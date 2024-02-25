from collections import defaultdict, OrderedDict
import glob
# import hf_olmo
import mmap
import numpy as np
import os
import random
import sys
import time
from transformers import AutoTokenizer

assert sys.byteorder == 'little'
PAGESIZE = int(os.sysconf(os.sysconf_names['SC_PAGESIZE']))

'''
Variable Symbol Value/Range Meaning
         S                  Number of shards of the datastore
         s      [0, S)      Index of datastore
tok_cnt         T           Number of tokens in the ds
ds_size         2 * T       Number of bytes in the ds
sa_cnt          T           Number of elements in the sa
sa_size         P * T       Number of bytes in the sa
ptr_size P      [1, 8]      Number of bytes per pointer
rank            [0, T)      Rank in the sa
ptr             [0, 2 * T)  Pointer into the ds, in bytes (must be even)
offset          [0, 2 * T)  Offset into the ds, in bytes (must be even)
'''

def bin_search(arr, val):
    lo, hi = -1, len(arr) # lo is always < val, hi is always >= val
    while hi - lo > 1:
        mi = (lo + hi) // 2
        if arr[mi] < val:
            lo = mi
        else:
            hi = mi
    return lo, hi

class NGramLanguageModeling(object):

    def __init__(self, consts, data_dir, eos_token_id):

        self.consts = consts
        self.data_dir = data_dir
        self.eos_token_id = eos_token_id

        assert os.path.isdir(self.data_dir), self.data_dir

        self.datastores = [] # Each datastore represents a chunk of the entire corpus
        if 'v4' in self.data_dir:
            self.version = 'v4'

            ds_path_base = os.path.join(self.data_dir, 'tokenized')
            sa_path_base = os.path.join(self.data_dir, 'table')
            do_path_base = os.path.join(self.data_dir, 'offset')

            ds_paths = sorted(glob.glob(f'{ds_path_base}*'))
            sa_paths = sorted(glob.glob(f'{sa_path_base}*'))
            do_paths = sorted(glob.glob(f'{do_path_base}*'))
            assert len(ds_paths) == len(sa_paths)
            assert len(ds_paths) == len(do_paths)
            self.num_shards = len(ds_paths)

            for (ds_path, sa_path, do_path) in zip(ds_paths, sa_paths, do_paths):
                f_ds = open(ds_path, 'rb')
                ds = mmap.mmap(f_ds.fileno(), 0, prot=mmap.PROT_READ)
                ds.madvise(mmap.MADV_RANDOM)
                f_sa = open(sa_path, 'rb')
                sa = mmap.mmap(f_sa.fileno(), 0, prot=mmap.PROT_READ)
                sa.madvise(mmap.MADV_RANDOM)
                f_do = open(do_path, 'rb')
                do = mmap.mmap(f_do.fileno(), 0, prot=mmap.PROT_READ)
                do.madvise(mmap.MADV_RANDOM)

                ds_size = os.path.getsize(ds_path)
                sa_size = os.path.getsize(sa_path)
                do_size = os.path.getsize(do_path)
                assert ds_size % 2 == 0 # 2 bytes per token
                tok_cnt = ds_size // 2 # total number of tokens
                assert sa_size % tok_cnt == 0
                ptr_size = sa_size // tok_cnt # size of each pointer
                assert do_size % 8 == 0 # 8 bytes per doc
                doc_cnt = do_size // 8 # total number of docs

                datastore = { 'ds': ds, 'sa': sa, 'do': do, 'tok_cnt': tok_cnt, 'ds_size': ds_size, 'ptr_size': ptr_size, 'doc_cnt': doc_cnt }
                self.datastores.append(datastore)
        else:
            raise ValueError(f'Datastore version not supported!')

        print (f'Loaded datastore {data_dir} with {self.num_shards} shards')

    def find(self, input_ids, hint_segments=None):
        '''
        Inputs:
        - input_ids [int]
        - hint_segments: [(int, int) * S]: hint from output of a previous search, whose input_ids was a prefix of the current input_ids
        Outputs:
        - segments [(int, int) * S]: starting and ending ranks in the sa, per datastore shard
        - cnt int: total number of occurrences of the prompt
        '''
        start_time = time.time()

        assert type(input_ids) == list
        if hint_segments is None:
            hint_segments = [None] * self.num_shards
        assert type(hint_segments) == list
        assert len(hint_segments) == self.num_shards

        input_buf = np.array(input_ids, dtype=np.uint16).view(np.uint8).tobytes()
        segments = []
        cnt = 0

        for datastore, hint_segment in zip(self.datastores, hint_segments):
            ds, sa, tok_cnt, ptr_size = datastore['ds'], datastore['sa'], datastore['tok_cnt'], datastore['ptr_size']
            if len(input_ids) == 0:
                segments.append((0, tok_cnt))
                cnt += tok_cnt
                continue

            def prefetch(lo, hi, depth=0):
                mi = (lo + hi) // 2 # sa index to inspect
                if mi == -1: # this may happen when lo=-1 and hi=0, and we skip prefetching
                    return
                if depth == 1: # fetch ds
                    ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                    ds.madvise(mmap.MADV_WILLNEED, ptr - ptr % PAGESIZE, len(input_buf) + ptr % PAGESIZE)
                elif depth == 3: # fetch sa
                    sa.madvise(mmap.MADV_WILLNEED, mi*ptr_size - mi*ptr_size % PAGESIZE, ptr_size + mi*ptr_size % PAGESIZE)
                    return
                prefetch(lo, mi, depth+1)
                prefetch(mi, hi, depth+1)

            # Search for the leftmost sa index that IS >= the prompt
            if hint_segment is None:
                lo, hi = -1, tok_cnt # lo is always < the prompt, hi is always >= the prompt
            else:
                lo, hi = hint_segment[0] - 1, hint_segment[1]
            while hi - lo > 1:
                prefetch(lo, hi)
                mi = (lo + hi) // 2 # sa index to inspect
                ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                if ds[ptr : ptr + len(input_buf)] < input_buf:
                    lo = mi
                else:
                    hi = mi
            start = hi

            # Search for the leftmost sa index that IS > the prompt
            if hint_segment is None:
                lo, hi = -1, tok_cnt # lo is always <= the prompt, hi is always > the prompt
            else:
                lo, hi = hint_segment[0] - 1, hint_segment[1]
            while hi - lo > 1:
                prefetch(lo, hi)
                mi = (lo + hi) // 2 # sa index to inspect
                ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                if ds[ptr : ptr + len(input_buf)] <= input_buf:
                    lo = mi
                else:
                    hi = mi
            end = hi

            assert start <= end
            segments.append((start, end))
            cnt += end - start

        assert cnt >= 0
        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {'cnt': cnt, 'segments': segments, 'latency': latency}

    def prob(self, prompt_ids, cont_id):
        '''
        Inputs:
        - prompt_ids [int]
        - cont_id int
        Outputs:
        - prob float
        - prompt_cnt int
        - cont_cnt int
        '''
        start_time = time.time()

        assert type(prompt_ids) == list
        assert type(cont_id) == int

        prompt_find_outputs = self.find(prompt_ids)

        prompt_cnt = prompt_find_outputs['cnt']
        cont_find_outputs = self.find(prompt_ids + [cont_id], hint_segments=prompt_find_outputs['segments'])
        cont_cnt = cont_find_outputs['cnt']
        prob = cont_cnt / prompt_cnt if prompt_cnt > 0 else -1.0

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {"prob": prob, "prompt_cnt": prompt_cnt, "cont_cnt": cont_cnt, "latency": latency}

    def ntd(self, prompt_ids):
        start_time = time.time()

        find_outputs = self.find(prompt_ids)

        if find_outputs['cnt'] == 0:
            end_time = time.time()
            latency = (end_time - start_time)*1000
            return {'prompt_cnt': 0, 'freq_by_token_id': {}, 'prob_by_token_id': {}, 'latency': latency}
        unit = 1 if find_outputs['cnt'] <= self.consts.MAX_CNT_FOR_NTD else 2**(int(np.ceil(np.log2(find_outputs['cnt'] / self.consts.MAX_CNT_FOR_NTD))))

        segments = find_outputs['segments']
        freq_by_token_id = defaultdict(int)
        for datastore, segment in zip(self.datastores, segments):
            ds, sa, ds_size, ptr_size = datastore['ds'], datastore['sa'], datastore['ds_size'], datastore['ptr_size']
            result = self._get_freq_by_token_id_approx(prompt_ids, segment, ds, sa, ds_size, ptr_size, unit)
            for token_id, freq in result.items():
                freq_by_token_id[token_id] += freq
        prompt_cnt = sum(freq_by_token_id.values())
        prob_by_token_id = {token_id: freq / prompt_cnt if prompt_cnt > 0 else -1.0 for token_id, freq in freq_by_token_id.items()}

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {'prompt_cnt': prompt_cnt, 'freq_by_token_id': freq_by_token_id, 'prob_by_token_id': prob_by_token_id, 'latency': latency}

    def _get_freq_by_token_id_approx(self, prompt_ids, segment, ds, sa, ds_size, ptr_size, unit, token_start=None, token_end=None):
        freq_by_token_id = OrderedDict()
        (start, end) = segment

        def prefetch(lo, hi, depth=0):
            mi = (lo + hi) // 2 # sa index to inspect
            if mi <= 0: # prefetching when mi <= 0 will cause page errors
                return
            if depth == 1: # fetch ds
                ptr = self.convert_rank_to_ptr(sa, mi-1, ptr_size)
                offset = ptr + 2 * len(prompt_ids)
                ds.madvise(mmap.MADV_WILLNEED, offset - offset % PAGESIZE, 2 + offset % PAGESIZE)
                ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                offset = ptr + 2 * len(prompt_ids)
                ds.madvise(mmap.MADV_WILLNEED, offset - offset % PAGESIZE, 2 + offset % PAGESIZE)
            elif depth == 3: # fetch sa
                sa.madvise(mmap.MADV_WILLNEED, (mi-1)*ptr_size - (mi-1)*ptr_size % PAGESIZE, 2*ptr_size + (mi-1)*ptr_size % PAGESIZE) # since we need both mi-1 and mi
                return
            prefetch(lo, mi, depth+1)
            prefetch(mi, hi, depth+1)
        prefetch(start, end)

        # Trivial case
        if end - start < 4 * unit:
            for rank in range(start, end, unit):
                rank_mid = rank + unit // 2 if rank + unit <= end else (rank + end) // 2
                ptr = self.convert_rank_to_ptr(sa, rank_mid, ptr_size)
                offset = ptr + 2 * len(prompt_ids)
                token_id = self.convert_offset_to_token_id(ds, offset)
                if token_id not in freq_by_token_id:
                    freq_by_token_id[token_id] = 0
                freq_by_token_id[token_id] += unit if rank + unit <= end else end - rank
            return freq_by_token_id

        # If start and end-1 has the same token, then we know this segment is all the same token
        if token_start is None:
            ptr_start = self.convert_rank_to_ptr(sa, start, ptr_size)
            token_start = self.convert_offset_to_token_id(ds, ptr_start + 2 * len(prompt_ids))
        if token_end is None:
            ptr_end = self.convert_rank_to_ptr(sa, end-1, ptr_size)
            token_end = self.convert_offset_to_token_id(ds, ptr_end + 2 * len(prompt_ids))
        if token_start == token_end:
            freq_by_token_id[token_start] = end - start
            return freq_by_token_id

        # Otherwise, we do divide and conquer
        mi = (start + end) // 2
        left_freq_by_token_id = self._get_freq_by_token_id_approx(prompt_ids, (start, mi), ds, sa, ds_size, ptr_size, unit, token_start=token_start)
        right_freq_by_token_id = self._get_freq_by_token_id_approx(prompt_ids, (mi, end), ds, sa, ds_size, ptr_size, unit, token_end=token_end)
        if next(reversed(left_freq_by_token_id)) != next(iter(right_freq_by_token_id)):
            freq_by_token_id = left_freq_by_token_id
            freq_by_token_id.update(right_freq_by_token_id)
        else:
            token_id, freq = left_freq_by_token_id.popitem(last=True)
            token_id2, freq2 = right_freq_by_token_id.popitem(last=False)
            assert token_id == token_id2
            freq_by_token_id = left_freq_by_token_id
            freq_by_token_id[token_id] = freq + freq2
            freq_by_token_id.update(right_freq_by_token_id)

        return freq_by_token_id

    def infgram_prob(self, prompt_ids, cont_id):
        start_time = time.time()

        assert type(prompt_ids) == list
        assert type(cont_id) == int

        L = len(prompt_ids)
        # binary lifting
        l_lo, l_hi = 0, 1 # l_lo always have non-zero count, l_hi always have zero count
        while True:
            if l_hi > L:
                l_hi = L + 1
                break
            result = self.find(prompt_ids[L-l_hi:])
            if result['cnt'] == 0:
                break
            l_lo = l_hi
            l_hi *= 2
        # binary search within [l_lo, l_hi)
        while l_hi - l_lo > 1:
            l_mid = (l_lo + l_hi) // 2
            result = self.find(prompt_ids[L-l_mid:])
            if result['cnt'] == 0:
                l_hi = l_mid
            else:
                l_lo = l_mid

        l = l_lo
        prompt_suffix_ids = prompt_ids[L-l:]
        result = self.prob(prompt_suffix_ids, cont_id)
        result['lfn'] = l + 1

        end_time = time.time()
        latency = (end_time - start_time)*1000
        result['latency'] = latency
        return result

    def infgram_ntd(self, prompt_ids):
        start_time = time.time()

        assert type(prompt_ids) == list

        L = len(prompt_ids)
        # binary lifting
        l_lo, l_hi = 0, 1 # l_lo always have non-zero count, l_hi always have zero count
        while True:
            if l_hi > L:
                l_hi = L + 1
                break
            result = self.find(prompt_ids[L-l_hi:])
            if result['cnt'] == 0:
                break
            l_lo = l_hi
            l_hi *= 2
        # binary search within [l_lo, l_hi)
        while l_hi - l_lo > 1:
            l_mid = (l_lo + l_hi) // 2
            result = self.find(prompt_ids[L-l_mid:])
            if result['cnt'] == 0:
                l_hi = l_mid
            else:
                l_lo = l_mid

        l = l_lo
        prompt_suffix_ids = prompt_ids[L-l:]
        result = self.ntd(prompt_suffix_ids)
        result['lfn'] = l + 1

        end_time = time.time()
        latency = (end_time - start_time)*1000
        result['latency'] = latency
        return result

    def search_docs(self, cnf, maxnum):
        start_time = time.time()

        if len(cnf) == 1:
            disj_clause = cnf[0]
            find_result = self.find_disj(disj_clause, approx=False)
            if find_result['cnt'] == 0:
                if len(disj_clause) == 1:
                    return {'error': f'The query term is not found in the corpus!'}
                else:
                    return {'error': f'None of the query terms is found in the corpus!'}

            # sample up to maxnum documents
            cnt, cnt_by_shard, segments_by_shard = find_result['cnt'], find_result['cnt_by_shard'], find_result['segments_by_shard']
            documents, idxs = [], []
            for _ in range(maxnum):
                s = random.choices(range(self.num_shards), weights=cnt_by_shard)[0]
                segments = segments_by_shard[s]
                cnt_by_segment = [end-start for (start, end) in segments]
                ss = random.choices(range(len(segments)), weights=cnt_by_segment)[0]
                (start, end) = segments[ss]
                rank = random.randrange(start, end) # left inclusive, right exclusive
                ptr = self.convert_rank_to_ptr(self.datastores[s]['sa'], rank, self.datastores[s]['ptr_size'])
                document = self.get_document(s, ptr, ptr, max_output_doc_tokens=self.consts.MAX_OUTPUT_DOC_TOKENS // maxnum)
                idx = sum(cnt_by_shard[:s]) + sum(cnt_by_segment[:ss]) + (rank - start)
                documents.append(document)
                idxs.append(idx)

            end_time = time.time()
            latency = (end_time - start_time)*1000
            return {'documents': documents, 'idxs': idxs, 'cnt': cnt, 'approx': False, 'latency': latency}

        find_result = self.find_cnf(cnf)
        if 0 in find_result['cnt_by_clause']:
            d = find_result['cnt_by_clause'].index(0)
            return {'error': f'None of the terms in the {d}-th disjunctive clause is found in the corpus!'}

        # sample up to maxnum documents
        cnt, valid_ptr_ranges_by_shard, approx = find_result['cnt'], find_result['valid_ptr_ranges_by_shard'], find_result['approx']
        valid_ptr_cnt_by_shard = [len(valid_ptr_ranges) for valid_ptr_ranges in valid_ptr_ranges_by_shard]
        valid_ptr_cnt = sum(valid_ptr_cnt_by_shard)
        if valid_ptr_cnt == 0:
            return {'error': 'Query is not found in the corpus! Try relaxing the constraints.'}
        documents, idxs = [], []
        for _ in range(maxnum):
            s = random.choices(range(self.num_shards), weights=valid_ptr_cnt_by_shard)[0]
            valid_ptr_ranges = valid_ptr_ranges_by_shard[s]
            i = random.randrange(0, len(valid_ptr_ranges)) # left inclusive, right exclusive
            ptr_range = valid_ptr_ranges[i]
            percentile = (sum(valid_ptr_cnt_by_shard[:s]) + i) / valid_ptr_cnt
            idx = int(percentile * cnt)
            (l, r) = ptr_range
            document = self.get_document(s, l, r, max_output_doc_tokens=self.consts.MAX_OUTPUT_DOC_TOKENS // maxnum)
            documents.append(document)
            idxs.append(idx)

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {'documents': documents, 'idxs': idxs, 'cnt': cnt, 'approx': approx, 'latency': latency}

    def find_cnf(self, cnf):
        assert len(cnf) > 0

        find_results = []
        cnt_by_clause = []
        for d in range(len(cnf)):
            disj_clause = cnf[d]
            find_result = self.find_disj(disj_clause, approx=True)
            find_results.append(find_result)
            cnt_by_clause.append(find_result['cnt'])
        if any([cnt == 0 for cnt in cnt_by_clause]):
            return {
                'cnt_by_clause': cnt_by_clause,
                'cnt': 0,
                'cnt_by_shard': [],
                'valid_ptr_ranges_by_shard': [],
                'subsampling_factor_by_shard': [],
                'approx': False,
            }

        cnt_by_shard = []
        valid_ptr_ranges_by_shard = []
        subsampling_factor_by_shard = []
        for s in range(self.num_shards):
            # sort find_results by cnt in increasing order
            find_results = sorted(find_results, key=lambda find_result: sum([end - start for (start, end) in find_result['segments_by_shard'][s]]))

            datastore = self.datastores[s]
            ds, sa, ptr_size, ds_size = datastore['ds'], datastore['sa'], datastore['ptr_size'], datastore['ds_size']
            find_result = find_results[0]
            valid_ptr_ranges = []
            for (start, end) in find_result['segments_by_shard'][s]:
                ptrs = self.convert_ranks_to_ptrs(sa, start, end, ptr_size)
                valid_ptr_ranges += [(ptr, ptr) for ptr in ptrs]
            subsampling_factor = find_result['subsampling_factor_by_shard'][s]

            # maintain valid ptr ranges
            # if there are Q terms and each term has M matches in the shard, the complexity is O(Q * M * log(M))
            for d in range(1, len(cnf)):
                find_result = find_results[d]
                ptrs = []
                for (start, end) in find_result['segments_by_shard'][s]:
                    ptrs += self.convert_ranks_to_ptrs(sa, start, end, ptr_size)
                ptrs = sorted(ptrs)
                new_valid_ptr_ranges = []
                for (l, r) in valid_ptr_ranges:
                    lo, _ = bin_search(ptrs, r)
                    new_l = -1 if lo == -1 else min(l, ptrs[lo])
                    _, hi = bin_search(ptrs, l)
                    new_r = ds_size if hi == len(ptrs) else max(r, ptrs[hi])
                    if new_l != -1 and new_l >= l - self.consts.MAX_DIFF_BYTES and new_r != ds_size and new_r <= r + self.consts.MAX_DIFF_BYTES: # +- (MAX_DIFF_BYTES / 2) tokens
                        new_valid_ptr_ranges.append((new_l, new_r))
                    else:
                        if new_l != -1 and new_l >= l - self.consts.MAX_DIFF_BYTES:
                            new_valid_ptr_ranges.append((new_l, r))
                        if new_r != ds_size and new_r <= r + self.consts.MAX_DIFF_BYTES:
                            new_valid_ptr_ranges.append((l, new_r))
                valid_ptr_ranges = new_valid_ptr_ranges
                subsampling_factor *= find_result['subsampling_factor_by_shard'][s]

            # remove ptr ranges that cross document boundary
            new_valid_ptr_ranges = []
            for (l, r) in valid_ptr_ranges:
                token_buf = np.frombuffer(ds[l:r], dtype=np.uint8)
                token_ids = token_buf.view(np.uint16).tolist()
                if 65535 in token_ids:
                    continue
                new_valid_ptr_ranges.append((l, r))
            valid_ptr_ranges = new_valid_ptr_ranges

            cnt = int(len(valid_ptr_ranges) * subsampling_factor)
            cnt_by_shard.append(cnt)
            valid_ptr_ranges_by_shard.append(valid_ptr_ranges)
            subsampling_factor_by_shard.append(subsampling_factor)

        cnt = sum(cnt_by_shard)
        approx = any([subsampling_factor > 1 for subsampling_factor in subsampling_factor_by_shard])
        return {
            'cnt_by_clause': cnt_by_clause,
            'cnt': cnt,
            'cnt_by_shard': cnt_by_shard,
            'valid_ptr_ranges_by_shard': valid_ptr_ranges_by_shard,
            'subsampling_factor_by_shard': subsampling_factor_by_shard,
            'approx': approx,
        }

    def find_disj(self, disj_clause, approx=False):
        '''
        Inputs:
        - disj_clause [[int]]: a disjunctive clause, where each element is a list of token ids
        '''
        np.random.seed(19260817)
        find_results = []
        for input_ids in disj_clause:
            find_result = self.find(input_ids)
            find_results.append(find_result)
        cnt_by_shard = []
        subsampling_factor_by_shard = []
        segments_by_shard = []
        for s in range(self.num_shards):
            segments = []
            for find_result in find_results:
                segments.append(find_result['segments'][s])
            cnt = sum([end - start for (start, end) in segments])
            subsampling_factor = 1
            if approx and cnt > self.consts.MAX_CLAUSE_FREQ_PER_SHARD:
                # TODO: This subsampling might not be uniform
                new_segments = []
                for (start, end) in segments:
                    length = end - start
                    new_length = int(np.ceil(length / cnt * self.consts.MAX_CLAUSE_FREQ_PER_SHARD))
                    new_start = start + np.random.randint(0, length - new_length + 1) # left inclusive, right exclusive
                    new_end = new_start + new_length
                    new_segments.append((new_start, new_end))
                new_cnt = sum([end - start for (start, end) in new_segments])
                assert new_cnt > 0
                assert new_cnt <= cnt
                subsampling_factor = cnt / new_cnt
                segments = new_segments
            cnt_by_shard.append(cnt)
            subsampling_factor_by_shard.append(subsampling_factor)
            segments_by_shard.append(segments)
        cnt = sum(cnt_by_shard)
        return {
            'cnt': cnt,
            'cnt_by_shard': cnt_by_shard,
            'segments_by_shard': segments_by_shard,
            'subsampling_factor_by_shard': subsampling_factor_by_shard,
        }

    def get_document(self, s, left_ptr, right_ptr, max_output_doc_tokens):
        datastore = self.datastores[s]
        max_prepend_tokens = max(1, (max_output_doc_tokens - (right_ptr - left_ptr) // 2) // 2)
        max_append_tokens = max(1, (max_output_doc_tokens - (right_ptr - left_ptr) // 2 + 1) // 2)

        ds, do, ds_size, doc_cnt = datastore['ds'], datastore['do'], datastore['ds_size'], datastore['doc_cnt']

        def prefetch(lo, hi, depth=0):
            mi = (lo + hi) // 2 # sa index to inspect
            if depth == 3: # fetch sa
                do.madvise(mmap.MADV_WILLNEED, mi*8 - mi*8 % PAGESIZE, 8 + mi*8 % PAGESIZE)
                return
            prefetch(lo, mi, depth+1)
            prefetch(mi, hi, depth+1)

        lo, hi = 0, doc_cnt # lo always <= the answer, hi always > the answer
        while hi - lo > 1:
            prefetch(lo, hi)
            mi = (lo + hi) // 2
            ptr = self.convert_doc_rank_to_ptr(do, mi, doc_cnt, ds_size)
            if ptr <= left_ptr:
                lo = mi
            else:
                hi = mi

        doc_ix = sum([self.datastores[_]['doc_cnt'] for _ in range(s)] + [0]) + lo

        doc_start_ptr = self.convert_doc_rank_to_ptr(do, lo, doc_cnt, ds_size) + 2 # +2 because we want to skip the document separator
        doc_end_ptr = self.convert_doc_rank_to_ptr(do, lo+1, doc_cnt, ds_size)
        doc_len = (doc_end_ptr - doc_start_ptr) // 2

        start_ptr = max(doc_start_ptr, left_ptr - 2 * max_prepend_tokens)
        end_ptr = min(doc_end_ptr, right_ptr + 2 * max_append_tokens)
        disp_len = (end_ptr - start_ptr) // 2

        token_buf = np.frombuffer(ds[start_ptr : end_ptr], dtype=np.uint8)
        token_ids = token_buf.view(np.uint16).tolist()
        token_offset = (left_ptr - start_ptr) // 2
        return {'token_ids': token_ids, 'token_offset': token_offset, 'doc_ix': doc_ix, 'doc_len': doc_len, 'disp_len': disp_len}

    def analyze_document(self, input_ids):
        assert type(input_ids)==list
        assert len(input_ids)>0

        L = len(input_ids)
        all_result_by_cutoff = []
        compute_time = 0
        for l in range(L):
            result_by_cutoff = {}
            for cutoff in reversed(range(0, l+1)):
                compute_time += 1
                if compute_time > 2 * self.consts.MAX_INPUT_DOC_TOKENS:
                    return None
                result = self.prob(input_ids[cutoff:l], input_ids[l])
                if result['prompt_cnt'] == 0:
                    break
                result_by_cutoff[cutoff] = result
            all_result_by_cutoff.append(result_by_cutoff)
            # print(l, result_by_cutoff)
        return all_result_by_cutoff

    def convert_offset_to_token_id(self, ds, offset):
        if offset >= len(ds):
            # This happens when we matched the very end of the ds.
            return self.eos_token_id
        token_buf = np.frombuffer(ds[offset:offset+2], dtype=np.uint8)
        token_id = int(token_buf.view(np.uint16)[0])
        # If you see \xff\xff, this actually means we're at the very end of a document.
        if token_id == 65535:
            token_id = self.eos_token_id
        return token_id

    def convert_rank_to_ptr(self, sa, rank, ptr_size):
        ptr_buf = np.frombuffer(sa[rank*ptr_size:(rank+1)*ptr_size], dtype=np.uint8)
        # Add right padding due to little-endianness
        ptr_buf = np.pad(ptr_buf, (0, 8 - ptr_size), 'constant', constant_values=0)
        ptr = int(ptr_buf.view(np.uint64)[0])
        return ptr

    def convert_ranks_to_ptrs(self, sa, rank_start, rank_end, ptr_size):
        ptr_buf = np.frombuffer(sa[rank_start*ptr_size:rank_end*ptr_size], dtype=np.uint8)
        # Add right padding due to little-endianness
        ptr_buf = np.pad(ptr_buf.reshape(-1, ptr_size), ((0, 0), (0, 8 - ptr_size)), 'constant', constant_values=0).reshape(-1)
        ptrs = ptr_buf.view(np.uint64).tolist()
        return ptrs

    def convert_doc_rank_to_ptr(self, do, rank, doc_cnt, ds_size):
        assert 0 <= rank and rank <= doc_cnt
        if rank == doc_cnt:
            return ds_size
        ptr_buf = np.frombuffer(do[rank*8:(rank+1)*8], dtype=np.uint8)
        ptr = int(ptr_buf.view(np.uint64)[0])
        return ptr


class NGramLanguageModelingUnion(NGramLanguageModeling):

    def __init__(self, consts, data_dirs, eos_token_id):

        self.consts = consts
        self.data_dirs = data_dirs
        self.eos_token_id = eos_token_id

        self.lms = [NGramLanguageModeling(consts, data_dir, eos_token_id) for data_dir in data_dirs]

    def find(self, input_ids):
        start_time = time.time()

        results = [lm.find(input_ids) for lm in self.lms]
        cnt_by_lm = [result['cnt'] for result in results]
        cnt = sum(cnt_by_lm)

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {'cnt': cnt, 'latency': latency}

    def prob(self, prompt_ids, cont_id):
        start_time = time.time()

        results = [lm.prob(prompt_ids, cont_id) for lm in self.lms]
        prompt_cnt = sum([result['prompt_cnt'] for result in results])
        cont_cnt = sum([result['cont_cnt'] for result in results])
        prob = cont_cnt / prompt_cnt if prompt_cnt > 0 else -1.0

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {"prob": prob, "prompt_cnt": prompt_cnt, "cont_cnt": cont_cnt, "latency": latency}

    def ntd(self, prompt_ids):
        start_time = time.time()

        results = [lm.ntd(prompt_ids) for lm in self.lms]
        prompt_cnt = sum([result['prompt_cnt'] for result in results])
        freq_by_token_id = defaultdict(int)
        for result in results:
            for token_id, freq in result['freq_by_token_id'].items():
                freq_by_token_id[token_id] += freq
        prob_by_token_id = {token_id: freq / prompt_cnt if prompt_cnt > 0 else -1.0 for token_id, freq in freq_by_token_id.items()}

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {'prompt_cnt': prompt_cnt, 'freq_by_token_id': freq_by_token_id, 'prob_by_token_id': prob_by_token_id, 'latency': latency}

    def search_docs(self, cnf, maxnum):
        start_time = time.time()

        if len(cnf) == 1:
            disj_clause = cnf[0]
            find_results = [lm.find_disj(disj_clause) for lm in self.lms]
            cnt_by_lm = [find_result['cnt'] for find_result in find_results]
            if sum(cnt_by_lm) == 0:
                if len(disj_clause) == 1:
                    return {'error': f'The query term is not found in the corpus!'}
                else:
                    return {'error': f'None of the query terms is found in the corpus!'}

            # sample up to maxnum documents
            documents, idxs = [], []
            for _ in range(maxnum):
                l = random.choices(range(len(self.lms)), weights=cnt_by_lm)[0]
                lm = self.lms[l]
                find_result = find_results[l]
                cnt, cnt_by_shard, segments_by_shard = find_result['cnt'], find_result['cnt_by_shard'], find_result['segments_by_shard']
                s = random.choices(range(lm.num_shards), weights=cnt_by_shard)[0]
                segments = segments_by_shard[s]
                cnt_by_segment = [end-start for (start, end) in segments]
                ss = random.choices(range(len(segments)), weights=cnt_by_segment)[0]
                (start, end) = segments[ss]
                rank = random.randrange(start, end) # left inclusive, right exclusive
                ptr = lm.convert_rank_to_ptr(lm.datastores[s]['sa'], rank, lm.datastores[s]['ptr_size'])
                document = lm.get_document(s, ptr, ptr, max_output_doc_tokens=lm.consts.MAX_OUTPUT_DOC_TOKENS // maxnum)
                idx = sum(cnt_by_shard[:s]) + sum(cnt_by_segment[:ss]) + (rank - start)
                documents.append(document)
                idxs.append(idx)

            end_time = time.time()
            latency = (end_time - start_time)*1000
            return {'documents': documents, 'idxs': idxs, 'cnt': cnt, 'approx': False, 'latency': latency}

        find_results = [lm.find_cnf(cnf) for lm in self.lms]

        cnt_by_lm = [find_result['cnt'] for find_result in find_results]
        if sum(cnt_by_lm) == 0:
            return {'error': 'Query is not found in the corpus! Try relaxing the constraints.'}

        # sample up to maxnum documents
        documents, idxs = [], []
        for _ in range(maxnum):
            l = random.choices(range(len(self.lms)), weights=cnt_by_lm)[0]
            lm = self.lms[l]
            find_result = find_results[l]
            cnt, valid_ptr_ranges_by_shard, approx = find_result['cnt'], find_result['valid_ptr_ranges_by_shard'], find_result['approx']
            valid_ptr_cnt_by_shard = [len(valid_ptr_ranges) for valid_ptr_ranges in valid_ptr_ranges_by_shard]
            valid_ptr_cnt = sum(valid_ptr_cnt_by_shard)
            s = random.choices(range(lm.num_shards), weights=valid_ptr_cnt_by_shard)[0]
            valid_ptr_ranges = valid_ptr_ranges_by_shard[s]
            i = random.randrange(0, len(valid_ptr_ranges)) # left inclusive, right exclusive
            ptr_range = valid_ptr_ranges[i]
            percentile = (sum(valid_ptr_cnt_by_shard[:s]) + i) / valid_ptr_cnt
            idx = int(percentile * cnt)
            (l, r) = ptr_range
            document = lm.get_document(s, l, r, max_output_doc_tokens=lm.consts.MAX_OUTPUT_DOC_TOKENS // maxnum)
            documents.append(document)
            idxs.append(idx)

        cnt = sum(cnt_by_lm)
        approx = any([find_result['approx'] for find_result in find_results])

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return {'documents': documents, 'idxs': idxs, 'cnt': cnt, 'approx': approx, 'latency': latency}


def main():
    tokenizer_type = 'llama'
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.environ['HF_TOKEN_DOWNLOAD'], add_bos_token=False, add_eos_token=False)
    tokenizer.decode([1, 2, 3])
    lm = NGramLanguageModeling(data_dir='/large_experiments/cmd/ngram_datastore/redpajama_bff2_v3_c8_llama2', eos_token_id=tokenizer.eos_token_id)

    # tokenizer_type = 'olmo'
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False)
    # tokenizer.decode([1, 2, 3])
    # lm = NGramLanguageModeling(data_dir='/data-dolmasample-olmo/v4_dolmasample_olmo', eos_token_id=tokenizer.eos_token_id)


    def tokenize(query):
        if tokenizer_type == 'gpt2':
            if query != '':
                query = ' ' + query
            input_ids = tokenizer.encode(query)
        elif tokenizer_type == 'llama':
            input_ids = tokenizer.encode(query)
            if len(input_ids) > 0 and input_ids[0] == 29871:
                input_ids = input_ids[1:]
        elif tokenizer_type == 'olmo':
            if query != '':
                query = ' ' + query
            input_ids = tokenizer.encode(query)
        else:
            raise NotImplementedError
        return input_ids

    ngram = 'natural language processing'
    input_ids = tokenize(ngram)
    print(input_ids)

    start_time = time.time()
    find_result = lm.find(input_ids)
    end_time = time.time()
    print(find_result)
    print(f'find: {end_time - start_time} seconds')
    print()

    start_time = time.time()
    prob_result = lm.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
    end_time = time.time()
    print(prob_result)
    print(f'prob: {end_time - start_time} seconds')
    print()

    start_time = time.time()
    ntd_result = lm.ntd(prompt_ids=input_ids[:-1])
    end_time = time.time()
    print(ntd_result)
    print(f'ntd: {end_time - start_time} seconds')
    print()

    input_ids = [random.randint(0, 65535) for _ in range(5)] + input_ids
    start_time = time.time()
    prob_result = lm.infgram_prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
    end_time = time.time()
    print(prob_result)
    print(f'infgram_prob: {end_time - start_time} seconds')
    print()

    start_time = time.time()
    ntd_result = lm.infgram_ntd(prompt_ids=input_ids[:-1])
    end_time = time.time()
    print(ntd_result)
    print(f'infgram_ntd: {end_time - start_time} seconds')
    print()

    # ngrams, ngrams_2 = ['natural language processing'], ['deep learning']
    ngrams, ngrams_2 = ['artificial intelligence', 'natural language processing'], ['machine learning', 'deep learning']
    disj_clause = [tokenize(ngram) for ngram in ngrams]
    disj_clause_2 = [tokenize(ngram) for ngram in ngrams_2]
    cnf = [disj_clause, disj_clause_2]
    start_time = time.time()
    result = lm.search_docs(cnf, maxnum=1)
    end_time = time.time()
    if type(result) == str:
        print(result)
    else:
        print(result['cnt'], result['approx'])
        document = result['documents'][0]
        token_ids = document['token_ids']
        token_offset = document['token_offset']
        string = tokenizer.decode(token_ids)
        prefix_string = tokenizer.decode(token_ids[:token_offset])
        char_offset = len(prefix_string)
        print(f'string: {string[:50]} ... {string[char_offset-100:char_offset+100]} ... {string[-50:]}')
    print(f'search_docs: {end_time - start_time} seconds')
    print()

    ngrams = ['Mount Tai', 'Shandong']
    disj_clause = [tokenize(ngram) for ngram in ngrams]
    cnf = [disj_clause]
    start_time = time.time()
    result = lm.search_docs(cnf, maxnum=10)
    end_time = time.time()
    if type(result) == str:
        print(result)
    else:
        print(result['cnt'], result['approx'])
        document = result['documents'][0]
        token_ids = document['token_ids']
        token_offset = document['token_offset']
        string = tokenizer.decode(token_ids)
        prefix_string = tokenizer.decode(token_ids[:token_offset])
        char_offset = len(prefix_string)
        print(f'string: {string[:50]} ... {string[char_offset-100:char_offset+100]} ... {string[-50:]}')
    print(f'search_docs: {end_time - start_time} seconds')
    print()

if __name__ == '__main__':
    main()
