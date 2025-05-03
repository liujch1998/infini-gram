import sys
from typing import Iterable, List, Optional, Tuple, Dict

from .models import *
from . import cpp_engine

class InfiniGramEngine:

    def __init__(self, index_dir: Iterable[str] | str, eos_token_id: int, vocab_size=65535, version=4, token_dtype='u16',
                 load_to_ram=False, ds_prefetch_depth=1, sa_prefetch_depth=3, od_prefetch_depth=3,
                 bow_ids_path: str = None, attribution_block_size: int = 512, precompute_unigram_logprobs: bool = False,
                 prev_shards_by_index_dir = {},
                 max_support=1000, max_clause_freq=50000, max_diff_tokens=100, maxnum=1, max_disp_len=1000,
                 ) -> None:

        assert sys.byteorder == 'little', 'This code is designed to run on little-endian machines only!'

        if type(index_dir) == str:
            index_dir = [index_dir]
        assert type(index_dir) == list and all(type(d) == str for d in index_dir)
        assert type(eos_token_id) == int and 0 <= eos_token_id and eos_token_id < vocab_size
        assert type(load_to_ram) == bool
        assert type(ds_prefetch_depth) == int and ds_prefetch_depth >= 0
        assert type(sa_prefetch_depth) == int and sa_prefetch_depth >= ds_prefetch_depth
        assert type(od_prefetch_depth) == int and od_prefetch_depth >= 0
        assert type(max_support) == int and max_support > 0
        assert type(max_clause_freq) == int and max_clause_freq > 0
        assert type(max_diff_tokens) == int and max_diff_tokens > 0
        assert type(maxnum) == int and maxnum > 0
        assert type(max_disp_len) == int and max_disp_len > 0

        self.max_support = max_support
        self.max_clause_freq = max_clause_freq
        self.max_diff_tokens = max_diff_tokens
        self.maxnum = maxnum
        self.max_disp_len = max_disp_len

        bow_ids = set()
        if bow_ids_path is not None:
            try:
                with open(bow_ids_path, 'r') as f:
                    for line in f:
                        bow_ids.add(int(line.strip()))
            except Exception as e:
                print(f"Error reading bow_ids_path: {e}")
                raise e

        if token_dtype == 'u8':
            self.token_id_max = 2**8 - 1
            engine_class = cpp_engine.Engine_U8
        elif token_dtype == 'u16':
            self.token_id_max = 2**16 - 1
            engine_class = cpp_engine.Engine_U16
        elif token_dtype == 'u32':
            self.token_id_max = 2**32 - 1
            engine_class = cpp_engine.Engine_U32
        else:
            raise ValueError(f'Unsupported token dtype: {token_dtype}')
        self.engine = engine_class(index_dir, eos_token_id, vocab_size, version, load_to_ram, ds_prefetch_depth, sa_prefetch_depth, od_prefetch_depth, bow_ids, attribution_block_size, precompute_unigram_logprobs, prev_shards_by_index_dir)

    def compute_unigram_counts(self, s: int) -> List[int]:
        return self.engine.compute_unigram_counts(s=s)

    def get_new_shards_by_index_dir(self):
        return self.engine.get_new_shards_by_index_dir()

    def check_query_ids(self, query_ids: QueryIdsType, allow_empty: bool) -> bool:
        if not (type(query_ids) == list and (allow_empty or len(query_ids) > 0)):
            return False
        for q in query_ids:
            if not (type(q) == int and 0 <= q and q <= self.token_id_max):
                return False
        return True

    def check_cnf(self, cnf: CnfType) -> bool:
        if not (type(cnf) == list and len(cnf) > 0):
            return False
        for disj_clause in cnf:
            if not (type(disj_clause) == list and len(disj_clause) > 0):
                return False
            for query_ids in disj_clause:
                if not (type(query_ids) == list and len(query_ids) > 0):
                    return False
                for q in query_ids:
                    if not (type(q) == int and 0 <= q and q <= self.token_id_max):
                        return False
        return True

    def find(self, input_ids: QueryIdsType) -> InfiniGramEngineResponse[FindResponse]:
        if not self.check_query_ids(input_ids, allow_empty=True):
            return {'error': f'input_ids must be a list of integers in range [0, {self.token_id_max}]'}
        result = self.engine.find(input_ids=input_ids)
        return {'cnt': result.cnt, 'segment_by_shard': result.segment_by_shard}

    def find_cnf(self, cnf: CnfType, max_clause_freq: Optional[int]=None, max_diff_tokens: Optional[int]=None) -> InfiniGramEngineResponse[FindCnfResponse]:
        if max_clause_freq is None:
            max_clause_freq = self.max_clause_freq
        if max_diff_tokens is None:
            max_diff_tokens = self.max_diff_tokens
        if not (type(max_clause_freq) == int and max_clause_freq > 0):
            return {'error': 'max_clause_freq must be a positive integer'}
        if not (type(max_diff_tokens) == int and max_diff_tokens > 0):
            return {'error': 'max_diff_tokens must be a positive integer'}
        if not self.check_cnf(cnf):
            return {'error': f'cnf must be a non-empty, triply-nested list of integers in range [0, {self.token_id_max}]'}
        result = self.engine.find_cnf(cnf=cnf, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)
        return {'cnt': result.cnt, 'approx': result.approx, 'ptrs_by_shard': result.ptrs_by_shard}

    def count(self, input_ids: QueryIdsType) -> InfiniGramEngineResponse[CountResponse]:
        if not self.check_query_ids(input_ids, allow_empty=True):
            return {'error': f'input_ids must be a list of integers in range [0, {self.token_id_max}]'}
        result = self.engine.count(input_ids=input_ids)
        return {'count': result.count, 'approx': result.approx}

    def count_cnf(self, cnf: CnfType, max_clause_freq=None, max_diff_tokens=None)-> InfiniGramEngineResponse[CountResponse]:
        if max_clause_freq is None:
            max_clause_freq = self.max_clause_freq
        if max_diff_tokens is None:
            max_diff_tokens = self.max_diff_tokens
        if not (type(max_clause_freq) == int and max_clause_freq > 0):
            return {'error': 'max_clause_freq must be a positive integer'}
        if not (type(max_diff_tokens) == int and max_diff_tokens > 0):
            return {'error': 'max_diff_tokens must be a positive integer'}
        if not self.check_cnf(cnf):
            return {'error': f'cnf must be a non-empty, triply-nested list of integers in range [0, {self.token_id_max}]'}
        result = self.engine.count_cnf(cnf=cnf, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)
        return {'count': result.count, 'approx': result.approx}

    def prob(self, prompt_ids: QueryIdsType, cont_id: int) -> InfiniGramEngineResponse[ProbResponse]:
        if not self.check_query_ids(prompt_ids, allow_empty=True):
            return {'error': f'prompt_ids must be a non-empty list of integers in range [0, {self.token_id_max}]'}
        if not (type(cont_id) == int and 0 <= cont_id and cont_id <= self.token_id_max):
            return {'error': f'cont_id must be an integer in range [0, {self.token_id_max}]'}
        result = self.engine.prob(prompt_ids=prompt_ids, cont_id=cont_id)
        return {'prompt_cnt': result.prompt_cnt, 'cont_cnt': result.cont_cnt, 'prob': result.prob}

    def ntd(self, prompt_ids: QueryIdsType, max_support: Optional[int]=None) -> InfiniGramEngineResponse[NtdResponse] :
        if max_support is None:
            max_support = self.max_support
        if not (type(max_support) == int and max_support > 0):
            return {'error': 'max_support must be a positive integer'}
        if not self.check_query_ids(prompt_ids, allow_empty=True):
            return {'error': f'prompt_ids must be a list of integers in range [0, {self.token_id_max}]'}
        result = self.engine.ntd(prompt_ids=prompt_ids, max_support=max_support)
        result_by_token_id: dict[int, DistTokenResult] = {token_id: {'cont_cnt': r.cont_cnt, 'prob': r.prob} for token_id, r in result.result_by_token_id.items()}
        return {'prompt_cnt': result.prompt_cnt, 'result_by_token_id': result_by_token_id, 'approx': result.approx}

    def infgram_prob(self, prompt_ids: QueryIdsType, cont_id: int) -> InfiniGramEngineResponse[InfGramProbResponse]:
        if not self.check_query_ids(prompt_ids, allow_empty=True):
            return {'error': f'prompt_ids must be a non-empty list of integers in range [0, {self.token_id_max}]'}
        if not (type(cont_id) == int and 0 <= cont_id and cont_id <= self.token_id_max):
            return {'error': f'cont_id must be an integer in range [0, {self.token_id_max}]'}
        result = self.engine.infgram_prob(prompt_ids=prompt_ids, cont_id=cont_id)
        return {'prompt_cnt': result.prompt_cnt, 'cont_cnt': result.cont_cnt, 'prob': result.prob, 'suffix_len': result.suffix_len}

    def infgram_ntd(self, prompt_ids: QueryIdsType, max_support: Optional[int] = None) -> InfiniGramEngineResponse[InfGramNtdResponse]:
        if max_support is None:
            max_support = self.max_support
        if not (type(max_support) == int and max_support > 0):
            return {'error': 'max_support must be a positive integer'}
        if not self.check_query_ids(prompt_ids, allow_empty=True):
            return {'error': f'prompt_ids must be a list of integers in range [0, {self.token_id_max}]'}
        result = self.engine.infgram_ntd(prompt_ids=prompt_ids, max_support=max_support)
        result_by_token_id:  dict[int, DistTokenResult] = {token_id: {'cont_cnt': r.cont_cnt, 'prob': r.prob} for token_id, r in result.result_by_token_id.items()}
        return {'prompt_cnt': result.prompt_cnt, 'result_by_token_id': result_by_token_id, 'approx': result.approx, 'suffix_len': result.suffix_len}

    def search_docs(self, input_ids: QueryIdsType, maxnum: Optional[int] = None, max_disp_len: Optional[int] = None) -> InfiniGramEngineResponse[SearchDocsResponse]:
        if maxnum is None:
            maxnum = self.maxnum
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(maxnum) == int and maxnum > 0):
            return {'error': 'maxnum must be a positive integer'}
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        if not self.check_query_ids(input_ids, allow_empty=True):
            return {'error': f'input_ids must be a list of integers in range [0, {self.token_id_max}]'}

        result = self.engine.search_docs(input_ids=input_ids, maxnum=maxnum, max_disp_len=max_disp_len)

        documents: List[DocResult] = [{'doc_ix': d.doc_ix, 'doc_len': d.doc_len, 'disp_len': d.disp_len, 'needle_offset': d.needle_offset, 'metadata': d.metadata, 'token_ids': d.token_ids, 'blocked': d.blocked} for d in result.docs] # type: ignore
        return {'cnt': result.cnt, 'approx': result.approx, 'idxs': result.idxs, 'documents': documents}

    def search_docs_cnf(self, cnf: CnfType, maxnum: Optional[int] = None, max_disp_len: Optional[int] = None, max_clause_freq: Optional[int] = None, max_diff_tokens: Optional[int] = None) -> InfiniGramEngineResponse[SearchDocsResponse]:
        if maxnum is None:
            maxnum = self.maxnum
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if max_clause_freq is None:
            max_clause_freq = self.max_clause_freq
        if max_diff_tokens is None:
            max_diff_tokens = self.max_diff_tokens
        if not (type(maxnum) == int and maxnum > 0):
            return {'error': 'maxnum must be a positive integer'}
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        if not (type(max_clause_freq) == int and max_clause_freq > 0):
            return {'error': 'max_clause_freq must be a positive integer'}
        if not (type(max_diff_tokens) == int and max_diff_tokens > 0):
            return {'error': 'max_diff_tokens must be a positive integer'}
        if not self.check_cnf(cnf):
            return {'error': f'cnf must be a non-empty, triply-nested list of integers in range [0, {self.token_id_max}]'}

        result = self.engine.search_docs_cnf(cnf=cnf, maxnum=maxnum, max_disp_len=max_disp_len, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)

        documents: List[DocResult] = [{'doc_ix': d.doc_ix, 'doc_len': d.doc_len, 'disp_len': d.disp_len, 'needle_offset': d.needle_offset, 'metadata': d.metadata, 'token_ids': d.token_ids, 'blocked': d.blocked} for d in result.docs]
        return {'cnt': result.cnt, 'approx': result.approx, 'idxs': result.idxs, 'documents': documents}

    def get_doc_by_rank(self, s: int, rank: int, max_disp_len: Optional[int] = None) -> InfiniGramEngineResponse[DocResult]:
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        num_shards = self.engine.get_num_shards()
        if not (type(s) == int and 0 <= s and s < num_shards):
            return {'error': f's must be an integer in range [0, {num_shards})'}
        tok_cnt = self.engine.get_tok_cnt(s=s)
        if not (type(rank) == int and 0 <= rank and rank < tok_cnt):
            return {'error': f'ptr must be an integer in range [0, {tok_cnt})'}

        result = self.engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=max_disp_len)
        return {'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked}

    def get_docs_by_ranks(self, list_of_s_and_rank: List[Tuple[int, int]], max_disp_len: Optional[int] = None) -> InfiniGramEngineResponse[List[DocResult]]:
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        num_shards = self.engine.get_num_shards()
        for s, rank in list_of_s_and_rank:
            if not (type(s) == int and 0 <= s and s < num_shards):
                return {'error': f's must be an integer in range [0, {num_shards})'}
            tok_cnt = self.engine.get_tok_cnt(s=s)
            if not (type(rank) == int and 0 <= rank and rank < tok_cnt):
                return {'error': f'ptr must be an integer in range [0, {tok_cnt})'}

        results = self.engine.get_docs_by_rank(list_of_s_and_rank=list_of_s_and_rank, max_disp_len=max_disp_len)
        return [{'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked} for result in results]

    def get_doc_by_ptr(self, s: int, ptr: int, max_disp_len: Optional[int] = None) -> InfiniGramEngineResponse[DocResult]:
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        num_shards = self.engine.get_num_shards()
        if not (type(s) == int and 0 <= s and s < num_shards):
            return {'error': f's must be an integer in range [0, {num_shards})'}
        ds_size = self.engine.get_ds_size(s=s)
        if not (type(ptr) == int and 0 <= ptr and ptr < ds_size and ptr % 2 == 0):
            return {'error': f'ptr must be an even integer in range [0, {ds_size})'}

        result = self.engine.get_doc_by_ptr(s=s, ptr=ptr, max_disp_len=max_disp_len)
        return {'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked}

    def get_docs_by_ptrs(self, list_of_s_and_ptr: List[Tuple[int, int]], max_disp_len: Optional[int] = None) -> InfiniGramEngineResponse[List[DocResult]]:
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        num_shards = self.engine.get_num_shards()
        for s, ptr in list_of_s_and_ptr:
            if not (type(s) == int and 0 <= s and s < num_shards):
                return {'error': f's must be an integer in range [0, {num_shards})'}
            ds_size = self.engine.get_ds_size(s=s)
            if not (type(ptr) == int and 0 <= ptr and ptr < ds_size and ptr % 2 == 0):
                return {'error': f'ptr must be an even integer in range [0, {ds_size})'}

        results = self.engine.get_docs_by_ptrs(list_of_s_and_ptr=list_of_s_and_ptr, max_disp_len=max_disp_len)
        return [{'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked} for result in results]

    def get_doc_by_ix(self, doc_ix: int, max_disp_len: Optional[int]=None) -> InfiniGramEngineResponse[DocResult]:
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        total_doc_cnt = self.engine.get_total_doc_cnt()
        if not (type(doc_ix) == int and 0 <= doc_ix and doc_ix < total_doc_cnt):
            return {'error': f'doc_ix must be an integer in range [0, {total_doc_cnt})'}

        result = self.engine.get_doc_by_ix(doc_ix=doc_ix, max_disp_len=max_disp_len)
        return {'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked}

    def get_docs_by_ixs(self, list_of_doc_ix: List[int], max_disp_len: Optional[int]=None) -> InfiniGramEngineResponse[List[DocResult]]:
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        total_doc_cnt = self.engine.get_total_doc_cnt()
        for doc_ix in list_of_doc_ix:
            if not (type(doc_ix) == int and 0 <= doc_ix and doc_ix < total_doc_cnt):
                return {'error': f'doc_ix must be an integer in range [0, {total_doc_cnt})'}

        results = self.engine.get_docs_by_ixs(list_of_doc_ix=list_of_doc_ix, max_disp_len=max_disp_len)
        return [{'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked} for result in results]

    def get_doc_by_rank_2(self, s: int, rank: int, needle_len: int, max_ctx_len: int) -> InfiniGramEngineResponse[DocResult]:
        if not (type(needle_len) == int and needle_len >= 0):
            return {'error': 'needle_len must be a non-negative integer'}
        if not (type(max_ctx_len) == int and max_ctx_len >= 0):
            return {'error': 'max_ctx_len must be a non-negative integer'}
        num_shards = self.engine.get_num_shards()
        if not (type(s) == int and 0 <= s and s < num_shards):
            return {'error': f's must be an integer in range [0, {num_shards})'}
        tok_cnt = self.engine.get_tok_cnt(s=s)
        if not (type(rank) == int and 0 <= rank and rank < tok_cnt):
            return {'error': f'ptr must be an integer in range [0, {tok_cnt})'}

        result = self.engine.get_doc_by_rank_2(s=s, rank=rank, needle_len=needle_len, max_ctx_len=max_ctx_len)
        return {'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked}

    def get_docs_by_ranks_2(self, requests: List[Tuple[int, int, int, int]]) -> InfiniGramEngineResponse[List[DocResult]]:
        num_shards = self.engine.get_num_shards()
        for s, rank, needle_len, max_ctx_len in requests:
            if not (type(needle_len) == int and needle_len >= 0):
                return {'error': 'needle_len must be a non-negative integer'}
            if not (type(max_ctx_len) == int and max_ctx_len >= 0):
                return {'error': 'max_ctx_len must be a non-negative integer'}
            if not (type(s) == int and 0 <= s and s < num_shards):
                return {'error': f's must be an integer in range [0, {num_shards})'}
            tok_cnt = self.engine.get_tok_cnt(s=s)
            if not (type(rank) == int and 0 <= rank and rank < tok_cnt):
                return {'error': f'ptr must be an integer in range [0, {tok_cnt})'}

        results = self.engine.get_docs_by_ranks_2(requests=requests)
        return [{'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked} for result in results]

    def get_doc_by_ptr_2(self, s: int, ptr: int, needle_len: int, max_ctx_len: int) -> InfiniGramEngineResponse[DocResult]:
        if not (type(needle_len) == int and needle_len >= 0):
            return {'error': 'needle_len must be a non-negative integer'}
        if not (type(max_ctx_len) == int and max_ctx_len >= 0):
            return {'error': 'max_ctx_len must be a non-negative integer'}
        num_shards = self.engine.get_num_shards()
        if not (type(s) == int and 0 <= s and s < num_shards):
            return {'error': f's must be an integer in range [0, {num_shards})'}
        ds_size = self.engine.get_ds_size(s=s)
        if not (type(ptr) == int and 0 <= ptr and ptr < ds_size and ptr % 2 == 0):
            return {'error': f'ptr must be an even integer in range [0, {ds_size})'}

        result = self.engine.get_doc_by_ptr_2(s=s, ptr=ptr, needle_len=needle_len, max_ctx_len=max_ctx_len)
        return {'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked}

    def get_docs_by_ptrs_2(self, requests: List[Tuple[int, int, int, int]]) -> InfiniGramEngineResponse[List[DocResult]]:
        num_shards = self.engine.get_num_shards()
        for s, ptr, needle_len, max_ctx_len in requests:
            if not (type(needle_len) == int and needle_len >= 0):
                return {'error': 'needle_len must be a non-negative integer'}
            if not (type(max_ctx_len) == int and max_ctx_len >= 0):
                return {'error': 'max_ctx_len must be a non-negative integer'}
            if not (type(s) == int and 0 <= s and s < num_shards):
                return {'error': f's must be an integer in range [0, {num_shards})'}
            ds_size = self.engine.get_ds_size(s=s)
            if not (type(ptr) == int and 0 <= ptr and ptr < ds_size and ptr % 2 == 0):
                return {'error': f'ptr must be an even integer in range [0, {ds_size})'}

        results = self.engine.get_docs_by_ptrs_2(requests=requests)
        return [{'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked} for result in results]

    def get_doc_by_ix_2(self, doc_ix: int, max_ctx_len: int) -> InfiniGramEngineResponse[DocResult]:
        if not (type(max_ctx_len) == int and max_ctx_len >= 0):
            return {'error': 'max_ctx_len must be a non-negative integer'}
        total_doc_cnt = self.engine.get_total_doc_cnt()
        if not (type(doc_ix) == int and 0 <= doc_ix and doc_ix < total_doc_cnt):
            return {'error': f'doc_ix must be an integer in range [0, {total_doc_cnt})'}

        result = self.engine.get_doc_by_ix_2(doc_ix=doc_ix, max_ctx_len=max_ctx_len)
        return {'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked}

    def get_docs_by_ixs_2(self, requests: List[Tuple[int, int]]) -> InfiniGramEngineResponse[List[DocResult]]:
        total_doc_cnt = self.engine.get_total_doc_cnt()
        for doc_ix, max_ctx_len in requests:
            if not (type(max_ctx_len) == int and max_ctx_len >= 0):
                return {'error': 'max_ctx_len must be a non-negative integer'}
            if not (type(doc_ix) == int and 0 <= doc_ix and doc_ix < total_doc_cnt):
                return {'error': f'doc_ix must be an integer in range [0, {total_doc_cnt})'}

        results = self.engine.get_docs_by_ixs_2(requests=requests)
        return [{'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked} for result in results]

    def get_total_doc_cnt(self) -> int:
        return self.engine.get_total_doc_cnt()

    def creativity(self, input_ids: QueryIdsType) -> InfiniGramEngineResponse[CreativityResponse]:
        if not self.check_query_ids(input_ids, allow_empty=True):
            return {'error': f'input_ids must be a list of integers in range [0, {self.token_id_max}]'}
        result = self.engine.creativity(input_ids=input_ids)
        return {'rs': result.rs}

    def attribute(self, input_ids: QueryIdsType, delim_ids: Iterable[int], min_len: int, max_cnt: int, enforce_bow: bool) -> AttributionResponse:
        if not self.check_query_ids(input_ids, allow_empty=True):
            return {'error': f'input_ids must be a list of integers in range [0, {self.token_id_max}]'}
        if not self.check_query_ids(delim_ids, allow_empty=True):
            return {'error': f'delim_ids must be a list of integers in range [0, {self.token_id_max}]'}
        if not (type(min_len) == int and min_len >= 0):
            return {'error': 'min_len must be a non-negative integer'}
        if not (type(max_cnt) == int and max_cnt > 0):
            return {'error': 'max_cnt must be a positive integer'}
        if not (type(enforce_bow) == bool):
            return {'error': 'enforce_bow must be a boolean'}

        result = self.engine.attribute(input_ids=input_ids, delim_ids=delim_ids, min_len=min_len, max_cnt=max_cnt, enforce_bow=enforce_bow)

        return {
            "spans": [
                AttributionSpan(
                    l=span.l,
                    r=span.r,
                    length=span.length,
                    count=span.count,
                    unigram_logprob_sum=span.unigram_logprob_sum,
                    docs=[
                        AttributionDoc(s=doc.s, ptr=doc.ptr)
                        for doc in span.docs
                    ],
                   # The spans we get back from result aren't typed so Pylance doesn't like us feeding them into AttributionSpan
                )  # type: ignore
                for span in result.spans
            ]
        }

class InfiniGramEngineDiff(InfiniGramEngine):

    def __init__(self, index_dir: Iterable[str] | str, index_dir_diff: Iterable[str] | str, eos_token_id: int, vocab_size=65535, version=4, token_dtype='u16',
                 load_to_ram=False, ds_prefetch_depth=1, sa_prefetch_depth=3, od_prefetch_depth=3,
                 bow_ids_path: str = None, attribution_block_size: int = 512, precompute_unigram_logprobs: bool = False,
                 prev_shards_by_index_dir = {},
                 max_support=1000, max_clause_freq=50000, max_diff_tokens=100, maxnum=1, max_disp_len=1000,
                 ) -> None:

        assert sys.byteorder == 'little', 'This code is designed to run on little-endian machines only!'

        if type(index_dir) == str:
            index_dir = [index_dir]
        assert type(index_dir) == list and all(type(d) == str for d in index_dir)
        if type(index_dir_diff) == str:
            index_dir_diff = [index_dir_diff]
        assert type(index_dir_diff) == list and all(type(d) == str for d in index_dir_diff)
        assert type(eos_token_id) == int and 0 <= eos_token_id and eos_token_id < vocab_size
        assert type(load_to_ram) == bool
        assert type(ds_prefetch_depth) == int and ds_prefetch_depth >= 0
        assert type(sa_prefetch_depth) == int and sa_prefetch_depth >= ds_prefetch_depth
        assert type(od_prefetch_depth) == int and od_prefetch_depth >= 0
        assert type(max_support) == int and max_support > 0
        assert type(max_clause_freq) == int and max_clause_freq > 0
        assert type(max_diff_tokens) == int and max_diff_tokens > 0
        assert type(maxnum) == int and maxnum > 0
        assert type(max_disp_len) == int and max_disp_len > 0

        self.max_support = max_support
        self.max_clause_freq = max_clause_freq
        self.max_diff_tokens = max_diff_tokens
        self.maxnum = maxnum
        self.max_disp_len = max_disp_len

        bow_ids = set()
        if bow_ids_path is not None:
            try:
                with open(bow_ids_path, 'r') as f:
                    for line in f:
                        bow_ids.add(int(line.strip()))
            except Exception as e:
                print(f"Error reading bow_ids_path: {e}")
                raise e

        if token_dtype == 'u8':
            self.token_id_max = 2**8 - 1
            engine_class = cpp_engine.EngineDiff_U8
        elif token_dtype == 'u16':
            self.token_id_max = 2**16 - 1
            engine_class = cpp_engine.EngineDiff_U16
        elif token_dtype == 'u32':
            self.token_id_max = 2**32 - 1
            engine_class = cpp_engine.EngineDiff_U32
        else:
            raise ValueError(f'Unsupported token dtype: {token_dtype}')
        self.engine = engine_class(index_dir, index_dir_diff, eos_token_id, vocab_size, version, load_to_ram, ds_prefetch_depth, sa_prefetch_depth, od_prefetch_depth, bow_ids, attribution_block_size, precompute_unigram_logprobs, prev_shards_by_index_dir)

    def get_docs_by_ptrs_2_grouped(self, requests: List[GetDocsByPtrsRequestWithTakedown]) -> InfiniGramEngineResponse[List[List[DocResult]]]:
        num_shards = self.engine.get_num_shards()
        for request in requests:
            needle_len = request['needle_len']
            max_ctx_len = request['max_ctx_len']
            if not (type(needle_len) == int and needle_len >= 0):
                return {'error': 'needle_len must be a non-negative integer'}
            if not (type(max_ctx_len) == int and max_ctx_len >= 0):
                return {'error': 'max_ctx_len must be a non-negative integer'}
            for doc in request['docs']:
                s = doc['s']
                ptr = doc['ptr']
                if not (type(s) == int and 0 <= s and s < num_shards):
                    return {'error': f's must be an integer in range [0, {num_shards})'}
                ds_size = self.engine.get_ds_size(s=s)
                if not (type(ptr) == int and 0 <= ptr and ptr < ds_size and ptr % 2 == 0):
                    return {'error': f'ptr must be an even integer in range [0, {ds_size})'}

        resultss = self.engine.get_docs_by_ptrs_2_grouped(requests=[
            (
                [(doc['s'], doc['ptr']) for doc in request['docs']],
                request['span_ids'],
                request['needle_len'],
                request['max_ctx_len'],
            ) for request in requests])
        return [[{'doc_ix': result.doc_ix, 'doc_len': result.doc_len, 'disp_len': result.disp_len, 'needle_offset': result.needle_offset, 'metadata': result.metadata, 'token_ids': result.token_ids, 'blocked': result.blocked} for result in results] for results in resultss]
