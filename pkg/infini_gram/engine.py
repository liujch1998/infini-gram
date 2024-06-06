import sys
from .cpp_engine import InfiniGramEngine as Engine

class InfiniGramEngine:

    def __init__(self, index_dir, eos_token_id,
                 load_to_ram=False, ds_prefetch_depth=1, sa_prefetch_depth=3, od_prefetch_depth=3,
                 max_support=1000, max_clause_freq=50000, max_diff_tokens=100, maxnum=1000, max_disp_len=1000,
                 ):

        assert sys.byteorder == 'little', 'This code is designed to run on little-endian machines only!'

        if type(index_dir) == str:
            index_dir = [index_dir]
        assert type(index_dir) == list and all(type(d) == str for d in index_dir)
        assert type(eos_token_id) == int and 0 <= eos_token_id and eos_token_id < 65535
        assert type(load_to_ram) == bool
        assert type(ds_prefetch_depth) == int and ds_prefetch_depth > 0
        assert type(sa_prefetch_depth) == int and sa_prefetch_depth > ds_prefetch_depth
        assert type(od_prefetch_depth) == int and od_prefetch_depth > 0
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

        self.engine = Engine(index_dir=index_dir, eos_token_id=eos_token_id,
                             load_to_ram=load_to_ram, ds_prefetch_depth=ds_prefetch_depth, sa_prefetch_depth=sa_prefetch_depth, od_prefetch_depth=od_prefetch_depth)

    def check_query_ids(self, query_ids, allow_empty):
        if not (type(query_ids) == list and (allow_empty or len(query_ids) > 0)):
            return False
        for q in query_ids:
            if not (type(q) == int and 0 <= q and q < 65535):
                return False
        return True

    def check_cnf(self, cnf):
        if not (type(cnf) == list and len(cnf) > 0):
            return False
        for disj_clause in cnf:
            if not (type(disj_clause) == list and len(disj_clause) > 0):
                return False
            for query_ids in disj_clause:
                if not (type(query_ids) == list and len(query_ids) > 0):
                    return False
                for q in query_ids:
                    if not (type(q) == int and 0 <= q and q < 65535):
                        return False
        return True

    def count(self, query_ids):
        if not self.check_query_ids(query_ids, allow_empty=True):
            return {'error': 'query_ids must be a list of integers in range [0, 65535)'}
        result = self.lm.count(input_ids=query_ids)
        return {'count': result.count, 'approx': result.approx}

    def count_cnf(self, query_ids, max_clause_freq=None, max_diff_tokens=None):
        if max_clause_freq is None:
            max_clause_freq = self.max_clause_freq
        if max_diff_tokens is None:
            max_diff_tokens = self.max_diff_tokens
        if not (type(max_clause_freq) == int and max_clause_freq > 0):
            return {'error': 'max_clause_freq must be a positive integer'}
        if not (type(max_diff_tokens) == int and max_diff_tokens > 0):
            return {'error': 'max_diff_tokens must be a positive integer'}
        if not self.check_cnf(query_ids):
            return {'error': 'query_ids must be a non-empty, triply-nested list of integers in range [0, 65535)'}
        result = self.engine.count_cnf(cnf=query_ids, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)
        return {'count': result.count, 'approx': result.approx}

    def prob(self, query_ids):
        if not self.check_query_ids(query_ids, allow_empty=False):
            return {'error': 'Query must be non-empty. If you are supplying query_ids, it must be a non-empty list of integers in range [0, 65535)'}
        result = self.engine.prob(prompt_ids=query_ids[:-1], cont_id=query_ids[-1])
        return {'prompt_cnt': result.prompt_cnt, 'cont_cnt': result.cont_cnt, 'prob': result.prob}

    def ntd(self, query_ids, max_support=None):
        if max_support is None:
            max_support = self.max_support
        if not (type(max_support) == int and max_support > 0):
            return {'error': 'max_support must be a positive integer'}
        if not self.check_query_ids(query_ids, allow_empty=True):
            return {'error': 'query_ids must be a list of integers in range [0, 65535)'}
        result = self.engine.ntd(prompt_ids=query_ids)
        result_by_token_id = {token_id: {'cont_cnt': r.cont_cnt, 'prob': r.prob} for token_id, r in result.result_by_token_id.items()}
        return {'prompt_cnt': result.prompt_cnt, 'result_by_token_id': result_by_token_id, 'approx': result.approx}

    def infgram_prob(self, query_ids):
        if not self.check_query_ids(query_ids, allow_empty=False):
            return {'error': 'Query must be non-empty. If you are supplying query_ids, it must be a non-empty list of integers in range [0, 65535)'}
        result = self.engine.infgram_prob(prompt_ids=query_ids[:-1], cont_id=query_ids[-1])
        return {'prompt_cnt': result.prompt_cnt, 'cont_cnt': result.cont_cnt, 'prob': result.prob, 'suffix_len': result.suffix_len}

    def infgram_ntd(self, query_ids, max_support=None):
        if max_support is None:
            max_support = self.max_support
        if not (type(max_support) == int and max_support > 0):
            return {'error': 'max_support must be a positive integer'}
        if not self.check_query_ids(query_ids, allow_empty=True):
            return {'error': 'query_ids must be a list of integers in range [0, 65535)'}
        result = self.engine.infgram_ntd(prompt_ids=query_ids)
        result_by_token_id = {token_id: {'cont_cnt': r.cont_cnt, 'prob': r.prob} for token_id, r in result.result_by_token_id.items()}
        return {'prompt_cnt': result.prompt_cnt, 'result_by_token_id': result_by_token_id, 'approx': result.approx, 'suffix_len': result.suffix_len}

    def search_docs(self, query_ids, maxnum=None, max_disp_len=None):
        if maxnum is None:
            maxnum = self.maxnum
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not (type(maxnum) == int and maxnum > 0):
            return {'error': 'maxnum must be a positive integer'}
        if not (type(max_disp_len) == int and max_disp_len > 0):
            return {'error': 'max_disp_len must be a positive integer'}
        if not self.check_query_ids(query_ids, allow_empty=True):
            return {'error': 'query_ids must be a list of integers in range [0, 65535)'}

        result = self.lm.search_docs(query_ids=query_ids, maxnum=maxnum, max_disp_len=max_disp_len)

        if result.cnt == 0:
            return {'message': '0 occurrences found', 'documents': []}
        message = f'{"Approximately " if result.approx else ""}{result.cnt} occurrences found. Displaying the documents of occurrences #{result.idxs}'
        documents = [{'doc_ix': d.doc_ix, 'doc_len': d.doc_len, 'disp_len': d.disp_len, 'metadata': d.metadata, 'token_ids': d.token_ids} for d in result.documents]
        return {'message': message, 'documents': documents}

    def search_docs_cnf(self, query_ids, maxnum=None, max_disp_len=None, max_clause_freq=None, max_diff_tokens=None):
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
        if not self.check_cnf(query_ids):
            return {'error': 'query_ids must be a non-empty, triply-nested list of integers in range [0, 65535)'}

        result = self.engine.search_docs_cnf(cnf=query_ids, maxnum=maxnum, max_disp_len=max_disp_len, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)

        if result.cnt == 0:
            return {'message': '0 occurrences found', 'documents': []}
        message = f'{"Approximately " if result.approx else ""}{result.cnt} occurrences found. Displaying the documents of occurrences #{result.idxs}'
        documents = [{'doc_ix': d.doc_ix, 'doc_len': d.doc_len, 'disp_len': d.disp_len, 'metadata': d.metadata, 'token_ids': d.token_ids} for d in result.documents]
        return {'message': message, 'documents': documents}

    # TODO: expose find(), find_cnf(), get_doc_by_rank(), get_doc_by_ptr()
