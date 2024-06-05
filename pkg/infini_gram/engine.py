import sys
from .cpp_engine import Config, NGramLanguageModeling, NGramLanguageModelingUnion

class InfiniGramEngine:

    def __init__(self, index_dir, eos_token_id, **kwargs):

        assert sys.byteorder == 'little', 'This code is designed to run on little-endian machines only!'

        MAX_CNT_FOR_NTD = kwargs.get('MAX_CNT_FOR_NTD', 1000)
        assert type(MAX_CNT_FOR_NTD) == int and MAX_CNT_FOR_NTD > 0
        MAX_OUTPUT_DOC_TOKENS = kwargs.get('MAX_OUTPUT_DOC_TOKENS', 5000)
        assert type(MAX_OUTPUT_DOC_TOKENS) == int and MAX_OUTPUT_DOC_TOKENS > 0
        MAX_CLAUSE_FREQ_PER_SHARD = kwargs.get('MAX_CLAUSE_FREQ_PER_SHARD', 50000)
        assert type(MAX_CLAUSE_FREQ_PER_SHARD) == int and MAX_CLAUSE_FREQ_PER_SHARD > 0
        MAX_DIFF_TOKENS = kwargs.get('MAX_DIFF_TOKENS', 100)
        assert type(MAX_DIFF_TOKENS) == int and MAX_DIFF_TOKENS > 0
        ds_prefetch_depth = kwargs.get('ds_prefetch_depth', 1)
        assert type(ds_prefetch_depth) == int and ds_prefetch_depth > 0
        sa_prefetch_depth = kwargs.get('sa_prefetch_depth', 3)
        assert type(sa_prefetch_depth) == int and sa_prefetch_depth > ds_prefetch_depth
        od_prefetch_depth = kwargs.get('od_prefetch_depth', 3)
        assert type(od_prefetch_depth) == int and od_prefetch_depth > 0
        config = Config(
            MAX_CNT_FOR_NTD, MAX_OUTPUT_DOC_TOKENS, MAX_CLAUSE_FREQ_PER_SHARD, MAX_DIFF_TOKENS, ds_prefetch_depth, sa_prefetch_depth, od_prefetch_depth,
        )

        assert type(eos_token_id) == int and 0 <= eos_token_id and eos_token_id < 65535

        if type(index_dir) == str:
            self.lm = NGramLanguageModeling(index_dir, eos_token_id, config)
        elif type(index_dir) == list and all(type(d) == str for d in index_dir):
            self.lm = NGramLanguageModelingUnion(index_dir, eos_token_id, config)
        else:
            raise ValueError('index_dir must be a string or a list of strings')

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
        if self.check_query_ids(query_ids, allow_empty=True):
            result = self.lm.count(input_ids=query_ids)
        else: # cnf
            if not self.check_cnf(query_ids):
                return {'error': 'query_ids must be a list (or a triply-nested list, i.e., a CNF) of integers in range [0, 65535)'}
            result = self.lm.count_cnf(cnf=query_ids)
        return {'count': result.count, 'approx': result.approx}

    def prob(self, query_ids):
        if not self.check_query_ids(query_ids, allow_empty=False):
            return {'error': 'Query must be non-empty. If you are supplying query_ids, it must be a non-empty list of integers in range [0, 65535)'}
        result = self.lm.prob(prompt_ids=query_ids[:-1], cont_id=query_ids[-1])
        return {'prompt_cnt': result.prompt_cnt, 'cont_cnt': result.cont_cnt, 'prob': result.prob}

    def ntd(self, query_ids):
        if not self.check_query_ids(query_ids, allow_empty=True):
            return {'error': 'query_ids must be a list of integers in range [0, 65535)'}
        result = self.lm.ntd(prompt_ids=query_ids)
        result_by_token_id = {token_id: {'cont_cnt': r.cont_cnt, 'prob': r.prob} for token_id, r in result.result_by_token_id.items()}
        return {'prompt_cnt': result.prompt_cnt, 'result_by_token_id': result_by_token_id, 'approx': result.approx}

    def infgram_prob(self, query_ids):
        if not self.check_query_ids(query_ids, allow_empty=False):
            return {'error': 'Query must be non-empty. If you are supplying query_ids, it must be a non-empty list of integers in range [0, 65535)'}
        result = self.lm.infgram_prob(prompt_ids=query_ids[:-1], cont_id=query_ids[-1])
        return {'prompt_cnt': result.prompt_cnt, 'cont_cnt': result.cont_cnt, 'prob': result.prob, 'suffix_len': result.suffix_len}

    def infgram_ntd(self, query_ids):
        if not self.check_query_ids(query_ids, allow_empty=True):
            return {'error': 'query_ids must be a list of integers in range [0, 65535)'}
        result = self.lm.infgram_ntd(prompt_ids=query_ids)
        result_by_token_id = {token_id: {'cont_cnt': r.cont_cnt, 'prob': r.prob} for token_id, r in result.result_by_token_id.items()}
        return {'prompt_cnt': result.prompt_cnt, 'result_by_token_id': result_by_token_id, 'approx': result.approx, 'suffix_len': result.suffix_len}

    def search_docs(self, query_ids, maxnum):
        if self.check_query_ids(query_ids, allow_empty=False):
            cnf = [[query_ids]]
        elif self.check_cnf(query_ids):
            cnf = query_ids
        else:
            return {'error': 'Query must be non-empty. If you are supplying query_ids, it must be a list (or a triply-nested list, i.e., a CNF) of integers in range [0, 65535)'}
        if not (type(maxnum) == int and maxnum > 0):
            return {'error': 'maxnum must be a positive integer'}
        result = self.lm.search_docs(cnf=cnf, maxnum=maxnum)

        if result.cnt == 0:
            if len(cnf) == 1:
                if len(cnf[0]) == 1:
                    error = 'The query term is not found in the corpus!'
                else:
                    error = 'None of the query terms is found in the corpus!'
            else:
                if 0 in result.cnt_by_clause:
                    error = f'None of the terms in the {result.cnt_by_clause.index(0)}-th disjunctive clause is found in the corpus!'
                else:
                    error = 'Query is not found in the corpus! Try relaxing the constraints.'
            return {'error': error}

        message = f'{"Approximately " if result.approx else ""}{result.cnt} occurrences found. Displaying the documents of occurrences #{result.idxs}'
        documents = [{'token_ids': d.token_ids, 'doc_ix': d.doc_ix, 'doc_len': d.doc_len, 'disp_len': d.disp_len} for d in result.documents]
        return {'message': message, 'documents': documents}
