import argparse
from flask import Flask, jsonify, request
import json
import os
import requests
import sys
import time
import traceback
from transformers import AutoTokenizer
sys.path.append('../pkg')
from infini_gram.engine import InfiniGramEngine

parser = argparse.ArgumentParser()
parser.add_argument('--MODE', type=str, default='api', choices=['api', 'dev', 'demo'])
parser.add_argument('--FLASK_PORT', type=int, default=5000)
parser.add_argument('--CONFIG_FILE', type=str, default='api_config.json')
parser.add_argument('--LOG_PATH', type=str, default=None)
# API limits
parser.add_argument('--MAX_QUERY_CHARS', type=int, default=1000)
parser.add_argument('--MAX_QUERY_TOKENS', type=int, default=500)
parser.add_argument('--MAX_CLAUSES_PER_CNF', type=int, default=4)
parser.add_argument('--MAX_TERMS_PER_CLAUSE', type=int, default=4)
# the following values must be no smaller than the defaults in engine.py
parser.add_argument('--MAX_SUPPORT', type=int, default=1000)
parser.add_argument('--MAX_CLAUSE_FREQ', type=int, default=500000)
parser.add_argument('--MAX_DIFF_TOKENS', type=int, default=1000)
parser.add_argument('--MAXNUM', type=int, default=10)
parser.add_argument('--MAX_DISP_LEN', type=int, default=10000)
args = parser.parse_args()

DOLMA_API_URL = os.environ.get(f'DOLMA_API_URL_{args.MODE.upper()}', None)

prev_shards_by_index_dir = {}

class Processor:

    def __init__(self, config):
        assert 'index_dir' in config and 'tokenizer' in config

        self.tokenizer_type = config['tokenizer']
        if self.tokenizer_type == 'gpt2':
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', add_bos_token=False, add_eos_token=False)
        elif self.tokenizer_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
        elif self.tokenizer_type == 'olmo':
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf", add_bos_token=False, add_eos_token=False)
        elif self.tokenizer_type == 'gptneox':
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-6.9b', add_bos_token=False, add_eos_token=False)
        else:
            raise NotImplementedError

        global prev_shards_by_index_dir
        self.engine = InfiniGramEngine(index_dir=config['index_dir'], eos_token_id=self.tokenizer.eos_token_id, ds_prefetch_depth=0, sa_prefetch_depth=0, od_prefetch_depth=0, prev_shards_by_index_dir=prev_shards_by_index_dir)
        prev_shards_by_index_dir = {
            **prev_shards_by_index_dir,
            **self.engine.get_new_shards_by_index_dir(),
        }

    def tokenize(self, query):
        if self.tokenizer_type == 'gpt2':
            if query != '':
                query = ' ' + query
            input_ids = self.tokenizer.encode(query)
        elif self.tokenizer_type == 'llama':
            input_ids = self.tokenizer.encode(query)
            if len(input_ids) > 0 and input_ids[0] == 29871:
                input_ids = input_ids[1:]
        elif self.tokenizer_type == 'olmo':
            if query != '':
                query = ' ' + query
            input_ids = self.tokenizer.encode(query)
        elif self.tokenizer_type == 'gptneox':
            if query != '':
                query = ' ' + query
            input_ids = self.tokenizer.encode(query)
        else:
            raise NotImplementedError
        return input_ids

    def process(self, query_type, query, query_ids, **kwargs):
        '''
        Preconditions: query_type is valid, and exactly one of query and query_ids exists.
        Postconditions: query_ids is a list of integers, or a triply-nested list of integers.
        Max input lengths, element types, and integer bounds are checked here, but min input lengths are not checked.
        '''
        # parse query
        if query is not None:
            if type(query) != str:
                return {'error': f'query must be a string!'}
            if len(query) > args.MAX_QUERY_CHARS:
                return {'error': f'Please limit your input to <= {args.MAX_QUERY_CHARS} characters!'}
            if not (' AND ' in query or ' OR ' in query): # simple query
                query_ids = self.tokenize(query)
            else: # CNF query
                clauses = query.split(' AND ')
                termss = [clause.split(' OR ') for clause in clauses]
                query_ids = [[self.tokenize(term) for term in terms] for terms in termss]

        # validate query_ids
        if type(query_ids) == list and all(type(input_id) == int for input_id in query_ids): # simple query
            if len(query_ids) > args.MAX_QUERY_TOKENS:
                return {'error': f'Please limit your input to <= {args.MAX_QUERY_TOKENS} tokens!'}
            if any(input_id < 0 or (input_id >= self.tokenizer.vocab_size and input_id != 65535) for input_id in query_ids):
                return {'error': f'Some item(s) in your query_ids are out-of-range!'}
            tokens = self.tokenizer.convert_ids_to_tokens(query_ids)
            is_cnf = False
        elif type(query_ids) == list and all([type(clause) == list and all([type(term) == list and all([type(input_id) == int for input_id in term]) for term in clause]) for clause in query_ids]):
            if sum(sum(len(term) for term in clause) for clause in query_ids) > args.MAX_QUERY_TOKENS:
                return {'error': f'Please limit your input to <= {args.MAX_QUERY_TOKENS} tokens!'}
            if len(query_ids) > args.MAX_CLAUSES_PER_CNF:
                return {'error': f'Please enter at most {args.MAX_CLAUSES_PER_CNF} disjunctive clauses!'}
            for clause in query_ids:
                if len(clause) > args.MAX_TERMS_PER_CLAUSE:
                    return {'error': f'Please enter at most {args.MAX_TERMS_PER_CLAUSE} terms in each disjunctive clause!'}
                for term in clause:
                    if any(input_id < 0 or (input_id >= self.tokenizer.vocab_size and input_id != 65535) for input_id in term):
                        return {'error': f'Some item(s) in your query_ids are out-of-range!'}
            tokens = [[self.tokenizer.convert_ids_to_tokens(term) for term in clause] for clause in query_ids]
            is_cnf = True
        else:
            return {'error': f'query_ids must be a list of integers, or a triply-nested list of integers!'}

        start_time = time.time()
        if is_cnf and query_type in ['count', 'search_docs']:
            result = getattr(self, f'{query_type}_cnf')(query_ids, **kwargs)
        else:
            result = getattr(self, query_type)(query_ids, **kwargs)
        end_time = time.time()
        result['latency'] = (end_time - start_time) * 1000
        result['token_ids'] = query_ids
        result['tokens'] = tokens

        return result

    def count(self, query_ids):
        return self.engine.count(input_ids=query_ids)

    def count_cnf(self, query_ids, max_clause_freq=None, max_diff_tokens=None):
        if max_clause_freq is not None and not (type(max_clause_freq) == int and 0 < max_clause_freq and max_clause_freq <= args.MAX_CLAUSE_FREQ):
            return {'error': f'max_clause_freq must be an integer in [1, {args.MAX_CLAUSE_FREQ}]!'}
        if max_diff_tokens is not None and not (type(max_diff_tokens) == int and 0 < max_diff_tokens and max_diff_tokens <= args.MAX_DIFF_TOKENS):
            return {'error': f'max_diff_tokens must be an integer in [1, {args.MAX_DIFF_TOKENS}]!'}
        return self.engine.count_cnf(cnf=query_ids, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)

    def prob(self, query_ids):
        if len(query_ids) == 0:
            return {'error': f'Please provide a non-empty query!'}
        return self.engine.prob(prompt_ids=query_ids[:-1], cont_id=query_ids[-1])

    def ntd(self, query_ids, max_support=None):
        if max_support is not None and not (type(max_support) == int and 0 < max_support and max_support <= args.MAX_SUPPORT):
            return {'error': f'max_support must be an integer in [1, {args.MAX_SUPPORT}]!'}
        result = self.engine.ntd(prompt_ids=query_ids, max_support=max_support)
        if 'result_by_token_id' in result:
            for token_id in result['result_by_token_id']:
                result['result_by_token_id'][token_id]['token'] = self.tokenizer.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')
        return result

    def infgram_prob(self, query_ids):
        if len(query_ids) == 0:
            return {'error': f'Please provide a non-empty query!'}
        result = self.engine.infgram_prob(prompt_ids=query_ids[:-1], cont_id=query_ids[-1])
        if 'suffix_len' in result:
            result['longest_suffix'] = self.tokenizer.decode(query_ids[-result['suffix_len']-1:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return result

    def infgram_ntd(self, query_ids, max_support=None):
        if max_support is not None and not (type(max_support) == int and 0 < max_support and max_support <= args.MAX_SUPPORT):
            return {'error': f'max_support must be an integer in [1, {args.MAX_SUPPORT}]!'}
        result = self.engine.infgram_ntd(prompt_ids=query_ids, max_support=max_support)
        if 'result_by_token_id' in result:
            for token_id in result['result_by_token_id']:
                result['result_by_token_id'][token_id]['token'] = self.tokenizer.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')
        if 'suffix_len' in result:
            result['longest_suffix'] = self.tokenizer.decode(query_ids[-result['suffix_len']:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return result

    def search_docs(self, query_ids, maxnum=None, max_disp_len=None):
        if maxnum is not None and not (type(maxnum) == int and 0 < maxnum and maxnum <= args.MAXNUM):
            return {'error': f'maxnum must be an integer in [1, {args.MAXNUM}]!'}
        if max_disp_len is not None and not (type(max_disp_len) == int and 0 < max_disp_len and max_disp_len <= args.MAX_DISP_LEN):
            return {'error': f'max_disp_len must be an integer in [1, {args.MAX_DISP_LEN}]!'}

        result = self.engine.search_docs(input_ids=query_ids, maxnum=maxnum, max_disp_len=max_disp_len)

        if 'error' in result:
            return result

        if result['cnt'] == 0:
            result['message'] = '0 occurrences found'
        else:
            result['message'] = f'{"Approximately " if result["approx"] else ""}{result["cnt"]} occurrences found. Displaying the documents of occurrences #{result["idxs"]}'
        for document in result['documents']:
            token_ids = document['token_ids']
            spans = [(token_ids, None)]
            if len(query_ids) > 0:
                needle = query_ids
                new_spans = []
                for span in spans:
                    if span[1] is not None:
                        new_spans.append(span)
                    else:
                        haystack = span[0]
                        new_spans += self._replace(haystack, needle, label='0')
                spans = new_spans
            spans = [(self.tokenizer.decode(token_ids), d) for (token_ids, d) in spans]
            document['spans'] = spans
        return result

    def search_docs_cnf(self, query_ids, maxnum=None, max_disp_len=None, max_clause_freq=None, max_diff_tokens=None):
        if maxnum is not None and not (type(maxnum) == int and 0 < maxnum and maxnum <= args.MAXNUM):
            return {'error': f'maxnum must be an integer in [1, {args.MAXNUM}]!'}
        if max_disp_len is not None and not (type(max_disp_len) == int and 0 < max_disp_len and max_disp_len <= args.MAX_DISP_LEN):
            return {'error': f'max_disp_len must be an integer in [1, {args.MAX_DISP_LEN}]!'}
        if max_clause_freq is not None and not (type(max_clause_freq) == int and 0 < max_clause_freq and max_clause_freq <= args.MAX_CLAUSE_FREQ):
            return {'error': f'max_clause_freq must be an integer in [1, {args.MAX_CLAUSE_FREQ}]!'}
        if max_diff_tokens is not None and not (type(max_diff_tokens) == int and 0 < max_diff_tokens and max_diff_tokens <= args.MAX_DIFF_TOKENS):
            return {'error': f'max_diff_tokens must be an integer in [1, {args.MAX_DIFF_TOKENS}]!'}

        result = self.engine.search_docs_cnf(cnf=query_ids, maxnum=maxnum, max_disp_len=max_disp_len, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)

        if 'error' in result:
            return result

        if result['cnt'] == 0:
            result['message'] = '0 occurrences found'
        else:
            result['message'] = f'{"Approximately " if result["approx"] else ""}{result["cnt"]} occurrences found. Displaying the documents of occurrences #{result["idxs"]}'
        cnf = query_ids
        for document in result['documents']:
            token_ids = document['token_ids']
            spans = [(token_ids, None)]
            for d, clause in enumerate(cnf):
                for needle in clause:
                    new_spans = []
                    for span in spans:
                        if span[1] is not None:
                            new_spans.append(span)
                        else:
                            haystack = span[0]
                            new_spans += self._replace(haystack, needle, label=f'{d}')
                    spans = new_spans
            spans = [(self.tokenizer.decode(token_ids), d) for (token_ids, d) in spans]
            document['spans'] = spans
        return result

    def _replace(self, haystack, needle, label):
        spans = []
        while True:
            pos = -1
            for p in range(len(haystack) - len(needle) + 1):
                if haystack[p:p+len(needle)] == needle:
                    pos = p
                    break
            if pos == -1:
                break
            if pos > 0:
                spans.append((haystack[:pos], None))
            spans.append((haystack[pos:pos+len(needle)], label))
            haystack = haystack[pos+len(needle):]
        if len(haystack) > 0:
            spans.append((haystack, None))
        return spans

    def find(self, query_ids):
        return self.engine.find(input_ids=query_ids)

    def find_cnf(self, query_ids, max_clause_freq=None, max_diff_tokens=None):
        if max_clause_freq is not None and not (type(max_clause_freq) == int and 0 < max_clause_freq and max_clause_freq <= args.MAX_CLAUSE_FREQ):
            return {'error': f'max_clause_freq must be an integer in [1, {args.MAX_CLAUSE_FREQ}]!'}
        if max_diff_tokens is not None and not (type(max_diff_tokens) == int and 0 < max_diff_tokens and max_diff_tokens <= args.MAX_DIFF_TOKENS):
            return {'error': f'max_diff_tokens must be an integer in [1, {args.MAX_DIFF_TOKENS}]!'}
        return self.engine.find_cnf(cnf=query_ids, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)

    def get_doc_by_rank(self, query_ids, s, rank, max_disp_len=None):
        if max_disp_len is not None and not (type(max_disp_len) == int and 0 < max_disp_len and max_disp_len <= args.MAX_DISP_LEN):
            return {'error': f'max_disp_len must be an integer in [1, {args.MAX_DISP_LEN}]!'}
        result = self.engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=max_disp_len)
        if 'error' in result:
            return result

        token_ids = result['token_ids']
        spans = [(token_ids, None)]
        if len(query_ids) > 0:
            needle = query_ids
            new_spans = []
            for span in spans:
                if span[1] is not None:
                    new_spans.append(span)
                else:
                    haystack = span[0]
                    new_spans += self._replace(haystack, needle, label='0')
            spans = new_spans
        spans = [(self.tokenizer.decode(token_ids), d) for (token_ids, d) in spans]
        result['spans'] = spans
        return result

    def get_doc_by_ptr(self, query_ids, s, ptr, max_disp_len=None):
        if max_disp_len is not None and not (type(max_disp_len) == int and 0 < max_disp_len and max_disp_len <= args.MAX_DISP_LEN):
            return {'error': f'max_disp_len must be an integer in [1, {args.MAX_DISP_LEN}]!'}
        result = self.engine.get_doc_by_ptr(s=s, ptr=ptr, max_disp_len=max_disp_len)
        if 'error' in result:
            return result

        cnf = query_ids
        token_ids = result['token_ids']
        spans = [(token_ids, None)]
        for d, clause in enumerate(cnf):
            for needle in clause:
                new_spans = []
                for span in spans:
                    if span[1] is not None:
                        new_spans.append(span)
                    else:
                        haystack = span[0]
                        new_spans += self._replace(haystack, needle, label=f'{d}')
                spans = new_spans
        spans = [(self.tokenizer.decode(token_ids), d) for (token_ids, d) in spans]
        result['spans'] = spans
        return result

PROCESSOR_BY_INDEX = {}
with open(args.CONFIG_FILE) as f:
    configs = json.load(f)
    for config in configs:
        PROCESSOR_BY_INDEX[config['name']] = Processor(config)

# save log under home directory
if args.LOG_PATH is None:
    args.LOG_PATH = f'/home/ubuntu/logs/flask_{args.MODE}.log'
log = open(args.LOG_PATH, 'a')
app = Flask(__name__)

@app.route('/', methods=['POST'])
def query():
    data = request.json
    print(data)
    log.write(json.dumps(data) + '\n')
    log.flush()

    index = data['corpus'] if 'corpus' in data else (data['index'] if 'index' in data else None)
    if DOLMA_API_URL is not None:
        try:
            response = requests.post(DOLMA_API_URL, json=data, timeout=30)
        except requests.exceptions.Timeout:
            return jsonify({'error': f'[Flask] Web request timed out. Please try again later.'}), 500
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'[Flask] Web request error: {e}'}), 500
        return jsonify(response.json()), response.status_code

    try:
        query_type = data['query_type']
        index = data['corpus'] if 'corpus' in data else data['index']
        for key in ['query_type', 'corpus', 'index', 'engine', 'source', 'timestamp']:
            if key in data:
                del data[key]
        if ('query' not in data and 'query_ids' not in data) or ('query' in data and 'query_ids' in data):
            return jsonify({'error': f'[Flask] Exactly one of query and query_ids must be present!'}), 400
        if 'query' in data:
            query = data['query']
            query_ids = None
            del data['query']
        else:
            query = None
            query_ids = data['query_ids']
            del data['query_ids']
    except KeyError as e:
        return jsonify({'error': f'[Flask] Missing required field: {e}'}), 400

    try:
        processor = PROCESSOR_BY_INDEX[index]
    except KeyError:
        return jsonify({'error': f'[Flask] Invalid index: {index}'}), 400
    if not hasattr(processor.engine, query_type):
        return jsonify({'error': f'[Flask] Invalid query_type: {query_type}'}), 400

    try:
        result = processor.process(query_type, query, query_ids, **data)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        return jsonify({'error': f'[Flask] Internal server error: {e}'}), 500
    return jsonify(result), 200

app.run(host='0.0.0.0', port=args.FLASK_PORT, threaded=False, processes=10)
