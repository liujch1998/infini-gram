import argparse
from flask import Flask, jsonify, request
import json
import numpy as np
import os
import requests
import sys
import time
import traceback
from transformers import AutoTokenizer
from engine import InfiniGramEngine

parser = argparse.ArgumentParser()
parser.add_argument('--MODE', type=str, default='api', choices=['api', 'dev', 'demo'])
parser.add_argument('--FLASK_PORT', type=int, default=5000)
# API limits
parser.add_argument('--MAX_QUERY_CHARS', type=int, default=1000)
parser.add_argument('--MAX_QUERY_TOKENS', type=int, default=500)
parser.add_argument('--MAX_CLAUSES_PER_CNF', type=int, default=4)
parser.add_argument('--MAX_TERMS_PER_CLAUSE', type=int, default=4)
parser.add_argument('--MAX_OUTPUT_NUM_DOCS', type=int, default=10)
# engine limits
parser.add_argument('--MAX_OUTPUT_DOC_TOKENS', type=int, default=5000)
parser.add_argument('--MAX_CNT_FOR_NTD', type=int, default=1000)
parser.add_argument('--MAX_CLAUSE_FREQ_PER_SHARD', type=int, default=50000)
parser.add_argument('--MAX_DIFF_TOKENS', type=int, default=100)
args = parser.parse_args()

DOLMA_API_URL = os.environ.get(f'DOLMA_API_URL_{args.MODE.upper()}', None)

class Processor:

    def __init__(self, config):
        assert 'dir' in config and 'tokenizer' in config

        self.tokenizer_type = config['tokenizer']
        if self.tokenizer_type == 'gpt2':
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', add_bos_token=False, add_eos_token=False)
        elif self.tokenizer_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.environ['HF_TOKEN_DOWNLOAD'], add_bos_token=False, add_eos_token=False)
        elif self.tokenizer_type == 'olmo':
            import hf_olmo
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False)
        else:
            raise NotImplementedError

        self.engine = InfiniGramEngine(
            index_dir=config['dir'],
            eos_token_id=self.tokenizer.eos_token_id,
            MAX_CNT_FOR_NTD=args.MAX_CNT_FOR_NTD,
            MAX_OUTPUT_DOC_TOKENS=args.MAX_OUTPUT_DOC_TOKENS,
            MAX_CLAUSE_FREQ_PER_SHARD=args.MAX_CLAUSE_FREQ_PER_SHARD,
            MAX_DIFF_TOKENS=args.MAX_DIFF_TOKENS,
            ds_prefetch_depth=config.get('ds_prefetch_depth', 1),
            sa_prefetch_depth=config.get('sa_prefetch_depth', 3),
            od_prefetch_depth=config.get('od_prefetch_depth', 3),
        )

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
            if any(input_id < 0 or input_id >= self.tokenizer.vocab_size for input_id in query_ids):
                return {'error': f'Some item(s) in your query_ids are out-of-range!'}
            tokens = self.tokenizer.convert_ids_to_tokens(query_ids)
        elif type(query_ids) == list and all([type(clause) == list and all([type(term) == list and all([type(input_id) == int for input_id in term]) for term in clause]) for clause in query_ids]):
            if sum(sum(len(term) for term in clause) for clause in query_ids) > args.MAX_QUERY_TOKENS:
                return {'error': f'Please limit your input to <= {args.MAX_QUERY_TOKENS} tokens!'}
            if len(query_ids) > args.MAX_CLAUSES_PER_CNF:
                return {'error': f'Please enter at most {args.MAX_CLAUSES_PER_CNF} disjunctive clauses!'}
            for clause in query_ids:
                if len(clause) > args.MAX_TERMS_PER_CLAUSE:
                    return {'error': f'Please enter at most {args.MAX_TERMS_PER_CLAUSE} terms in each disjunctive clause!'}
                for term in clause:
                    if any(input_id < 0 or input_id >= self.tokenizer.vocab_size for input_id in term):
                        return {'error': f'Some item(s) in your query_ids are out-of-range!'}
            tokens = [[self.tokenizer.convert_ids_to_tokens(term) for term in clause] for clause in query_ids]
        else:
            return {'error': f'query_ids must be a list of integers, or a triply-nested list of integers!'}

        start_time = time.time()
        result = getattr(self, query_type)(query_ids, **kwargs)
        end_time = time.time()
        result['latency'] = end_time - start_time
        result['token_ids'] = query_ids
        result['tokens'] = tokens

        return result

    def count(self, query_ids):
        return self.engine.count(query_ids)

    def prob(self, query_ids):
        return self.engine.prob(query_ids)

    def ntd(self, query_ids):
        result = self.engine.ntd(query_ids)
        if 'result_by_token_id' in result:
            for token_id in result['result_by_token_id']:
                result['result_by_token_id'][token_id]['token'] = self.tokenizer.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')
        return result

    def infgram_prob(self, query_ids):
        result = self.engine.infgram_prob(query_ids)
        if 'suffix_len' in result:
            result['longest_suffix'] = self.tokenizer.decode(query_ids[-result['suffix_len']-1:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return result

    def infgram_ntd(self, query_ids):
        result = self.engine.infgram_ntd(query_ids)
        if 'result_by_token_id' in result:
            for token_id in result['result_by_token_id']:
                result['result_by_token_id'][token_id]['token'] = self.tokenizer.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')
        if 'suffix_len' in result:
            result['longest_suffix'] = self.tokenizer.decode(query_ids[-result['suffix_len']:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return result

    def search_docs(self, query_ids, maxnum):
        if maxnum > args.MAX_OUTPUT_NUM_DOCS:
            return {'error': f'Please request at most {args.MAX_OUTPUT_NUM_DOCS} documents!'}

        result = self.engine.search_docs(query_ids, maxnum)

        if 'documents' in result:
            if type(query_ids) == list and all(type(input_id) == int for input_id in query_ids): # simple query
                cnf = [[query_ids]]
            else:
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

PROCESSOR_BY_INDEX = {}
with open('api_config_test.json') as f:
    configs = json.load(f)
    for config in configs:
        PROCESSOR_BY_INDEX[config['name']] = Processor(config)

log = open(f'flask_{args.MODE}.log', 'a')
app = Flask(__name__)

@app.route('/', methods=['POST'])
def query():
    data = request.json
    print(data)
    log.write(json.dumps(data) + '\n')
    log.flush()

    index = data['corpus'] if 'corpus' in data else (data['index'] if 'index' in data else None)
    if 'dolma-v1_6' in index and DOLMA_API_URL is not None:
        try:
            response = requests.post(DOLMA_API_URL, json=data, timeout=10)
        except requests.exceptions.Timeout:
            return jsonify({'error': f'[Flask] Web request timed out. Please try again later.'}), 500
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'[Flask] Web request error: {e}'}), 500
        return jsonify(response.json()), response.status_code

    try:
        query_type = data['query_type']
        del data['query_type']
        if 'corpus' in data:
            index = data['corpus']
            del data['corpus']
        else:
            index = data['index']
            del data['index']
        if 'engine' in data:
            del data['engine']
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

app.run(host='0.0.0.0', port=args.FLASK_PORT, threaded=False)
