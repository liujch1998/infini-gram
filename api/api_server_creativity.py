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
parser.add_argument('--MODE', type=str, default='creativity', choices=['creativity'])
parser.add_argument('--FLASK_PORT', type=int, default=5010)
parser.add_argument('--CONFIG_FILE', type=str, default='api_config.json')
parser.add_argument('--LOG_PATH', type=str, default=None)
# API limits
parser.add_argument('--MAX_QUERY_CHARS', type=int, default=1000)
parser.add_argument('--MAX_QUERY_TOKENS', type=int, default=500)
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
            query_ids = self.tokenize(query)

        # validate query_ids
        if type(query_ids) == list and all(type(input_id) == int for input_id in query_ids): # simple query
            if len(query_ids) > args.MAX_QUERY_TOKENS:
                return {'error': f'Please limit your input to <= {args.MAX_QUERY_TOKENS} tokens!'}
            if any(input_id < 0 or input_id >= self.tokenizer.vocab_size for input_id in query_ids):
                return {'error': f'Some item(s) in your query_ids are out-of-range!'}
            tokens = self.tokenizer.convert_ids_to_tokens(query_ids)
        else:
            return {'error': f'query_ids must be a list of integers!'}

        start_time = time.time()
        result = getattr(self, query_type)(query_ids, **kwargs)
        end_time = time.time()
        result['latency'] = (end_time - start_time) * 1000
        result['token_ids'] = query_ids
        result['tokens'] = tokens

        return result

    def creativity(self, query_ids):
        if len(query_ids) == 0:
            return {'error': f'Please provide a non-empty query!'}
        result = self.engine.creativity(input_ids=query_ids)
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

app.run(host='0.0.0.0', port=args.FLASK_PORT, threaded=False)
