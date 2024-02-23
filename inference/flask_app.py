import argparse
from flask import Flask, jsonify, request
import json
import numpy as np
import os
import socket
import subprocess
import sys
import time
import traceback
from transformers import AutoTokenizer
from infini_gram import NGramLanguageModeling

parser = argparse.ArgumentParser()
parser.add_argument('--MODE', type=str, default='api', choices=['api', 'dev', 'demo'])
parser.add_argument('--FLASK_PORT', type=int, default=5000)
parser.add_argument('--CPP_PORT', type=int, default=6000)
parser.add_argument('--MAX_QUERY_CHARS', type=int, default=1000)
parser.add_argument('--MAX_INPUT_DOC_TOKENS', type=int, default=1000)
parser.add_argument('--MAX_OUTPUT_DOC_TOKENS', type=int, default=5000)
parser.add_argument('--MAX_OUTPUT_NUM_DOCS', type=int, default=10)
parser.add_argument('--MAX_CNT_FOR_NTD', type=int, default=1000)
parser.add_argument('--MAX_CLAUSE_FREQ_PER_SHARD', type=int, default=50000)
parser.add_argument('--MAX_DIFF_TOKENS', type=int, default=100)
parser.add_argument('--MAX_CLAUSES_IN_CNF', type=int, default=4)
parser.add_argument('--MAX_TERMS_IN_DISJ_CLAUSE', type=int, default=4)
consts = parser.parse_args()
consts.MAX_DIFF_BYTES = 2 * consts.MAX_DIFF_TOKENS

ENGINES = ['python', 'c++']
# C++ engine
SOCKET_IN_BUFFER_SIZE = 2048 # This number is also hard-coded in infini_gram.cpp
SOCKET_OUT_BUFFER_SIZE = 65536 # This number is also hard-coded in infini_gram.cpp

class CppProcessor:
    def __init__(self):
        try:
            os.popen(f'g++ -std=c++17 -O3 infini_gram.cpp -o infini_gram_{consts.MODE}').read()
            subprocess.Popen(f'./infini_gram_{consts.MODE} {consts.CPP_PORT} {consts.MAX_CNT_FOR_NTD} {consts.MAX_OUTPUT_DOC_TOKENS} {consts.MAX_CLAUSE_FREQ_PER_SHARD} {consts.MAX_DIFF_TOKENS} >> ~/cpp_engine_{consts.MODE}.log 2>&1', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.s.settimeout(1.0) # in seconds
            self.s.connect(('localhost', consts.CPP_PORT))
            print(f'C++ processor initialized')
        except Exception as e:
            print(f'C++ processor failed to initialize: {e}')
            self.s = None

    def process(self, request):
        if self.s is None:
            return {'error': 'C++ processor failed to initialize'}
        try:
            request_buf = json.dumps(request).encode('utf-8')
            request_buf += b'\0' * (SOCKET_IN_BUFFER_SIZE - len(request_buf))
            bytes_sent = 0
            while bytes_sent < SOCKET_IN_BUFFER_SIZE:
                ret = self.s.send(request_buf[bytes_sent:])
                if ret == 0:
                    break
                bytes_sent += ret
            if bytes_sent != SOCKET_IN_BUFFER_SIZE:
                print(f'C++ socket send failed')
                return {'error': 'C++ socket send failed'}
            bytes_read = 0
            response_buf = b''
            while bytes_read < SOCKET_OUT_BUFFER_SIZE:
                ret = self.s.recv(SOCKET_OUT_BUFFER_SIZE - bytes_read)
                if len(ret) == 0:
                    break
                response_buf += ret
                bytes_read += len(ret)
            if len(response_buf) != SOCKET_OUT_BUFFER_SIZE:
                print(f'C++ socket recv failed')
                return {'error': 'C++ socket recv failed'}
            # response_buf = self.s.recv(SOCKET_OUT_BUFFER_SIZE)
            response = json.loads(response_buf.decode().rstrip('\0'))
            # postprocess to recover tuples and maps
            if 'segments' in response:
                response['segments'] = [(s[0], s[1]) for s in response['segments']]
            if 'freq_by_token_id' in response:
                response['freq_by_token_id'] = {_[0]: _[1] for _ in response['freq_by_token_id']}
            if 'prob_by_token_id' in response:
                response['prob_by_token_id'] = {_[0]: _[1] for _ in response['prob_by_token_id']}
            return response
        except socket.timeout:
            print('C++ socket timeout')
            return {'error': 'C++ socket timeout'}
        except Exception as e:
            print(f'C++ comm error: {e}')
            return {'error': f'C++ comm error: {e}'}

cpp_processor = CppProcessor()

class NGramProcessor:

    def __init__(self, tokenizer_type, corpus, data_dir):
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == 'gpt2':
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', add_bos_token=False, add_eos_token=False)
        elif tokenizer_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.environ['HF_TOKEN_DOWNLOAD'], add_bos_token=False, add_eos_token=False)
        elif tokenizer_type == 'olmo':
            import hf_olmo
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False)
        else:
            raise NotImplementedError
        self.corpus = corpus
        self.lm = NGramLanguageModeling(consts=consts, data_dir=data_dir, eos_token_id=self.tokenizer.eos_token_id)

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

    def count(self, engine, query):
        if len(query) > consts.MAX_QUERY_CHARS:
            return {'error': f'Please limit your input to <= {consts.MAX_QUERY_CHARS} characters!'}
        input_ids = self.tokenize(query)
        _ = " ".join(['"' + token.replace('Ġ', ' ') + '"' for token in self.tokenizer.convert_ids_to_tokens(input_ids)])
        tokenized = f'[{_}] {str(input_ids)}'

        if engine == 'python':
            result = self.lm.find(input_ids)
        elif engine == 'c++':
            result = cpp_processor.process({'query_type': 'count', 'corpus': self.corpus, 'input_ids': input_ids})

        output = {'count': result['cnt']} if 'error' not in result else {'error': result['error']}
        output['tokenized'] = tokenized
        if 'latency' in result:
            output['latency'] = result['latency']
        return output

    def prob(self, engine, query):
        if len(query) > consts.MAX_QUERY_CHARS:
            return {'error': f'Please limit your input to <= {consts.MAX_QUERY_CHARS} characters!'}
        if query == '':
            return {'error': 'Please enter at least one token!'}
        input_ids = self.tokenize(query)
        _ = " ".join(['"' + token.replace('Ġ', ' ') + '"' for token in self.tokenizer.convert_ids_to_tokens(input_ids)])
        tokenized = f'[{_}] {str(input_ids)}'

        if engine == 'python':
            result = self.lm.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
        elif engine == 'c++':
            result = cpp_processor.process({'query_type': 'prob', 'corpus': self.corpus, 'input_ids': input_ids})

        output = {'prob': result['prob'], 'prompt_cnt': result['prompt_cnt'], 'cont_cnt': result['cont_cnt']} if 'error' not in result else {'error': result['error']}
        output['tokenized'] = tokenized
        if 'latency' in result:
            output['latency'] = result['latency']
        return output

    def ntd(self, engine, query):
        if len(query) > consts.MAX_QUERY_CHARS:
            return {'error': f'Please limit your input to <= {consts.MAX_QUERY_CHARS} characters!'}
        input_ids = self.tokenize(query)
        _ = " ".join(['"' + token.replace('Ġ', ' ') + '"' for token in self.tokenizer.convert_ids_to_tokens(input_ids)])
        tokenized = f'[{_}] {str(input_ids)}'

        if engine == 'python':
            result = self.lm.ntd(prompt_ids=input_ids)
        elif engine == 'c++':
            result = cpp_processor.process({'query_type': 'ntd', 'corpus': self.corpus, 'input_ids': input_ids})

        if 'error' in result:
            output = {'error': result['error']}
        else:
            ntd = {}
            for token_id in result['prob_by_token_id']:
                prob = result['prob_by_token_id'][token_id]
                freq = result['freq_by_token_id'][token_id]
                token = self.tokenizer.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')
                ntd[f'{token} ({freq} / {result["prompt_cnt"]})'] = prob
            output = {'ntd': ntd}
        output['tokenized'] = tokenized
        if 'latency' in result:
            output['latency'] = result['latency']
        return output

    def infgram_prob(self, engine, query):
        if len(query) > consts.MAX_QUERY_CHARS:
            return {'error': f'Please limit your input to <= {consts.MAX_QUERY_CHARS} characters!'}
        if query == '':
            return {'error': 'Please enter at least one token!'}
        input_ids = self.tokenize(query)
        _ = " ".join(['"' + token.replace('Ġ', ' ') + '"' for token in self.tokenizer.convert_ids_to_tokens(input_ids)])
        tokenized = f'[{_}] {str(input_ids)}'

        if engine == 'python':
            result = self.lm.infgram_prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
        elif engine == 'c++':
            result = cpp_processor.process({'query_type': 'infgram_prob', 'corpus': self.corpus, 'input_ids': input_ids})

        output = {'prob': result['prob'], 'prompt_cnt': result['prompt_cnt'], 'cont_cnt': result['cont_cnt']} if 'error' not in result else {'error': result['error']}
        if 'prob' in output and output['prob'] == -1.0:
            output = {'error': 'Fatal error: (n-1)-gram is not found in the corpus! This should not happen.'}
        if 'lfn' in result:
            longest_suffix_ids = input_ids[-result['lfn']:-1]
            longest_suffix = self.tokenizer.decode(longest_suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            output['longest_suffix'] = longest_suffix
        output['tokenized'] = tokenized
        if 'latency' in result:
            output['latency'] = result['latency']
        return output

    def infgram_ntd(self, engine, query):
        if len(query) > consts.MAX_QUERY_CHARS:
            return {'error': f'Please limit your input to <= {consts.MAX_QUERY_CHARS} characters!'}
        input_ids = self.tokenize(query)
        _ = " ".join(['"' + token.replace('Ġ', ' ') + '"' for token in self.tokenizer.convert_ids_to_tokens(input_ids)])
        tokenized = f'[{_}] {str(input_ids)}'

        if engine == 'python':
            result = self.lm.infgram_ntd(prompt_ids=input_ids)
        elif engine == 'c++':
            result = cpp_processor.process({'query_type': 'infgram_ntd', 'corpus': self.corpus, 'input_ids': input_ids})

        if 'error' in result:
            output = {'error': result['error']}
        elif result['prompt_cnt'] == 0:
            output = {'error': 'Fatal error: prompt_cnt is 0! This should not happen.'}
        else:
            ntd = {}
            for token_id in result['prob_by_token_id']:
                prob = result['prob_by_token_id'][token_id]
                freq = result['freq_by_token_id'][token_id]
                token = self.tokenizer.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')
                ntd[f'{token} ({freq} / {result["prompt_cnt"]})'] = prob
            output = {'ntd': ntd}
        if 'lfn' in result:
            longest_suffix_ids = input_ids[-(result['lfn']-1):]
            longest_suffix = self.tokenizer.decode(longest_suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            output['longest_suffix'] = longest_suffix
        output['tokenized'] = tokenized
        if 'latency' in result:
            output['latency'] = result['latency']
        return output

    def search_docs(self, engine, query, maxnum):
        '''
        returns:
        - outputs (list, guaranteed to have length 0 or maxnum)
        - output_tokens (str)
        - message (str)
        '''
        if maxnum < 1:
            return {'error': f'Please request at least one document!'}
        if maxnum > consts.MAX_OUTPUT_NUM_DOCS:
            return {'error': f'Please request at most {consts.MAX_OUTPUT_NUM_DOCS} documents!'}
        if len(query) > consts.MAX_QUERY_CHARS:
            return {'error': f'Please limit your input to <= {consts.MAX_QUERY_CHARS} characters!'}
        clauses = query.split(' AND ')
        ngramss = [clause.split(' OR ') for clause in clauses]
        if len(ngramss) == 0:
            return {'error': 'Please enter at least one token!'}
        if len(ngramss) > consts.MAX_CLAUSES_IN_CNF:
            return {'error': f'Please enter at most {consts.MAX_CLAUSES_IN_CNF} disjunctive clauses!'}
        for ngrams in ngramss:
            if len(ngrams) == 0:
                return {'error': 'One of the disjunctive clauses appears to be empty, please enter a valid query!'}
            if len(ngrams) > consts.MAX_TERMS_IN_DISJ_CLAUSE:
                return {'error': f'Please enter at most {consts.MAX_TERMS_IN_DISJ_CLAUSE} terms in each disjunctive clause!'}
            if any([ngram == '' for ngram in ngrams]):
                return {'error': 'One of the terms appear to be empty, please enter a valid query!'}
        cnf = [[self.tokenize(ngram) for ngram in ngrams] for ngrams in ngramss]
        tokenized = []
        for disj_ix in range(len(cnf)):
            output_tokenss = []
            disj_clause = cnf[disj_ix]
            for input_ids in disj_clause:
                _ = " ".join(['"' + token.replace('Ġ', ' ') + '"' for token in self.tokenizer.convert_ids_to_tokens(input_ids)])
                output_tokens = f'[{_}] {str(input_ids)}'
                output_tokenss.append(output_tokens)
            tokenized.append('\n'.join(output_tokenss))
        tokenized = '\n\n'.join(tokenized)

        if engine == 'python':
            result = self.lm.search_docs(cnf, maxnum)
        elif engine == 'c++':
            result = cpp_processor.process({'query_type': 'search_docs', 'corpus': self.corpus, 'cnf': cnf, 'maxnum': maxnum})

        if 'error' in result:
            output = {'error': result['error']}
        else:
            cnt = result['cnt']
            approx = result['approx']
            message = f'{"Approximately " if approx else ""}{cnt} occurrences found. Displaying the documents of occurrences #{result["idxs"]}'
            docs = []
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
                doc = [(self.tokenizer.decode(token_ids), d) for (token_ids, d) in spans]
                docs.append(doc)
            output = {'message': message, 'docs': docs}
        output['tokenized'] = tokenized
        if 'latency' in result:
            output['latency'] = result['latency']
        return output

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

    def analyze_document(self, engine, query):
        if query == '':
            return {'html': '<p>Please enter at least one token!</p>'}
        input_ids = self.tokenize(query)
        if len(input_ids) == 0:
            return {'html': '<p>Please enter at least one token!</p>'}
        if len(input_ids) > consts.MAX_INPUT_DOC_TOKENS:
            return {'html': f'<p>Please limit your document to <= {consts.MAX_INPUT_DOC_TOKENS} tokens.</p>'}
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        all_result_by_cutoff = self.lm.analyze_document(input_ids)
        if all_result_by_cutoff is None:
            return {'html': f'<p>Compute time exceeded.</p>'}
        html = ''
        line_len = 0
        for i, (token, result_by_cutoff) in enumerate(zip(tokens, all_result_by_cutoff)):
            if line_len >= 100 and token.startswith('Ġ') and token != 'Ċ':
                html += '<br/>'
                line_len = 0
            earliest_cutoff = sorted(result_by_cutoff.keys())[0]
            prob = result_by_cutoff[earliest_cutoff]['prob']
            lfn = len(result_by_cutoff)
            r = 1.0 - prob
            g = 0.0
            b = prob
            a = 0.1 + 0.8 * (min(5, np.log2(lfn)) / 5.0)
            color = f'({int(255*r)}, {int(255*g)}, {int(255*b)}, {a:.2f})'

            data_hl = {}
            for cutoff, result in result_by_cutoff.items():
                prob = result['prob']
                cnt = result['prompt_cnt']
                rr = 1.0 - prob
                gg = 0.0
                bb = prob
                aa = 0.1 + 0.8 * (min(15, np.log2(cnt)) / 15.0)
                ccolor = f'({int(255*rr)}, {int(255*gg)}, {int(255*bb)}, {aa:.2f})'
                data_hl[cutoff] = ccolor
            data_hl = json.dumps(data_hl)
            title_hl = ""
            for cutoff, result in result_by_cutoff.items():
                token_disp = token.replace("Ġ", " ").replace("Ċ", "\\\n").replace('"', '&quot;')
                prompt_disp = "".join([token.replace("Ġ", " ").replace("Ċ", "\\\n").replace('"', '&quot;') for token in tokens[cutoff:i]]) if cutoff < i else "''"
                title_hl += f'P({token_disp} | {prompt_disp}) = {result["cont_cnt"]} / {result["prompt_cnt"]} = {result["prob"]:.4f}\n'
            title_hl = title_hl.strip('\n')

            if token == 'Ċ':
                disp_token = '\\n'
                is_linebreak = True
            else:
                disp_token = token.replace('Ġ', '&nbsp;')
                is_linebreak = False
            onmouseover = "var spans = document.getElementById('hldoc').children; for (var i = 0; i < spans.length; i++) { spans[i].style.backgroundColor = ''; } var data_hl = JSON.parse(this.getAttribute('data-hl')); for (var cutoff in data_hl) { var span = document.getElementById('hldoc-token-' + cutoff); span.style.cssText = 'background-color: rgba' + data_hl[cutoff]; + ';'; }"
            onmouseout = "var spans = document.getElementById('hldoc').children; for (var i = 0; i < spans.length; i++) { spans[i].style.cssText = spans[i].getAttribute('class'); }"
            html += f'<span id="hldoc-token-{i}" style="background-color: rgba{color};" class="background-color: rgba{color};" title="{title_hl}"' + f" data-hl='{data_hl}'" + f' onmouseover="{onmouseover}" onmouseout="{onmouseout}">{disp_token}</span>'
            if is_linebreak:
                html += '<br/>'
                line_len = 0
            else:
                line_len += len(token)
        html = '<div><p id="hldoc" style="font-size: 16px;">' + html.strip(' ') + '</p></div>'
        return {'html': html}

PROCESSOR_BY_CORPUS = {}
with open('indexes.json') as f:
    configs = json.load(f)
    for config in configs:
        assert all([k in config for k in ['corpus', 'dir', 'tokenizer', 'eos_token_id']])
        PROCESSOR_BY_CORPUS[config['corpus']] = NGramProcessor(
            tokenizer_type=config['tokenizer'],
            corpus=config['corpus'],
            data_dir=config['dir'],
        )

log = open(f'../flask_{consts.MODE}.log', 'a')
app = Flask(__name__)

@app.route('/', methods=['POST'])
@app.route('/api', methods=['POST'])
def query():
    data = request.json
    print(data)
    log.write(json.dumps(data) + '\n')
    log.flush()

    try:
        query_type = data['query_type']
        corpus = data['corpus']
        engine = 'c++' if 'engine' not in data else data['engine']
        query = data['query']
        if query_type == 'search_docs':
            maxnum = data['maxnum']
    except KeyError as e:
        return jsonify({'error': f'[Flask] Missing required field: {e}'}), 400
    try:
        processor = PROCESSOR_BY_CORPUS[corpus]
    except KeyError:
        return jsonify({'error': f'[Flask] Invalid corpus: {corpus}'}), 400
    if engine not in ENGINES:
        return jsonify({'error': f'[Flask] Invalid engine: {engine}'}), 400
    if not hasattr(processor, query_type):
        return jsonify({'error': f'[Flask] Invalid query_type: {query_type}'}), 400

    try:
        if query_type == 'search_docs':
            result = getattr(processor, query_type)(engine, query, maxnum)
        else:
            result = getattr(processor, query_type)(engine, query)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        return jsonify({'error': f'[Flask] Internal server error: {e}'}), 500
    return jsonify(result), 200

app.run(host='0.0.0.0', port=consts.FLASK_PORT, threaded=False)

