from infini_gram.engine import InfiniGramEngine
import transformers
from termcolor import colored
import sys
import time
import json
import random

tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', add_bos_token=False, add_eos_token=False)
delim_ids = [13, 29889] # \n is 13; . is 29889

def decode(token_ids):
    # trick to preserve the potential leading space
    return tokenizer.decode([7575] + token_ids, clean_up_tokenization_spaces=False)[4:]

def format_doc(doc, span_ids):
    token_ids = doc['token_ids']
    segments = []
    while True:
        pos = -1
        for p in range(len(token_ids) - len(span_ids) + 1):
            if token_ids[p:p+len(span_ids)] == span_ids:
                pos = p
                break
        if pos == -1:
            break
        if pos > 0:
            segments.append((token_ids[:pos], False))
        segments.append((token_ids[pos:pos+len(span_ids)], True))
        token_ids = token_ids[pos+len(span_ids):]
    if len(token_ids) > 0:
        segments.append((token_ids, False))

    output = ''
    for (token_ids, is_match) in segments:
        if is_match:
            output += colored(decode(token_ids), 'green')
        else:
            output += decode(token_ids)

    return output

def main():
    engine = InfiniGramEngine(
        index_dir=['/data-v4-dolma-v1_7-s0-llama/v4_dolma-v1_7-s0_llama', '/data-v4-dolma-v1_7-s1-llama/v4_dolma-v1_7-s1_llama'],
        bow_ids_path='infini_gram/llama-2_bow_ids.txt',
        eos_token_id=2, ds_prefetch_depth=0, sa_prefetch_depth=0,
    )

    filepath = sys.argv[1]
    print('='*80)
    print(f'Input file: {filepath}')
    print('='*80)

    text = open(filepath, 'r').read()
    print('Model output:')
    print(text)
    print('='*80)

    # with open('att_input/olmo-7b_old_man_sea.json') as f:
    #     items = json.load(f)
    #     item = random.choice(items)
    #     text = item['text']

    input_ids = tokenizer.encode(text)

    start_time = time.time()
    attribution_result = engine.attribute(input_ids=input_ids, delim_ids=delim_ids, min_len=10, max_cnt=10000, enforce_bow=True)
    latency = time.time() - start_time
    print(f'Attribution latency: {latency:.3f}s')
    print('='*80)

    start_time = time.time()
    spans = attribution_result["spans"]
    for span in spans:
        docs = []
        for doc in span['docs'][:1]:
            docs.append(engine.get_doc_by_ptr(s=doc['s'], ptr=doc['ptr'], max_disp_len=500))
        span['docs'] = docs
    latency = time.time() - start_time
    print(f'Doc fetch latency: {latency:.3f}s')
    print('='*80)

    print(f'Number of interesting spans: {len(spans)}')
    print('Interesting spans:')
    for span in spans:
        disp_span = tokenizer.decode(input_ids[span['l']:span['r']]).replace('\n', '\\n')
        print(f'\tl = {span["l"]}, r = {span["r"]}, length = {span["length"]}, count = {span["count"]}, span = "{disp_span}"')
    print('='*80)

    for span in spans:
        disp_span = tokenizer.decode(input_ids[span['l']:span['r']]).replace('\n', '\\n')
        print(f'l = {span["l"]}, r = {span["r"]}, length = {span["length"]}, count = {span["count"]}, span = "{disp_span}"')
        for d, doc in enumerate(span['docs']):
            print()
            print(f'Doc #{d}:')
            print(format_doc(doc, span_ids=input_ids[span['l']:span['r']]))
        print('-'*80)

if __name__ == '__main__':
    main()
