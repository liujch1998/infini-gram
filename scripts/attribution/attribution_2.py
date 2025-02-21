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

def format_doc(doc):
    output = ''
    token_ids = doc['token_ids']
    disp_offset = doc['disp_offset']
    i = 0
    for (token_offset, span) in doc['token_offset_span_pairs']:
        if i > (token_offset - disp_offset):
            continue
        output += decode(token_ids[i:(token_offset - disp_offset)])
        output += colored(decode(token_ids[(token_offset - disp_offset):(token_offset - disp_offset + span[1] - span[0])]), 'green')
        i = token_offset - disp_offset + span[1] - span[0]
    output += decode(token_ids[i:])
    return output

def main():
    engine = InfiniGramEngine(index_dir=['/data-v4-dolma-v1_7-s0-llama/v4_dolma-v1_7-s0_llama', '/data-v4-dolma-v1_7-s1-llama/v4_dolma-v1_7-s1_llama'], eos_token_id=2, ds_prefetch_depth=0, sa_prefetch_depth=0)

    filepath = sys.argv[1]
    print('='*80)
    print(f'Input file: {filepath}')
    print('='*80)

    text = open(filepath, 'r').read()
    print('Model output:')
    print(text)
    print('='*80)

    input_ids = tokenizer.encode(text)
    # spans = engine.compute_interesting_spans(input_ids=input_ids, stop_ids=stop_ids)

    start_time = time.time()
    attribution_result = engine.attribute_2(input_ids=input_ids, delim_ids=delim_ids, min_len=7, max_cnt=10000, max_docs=10, max_disp_len=500)
    latency = time.time() - start_time
    print(f'Attribution latency: {latency:.3f}s')
    print('='*80)

    spans = attribution_result["spans"]
    print(f'Number of interesting spans: {len(spans)}')
    print('Interesting spans:')
    for (start, end) in spans:
        length = end - start
        count = engine.count(input_ids=input_ids[start:end])['count']
        disp_span = tokenizer.decode(input_ids[start:end]).replace('\n', '\\n')
        print(f'\tl = {start}, r = {end}, length = {length}, count = {count}, span = "{disp_span}"')
    print('='*80)

    docs = attribution_result["docs"]
    print(f'Number of docs retrieved: {len(docs)}')
    print('Metadata of each document:')
    for doc in docs:
        printed_doc = {k: v for k, v in doc.items() if k not in ['token_ids', 'metadata']}
        print(f'\t{printed_doc}')
    print('='*80)

    for i in range(min(5, len(docs))):
        print(f'Doc #{i}:')
        # print(docs[i]['metadata'])
        print(format_doc(docs[i]))
        print('-'*80)

if __name__ == '__main__':
    main()
