# pip install transformers tqdm termcolor pybind11
# huggingface-cli login

from termcolor import colored
import csv
import sys
import time
import numpy as np
import transformers
sys.path = ['../pkg'] + sys.path
from infini_gram.engine import InfiniGramEngine, InfiniGramEngineDiff

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
    text = 'Puigdemont calls for talks with Spain Published duration 22 December 2017 Related Topics Catalonia independence protests'
    # text = '''Jacob Bernoulli (1654–1705) was a Swiss mathematician and a member of the Bernoulli family'''
    input_ids = tokenizer.encode(text)

    print('Before takedown:')

    engine = InfiniGramEngine(
        index_dir=['../index/v4_pileval_llama'],
        eos_token_id=2, bow_ids_path='./llama-2_bow_ids.txt', precompute_unigram_logprobs=True,
        ds_prefetch_depth=0, sa_prefetch_depth=0, od_prefetch_depth=0,
    )

    attribution_result = engine.attribute(input_ids=input_ids, delim_ids=delim_ids, min_len=1, max_cnt=1000000, enforce_bow=True)
    spans = attribution_result["spans"]
    for span in spans:
        span['docs'] = engine.get_docs_by_ptrs_2_grouped(requests=[(doc['s'], doc['ptr'], span['length'], 20) for doc in span['docs'][:3]])

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
            print(f'Doc #{d}: doc_ix = {doc["doc_ix"]}, doc_len = {doc["doc_len"]}, disp_len = {doc["disp_len"]}, needle_offset = {doc["needle_offset"]}, blocked = {doc["blocked"]}')
            print(format_doc(doc, span_ids=input_ids[span['l']:span['r']]))
        print('-'*80)
    print('='*80)


    print('After takedown:')

    engine = InfiniGramEngineDiff(
        index_dir=['../index/v4_pileval_llama'],
        # index_dir_diff=['../index/v4_pileval-0doc_llama'],
        index_dir_diff=['../index/v4_pileval-1doc_llama'],
        # index_dir_diff=['../index/v4_pileval-p50doc_llama'],
        eos_token_id=2, bow_ids_path='./llama-2_bow_ids.txt', precompute_unigram_logprobs=True,
        ds_prefetch_depth=0, sa_prefetch_depth=0, od_prefetch_depth=0,
    )

    attribution_result = engine.attribute(input_ids=input_ids, delim_ids=delim_ids, min_len=1, max_cnt=1000000, enforce_bow=True)
    spans = attribution_result["spans"]
    docss = engine.get_docs_by_ptrs_2_grouped(requests=[{'docs': span['docs'][:3], 'span_ids': input_ids[span['l']:span['r']], 'needle_len': span['length'], 'max_ctx_len': 20} for span in spans])
    for span, docs in zip(spans, docss):
        span['docs'] = docs

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
            print(f'Doc #{d}: doc_ix = {doc["doc_ix"]}, doc_len = {doc["doc_len"]}, disp_len = {doc["disp_len"]}, needle_offset = {doc["needle_offset"]}, blocked = {doc["blocked"]}')
            print(format_doc(doc, span_ids=input_ids[span['l']:span['r']]))
        print('-'*80)
    print('='*80)


if __name__ == '__main__':
    main()
