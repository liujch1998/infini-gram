# pip install transformers tqdm termcolor pybind11
# huggingface-cli login

from termcolor import colored
import csv
import sys
import time
import numpy as np
import transformers
sys.path = ['../../pkg'] + sys.path
from infini_gram.engine import InfiniGramEngine

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
        index_dir=[
            '/weka/oe-training-default/jiachengl/he-infinigram-api/index/v4_olmoe-mix-0924-dclm_llama',
            '/weka/oe-training-default/jiachengl/he-infinigram-api/index/v4_olmoe-mix-0924-nodclm_llama',
            '/weka/oe-training-default/jiachengl/he-infinigram-api/index/v4_olmo-2-1124-13b-anneal-adapt_llama',
        ],
        eos_token_id=2, bow_ids_path='./llama-2_bow_ids.txt', precompute_unigram_logprobs=True,
        ds_prefetch_depth=0, sa_prefetch_depth=0, od_prefetch_depth=0,
    )

    # name = sys.argv[1]
    # text = open(f'input/{name}.txt', 'r').read()
    # print('Model output:')
    # print(text)
    # print('='*80)

    text = '''Celine Dion is a Canadian singer known for her powerful voice and wide vocal range. She has achieved international fame and is one of the best-selling artists of all time. Born on March 30, 1968, in Charlemagne, Quebec, Canada, Dion began performing at a young age and gained prominence in the music industry in the 1980s.

Her career took off when she won the 1988 Eurovision Song Contest with the song "Ne partez pas sans moi." This victory launched her into international stardom.'''
    # text = '''Taylor Swift is an American singer-songwriter and musician known for her narrative songwriting which often draws on her personal life. She was born on December 13, 1989, in Reading, Pennsylvania.'''

    input_ids = tokenizer.encode(text)

    start_time = time.time()
    attribution_result = engine.attribute(input_ids=input_ids, delim_ids=delim_ids, min_len=1, max_cnt=1000000, enforce_bow=True)
    spans = attribution_result["spans"]
    latency = time.time() - start_time
    print(f'Attribution latency: {latency:.3f}s')
    print('='*80)

    # start_time = time.time()
    # for span in spans:
    #     span['docs'] = engine.get_docs_by_ptrs(list_of_s_and_ptr=[(doc['s'], doc['ptr']) for doc in span['docs']], max_disp_len=200)
    # latency = time.time() - start_time
    # print(f'Doc fetch latency: {latency:.3f}s')
    # print('='*80)

    print(f'Number of interesting spans: {len(spans)}')
    print('Interesting spans:')
    for span in spans:
        disp_span = tokenizer.decode(input_ids[span['l']:span['r']]).replace('\n', '\\n')
        print(f'\tl = {span["l"]}, r = {span["r"]}, length = {span["length"]}, count = {span["count"]}, span = "{disp_span}"')
    print('='*80)

    text = ' Dion is a Canadian singer'
    input_ids = tokenizer.encode(text)[1:]
    find_result = engine.find(input_ids=input_ids)
    print(find_result)
    for s, segment in enumerate(find_result['segment_by_shard']):
        b, e = segment
        print(s)
        prev_doc = engine.get_doc_by_rank(s=s, rank=b-1, max_disp_len=20)
        print(tokenizer.decode(prev_doc['token_ids']))
        first_doc = engine.get_doc_by_rank(s=s, rank=b, max_disp_len=20)
        print(tokenizer.decode(first_doc['token_ids']))
        last_doc = engine.get_doc_by_rank(s=s, rank=e-1, max_disp_len=20)
        print(tokenizer.decode(last_doc['token_ids']))
        next_doc = engine.get_doc_by_rank(s=s, rank=e, max_disp_len=20)
        print(tokenizer.decode(next_doc['token_ids']))

    # for span in spans:
    #     disp_span = tokenizer.decode(input_ids[span['l']:span['r']]).replace('\n', '\\n')
    #     print(f'l = {span["l"]}, r = {span["r"]}, length = {span["length"]}, count = {span["count"]}, span = "{disp_span}"')
    #     for d, doc in enumerate(span['docs']):
    #         print()
    #         print(f'Doc #{d}:')
    #         print(format_doc(doc, span_ids=input_ids[span['l']:span['r']]))
    #     print('-'*80)

    # with open(f'span/{name}.csv', 'w') as f:
    #     writer = csv.DictWriter(f, fieldnames=['l', 'r', 'span', 'length', 'count', 'unigram_logprob_sum'])
    #     writer.writeheader()
    #     for span in spans:
    #         disp_span = tokenizer.decode(input_ids[span['l']:span['r']]).replace('\n', '\\n')
    #         writer.writerow({
    #             'l': span['l'],
    #             'r': span['r'],
    #             'span': disp_span,
    #             'length': span['length'],
    #             'count': span['count'],
    #             'unigram_logprob_sum': span['unigram_logprob_sum'],
    #         })

if __name__ == '__main__':
    main()
