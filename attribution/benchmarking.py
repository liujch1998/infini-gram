# pip install transformers tqdm pybind11
# huggingface-cli login

import json
import numpy as np
import random
import time
from tqdm import tqdm
import transformers
from infini_gram.engine import InfiniGramEngine

tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', add_bos_token=False, add_eos_token=False)
delim_ids = [13, 29889] # \n is 13; . is 29889

engine = InfiniGramEngine(
    index_dir=[
        '/weka/oe-training-default/jiachengl/index/v4_olmoe-mix-0924-dclm_llama',
        '/weka/oe-training-default/jiachengl/index/v4_olmoe-mix-0924-nodclm_llama',
        # '/weka/oe-training-default/jiachengl/index/v4_tulu-v3.1-mix-preview-4096-OLMoE_llama',
        # '/weka/oe-training-default/jiachengl/index/v4_ultrafeedback-binarized-cleaned_llama',
    ],
    # index_dir=[
    #     './index/v4_pileval_llama',
    # ],
    # index_dir=[
    #     './index/v4_pileval_llama_replica0',
    #     './index/v4_pileval_llama_replica1',
    #     './index/v4_pileval_llama_replica2',
    #     './index/v4_pileval_llama_replica3',
    #     './index/v4_pileval_llama_replica4',
    #     './index/v4_pileval_llama_replica5',
    #     './index/v4_pileval_llama_replica6',
    #     './index/v4_pileval_llama_replica7',
    #     './index/v4_pileval_llama_replica8',
    #     './index/v4_pileval_llama_replica9',
    #     './index/v4_pileval_llama_replica10',
    #     './index/v4_pileval_llama_replica11',
    #     './index/v4_pileval_llama_replica12',
    # ],
    eos_token_id=2, bow_ids_path='./llama-2_bow_ids.txt',
    ds_prefetch_depth=0, sa_prefetch_depth=0, od_prefetch_depth=0,
)

with open('/weka/oe-training-default/jiachengl/raw/tulu-v3.1-mix-preview-4096-OLMoE/0.jsonl') as f:
    lines = f.readlines()
    ds = [json.loads(l) for l in lines]
sampled_ds = random.sample(ds, 10)

lengths = []
num_spans = []
num_docs = []
attribute_latencies = []
doc_latencies = []

for item in tqdm(sampled_ds):
    text = item['text'].split('\n<|assistant|>\n')[-1]
    # print(text)
    input_ids = tokenizer.encode(text)
    lengths.append(len(input_ids))
    print(input_ids)

    start_time = time.time()
    attribution_result = engine.attribute(input_ids=input_ids, delim_ids=delim_ids, min_len=1, max_cnt=10, enforce_bow=True)
    num_spans.append(len(attribution_result['spans']))
    num_docs.append(sum([len(span['docs']) for span in attribution_result['spans']]))
    latency = time.time() - start_time
    attribute_latencies.append(latency)

    start_time = time.time()
    list_of_s_and_ptr = [(doc['s'], doc['ptr']) for span in attribution_result['spans'] for doc in span['docs']]
    docs = engine.get_docs_by_ptrs(list_of_s_and_ptr=list_of_s_and_ptr, max_disp_len=100)
    latency = time.time() - start_time
    doc_latencies.append(latency)

print(f'Avg length: {np.mean(lengths):.2f}')
print(f'Avg num spans: {np.mean(num_spans):.2f}')
print(f'Avg num docs: {np.mean(num_docs):.2f}')
print(f'Avg attribution latency: {int(np.mean(attribute_latencies) * 1000)}ms')
print(f'Avg doc fetch latency: {int(np.mean(doc_latencies) * 1000)}ms')
