# python test/engine_test.py

import numpy as np
import random
import time
from infini_gram.engine import InfiniGramEngine

def main():
    engine = InfiniGramEngine(index_dir='../index/v4_pileval_llama', eos_token_id=2)
    # engine = InfiniGramEngine(index_dir='../index/v4_pileval_llama_u32', eos_token_id=2, token_dtype='u32')
    # engine = InfiniGramEngine(index_dir='../index/v5_pileval_llama', eos_token_id=2, version=5)
    # engine = InfiniGramEngine(index_dir='../index/v4_dolma-v1_6-sample_llama', eos_token_id=2)
    # engine = InfiniGramEngine(index_dir=['../index/v4_pileval_llama', '../index/v4_dolma-v1_6-sample_llama'], eos_token_id=2)

    # times = []
    # for it in range(100):
    #     query_ids = [random.randint(0, 65535) for _ in range(5)]
    #     start_time = time.time()
    #     result = engine.count(query_ids=query_ids)
    #     end_time = time.time()
    #     times.append(end_time - start_time)
    # print('Average time:', np.mean(times))
    # exit()

    input_ids = [5613, 4086, 9068]

    print(engine.count(input_ids=input_ids))
    print()
    print(engine.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1]))
    print()
    print(engine.ntd(prompt_ids=input_ids[:-1]))
    print()
    print(engine.infgram_prob(prompt_ids=[3234, 4324] + input_ids[:-1], cont_id=input_ids[-1]))
    print()
    print(engine.infgram_ntd(prompt_ids=[3234, 4324] + input_ids[:-1]))
    print()
    print(engine.search_docs(input_ids=input_ids))
    print()
    find_result = engine.find(input_ids=input_ids)
    rank = find_result['segment_by_shard'][0][0]
    doc = engine.get_doc_by_rank_2(s=0, rank=rank, needle_len=len(input_ids), max_ctx_len=10)
    print(doc)
    print()


if __name__ == '__main__':
    main()
