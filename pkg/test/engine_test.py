# python test/engine_test.py

import numpy as np
import random
import time
from infini_gram.engine import InfiniGramEngine

def main():
    engine = InfiniGramEngine(index_dir='../index/v4_pileval_llama', eos_token_id=2)
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

if __name__ == '__main__':
    main()
