import numpy as np
import random
import time
from engine import InfiniGramEngine

def main():
    engine = InfiniGramEngine(
        index_dir='/gscratch/xlab/liujc/ha-infini-gram/index/v4_pileval_llama',
        eos_token_id=2,
    )
    # engine = InfiniGramEngine(
    #     index_dir=['/gscratch/xlab/liujc/ha-infini-gram/index/v4_pileval_llama', '/gscratch/xlab/liujc/ha-infini-gram/index/v4_pileval_llama'],
    #     eos_token_id=2,
    # )

    # times = []
    # for it in range(100):
    #     query_ids = [random.randint(0, 65535) for _ in range(5)]
    #     start_time = time.time()
    #     result = engine.count(query_ids=query_ids)
    #     end_time = time.time()
    #     times.append(end_time - start_time)
    # print('Average time:', np.mean(times))
    # exit()

    query_ids = [5613, 4086, 9068]

    print(engine.count(query_ids=query_ids))
    print()
    print(engine.prob(query_ids=query_ids))
    print()
    print(engine.ntd(query_ids=query_ids[:-1]))
    print()
    print(engine.infgram_prob(query_ids=[3234, 4324] + query_ids))
    print()
    print(engine.infgram_ntd(query_ids=[3234, 4324] + query_ids[:-1]))
    print()
    print(engine.search_docs(query_ids=query_ids, maxnum=10))
    print()

if __name__ == '__main__':
    main()
