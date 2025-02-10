import requests
import multiprocessing as mp
import random

NUM_TOKENS = 2
NUM_CONCURRENT_REQUESTS = 200

PAYLOAD = {
    'query_ids': None,
    'query_type': 'count',
    'index': 'v4_dolma-v1_7_llama',
}

url = 'http://localhost:5001'

def issue_request(query_ids):
    payload = PAYLOAD.copy()
    payload['query_ids'] = query_ids
    payload['query_type'] = random.choice(['count', 'prob', 'ntd', 'infgram_prob', 'infgram_ntd'])
    return requests.post(url, json=payload).json()

with mp.get_context('fork').Pool(NUM_CONCURRENT_REQUESTS) as p:
    query_idss = []
    for i in range(NUM_CONCURRENT_REQUESTS):
        query_ids = []
        for j in range(NUM_TOKENS):
            query_ids.append(random.randint(0, 30000))
        query_idss.append(query_ids)
    results = p.map(issue_request, query_idss)
    for result in results:
        print(result)

