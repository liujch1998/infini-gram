import requests

api_url = 'http://localhost:5000/'
corpus = 'v4_pileval_llama'

queries = [
    {'corpus': corpus, 'query_type': 'count', 'query': ''},
    {'corpus': corpus, 'query_type': 'count', 'query': 'natural language processing'},
    {'corpus': corpus, 'query_type': 'count', 'query': 'fhsdkcdshfsdf'},
    {'corpus': corpus, 'query_type': 'count', 'query': 'natural language processing OR artificial intelligence'},
    {'corpus': corpus, 'query_type': 'count', 'query': 'natural language processing AND deep learning'},
    {'corpus': corpus, 'query_type': 'count', 'query': 'natural language processing OR artificial intelligence AND deep learning'},
    {'corpus': corpus, 'query_type': 'prob', 'query': 'natural language processing'},
    {'corpus': corpus, 'query_type': 'prob', 'query': 'natural language apple'},
    {'corpus': corpus, 'query_type': 'prob', 'query': 'fhsdkcdshfsdf processing'},
    {'corpus': corpus, 'query_type': 'ntd', 'query': 'natural language'},
    {'corpus': corpus, 'query_type': 'ntd', 'query': ''},
    {'corpus': corpus, 'query_type': 'infgram_prob', 'query': 'fhsdkcdshfsdf natural language processing'},
    {'corpus': corpus, 'query_type': 'infgram_ntd', 'query': 'fhsdkcdshfsdf natural language'},
    {'corpus': corpus, 'query_type': 'search_docs', 'maxnum': 1, 'query': ''},
    {'corpus': corpus, 'query_type': 'search_docs', 'maxnum': 1, 'query': 'natural language processing'},
    {'corpus': corpus, 'query_type': 'search_docs', 'maxnum': 1, 'query': 'fhsdkcdshfsdf'},
    {'corpus': corpus, 'query_type': 'search_docs', 'maxnum': 1, 'query': 'natural language processing OR artificial intelligence'},
    {'corpus': corpus, 'query_type': 'search_docs', 'maxnum': 1, 'query': 'natural language processing AND deep learning'},
    {'corpus': corpus, 'query_type': 'search_docs', 'maxnum': 1, 'query': 'natural language processing OR artificial intelligence AND deep learning'},
]

for query in queries:
    print(query)
    result = requests.post(api_url, json=query).json()
    print(result)
    print()
