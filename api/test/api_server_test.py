import requests

api_url = 'http://localhost:5000/'
index = 'v4_pileval_llama'

queries = [
    {'index': index, 'query_type': 'count', 'query': 'natural language processing'},
    {'index': index, 'query_type': 'count', 'query': ''},
    {'index': index, 'query_type': 'count', 'query': 'fhsdkcdshfsdf'},
    {'index': index, 'query_type': 'count', 'query': 'natural language processing OR artificial intelligence', 'max_clause_freq': 50000, 'max_diff_tokens': 20},
    {'index': index, 'query_type': 'count', 'query': 'natural language processing AND deep learning', 'max_clause_freq': 50000, 'max_diff_tokens': 20},
    {'index': index, 'query_type': 'count', 'query': 'natural language processing AND deep learning'},
    {'index': index, 'query_type': 'count', 'query': 'natural language processing OR artificial intelligence AND deep learning', 'max_clause_freq': 50000, 'max_diff_tokens': 20},
    {'index': index, 'query_type': 'prob', 'query': 'natural language processing'},
    {'index': index, 'query_type': 'prob', 'query': 'natural language apple'},
    {'index': index, 'query_type': 'prob', 'query': 'fhsdkcdshfsdf processing'},
    {'index': index, 'query_type': 'ntd', 'query': 'natural language', 'max_support': 10},
    {'index': index, 'query_type': 'ntd', 'query': 'natural language'},
    {'index': index, 'query_type': 'ntd', 'query': '', 'max_support': 10},
    {'index': index, 'query_type': 'infgram_prob', 'query': 'fhsdkcdshfsdf natural language processing'},
    {'index': index, 'query_type': 'infgram_ntd', 'query': 'fhsdkcdshfsdf natural language', 'max_support': 10},
    {'index': index, 'query_type': 'infgram_ntd', 'query': 'fhsdkcdshfsdf natural language'},
    {'index': index, 'query_type': 'search_docs', 'query': 'natural language processing', 'maxnum': 1, 'max_disp_len': 20},
    {'index': index, 'query_type': 'search_docs', 'query': 'natural language processing', 'maxnum': 10, 'max_disp_len': 20},
    {'index': index, 'query_type': 'search_docs', 'query': 'natural language processing'},
    {'index': index, 'query_type': 'search_docs', 'query': '', 'maxnum': 1, 'max_disp_len': 20},
    {'index': index, 'query_type': 'search_docs', 'query': 'fhsdkcdshfsdf', 'maxnum': 1, 'max_disp_len': 20},
    {'index': index, 'query_type': 'search_docs', 'query': 'natural language processing OR artificial intelligence', 'maxnum': 1, 'max_disp_len': 20, 'max_clause_freq': 50000, 'max_diff_tokens': 20},
    {'index': index, 'query_type': 'search_docs', 'query': 'natural language processing AND deep learning', 'maxnum': 1, 'max_disp_len': 20, 'max_clause_freq': 50000, 'max_diff_tokens': 20},
    {'index': index, 'query_type': 'search_docs', 'query': 'natural language processing AND deep learning', 'maxnum': 1, 'max_disp_len': 200},
    {'index': index, 'query_type': 'search_docs', 'query': 'natural language processing OR artificial intelligence AND deep learning', 'maxnum': 1, 'max_disp_len': 20, 'max_clause_freq': 50000, 'max_diff_tokens': 20},
]

for query in queries:
    print(query)
    result = requests.post(api_url, json=query).json()
    print(result)
    print()
