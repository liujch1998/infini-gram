import socket
import json

OUT_BUFFER_SIZE = 65536

CPP_PORT = 3786
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', CPP_PORT))

request = {
    'query_type': 'count',
    'corpus': 'redpajama_bff2_v3_c8_llama2',
    'input_ids': [5613, 4086, 9068],
}
s.sendall(json.dumps(request).encode('utf-8'))
response = s.recv(OUT_BUFFER_SIZE).decode()
print(response)

request = {
    'query_type': 'compute_prob',
    'corpus': 'redpajama_bff2_v3_c8_llama2',
    'input_ids': [5613, 4086, 9068],
}
s.sendall(json.dumps(request).encode('utf-8'))
response = s.recv(OUT_BUFFER_SIZE).decode()
print(response)

request = {
    'query_type': 'get_next_token_distribution_fast_approx',
    'corpus': 'redpajama_bff2_v3_c8_llama2',
    'input_ids': [5613, 4086],
}
s.sendall(json.dumps(request).encode('utf-8'))
response = s.recv(OUT_BUFFER_SIZE).decode()
print(response)

s.close()
