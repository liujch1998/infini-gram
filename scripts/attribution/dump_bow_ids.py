import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', add_bos_token=False, add_eos_token=False)

token_to_id = tokenizer.get_vocab()

bow_token_ids = []
for token, token_id in token_to_id.items():
    if token_id <= 258 or token_id >= 29871 or token[0] == '▁':
        bow_token_ids.append(token_id)

bow_token_ids = sorted(bow_token_ids)

for token_id in bow_token_ids:
    print(token_id)
