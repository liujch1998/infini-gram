from infini_gram.engine import InfiniGramEngine

INDEX_DIR = '/weka/oe-training-default/jiachengl/he-infinigram-api/index'
INDEX_NAMES = [
    'v4_pileval_llama',
    'v4_olmoe-mix-0924-dclm_llama',
    'v4_olmoe-mix-0924-nodclm_llama',
    'v4_tulu-v3.1-mix-preview-4096-OLMoE_llama',
    'v4_ultrafeedback-binarized-cleaned_llama',
    'v4_olmo-2-1124-13b-anneal-adapt_llama',
    'v4_olmoe-0125-1b-7b-anneal-adapt_llama',
    'v4_olmo-2-0325-32b-anneal-adapt_llama',
]

for INDEX_NAME in INDEX_NAMES:
    print(INDEX_NAME)
    engine = InfiniGramEngine(
        index_dir=f'{INDEX_DIR}/{INDEX_NAME}',
        eos_token_id=2,
        precompute_unigram_logprobs=False,
    )
    for s in range(engine.engine.get_num_shards()):
        unigram_counts = engine.compute_unigram_counts(s=s)
        # for token_id, count in enumerate(unigram_counts):
        #     print(token_id, count)
        with open(f'{INDEX_DIR}/{INDEX_NAME}/unigram.{s}', 'w') as f:
            for count in unigram_counts:
                f.write(f'{count}\n')
