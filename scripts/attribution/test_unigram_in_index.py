from infini_gram.engine import InfiniGramEngine

INDEX_DIR = '/weka/oe-training-default/jiachengl/index'
# INDEX_DIR = '../index'
INDEX_NAMES = [
    'v4_pileval_llama',
    # 'v4_olmoe-mix-0924-dclm_llama',
    # 'v4_olmoe-mix-0924-nodclm_llama',
    # 'v4_tulu-v3.1-mix-preview-4096-OLMoE_llama',
    # 'v4_ultrafeedback-binarized-cleaned_llama',
    # 'v4_olmo-2-1124-13b-anneal-adapt_llama',
]

for INDEX_NAME in INDEX_NAMES:
    print(INDEX_NAME)
    engine = InfiniGramEngine(
        index_dir=f'{INDEX_DIR}/{INDEX_NAME}',
        eos_token_id=2,
        precompute_unigram_logprobs=True,
    )
