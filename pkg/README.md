# infini-gram

This package allows you to run the infini-gram engine on any existing infini-gram index.

(We will soon support building your own indexes on datasets of your choice. Stay tuned!)

To learn more about infini-gram:
* Paper: <https://arxiv.org/abs/2401.17377>
* Project Home: <https://infini-gram.io>
* Web Interface: <https://infini-gram.io/demo>
* API Endpoint: <https://infini-gram.io/api_doc>

## Overview

Running the infini-gram engine is easy!
Here's a minimal example:

```python
from infini_gram.engine import InfiniGramEngine

engine = InfiniGramEngine(index_dir='index/v4_pileval_llama', eos_token_id=2)

engine.count(query_ids=[5613, 4086, 9068]) # natural language processing
# returns {'count': 76, 'approx': False}
```

## Usage

1. Install this package.
```bash
pip install infini-gram
```

2. Download the infini-gram index that you would like to query. See details in the "Pre-built Indexes" section below.

3. Start a Python program and create the engine.
```python
from infini_gram.engine import InfiniGramEngine

engine = InfiniGramEngine(index_dir='{LOCAL_INDEX_PATH}', eos_token_id=2) # eos_token_id should match the tokenizer used for this index!
```

4. Query the engine. Here are some example queries:
```python
engine.count(query_ids=[5613, 4086, 9068]) # natural language processing
engine.count(query_ids=[[[5613, 4086, 9068], [23116, 21082]]]) # natural language processing OR artificial intelligence
engine.count(query_ids=[[[5613, 4086, 9068]], [[6483, 6509]]]) # natural language processing AND deep learning
engine.count(query_ids=[[[5613, 4086, 9068], [23116, 21082]], [[6483, 6509]]]) # (natural language processing OR artificial intelligence) AND deep learning
engine.prob(query_ids=[5613, 4086, 9068]) # natural language processing
engine.ntd(query_ids=[5613, 4086]) # natural language
engine.infgram_prob(query_ids=[3234, 4324] + [5613, 4086, 9068]) # * * natural language processing
engine.infgram_ntd(query_ids=[3234, 4324] + [5613, 4086]) # * * natural language
engine.search_docs(query_ids=[5613, 4086, 9068], maxnum=10) # natural language processing
```

The interface is essentially identical to our [API endpoint](https://infini-gram.io/api_doc), with the following distinctions:
  * The input only takes `query_ids`. You need to tokenize your query yourself.
  * The output does not contain fields `token_ids`, `tokens`, and `latency`.

## Pre-built Indexes

We have made the following indexes publicly available on AWS S3.

Smaller indexes are stored in the `s3://infini-gram-lite` bucket and can be downloaded for free and without an AWS account.
To download, run command `aws s3 cp --no-sign-request --recursive {S3_URL} {LOCAL_INDEX_PATH}`

Larger indexes are stored in the `s3://infini-gram` bucket.
To download these indexes, you need to pay for the data transfer fee (~$0.09 per GB according to [AWS S3 pricing](https://aws.amazon.com/s3/pricing/)).
Make sure you have correctly set up your AWS credentials before downloading these indexes.
To download, run command `aws s3 cp --request-payer requester --recursive {S3_URL} {LOCAL_INDEX_PATH}`

| Name | Documents | Tokens | Storage | Corpus | Tokenizer | S3 URL |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `v4_rpj_llama_s4` | 931,361,530 | 1,385,942,948,192 | 8.9TiB | [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram/index/v4_rpj_llama_s4> |
| `v4_piletrain_llama` | 210,607,728 | 383,299,322,520 | 2.5TiB | [Pile-train](https://huggingface.co/datasets/EleutherAI/pile) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram/index/v4_piletrain_llama> |
| `v4_c4train_llama` | 364,868,892 | 198,079,554,945 | 1.3TiB | [C4-train](https://huggingface.co/datasets/allenai/c4) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram/index/v4_c4train_llama> |
| `v4_pileval_llama` | 214,670 | 393,769,120 | 2.3GiB | [Pile-val](https://huggingface.co/datasets/EleutherAI/pile) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram-lite/index/v4_pileval_llama> |
| `v4_pileval_gpt2` | 214,670 | 383,326,404 | 2.2GiB | [Pile-val](https://huggingface.co/datasets/EleutherAI/pile) | [GPT-2](https://huggingface.co/gpt2) | <s3://infini-gram-lite/index/v4_pileval_gpt2> |
| `v4_dolmasample_olmo` | 13,095,416 | 8,039,098,124 | 53GiB | [Dolma-v1.6-sample](https://huggingface.co/datasets/allenai/dolma) | [OLMo](https://huggingface.co/allenai/OLMo-7B) | <s3://infini-gram-lite/index/v4_dolmasample_olmo> |

## Limitations

1. This package should work on any Linux distribution. Sorry no MacOS or Windows support :)
1. Supported architectures are `x86_64` and `i686`.
1. Your system needs to be little-endian.
1. Please make sure you have Python >=3.8 (and strictly speaking, CPython).

## License

This package is licensed under the [UW Academic Software License](https://infini-gram.io/LICENSE).
Use by universities and non-profit institutions is allowed.
Commercial use is not allowed.
A copy of the license is enclosed with the package distribution.

## Citation

If you find infini-gram useful, please kindly cite our paper:
```bibtex
@article{Liu2024InfiniGram,
  title={Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens},
  author={Liu, Jiacheng and Min, Sewon and Zettlemoyer, Luke and Choi, Yejin and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2401.17377},
  year={2024}
}
```
