# infini-gram

This package allows you to run the infini-gram engine on any existing infini-gram index.

(We will soon support building your own index on datasets of your choice. Stay tuned!)

To learn more about infini-gram:
* Paper: <https://arxiv.org/abs/2401.17377>
* Project Home: <https://infini-gram.io>
* API Endpoint: <https://infini-gram.io/api_doc>
* Web Interface: <https://infini-gram.io/demo>

## Overview

Running the infini-gram engine is easy!
Here's a minimal example:

```python
from infini_gram.engine import InfiniGramEngine

engine = InfiniGramEngine(index_dir='/index/v4_pileval_llama', eos_token_id=2)

engine.count(query_ids=[5613, 4086, 9068]) # natural language processing
# returns {'count': 76, 'approx': False}
```

## Usage

1. Install this package.
```bash
pip install infini-gram
```

2. Download the infini-gram index that you would like to query. See details

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
You may download the index with `aws s3 cp --recursive s3://infini-gram/index/{INDEX_NAME} {LOCAL_INDEX_PATH}`

| Name | Documents | Tokens | Storage | Corpus | Tokenizer |
| --- | ---: | ---: | ---: | --- | --- |
| `v4_c4train_llama` | 364,868,892 | 198,079,554,945 | 1.3TiB | C4-train | Llama-2 |
| `v4_pileval_llama` | 214,670 | 393,769,120 | 2.3GiB | Pile-val | Llama-2 |
| `v4_pileval_gpt2` | 214,670 | 383,326,404 | 2.2GiB | Pile-val | GPT-2 |
| `v4_dolmasample_olmo` | 13,095,416 | 8,039,098,124 | 53GiB | Dolma-sample | OLMo |
<!-- | `v4_rpj_llama_s4` | 931,361,530 | 1,385,942,948,192 | 8.9TiB | RedPajama | Llama-2 | -->
<!-- | `v4_piletrain_llama` | 210,607,728 | 383,299,322,520 | 2.5TiB | Pile-train | Llama-2 | -->

## Limitations

1. This package should work on any Linux distribution. Sorry no MacOS or Windows support :)
1. Your system needs to be little-endian.
1. Please make sure you have Python >=3.8.

## License

This package is licensed under the UW Academic Software License.
Commercial use is not allowed.

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
