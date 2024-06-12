---
layout: page
title: Python Package
---

## Python Package

Infini-gram offers a Python package, which allows you to run the infini-gram engine on your own machine.
You can access all functionalities offered by the API Endpoint and the Web Interface, plus a little extra, while being spare of the annoying network latency and rate limits.

Currently the package supports running the engine on existing infini-gram indexes, which we have opened up for download.
We will soon support building new indexes yourself on datasets of your choice.

## Overview

Running the infini-gram engine is easy!
Here's a minimal example:

```python
>>> from infini_gram.engine import InfiniGramEngine
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
>>> engine = InfiniGramEngine(index_dir='index/v4_pileval_llama', eos_token_id=tokenizer.eos_token_id)

>>> input_ids = tokenizer.encode(' natural language processing') # prepend space to indicate beginning of word
>>> input_ids
[5613, 4086, 9068]
>>> engine.count(input_ids=input_ids)
{'count': 76, 'approx': False}
```

## This Python Package vs. the API Endpoint

The Python package has all the functionalities of the API endpoint, plus a few extra features:
1. There is no hard upper limit on query parameters (`max_support`, `max_clause_freq`, `max_diff_tokens`, `maxnum`, `max_disp_len`). The only limit will be your machine's compute power.
1. In addition to retrieving random documents that match your query, you can enumerate these documents sequentially.

There are a few other distinctions:
1. Inputting query strings is not allowed. You need to tokenize your query yourself.
1. CNF queries have separate method names (`count_cnf()`, `search_docs_cnf()`) from simple queries (`count()`, `search_docs()`).
1. The input field `query_ids` is replaced by more specific names: `input_ids`, `cnf`, `prompt_ids` and/or `cont_ids`.
1. The output does not contain fields `token_ids`, `tokens`, and `latency`.

## Installation

1. Check your system and make sure it satisfies the following requirements:
  * This package should work on any Linux distribution. Sorry no MacOS or Windows support :)
  * Supported architectures are `x86_64` and `i686`.
  * Your system needs to be little-endian. This should be the case for most modern machines.
  * Please make sure you have Python >=3.8 (and strictly speaking, CPython, not PyPy or some other implementations).

2. Install this package: `pip install infini-gram`

3. Download the infini-gram index that you would like to query. See details in the "Pre-built Indexes" section below.

### Pre-built Indexes

We have made the following indexes publicly available on AWS S3.

Smaller indexes are stored in the `s3://infini-gram-lite` bucket and can be downloaded for free and without an AWS account.
These indexes are `v4_pileval_llama`, `v4_pileval_gpt2`, and `v4_dolmasample_olmo`.
To download, run command `aws s3 cp --no-sign-request --recursive {S3_URL} {LOCAL_INDEX_PATH}`

Larger indexes are stored in the `s3://infini-gram` bucket.
To download these indexes, you need to pay for the data transfer fee (~$0.09 per GB according to [AWS S3 pricing](https://aws.amazon.com/s3/pricing/)).
Make sure you have correctly set up your AWS credentials before downloading these indexes.
These indexes are `v4_rpj_llama_s4`, `v4_piletrain_llama`, and `v4_c4train_llama`.
To download, run command `aws s3 cp --request-payer requester --recursive {S3_URL} {LOCAL_INDEX_PATH}`

| Name | Documents | Tokens | Storage | Corpus | Tokenizer | S3 URL |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `v4_rpj_llama_s4` | 931,361,530 | 1,385,942,948,192 | 8.9TiB | [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram/index/v4_rpj_llama_s4> |
| `v4_piletrain_llama` | 210,607,728 | 383,299,322,520 | 2.5TiB | [Pile-train](https://huggingface.co/datasets/EleutherAI/pile) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram/index/v4_piletrain_llama> |
| `v4_c4train_llama` | 364,868,892 | 198,079,554,945 | 1.3TiB | [C4-train](https://huggingface.co/datasets/allenai/c4) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram/index/v4_c4train_llama> |
| `v4_pileval_llama` | 214,670 | 393,769,120 | 2.3GiB | [Pile-val](https://huggingface.co/datasets/EleutherAI/pile) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) | <s3://infini-gram-lite/index/v4_pileval_llama> |
| `v4_pileval_gpt2` | 214,670 | 383,326,404 | 2.2GiB | [Pile-val](https://huggingface.co/datasets/EleutherAI/pile) | [GPT-2](https://huggingface.co/gpt2) | <s3://infini-gram-lite/index/v4_pileval_gpt2> |
| `v4_dolmasample_olmo` | 13,095,416 | 8,039,098,124 | 53GiB | [Dolma-v1.6-sample](https://huggingface.co/datasets/allenai/dolma) | [OLMo](https://huggingface.co/allenai/OLMo-7B) | <s3://infini-gram-lite/index/v4_dolmasample_olmo> |

## Usage

Prior to submitting any type of queries, you need to instatiate the engine with the index you would like to query.
As an example, below we create an engine with the index for Pile-val (the validation set of Pile), which was created using the Llama-2 tokenizer:
```python
>>> from infini_gram.engine import InfiniGramEngine
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False) # the tokenizer should match that of the index you load below
>>> engine = InfiniGramEngine(index_dir='index/v4_pileval_llama', eos_token_id=tokenizer.eos_token_id) # please replace index_dir with the local directory where you store the index
```

### 1. Count an n-gram (**`count()`** and **`count_cnf()`**)

This query type counts the number of times the query string appears in the corpus.
For example, to find out the number of occurrences of n-gram `natural language processing` in the Pile-val corpus,
```python
>>> input_ids = tokenizer.encode(' natural language processing')
>>> input_ids
[5613, 4086, 9068]

>>> engine.count(input_ids=input_ids)
{'count': 76, 'approx': False}
```
The `approx` field indicates whether the count is approximate.
For simple queries with a single n-gram term, this is always False (the count is always exact).
As you will see later, count for complex queries may be approximate.

If you submit an empty query, the engine returns the total number of tokens in the corpus.
```python
>>> engine.count(input_ids=[])
{'count': 393769120, 'approx': False}
```

You can also make more complex queries by connecting multiple n-grams with the AND/OR operators, in the [CNF format](https://en.wikipedia.org/wiki/Conjunctive_normal_form), in which case the engine counts the number of times where this logical constraint is satisfied.
A CNF query is a triply-nested list.
The top-level is a list of disjunctive clauses (which are eventually connected with the AND operator).
Each disjuctive clause is a list of n-gram terms (which are eventually connected with the OR operator).
And each n-gram term has the same format as `input_ids` above, i.e., a list of token ids.
```python
# natural language processing OR artificial intelligence
>>> cnf = [
...     [tokenizer.encode(' natural language processing'), tokenizer.encode(' artificial intelligence')]
... ]
>>> cnf
[[[5613, 4086, 9068], [23116, 21082]]]

>>> engine.count_cnf(cnf=cnf)
{'count': 499, 'approx': False}

# natural language processing AND deep learning
>>> cnf = [
...     [tokenizer.encode(' natural language processing')],
...     [tokenizer.encode(' deep learning')],
... ]
>>> cnf
[[[5613, 4086, 9068]], [[6483, 6509]]]

>>> engine.count_cnf(cnf=cnf)
{'count': 6, 'approx': False}

# (natural language processing OR artificial intelligence) AND deep learning
>>> cnf = [
...     [tokenizer.encode(' natural language processing'), tokenizer.encode(' artificial intelligence')],
...     [tokenizer.encode(' deep learning')],
... ])
>>> cnf
[[[5613, 4086, 9068], [23116, 21082]], [[6483, 6509]]]

>>> engine.count_cnf(cnf=cnf)
{'count': 19, 'approx': False}
```

**CNF queries and approximation:**
In case the CNF query contains AND operator(s), the engine needs to enumerate all occurrences of each clause and pick cases where they co-occur within reasonable distance.
This distance is controlled by the optional parameter `max_diff_tokens`, which has a default value of 100.
Increasing this value and you may get more counts:
```python
# natural language processing AND deep learning
>>> engine.count_cnf(cnf=[
...     [tokenizer.encode(' natural language processing')],
...     [tokenizer.encode(' deep learning')],
... ], max_diff_tokens=1000)
{'count': 14, 'approx': False}
```
However, if one of the clauses have a too high count, it will be inpractical to enumerate all its occurrences.
Our solution is to take a subsample of its occurrences when the count is higher than a threshold, controlled by the optional parameter `max_clause_freq`, which has a default value of 50000.
When subsampling happens on any of the clauses, the count will be reported as approximate:
```python
>>> engine.count(input_ids=[tokenizer.encode(' this')])
{'count': 739845, 'approx': False}
>>> engine.count(input_ids=[tokenizer.encode(' that')])
{'count': 1866317, 'approx': False}

# this AND that
>>> engine.count_cnf(cnf=[[tokenizer.encode(' this')], [tokenizer.encode(' that')]])
{'count': 982128, 'approx': True}
```
Increasing this value and you will get more accurate estimate of the count, and when this value is larger than (or equal to) the count of all clauses, the count becomes exact.
```python
>>> engine.count_cnf(cnf=[[tokenizer.encode(' this')], [tokenizer.encode(' that')]], max_clause_freq=500000)
{'count': 430527, 'approx': True}

>>> engine.count_cnf(cnf=[[tokenizer.encode(' this')], [tokenizer.encode(' that')]], max_clause_freq=2000000)
{'count': 480107, 'approx': False}
```

### 2. Prob of the last token (**`prob()`**)

This query type computes the n-gram LM probability of a token conditioning on a preceding prompt.

For example, to compute `P(processing | natural language)`:
```python
>>> input_ids = tokenizer.encode(' natural language processing')
>>> input_ids
[5613, 4086, 9068]

>>> engine.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
{'prompt_cnt': 257, 'cont_cnt': 76, 'prob': 0.29571984435797666}
```
In this case, `prompt_cnt` is the count of the 2-gram `natural language`, `cont_cnt` is the count of the 3-gram `natural language processing`, and `prob` is the division of these two counts.

If the prompt cannot be found in the corpus, the probability would be 0/0=NaN.
In these cases we report `prob = -1.0` to indicate an error:
```python
>>> input_ids = tokenizer.encode(' I love natural language processing')
>>> input_ids
[306, 5360, 5613, 4086, 9068]

>>> engine.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
{'prompt_cnt': 0, 'cont_cnt': 0, 'prob': -1.0}
```

### 3. Next-token distribution (**`ntd()`**)

This query type computes the n-gram LM next-token distribution conditioning on a preceding prompt.

For example, this will return the token distribution following `natural language`:
```python
>>> input_ids = tokenizer.encode(' natural language')
>>> input_ids
[5613, 4086]

>>> engine.ntd(prompt_ids=input_ids)
{'prompt_cnt': 257, 'result_by_token_id': {13: {'cont_cnt': 1, 'prob': 0.0038910505836575876}, 297: {'cont_cnt': 1, 'prob': 0.0038910505836575876}, ..., 30003: {'cont_cnt': 1, 'prob': 0.0038910505836575876}}, 'approx': False}
```
`result_by_token_id` is a dict that maps token id to the probability of that token as a continuation of the prompt.

If the prompt cannot be found in the corpus, you will get an empty distribution:
```python
>>> input_ids = tokenizer.encode(' I love natural language processing')
>>> input_ids
[306, 5360, 5613, 4086, 9068]

>>> engine.ntd(prompt_ids=input_ids[:-1])
{'prompt_cnt': 0, 'result_by_token_id': {}, 'approx': False}
```

**Approximation:**
For each occurrence of the prompt, the engine needs to inspect the token appearing after it.
This is time-consuming and not feasible when `prompt_cnt` is large.
After this prompt count crosses a threshold, the engine needs to downsample the number of cases it inspects, and the resulting distribution will become approximate (which will be reflected in the `approx` field).
This threshold is controlled by the optional parameter `max_support`, which has a default value of 1000.
For example, to get the unigram token distribution, you can query with an empty prompt and the result will be approximate:
```python
>>> engine.ntd(prompt_ids=[])
{'prompt_cnt': 393769120, 'result_by_token_id': {12: {'cont_cnt': 1013873, 'prob': 0.00257479052699714}, 13: {'cont_cnt': 14333030, 'prob': 0.03639957851443506}, ..., 30934: {'cont_cnt': 489584, 'prob': 0.0012433275621003496}}, 'approx': True}
```

### 4. ∞-gram prob (**`infgram_prob()`**)

This query type computes the ∞-gram LM probability of a token conditioning on a preceding prompt.
It uses the longest suffix of the prompt that has a non-zero count in the corpus.

```python
>>> input_ids = tokenizer.encode(' I love natural language processing')
>>> input_ids
[306, 5360, 5613, 4086, 9068]

>>> engine.infgram_prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
{'prompt_cnt': 257, 'cont_cnt': 76, 'prob': 0.29571984435797666, 'suffix_len': 2}
```
The field `suffix_len` indicates the number of tokens in the longest suffix of the prompt.
In this case, since `[5613, 4086]` can be found in the corpus, but `[5360, 5613, 4086]` cannot, the longest suffix is `[5613, 4086]`, which has length 2.

### 5. ∞-gram next-token distribution (**`infgram_ntd()`**)

This query type computes the ∞-gram LM next-token distribution conditioning on a preceding prompt.

```python
>>> input_ids = tokenizer.encode(' I love natural language')
>>> input_ids
[306, 5360, 5613, 4086]

>>> engine.infgram_ntd(prompt_ids=input_ids, max_support=10)
{'prompt_cnt': 257, 'result_by_token_id': {297: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 470: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 508: {'cont_cnt': 1, 'prob': 0.0038910505836575876}, 8004: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 9068: {'cont_cnt': 96, 'prob': 0.3735408560311284}, 24481: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 29889: {'cont_cnt': 32, 'prob': 0.1245136186770428}}, 'approx': True, 'suffix_len': 2}
```

### 6. Search documents (**`search_docs()`** and **`search_docs_cnf()`**)

This query type returns a few random documents in the corpus that match your query.
You can view this as an extension to `count()`.
In addition to returning the count of occurrences matching your query, it will also return a random sample (with replacement) of documents containing these occurrences.

For example, to get a random document containing `natural language processing`, you can use the `search_docs()` method:
```python
>>> input_ids = tokenizer.encode(' natural language processing')
>>> input_ids
[5613, 4086, 9068]

>>> engine.search_docs(input_ids=input_ids, maxnum=1, max_disp_len=10)
{'cnt': 76, 'approx': False, 'idxs': [54], 'documents': [{'doc_ix': 142405, 'doc_len': 19238, 'disp_len': 10, 'metadata': '', 'token_ids': [4475, 304, 9045, 2562, 322, 5613, 4086, 9068, 29889, 13]}]}
```
The `document` field is a list of documents.
Please check out the meaning of these fields in the [API Documentation](/api_doc).
`maxnum` controls the number of documents to sample (default value is 1), and `max_disp_len` controls the max number of tokens to return per documents (default value is 1000).
Note that the sampling is **with replacement**, so you might get duplicate documents if you ask for more than one documents.
To get a new batch of random documents, simply run this query again.

CNF queries are also supported!
The method name is `search_docs_cnf()` and its protocol is same as `count_cnf()`:
```python
# natural language processing AND deep learning
>>> cnf = [
...     [tokenizer.encode(' natural language processing')],
...     [tokenizer.encode(' deep learning')],
... ]
>>> cnf
[[[5613, 4086, 9068]], [[6483, 6509]]]

>>> engine.search_docs_cnf(cnf=cnf, maxnum=1, max_disp_len=20)
{'cnt': 6, 'approx': False, 'idxs': [2], 'documents': [{'doc_ix': 191568, 'doc_len': 3171, 'disp_len': 20, 'metadata': '', 'token_ids': [29889, 450, 1034, 13364, 508, 367, 4340, 1304, 304, 7945, 6483, 6509, 2729, 5613, 4086, 9068, 9595, 1316, 408, 10013]}]}
```
Again, you can also use `max_clause_freq` and `max_diff_tokens` to control the behavior of CNF queries.
Note that when the CNF query contains AND operator(s) the the count is approximate, the actual number of retrievable documents will be fewer than the reported count (since this count accounted for the subsampling).

### 6.1 Enumerating all documents containing your query

`search_docs()` and `search_docs_cnf()` returns a random sample of the matching documents.
If you want to enumerate all these documents sequentially, there's a way to do it but it's slightly more complicated.

For simple queries, you need to first call `find()` to get information about where the matching documents are located.
```python
>>> input_ids = tokenizer.encode(' natural language processing')
>>> input_ids
[5613, 4086, 9068]

>>> engine.find(input_ids=input_ids)
{'cnt': 76, 'segment_by_shard': [(365362993, 365363069)]}
```
The returned `segment_by_shard` is a list of 2-tuples, each tuple represents a range of "ranks" in one of the shards of the index, and each rank can be traced back to a matched document in that shard.
The length of this list is equal to the total number of shards.
For example, if you want to retrieve the first matched document in shard 0, you can do
```python
>>> engine.get_doc_by_rank(s=0, rank=365362993, max_disp_len=10)
{'doc_ix': 47865, 'doc_len': 12932, 'disp_len': 10, 'metadata': '', 'token_ids': [363, 5164, 11976, 1316, 408, 5613, 4086, 9068, 518, 29992]}
```
The returned dict represents a document.
You can see that the query input_ids `[5613, 4086, 9068]` is present in this document.

The ranges are left-inclusive and right-exclusive.
To enumerate all documents, you can do something like
```python
>>> find_result = engine.find(input_ids=input_ids)
>>> for s, (start, end) in enumerate(find_result['segment_by_shard']):
...     for rank in range(start, end):
...         doc = engine.get_doc_by_rank(s=s, rank=rank)
```

For CNF queries, you need to first call `find_cnf()` which returns locations of matching documents in a different protocol:
```python
# natural language processing AND deep learning
>>> cnf = [
...     [tokenizer.encode(' natural language processing')],
...     [tokenizer.encode(' deep learning')],
... ]
>>> cnf
[[[5613, 4086, 9068]], [[6483, 6509]]]

>>> engine.find_cnf(cnf=cnf)
{'cnt': 6, 'approx': False, 'ptrs_by_shard': [[717544382, 377178100, 706194108, 25563710, 250933686, 706194476]]}
```
Note that the returned field is not `segment_by_shard` but rather `ptrs_by_shard`.
For each shard, instead of having a range of "ranks", now we get a list of "pointers", and each pointer can be traced back to a matched document in that shard of the index.
The length of the outer list is equal to the total number of shards.
To get documents with these pointers, you need to call a different helper function:
```python
# Get the document at pointer #2 in shard 0
>>> engine.get_doc_by_ptr(s=0, ptr=706194108, max_disp_len=20)
{'doc_ix': 191568, 'doc_len': 3171, 'disp_len': 20, 'metadata': '', 'token_ids': [29889, 450, 1034, 13364, 508, 367, 4340, 1304, 304, 7945, 6483, 6509, 2729, 5613, 4086, 9068, 9595, 1316, 408, 10013]}
```
You can see that both `[5613, 4086, 9068]` and `[6483, 6509]` are present in this document.
(For illustration I use a small `max_disp_len`; since the default `max_diff_tokens = 100`, you might need to increase `max_disp_len` to see the document covering all clauses in the CNF query.)

To enumerate all documents, you can do something like
```python
>>> find_result = engine.find_cnf(cnf=cnf)
>>> for s, ptrs in enumerate(find_result['ptrs_by_shard']):
...     for ptr in ptrs:
...         doc = engine.get_doc_by_ptr(s=s, ptr=ptr)
```

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
