---
layout: page
redirect_to:
  - https://infini-gram.readthedocs.io/en/latest/api.html
title: API Endpoint
---

## API Endpoint

The infini-gram API endpoint is `https://api.infini-gram.io/`.
Please make regular HTTP POST requests.
In your request, please include a JSON payload, and the response will also contain a JSON payload.

Most queries are processed within 10s of milliseconds.
You should receive the response within a fraction of a second.
If you are experiencing longer latencies, it might be due to network delays or heavy traffic.

**Please wrap your requests in an exception-handling loop. We can't guarantee 100% uptime or successful processing of queries, so it may be wise to catch errors and retry the failed requests.**

If you find infini-gram useful, please kindly cite our paper:
```bibtex
@article{Liu2024InfiniGram,
  title={Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens},
  author={Liu, Jiacheng and Min, Sewon and Zettlemoyer, Luke and Choi, Yejin and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2401.17377},
  year={2024}
}
```

---
<br/>

## Updates

### 2024-06-28

* Dolma v1.7 is now available! We will deprecate the Dolma v1.6 index (`v4_dolma-v1_6_llama`) on July 7.

### 2024-06-12

* The `count` query now has two optional fields, `max_clause_freq` and `max_diff_tokens`, both are for customizing CNF queries.
* The `ntd` and `infgram_ntd` queries now has an optional field, `max_support`, which is to customize the accuracy of result.
* The `search_docs` query now has four optional fields. `max_clause_freq` and `max_diff_tokens` which are similar to those in `count` queries. `max_disp_len` which controls how many tokens to return per document. `maxnum` is now optional with a default value of 1.

### 2024-06-06

* The input field `corpus` is renamed to `index`. Support for `corpus` will be discontinued sometime in the future. Please update your scripts accordingly.
* `count`, `ntd`, and `infgram_ntd` queries now returns an extra field `approx`.
* `infgram_prob` and `infgram_ntd` queries now returns an extra field `suffix_len`.
* For `search_docs` queries, each returned document now contains an extra field `token_ids`.

### 2024-05-08

* The `count` query now supports CNF inputs, similar to `search_docs`.

### 2024-04-15

* We're lifting the restriction on concurrent requests and sleeping between requests. Now it should be OK to issue concurrent requests. Though our server is serving a lot of requests and you may experience longer latency.

### 2024-03-02

* The API now supports inputting a list of token IDs in place of a string as the query. Check out the `query_ids` field.

### 2024-02-23

* The output field `tokenized` is deprecated and replaced by `token_ids` and `tokens` in all query types (except in `search_docs`, where the new fields are `token_idsss` and `tokensss`). The `tokenized` field will be removed on 2024-03-01.
* For `ntd` and `infgram_ntd` queries, there is now a new output field `prompt_cnt`, and the output field `ntd` is deprecated and replaced by `result_by_token_id`. The `ntd` field will be removed on 2024-03-01.
* For `search_docs` queries, the output field `docs` is deprecated and replaced by `documents`, which contains additional metadata of the retrieved documents. The `docs` field will be removed on 2024-03-01.

---
<br/>

## Example usage

**From Shell:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"index": "v4_rpj_llama_s4", "query_type": "count", "query": "University of Washington"}' https://api.infini-gram.io/
```
Outputs:
```json
{"count":2349754,"approx":false,"latency":0.73,"token_ids":[3014,310,7660],"tokens":["\u2581University","\u2581of","\u2581Washington"]}
```

**From Python:**
```python
import requests

payload = {
    'index': 'v4_rpj_llama_s4',
    'query_type': 'count',
    'query': 'University of Washington',
}
result = requests.post('https://api.infini-gram.io/', json=payload).json()
print(result)
```
Outputs:
```
{'count': 2349754, 'approx': False, 'latency': 0.724, 'token_ids': [3014, 310, 7660], 'tokens': ['▁University', '▁of', '▁Washington']}
```

---
<br/>

## Overview

**Available indexes:**

We have built the infini-gram indexes on several corpora, and you may query them through the API.

| Name | Documents | Tokens | Corpus | Tokenizer |
| --- | ---: | ---: | --- | --- |
| `v4_dolma-v1_7_llama` | 3,403,336,408 | 2,604,642,372,173 | [Dolma-v1.7](https://huggingface.co/datasets/allenai/dolma) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| `v4_rpj_llama_s4` | 931,361,530 | 1,385,942,948,192 | [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| `v4_piletrain_llama` | 210,607,728 | 383,299,322,520 | [Pile-train](https://huggingface.co/datasets/EleutherAI/pile) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| `v4_c4train_llama` | 364,868,892 | 198,079,554,945 | [C4-train](https://huggingface.co/datasets/allenai/c4) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| `v4_dolma-v1_6-sample_llama` | 13,095,416 | 9,178,218,956 | [Dolma-v1.6-sample](https://huggingface.co/datasets/allenai/dolma) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| `v4_dolmasample_olmo` | 13,095,416 | 8,039,098,124 | [Dolma-v1.6-sample](https://huggingface.co/datasets/allenai/dolma) | [OLMo](https://huggingface.co/allenai/OLMo-7B) |
| `v4_pileval_llama` | 214,670 | 393,769,120 | [Pile-val](https://huggingface.co/datasets/EleutherAI/pile) | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| `v4_pileval_gpt2` | 214,670 | 383,326,404 | [Pile-val](https://huggingface.co/datasets/EleutherAI/pile) | [GPT-2](https://huggingface.co/gpt2) |

**Input parameters:**

In general, the request JSON payload should be a dict containing the following fields:

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `index` | The index to search in | E.g., `v4_rpj_llama_s4`. See full list in the table in the "Available indexes" section above. |
| `query_type` | One of the six supported query types | `count`, `prob`, `ntd`, `infgram_prob`, `infgram_ntd`, `search_docs` |
| `query` or `query_ids` | The query (semantic depends on query type) | If `query`: Any string. If `query_ids`: A list of integers. (Empty may be OK depending on query type) |

For certain query types, additional fields may be required.
Please see the specific query type below for more details.

If you supply a `query` string, it will be tokenized into an n-gram, and we only find results that match the token boundaries. For example, doing a count query with query=`a` will not give you the count of the letter `a`, but rather the count of the token `_a`.

**Output parameters:**

If an error occurred (e.g., malformed input, internal server error), the response JSON dict will contain a key `error` with a string value describing the error.
Please check this key first before processing the rest of the response.

If the query was successful, the response JSON dict will contain the following fields:

| Key | Description | Value Range |
| --- | --- | --- |
| `token_ids` | The token IDs in the tokenized query | A list of integers |
| `tokens` | The tokens in the tokenized query | A list of strings |
| `latency` | The processing time in the engine. This does not include network latency. | A non-negative float number, in milliseconds |

In addition, the response JSON dict will contain results specific to the query type.
Please see the specific query type below for more details.

---
<br/>

## 1. Count an n-gram (**count**)

This query type counts the number of times the query string appears in the corpus.
If the query is an empty string, the total number of tokens in the corpus will be returned.

You can simply enter a string, in which we count the number of occurrences of the string.
You can also connect multiple strings with the AND/OR operators, in the [CNF format](https://en.wikipedia.org/wiki/Conjunctive_normal_form), in which case we count the number of times where this logical constraint is satisfied.

**Examples:**

1. If you query `natural language processing`, the API returns the number of occurrences of `natural language processing`.
2. If you query `natural language processing AND deep learning`, the API returns the number of co-occurrences of `natural language processing` and `deep learning`.
3. If you query `natural language processing OR artificial intelligence AND deep learning OR machine learning`, the API returns the number of co-occurrences of {one of `natural language processing` / `artificial intelligence`}, and {one of `deep learning` / `machine learning`}.

**Notes on CNF queries:**

* When you write a query in CNF, note that **OR has higher precedence than AND** (which is contrary to conventions in boolean algebra).
* In AND queries, we can only examine co-occurrences where adjacent clauses are separated by no more than 100 tokens. You can customize this value by supplying a field `max_diff_tokens` with an integer value within range [1, 1000].
* In AND queries, if a clause has more than 50000 matches, the count will be **approximate**: we will estimate the count by examining a random subset of 50000 documents out of all documents containing that clause. You can customize this value by supplying a field `max_clause_freq` with an integer value within range [1, 500000].
* The above subsampling mechanism might cause a zero count on conjuction of some simple clauses (e.g., `birds AND oil`).

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `index` | see overview | see overview |
| `query_type` | see overview | `count` |
| `query` or `query_ids` | The n-gram to count | If `query`: A string (empty is OK), or several non-empty strings connected with the AND/OR operators. If `query_ids`: A list of integers, or a triply-nested list of integers (see below for details). |
| [Optional] `max_clause_freq` | For CNF queries only. The maximum clause frequency before subsampling happens. | An integer in range [1, 500000], default = 50000 |
| [Optional] `max_diff_tokens` | For CNF queries only. The maximum distance between adjacent clauses. | An integer in range [1, 1000], default = 100 |

If you input `query_ids`, it should be either a list of integers (for simple queries), or a list of list of list of integers (for CNF queries).
In case of CNF queries:
The inner-most list is a list of token IDs for a query term;
The second-level list represent a disjunctive clause, i.e., query terms connected with OR;
The outer-most list represents a CNF, i.e., disjuctive clauses connected with AND.

Here are some examples of equivalent `query` and `query_ids` (Assuming a Llama-2 tokenizer):
* `natural language processing` <==> `[5613, 4086, 9068]`, or `[[[5613, 4086, 9068]]]` if you will
* `natural language processing OR deep learning` <==> `[[[5613, 4086, 9068], [6483, 6509]]]`
* `natural language processing AND deep learning` <==> `[[[5613, 4086, 9068]], [[6483, 6509]]]`
* `natural language processing OR artificial intelligence AND deep learning OR machine learning` <==> `[[[5613, 4086, 9068], [23116, 21082]], [[6483, 6509], [4933, 6509]]]`

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `token_ids` | see overview | see overview |
| `tokens` | see overview | see overview |
| `latency` | see overview | see overview |
| `count` | The number of occurrences of the query | A non-negative integer |
| `approx` | Whether the count is approximate | False (for exact) and True (for approximate) |

---
<br/>

## 2. Prob of the last token (**prob**)

This query type computes the n-gram LM probability of the last token of the query conditioning on all preceding tokens.
It treats your query as an n-gram, counts the full n-gram and also the (n-1)-gram that excludes the last token, and takes the division of the two counts.

**Examples:**

* If you query `natural language processing`, the API returns `P(processing | natural language) = Cnt(natural language processing) / Cnt(natural language)`.

**Notes:**

* If the (n-1)-gram is not found in the corpus, the returned probability will be -1.0 (to denote NaN).

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `index` | see overview | see overview |
| `query_type` | see overview | `prob` |
| `query` or `query_ids` | The n-gram to query | Any non-empty string or list of integers |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `token_ids` | see overview | see overview |
| `tokens` | see overview | see overview |
| `latency` | see overview | see overview |
| `prob` | The n-gram LM probability | A real number in range [0, 1], or -1.0 (denoting NaN) |
| `prompt_cnt` | The count of the (n-1)-gram | A non-negative integer |
| `cont_cnt` | The count of the n-gram | A non-negative integer |

---
<br/>

## 3. Next-token distribution (**ntd**)

This query type treats your query as the (n-1)-gram as in query type 2, and returns the full distribution of the next token.

**Examples:**

* If you query `natural language`, the API returns `P(* | natural language)` for all possible tokens `*`.

**Notes:**

* If the (n-1)-gram is not found in the corpus, you will get an empty distribution.
* If the (n-1)-gram appears more than 1000 times in the corpus, the result will be **approximate**: we will estimate the distribution by examining a subset of 1000 occurrences of the (n-1)-gram. You can customize this value by supplying a field `max_support` with an integer value within range [1, 1000].

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `index` | see overview | see overview |
| `query_type` | see overview | `ntd` |
| `query` or `query_ids` | The (n-1)-gram to query | Any string or list of integers (empty is OK) |
| [Optional] `max_support` | The maximum (n-1)-gram frequency before subsampling happens. | An integer in range [1, 1000], default = 1000 |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `token_ids` | see overview | see overview |
| `tokens` | see overview | see overview |
| `latency` | see overview | see overview |
| `prompt_cnt` | The count of the (n-1)-gram | A non-negative integer |
| `result_by_token_id` | The next token distribution | A dict that maps token IDs to results. Each result is a dict with the following keys: `token` (str, the token string), `prob` (float, the probability of this token), `cont_cnt` (int, the count of the n-gram formed by appending this token) |
| `approx` | Whether the distribution is approximate | False (for exact) and True (for approximate) |

---
<br/>

## 4. ∞-gram prob (**infgram_prob**)

This query type computes the ∞-gram LM probability of the last token of the query conditioning on all preceding tokens.
In contrast to n-gram, the ∞-gram LM uses the longest possible (n-1)-gram suffix as context, as long as the count of this (n-1)-gram is non-zero.
For more details on the ∞-gram LM, please refer to our paper.

**Examples:**

* If you query `I love natural language processing`, and `natural language` appears in the corpus but `love natural language` does not, then the API returns `P(processing | natural language)`.

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `index` | see overview | see overview |
| `query_type` | see overview | `infgram_prob` |
| `query` or `query_ids` | The sequence to query | Any non-empty string or list of integers |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `token_ids` | see overview | see overview |
| `tokens` | see overview | see overview |
| `latency` | see overview | see overview |
| `prob` | The n-gram LM probability | A real number in range [0, 1] |
| `prompt_cnt` | The count of the (n-1)-gram | A non-negative integer |
| `cont_cnt` | The count of the n-gram | A non-negative integer |
| `suffix_len` | The number of tokens in the longest suffix used to compute the ∞-gram probability | A non-negative integer |
| `longest_suffix` | The longest suffix used to compute the ∞-gram probability | A string (may be empty) |

---
<br/>

## 5. ∞-gram next-token distribution (**infgram_ntd**)

This query type computes the full next-token distribution according to the ∞-gram LM.
It uses the longest possible (n-1)-gram suffix of the query as context, as long as the count of this (n-1)-gram is non-zero.
For more details on the ∞-gram LM, please refer to our paper.

**Examples:**

* If you query `I love natural language`, and `natural language` appears in the corpus but `love natural language` does not, then the API returns `P(* | natural language)` for all possible tokens `*`.

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `index` | see overview | see overview |
| `query_type` | see overview | `infgram_ntd` |
| `query` or `query_ids` | The sequence to query | Any string or list of integers (empty is OK) |
| [Optional] `max_support` | The maximum (n-1)-gram frequency before subsampling happens. | An integer in range [1, 1000], default = 1000 |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `token_ids` | see overview | see overview |
| `tokens` | see overview | see overview |
| `latency` | see overview | see overview |
| `prompt_cnt` | The count of the (n-1)-gram (where (n-1) is the number of tokens in the longest suffix) | A non-negative integer |
| `result_by_token_id` | The next token distribution | A dict that maps token IDs to results. Each result is a dict with the following keys: `token` (str, the token string), `prob` (float, the probability of this token), `cont_cnt` (int, the count of the n-gram formed by appending this token) |
| `approx` | Whether the distribution is approximate | False (for exact) and True (for approximate) |
| `suffix_len` | The number of tokens in the longest suffix used to compute the ∞-gram probability | A non-negative integer |
| `longest_suffix` | The longest suffix used to compute the ∞-gram probability | A string (may be empty) |

---
<br/>

## 6. Search documents (**search_docs**)

This query type returns a few random documents in the corpus that match your query.

You can simply enter a string, in which case the documents returned would contain the string.
You can also connect multiple strings with the AND/OR operators, in the [CNF format](https://en.wikipedia.org/wiki/Conjunctive_normal_form), in which case the returned documents contains strings such that it satisfies this logical constraint.

**Examples:**

1. If you query `natural language processing`, the documents returned would contain the string `natural language processing`.
2. If you query `natural language processing AND deep learning`, the documents returned would contain both `natural language processing` and `deep learning`.
3. If you query `natural language processing OR artificial intelligence AND deep learning OR machine learning`, the documents returned would contain at least one of `natural language processing` / `artificial intelligence`, and also at least one of `deep learning` / `machine learning`.

If you want another batch of random documents, simply submit the same query again :)

**Notes on CNF queries:**

* When you write a query in CNF, note that **OR has higher precedence than AND** (which is contrary to conventions in boolean algebra).
* In AND queries, we can only examine co-occurrences where adjacent clauses are separated by no more than 100 tokens. You can customize this value by supplying a field `max_diff_tokens` with an integer value within range [1, 1000].
* In AND queries, if a clause has more than 50000 matches, the count will be **approximate**: we will estimate the count by examining a random subset of 50000 documents out of all documents containing that clause. You can customize this value by supplying a field `max_clause_freq` with an integer value within range [1, 500000].
* The above subsampling mechanism might cause a zero count on conjuction of some simple clauses (e.g., `birds AND oil`).

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `index` | see overview | see overview |
| `query_type` | see overview | `search_docs` |
| `query` or `query_ids` | The search query | If `query`: A non-empty string, or several such strings connected with the AND/OR operators. If `query_ids`: A list of integers, or a triply-nested list of integers (see below for details). |
| [Optional] `maxnum` | The max number of documents to return | An integer in range [1, 10], default = 1 |
| [Optional] `max_disp_len` | The max number of tokens to return for each document | An integer in range [1, 10000], default = 1000 |
| [Optional] `max_clause_freq` | For CNF queries only. The maximum clause frequency before subsampling happens. | An integer in range [1, 500000], default = 50000 |
| [Optional] `max_diff_tokens` | For CNF queries only. The maximum distance between adjacent clauses. | An integer in range [1, 1000], default = 100 |

If you input `query_ids`, it should be either a list of integers (for simple queries), or a list of list of list of integers (for CNF queries).
In case of CNF queries:
The inner-most list is a list of token IDs for a query term;
The second-level list represent a disjunctive clause, i.e., query terms connected with OR;
The outer-most list represents a CNF, i.e., disjuctive clauses connected with AND.

Here are some examples of equivalent `query` and `query_ids` (Assuming a Llama-2 tokenizer):
* `natural language processing` <==> `[5613, 4086, 9068]`, or `[[[5613, 4086, 9068]]]` if you will
* `natural language processing OR deep learning` <==> `[[[5613, 4086, 9068], [6483, 6509]]]`
* `natural language processing AND deep learning` <==> `[[[5613, 4086, 9068]], [[6483, 6509]]]`
* `natural language processing OR artificial intelligence AND deep learning OR machine learning` <==> `[[[5613, 4086, 9068], [23116, 21082]], [[6483, 6509], [4933, 6509]]]`

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `token_ids` | The token IDs in the tokenized query | A list of integers, or a triply-nested list of integers |
| `tokens` | The tokens in the tokenized query | A list of strings, or a triply-nested list of strings |
| `latency` | see overview | see overview |
| `cnt` | The number of occurrences of the query | A non-negative integer |
| `approx` | Whether the count is approximate | False (for exact) and True (for approximate) |
| `idxs` | The indexes of the returned documents within all matches | A list of non-negative integers |
| `documents` | A list of documents randomly sampled from those that match the query | A list of Documents, where each Document is a dict with the following keys: `doc_ix` (int, the index of this document in the corpus), `doc_len` (int, the total number of tokens in this document), `disp_len` (int, the number of tokens returned after truncation), `token_ids` (a list of integers: the tokenized version of the document), `spans` (a list of tuples: each tuple's first element is a span of text and it second element is a string marking the index of the clause that this span matches; if this span does not match any clause, this second element is NULL) |
| `message` | A message describing the total number of matched documents | A string |

---
<br/>

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
