---
layout: page
title: API Doc
---

The infini-gram API endpoint is `https://api.infini-gram.io/`.
Please make regular HTTP POST requests.
In your request, please include a JSON payload, and the response will also contain a JSON payload.

---
<br/>

## Overview

**Input parameters:**

In general, the request JSON payload should be a dict containing the following fields:

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `corpus` | The corpus to search in (and the tokenizer to use) | `v4_pileval_gpt2`, `v4_pileval_llama`, `v4_piletrain_llama`, `v4_rpj_llama_s4`, `v4_dolmasample_olmo` |
| `query_type` | One of the six supported query types | `count`, `prob`, `ntd`, `infgram_prob`, `infgram_ntd`, `search_docs` |
| `query` | The query (semantic depends on query type) | Any string (empty may be OK depending on query type) |

For certain query types, additional fields may be required.
Please see the specific query type below for more details.

**Output parameters:**

If an error occurred (e.g., malformed input, internal server error), the response JSON dict will contain a key `error` with a string value describing the error.
Please check this key first before processing the rest of the response.

If the query was successful, the response JSON dict will contain the following fields:

| Key | Description | Value Range |
| --- | --- | --- |
| `tokenized` | Debug information about the post-tokenization query | A string |
| `latency` | The processing time in the engine. This does not include network latency. | A non-negative float number, in milliseconds |

In addition, the response JSON dict will contain results specific to the query type.
Please see the specific query type below for more details.

---
<br/>

## 1. Count an n-gram (`count`)

This query type counts the number of times the query string appears in the corpus.
If the query is an empty string, the total number of tokens in the corpus will be returned.

**Example:**
If you query `natural language processing`, the API returns Cnt(`natural language processing`).

**Note:**
The query string will be tokenized into an n-gram, and we only count those occurrences that match the token boundaries.
For example, querying `a` will not give you the count of the letter `a`, but rather the count of the token `_a`.

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `corpus` | see overview | see overview |
| `query_type` | see overview | `count` |
| `query` | The n-gram to count | Any string (empty is OK) |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `tokenized` | see overview | see overview |
| `latency` | see overview | see overview |
| `count` | The count of the query n-gram | A non-negative integer |

---
<br/>

## 2. Prob of the last token (`prob`)

This query type computes the n-gram LM probability of the last token of the query conditioning on all preceding tokens.
It treats your query as an n-gram, counts the full n-gram and also the (n-1)-gram that excludes the last token, and takes the division of the two counts.

**Example:**
If you query `natural language processing`, the API returns P(`processing` | `natural language`) = Cnt(`natural language processing`) / Cnt(`natural language`).

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `corpus` | see overview | see overview |
| `query_type` | see overview | `prob` |
| `query` | The n-gram to query | Any non-empty string |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `tokenized` | see overview | see overview |
| `latency` | see overview | see overview |
| `prob` | The n-gram LM probability | A real number in range [0, 1], or -1.0 (denoting NaN) |
| `prompt_cnt` | The count of the n-1 gram | A non-negative integer |
| `cont_cnt` | The count of the n-gram | A non-negative integer |

---
<br/>

## 3. Next-token distribution (`ntd`)

This query type treats your query as the (n-1)-gram as in query type 2, and returns the full distribution of the next token.

**Example:**
If you query `natural language`, the API returns P(* | `natural language`) for all possible tokens *.

**Note:**
If the query appears more than 1000 times in the corpus, the distribution returned will be approximate.

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `corpus` | see overview | see overview |
| `query_type` | see overview | `ntd` |
| `query` | The (n-1)-gram to query | Any string (empty is OK) |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `tokenized` | see overview | see overview |
| `latency` | see overview | see overview |
| `ntd` | The next-token distribution | A dict that maps tokens to probabilities. (Each key is a string containing the token and the frequencies, if you only want the token you might need to do some parsing) |

---
<br/>

## 4. ∞-gram prob (`infgram_prob`)

This query type computes the ∞-gram LM probability of the last token of the query conditioning on all preceding tokens.
In contrast to n-gram, the ∞-gram LM uses the longest possible (n-1)-gram suffix as context, as long as the count of this (n-1)-gram is non-zero.
For more details on the ∞-gram LM, please refer to our paper.

**Example:**
If you query `I love natural language processing`, and `natural language` appears in the corpus but `love natural language` does not, then the API returns P(`processing` | `natural language`).

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `corpus` | see overview | see overview |
| `query_type` | see overview | `infgram_prob` |
| `query` | The sequence to query | Any non-empty string |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `tokenized` | see overview | see overview |
| `latency` | see overview | see overview |
| `prob` | The n-gram LM probability | A real number in range [0, 1] |
| `prompt_cnt` | The count of the n-1 gram | A non-negative integer |
| `cont_cnt` | The count of the n-gram | A non-negative integer |
| `longest_suffix` | The longest suffix used to compute the ∞-gram probability | A string (may be empty) |

---
<br/>

## 5. ∞-gram next-token distribution (`infgram_ntd`)

This query type computes the full next-token distribution according to the ∞-gram LM.
It uses the longest possible (n-1)-gram suffix of the query as context, as long as the count of this (n-1)-gram is non-zero.
For more details on the ∞-gram LM, please refer to our paper.

**Example:**
If you query `I love natural language`, and `natural language` appears in the corpus but `love natural language` does not, then the API returns P(* | `natural language`) for all possible tokens *.

**Input parameters:**

| Key | Description | Acceptable Values |
| --- | --- | --- |
| `corpus` | see overview | see overview |
| `query_type` | see overview | `infgram_ntd` |
| `query` | The (n-1)-gram to query | Any string (empty is OK) |

**Output parameters:**

| Key | Description | Value Range |
| --- | --- | --- |
| `tokenized` | see overview | see overview |
| `latency` | see overview | see overview |
| `ntd` | The next-token distribution | A dict that maps tokens to probabilities. (Each key is a string containing the token and the frequencies, if you only want the token you might need to do some parsing) |
| `longest_suffix` | The longest suffix used to compute the ∞-gram probability | A string (may be empty) |
