CHANGELOG
=========

2024-06-28
----------

* Dolma v1.7 is now available! We will deprecate the Dolma v1.6 index (``v4_dolma-v1_6_llama``) on July 7.

2024-06-12
----------

* The ``count`` query now has two optional fields, ``max_clause_freq`` and ``max_diff_tokens``, both are for customizing CNF queries.
* The ``ntd`` and ``infgram_ntd`` queries now has an optional field, ``max_support``, which is to customize the accuracy of result.
* The ``search_docs`` query now has four optional fields. ``max_clause_freq`` and ``max_diff_tokens`` which are similar to those in ``count`` queries. ``max_disp_len`` which controls how many tokens to return per document. ``maxnum`` is now optional with a default value of 1.

2024-06-06
----------

* The input field ``corpus`` is renamed to ``index``. Support for ``corpus`` will be discontinued sometime in the future. Please update your scripts accordingly.
* ``count``, ``ntd``, and ``infgram_ntd`` queries now returns an extra field ``approx``.
* ``infgram_prob`` and ``infgram_ntd`` queries now returns an extra field ``suffix_len``.
* For ``search_docs`` queries, each returned document now contains an extra field ``token_ids``.

2024-05-08
----------

* The ``count`` query now supports CNF inputs, similar to ``search_docs``.

2024-04-15
----------

* We're lifting the restriction on concurrent requests and sleeping between requests. Now it should be OK to issue concurrent requests. Though our server is serving a lot of requests and you may experience longer latency.

2024-03-02
----------

* The API now supports inputting a list of token IDs in place of a string as the query. Check out the ``query_ids`` field.

2024-02-23
----------

* The output field ``tokenized`` is deprecated and replaced by ``token_ids`` and ``tokens`` in all query types (except in ``search_docs``, where the new fields are ``token_idsss`` and ``tokensss``). The ``tokenized`` field will be removed on 2024-03-01.
* For ``ntd`` and ``infgram_ntd`` queries, there is now a new output field ``prompt_cnt``, and the output field ``ntd`` is deprecated and replaced by ``result_by_token_id``. The ``ntd`` field will be removed on 2024-03-01.
* For ``search_docs`` queries, the output field ``docs`` is deprecated and replaced by ``documents``, which contains additional metadata of the retrieved documents. The ``docs`` field will be removed on 2024-03-01.
