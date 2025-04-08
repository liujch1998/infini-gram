Python Package
==============

Infini-gram offers a Python package, which allows you to run the infini-gram engine on your own machine with indexes stored locally.
You can access all functionalities offered by the API Endpoint and the Web Interface, plus a little extra, while sparing yourself the annoying network latency and rate limits.

You can run the engine on our pre-built indexes, which we have made available for download.
You can also build new indexes on datasets of your choice.

Getting Started
---------------

To make queries on a local index, you first need to instantiate an engine with this index, and then you can make queries by invoking the appropriate methods in the engine.
Here's a minimal example to get started:

.. code-block:: python

   >>> from infini_gram.engine import InfiniGramEngine
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
   >>> engine = InfiniGramEngine(index_dir='index/v4_pileval_llama', eos_token_id=tokenizer.eos_token_id)

   >>> input_ids = tokenizer.encode('natural language processing')
   >>> input_ids
   [5613, 4086, 9068]
   >>> engine.count(input_ids=input_ids)
   {'count': 76, 'approx': False}

You can read about other query types in the :ref:`usage` section below.

This Python Package vs. the API Endpoint
----------------------------------------

The Python package has all functionalities of the API endpoint, plus a few extra features:

1. There is no hard upper limit on query parameters (e.g., ``max_support``, ``max_clause_freq``, ``max_diff_tokens``, ``max_disp_len``). The only limit will be your machine's compute power.

There are a few other distinctions:

1. Inputting query strings is not allowed. You need to tokenize your query yourself.
2. CNF queries have separate method names (``count_cnf()``, ``find_cnf()``) from simple queries (``count()``, ``find()``).
3. The input field ``query_ids`` is replaced with more specific names (e.g., ``input_ids``, ``cnf``, ``prompt_ids``, ``cont_ids``).
4. The output does not contain fields ``token_ids``, ``tokens``, and ``latency``.

Installation
------------

1. Check your system and make sure it satisfies the following requirements:
  * You have Linux or MacOS. Sorry no Windows support :)
  * Supported architectures: x86_64 and i686 on Linux; x86_64 and arm64 on MacOS.
  * Your system needs to be little-endian. This should be the case for most modern machines.
  * Make sure you have Python >=3.11 (and strictly speaking, CPython, not PyPy or some other implementations).

2. Install this package: ``pip install infini-gram``

3. If you'd like to run the engine on one of our pre-built indexes, download the index that you would like to query. For sake of performance, it is strongly recommended that you put the index on an SSD. See details in the :ref:`pre-built-indexes` section below.

4. If none of the pre-built indexes fit your need, you can build new indexes on datasets of your own choice. See details in :doc:`indexing`.

.. _pre-built-indexes:

Pre-built Indexes
~~~~~~~~~~~~~~~~~

We have made the following indexes publicly available on AWS S3:

.. list-table::
   :header-rows: 1

   * - Name
     - Documents
     - Tokens
     - Storage
     - Corpus
     - Tokenizer
     - S3 URL
   * - ``v4_olmoe-mix-0924-dclm_llama``
     - 2,948,096,911
     - 4,341,627,197,578
     - 33TiB
     - `olmoe-mix-0924 <https://huggingface.co/datasets/allenai/olmoe-mix-0924>`_ (the DCLM part)
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_olmoe-mix-0924-dclm_llama <s3://infini-gram/index/v4_olmoe-mix-0924-dclm_llama>`_
   * - ``v4_olmoe-mix-0924-nodclm_llama``
     - 133,343,623
     - 233,848,504,469
     - 1.8TiB
     - `olmoe-mix-0924 <https://huggingface.co/datasets/allenai/olmoe-mix-0924>`_ (everything except DCLM)
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_olmoe-mix-0924-nodclm_llama <s3://infini-gram/index/v4_olmoe-mix-0924-nodclm_llama>`_
   * - ``v4_olmo-2-0325-32b-anneal-adapt_llama``
     - 82,461,386
     - 35,153,386,430
     - 268GiB
     - `dolmino-mix-1124 <https://huggingface.co/datasets/allenai/dolmino-mix-1124>`_ (except those already in pre-training); `SFT <https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225>`_; `DPO <https://huggingface.co/datasets/allenai/olmo-2-0325-32b-preference-mix>`_; `RLVR <https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_olmo-2-0325-32b-anneal-adapt_llama <s3://infini-gram/index/v4_olmo-2-0325-32b-anneal-adapt_llama>`_
   * - ``v4_olmo-2-1124-13b-anneal-adapt_llama``
     - 82,534,460
     - 35,273,912,238
     - 269GiB
     - `dolmino-mix-1124 <https://huggingface.co/datasets/allenai/dolmino-mix-1124>`_ (except those already in pre-training); `SFT <https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture>`_; `DPO <https://huggingface.co/datasets/allenai/olmo-2-1124-13b-preference-mix>`_; `RLVR <https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_olmo-2-1124-13b-anneal-adapt_llama <s3://infini-gram/index/v4_olmo-2-1124-13b-anneal-adapt_llama>`_
   * - ``v4_olmoe-0125-1b-7b-anneal-adapt_llama``
     - 82,513,183
     - 35,262,277,074
     - 269GiB
     - `dolmino-mix-1124 <https://huggingface.co/datasets/allenai/dolmino-mix-1124>`_ (except those already in pre-training); `SFT <https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture>`_; `DPO <https://huggingface.co/datasets/allenai/olmoe-0125-1b-7b-preference-mix>`_; `RLVR <https://huggingface.co/datasets/allenai/RLVR-GSM>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_olmoe-0125-1b-7b-anneal-adapt_llama <s3://infini-gram/index/v4_olmoe-0125-1b-7b-anneal-adapt_llama>`_
   * - ``v4_dolma-v1_7_llama``
     - 3,403,336,408
     - 2,604,642,372,173
     - 20TiB
     - `Dolma-v1.7 <https://huggingface.co/datasets/allenai/dolma>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_dolma-v1_7_llama <s3://infini-gram/index/v4_dolma-v1_7_llama>`_
   * - ``v4_rpj_llama_s4``
     - 931,361,530
     - 1,385,942,948,192
     - 8.9TiB
     - `RedPajama <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_rpj_llama_s4 <s3://infini-gram/index/v4_rpj_llama_s4>`_
   * - ``v4_piletrain_llama``
     - 210,607,728
     - 383,299,322,520
     - 2.5TiB
     - `Pile-train <https://huggingface.co/datasets/EleutherAI/pile>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_piletrain_llama <s3://infini-gram/index/v4_piletrain_llama>`_
   * - ``v4_c4train_llama``
     - 364,868,892
     - 198,079,554,945
     - 1.3TiB
     - `C4-train <https://huggingface.co/datasets/allenai/c4>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_c4train_llama <s3://infini-gram/index/v4_c4train_llama>`_
   * - ``v4_dolma-v1_6-sample_llama``
     - 13,095,416
     - 9,178,218,956
     - 62GiB
     - `Dolma-v1.6-sample <https://huggingface.co/datasets/allenai/dolma>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram/index/v4_dolma-v1_6-sample_llama <s3://infini-gram/index/v4_dolma-v1_6-sample_llama>`_
   * - ``v4_dolmasample_olmo``
     - 13,095,416
     - 8,039,098,124
     - 53GiB
     - `Dolma-v1.6-sample <https://huggingface.co/datasets/allenai/dolma>`_
     - `OLMo <https://huggingface.co/allenai/OLMo-7B>`_
     - `s3://infini-gram-lite/index/v4_dolmasample_olmo <s3://infini-gram-lite/index/v4_dolmasample_olmo>`_
   * - ``v4_pileval_llama``
     - 214,670
     - 393,769,120
     - 2.3GiB
     - `Pile-val <https://huggingface.co/datasets/EleutherAI/pile>`_
     - `Llama-2 <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_
     - `s3://infini-gram-lite/index/v4_pileval_llama <s3://infini-gram-lite/index/v4_pileval_llama>`_
   * - ``v4_pileval_gpt2``
     - 214,670
     - 383,326,404
     - 2.2GiB
     - `Pile-val <https://huggingface.co/datasets/EleutherAI/pile>`_
     - `GPT-2 <https://huggingface.co/gpt2>`_
     - `s3://infini-gram-lite/index/v4_pileval_gpt2 <s3://infini-gram-lite/index/v4_pileval_gpt2>`_

Smaller indexes are stored in the ``s3://infini-gram-lite`` bucket and can be downloaded for free and without an AWS account.
These indexes are ``v4_pileval_llama``, ``v4_pileval_gpt2``, and ``v4_dolmasample_olmo``.
To download, run command:

.. code-block:: bash

   aws s3 cp --no-sign-request --recursive {S3_URL} {LOCAL_INDEX_PATH}

Larger indexes are stored in the ``s3://infini-gram`` bucket.
To download these indexes, you need to pay for the data transfer fee (~$0.09 per GB according to `AWS S3 pricing <https://aws.amazon.com/s3/pricing/>`_).
Make sure you have correctly set up your AWS credentials before downloading these indexes.
These indexes are ``v4_rpj_llama_s4``, ``v4_piletrain_llama``, and ``v4_c4train_llama``.
To download, run command:

.. code-block:: bash

   aws s3 cp --request-payer requester --recursive {S3_URL} {LOCAL_INDEX_PATH}

.. _usage:

Query Types
-----------

Prior to submitting any type of queries, you need to instatiate the engine with the index you would like to query.
As an example, below we create an engine with the index for Pile-val (the validation set of Pile), which was created using the Llama-2 tokenizer:

.. code-block:: python

   >>> from infini_gram.engine import InfiniGramEngine
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False) # the tokenizer should match that of the index you load below
   >>> engine = InfiniGramEngine(index_dir='index/v4_pileval_llama', eos_token_id=tokenizer.eos_token_id) # please replace index_dir with the local directory where you store the index

1. Count an n-gram (or a CNF of multiple n-grams)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This query type counts the number of occurrences of an n-gram, or a CNF of multiple n-grams.

1.1 Count simple queries
^^^^^^^^^^^^^^^^^^^^^^^^

With simple queries, the engine counts the number of occurrences of a single n-gram in the corpus.

For example, to find out the number of occurrences of n-gram ``natural language processing`` in the Pile-val corpus:

.. code-block:: python

   >>> input_ids = tokenizer.encode('natural language processing')
   >>> input_ids
   [5613, 4086, 9068]

   >>> engine.count(input_ids=input_ids)
   {'count': 76, 'approx': False}

The ``approx`` field indicates whether the count is approximate.
For simple queries with a single n-gram term, this is always False (the count is always exact).
As you will see later, count for complex queries may be approximate.

If you submit an empty query, the engine returns the total number of tokens in the corpus:

.. code-block:: python

   >>> engine.count(input_ids=[])
   {'count': 393769120, 'approx': False}

1.2 Count CNF queries
^^^^^^^^^^^^^^^^^^^^^

You can make more complex queries by connecting multiple n-grams with the AND/OR operators, in the `CNF format <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`_, in which case the engine counts the number of times where this logical constraint is satisfied in the corpus.
A CNF query is a triply-nested list.
The top-level is a list of disjunctive clauses (which are eventually connected with the AND operator).
Each disjuctive clause is a list of n-gram terms (which are eventually connected with the OR operator).
And each n-gram term has the same format as ``input_ids`` above, i.e., a list of token ids.

.. code-block:: python

   # natural language processing OR artificial intelligence
   >>> cnf = [
   ...     [tokenizer.encode('natural language processing'), tokenizer.encode('artificial intelligence')]
   ... ]
   >>> cnf
   [[[5613, 4086, 9068], [23116, 21082]]]

   >>> engine.count_cnf(cnf=cnf)
   {'count': 499, 'approx': False}

.. code-block:: python

   # natural language processing AND deep learning
   >>> cnf = [
   ...     [tokenizer.encode('natural language processing')],
   ...     [tokenizer.encode('deep learning')],
   ... ]
   >>> cnf
   [[[5613, 4086, 9068]], [[6483, 6509]]]

   >>> engine.count_cnf(cnf=cnf)
   {'count': 6, 'approx': False}

.. code-block:: python

   # (natural language processing OR artificial intelligence) AND deep learning
   >>> cnf = [
   ...     [tokenizer.encode('natural language processing'), tokenizer.encode('artificial intelligence')],
   ...     [tokenizer.encode('deep learning')],
   ... ])
   >>> cnf
   [[[5613, 4086, 9068], [23116, 21082]], [[6483, 6509]]]

   >>> engine.count_cnf(cnf=cnf)
   {'count': 19, 'approx': False}

**Approximation:**
In case the CNF query contains AND operator(s), the engine needs to enumerate all occurrences of each clause and pick cases where they co-occur within reasonable distance.
This distance is controlled by the optional parameter ``max_diff_tokens``, which has a default value of 100.
Increasing this value and you may get more counts:

.. code-block:: python

   # natural language processing AND deep learning
   >>> engine.count_cnf(cnf=[
   ...     [tokenizer.encode('natural language processing')],
   ...     [tokenizer.encode('deep learning')],
   ... ], max_diff_tokens=1000)
   {'count': 14, 'approx': False}

However, if one of the clauses have a too high count, it will be inpractical to enumerate all its occurrences.
Our solution is to take a subsample of its occurrences when the count is higher than a threshold, controlled by the optional parameter ``max_clause_freq``, which has a default value of 50000.
When subsampling happens on any of the clauses, the count will be reported as approximate:

.. code-block:: python

   >>> engine.count(input_ids=tokenizer.encode('this'))
   {'count': 739845, 'approx': False}
   >>> engine.count(input_ids=tokenizer.encode('that'))
   {'count': 1866317, 'approx': False}

   # this AND that
   >>> engine.count_cnf(cnf=[[tokenizer.encode('this')], [tokenizer.encode('that')]])
   {'count': 982128, 'approx': True}

Increasing this value and you will get more accurate estimate of the count, and when this value is larger than (or equal to) the count of all clauses, the count becomes exact:

.. code-block:: python

   >>> engine.count_cnf(cnf=[[tokenizer.encode('this')], [tokenizer.encode('that')]], max_clause_freq=500000)
   {'count': 430527, 'approx': True}

   >>> engine.count_cnf(cnf=[[tokenizer.encode('this')], [tokenizer.encode('that')]], max_clause_freq=2000000)
   {'count': 480107, 'approx': False}

2. Prob of the last token
~~~~~~~~~~~~~~~~~~~~~~~~~

This query type computes the n-gram LM probability of a token conditioning on a preceding prompt.

For example, to compute ``P(processing | natural language)``:

.. code-block:: python

   >>> input_ids = tokenizer.encode('natural language processing')
   >>> input_ids
   [5613, 4086, 9068]

   >>> engine.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
   {'prompt_cnt': 257, 'cont_cnt': 76, 'prob': 0.29571984435797666}

In this case, ``prompt_cnt`` is the count of the 2-gram ``natural language``, ``cont_cnt`` is the count of the 3-gram ``natural language processing``, and ``prob`` is the division of these two counts.

If the prompt cannot be found in the corpus, the probability would be 0/0=NaN.
In these cases we report ``prob = -1.0`` to indicate an error:

.. code-block:: python

   >>> input_ids = tokenizer.encode('I love natural language processing')
   >>> input_ids
   [306, 5360, 5613, 4086, 9068]

   >>> engine.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
   {'prompt_cnt': 0, 'cont_cnt': 0, 'prob': -1.0}

3. Next-token distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

This query type computes the n-gram LM next-token distribution conditioning on a preceding prompt.

For example, this will return the token distribution following ``natural language``:

.. code-block:: python

   >>> input_ids = tokenizer.encode('natural language')
   >>> input_ids
   [5613, 4086]

   >>> engine.ntd(prompt_ids=input_ids)
   {'prompt_cnt': 257, 'result_by_token_id': {13: {'cont_cnt': 1, 'prob': 0.0038910505836575876}, 297: {'cont_cnt': 1, 'prob': 0.0038910505836575876}, ..., 30003: {'cont_cnt': 1, 'prob': 0.0038910505836575876}}, 'approx': False}

``result_by_token_id`` is a dict that maps token id to the probability of that token as a continuation of the prompt.

If the prompt cannot be found in the corpus, you will get an empty distribution:

.. code-block:: python

   >>> input_ids = tokenizer.encode('I love natural language processing')
   >>> input_ids
   [306, 5360, 5613, 4086, 9068]

   >>> engine.ntd(prompt_ids=input_ids[:-1])
   {'prompt_cnt': 0, 'result_by_token_id': {}, 'approx': False}

**Approximation:**
For each occurrence of the prompt, the engine needs to inspect the token appearing after it.
This is time-consuming and not feasible when ``prompt_cnt`` is large.
After this prompt count crosses a threshold, the engine needs to downsample the number of cases it inspects, and the resulting distribution will become approximate (which will be reflected in the ``approx`` field).
This threshold is controlled by the optional parameter ``max_support``, which has a default value of 1000.
For example, to get the unigram token distribution, you can query with an empty prompt and the result will be approximate:

.. code-block:: python

   >>> engine.ntd(prompt_ids=[])
   {'prompt_cnt': 393769120, 'result_by_token_id': {12: {'cont_cnt': 1013873, 'prob': 0.00257479052699714}, 13: {'cont_cnt': 14333030, 'prob': 0.03639957851443506}, ..., 30934: {'cont_cnt': 489584, 'prob': 0.0012433275621003496}}, 'approx': True}

4. ∞-gram prob
~~~~~~~~~~~~~~

This query type computes the ∞-gram LM probability of a token conditioning on a preceding prompt.
It uses the longest suffix of the prompt that has a non-zero count in the corpus.

.. code-block:: python

   >>> input_ids = tokenizer.encode('I love natural language processing')
   >>> input_ids
   [306, 5360, 5613, 4086, 9068]

   >>> engine.infgram_prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1])
   {'prompt_cnt': 257, 'cont_cnt': 76, 'prob': 0.29571984435797666, 'suffix_len': 2}

The field ``suffix_len`` indicates the number of tokens in the longest suffix of the prompt.
In this case, since ``[5613, 4086]`` can be found in the corpus, but ``[5360, 5613, 4086]`` cannot, the longest suffix is ``[5613, 4086]``, which has length 2.

5. ∞-gram next-token distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This query type computes the ∞-gram LM next-token distribution conditioning on a preceding prompt.

.. code-block:: python

   >>> input_ids = tokenizer.encode('I love natural language')
   >>> input_ids
   [306, 5360, 5613, 4086]

   >>> engine.infgram_ntd(prompt_ids=input_ids, max_support=10)
   {'prompt_cnt': 257, 'result_by_token_id': {297: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 470: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 508: {'cont_cnt': 1, 'prob': 0.0038910505836575876}, 8004: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 9068: {'cont_cnt': 96, 'prob': 0.3735408560311284}, 24481: {'cont_cnt': 32, 'prob': 0.1245136186770428}, 29889: {'cont_cnt': 32, 'prob': 0.1245136186770428}}, 'approx': True, 'suffix_len': 2}

6. Search documents
~~~~~~~~~~~~~~~~~~~

This query type returns documents in the corpus that match your query.

6.1 Search with simple queries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With simple queries, the engine can return documents containing a single n-gram.

First, you need to call ``find()`` to get information about where the matching documents are located.

.. code-block:: python

   >>> input_ids = tokenizer.encode('natural language processing')
   >>> input_ids
   [5613, 4086, 9068]

   >>> engine.find(input_ids=input_ids)
   {'cnt': 76, 'segment_by_shard': [(365362993, 365363069)]}

The returned ``segment_by_shard`` is a list of 2-tuples, each tuple represents a range of "ranks" in one of the shards of the index, and each rank can be traced back to a matched document in that shard.
The length of this list is equal to the total number of shards.
For example, if you want to retrieve the first matched document in shard 0, you can do

.. code-block:: python

   >>> engine.get_doc_by_rank(s=0, rank=365362993, max_disp_len=10)
   {'doc_ix': 47865, 'doc_len': 12932, 'disp_len': 10, 'metadata': '', 'token_ids': [363, 5164, 11976, 1316, 408, 5613, 4086, 9068, 518, 29992]}

The returned dict represents a document.
You can see that the query input_ids ``[5613, 4086, 9068]`` is present in this document.

The ranges are left-inclusive and right-exclusive.
To enumerate all documents, you can do something like

.. code-block:: python

   >>> find_result = engine.find(input_ids=input_ids)
   >>> for s, (start, end) in enumerate(find_result['segment_by_shard']):
   ...     for rank in range(start, end):
   ...         doc = engine.get_doc_by_rank(s=s, rank=rank)

6.2 Search with CNF queries
^^^^^^^^^^^^^^^^^^^^^^^^^^^

With CNF queries, the engine can return documents that satisfy the logical constraint specified in the CNF.

You need to first call ``find_cnf()`` which returns locations of matching documents in a different protocol:

.. code-block:: python

   # natural language processing AND deep learning
   >>> cnf = [
   ...     [tokenizer.encode('natural language processing')],
   ...     [tokenizer.encode('deep learning')],
   ... ]
   >>> cnf
   [[[5613, 4086, 9068]], [[6483, 6509]]]

   >>> engine.find_cnf(cnf=cnf)
   {'cnt': 6, 'approx': False, 'ptrs_by_shard': [[717544382, 377178100, 706194108, 25563710, 250933686, 706194476]]}

Note that the returned field is not ``segment_by_shard`` but rather ``ptrs_by_shard``.
For each shard, instead of having a range of "ranks", now we get a list of "pointers", and each pointer can be traced back to a matched document in that shard of the index.
The length of the outer list is equal to the total number of shards.
To get documents with these pointers, you need to call a different helper function:

.. code-block:: python

   # Get the document at pointer #2 in shard 0
   >>> engine.get_doc_by_ptr(s=0, ptr=706194108, max_disp_len=20)
   {'doc_ix': 191568, 'doc_len': 3171, 'disp_len': 20, 'metadata': '', 'token_ids': [29889, 450, 1034, 13364, 508, 367, 4340, 1304, 304, 7945, 6483, 6509, 2729, 5613, 4086, 9068, 9595, 1316, 408, 10013]}

You can see that both ``[5613, 4086, 9068]`` and ``[6483, 6509]`` are present in this document.
(For illustration I use a small ``max_disp_len``; since the default ``max_diff_tokens = 100``, you might need to increase ``max_disp_len`` to see the document covering all clauses in the CNF query.)

To enumerate all documents, you can do something like

.. code-block:: python

   >>> find_result = engine.find_cnf(cnf=cnf)
   >>> for s, ptrs in enumerate(find_result['ptrs_by_shard']):
   ...     for ptr in ptrs:
   ...         doc = engine.get_doc_by_ptr(s=s, ptr=ptr)
