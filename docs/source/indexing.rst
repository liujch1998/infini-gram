Indexing Custom Datasets
========================

If the dataset you'd like to query does not have a pre-built index, you can build the index yourself.

For example, to index the training set of Pile using the Llama-2 tokenizer, you can run:

.. code-block:: bash

   python -m infini_gram.indexing \
       --data_dir /dir/of/pile/train \
       --save_dir /dir/to/save/the/index \
       --tokenizer llama \
       --cpus 64 --mem 512 \
       --shards 2 --add_metadata \
       --ulimit 1048576

This assumes that your system has 64 CPU cores and 512 GiB of RAM available to the program, and will shard the index in 2 ways.

**Estimate the number of shards:**
Before we can build the index, we need to estimate the number of shards $S$ to use.
There are two considerations:

1. Each shard of tokenized corpus must have no more than $2^{39} \approx 500\text{B}$ tokens.
2. Each shard of tokenized corpus must fit in the RAM. If your machine has $M$ bytes of RAM, then the shard should be no bigger than $0.8 \cdot M$ bytes (to leave some buffer), and thus each shard should have no more than $0.4 \cdot M$ tokens.

**Estimate the amount of disk space required:**
Before building the index, you might want to check that you have enough disk space.
Aside from the original corpus, the index will consume roughly 7 bytes per token, so for a corpus of $N$ tokens, you will need $7 N$ bytes of disk space.
If you include document metadata in the index (``--add_metadata``), you will need a bit more space.
In addition, we also need some disk space to store temporary files, which is roughly $12 N / S$ bytes.

Prior to running this command, make sure you have the dataset files stored under ``--data_dir`` or its subdirectories.
Each file should be a JSONL file (ending in ``.jsonl``) or its compressed format (ending in ``.gz`` or ``.zst``), and each line should be a dict with a field ``text`` and optionally some other fields (treated as metadata).
If you have the data files in a different format, feel free to head to the installation path of this package, open ``indexing.py``, and edit the ``load_file()`` function.

The final index files will be stored under ``--save_dir``.
During indexing, some temporary files will be created under the same directory, and they will be automatically removed when indexing is done.
If you would like these temporary files to utilize a different storage location, you may specify this with ``--temp_dir``.

The available tokenizers are ``{gpt2, llama, olmo}``.
If you would like to use a different tokenizer, feel free to head to the installation path of this package, open ``indexing.py``, and add your tokenizer to the ``tokenize()`` function.
The vocab size of the tokenizer must be no larger than 65535.

The ulimit argument raises the max number of open files allowed by the system.
Some system does not allow raising this limit too much, and in such case you can try specifying a smaller value.

Use ``python -m infini_gram.indexing -h`` for additional help on using this program.
