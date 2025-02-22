# 📖 infini-gram

This repo hosts the code of the [infini-gram search engine](https://infini-gram.io/), which is described in this paper: [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377).

To learn more about infini-gram:
* Paper: <https://arxiv.org/abs/2401.17377>
* Project Home: <https://infini-gram.io>
* Web Interface: <https://infini-gram.io/demo>
* API Endpoint: <https://infini-gram.io/api_doc>
* Python Package: <https://pypi.org/project/infini-gram>
* Code: <https://github.com/liujch1998/infini-gram>

## Overview

Infini-gram is an engine that processes n-gram queries on massive text corpora with extremely low latency.
It can count arbitrarily long strings in trillion-token corpora and retrieve their containing documents, all within dozens of milliseconds.

Infini-gram is powered by an index based on suffix arrays.
This repo contains everything you might need to build an infini-gram index for the corpus of your choice, and to perform inference on this index.

Depending on your use case and needs, there are several options for you:
1. If you'd like to explore infini-gram or query in small volume, our [Web Interface](https://hf.co/spaces/liujch1998/infini-gram) is the best place to start.
2. If you'd like to programmatically query infini-gram in moderate to large volume, we offer a free and easy-to-use API endpoint, please check out the [API documentation](https://infini-gram.io/api_doc).
3. If you'd like to serve the inference engine yourself on an existing index, or build index on a new dataset, you can do so via our [Python Package](https://pypi.org/project/infini-gram).
4. If you'd like to customize the indexing or the inference engine, or you're simply curious about how things work, you're welcome to dive into this repo and mess with it!

## Structure of the Index

Each index is a directory with several regular files.

The first notion to understand is **sharding**.
For sake of efficiency, we need to limit the number of tokens in each index to 500B tokens.
If the corpus has more than 500B tokens, we break it into multiple shards, and index each shard separately.
Fortunately, the infini-gram index is additive, and thus we can recover the full index by taking the union of all shards.

If we break into $S$ shards, and we number them from $0$ to $S-1$, then the index for shard $s$ consists of the following files:

* `tokenized.{s}` -- The byte array for the tokenized version of the corpus shard. This is a binary file that is $2t$ bytes long, where $t$ is the number of tokens in this shard. Each contiguous two bytes represent a token id.
* `table.{s}` -- The suffix array on the byte array `tokenized.{s}`. This is a binary file that is $k \cdot t$ bytes long, where $k$ is the number of bytes in each pointer: $k = \lceil \frac{1}{8}\log_2{2t} \rceil$. For shards with no more than 500B tokens, $k = 5$. Each contiguous $k$ bytes represent a pointer into the `tokenized.{s}` byte array.
* `offset.{s}` -- The document offsets. This is a binary file that is $8d$ bytes long, where $d$ is the number of documents in this shard. Each contiguous 8 bytes is a pointer into the `tokenized.{s}` byte array which is the beginning of a document.
* [Optional] `metadata.{s}` -- The metadata for the documents. This is a text file with $d$ lines, where each line is a comma-separated list of three fields: the document ID (if any), the relative path of the containing file in the original corpus, and the line number in that file (0-indexed).
* [Optional] `metaoff.{s}` -- The offsets of the metadata. This is a binary file that is $8d$ bytes long, where $d$ is the number of documents in this shard. Each contiguous 8 bytes is a pointer into the `metadata.{s}` file which is the beginning of a document's metadata.

### Endianness

All binary files in our indexes use **little-endian** byte order.
Please make sure to check that `sys.byteorder == 'little'` in your training and inference code.

One consequence of using little-endian is that the suffix array is not a dictionary order of the token id sequences. For example, what follows token id `0x0000` is token id `0x0100` (which is `00 01` in little-endian byte array).
This does not matter because we are doing exact match search.

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
