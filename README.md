# 📖 infini-gram

This repo hosts the code of the [infini-gram search engine](https://infini-gram.io/), which is described in this paper: [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377).

* Paper: <https://arxiv.org/abs/2401.17377>
* Project page: <https://infini-gram.io>
* API: <https://infini-gram.io/api_doc>
* Demo: <https://hf.co/spaces/liujch1998/infini-gram>
* Code: <https://github.com/liujch1998/infini-gram>

## Overview

Infini-gram is an engine that processes n-gram queries on massive text corpora with extremely low latency.
It can count arbitrarily long strings in trillion-token corpora and retrieve their containing documents, all within dozens of milliseconds.

Infini-gram is powered by an index based on suffix arrays.
This repo contains everything you might need to build an infini-gram index for the corpus of your choice, and to perform inference on this index.

Depending on your use case and needs, there are several options for you:
1. If you'd like to explore infini-gram or query in small volume, our [HF demo](https://hf.co/spaces/liujch1998/infini-gram) is the best place to start.
2. If you'd like to programmatically query infini-gram in moderate to large volume, we offer a free and easy-to-use API endpoint, please check out the [API documentation](https://infini-gram.io/api_doc).
3. If you are querying one of the indexes supported in the demo or the API, but need to customize the inference engine (e.g., perhaps you want to retrieve all documents containing a query string, rather than a random set of 10 documents), please head to the `inference/` folder. The README there will guide you to acquire a pre-built index and set up the inference service.
4. If you'd like to build an infini-gram index for a text corpus of your choice, please head to the `train/` folder and follow the README there. After you have built the index, you may serve it using the inference code.

## Pre-built Indexes

Here are the indexes we serve in the demo and the API endpoint, and some of their statistics:

| Name | Documents | Tokens | Storage | Corpus | Tokenizer |
| --- | ---: | ---: | ---: | --- | --- |
| `v4_rpj_llama_s4` | 931,361,530 | 1,385,942,948,192 | 8.9TiB | RedPajama | Llama |
| `v4_piletrain_llama` | 210,607,728 | 383,299,322,520 | 2.5TiB | Pile-train | Llama |
| `v4_c4train_llama` | 364,868,892 | 198,079,554,945 | 1.3TiB | C4-train | Llama |
| `v4_pileval_llama` | 214,670 | 393,769,120 | 2.3GiB | Pile-val | Llama |
| `v4_pileval_gpt2` | 214,670 | 383,326,404 | 2.2GiB | Pile-val | GPT-2 |
| `v4_dolmasample_olmo` | 13,095,416 | 8,039,098,124 | 53GiB | Dolma-sample | OLMo |

You may download / restore the indexes here:
* `v4_rpj_llama_s4`: <https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#SnapshotDetails:snapshotId=snap-0fb56adcc3c1a1ca7>
* `v4_piletrain_llama`: <https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#SnapshotDetails:snapshotId=snap-0d3937dcbb635c119>
* `v4_c4train_llama`: <https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#SnapshotDetails:snapshotId=snap-0dabdf10bf6930394>

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
