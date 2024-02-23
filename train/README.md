# infini-gram

This repo hosts the code for building the Suffix Array based index in the infini-gram paper: [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)

You may use this code to build an index of any text corpus, which enables very fast n-gram/∞-gram queries.

* Paper: <https://arxiv.org/abs/2401.17377>
* Code: <https://github.com/liujch1998/infini-gram>
* Demo: <https://hf.co/spaces/liujch1998/infini-gram>
* API: (coming soon)

## Pre-built indexes

Our demo hosts the following infini-gram indexes:

| Name | Tokens | Storage Size | Corpus | Tokenizer | Unlimited Len |
| --- | --- | --- | --- | --- | --- |
| `v3_pileval` | 383,755,744 | 2.2GiB | Pile-val | GPT-2 | yes |
| `v3_pileval_llama` | 394,413,130 | 2.3GiB | Pile-val | LLaMA-2 | yes |
| `v3_rpj_llama_c4` | 1,385,642,556,650 | 8.9TiB | RedPajama | LLaMA-2 | no |
| `v4_piletrain_llama` | 383,299,322,520 | 2.4TiB | Pile-train | LLaMA-2 | yes |
| `v4_dolmasample_olmo` | 8,039,098,124 | 53GiB | Dolma-sample | OLMo | yes |

Each index is stored as an AWS EBS volume, with volume name `infinigram-i-<name>`.

If you would like to build the index for other corpora, please follow the instructions below.

## Setup

1. Create a conda environment and activate it:
```bash
conda env create -f environment.yml
conda activate infini-gram
```

2. Install Rust and compile the Rust part of the code:
```bash
sudo apt-get install gcc
source "$HOME/.cargo/env"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo build --release
```

## Building the index

Build the index by running the following command:
```bash
python main.py \
    --data_dir [/path/to/data/dir] \
    --save_dir [/path/to/save/dir] \
    --tokenizer gpt2 \
    --cpus 64 --mem 256
```
* `--data_dir` is the directory that contains the text corpus. The corpus may be stored as several files, and they may be located in subdirectories. Each file must be in `.jsonl`, `.gz` or `.zst` format. Each line should be a JSON object with a `text` field.
* `--save_dir` is the directory to save the index.
* `--tokenizer` is the tokenizer to use. Currently, we support `gpt2` and `llama`.
* `--cpus` is the number of CPUs to use. It should be no more than the number of CPUs on the machine, and typically should be a power of 2.
* `--mem` is the amount of memory to use, in GB. It should be no more than the amount of memory on the machine.

If your corpus is very large, you may need to shard the index by specifying `--shards`. To build the index, the tokenized version of each shard must fit in memory. Use this as a guideline when deciding the number of shards.

If you'd like to speed up the indexing process, you may distribute the work across multiple machines. Run the same command on each machine with the same `--workers W` and a different `--worker_id [0, W)`. Note that the number of workers $W$ must be a divisor of the number of shards $S$. After all workers are finished, move all `tokenized.*` and `table.*` files to the same directory.

## Structure of the index

Each index is a directory with the following files:
* `tokenized.{s}` -- The tokenized version of the corpus shard. This file is a binary file that is $2t$ bytes long, where $t$ is the total number of tokens. Each contiguous two bytes represent a token id.
* `table.{s}` -- The suffix array on the byte table `tokenized.{s}`. This file is a binary file that is $k \cdot t$ bytes long, where $k$ is the number of bytes in each pointer: $k = \lceil \frac{1}{8}\log_2{2t} \rceil$. For the Pile-val dataset, $k = 4$. Each contiguous $k$ bytes represent a pointer to the `tokenized.{s}` byte table.

**Sharding**: In case the index is sharded into $S$-ways, there will be $S$ tokenized files `tokenized.0` ... `tokenized.{S-1}`, and $S$ table files `table.0` ... `table.{S-1}`. Each pair of `tokenized.{s}` and `table.{s}` is self-contained.

### Endianness

All binary files in our datastores use **little-endian** byte order.
Please make sure that `sys.byteorder == 'little'` in your inference code.

One consequence of using little-endian is that the suffix array is not a dictionary order of the token id sequences. For example, what follows token id `0x0000` is token id `0x0100` (which is "00 01" in little-endian byte array).
This does not matter because we are doing exact match search.

## Inference with the index

The algorithms for querying the index are described in our paper. You may refer to the implementation in our [HF Spaces demo](https://huggingface.co/spaces/liujch1998/infini-gram).

## Acknowledgements

Our suffix array implementation is adapted from Google's [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)

## Citation

If you find this repo useful, please consider citing our paper:
```bibtex
@article{Liu2024InfiniGram,
  title={Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens},
  author={Liu, Jiacheng and Min, Sewon and Zettlemoyer, Luke and Choi, Yejin and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2401.17377},
  year={2024}
}
```
