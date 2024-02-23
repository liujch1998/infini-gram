# 📖 infini-gram (inference)

This is the source code of the inference part of the [infini-gram search engine](https://infini-gram.io/).

**Target audience:**
If you would like to host a service similar to the one that powers the infini-gram [demo](https://huggingface.co/spaces/liujch1998/infini-gram) or [API](https://infini-gram.io/api_doc), then this repository is for you.
You must have already obtained the index on some datasets by either downloading from our released [pre-built indexes]() or building your own using our [training code](https://github.com/liujch1998/infini-gram-t).

## Overview

Running this code will spin up a Flask server and two inference engines in your machine.
This server will listen to incoming HTTP requests and respond with the query results.
Each query will be processed by either the C++ engine (faster, default) or the Python engine (slower), depending on the query's specification.
These two engines should produce identical results (up to some difference in the random number generators), while the C++ engine can use multi-threading and is thus faster.

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

[TODO: Add download links]

## System Requirements

Hosting the inference service requires no GPU and very minimal CPU and RAM.
All indexes will be accessed as memory-mapped files.

The biggest requirement is disk space.
The infini-gram index is roughly 7 bytes per token.
So if you're serving a corpus of 1 billion tokens, you will need 7GB of disk space; if you're serving a corpus of 1 trillion tokens, you will need 7TB of disk space.

If your machine does not have enough disk space, you may refer to our sample guide below, [Serving Massive Corpora with AWS]().

## Setup

1. Install g++.
```bash
sudo apt-get update
sudo apt-get install g++
```

2. Install conda (follow the [official instructions](https://docs.anaconda.com/free/miniconda/miniconda-install/)).

3. Clone this repository.
```bash
git clone git@github.com:liujch1998/infini-gram-i.git
cd infini-gram-i
```

4. Create a conda environment.
```bash
conda env create -f environment.yml
conda activate infini-gram-i
```

5. Set environment variable `HF_TOKEN_DOWNLOAD` to be your HF token with read access. This is necessary for downloading gated tokenizers such as Llama and OLMo. It is recommended to also add this line to your `~/.bashrc`.
```bash
export HF_TOKEN_DOWNLOAD=[YOUR_HF_READ_TOKEN]
```

## Run

1. Register the index(es) you would like to serve, by editing the `indexes.json` file. Replace the example entries with those that suit your own index(es).
For each index, create a dict that contains the following fields:

| Field | Description |
| --- | --- |
| `corpus` | A convenient name for the index. |
| `dir` | The directory where the index is stored. |
| `tokenizer` | The name of the tokenizer used to build the index. Options: [`gpt2`, `llama`, `olmo`]. |
| `eos_token_id` | The EOS token ID of the tokenizer. |

There are also a few optional fields, `*_prefetch_depth`.
These are parameters for controlling the memory prefetching behavior of the C++ engine for optimal performance.
If you don't know what they do, you can just leave out these fields.

2. Start the Flask server.
```bash
python flask_app.py
```

3. Send a test query to the server!
```bash
curl -X POST -H "Content-Type: application/json" -d '{"corpus": "[YOUR_CORPUS_NAME]", "query_type": "count", "query": "Hello, world!"}' http://localhost:5000/
```

### Logging

The service will log into two files, `~/flask_{MODE}.log` and `~/cpp_engine_{MODE}.log`.
If you encounter any issues, please refer to these logs for more information.

## Serving Massive Corpora with AWS

If you want to host an index bigger than your machine's disk space, you may consider using cloud services.
Here I will go through my recipe for setting things up on AWS EC2.

### Step 1: Create a volume that stores the index

Create a volume with type = `gp3`, and with maximum IOPS (`16000`) and throughput (`1000` MiB/s).
After all index data is written, you may reduce to minumum IOPS (`3000`) and throughput (`125` MiB/s) to save cost.
Note that in some of the cheapest regions, a `gp3` volume costs $0.08 per GiB-month.

**Volume size:** When creating volumes, please make sure to reserve 10% more space than the index size, since the actual disk space seems to be less than what you reserve.

If you'd like to use one of our released pre-built indexes, you can restore a volume from the corresponding snapshot we share.

### Step 2: Launch an instance

### Step 3: SSH into the instance and do the regular setup (refer to the Setup section above)

### Step 4: Attach the volume(s) to the instance

### Step 5: Mount the volume(s) in the instance

### Step 6: Start the Flask server

### Optional: Associate a static IP address to the instance

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
