# 📖 infini-gram (inference)

This is the source code of the inference part of the [infini-gram search engine](https://infini-gram.io/).

**Target audience:**
If you would like to host a service similar to the one that powers the infini-gram [demo](https://huggingface.co/spaces/liujch1998/infini-gram) or [API](https://infini-gram.io/api_doc), then this repository is for you.
You must have already obtained the index on some datasets by either downloading from our released [pre-built indexes]() or building your own using our [training code](https://github.com/liujch1998/infini-gram/tree/master/train).

## Overview

Running this code will spin up a Flask server and two inference engines in your machine.
This server will listen to incoming HTTP requests and respond with the query results.
Each query will be processed by either the C++ engine (faster, default) or the Python engine (slower), depending on the query's specification.
These two engines should produce identical results (up to some difference in the random number generators), while the C++ engine can use multi-threading and is thus faster.

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
git clone git@github.com:liujch1998/infini-gram.git
cd infini-gram/inference
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

For reference, we use an instance of type `m7g.2xlarge` (8 CPUs, 32GiB RAM) for serving the infini-gram demo and API endpoint, which costs $0.32/hr.
We choose the Ubuntu 64-bit ARM image, and a 32GiB gp2 volume.
It should be OK to use a smaller instance, but you may get higher latency when querying large indexes due to insufficient parallelization.

### Step 3: SSH into the instance and do the regular setup (refer to the Setup section above)

### Step 4: Attach the volume(s) to the instance

From the AWS console, select the volume(s) that stores the index, and attach it to the instance.

### Step 5: Mount the volume(s) in the instance

(If the instance is rebooted, you will need to do this step again.)

In the instance, use `lsblk` to locate the attached volume(s) (the first one is likely to be `/dev/nvme1n1`).
Mount this volume by running something like:
```bash
sudo mkdir /data
sudo mount /dev/nvme1n1 /data
sudo chown ubuntu:ubuntu /data
```
If you're serving multiple indexes, you may want to mount each volume to a different directory.

### Step 6: Start the Flask server (refer to the Run section above)

### Optional: Associate a static IP address to the instance
