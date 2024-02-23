# 📖 infini-gram (train)

This is the source code of the training part of the [infini-gram search engine](https://infini-gram.io/).

**Target audience:**
If you would like to build an infini-gram index for a text corpus of your choice, which can be subsequently served by the [inference code](https://github.com/liujch1998/infini-gram/tree/master/inference).

## System Requirements

Building the index requires no GPU, but may require a significant amount of CPU, RAM, and disk space, depending on the size of your corpus.
For reference, we built all our indexes on an AWS EC2 instance of type `r6in.32xlarge` (128 CPUs, 1TiB RAM), and the indexing of RedPajama used 19TiB of disk space.

If you want to index a very big corpus and your machine is too small, you may refer to our sample guide below, [Indexing Massive Corpora with AWS]().

## Setup

1. Install conda (follow the [official instructions](https://docs.anaconda.com/free/miniconda/miniconda-install/)).

2. Clone this repository.
```bash
git clone git@github.com:liujch1998/infini-gram.git
cd infini-gram/train
```

3. Create a conda environment.
```bash
conda env create -f environment.yml
conda activate infini-gram-t
```

4. Install Rust and compile the Rust part of the code:
```bash
sudo apt-get update
sudo apt-get install gcc
source "$HOME/.cargo/env"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo build --release
```

## Building the Index

**Estimate the number of shards:**
Before we can build the index, we need to estimate the number of shards $S$ to use.
(See README in the root directory for more information on sharding.)
There are two considerations:
1. Each shard of tokenized corpus must have no more than $2^{39} \approx 500\text{B}$ tokens.
2. Each shard of tokenized corpus must fit in the RAM. If your machine has $M$ bytes of RAM, then the shard should be no bigger than $0.8 \cdot M$ bytes (to leave some buffer), and thus each shard should have no more than $0.4 \cdot M$ tokens.

**Estimate the amount of disk space required:**
Before buildin gthe index, you might want to check that you have enough disk space.
Aside from the original corpus, the index will consume roughly 7 bytes per token, plus some metadata, so for a corpus of $N$ tokens, let's say you will need $8N$ bytes of disk space.
In addition, we also need some swap disk space, which is roughly $12 N / S$ bytes.

Build the index by running the following command:
```bash
python main.py \
    --data_dir [/path/to/data/dir] \
    --temp_dir [/path/to/temp/dir] \
    --save_dir [/path/to/save/dir] \
    --tokenizer [TOKENIZER] \
    --cpus [CPUS] --mem [MEM]
```
* `--data_dir` is the directory that contains the text corpus. The corpus may be stored as several files, and they may be located in subdirectories. Each file must be in `.jsonl`, `.gz` or `.zst` format. Each line should be a JSON object with a `text` field.
* `--temp_dir` is the location of the swap space. It will be cleaned up after the indexing process.
* `--save_dir` is the directory to save the index.
* `--tokenizer` is the tokenizer to use. Currently, we support `gpt2`, `llama`, and `olmo`.
* `--cpus` is the number of CPUs to use. It should be no more than the number of CPUs on your machine, and typically should be a power of 2.
* `--mem` is the amount of RAM available to use, in GiB. It should be no more than 80% of the total RAM on your machine.

If you'd like to speed up the indexing process, you may distribute the work across multiple machines. Run the same command on each machine with the same `--workers W` and a different `--worker_id [0, W)`. Note that the number of workers $W$ must be a divisor of the number of shards $S$. After all workers are finished, move all index files to the same directory.

## Indexing Massive Corpora with AWS

If you want to index a very big corpus and your machine is too small, you may consider using cloud services.
Here I will go through my recipe for setting things up on AWS EC2.

### Step 1: Create volumes

Create three volumes: the first one to store the original corpus, the second one to be the swap space (should have $12 N / S$ bytes), and the third one to store the index (should have $8 N$ bytes).
Create all volumes with type = `gp3`, and with maximum IOPS (`16000`) and throughput (`1000` MiB/s).
After all index data is written, you may delete the swap volume, and for others reduce to minumum IOPS (`3000`) and throughput (`125` MiB/s) to save cost.
Note that in some of the cheapest regions, a `gp3` volume costs $0.08 per GiB-month.

**Volume size:** When creating volumes, please make sure to reserve 10% more space than the index size, since the actual disk space seems to be less than what you reserve.

### Step 2: Launch an instance

For reference, we use an instance of type `r6in.32xlarge` (128 CPUs, 1TiB RAM), which costs $11.16/hr.
We choose the Ubuntu 64-bit x86 image, and a 32GiB gp2 volume.

### Step 3: SSH into the instance and do the regular setup (refer to the Setup section above)

### Step 4: Attach all three volumes to the instance

From the AWS console, select the volumes and attach them to the instance.

### Step 5: Format and mount all three volumes in the instance

In the instance, use `lsblk` to locate the attached volumes (the first one is likely to be `/dev/nvme1n1`).
Format and mount the volumes by running something like:
```bash
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir /data_c
sudo mount /dev/nvme1n1 /data_c
sudo chown ubuntu:ubuntu /data_c

sudo mkfs -t ext4 /dev/nvme2n1
sudo mkdir /data_t
sudo mount /dev/nvme2n1 /data_t
sudo chown ubuntu:ubuntu /data_t

sudo mkfs -t ext4 /dev/nvme3n1
sudo mkdir /data_i
sudo mount /dev/nvme3n1 /data_i
sudo chown ubuntu:ubuntu /data_i
```

### Step 6: Download the corpus into /data_c

### Step 7: Start the indexing process (refer to the Building the Index section above)

### Step 8: Umount all three volumes in the instance
```bash
sudo umount /data_c
sudo umount /data_t
sudo umount /data_i
```

### Step 9: Detach all three volumes from the instance

From the AWS console, select the volumes and detach them from the instance.

### Step 10: Delete the swap volume; Reduce IOPS and throughput for the other two volumes

## Acknowledgements

Our suffix array implementation is adapted from Google's [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)
