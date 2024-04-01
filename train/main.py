import argparse
import glob
import gzip
import json
import multiprocessing as mp
import numpy as np
import os
import resource
import shutil
import struct
import sys
import time
from tqdm import tqdm
import transformers
import zstandard as zstd

assert sys.byteorder == 'little'
HACK = 100000
transformers.utils.logging.set_verbosity(40) # suppress warnings
resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--temp_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=True, choices=['gpt2', 'llama', 'olmo'])
parser.add_argument('--doc_sep', type=bytes, default=b'\xff\xff')
parser.add_argument('--batch_size', type=int, default=65536, help='Typically should be a power of 2.')
parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Typically should be a power of 2.')
parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GB available to the program.')
parser.add_argument('--shards', type=int, default=1, help='Typically should be a power of 2.')
parser.add_argument('--workers', type=int, default=1, help='Must be a divisor of shards. Typically should be a power of 2.')
parser.add_argument('--worker_id', type=int, default=0)
args = parser.parse_args()
args.data_dir = args.data_dir.rstrip('/')
args.temp_dir = args.temp_dir.rstrip('/')
args.save_dir = args.save_dir.rstrip('/')

assert args.batch_size > 0
assert args.cpus > 0
assert args.shards > 0
assert args.workers > 0
assert 0 <= args.worker_id < args.workers
assert args.shards % args.workers == 0

assert os.path.exists(args.data_dir)
os.makedirs(args.temp_dir, exist_ok=True)
os.makedirs(args.save_dir, exist_ok=True)

# ======== Step 1 (tokenize) ======== #

ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]
od_paths = [os.path.join(args.save_dir, f'offset.{i}') for i in range(args.worker_id, args.shards, args.workers)]
mt_paths = [os.path.join(args.save_dir, f'metadata.{i}') for i in range(args.worker_id, args.shards, args.workers)]
om_paths = [os.path.join(args.save_dir, f'metaoff.{i}') for i in range(args.worker_id, args.shards, args.workers)]
if all([os.path.exists(ds_path) and os.access(ds_path, os.R_OK) and not os.access(ds_path, os.W_OK) and not os.access(ds_path, os.X_OK) for ds_path in ds_paths]) \
    and all([os.path.exists(od_path) and os.access(od_path, os.R_OK) and not os.access(od_path, os.W_OK) and not os.access(od_path, os.X_OK) for od_path in od_paths]) \
    and all([os.path.exists(mt_path) and os.access(mt_path, os.R_OK) and not os.access(mt_path, os.W_OK) and not os.access(mt_path, os.X_OK) for mt_path in mt_paths]) \
    and all([os.path.exists(om_path) and os.access(om_path, os.R_OK) and not os.access(om_path, os.W_OK) and not os.access(om_path, os.X_OK) for om_path in om_paths]):
    print('Step 1 (tokenize): Skipped. All tokenized files already exist.')
else:
    print('Step 1 (tokenize): Starting ...')

    if args.tokenizer == 'gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama':
        tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN_DOWNLOAD'), use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    elif args.tokenizer == 'olmo':
        from dolma.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained('allenai/gpt-neox-olmo-dolma-v1_5', bos_token_id=None, eos_token_id=None, pad_token_id=1, segment_before_tokenization=True)
    else:
        raise ValueError(f'Unknown tokenizer: {args.tokenizer}')

    def load_file(path):
        if path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        elif path.endswith('.zst'):
            with open(path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    decompressed_data = reader.read().decode('utf-8')
                lines = decompressed_data.split('\n')
                if lines[-1] == '':
                    lines = lines[:-1]
        elif path.endswith('.jsonl'):
            with open(path, encoding='utf-8') as f:
                lines = f.readlines()
        else:
            raise ValueError(f'Unknown file type: {path}')
        return lines

    def tok(line):
        js = json.loads(line.strip('\n'))
        tok_text = tokenizer.encode(js['text'])
        byte_arr = np.array(tok_text, dtype=np.uint16).view(np.uint8).tobytes()
        ID = js["id"] if "id" in js else ""
        return byte_arr, ID
    def extract_text(line):
        js = json.loads(line.strip('\n'))
        text = js['text']
        ID = js["id"] if "id" in js else ""
        return text, ID
    def convert_to_bytes(tok_text):
        return np.array(tok_text, dtype=np.uint16).view(np.uint8).tobytes()

    data_paths = glob.glob(f'{args.data_dir}/**/*.json*', recursive=True)
    data_paths = list(sorted(data_paths))
    ds_fouts = [open(ds_path, 'wb') for ds_path in ds_paths]
    od_fouts = [open(od_path, 'wb') for od_path in od_paths]
    mt_fouts = [open(mt_path, 'w') for mt_path in mt_paths]
    om_fouts = [open(om_path, 'wb') for om_path in om_paths]
    with mp.get_context('fork').Pool(args.cpus) as p:
        ods = [0 for _ in od_fouts]
        oms = [0 for _ in om_fouts]
        for data_path in tqdm(data_paths):
            rel_data_path = data_path[len(args.data_dir)+1:]
            lines = load_file(data_path)
            for offset in tqdm(range(0, len(lines), args.workers*args.batch_size), total=len(range(0, len(lines), args.workers*args.batch_size))):
                batch_lines = lines[(offset+args.worker_id):(offset+args.workers*args.batch_size):args.workers]
                results = p.map(tok, batch_lines)
                for i, (byte_arr, ID) in enumerate(results):
                    content = args.doc_sep + byte_arr
                    j = i % (args.shards // args.workers)
                    ds_fouts[j].write(content)
                    od_fouts[j].write(np.array([ods[j]], dtype=np.uint64).view(np.uint8).tobytes())
                    ods[j] += len(content)
                    linenum = (offset + args.worker_id) + args.workers * i
                    metadata = f'{ID},{rel_data_path},{linenum}\n'
                    mt_fouts[j].write(metadata)
                    om_fouts[j].write(np.array([oms[j]], dtype=np.uint64).view(np.uint8).tobytes())
                    oms[j] += len(metadata)
            del lines

    for ds_fout in ds_fouts:
        ds_fout.close()
    for od_fout in od_fouts:
        od_fout.close()
    for mt_fout in mt_fouts:
        mt_fout.close()
    for om_fout in om_fouts:
        om_fout.close()

    for ds_path in ds_paths:
        os.chmod(ds_path, 0o444)
    for od_path in od_paths:
        os.chmod(od_path, 0o444)
    for mt_path in mt_paths:
        os.chmod(mt_path, 0o444)
    for om_path in om_paths:
        os.chmod(om_path, 0o444)

# ======== Step 2 (build suffix array) ======== #

print('Step 2 (build suffix array): starting ...')

for t, ds_path in enumerate(ds_paths):
    sa_path = ds_path.replace('tokenized', 'table')
    if os.path.exists(sa_path) and os.access(sa_path, os.R_OK) and not os.access(sa_path, os.W_OK) and not os.access(sa_path, os.X_OK):
        print(f'Shard {t} / {len(ds_paths)}: Skipped. Table already exists.')
        continue

    start_time_all = time.time()

    # -------- Step 2.1 (make-part) -------- #

    print(f'Shard {t} / {len(ds_paths)}: make-part ...')
    start_time = time.time()

    tok_size = os.path.getsize(ds_path)
    mem_bytes = args.mem * 1024**3
    num_job_batches = 1
    while num_job_batches * (mem_bytes // 8) < tok_size:
        num_job_batches *= 2
    parallel_jobs = args.cpus
    total_jobs = num_job_batches * parallel_jobs
    print(f'Using {num_job_batches} batches of {parallel_jobs} jobs each, for a total of {total_jobs} jobs.')

    S = tok_size // total_jobs
    # Make sure that parts contain whole tokens (2 bytes)
    if S % 2 == 1:
        S += 1

    parts_dir = os.path.join(args.temp_dir, f'parts-{args.worker_id}')
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)

    ranges, files = [], []
    for batch_start in tqdm(list(range(0, total_jobs, parallel_jobs))):
        batch_end = min(batch_start+parallel_jobs, total_jobs)
        batch_ranges, batch_files = [], []
        for i in range(batch_start, batch_end):
            s, e = i*S, min((i+1)*S+HACK, tok_size)
            batch_ranges.append((s, e))
            batch_files.append(os.path.join(parts_dir, f'{s}-{e}'))
        ranges += batch_ranges
        files += batch_files
        wait = []
        for (s, e) in batch_ranges:
            cmd = f'./target/release/suffix_array make-part --data-file {ds_path} --parts-dir {parts_dir} --start-byte {s} --end-byte {e}'
            wait.append(os.popen(cmd))
        [x.read() for x in wait]

    end_time = time.time()
    print(f'Shard {t} / {len(ds_paths)}: make-part done. Took {end_time-start_time:.2f} seconds')

    # -------- Step 2.2 (merge) -------- #

    print(f'Shard {t} / {len(ds_paths)}: merge ...')
    start_time = time.time()

    merged_dir = os.path.join(args.temp_dir, f'merged-{args.worker_id}')
    shutil.rmtree(merged_dir, ignore_errors=True)
    os.makedirs(merged_dir)

    cmd = f'./target/release/suffix_array merge --merged-dir {merged_dir} --suffix-path {" --suffix-path ".join(files)} --num-threads {args.cpus} --hacksize {HACK}'
    pipe = os.popen(cmd)
    output = pipe.read()
    if pipe.close() is not None:
        print('Something went wrong with merging.')
        exit(1)

    shutil.rmtree(parts_dir)

    end_time = time.time()
    print(f'Shard {t} / {len(ds_paths)}: merge done. Took {end_time-start_time:.2f} seconds')

    # -------- Step 2.3 (concat) -------- #

    print(f'Shard {t} / {len(ds_paths)}: concat ...')
    start_time = time.time()

    os.popen(f'cat {merged_dir}/* > {sa_path}').read()
    shutil.rmtree(merged_dir)

    end_time = time.time()
    print(f'Shard {t} / {len(ds_paths)}: concat done. Took {end_time-start_time:.2f} seconds')

    # -------- Step 2.4 (verify) -------- #

    if not os.path.exists(sa_path):
        print('Failed to create table')
        exit(1)

    table_size = os.path.getsize(sa_path)
    if table_size % (tok_size // 2) != 0:
        print('File size is wrong')
        exit(1)

    os.chmod(sa_path, 0o444)

    end_time_all = time.time()
    print(f'Shard {t} / {len(ds_paths)}: Done. Took {end_time_all-start_time_all:.2f} seconds')
