import argparse
import glob
import gzip
import json
import multiprocessing as mp
import numpy as np
import os
import resource
import shutil
import sys
import time
from tqdm import tqdm

HACK = 100000

tokenizer = None

def load_file(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    elif path.endswith('.zst'):
        with open(path, 'rb') as f:
            import zstandard as zstd
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
    global tokenizer
    metadata = json.loads(line.strip('\n'))
    tok_text = tokenizer.encode(metadata['text'])
    del metadata['text']
    byte_arr = np.array(tok_text, dtype=np.uint16).view(np.uint8).tobytes()
    return byte_arr, metadata
# def extract_text(line):
#     js = json.loads(line.strip('\n'))
#     text = js['text']
#     ID = js["id"] if "id" in js else ""
#     return text, ID
# def convert_to_bytes(tok_text):
#     return np.array(tok_text, dtype=np.uint16).view(np.uint8).tobytes()

def tokenize(args):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    od_paths = [os.path.join(args.save_dir, f'offset.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    mt_paths = [os.path.join(args.save_dir, f'metadata.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    om_paths = [os.path.join(args.save_dir, f'metaoff.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    if all([os.path.exists(ds_path) for ds_path in ds_paths]) \
        and all([os.path.exists(od_path) for od_path in od_paths]):
        print('Step 1 (tokenize): Skipped. All tokenized files already exist.')
        return

    print('Step 1 (tokenize): Starting ...')

    import transformers
    transformers.utils.logging.set_verbosity(40) # suppress warnings
    global tokenizer
    if args.tokenizer == 'gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama':
        tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'), use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    elif args.tokenizer == 'olmo':
        tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False)
        # # The following is a faster version, but the result is a bit different
        # from dolma.tokenizer import Tokenizer
        # tokenizer = Tokenizer.from_pretrained('allenai/gpt-neox-olmo-dolma-v1_5', bos_token_id=None, eos_token_id=None, pad_token_id=1, segment_before_tokenization=True)
    else:
        raise ValueError(f'Unknown tokenizer: {args.tokenizer}')

    data_paths = glob.glob(f'{args.data_dir}/**/*.json*', recursive=True)
    data_paths = list(sorted(data_paths))
    ds_fouts = [open(ds_path, 'wb') for ds_path in ds_paths]
    od_fouts = [open(od_path, 'wb') for od_path in od_paths]
    if args.add_metadata:
        mt_fouts = [open(mt_path, 'w') for mt_path in mt_paths]
        om_fouts = [open(om_path, 'wb') for om_path in om_paths]
    with mp.get_context('fork').Pool(args.cpus) as p:
        ods = [0 for _ in od_fouts]
        if args.add_metadata:
            oms = [0 for _ in om_fouts]
        for data_path in tqdm(data_paths):
            rel_data_path = data_path[len(args.data_dir)+1:]
            lines = load_file(data_path)
            for offset in tqdm(range(0, len(lines), args.workers*args.batch_size), total=len(range(0, len(lines), args.workers*args.batch_size))):
                batch_lines = lines[(offset+args.worker_id):(offset+args.workers*args.batch_size):args.workers]
                results = p.map(tok, batch_lines)
                for i, (byte_arr, metadata) in enumerate(results):
                    content = args.doc_sep + byte_arr
                    j = i % (args.shards // args.workers)
                    ds_fouts[j].write(content)
                    od_fouts[j].write(np.array([ods[j]], dtype=np.uint64).view(np.uint8).tobytes())
                    ods[j] += len(content)
                    if args.add_metadata:
                        linenum = (offset + args.worker_id) + args.workers * i
                        mt = json.dumps({'path': rel_data_path, 'linenum': linenum, 'metadata': metadata}) + '\n'
                        mt_fouts[j].write(mt)
                        om_fouts[j].write(np.array([oms[j]], dtype=np.uint64).view(np.uint8).tobytes())
                        oms[j] += len(mt)
            del lines

    for ds_fout in ds_fouts:
        ds_fout.close()
    for od_fout in od_fouts:
        od_fout.close()
    if args.add_metadata:
        for mt_fout in mt_fouts:
            mt_fout.close()
        for om_fout in om_fouts:
            om_fout.close()

def build_sa(args):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    print('Step 2 (build suffix array): starting ...')

    for t, ds_path in enumerate(ds_paths):
        sa_path = ds_path.replace('tokenized', 'table')
        if os.path.exists(sa_path):
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
                cmd = f'./rust_indexing make-part --data-file {ds_path} --parts-dir {parts_dir} --start-byte {s} --end-byte {e}'
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

        cmd = f'./rust_indexing merge --merged-dir {merged_dir} --suffix-path {" --suffix-path ".join(files)} --num-threads {args.cpus} --hacksize {HACK}'
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

        end_time_all = time.time()
        print(f'Shard {t} / {len(ds_paths)}: Done. Took {end_time_all-start_time_all:.2f} seconds')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the raw text corpus. Must be absolute path.')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory where temporary indexing files are stored. Must be absolute path.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where the final index files are stored. Must be absolute path.')
    parser.add_argument('--tokenizer', type=str, required=True, choices=['gpt2', 'llama', 'olmo'])
    parser.add_argument('--doc_sep', type=bytes, default=b'\xff\xff')
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size for tokenization.')
    parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Number of CPU cores available to the program.')
    parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GiB available to the program.')
    parser.add_argument('--shards', type=int, default=1, help='Number of shards to split the index into.')
    parser.add_argument('--workers', type=int, default=1, help='Total number of workers. Must be a divisor of shards.')
    parser.add_argument('--worker_id', type=int, default=0, help='The worker ID of this process. Must be in range [0, workers).')
    parser.add_argument('--add_metadata', default=False, action='store_true', help='Whether to store document metadata in the index.')
    parser.add_argument('--ulimit', type=int, default=1048576, help='Maximum number of open files allowed.')
    args = parser.parse_args()
    if args.temp_dir is None:
        args.temp_dir = args.save_dir
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

    assert sys.byteorder == 'little'
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.ulimit, args.ulimit))

    tokenize(args)
    build_sa(args)

if __name__ == '__main__':
    main()
