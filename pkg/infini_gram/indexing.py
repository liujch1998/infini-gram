import argparse
from collections import defaultdict
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
token_dtype = None
version = None

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
    global tokenizer, token_dtype, version
    metadata = json.loads(line.strip('\n'))
    if tokenizer is None:
        byte_arr = metadata['text'].encode('utf-8')
        if version == 5:
            byte_arr = byte_arr[::-1].copy()
    else:
        text = tokenizer.encode(metadata['text'])
        if version == 5:
            text = text[::-1].copy()
        byte_arr = np.array(text, dtype=token_dtype).view(np.uint8).tobytes()
    del metadata['text']
    return byte_arr, metadata

def tokenize(args):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    od_paths = [os.path.join(args.save_dir, f'offset.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    mt_paths = [os.path.join(args.save_dir, f'metadata.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    om_paths = [os.path.join(args.save_dir, f'metaoff.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    ug_paths = [os.path.join(args.save_dir, f'unigram.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    if all([os.path.exists(ds_path) for ds_path in ds_paths]) \
        and all([os.path.exists(od_path) for od_path in od_paths]):
        print('Step 1 (tokenize): Skipped. All tokenized files already exist.')
        return

    print('Step 1 (tokenize): Starting ...')

    import transformers
    transformers.utils.logging.set_verbosity(40) # suppress warnings
    global tokenizer, token_dtype
    if args.tokenizer is None:
        tokenizer = None
    elif args.tokenizer == 'gpt2':
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
    if args.add_unigram:
        ug_fouts = [open(ug_path, 'w') for ug_path in ug_paths]
        unigram_counts = [defaultdict(int) for ug_path in ug_paths]
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
                    if args.add_unigram:
                        token_ids = np.frombuffer(content, dtype=np.uint8).view(token_dtype)
                        for token_id in token_ids:
                            unigram_counts[j][token_id] += 1
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
    if args.add_unigram:
        for j, ug_fout in enumerate(ug_fouts):
            for token_id, count in sorted(unigram_counts[j].items()):
                ug_fout.write(f'{token_id} {count}\n')
            ug_fout.close()

def build_sa(args):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    for t, ds_path in enumerate(ds_paths):
        print(f'Shard {t} / {len(ds_paths)}', flush=True)

        sa_path = ds_path.replace('tokenized', 'table')
        if os.path.exists(sa_path):
            print(f'Step 2 (build_sa): Skipped. File already exists.', flush=True)
            continue

        print('Step 2 (build_sa): Starting ...', flush=True)
        start_time_all = time.time()

        # -------- Step 2.1 (make-part) -------- #

        print(f'\tStep 2.1 (make-part): Starting ...', flush=True)
        start_time = time.time()

        ds_size = os.path.getsize(ds_path)
        ratio = int(np.ceil(np.log2(ds_size) / 8))
        mem_bytes = args.mem * 1024**3
        num_job_batches = 1
        while num_job_batches * (mem_bytes // (12 if args.token_width == 1 else 8)) < ds_size:
            num_job_batches *= 2
        parallel_jobs = args.cpus
        total_jobs = num_job_batches * parallel_jobs
        print(f'Using {num_job_batches} batches of {parallel_jobs} jobs each, for a total of {total_jobs} jobs.', flush=True)

        S = ds_size // total_jobs
        # Make sure that parts contain whole tokens
        if S % args.token_width != 0:
            S += args.token_width - S % args.token_width

        parts_dir = os.path.join(args.temp_dir, f'parts-{args.worker_id}')
        shutil.rmtree(parts_dir, ignore_errors=True)
        os.makedirs(parts_dir)

        for batch_start in tqdm(list(range(0, total_jobs, parallel_jobs))):
            batch_end = min(batch_start+parallel_jobs, total_jobs)
            batch_ranges = []
            for i in range(batch_start, batch_end):
                s, e = i*S, min((i+1)*S+HACK, ds_size)
                batch_ranges.append((s, e))
            pipes = []
            for (s, e) in batch_ranges:
                pipes.append(os.popen(f'./rust_indexing make-part --data-file {ds_path} --parts-dir {parts_dir} --start-byte {s} --end-byte {e} --ratio {ratio} --token-width {args.token_width}'))
            [pipe.read() for pipe in pipes]
            if any([pipe.close() is not None for pipe in pipes]):
                print('\tStep 2.1 (make-part): Something went wrong', flush=True)
                exit(1)

        end_time = time.time()
        print(f'\tStep 2.1 (make-part): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        # -------- Step 2.2 (merge) -------- #

        print(f'\tStep 2.2 (merge): Starting ...', flush=True)
        start_time = time.time()

        merged_dir = os.path.join(args.temp_dir, f'merged-{args.worker_id}')
        shutil.rmtree(merged_dir, ignore_errors=True)
        os.makedirs(merged_dir)

        pipe = os.popen(f'./rust_indexing merge --data-file {ds_path} --parts-dir {parts_dir} --merged-dir {merged_dir} --num-threads {args.cpus} --hacksize {HACK} --ratio {ratio} --token-width {args.token_width}')
        pipe.read()
        if pipe.close() is not None:
            print('\tStep 2.2 (merge): Something went wrong', flush=True)
            exit(1)

        shutil.rmtree(parts_dir)

        end_time = time.time()
        print(f'\tStep 2.2 (merge): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        # -------- Step 2.3 (concat) -------- #

        print(f'\tStep 2.3 (concat): Starting ...', flush=True)
        start_time = time.time()

        pipe = os.popen(f'./rust_indexing concat --data-file {ds_path} --merged-dir {merged_dir} --merged-file {sa_path} --num-threads {args.cpus} --ratio {ratio} --token-width {args.token_width}')
        pipe.read()
        if pipe.close() is not None:
            print('\tStep 2.3 (concat): Something went wrong', flush=True)
            exit(1)

        shutil.rmtree(merged_dir)

        end_time = time.time()
        print(f'\tStep 2.3 (concat): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        end_time_all = time.time()
        print(f'Step 2 (build_sa): Done. Took {end_time_all-start_time_all:.2f} seconds', flush=True)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the raw text corpus. Must be absolute path.')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory where temporary indexing files are stored. Must be absolute path.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where the final index files are stored. Must be absolute path.')
    parser.add_argument('--version', type=int, default=4, choices=[4, 5], help='Version of the index.')
    parser.add_argument('--tokenizer', type=str, default=None, choices=[None, 'gpt2', 'llama', 'olmo'])
    parser.add_argument('--token_dtype', type=str, default='u16', choices=['u8', 'u16', 'u32'], help='Data type for tokens.')
    parser.add_argument('--add_metadata', default=False, action='store_true', help='Whether to store document metadata in the index.')
    parser.add_argument('--add_unigram', default=False, action='store_true', help='Whether to precompute unigram counts.')
    parser.add_argument('--shards', type=int, default=1, help='Number of shards to split the index into.')
    parser.add_argument('--workers', type=int, default=1, help='Total number of workers. Must be a divisor of shards.')
    parser.add_argument('--worker_id', type=int, default=0, help='The worker ID of this process. Must be in range [0, workers).')
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size for tokenization.')
    parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Number of CPU cores available to the program.')
    parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GiB available to the program.')
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

    global token_dtype, version
    if args.token_dtype == 'u8':
        token_dtype = np.uint8
        args.token_width = 1
        args.doc_sep = b'\xff'
    elif args.token_dtype == 'u16':
        token_dtype = np.uint16
        args.token_width = 2
        args.doc_sep = b'\xff\xff'
    elif args.token_dtype == 'u32':
        token_dtype = np.uint32
        args.token_width = 4
        args.doc_sep = b'\xff\xff\xff\xff'
    else:
        raise ValueError(f'Unknown token_dtype: {args.token_dtype}')
    version = args.version

    assert os.path.exists(args.data_dir)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    assert sys.byteorder == 'little'
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.ulimit, args.ulimit))

    tokenize(args)
    build_sa(args)

if __name__ == '__main__':
    main()
