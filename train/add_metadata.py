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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=65536, help='Typically should be a power of 2.')
parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Typically should be a power of 2.')
parser.add_argument('--shards', type=int, default=1, help='Typically should be a power of 2.')
parser.add_argument('--workers', type=int, default=1, help='Must be a divisor of shards. Typically should be a power of 2.')
parser.add_argument('--worker_id', type=int, default=0)
args = parser.parse_args()
args.data_dir = args.data_dir.rstrip('/')
args.save_dir = args.save_dir.rstrip('/')

assert args.batch_size > 0
assert args.cpus > 0
assert args.shards > 0
assert args.workers > 0
assert 0 <= args.worker_id < args.workers
assert args.shards % args.workers == 0

assert os.path.exists(args.data_dir)
os.makedirs(args.save_dir, exist_ok=True)

# ======== Step 1 (tokenize) ======== #

mt_paths = [os.path.join(args.save_dir, f'metadata.{i}') for i in range(args.worker_id, args.shards, args.workers)]
om_paths = [os.path.join(args.save_dir, f'metaoff.{i}') for i in range(args.worker_id, args.shards, args.workers)]

print('Step 1 (tokenize): Starting ...')

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

def get_metadata(line):
    metadata = json.loads(line.strip('\n'))
    del metadata['text']
    return metadata

data_paths = glob.glob(f'{args.data_dir}/**/*.json*', recursive=True)
data_paths = list(sorted(data_paths))
mt_fouts = [open(mt_path, 'w') for mt_path in mt_paths]
om_fouts = [open(om_path, 'wb') for om_path in om_paths]
with mp.get_context('fork').Pool(args.cpus) as p:
    oms = [0 for _ in om_fouts]
    for data_path in tqdm(data_paths):
        rel_data_path = data_path[len(args.data_dir)+1:]
        lines = load_file(data_path)
#        item = json.loads(lines[0].strip('\n'))
#        print(item.keys())
#        for k, v in item.items():
#            if k != 'text':
#                print(k, v)
#        exit()
        for offset in tqdm(range(0, len(lines), args.workers*args.batch_size), total=len(range(0, len(lines), args.workers*args.batch_size))):
            batch_lines = lines[(offset+args.worker_id):(offset+args.workers*args.batch_size):args.workers]
            metadatas = p.map(get_metadata, batch_lines)
            for i, metadata in enumerate(metadatas):
                j = i % (args.shards // args.workers)
                linenum = (offset + args.worker_id) + args.workers * i
                mt = json.dumps({'path': rel_data_path, 'linenum': linenum, 'metadata': metadata}) + '\n'
                mt_fouts[j].write(mt)
                om_fouts[j].write(np.array([oms[j]], dtype=np.uint64).view(np.uint8).tobytes())
                oms[j] += len(mt)
        del lines

for mt_fout in mt_fouts:
    mt_fout.close()
for om_fout in om_fouts:
    om_fout.close()

