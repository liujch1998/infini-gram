'''
import glob

paths = glob.glob('/data_c/dolma-v1_6/**/*.aria2')

with open(f'/home/ubuntu/dolma/urls/v1_6.resume.aria', 'w') as f:
    for path in paths:
        PREFIX = '/data_c/dolma-v1_6/'
        assert path.startswith(PREFIX)
        assert path.endswith('.aria2')
        rel_path = path[len(PREFIX):-len('.aria2')]
        segments = rel_path.split('/')
        d, fn = '/'.join(segments[:-1]), segments[-1]
        f.write(f'https://olmo-data.org/dolma-v1_6/{rel_path}\n')
        f.write(f'  dir=/data_c/dolma-v1_6/{d}\n')
        f.write(f'  out={fn}\n')
'''

'''
import os

with open('/home/ubuntu/dolma/urls/v1_6.txt') as f:
    lines = f.readlines()

with open(f'/home/ubuntu/dolma/urls/v1_6.resume.aria', 'w') as f:
    for line in lines:
        PREFIX = 'https://olmo-data.org/dolma-v1_6/'
        assert line.startswith(PREFIX)
        rel_path = line[len(PREFIX):].rstrip('\n')
        segments = rel_path.split('/')
        d, fn = '/'.join(segments[:-1]), segments[-1]
        LOCAL_PREFIX = '/data_c/dolma-v1_6/'
        path = LOCAL_PREFIX + rel_path
        if not os.path.exists(path):
            f.write(line)
            f.write(f'  dir=/data_c/dolma-v1_6/{d}\n')
            f.write(f'  out={fn}\n')
'''

import glob

paths = glob.glob('/data_c/dolma-v1_6/cc_en_tail/*')
ixs = []
for path in paths:
    fn = path.split('/')[-1]
    PREFIX = 'cc_en_tail-'
    SUFFIX = '.json.gz'
    assert fn.startswith(PREFIX)
    assert fn.endswith(SUFFIX)
    ix = int(fn[len(PREFIX):-len(SUFFIX)])
    ixs.append(ix)
for ix in range(max(ixs) + 1):
    if ix not in ixs:
        print(ix)

