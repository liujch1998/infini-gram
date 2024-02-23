with open('/home/ubuntu/dolma/urls/v1_6-sample.txt') as f:
    lines = f.readlines()

with open(f'/home/ubuntu/dolma/urls/v1_6-sample.aria', 'w') as f:
    for line in lines:
        PREFIX = 'https://olmo-data.org/dolma-v1_6-8B-sample/'
        assert line.startswith(PREFIX)
        path = line[len(PREFIX):].split('/')
        d, fn = '/'.join(path[:-1]), path[-1].strip('\n')
        f.write(line)
        f.write(f'  dir=/data_c/dolma-v1_6-sample/{d}\n')
        f.write(f'  out={fn}\n')

