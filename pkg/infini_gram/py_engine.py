import sys
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import boto3

@dataclass
class DatastoreShard:
    ds: str
    sa: str
    tok_cnt: int
    ds_size: int
    ptr_size: int
    od: str
    doc_cnt: int
    mt: Optional[str] = None
    mt_size: Optional[int] = None
    om: Optional[str] = None

@dataclass
class FindResult:
    cnt: int
    segment_by_shard: List[Tuple[int, int]]

@dataclass
class CountResult:
    count: int
    approx: bool

@dataclass
class ProbResult:
    prompt_cnt: int
    cont_cnt: int
    prob: float

@dataclass
class DistTokenResult:
    cont_cnt: int
    prob: float

@dataclass
class DistResult:
    prompt_cnt: int
    result_by_token_id: Dict[int, DistTokenResult]
    approx: bool

@dataclass
class InfgramProbResult:
    prompt_cnt: int
    cont_cnt: int
    prob: float
    suffix_len: int

@dataclass
class InfgramDistResult:
    prompt_cnt: int
    result_by_token_id: Dict[int, DistTokenResult]
    approx: bool
    suffix_len: int

@dataclass
class DocResult:
    doc_ix: int
    doc_len: int
    disp_len: int
    needle_offset: int
    metadata: str
    token_ids: List[int]
    blocked: bool

class Engine:

    def __init__(self,
        token_width: int,
        s3_names: list[str], eos_token_id: int, vocab_size: int, version: int,
    ):

        assert token_width in [1, 2, 4]
        assert version == 4, 'No support for v5 index in Python engine!'
        self.token_width = token_width
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.version = version
        self.doc_sep_id = (2 ** (token_width * 8)) - 1

        assert sys.byteorder == 'little'

        self.s3 = boto3.client('s3')
        self.shards = []
        self.num_shards = 0

        for s3_name in s3_names:
            response = self.s3.list_objects_v2(Bucket='infini-gram', Prefix=f'index/{s3_name}')
            if 'Contents' not in response:
                print(f'Error listing objects in index {s3_name} on S3!')
                return
            ds_paths, sa_paths, od_paths, mt_paths, om_paths = [], [], [], [], []
            for obj in response['Contents']:
                file_name = obj['Key']
                if 'tokenized' in file_name:
                    ds_paths.append(file_name)
                elif 'table' in file_name:
                    sa_paths.append(file_name)
                elif 'offset' in file_name:
                    od_paths.append(file_name)
                elif 'metadata' in file_name:
                    mt_paths.append(file_name)
                elif 'metaoff' in file_name:
                    om_paths.append(file_name)
            ds_paths.sort()
            sa_paths.sort()
            od_paths.sort()
            mt_paths.sort()
            om_paths.sort()
            assert len(ds_paths) == len(sa_paths) == len(od_paths)
            assert len(mt_paths) == 0 or len(mt_paths) == len(ds_paths)
            assert len(om_paths) == len(mt_paths)

            for i in range(len(ds_paths)):
                # get ds_size by querying the object size on s3
                ds_size = self.s3.head_object(Bucket='infini-gram', Key=ds_paths[i])['ContentLength']
                sa_size = self.s3.head_object(Bucket='infini-gram', Key=sa_paths[i])['ContentLength']
                od_size = self.s3.head_object(Bucket='infini-gram', Key=od_paths[i])['ContentLength']

                assert ds_size % token_width == 0
                tok_cnt = ds_size // token_width
                assert sa_size % tok_cnt == 0
                ptr_size = sa_size // tok_cnt
                assert od_size % 8 == 0
                doc_cnt = od_size // 8

                if len(mt_paths) == 0:
                    self.shards.append(DatastoreShard(ds=ds_paths[i], sa=sa_paths[i], tok_cnt=tok_cnt, ds_size=ds_size, ptr_size=ptr_size, od=od_paths[i], doc_cnt=doc_cnt))
                else:
                    mt_size = self.s3.head_object(Bucket='infini-gram', Key=mt_paths[i])['ContentLength']
                    om_size = self.s3.head_object(Bucket='infini-gram', Key=om_paths[i])['ContentLength']

                    assert om_size == doc_cnt * 8

                    self.shards.append(DatastoreShard(ds=ds_paths[i], sa=sa_paths[i], tok_cnt=tok_cnt, ds_size=ds_size, ptr_size=ptr_size, od=od_paths[i], doc_cnt=doc_cnt, mt=mt_paths[i], mt_size=mt_size, om=om_paths[i]))

        self.num_shards = len(self.shards)

    def get_bytes(self, key: str, b: int, e: int) -> bytes:

        response = self.s3.get_object(Bucket='infini-gram', Key=key, Range=f'bytes={b}-{e - 1}')
        return response['Body'].read()

    def find(self, input_ids: List[int]) -> FindResult:

        hint_segment_by_shard = [(0, shard.tok_cnt) for shard in self.shards]
        return self._find(input_ids, hint_segment_by_shard)

    def _find(self, input_ids: List[int], hint_segment_by_shard: List[Tuple[int, int]]) -> FindResult:

        assert len(hint_segment_by_shard) == self.num_shards

        # convert input_ids to bytes, where each int should take up token_width bytes
        input_bytes = b''.join([input_id.to_bytes(self.token_width, 'little') for input_id in input_ids])
        num_bytes = len(input_ids) * self.token_width

        with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            futures = [executor.submit(self._find_thread, s, input_bytes, num_bytes, hint_segment_by_shard[s]) for s in range(self.num_shards)]
            segment_by_shard = [future.result() for future in futures]

        cnt = sum(segment[1] - segment[0] for segment in segment_by_shard)

        return FindResult(cnt=cnt, segment_by_shard=segment_by_shard)

    def _find_thread(self, s: int, input_bytes: bytes, num_bytes: int, hint_segment: Tuple[int, int]) -> Tuple[int, int]:

        shard = self.shards[s]

        if num_bytes == 0:
            return (0, shard.tok_cnt)

        lo, hi = hint_segment
        while lo < hi:
            mi = (lo + hi - 1) // 2
            ptr = self._convert_rank_to_ptr(s, mi)

            ds_bytes = self.get_bytes(shard.ds, ptr, min(ptr + num_bytes, shard.ds_size))
            if ds_bytes < input_bytes:
                lo = mi + 1
            elif ds_bytes > input_bytes:
                hi = mi
            else:
                break
        if lo == hi:
            return (lo, lo)

        # search left boundary in (lo-1, mi], which should be >= query
        l, r = lo - 1, mi
        while r - l > 1:
            m = (l + r) // 2
            ptr = self._convert_rank_to_ptr(s, m)
            ds_bytes = self.get_bytes(shard.ds, ptr, min(ptr + num_bytes, shard.ds_size))
            if ds_bytes < input_bytes:
                l = m
            else:
                r = m
        left = r

        # search right boundary in (mi, hi], which should be > query
        l, r = mi, hi
        while r - l > 1:
            m = (l + r) // 2
            ptr = self._convert_rank_to_ptr(s, m)
            ds_bytes = self.get_bytes(shard.ds, ptr, min(ptr + num_bytes, shard.ds_size))
            if input_bytes < ds_bytes:
                r = m
            else:
                l = m
        right = r

        return (left, right)

    def count(self, input_ids: List[int]) -> CountResult:

        find_result = self.find(input_ids)
        return CountResult(count=find_result.cnt, approx=False)

    def get_doc_by_rank(self, s: int, rank: int, max_disp_len: int) -> DocResult:

        assert 0 <= s < self.num_shards
        shard = self.shards[s]
        assert 0 <= rank < shard.tok_cnt

        ptr = self._convert_rank_to_ptr(s, rank)
        return self.get_doc_by_ptr(s, ptr, max_disp_len)

    def get_doc_by_ptr(self, s: int, ptr: int, max_disp_len: int) -> DocResult:

        assert 0 <= s < self.num_shards
        shard = self.shards[s]
        assert 0 <= ptr < shard.ds_size
        assert ptr % self.token_width == 0

        max_prepend_tokens = max_disp_len // 2
        max_append_tokens = (max_disp_len + 1) // 2

        lo, hi = 0, shard.doc_cnt
        while hi - lo > 1:
            mi = (lo + hi) // 2
            p = self._convert_doc_ix_to_ptr(s, mi)
            if p <= ptr:
                lo = mi
            else:
                hi = mi

        local_doc_ix = lo
        doc_ix = 0
        for _ in range(s):
            doc_ix += self.shards[_].doc_cnt
        doc_ix += local_doc_ix

        doc_start_ptr = self._convert_doc_ix_to_ptr(s, local_doc_ix) + self.token_width
        doc_end_ptr = self._convert_doc_ix_to_ptr(s, local_doc_ix + 1)
        doc_len = (doc_end_ptr - doc_start_ptr) // self.token_width

        disp_start_ptr = max(doc_start_ptr, 0 if ptr < self.token_width * max_prepend_tokens else (ptr - self.token_width * max_prepend_tokens))
        disp_end_ptr = min(doc_end_ptr, ptr + self.token_width * max_append_tokens)
        disp_len = (disp_end_ptr - disp_start_ptr) // self.token_width
        needle_offset = (ptr - disp_start_ptr) // self.token_width

        metadata = ''
        if shard.mt:
            meta_start_ptr = self._convert_doc_ix_to_meta_ptr(s, local_doc_ix)
            meta_end_ptr = self._convert_doc_ix_to_meta_ptr(s, local_doc_ix + 1)
            metadata = self.get_bytes(shard.mt, meta_start_ptr, meta_end_ptr).decode('utf-8')

        token_bytes = self.get_bytes(shard.ds, disp_start_ptr, disp_end_ptr)
        token_ids = [int.from_bytes(token_bytes[i:i + self.token_width], 'little') for i in range(0, len(token_bytes), self.token_width)]

        return DocResult(doc_ix=doc_ix, doc_len=doc_len, disp_len=disp_len, needle_offset=needle_offset, metadata=metadata, token_ids=token_ids, blocked=False)

    def get_num_shards(self) -> int:
        return self.num_shards

    def get_tok_cnt(self, s: int) -> int:
        assert 0 <= s < self.num_shards
        return self.shards[s].tok_cnt

    def get_ds_size(self, s: int) -> int:
        assert 0 <= s < self.num_shards
        return self.shards[s].ds_size

    def get_total_tok_cnt(self) -> int:
        return sum(shard.tok_cnt for shard in self.shards)

    def get_total_doc_cnt(self) -> int:
        return sum(shard.doc_cnt for shard in self.shards)

    def _convert_ptr_to_token_id(self, s: int, ptr: int) -> int:
        shard = self.shards[s]
        assert ptr % self.token_width == 0
        assert ptr <= shard.ds_size
        if ptr == shard.ds_size:
            return self.eos_token_id
        token_id_bytes = self.get_bytes(shard.ds, ptr, ptr + self.token_width)
        token_id = int.from_bytes(token_id_bytes, 'little')
        if token_id == self.doc_sep_id:
            token_id = self.eos_token_id
        return token_id

    def _convert_rank_to_ptr(self, s: int, rank: int) -> int:
        shard = self.shards[s]
        assert rank < shard.tok_cnt
        ptr_bytes = self.get_bytes(shard.sa, rank * shard.ptr_size, rank * shard.ptr_size + shard.ptr_size)
        ptr = int.from_bytes(ptr_bytes, 'little')
        return ptr

    def _convert_doc_ix_to_ptr(self, s: int, doc_ix: int) -> int:
        shard = self.shards[s]
        assert doc_ix <= shard.doc_cnt
        if doc_ix == shard.doc_cnt:
            return shard.ds_size
        ptr_bytes = self.get_bytes(shard.od, doc_ix * 8, doc_ix * 8 + 8)
        ptr = int.from_bytes(ptr_bytes, 'little')
        return ptr

    def _convert_doc_ix_to_meta_ptr(self, s: int, doc_ix: int) -> int:
        shard = self.shards[s]
        assert doc_ix <= shard.doc_cnt
        if doc_ix == shard.doc_cnt:
            return shard.mt_size
        ptr_bytes = self.get_bytes(shard.om, doc_ix * 8, doc_ix * 8 + 8)
        ptr = int.from_bytes(ptr_bytes, 'little')
        return ptr
