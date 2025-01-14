from typing import Any, Iterable, List, Tuple, TypeAlias, TypeVar, TypedDict

QueryIdsType: TypeAlias = Iterable[int]
CnfType: TypeAlias = Iterable[Iterable[QueryIdsType]]

class ErrorResponse(TypedDict):
    error: str

T = TypeVar("T")
InfiniGramEngineResponse: TypeAlias = ErrorResponse | T


class FindResponse(TypedDict):
    cnt: int
    segment_by_shard: List[Tuple[int, int]]

class FindCnfResponse(TypedDict):
    cnt: int
    approx: bool
    ptrs_by_shard: List[List[int]]

class CountResponse(TypedDict):
    count: int
    approx: bool

class ProbResponse(TypedDict):
    prompt_cnt: int
    cont_cnt: int
    prob: float

class DistTokenResult(TypedDict):
    cont_cnt: int
    prob: float

class NtdResponse(TypedDict):
    prompt_cnt: int
    result_by_token_id: dict[int, DistTokenResult]
    approx: bool

class InfGramProbResponse(ProbResponse, TypedDict):
    suffix_len: int

class InfGramNtdResponse(NtdResponse, TypedDict):
    prompt_cnt: int
    result_by_token_id: dict[int, DistTokenResult]
    approx: bool
    suffix_len: int

class DocResult(TypedDict):
    doc_ix: int
    doc_len: int
    disp_len: int
    needle_offset: int
    metadata: str
    token_ids: List[int]

class SearchDocsResponse(TypedDict):
    cnt: int
    approx: bool
    idxs: List[int]
    documents: List[DocResult]

class CreativityResponse(TypedDict):
    rs: List[int]

class AttributionDoc(TypedDict):
    s: int
    ptr: int

class AttributionSpan(TypedDict):
    l: int
    r: int
    length: int
    count: int
    unigram_logprob_sum: float
    docs: List[AttributionDoc]

class AttributionResponse(TypedDict):
    spans: List[AttributionSpan]

class Attribution2Doc(TypedDict):
    doc_ix: int
    doc_len: int
    disp_len: int
    disp_offset: int
    metadata: str
    token_ids: List[int]
    token_offset_span_pairs: List[Tuple[int, Tuple[int, int]]]
    total_matched_len: int

class Attribution2Response(TypedDict):
    spans: List[Any]
    docs: List[Attribution2Doc]
