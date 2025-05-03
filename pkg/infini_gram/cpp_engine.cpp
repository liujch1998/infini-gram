// c++ -std=c++20 -O3 -shared -fPIC $(python3 -m pybind11 --includes) infini_gram/cpp_engine.cpp -o infini_gram/cpp_engine$(python3-config --extension-suffix)

#include "cpp_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpp_engine, m) {

    py::class_<DatastoreShard>(m, "DatastoreShard")
        .def_readwrite("ds", &DatastoreShard::ds)
        .def_readwrite("sa", &DatastoreShard::sa)
        .def_readwrite("tok_cnt", &DatastoreShard::tok_cnt)
        .def_readwrite("ds_size", &DatastoreShard::ds_size)
        .def_readwrite("ptr_size", &DatastoreShard::ptr_size)
        .def_readwrite("od", &DatastoreShard::od)
        .def_readwrite("doc_cnt", &DatastoreShard::doc_cnt)
        .def_readwrite("mt", &DatastoreShard::mt)
        .def_readwrite("mt_size", &DatastoreShard::mt_size)
        .def_readwrite("om", &DatastoreShard::om);

    py::class_<FindResult>(m, "FindResult")
        .def_readwrite("cnt", &FindResult::cnt)
        .def_readwrite("segment_by_shard", &FindResult::segment_by_shard);

    py::class_<FindCnfResult>(m, "FindCnfResult")
        .def_readwrite("cnt", &FindCnfResult::cnt)
        .def_readwrite("approx", &FindCnfResult::approx)
        .def_readwrite("ptrs_by_shard", &FindCnfResult::ptrs_by_shard);

    py::class_<CountResult>(m, "CountResult")
        .def_readwrite("count", &CountResult::count)
        .def_readwrite("approx", &CountResult::approx);

    py::class_<ProbResult>(m, "ProbResult")
        .def_readwrite("prompt_cnt", &ProbResult::prompt_cnt)
        .def_readwrite("cont_cnt", &ProbResult::cont_cnt)
        .def_readwrite("prob", &ProbResult::prob);

    py::class_<DistTokenResult>(m, "DistTokenResult")
        .def_readwrite("cont_cnt", &DistTokenResult::cont_cnt)
        .def_readwrite("prob", &DistTokenResult::prob);

    py::class_<DistResult<U8>>(m, "DistResult_U8")
        .def_readwrite("prompt_cnt", &DistResult<U8>::prompt_cnt)
        .def_readwrite("result_by_token_id", &DistResult<U8>::result_by_token_id)
        .def_readwrite("approx", &DistResult<U8>::approx);

    py::class_<DistResult<U16>>(m, "DistResult_U16")
        .def_readwrite("prompt_cnt", &DistResult<U16>::prompt_cnt)
        .def_readwrite("result_by_token_id", &DistResult<U16>::result_by_token_id)
        .def_readwrite("approx", &DistResult<U16>::approx);

    py::class_<DistResult<U32>>(m, "DistResult_U32")
        .def_readwrite("prompt_cnt", &DistResult<U32>::prompt_cnt)
        .def_readwrite("result_by_token_id", &DistResult<U32>::result_by_token_id)
        .def_readwrite("approx", &DistResult<U32>::approx);

    py::class_<InfgramProbResult>(m, "InfgramProbResult")
        .def_readwrite("prompt_cnt", &InfgramProbResult::prompt_cnt)
        .def_readwrite("cont_cnt", &InfgramProbResult::cont_cnt)
        .def_readwrite("prob", &InfgramProbResult::prob)
        .def_readwrite("suffix_len", &InfgramProbResult::suffix_len);

    py::class_<InfgramDistResult<U8>>(m, "InfgramDistResult_U8")
        .def_readwrite("prompt_cnt", &InfgramDistResult<U8>::prompt_cnt)
        .def_readwrite("result_by_token_id", &InfgramDistResult<U8>::result_by_token_id)
        .def_readwrite("approx", &InfgramDistResult<U8>::approx)
        .def_readwrite("suffix_len", &InfgramDistResult<U8>::suffix_len);

    py::class_<InfgramDistResult<U16>>(m, "InfgramDistResult_U16")
        .def_readwrite("prompt_cnt", &InfgramDistResult<U16>::prompt_cnt)
        .def_readwrite("result_by_token_id", &InfgramDistResult<U16>::result_by_token_id)
        .def_readwrite("approx", &InfgramDistResult<U16>::approx)
        .def_readwrite("suffix_len", &InfgramDistResult<U16>::suffix_len);

    py::class_<InfgramDistResult<U32>>(m, "InfgramDistResult_U32")
        .def_readwrite("prompt_cnt", &InfgramDistResult<U32>::prompt_cnt)
        .def_readwrite("result_by_token_id", &InfgramDistResult<U32>::result_by_token_id)
        .def_readwrite("approx", &InfgramDistResult<U32>::approx)
        .def_readwrite("suffix_len", &InfgramDistResult<U32>::suffix_len);

    py::class_<DocResult<U8>>(m, "DocResult_U8")
        .def_readwrite("doc_ix", &DocResult<U8>::doc_ix)
        .def_readwrite("doc_len", &DocResult<U8>::doc_len)
        .def_readwrite("disp_len", &DocResult<U8>::disp_len)
        .def_readwrite("needle_offset", &DocResult<U8>::needle_offset)
        .def_readwrite("metadata", &DocResult<U8>::metadata)
        .def_readwrite("token_ids", &DocResult<U8>::token_ids)
        .def_readwrite("blocked", &DocResult<U8>::blocked);

    py::class_<DocResult<U16>>(m, "DocResult_U16")
        .def_readwrite("doc_ix", &DocResult<U16>::doc_ix)
        .def_readwrite("doc_len", &DocResult<U16>::doc_len)
        .def_readwrite("disp_len", &DocResult<U16>::disp_len)
        .def_readwrite("needle_offset", &DocResult<U16>::needle_offset)
        .def_readwrite("metadata", &DocResult<U16>::metadata)
        .def_readwrite("token_ids", &DocResult<U16>::token_ids)
        .def_readwrite("blocked", &DocResult<U16>::blocked);

    py::class_<DocResult<U32>>(m, "DocResult_U32")
        .def_readwrite("doc_ix", &DocResult<U32>::doc_ix)
        .def_readwrite("doc_len", &DocResult<U32>::doc_len)
        .def_readwrite("disp_len", &DocResult<U32>::disp_len)
        .def_readwrite("needle_offset", &DocResult<U32>::needle_offset)
        .def_readwrite("metadata", &DocResult<U32>::metadata)
        .def_readwrite("token_ids", &DocResult<U32>::token_ids)
        .def_readwrite("blocked", &DocResult<U32>::blocked);

    py::class_<SearchDocsResult<U8>>(m, "SearchDocsResult_U8")
        .def_readwrite("cnt", &SearchDocsResult<U8>::cnt)
        .def_readwrite("approx", &SearchDocsResult<U8>::approx)
        .def_readwrite("idxs", &SearchDocsResult<U8>::idxs)
        .def_readwrite("docs", &SearchDocsResult<U8>::docs);

    py::class_<SearchDocsResult<U16>>(m, "SearchDocsResult_U16")
        .def_readwrite("cnt", &SearchDocsResult<U16>::cnt)
        .def_readwrite("approx", &SearchDocsResult<U16>::approx)
        .def_readwrite("idxs", &SearchDocsResult<U16>::idxs)
        .def_readwrite("docs", &SearchDocsResult<U16>::docs);

    py::class_<SearchDocsResult<U32>>(m, "SearchDocsResult_U32")
        .def_readwrite("cnt", &SearchDocsResult<U32>::cnt)
        .def_readwrite("approx", &SearchDocsResult<U32>::approx)
        .def_readwrite("idxs", &SearchDocsResult<U32>::idxs)
        .def_readwrite("docs", &SearchDocsResult<U32>::docs);

    py::class_<CreativityResult>(m, "CreativityResult")
        .def_readwrite("rs", &CreativityResult::rs);

    py::class_<AttributionDoc>(m, "AttributionDoc")
        .def_readwrite("s", &AttributionDoc::s)
        .def_readwrite("ptr", &AttributionDoc::ptr);

    py::class_<AttributionSpan>(m, "AttributionSpan")
        .def_readwrite("l", &AttributionSpan::l)
        .def_readwrite("r", &AttributionSpan::r)
        .def_readwrite("length", &AttributionSpan::length)
        .def_readwrite("count", &AttributionSpan::count)
        .def_readwrite("unigram_logprob_sum", &AttributionSpan::unigram_logprob_sum)
        .def_readwrite("docs", &AttributionSpan::docs);

    py::class_<AttributionResult>(m, "AttributionResult")
        .def_readwrite("spans", &AttributionResult::spans);

    py::class_<Engine<U8>>(m, "Engine_U8")
        .def(py::init<const vector<string>, const U8, const U8, const size_t, const bool, const size_t, const size_t, const size_t, const set<U8>, const size_t, const bool, const map<string, vector<DatastoreShard>>>())
        .def("get_new_shards_by_index_dir", &Engine<U8>::get_new_shards_by_index_dir, py::call_guard<py::gil_scoped_release>())
        .def("compute_unigram_counts", &Engine<U8>::compute_unigram_counts, "s"_a)
        .def("find", &Engine<U8>::find, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("find_cnf", &Engine<U8>::find_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("count", &Engine<U8>::count, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("count_cnf", &Engine<U8>::count_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("prob", &Engine<U8>::prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
        .def("ntd", &Engine<U8>::ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
        .def("infgram_prob", &Engine<U8>::infgram_prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
        .def("infgram_ntd", &Engine<U8>::infgram_ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
        .def("search_docs", &Engine<U8>::search_docs, py::call_guard<py::gil_scoped_release>(), "input_ids"_a, "maxnum"_a, "max_disp_len"_a)
        .def("search_docs_cnf", &Engine<U8>::search_docs_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "maxnum"_a, "max_disp_len"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("get_doc_by_rank", &Engine<U8>::get_doc_by_rank, py::call_guard<py::gil_scoped_release>(), "s"_a, "rank"_a, "max_disp_len"_a)
        .def("get_docs_by_ranks", &Engine<U8>::get_docs_by_ranks, py::call_guard<py::gil_scoped_release>(), "list_of_s_and_rank"_a, "max_disp_len"_a)
        .def("get_doc_by_ptr", &Engine<U8>::get_doc_by_ptr, py::call_guard<py::gil_scoped_release>(), "s"_a, "ptr"_a, "max_disp_len"_a)
        .def("get_docs_by_ptrs", &Engine<U8>::get_docs_by_ptrs, py::call_guard<py::gil_scoped_release>(), "list_of_s_and_ptr"_a, "max_disp_len"_a)
        .def("get_doc_by_ix", &Engine<U8>::get_doc_by_ix, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_disp_len"_a)
        .def("get_docs_by_ixs", &Engine<U8>::get_docs_by_ixs, py::call_guard<py::gil_scoped_release>(), "list_of_doc_ix"_a, "max_disp_len"_a)
        .def("get_doc_by_rank_2", &Engine<U8>::get_doc_by_rank_2, py::call_guard<py::gil_scoped_release>(), "s"_a, "rank"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ranks_2", &Engine<U8>::get_docs_by_ranks_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_doc_by_ptr_2", &Engine<U8>::get_doc_by_ptr_2, py::call_guard<py::gil_scoped_release>(), "s"_a, "ptr"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ptrs_2", &Engine<U8>::get_docs_by_ptrs_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_doc_by_ix_2", &Engine<U8>::get_doc_by_ix_2, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_ctx_len"_a)
        .def("get_docs_by_ixs_2", &Engine<U8>::get_docs_by_ixs_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_num_shards", &Engine<U8>::get_num_shards, py::call_guard<py::gil_scoped_release>())
        .def("get_tok_cnt", &Engine<U8>::get_tok_cnt, py::call_guard<py::gil_scoped_release>(), "s"_a)
        .def("get_ds_size", &Engine<U8>::get_ds_size, py::call_guard<py::gil_scoped_release>(), "s"_a)
        .def("get_total_doc_cnt", &Engine<U8>::get_total_doc_cnt, py::call_guard<py::gil_scoped_release>())
        .def("creativity", &Engine<U8>::creativity, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("attribute", &Engine<U8>::attribute, py::call_guard<py::gil_scoped_release>(), "input_ids"_a, "delim_ids"_a, "min_len"_a, "max_cnt"_a, "enforce_bow"_a);

    py::class_<Engine<U16>>(m, "Engine_U16")
        .def(py::init<const vector<string>, const U16, const U16, const size_t, const bool, const size_t, const size_t, const size_t, const set<U16>, const size_t, const bool, const map<string, vector<DatastoreShard>>>())
        .def("get_new_shards_by_index_dir", &Engine<U16>::get_new_shards_by_index_dir, py::call_guard<py::gil_scoped_release>())
        .def("compute_unigram_counts", &Engine<U16>::compute_unigram_counts, "s"_a)
        .def("find", &Engine<U16>::find, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("find_cnf", &Engine<U16>::find_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("count", &Engine<U16>::count, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("count_cnf", &Engine<U16>::count_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("prob", &Engine<U16>::prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
        .def("ntd", &Engine<U16>::ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
        .def("infgram_prob", &Engine<U16>::infgram_prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
        .def("infgram_ntd", &Engine<U16>::infgram_ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
        .def("search_docs", &Engine<U16>::search_docs, py::call_guard<py::gil_scoped_release>(), "input_ids"_a, "maxnum"_a, "max_disp_len"_a)
        .def("search_docs_cnf", &Engine<U16>::search_docs_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "maxnum"_a, "max_disp_len"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("get_doc_by_rank", &Engine<U16>::get_doc_by_rank, py::call_guard<py::gil_scoped_release>(), "s"_a, "rank"_a, "max_disp_len"_a)
        .def("get_docs_by_ranks", &Engine<U16>::get_docs_by_ranks, py::call_guard<py::gil_scoped_release>(), "list_of_s_and_rank"_a, "max_disp_len"_a)
        .def("get_doc_by_ptr", &Engine<U16>::get_doc_by_ptr, py::call_guard<py::gil_scoped_release>(), "s"_a, "ptr"_a, "max_disp_len"_a)
        .def("get_docs_by_ptrs", &Engine<U16>::get_docs_by_ptrs, py::call_guard<py::gil_scoped_release>(), "list_of_s_and_ptr"_a, "max_disp_len"_a)
        .def("get_doc_by_ix", &Engine<U16>::get_doc_by_ix, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_disp_len"_a)
        .def("get_docs_by_ixs", &Engine<U16>::get_docs_by_ixs, py::call_guard<py::gil_scoped_release>(), "list_of_doc_ix"_a, "max_disp_len"_a)
        .def("get_doc_by_rank_2", &Engine<U16>::get_doc_by_rank_2, py::call_guard<py::gil_scoped_release>(), "s"_a, "rank"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ranks_2", &Engine<U16>::get_docs_by_ranks_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_doc_by_ptr_2", &Engine<U16>::get_doc_by_ptr_2, py::call_guard<py::gil_scoped_release>(), "s"_a, "ptr"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ptrs_2", &Engine<U16>::get_docs_by_ptrs_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_doc_by_ix_2", &Engine<U16>::get_doc_by_ix_2, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_ctx_len"_a)
        .def("get_docs_by_ixs_2", &Engine<U16>::get_docs_by_ixs_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_num_shards", &Engine<U16>::get_num_shards, py::call_guard<py::gil_scoped_release>())
        .def("get_tok_cnt", &Engine<U16>::get_tok_cnt, py::call_guard<py::gil_scoped_release>(), "s"_a)
        .def("get_ds_size", &Engine<U16>::get_ds_size, py::call_guard<py::gil_scoped_release>(), "s"_a)
        .def("get_total_doc_cnt", &Engine<U16>::get_total_doc_cnt, py::call_guard<py::gil_scoped_release>())
        .def("creativity", &Engine<U16>::creativity, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("attribute", &Engine<U16>::attribute, py::call_guard<py::gil_scoped_release>(), "input_ids"_a, "delim_ids"_a, "min_len"_a, "max_cnt"_a, "enforce_bow"_a);

    py::class_<Engine<U32>>(m, "Engine_U32")
        .def(py::init<const vector<string>, const U32, const U32, const size_t, const bool, const size_t, const size_t, const size_t, const set<U32>, const size_t, const bool, const map<string, vector<DatastoreShard>>>())
        .def("get_new_shards_by_index_dir", &Engine<U32>::get_new_shards_by_index_dir, py::call_guard<py::gil_scoped_release>())
        .def("compute_unigram_counts", &Engine<U32>::compute_unigram_counts, "s"_a)
        .def("find", &Engine<U32>::find, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("find_cnf", &Engine<U32>::find_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("count", &Engine<U32>::count, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("count_cnf", &Engine<U32>::count_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("prob", &Engine<U32>::prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
        .def("ntd", &Engine<U32>::ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
        .def("infgram_prob", &Engine<U32>::infgram_prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
        .def("infgram_ntd", &Engine<U32>::infgram_ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
        .def("search_docs", &Engine<U32>::search_docs, py::call_guard<py::gil_scoped_release>(), "input_ids"_a, "maxnum"_a, "max_disp_len"_a)
        .def("search_docs_cnf", &Engine<U32>::search_docs_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "maxnum"_a, "max_disp_len"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("get_doc_by_rank", &Engine<U32>::get_doc_by_rank, py::call_guard<py::gil_scoped_release>(), "s"_a, "rank"_a, "max_disp_len"_a)
        .def("get_docs_by_ranks", &Engine<U32>::get_docs_by_ranks, py::call_guard<py::gil_scoped_release>(), "list_of_s_and_rank"_a, "max_disp_len"_a)
        .def("get_doc_by_ptr", &Engine<U32>::get_doc_by_ptr, py::call_guard<py::gil_scoped_release>(), "s"_a, "ptr"_a, "max_disp_len"_a)
        .def("get_docs_by_ptrs", &Engine<U32>::get_docs_by_ptrs, py::call_guard<py::gil_scoped_release>(), "list_of_s_and_ptr"_a, "max_disp_len"_a)
        .def("get_doc_by_ix", &Engine<U32>::get_doc_by_ix, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_disp_len"_a)
        .def("get_docs_by_ixs", &Engine<U32>::get_docs_by_ixs, py::call_guard<py::gil_scoped_release>(), "list_of_doc_ix"_a, "max_disp_len"_a)
        .def("get_doc_by_rank_2", &Engine<U32>::get_doc_by_rank_2, py::call_guard<py::gil_scoped_release>(), "s"_a, "rank"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ranks_2", &Engine<U32>::get_docs_by_ranks_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_doc_by_ptr_2", &Engine<U32>::get_doc_by_ptr_2, py::call_guard<py::gil_scoped_release>(), "s"_a, "ptr"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ptrs_2", &Engine<U32>::get_docs_by_ptrs_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_doc_by_ix_2", &Engine<U32>::get_doc_by_ix_2, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_ctx_len"_a)
        .def("get_docs_by_ixs_2", &Engine<U32>::get_docs_by_ixs_2, py::call_guard<py::gil_scoped_release>(), "requests"_a)
        .def("get_num_shards", &Engine<U32>::get_num_shards, py::call_guard<py::gil_scoped_release>())
        .def("get_tok_cnt", &Engine<U32>::get_tok_cnt, py::call_guard<py::gil_scoped_release>(), "s"_a)
        .def("get_ds_size", &Engine<U32>::get_ds_size, py::call_guard<py::gil_scoped_release>(), "s"_a)
        .def("get_total_doc_cnt", &Engine<U32>::get_total_doc_cnt, py::call_guard<py::gil_scoped_release>())
        .def("creativity", &Engine<U32>::creativity, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
        .def("attribute", &Engine<U32>::attribute, py::call_guard<py::gil_scoped_release>(), "input_ids"_a, "delim_ids"_a, "min_len"_a, "max_cnt"_a, "enforce_bow"_a);

    py::class_<EngineDiff<U8>, Engine<U8>>(m, "EngineDiff_U8")
        .def(py::init<const vector<string>, const vector<string>, const U8, const U8, const size_t, const bool, const size_t, const size_t, const size_t, const set<U8>, const size_t, const bool, const map<string, vector<DatastoreShard>>>())
        .def("get_docs_by_ptrs_2_grouped", &EngineDiff<U8>::get_docs_by_ptrs_2_grouped, py::call_guard<py::gil_scoped_release>(), "requests"_a);

    py::class_<EngineDiff<U16>, Engine<U16>>(m, "EngineDiff_U16")
        .def(py::init<const vector<string>, const vector<string>, const U16, const U16, const size_t, const bool, const size_t, const size_t, const size_t, const set<U16>, const size_t, const bool, const map<string, vector<DatastoreShard>>>())
        .def("get_docs_by_ptrs_2_grouped", &EngineDiff<U16>::get_docs_by_ptrs_2_grouped, py::call_guard<py::gil_scoped_release>(), "requests"_a);

    py::class_<EngineDiff<U32>, Engine<U32>>(m, "EngineDiff_U32")
        .def(py::init<const vector<string>, const vector<string>, const U32, const U32, const size_t, const bool, const size_t, const size_t, const size_t, const set<U32>, const size_t, const bool, const map<string, vector<DatastoreShard>>>())
        .def("get_docs_by_ptrs_2_grouped", &EngineDiff<U32>::get_docs_by_ptrs_2_grouped, py::call_guard<py::gil_scoped_release>(), "requests"_a);
}
