// c++ -std=c++17 -O3 -shared -fPIC $(python3 -m pybind11 --includes) infini_gram/cpp_engine.cpp -o infini_gram/cpp_engine$(python3-config --extension-suffix)

#include "cpp_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpp_engine, m) {

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

    py::class_<DistResult>(m, "DistResult")
        .def_readwrite("prompt_cnt", &DistResult::prompt_cnt)
        .def_readwrite("result_by_token_id", &DistResult::result_by_token_id)
        .def_readwrite("approx", &DistResult::approx);

    py::class_<InfgramProbResult>(m, "InfgramProbResult")
        .def_readwrite("prompt_cnt", &InfgramProbResult::prompt_cnt)
        .def_readwrite("cont_cnt", &InfgramProbResult::cont_cnt)
        .def_readwrite("prob", &InfgramProbResult::prob)
        .def_readwrite("suffix_len", &InfgramProbResult::suffix_len);

    py::class_<InfgramDistResult>(m, "InfgramDistResult")
        .def_readwrite("prompt_cnt", &InfgramDistResult::prompt_cnt)
        .def_readwrite("result_by_token_id", &InfgramDistResult::result_by_token_id)
        .def_readwrite("approx", &InfgramDistResult::approx)
        .def_readwrite("suffix_len", &InfgramDistResult::suffix_len);

    py::class_<DocResult>(m, "DocResult")
        .def_readwrite("doc_ix", &DocResult::doc_ix)
        .def_readwrite("doc_len", &DocResult::doc_len)
        .def_readwrite("disp_len", &DocResult::disp_len)
        .def_readwrite("needle_offset", &DocResult::needle_offset)
        .def_readwrite("metadata", &DocResult::metadata)
        .def_readwrite("token_ids", &DocResult::token_ids);

    py::class_<SearchDocsResult>(m, "SearchDocsResult")
        .def_readwrite("cnt", &SearchDocsResult::cnt)
        .def_readwrite("approx", &SearchDocsResult::approx)
        .def_readwrite("idxs", &SearchDocsResult::idxs)
        .def_readwrite("docs", &SearchDocsResult::docs);

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

    py::class_<Attribution2Doc>(m, "Attribution2Doc")
        .def_readwrite("s", &Attribution2Doc::s)
        .def_readwrite("local_doc_ix", &Attribution2Doc::local_doc_ix)
        .def_readwrite("doc_start_ptr", &Attribution2Doc::doc_start_ptr)
        .def_readwrite("doc_end_ptr", &Attribution2Doc::doc_end_ptr)
        .def_readwrite("doc_ix", &Attribution2Doc::doc_ix)
        .def_readwrite("doc_len", &Attribution2Doc::doc_len)
        .def_readwrite("disp_len", &Attribution2Doc::disp_len)
        .def_readwrite("disp_offset", &Attribution2Doc::disp_offset)
        .def_readwrite("metadata", &Attribution2Doc::metadata)
        .def_readwrite("token_ids", &Attribution2Doc::token_ids)
        .def_readwrite("token_offset_span_pairs", &Attribution2Doc::token_offset_span_pairs)
        .def_readwrite("total_matched_len", &Attribution2Doc::total_matched_len);

    py::class_<Attribution2Result>(m, "Attribution2Result")
        .def_readwrite("spans", &Attribution2Result::spans)
        .def_readwrite("docs", &Attribution2Result::docs);

    py::class_<Engine>(m, "Engine")
        .def(py::init<const vector<string>, const U16, const bool, const size_t, const size_t, const size_t, const set<U16>, const bool>())
        .def("find", &Engine::find, "input_ids"_a)
        .def("find_cnf", &Engine::find_cnf, "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("count", &Engine::count, "input_ids"_a)
        .def("count_cnf", &Engine::count_cnf, "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("prob", &Engine::prob, "prompt_ids"_a, "cont_id"_a)
        .def("ntd", &Engine::ntd, "prompt_ids"_a, "max_support"_a)
        .def("infgram_prob", &Engine::infgram_prob, "prompt_ids"_a, "cont_id"_a)
        .def("infgram_ntd", &Engine::infgram_ntd, "prompt_ids"_a, "max_support"_a)
        .def("search_docs", &Engine::search_docs, "input_ids"_a, "maxnum"_a, "max_disp_len"_a)
        .def("search_docs_cnf", &Engine::search_docs_cnf, "cnf"_a, "maxnum"_a, "max_disp_len"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("get_doc_by_rank", &Engine::get_doc_by_rank, "s"_a, "rank"_a, "max_disp_len"_a)
        .def("get_docs_by_ranks", &Engine::get_docs_by_ranks, "list_of_s_and_rank"_a, "max_disp_len"_a)
        .def("get_doc_by_ptr", &Engine::get_doc_by_ptr, "s"_a, "ptr"_a, "max_disp_len"_a)
        .def("get_docs_by_ptrs", &Engine::get_docs_by_ptrs, "list_of_s_and_ptr"_a, "max_disp_len"_a)
        .def("get_doc_by_ix", &Engine::get_doc_by_ix, "doc_ix"_a, "max_disp_len"_a)
        .def("get_docs_by_ixs", &Engine::get_docs_by_ixs, "list_of_doc_ix"_a, "max_disp_len"_a)
        .def("get_doc_by_rank_2", &Engine::get_doc_by_rank_2, "s"_a, "rank"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ranks_2", &Engine::get_docs_by_ranks_2, "list_of_s_and_rank"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_doc_by_ptr_2", &Engine::get_doc_by_ptr_2, "s"_a, "ptr"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_docs_by_ptrs_2", &Engine::get_docs_by_ptrs_2, "list_of_s_and_ptr"_a, "needle_len"_a, "max_ctx_len"_a)
        .def("get_num_shards", &Engine::get_num_shards)
        .def("get_tok_cnt", &Engine::get_tok_cnt, "s"_a)
        .def("get_ds_size", &Engine::get_ds_size, "s"_a)
        .def("get_total_doc_cnt", &Engine::get_total_doc_cnt)
        .def("creativity", &Engine::creativity, "input_ids"_a)
        .def("attribute", &Engine::attribute, "input_ids"_a, "delim_ids"_a, "min_len"_a, "max_cnt"_a, "enforce_bow"_a)
        .def("attribute_2", &Engine::attribute_2, "input_ids"_a, "delim_ids"_a, "min_len"_a, "max_cnt"_a, "max_docs"_a, "max_disp_len"_a);
}
