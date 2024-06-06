// c++ -std=c++17 -O3 -shared -fPIC $(python3 -m pybind11 --includes) cpp_engine.cpp -o cpp_engine$(python3-config --extension-suffix)

#include "cpp_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpp_engine, m) {

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
        .def_readwrite("metadata", &DocResult::metadata);
        .def_readwrite("token_ids", &DocResult::token_ids);

    py::class_<SearchDocsResult>(m, "SearchDocsResult")
        .def_readwrite("cnt", &SearchDocsResult::cnt)
        .def_readwrite("approx", &SearchDocsResult::approx)
        .def_readwrite("idxs", &SearchDocsResult::idxs)
        .def_readwrite("docs", &SearchDocsResult::docs);

    py::class_<InfiniGramEngine>(m, "InfiniGramEngine")
        .def(py::init<const string, const U16, const Config>())
        .def("count", &InfiniGramEngine::count, "input_ids"_a)
        .def("count_cnf", &InfiniGramEngine::count_cnf, "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("prob", &InfiniGramEngine::prob, "prompt_ids"_a, "cont_id"_a)
        .def("ntd", &InfiniGramEngine::ntd, "prompt_ids"_a, "max_support"_a)
        .def("infgram_prob", &InfiniGramEngine::infgram_prob, "prompt_ids"_a, "cont_id"_a)
        .def("infgram_ntd", &InfiniGramEngine::infgram_ntd, "prompt_ids"_a, "max_support"_a)
        .def("search_docs", &InfiniGramEngine::search_docs, "cnf"_a, "maxnum"_a, "max_disp_len"_a)
        .def("search_docs_cnf", &InfiniGramEngine::search_docs_cnf, "cnf"_a, "maxnum"_a, "max_disp_len"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("find", &InfiniGramEngine::find, "input_ids"_a)
        .def("find_cnf", &InfiniGramEngine::find_cnf, "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
        .def("get_doc_by_rank", &InfiniGramEngine::get_doc_by_rank, "s"_a, "rank"_a, "max_disp_len"_a)
        .def("get_doc_by_ptr", &InfiniGramEngine::get_doc_by_ptr, "s"_a, "ptr"_a, "max_disp_len"_a);
}
