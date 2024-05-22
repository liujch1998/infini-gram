// c++ -std=c++17 -O3 -shared -fPIC $(python3 -m pybind11 --includes) cpp_engine.cpp -o cpp_engine$(python3-config --extension-suffix)

#include "cpp_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpp_engine, m) {
    py::class_<Config>(m, "Config")
        .def(py::init<U64, U64, U64, U64, size_t, size_t, size_t>())
        .def_readwrite("MAX_CNT_FOR_NTD", &Config::MAX_CNT_FOR_NTD)
        .def_readwrite("MAX_OUTPUT_DOC_TOKENS", &Config::MAX_OUTPUT_DOC_TOKENS)
        .def_readwrite("MAX_CLAUSE_FREQ_PER_SHARD", &Config::MAX_CLAUSE_FREQ_PER_SHARD)
        .def_readwrite("MAX_DIFF_TOKENS", &Config::MAX_DIFF_TOKENS)
        .def_readwrite("ds_prefetch_depth", &Config::ds_prefetch_depth)
        .def_readwrite("sa_prefetch_depth", &Config::sa_prefetch_depth)
        .def_readwrite("od_prefetch_depth", &Config::od_prefetch_depth);

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

    py::class_<SearchDocResult>(m, "SearchDocResult")
        .def_readwrite("doc_ix", &SearchDocResult::doc_ix)
        .def_readwrite("doc_len", &SearchDocResult::doc_len)
        .def_readwrite("disp_len", &SearchDocResult::disp_len)
        .def_readwrite("token_ids", &SearchDocResult::token_ids);

    py::class_<SearchDocsResult>(m, "SearchDocsResult")
        .def_readwrite("cnt", &SearchDocsResult::cnt)
        .def_readwrite("approx", &SearchDocsResult::approx)
        .def_readwrite("cnt_by_clause", &SearchDocsResult::cnt_by_clause)
        .def_readwrite("idxs", &SearchDocsResult::idxs)
        .def_readwrite("documents", &SearchDocsResult::documents);

    py::class_<NGramLanguageModeling>(m, "NGramLanguageModeling")
        .def(py::init<const string, const U16, const Config>())
        .def("count", &NGramLanguageModeling::count, "input_ids"_a)
        .def("count_cnf", &NGramLanguageModeling::count_cnf, "cnf"_a)
        .def("prob", &NGramLanguageModeling::prob, "prompt_ids"_a, "cont_id"_a)
        .def("ntd", &NGramLanguageModeling::ntd, "prompt_ids"_a)
        .def("infgram_prob", &NGramLanguageModeling::infgram_prob, "prompt_ids"_a, "cont_id"_a)
        .def("infgram_ntd", &NGramLanguageModeling::infgram_ntd, "prompt_ids"_a)
        .def("search_docs", &NGramLanguageModeling::search_docs, "cnf"_a, "maxnum"_a);

    py::class_<NGramLanguageModelingUnion, NGramLanguageModeling>(m, "NGramLanguageModelingUnion")
        .def(py::init<const vector<string>, const U16, const Config>());
}
