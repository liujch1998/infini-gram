#include "../inference/infini_gram.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpp_engine, m) {
    py::class_<Consts>(m, "Consts")
        .def(py::init<U64, U64, U64, U64, U64>())
        .def_readwrite("MAX_CNT_FOR_NTD", &Consts::MAX_CNT_FOR_NTD)
        .def_readwrite("MAX_OUTPUT_DOC_TOKENS", &Consts::MAX_OUTPUT_DOC_TOKENS)
        .def_readwrite("MAX_CLAUSE_FREQ_PER_SHARD", &Consts::MAX_CLAUSE_FREQ_PER_SHARD)
        .def_readwrite("MAX_DIFF_TOKENS", &Consts::MAX_DIFF_TOKENS)
        .def_readwrite("MAX_DIFF_BYTES", &Consts::MAX_DIFF_BYTES);

    py::class_<FindResult>(m, "FindResult")
        .def_readwrite("cnt", &FindResult::cnt);

    py::class_<ProbResult>(m, "ProbResult")
        .def_readwrite("prob", &ProbResult::prob)
        .def_readwrite("prompt_cnt", &ProbResult::prompt_cnt)
        .def_readwrite("cont_cnt", &ProbResult::cont_cnt);

    py::class_<DistResult>(m, "DistResult")
        .def_readwrite("prompt_cnt", &DistResult::prompt_cnt)
        .def_readwrite("freq_by_token_id", &DistResult::freq_by_token_id)
        .def_readwrite("prob_by_token_id", &DistResult::prob_by_token_id);

    py::class_<InfgramProbResult>(m, "InfgramProbResult")
        .def_readwrite("lfn", &InfgramProbResult::lfn)
        .def_readwrite("prob", &InfgramProbResult::prob)
        .def_readwrite("prompt_cnt", &InfgramProbResult::prompt_cnt)
        .def_readwrite("cont_cnt", &InfgramProbResult::cont_cnt);

    py::class_<InfgramDistResult>(m, "InfgramDistResult")
        .def_readwrite("lfn", &InfgramDistResult::lfn)
        .def_readwrite("prompt_cnt", &InfgramDistResult::prompt_cnt)
        .def_readwrite("freq_by_token_id", &InfgramDistResult::freq_by_token_id)
        .def_readwrite("prob_by_token_id", &InfgramDistResult::prob_by_token_id);

    py::class_<SearchDocResult>(m, "SearchDocResult")
        .def_readwrite("token_ids", &SearchDocResult::token_ids)
        .def_readwrite("token_offset", &SearchDocResult::token_offset)
        .def_readwrite("doc_ix", &SearchDocResult::doc_ix)
        .def_readwrite("doc_len", &SearchDocResult::doc_len)
        .def_readwrite("disp_len", &SearchDocResult::disp_len);

    py::class_<SearchDocsResult>(m, "SearchDocsResult")
        .def_readwrite("documents", &SearchDocsResult::documents)
        .def_readwrite("idxs", &SearchDocsResult::idxs)
        .def_readwrite("cnt_by_clause", &SearchDocsResult::cnt_by_clause)
        .def_readwrite("cnt", &SearchDocsResult::cnt)
        .def_readwrite("approx", &SearchDocsResult::approx);

    py::class_<NGramLanguageModeling>(m, "NGramLanguageModeling")
        .def(py::init<const string, const U16, const size_t, const size_t, const size_t, const Consts>())
        .def("find", &NGramLanguageModeling::find, "input_ids"_a, "hint_segment_by_shard"_a = vector<pair<U64, U64>>())
        .def("prob", &NGramLanguageModeling::prob, "prompt_ids"_a, "cont_id"_a)
        .def("ntd", &NGramLanguageModeling::ntd, "prompt_ids"_a)
        .def("infgram_prob", &NGramLanguageModeling::infgram_prob, "prompt_ids"_a, "cont_id"_a)
        .def("infgram_ntd", &NGramLanguageModeling::infgram_ntd, "prompt_ids"_a)
        .def("search_docs", &NGramLanguageModeling::search_docs, "cnf"_a, "maxnum"_a);

    py::class_<NGramLanguageModelingUnion, NGramLanguageModeling>(m, "NGramLanguageModelingUnion")
        .def(py::init<const vector<string>, const U16, const size_t, const size_t, const size_t, const Consts>());
}
