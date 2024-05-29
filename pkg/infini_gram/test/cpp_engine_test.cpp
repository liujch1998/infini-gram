// g++ -std=c++17 -O3 -pthread cpp_engine_test.cpp -o cpp_engine_test

#include "../cpp_engine.h"

void print_search_docs_result(const SearchDocsResult &result) {
    cout << "cnt: " << result.cnt << endl;
    cout << "approx: " << result.approx << endl;
    cout << "cnt_by_clause: [ "; for (auto cnt : result.cnt_by_clause) cout << cnt << " "; cout << "]" << endl;
    cout << "idxs: [ "; for (auto idx : result.idxs) cout << idx << " "; cout << "]" << endl;
    cout << "documents: " << endl;
    for (auto &doc : result.documents) {
        cout << "  { doc_ix: " << doc.doc_ix << ", doc_len: " << doc.doc_len << ", disp_len: " << doc.disp_len << ", token_ids: [ ";
        for (auto token_id : doc.token_ids) cout << token_id << " ";
        cout << "] }" << endl;
    }
    cout << endl;
}

int main() {
    srand(time(0));
    std::mt19937 gen(19260817);
    std::uniform_int_distribution<U16> dis(0, 30000);

    const Config config = {
        .MAX_CNT_FOR_NTD = 1000,
        .MAX_OUTPUT_DOC_TOKENS = 100,
        .MAX_CLAUSE_FREQ_PER_SHARD = 50000,
        .MAX_DIFF_TOKENS = 100,
        .ds_prefetch_depth = 1,
        .sa_prefetch_depth = 3,
        .od_prefetch_depth = 3,
    };

    auto lm = NGramLanguageModeling("../../../index/v4_pileval_llama", 2, config);
    // auto lm = NGramLanguageModelingUnion({"/gscratch/xlab/liujc/ha-infini-gram/index/v4_pileval_llama", "/gscratch/xlab/liujc/ha-infini-gram/index/v4_pileval_llama"}, 2, config);

    {
        cout << "count, empty query" << endl;
        vector<U16> input_ids = {};
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.count(input_ids);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, simple query" << endl;
        vector<U16> input_ids = {5613, 4086, 9068}; // natural language processing
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.count(input_ids);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, simple query, zero count" << endl;
        vector<U16> input_ids = {dis(gen), dis(gen), dis(gen)}; // * * *
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.count(input_ids);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, OR query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
        };
        auto result = lm.count_cnf(cnf);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, AND query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}}, // natural language processing
            {{6483, 6509}}, // deep learning
        };
        auto result = lm.count_cnf(cnf);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, CNF query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
            {{6483, 6509}}, // deep learning
        };
        auto result = lm.count_cnf(cnf);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "prob" << endl;
        vector<U16> prompt_ids = {5613, 4086}; // natural language
        U16 cont_id = 9068; // processing
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        cout << "cont_id: " << cont_id << endl;
        auto result = lm.prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << endl;
    }
    {
        cout << "prob, empty prompt" << endl;
        vector<U16> prompt_ids = {};
        U16 cont_id = 9068; // processing
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        cout << "cont_id: " << cont_id << endl;
        auto result = lm.prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << endl;
    }
    {
        cout << "prob, zero cont_cnt" << endl;
        vector<U16> prompt_ids = {5613, 4086}; // natural language
        U16 cont_id = dis(gen); // *
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        cout << "cont_id: " << cont_id << endl;
        auto result = lm.prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << endl;
    }
    {
        cout << "prob, zero prompt_cnt" << endl;
        vector<U16> prompt_ids = {dis(gen), dis(gen), 5613, 4086}; // * * natural language
        U16 cont_id = 9068; // processing
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        cout << "cont_id: " << cont_id << endl;
        auto result = lm.prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << endl;
    }
    {
        cout << "ntd, exact" << endl;
        vector<U16> prompt_ids = {5613, 4086}; // natural language
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.ntd(prompt_ids);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "ntd, approx" << endl;
        vector<U16> prompt_ids = {5613}; // natural
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.ntd(prompt_ids);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "ntd, empty prompt" << endl;
        vector<U16> prompt_ids = {};
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.ntd(prompt_ids);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "infgram_prob" << endl;
        vector<U16> prompt_ids = {dis(gen), dis(gen), 5613, 4086}; // * * natural language
        U16 cont_id = 9068; // processing
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        cout << "cont_id: " << cont_id << endl;
        auto result = lm.infgram_prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << "suffix_len: " << result.suffix_len << endl;
        cout << endl;
    }
    {
        cout << "infgram_prob, whole context" << endl;
        vector<U16> prompt_ids = {5613, 4086}; // natural language
        U16 cont_id = 9068; // processing
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        cout << "cont_id: " << cont_id << endl;
        auto result = lm.infgram_prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << "suffix_len: " << result.suffix_len << endl;
        cout << endl;
    }
    {
        cout << "infgram_prob, no context" << endl;
        vector<U16> prompt_ids = {65534}; // *
        U16 cont_id = 9068; // processing
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        cout << "cont_id: " << cont_id << endl;
        auto result = lm.infgram_prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << "suffix_len: " << result.suffix_len << endl;
        cout << endl;
    }
    {
        cout << "infgram_ntd" << endl;
        vector<U16> prompt_ids = {dis(gen), dis(gen), 5613, 4086}; // * * natural language
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.infgram_ntd(prompt_ids);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << "suffix_len: " << result.suffix_len << endl;
        cout << endl;
    }
    {
        cout << "infgram_ntd, whole context" << endl;
        vector<U16> prompt_ids = {5613, 4086}; // natural language
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.infgram_ntd(prompt_ids);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << "suffix_len: " << result.suffix_len << endl;
        cout << endl;
    }
    {
        cout << "infgram_ntd, no context" << endl;
        vector<U16> prompt_ids = {65534}; // *
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = lm.infgram_ntd(prompt_ids);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << "suffix_len: " << result.suffix_len << endl;
        cout << endl;
    }
    {
        cout << "search_docs, empty query" << endl;
        vector<vector<vector<U16>>> cnf = {{{}}};
        size_t maxnum = 1;
        auto result = lm.search_docs(cnf, maxnum);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, simple query" << endl;
        vector<vector<vector<U16>>> cnf = {{{5613, 4086, 9068}}}; // natural language processing
        size_t maxnum = 5;
        auto result = lm.search_docs(cnf, maxnum);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, simple query, zero count" << endl;
        vector<vector<vector<U16>>> cnf = {{{dis(gen), dis(gen), dis(gen)}}}; // * * *
        size_t maxnum = 1;
        auto result = lm.search_docs(cnf, maxnum);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, OR query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
        };
        size_t maxnum = 1;
        auto result = lm.search_docs(cnf, maxnum);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, AND query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}}, // natural language processing
            {{6483, 6509}}, // deep learning
        };
        size_t maxnum = 1;
        auto result = lm.search_docs(cnf, maxnum);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, CNF query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
            {{6483, 6509}}, // deep learning
        };
        size_t maxnum = 1;
        auto result = lm.search_docs(cnf, maxnum);
        print_search_docs_result(result);
    }

    return 0;
}
