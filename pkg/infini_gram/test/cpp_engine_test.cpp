// g++ -std=c++17 -O3 -pthread cpp_engine_test.cpp -o cpp_engine_test

#include "../cpp_engine.h"

void print_search_docs_result(const SearchDocsResult &result) {
    cout << "cnt: " << result.cnt << endl;
    cout << "approx: " << result.approx << endl;
    cout << "idxs: [ "; for (auto idx : result.idxs) cout << idx << " "; cout << "]" << endl;
    cout << "docs: " << endl;
    for (auto &doc : result.docs) {
        cout << "  { doc_ix: " << doc.doc_ix << ", doc_len: " << doc.doc_len << ", disp_len: " << doc.disp_len << ", metadata: " << doc.metadata << ", token_ids: [ ";
        for (auto token_id : doc.token_ids) cout << token_id << " ";
        cout << "] }" << endl;
    }
    cout << endl;
}

int main() {
    srand(time(0));
    std::mt19937 gen(19260817);
    std::uniform_int_distribution<U16> dis(0, 65534); // left inclusive, right inclusive

    U64 max_support = 10;
    U64 max_clause_freq = 50000;
    U64 max_diff_tokens = 10;
    U64 max_disp_len = 20;
    size_t maxnum = 2;

    // auto engine = InfiniGramEngine({"../../../index/v4_pileval_llama"}, 2);
    auto engine = InfiniGramEngine({"../../../index/v4_dolma-v1_6-sample_llama"}, 2);
    // auto engine = InfiniGramEngine({"../../../index/v4_pileval_llama", "../../../index/v4_dolma-v1_6-sample_llama"}, 2);

    {
        cout << "count, empty query" << endl;
        vector<U16> input_ids = {};
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.count(input_ids);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, simple query" << endl;
        vector<U16> input_ids = {5613, 4086, 9068}; // natural language processing
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.count(input_ids);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, simple query, zero count" << endl;
        vector<U16> input_ids = {dis(gen), dis(gen), dis(gen)}; // * * *
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.count(input_ids);
        cout << "count: " << result.count << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "count, OR query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
        };
        auto result = engine.count_cnf(cnf, max_clause_freq, max_diff_tokens);
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
        auto result = engine.count_cnf(cnf, max_clause_freq, max_diff_tokens);
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
        auto result = engine.count_cnf(cnf, max_clause_freq, max_diff_tokens);
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
        auto result = engine.prob(prompt_ids, cont_id);
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
        auto result = engine.prob(prompt_ids, cont_id);
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
        auto result = engine.prob(prompt_ids, cont_id);
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
        auto result = engine.prob(prompt_ids, cont_id);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "cont_cnt: " << result.cont_cnt << endl;
        cout << "prob: " << result.prob << endl;
        cout << endl;
    }
    {
        cout << "ntd, exact" << endl;
        vector<U16> prompt_ids = {5613, 4086}; // natural language
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.ntd(prompt_ids, max_support);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "ntd, approx" << endl;
        vector<U16> prompt_ids = {5613}; // natural
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.ntd(prompt_ids, max_support);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << endl;
    }
    {
        cout << "ntd, empty prompt" << endl;
        vector<U16> prompt_ids = {};
        cout << "prompt_ids: [ "; for (auto id : prompt_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.ntd(prompt_ids, max_support);
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
        auto result = engine.infgram_prob(prompt_ids, cont_id);
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
        auto result = engine.infgram_prob(prompt_ids, cont_id);
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
        auto result = engine.infgram_prob(prompt_ids, cont_id);
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
        auto result = engine.infgram_ntd(prompt_ids, max_support);
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
        auto result = engine.infgram_ntd(prompt_ids, max_support);
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
        auto result = engine.infgram_ntd(prompt_ids, max_support);
        cout << "prompt_cnt: " << result.prompt_cnt << endl;
        cout << "result_by_token_id: { "; for (auto &[token_id, r] : result.result_by_token_id) cout << token_id << " => { cont_cnt: " << r.cont_cnt << ", prob: " << r.prob << " }, "; cout << "}" << endl;
        cout << "approx: " << result.approx << endl;
        cout << "suffix_len: " << result.suffix_len << endl;
        cout << endl;
    }
    {
        cout << "search_docs, empty query" << endl;
        vector<U16> input_ids = {};
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.search_docs(input_ids, maxnum, max_disp_len);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, simple query" << endl;
        vector<U16> input_ids = {5613, 4086, 9068}; // natural language processing
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.search_docs(input_ids, maxnum, max_disp_len);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, simple query, zero count" << endl;
        vector<U16> input_ids = {dis(gen), dis(gen), dis(gen)}; // * * *
        cout << "input_ids: [ "; for (auto id : input_ids) cout << id << " "; cout << "]" << endl;
        auto result = engine.search_docs(input_ids, maxnum, max_disp_len);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, OR query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
        };
        auto result = engine.search_docs_cnf(cnf, maxnum, max_disp_len, max_clause_freq, max_diff_tokens);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, AND query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}}, // natural language processing
            {{6483, 6509}}, // deep learning
        };
        auto result = engine.search_docs_cnf(cnf, maxnum, max_disp_len, max_clause_freq, max_diff_tokens);
        print_search_docs_result(result);
    }
    {
        cout << "search_docs, CNF query" << endl;
        vector<vector<vector<U16>>> cnf = {
            {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
            {{6483, 6509}}, // deep learning
        };
        auto result = engine.search_docs_cnf(cnf, maxnum, max_disp_len, max_clause_freq, max_diff_tokens);
        print_search_docs_result(result);
    }

    return 0;
}
