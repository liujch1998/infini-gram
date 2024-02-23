#include "infini_gram.h"

int main() {
    srand(time(0));
    std::mt19937 gen(19260817);
    std::uniform_int_distribution<U16> dis(0, 30000);

    const Consts consts = {
        .MAX_CNT_FOR_NTD = 1000,
        .MAX_OUTPUT_DOC_TOKENS = 5000,
        .MAX_CLAUSE_FREQ_PER_SHARD = 50000,
        .MAX_DIFF_TOKENS = 100,
        .MAX_DIFF_BYTES = sizeof(U16) * 100,
    };

    auto ngram_lm = NGramLanguageModeling("/large_experiments/cmd/ngram_datastore/redpajama_bff2_v3_c8_llama2", 2, 1, 2, 3, consts);

    vector<U16> input_ids = {5613, 4086, 9068};
    // vector<U16> prompt_ids = {dis(gen), dis(gen), dis(gen), dis(gen), dis(gen)};
    // for (auto token_id : prompt_ids) cout << token_id << " "; cout << endl;
    for (auto id : input_ids) cerr << id << " "; cerr << endl;
    auto start_time = chrono::high_resolution_clock::now();
    auto find_result = ngram_lm.find(input_ids);
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "time: " << duration.count() << " ms" << endl;
    cerr << "cnt: " << find_result.cnt << endl;
    cerr << endl;

    vector<U16> prompt_ids = {5613, 4086};
    U16 cont_id = 9068;
    start_time = chrono::high_resolution_clock::now();
    auto prob_result = ngram_lm.prob(prompt_ids, cont_id);
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "time: " << duration.count() << " ms" << endl;
    cerr << "prob: " << prob_result.prob << endl;
    cerr << endl;

    start_time = chrono::high_resolution_clock::now();
    auto ntd_result = ngram_lm.ntd(prompt_ids);
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "time: " << duration.count() << " ms" << endl;
    cerr << "ntd prompt_cnt: " << ntd_result.prompt_cnt << endl;
    cerr << endl;

    prompt_ids = {dis(gen), dis(gen), dis(gen), dis(gen), 5613, 4086};
    start_time = chrono::high_resolution_clock::now();
    auto infgram_prob_result = ngram_lm.infgram_prob(prompt_ids, cont_id);
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "time: " << duration.count() << " ms" << endl;
    cerr << "infgram_prob: " << infgram_prob_result.prob << endl;
    cerr << endl;

    start_time = chrono::high_resolution_clock::now();
    auto infgram_ntd_result = ngram_lm.infgram_ntd(prompt_ids);
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "time: " << duration.count() << " ms" << endl;
    cerr << "infgram_ntd prompt_cnt: " << infgram_ntd_result.prompt_cnt << endl;
    cerr << endl;

    // vector<vector<vector<U16>>> cnf = {
    //     {{dis(gen), dis(gen), dis(gen), dis(gen)}}, // some random stuff
    // };
    // vector<vector<vector<U16>>> cnf = {
    //     {{5613, 4086, 9068}}, // natural language processing
    // };
    // vector<vector<vector<U16>>> cnf = {
    //     {{5613, 4086, 9068}, {23116, 21082}}, // natural language processing OR artificial intelligence
    // };
    vector<vector<vector<U16>>> cnf = {
        {{5613, 4086, 9068}}, // natural language processing
        {{6483, 6509}}, // deep learning
    };
    size_t maxnum = 1;
    start_time = chrono::high_resolution_clock::now();
    auto search_docs_result = ngram_lm.search_docs(cnf, maxnum);
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "time: " << duration.count() << " ms" << endl;
    cerr << "search_docs cnt: " << search_docs_result.cnt << endl;
    // if (search_docs_result.cnt != 0) {
    //     for (auto d = 0; d < maxnum; d++) {
    //         auto &doc = search_docs_result.documents[d];
    //         cerr << "doc " << d << " idx: " << search_docs_result.idxs[d] << " token_offset: " << doc.token_offset << endl;
    //         cerr << "token_ids: ";
    //         for (auto token_id : doc.token_ids) {
    //             cerr << token_id << " ";
    //         }
    //         cerr << endl;
    //     }
    // }
    cerr << endl;

    return 0;
}
