#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include "json.hpp" // Include the JSON library
#include "infini_gram.h"

using namespace std;
using json = nlohmann::json;

const int MAX_QUEUE = 42;
const size_t SOCKET_IN_BUFFER_SIZE = 4096;
const size_t SOCKET_OUT_BUFFER_SIZE = 65536;

void send_response(int socket, const json& response) {
    string response_str = response.dump();
    if (response_str.size() >= SOCKET_OUT_BUFFER_SIZE) {
        string error = "[C++] Response too large: " + to_string(response_str.size()) + " bytes";
        cout << error << endl;
        json error_response = {{"error", error}};
        response_str = error_response.dump();
    }
    char out_buffer[SOCKET_OUT_BUFFER_SIZE] = {0};
    strcpy(out_buffer, response_str.c_str());
    char out_buffer_print[128] = {0};
    strncpy(out_buffer_print, out_buffer, 127);
    cout << "Response (" << response_str.size() << " bytes): " << out_buffer_print << " ..." << endl;
    size_t bytes_sent = 0;
    while (bytes_sent < SOCKET_OUT_BUFFER_SIZE) {
        auto ret = send(socket, out_buffer + bytes_sent, SOCKET_OUT_BUFFER_SIZE - bytes_sent, 0);
        if (ret == -1) {
            cout << "[C++] Socket send failed" << endl;
            break;
        }
        bytes_sent += ret;
    }
}

int main(int argc, char const *argv[]) {
    cout << endl;
    cout << "C++ engine rebooted" << endl;

    if (argc != 6) {
        cout << "Usage: " << argv[0] << " <PORT> <MAX_CNT_FOR_NTD> <MAX_OUTPUT_DOC_TOKENS> <MAX_CLAUSE_FREQ_PER_SHARD> <MAX_DIFF_TOKENS>" << endl;
        return 1;
    }
    int PORT = std::stoi(argv[1]);

    const Consts consts = {
        .MAX_CNT_FOR_NTD = (U64)std::stoi(argv[2]),
        .MAX_OUTPUT_DOC_TOKENS = (U64)std::stoi(argv[3]),
        .MAX_CLAUSE_FREQ_PER_SHARD = (U64)std::stoi(argv[4]),
        .MAX_DIFF_TOKENS = (U64)std::stoi(argv[5]),
        .MAX_DIFF_BYTES = sizeof(U16) * (U64)std::stoi(argv[5]),
    };

    map<string, shared_ptr<const NGramLanguageModeling>> LM_BY_CORPUS;
    ifstream f("indexes.json");
    const json configs = json::parse(f);
    f.close();
    for (const auto& config : configs) {
        const string corpus = config["corpus"];
        const U16 eos_token_id = (U16)config["eos_token_id"];
        const size_t ds_prefetch_depth = config.contains("ds_prefetch_depth") ? (size_t)config["ds_prefetch_depth"] : 1;
        const size_t sa_prefetch_depth = config.contains("sa_prefetch_depth") ? (size_t)config["sa_prefetch_depth"] : 3;
        const size_t od_prefetch_depth = config.contains("od_prefetch_depth") ? (size_t)config["od_prefetch_depth"] : 3;
        unique_ptr<NGramLanguageModeling> lm;
        if (config.contains("dirs")) {
            lm = make_unique<NGramLanguageModelingUnion>(config["dirs"], eos_token_id, ds_prefetch_depth, sa_prefetch_depth, od_prefetch_depth, consts);
        } else {
            lm = make_unique<NGramLanguageModeling>(config["dir"], eos_token_id, ds_prefetch_depth, sa_prefetch_depth, od_prefetch_depth, consts);
        }
        LM_BY_CORPUS.insert(make_pair(corpus, std::move(lm)));
    }

    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        cerr << "Failed to create socket fd" << endl;
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        cerr << "Failed to setsockopt" << endl;
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind the socket to the address and port number
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) {
        cerr << "Failed to bind port" << endl;
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    if (listen(server_fd, MAX_QUEUE) < 0) {
        cerr << "Failed to listen for connections" << endl;
        exit(EXIT_FAILURE);
    }

    // Accept a connection
    new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
    if (new_socket < 0) {
        cerr << "Failed to accept" << endl;
    }

    while (true) {
        char in_buffer[SOCKET_IN_BUFFER_SIZE] = {0};
        // keep reading until the buffer is full
        size_t bytes_read = 0;
        while (bytes_read < SOCKET_IN_BUFFER_SIZE) {
            auto ret = recv(new_socket, in_buffer + bytes_read, SOCKET_IN_BUFFER_SIZE - bytes_read, 0);
            if (ret == -1) {
                break;
            }
            bytes_read += ret;
        }
        if (bytes_read < SOCKET_IN_BUFFER_SIZE) {
            cout << "[C++] Socket read failed" << endl;
            continue;
        }
        // auto ret = read(new_socket, in_buffer, SOCKET_IN_BUFFER_SIZE);
        // if (ret != SOCKET_IN_BUFFER_SIZE) {
        //     cout << "[C++] Socket read failed" << endl;
        //     continue;
        // }
        cout << "Request (" << strlen(in_buffer) << " bytes): " << in_buffer << endl;
        json request, response;
        string query_type, corpus;
        vector<U16> input_ids;
        vector<vector<vector<U16>>> cnf;
        size_t maxnum;
        try {
            request = json::parse(in_buffer);
            query_type = request["query_type"];
            corpus = request["corpus"];
            if (query_type == "search_docs") {
                cnf = request["cnf"].get<vector<vector<vector<U16>>>>();
                maxnum = request["maxnum"];
            } else {
                input_ids = request["input_ids"].get<vector<U16>>();
            }
        } catch (json::parse_error& e) {
            string error = "[C++] Failed to parse request JSON: " + string(e.what());
            cout << error << endl;
            response = {{"error", error}};
            send_response(new_socket, response);
            continue;
        }

        shared_ptr<const NGramLanguageModeling> lm;
        try {
            lm = LM_BY_CORPUS.at(corpus);
        } catch (out_of_range& e) {
            string error = "[C++] Invalid corpus: " + corpus;
            cout << error << endl;
            response = {{"error", error}};
            send_response(new_socket, response);
            continue;
        }

        try {
            auto start_time = chrono::high_resolution_clock::now();
            if (query_type == "count") {
                FindResult find_result = lm->find(input_ids);
                response = {
                    {"cnt", find_result.cnt},
                    {"segments", find_result.segment_by_shard},
                };
            } else if (query_type == "prob") {
                vector<U16> prompt_ids = {input_ids.begin(), input_ids.end() - 1};
                U16 cont_id = input_ids.back();
                ProbResult prob_result = lm->prob(prompt_ids, cont_id);
                response = {
                    {"prob", prob_result.prob},
                    {"prompt_cnt", prob_result.prompt_cnt},
                    {"cont_cnt", prob_result.cont_cnt},
                };
            } else if (query_type == "ntd") {
                DistResult dist_result = lm->ntd(input_ids);
                response = {
                    {"prompt_cnt", dist_result.prompt_cnt},
                    {"freq_by_token_id", dist_result.freq_by_token_id},
                    {"prob_by_token_id", dist_result.prob_by_token_id},
                };
            } else if (query_type == "infgram_prob") {
                vector<U16> prompt_ids = {input_ids.begin(), input_ids.end() - 1};
                U16 cont_id = input_ids.back();
                InfgramProbResult infgram_prob_result = lm->infgram_prob(prompt_ids, cont_id);
                response = {
                    {"lfn", infgram_prob_result.lfn},
                    {"prob", infgram_prob_result.prob},
                    {"prompt_cnt", infgram_prob_result.prompt_cnt},
                    {"cont_cnt", infgram_prob_result.cont_cnt},
                };
            } else if (query_type == "infgram_ntd") {
                InfgramDistResult infgram_ntd_result = lm->infgram_ntd(input_ids);
                response = {
                    {"lfn", infgram_ntd_result.lfn},
                    {"prompt_cnt", infgram_ntd_result.prompt_cnt},
                    {"freq_by_token_id", infgram_ntd_result.freq_by_token_id},
                    {"prob_by_token_id", infgram_ntd_result.prob_by_token_id},
                };
            } else if (query_type == "search_docs") {
                SearchDocsResult search_docs_result = lm->search_docs(cnf, maxnum);
                if (search_docs_result.cnt == 0) {
                    string error;
                    if (cnf.size() == 1) {
                        if (cnf[0].size() == 1) {
                            error = "The query term is not found in the corpus!";
                        } else {
                            error = "None of the query terms is found in the corpus!";
                        }
                    } else {
                        auto &cnt_by_clause = search_docs_result.cnt_by_clause;
                        auto it = find(cnt_by_clause.begin(), cnt_by_clause.end(), 0);
                        if (it != cnt_by_clause.end()) {
                            error = "None of the terms in the " + to_string(it - cnt_by_clause.begin()) + "-th disjunctive clause is found in the corpus!";
                        } else {
                            error = "Query is not found in the corpus! Try relaxing the constraints.";
                        }
                    }
                    response = {{"error", error}};
                } else {
                    vector<json> documents;
                    for (auto &d : search_docs_result.documents) {
                        json document = {
                            {"token_ids", d.token_ids},
                            {"token_offset", d.token_offset},
                            {"doc_ix", d.doc_ix},
                            {"doc_len", d.doc_len},
                            {"disp_len", d.disp_len},
                        };
                        documents.push_back(document);
                    }
                    response = {
                        {"documents", documents},
                        {"idxs", search_docs_result.idxs},
                        {"cnt_by_clause", search_docs_result.cnt_by_clause},
                        {"cnt", search_docs_result.cnt},
                        {"approx", search_docs_result.approx},
                    };
                }
            } else {
                response = {{"error", "[C++] Invalid query_type: " + query_type}};
            }
            auto end_time = chrono::high_resolution_clock::now();
            auto latency = (double)chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000;
            response["latency"] = latency;
            send_response(new_socket, response);
        } catch (exception& e) {
            string error = "[C++] Exception: " + string(e.what());
            cout << error << endl;
            response = {{"error", error}};
            send_response(new_socket, response);
        }
    }

    // Close the socket
    close(server_fd);
    return 0;
}
