#include <cassert>
#include <cstdint> // for uint64_t
#include <cstring> // for memcpy
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/mman.h> // for mmap, munmap
#include <sys/stat.h> // for struct stat
#include <fcntl.h> // for O_RDONLY
#include <unistd.h> // for close
#include <algorithm>
#include <random>
#include <thread>
#include <fstream>

#define U64 uint64_t
#define U32 uint32_t
#define U16 uint16_t
#define U8 uint8_t

using namespace std;
namespace fs = std::filesystem;

void assert_little_endian() {
    unsigned int i = 1;
    char *c = (char*)&i;
    assert (*c);
}
const auto PAGESIZE = sysconf(_SC_PAGESIZE);

struct DatastoreShard {
    U8* ds;
    U8* sa;
    U64 tok_cnt;
    U64 ds_size;
    U8 ptr_size;
    U8* od;
    U64 doc_cnt;
    U8* mt;
    U64 mt_size;
    U8* om;
};
struct FindResult {
    U64 cnt;
    vector<pair<U64, U64>> segment_by_shard;
};
struct FindDisjResult {
    U64 cnt;
    vector<U64> cnt_by_shard;
    vector<vector<pair<U64, U64>>> segment_by_term_by_shard;
    vector<double> subsampling_factor_by_shard;
};
struct FindCnfResult {
    U64 cnt;
    bool approx;
    vector<vector<U64>> ptrs_by_shard;
};
struct CountResult {
    U64 count;
    bool approx;
};
struct ProbResult {
    U64 prompt_cnt;
    U64 cont_cnt;
    double prob;
};
struct DistTokenResult {
    U64 cont_cnt;
    double prob;
};
struct DistResult {
    U64 prompt_cnt;
    map<U16, DistTokenResult> result_by_token_id;
    bool approx;
};
struct InfgramProbResult {
    U64 prompt_cnt;
    U64 cont_cnt;
    double prob;
    U64 suffix_len;
};
struct InfgramDistResult {
    U64 prompt_cnt;
    map<U16, DistTokenResult> result_by_token_id;
    bool approx;
    U64 suffix_len;
};
struct DocResult {
    U64 doc_ix;
    U64 doc_len;
    U64 disp_len;
    string metadata;
    vector<U16> token_ids;
};
struct SearchDocsResult {
    U64 cnt;
    bool approx;
    vector<U64> idxs;
    vector<DocResult> docs;
};

class InfiniGramEngine {

public:

    InfiniGramEngine(
        const vector<string> index_dirs, const U16 eos_token_id,
        const bool load_to_ram = false,
        const size_t ds_prefetch_depth = 1, const size_t sa_prefetch_depth = 3, const size_t od_prefetch_depth = 3)
        : _eos_token_id(eos_token_id), _load_to_ram(load_to_ram), _ds_prefetch_depth(ds_prefetch_depth), _sa_prefetch_depth(sa_prefetch_depth), _od_prefetch_depth(od_prefetch_depth) {

        assert_little_endian();

        for (const auto &index_dir : index_dirs) {
            assert (fs::exists(index_dir));

            vector<string> ds_paths, sa_paths, od_paths, mt_paths, om_paths;
            for (const auto & entry : fs::directory_iterator(index_dir)) {
                if (entry.path().string().find("tokenized") != string::npos) {
                    ds_paths.push_back(entry.path());
                } else if (entry.path().string().find("table") != string::npos) {
                    sa_paths.push_back(entry.path());
                } else if (entry.path().string().find("offset") != string::npos) {
                    od_paths.push_back(entry.path());
                } else if (entry.path().string().find("metadata") != string::npos) {
                    mt_paths.push_back(entry.path());
                } else if (entry.path().string().find("metaoff") != string::npos) {
                    om_paths.push_back(entry.path());
                }
            }
            sort(ds_paths.begin(), ds_paths.end());
            sort(sa_paths.begin(), sa_paths.end());
            sort(od_paths.begin(), od_paths.end());
            sort(mt_paths.begin(), mt_paths.end());
            sort(om_paths.begin(), om_paths.end());
            assert (sa_paths.size() == ds_paths.size());
            assert (od_paths.size() == ds_paths.size());
            assert (mt_paths.size() == 0 || mt_paths.size() == ds_paths.size());
            assert (om_paths.size() == mt_paths.size());

            for (auto s = 0; s < ds_paths.size(); s++) {
                auto [ds, ds_size] = load_file(ds_paths[s]);
                auto [sa, sa_size] = load_file(sa_paths[s]);
                auto [od, od_size] = load_file(od_paths[s]);

                assert (ds_size % sizeof(U16) == 0);
                U64 tok_cnt = ds_size / sizeof(U16);
                assert (sa_size % tok_cnt == 0);
                U8 ptr_size = (U8)(sa_size / tok_cnt);
                assert (od_size % sizeof(U64) == 0);
                U64 doc_cnt = od_size / sizeof(U64);

                if (mt_paths.size() == 0) {
                    auto shard = DatastoreShard{ds, sa, tok_cnt, ds_size, ptr_size, od, doc_cnt};
                    _shards.push_back(shard);
                } else {
                    auto [mt, mt_size] = load_file(mt_paths[s]);
                    auto [om, om_size] = load_file(om_paths[s]);

                    assert (om_size == doc_cnt * sizeof(U64));

                    auto shard = DatastoreShard{ds, sa, tok_cnt, ds_size, ptr_size, od, doc_cnt, mt, mt_size, om};
                    _shards.push_back(shard);
                }
            }
        }

        _num_shards = _shards.size();
        assert (_num_shards > 0);
    }

    ~InfiniGramEngine() {
        for (auto shard : _shards) {
            unload_file(shard.ds, shard.ds_size);
            unload_file(shard.sa, shard.tok_cnt * shard.ptr_size);
            unload_file(shard.od, shard.doc_cnt * sizeof(U64));
            if (shard.mt) {
                unload_file(shard.mt, shard.mt_size);
                unload_file(shard.om, shard.doc_cnt * sizeof(U64));
            }
        }
    }

    pair<U8*, U64> load_file(const string &path) {
        if (_load_to_ram) {
            ifstream f(path, ios::binary);
            assert (f.is_open());
            f.seekg(0, ios::end);
            U64 size = f.tellg();
            f.seekg(0, ios::beg);
            U8 *buf = new U8[size];
            f.read(reinterpret_cast<char*>(buf), size);
            f.close();
            return {buf, size};
        } else {
            int f = open(path.c_str(), O_RDONLY);
            assert (f != -1);
            struct stat s;
            auto fstat_ret = fstat(f, &s);
            assert (fstat_ret != -1);
            U8 *ptr = static_cast<U8*>(mmap(NULL, s.st_size, PROT_READ, MAP_PRIVATE, f, 0));
            assert (ptr != MAP_FAILED);
            madvise(ptr, s.st_size, MADV_RANDOM);
            close(f);
            return {ptr, s.st_size};
        }
    }

    void unload_file(U8* ptr, U64 size) {
        if (_load_to_ram) {
            delete[] ptr;
        } else {
            munmap(ptr, size);
        }
    }

    FindResult find(const vector<U16> &input_ids) const {

        vector<pair<U64, U64>> hint_segment_by_shard;
        for (const auto &shard : _shards) {
            hint_segment_by_shard.push_back({0, shard.tok_cnt});
        }
        return _find(input_ids, hint_segment_by_shard);
    }

    FindResult _find(const vector<U16> &input_ids, const vector<pair<U64, U64>> &hint_segment_by_shard) const {

        assert (hint_segment_by_shard.size() == _num_shards);

        const U8 *input_buf = reinterpret_cast<const U8*>(input_ids.data());
        U64 num_bytes = input_ids.size() * sizeof(U16);

        vector<U64> start_thread_outputs(_num_shards);
        vector<U64> end_thread_outputs(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.emplace_back(&InfiniGramEngine::_find_thread, this, s,
                &input_ids, input_buf, num_bytes, hint_segment_by_shard[s], true, &start_thread_outputs[s]);
            threads.emplace_back(&InfiniGramEngine::_find_thread, this, s,
                &input_ids, input_buf, num_bytes, hint_segment_by_shard[s], false, &end_thread_outputs[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        U64 cnt = 0;
        vector<pair<U64, U64>> segment_by_shard;
        for (auto s = 0; s < _num_shards; s++) {
            assert (start_thread_outputs[s] <= end_thread_outputs[s]);
            cnt += end_thread_outputs[s] - start_thread_outputs[s];
            segment_by_shard.push_back({start_thread_outputs[s], end_thread_outputs[s]});
        }

        return FindResult{ .cnt = cnt, .segment_by_shard = segment_by_shard, };
    }

    void _find_thread(
        const size_t s,
        const vector<U16>* const input_ids,
        const U8* const input_buf,
        const U64 num_bytes,
        const pair<U64, U64> hint_segment,
        const bool finding_start,
        U64* const thread_output) const {

        const auto &shard = _shards[s];

        if (input_ids->empty()) {
            *thread_output = finding_start ? 0 : shard.tok_cnt;
            return;
        }

        U64 lo = hint_segment.first - 1, hi = hint_segment.second;

        if (finding_start) { // Search for the leftmost sa index that IS >= the prompt
            while (hi - lo > 1) {
                _prefetch_find(shard, num_bytes, lo, hi);
                U64 mi = (lo + hi) >> 1;
                U64 ptr = _convert_rank_to_ptr(shard, mi);
                bool less = std::lexicographical_compare(
                    shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size),
                    input_buf, input_buf + num_bytes);
                if (less) {
                    lo = mi;
                } else {
                    hi = mi;
                }
            }
        } else { // Search for the leftmost sa index that IS > the prompt
            while (hi - lo > 1) {
                _prefetch_find(shard, num_bytes, lo, hi);
                U64 mi = (lo + hi) >> 1;
                U64 ptr = _convert_rank_to_ptr(shard, mi);
                bool less = std::lexicographical_compare(
                    input_buf, input_buf + num_bytes,
                    shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size));
                if (!less) {
                    lo = mi;
                } else {
                    hi = mi;
                }
            }
        }
        *thread_output = hi;
    }

    FindDisjResult find_disj(const vector<vector<U16>> &disj_clause, const U64 max_clause_freq) const {

        vector<FindResult> find_result_by_term;
        for (const auto &term : disj_clause) {
            find_result_by_term.push_back(find(term));
        }
        vector<U64> cnt_by_shard(_num_shards);
        vector<vector<pair<U64, U64>>> segment_by_term_by_shard(_num_shards);
        vector<double> subsampling_factor_by_shard(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.emplace_back(&InfiniGramEngine::_find_disj_thread, this, s,
                &find_result_by_term, max_clause_freq / _num_shards, &cnt_by_shard[s], &segment_by_term_by_shard[s], &subsampling_factor_by_shard[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        U64 cnt = std::accumulate(cnt_by_shard.begin(), cnt_by_shard.end(), (U64)0);
        return FindDisjResult{
            .cnt = cnt,
            .cnt_by_shard = cnt_by_shard,
            .segment_by_term_by_shard = segment_by_term_by_shard,
            .subsampling_factor_by_shard = subsampling_factor_by_shard,
        };
    }

    void _find_disj_thread(
        const size_t s,
        const vector<FindResult>* const find_result_by_term,
        const U64 max_clause_freq_per_shard,
        U64* out_cnt,
        vector<pair<U64, U64>>* out_segment_by_term,
        double* out_subsampling_factor) const {

        std::mt19937 gen(19260817);

        U64 cnt = 0;
        vector<pair<U64, U64>> segment_by_term;
        for (const auto &find_result : *find_result_by_term) {
            const auto &segment = find_result.segment_by_shard[s];
            segment_by_term.push_back(segment);
            cnt += segment.second - segment.first;
        }
        double subsampling_factor = 1.0;
        if (cnt > max_clause_freq_per_shard) {
            // TODO: This subsampling might not be uniform
            U64 new_cnt = 0;
            vector<pair<U64, U64>> new_segment_by_term;
            for (const auto &[start, end] : segment_by_term) {
                U64 length = end - start;
                U64 new_length = (U64)((length * max_clause_freq_per_shard + (cnt - 1)) / cnt); // ceil div
                std::uniform_int_distribution<U64> dis(0, length - new_length); // left inclusive, right inclusive
                U64 new_start = start + dis(gen);
                U64 new_end = new_start + new_length;
                new_cnt += new_length;
                new_segment_by_term.push_back({new_start, new_end});
            }
            assert (new_cnt > 0);
            assert (new_cnt <= cnt);
            subsampling_factor = cnt / new_cnt;
            segment_by_term = new_segment_by_term;
        }
        *out_cnt = cnt;
        *out_segment_by_term = segment_by_term;
        *out_subsampling_factor = subsampling_factor;
    }

    FindCnfResult find_cnf(const vector<vector<vector<U16>>> &cnf, const U64 max_clause_freq, const U64 max_diff_tokens) const {

        assert (cnf.size() > 0);

        vector<FindDisjResult> find_disj_result_by_clause;
        for (const auto &clause : cnf) {
            auto find_disj_result = find_disj(clause, max_clause_freq);
            find_disj_result_by_clause.push_back(find_disj_result);
        }
        for (const auto &find_disj_result : find_disj_result_by_clause) {
            if (find_disj_result.cnt == 0) {
                return FindCnfResult{ .cnt = 0, .approx = false, .ptrs_by_shard = {} };
            }
        }

        vector<U64> cnt_by_shard(_num_shards);
        vector<vector<pair<U64, U64>>> valid_ptr_ranges_by_shard(_num_shards);
        vector<double> subsampling_factor_by_shard(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.emplace_back(&InfiniGramEngine::_find_cnf_thread, this, s,
                &find_disj_result_by_clause, max_diff_tokens, &cnt_by_shard[s], &valid_ptr_ranges_by_shard[s], &subsampling_factor_by_shard[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        U64 cnt = accumulate(cnt_by_shard.begin(), cnt_by_shard.end(), (U64)0);
        bool approx = any_of(subsampling_factor_by_shard.begin(), subsampling_factor_by_shard.end(), [](double f) { return f != 1.0; });
        vector<vector<U64>> ptrs_by_shard(_num_shards);
        for (auto s = 0; s < _num_shards; s++) {
            for (const auto &[l, r] : valid_ptr_ranges_by_shard[s]) {
                ptrs_by_shard[s].push_back(l);
            }
        }
        return FindCnfResult{ .cnt = cnt, .approx = approx, .ptrs_by_shard = ptrs_by_shard, };
    }

    void _find_cnf_thread(
        const size_t s,
        const vector<FindDisjResult>* const _find_disj_result_by_clause,
        const U64 max_diff_tokens,
        U64* out_cnt,
        vector<pair<U64, U64>>* out_valid_ptr_ranges,
        double* out_subsampling_factor) const {

        // sort find_disj_result_by_clause by cnt in of this shard in increasing order
        vector<FindDisjResult> find_disj_result_by_clause = *_find_disj_result_by_clause;
        sort(find_disj_result_by_clause.begin(), find_disj_result_by_clause.end(), [s](const FindDisjResult &a, const FindDisjResult &b) {
            return a.cnt_by_shard[s] < b.cnt_by_shard[s];
        });

        const DatastoreShard &shard = _shards[s];
        auto &find_disj_result = find_disj_result_by_clause[0];
        vector<pair<U64, U64>> valid_ptr_ranges;
        for (const auto &[start, end] : find_disj_result.segment_by_term_by_shard[s]) {
            vector<U64> ptrs = _convert_ranks_to_ptrs(shard, start, end);
            for (const auto ptr : ptrs) {
                valid_ptr_ranges.push_back({ptr, ptr});
            }
        }
        double subsampling_factor = find_disj_result.subsampling_factor_by_shard[s];

        // maintain valid ptr ranges
        // if there are Q terms and each term has M matches in the shard, the complexity is O(Q * M * log(M))
        for (auto d = 1; d < find_disj_result_by_clause.size(); d++) {
            auto &find_disj_result = find_disj_result_by_clause[d];
            vector<U64> ptrs;
            for (const auto &[start, end] : find_disj_result.segment_by_term_by_shard[s]) {
                vector<U64> new_ptrs = _convert_ranks_to_ptrs(shard, start, end);
                ptrs.insert(ptrs.end(), new_ptrs.begin(), new_ptrs.end());
            }
            sort(ptrs.begin(), ptrs.end());
            vector<pair<U64, U64>> new_valid_ptr_ranges;
            for (const auto& [l, r] : valid_ptr_ranges) {
                auto lo = _bin_search(ptrs, r).first; // find the last match that is < r
                U64 new_l = lo == (U64)-1 ? -1 : min(l, ptrs[lo]);
                auto hi = _bin_search(ptrs, l).second; // find the first match that is >= l
                U64 new_r = hi == ptrs.size() ? -1 : max(r, ptrs[hi]);
                if (new_l != (U64)-1 && new_l + max_diff_tokens * sizeof(U16) >= l && new_r != (U64)-1 && new_r <= r + max_diff_tokens * sizeof(U16)) { // +- MAX_DIFF_TOKENS tokens
                    new_valid_ptr_ranges.push_back({new_l, new_r});
                } else {
                    if (new_l != (U64)-1 && new_l + max_diff_tokens * sizeof(U16) >= l) {
                        new_valid_ptr_ranges.push_back({new_l, l});
                    }
                    if (new_r != (U64)-1 && new_r <= r + max_diff_tokens * sizeof(U16)) {
                        new_valid_ptr_ranges.push_back({r, new_r});
                    }
                }
            }
            valid_ptr_ranges = new_valid_ptr_ranges;
            subsampling_factor *= find_disj_result.subsampling_factor_by_shard[s];
        }

        // remove ptr ranges that cross document boundary
        vector<pair<U64, U64>> new_valid_ptr_ranges;
        const vector<U8> doc_sep = {0xff, 0xff};
        for (const auto& [l, r] : valid_ptr_ranges) {
            auto it = search(shard.ds + l, shard.ds + r, doc_sep.begin(), doc_sep.end());
            if (it == shard.ds + r) {
                new_valid_ptr_ranges.push_back({l, r});
            }
        }
        valid_ptr_ranges = new_valid_ptr_ranges;

        U64 cnt = (U64)(valid_ptr_ranges.size() * subsampling_factor);
        *out_cnt = cnt;
        *out_valid_ptr_ranges = valid_ptr_ranges;
        *out_subsampling_factor = subsampling_factor;
    }

    CountResult count(const vector<U16> &input_ids) const {
        auto find_result = find(input_ids);
        return CountResult{ .count = find_result.cnt, .approx = false, };
    }

    CountResult count_cnf(const vector<vector<vector<U16>>> &cnf, const U64 max_clause_freq, const U64 max_diff_tokens) const {
        auto find_cnf_result = find_cnf(cnf, max_clause_freq, max_diff_tokens);
        return CountResult{ .count = find_cnf_result.cnt, .approx = find_cnf_result.approx, };
    }

    ProbResult prob(const vector<U16> &prompt_ids, const U16 cont_id) const {

        auto prompt_find_result = find(prompt_ids);
        U64 prompt_cnt = prompt_find_result.cnt;
        if (prompt_cnt == 0) {
            return ProbResult{ .prompt_cnt = 0, .cont_cnt = 0, .prob = -1.0 };
        }
        vector<U16> input_ids = {prompt_ids.begin(), prompt_ids.end()};
        input_ids.push_back(cont_id);
        auto cont_find_result = _find(input_ids, prompt_find_result.segment_by_shard);
        U64 cont_cnt = cont_find_result.cnt;
        double prob = (double)cont_cnt / prompt_cnt;

        return ProbResult{ .prompt_cnt = prompt_cnt, .cont_cnt = cont_cnt, .prob = prob };
    }

    DistResult ntd(const vector<U16> &prompt_ids, const U64 max_support) const {

        auto prompt_find_result = find(prompt_ids);
        if (prompt_find_result.cnt == 0) {
            return DistResult{ .prompt_cnt = 0, .result_by_token_id = {}, .approx = false, };
        }
        U64 unit = 1;
        while (prompt_find_result.cnt > unit * max_support) {
            unit <<= 1;
        }
        bool approx = (unit > 1);

        vector<map<U16, U64>> freq_by_token_id_by_shard(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.emplace_back(&InfiniGramEngine::_get_freq_by_token_id_approx, this,
                s, prompt_ids.size() * sizeof(U16), prompt_find_result.segment_by_shard[s], unit, nullptr, nullptr, &freq_by_token_id_by_shard[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        map<U16, U64> freq_by_token_id = {};
        for (auto s = 0; s < _num_shards; s++) {
            for (auto& [token_id, freq] : freq_by_token_id_by_shard[s]) {
                freq_by_token_id[token_id] += freq;
            }
        }

        U64 prompt_cnt = 0;
        for (auto& [token_id, freq] : freq_by_token_id) {
            prompt_cnt += freq;
        }
        assert (prompt_cnt == prompt_find_result.cnt);
        map<U16, DistTokenResult> result_by_token_id = {};
        for (auto& [token_id, freq] : freq_by_token_id) {
            result_by_token_id[token_id] = DistTokenResult{freq, (double)freq / prompt_cnt};
        }

        return DistResult{ .prompt_cnt = prompt_cnt, .result_by_token_id = result_by_token_id, .approx = approx, };
    }

    void _get_freq_by_token_id_approx(
        const size_t s,
        const U64 num_bytes,
        const pair<U64, U64> segment,
        const U64 unit,
        const U16* const token_start,
        const U16* const token_end,
        map<U16, U64>* const out_freq_by_token_id) const {

        const auto& shard = _shards[s];
        U64 start = segment.first, end = segment.second;

        _prefetch_ntd(shard, num_bytes, start, end);

        if (end - start < 4 * unit) {
            for (U64 rank = start; rank < end; rank += unit) {
                U64 rank_mid = (rank + unit <= end) ? (rank + (unit >> 1)) : ((rank + end) >> 1);
                U64 ptr = _convert_rank_to_ptr(shard, rank_mid);
                U16 token_id = _convert_ptr_to_token_id(shard, ptr + num_bytes);
                (*out_freq_by_token_id)[token_id] += (rank + unit <= end) ? unit : (end - rank);
            }
            return;
        }

        // If start and end-1 has the same token, then we know this segment is all the same token
        U16 new_token_start = 0, new_token_end = 0;
        if (token_start) {
            new_token_start = *token_start;
        } else {
            U64 ptr_start = _convert_rank_to_ptr(shard, start);
            new_token_start = _convert_ptr_to_token_id(shard, ptr_start + num_bytes);
        }
        if (token_end) {
            new_token_end = *token_end;
        } else {
            U64 ptr_end = _convert_rank_to_ptr(shard, end - 1);
            new_token_end = _convert_ptr_to_token_id(shard, ptr_end + num_bytes);
        }
        if (new_token_start == new_token_end) {
            (*out_freq_by_token_id)[new_token_start] = end - start;
            return;
        }

        // Otherwise, we do divide and conquer
        U64 mi = (start + end) >> 1;
        pair<U64, U64> left_segment = {start, mi}, right_segment = {mi, end};
        map<U16, U64> left_thread_output = {}, right_thread_output = {};
        auto left_thread = thread(&InfiniGramEngine::_get_freq_by_token_id_approx, this,
            s, num_bytes, left_segment, unit, &new_token_start, nullptr, &left_thread_output);
        auto right_thread = thread(&InfiniGramEngine::_get_freq_by_token_id_approx, this,
            s, num_bytes, right_segment, unit, nullptr, &new_token_end, &right_thread_output);
        left_thread.join();
        right_thread.join();
        // TODO: This map merge is not efficient. Need to hack into the endianness of token_ids.
        for (auto& [token_id, freq] : left_thread_output) {
            (*out_freq_by_token_id)[token_id] += freq;
        }
        for (auto& [token_id, freq] : right_thread_output) {
            (*out_freq_by_token_id)[token_id] += freq;
        }
    }

    InfgramProbResult infgram_prob(const vector<U16> &prompt_ids, const U16 cont_id) const {

        size_t L = prompt_ids.size();
        // binary lifting
        size_t l_lo = 0, l_hi = 1;
        while (true) {
            if (l_hi > L) { l_hi = L + 1; break; }
            const vector<U16> prompt_suffix_ids{prompt_ids.begin() + L - l_hi, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) break;
            l_lo = l_hi;
            l_hi <<= 1;
        }
        // binary search within [l_lo, l_hi)
        while (l_hi - l_lo > 1) {
            size_t l_mid = (l_lo + l_hi) >> 1;
            const vector<U16> prompt_suffix_ids{prompt_ids.begin() + L - l_mid, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) {
                l_hi = l_mid;
            } else {
                l_lo = l_mid;
            }
        }

        size_t suffix_len = l_lo;
        const vector<U16> prompt_suffix_ids{prompt_ids.begin() + L - suffix_len, prompt_ids.end()};
        auto result = prob(prompt_suffix_ids, cont_id);

        return InfgramProbResult{
            .prompt_cnt = result.prompt_cnt,
            .cont_cnt = result.cont_cnt,
            .prob = result.prob,
            .suffix_len = suffix_len,
        };
    }

    InfgramDistResult infgram_ntd(const vector<U16> &prompt_ids, const U64 max_support) const {

        size_t L = prompt_ids.size();
        // binary lifting
        size_t l_lo = 0, l_hi = 1;
        while (true) {
            if (l_hi > L) { l_hi = L + 1; break; }
            const vector<U16> prompt_suffix_ids{prompt_ids.begin() + L - l_hi, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) break;
            l_lo = l_hi;
            l_hi <<= 1;
        }
        // binary search within [l_lo, l_hi)
        while (l_hi - l_lo > 1) {
            size_t l_mid = (l_lo + l_hi) >> 1;
            const vector<U16> prompt_suffix_ids{prompt_ids.begin() + L - l_mid, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) {
                l_hi = l_mid;
            } else {
                l_lo = l_mid;
            }
        }

        size_t suffix_len = l_lo;
        const vector<U16> prompt_suffix_ids{prompt_ids.begin() + L - suffix_len, prompt_ids.end()};
        auto result = ntd(prompt_suffix_ids, max_support);

        return InfgramDistResult{
            .prompt_cnt = result.prompt_cnt,
            .result_by_token_id = result.result_by_token_id,
            .approx = result.approx,
            .suffix_len = suffix_len,
        };
    }

    SearchDocsResult search_docs(const vector<U16> &input_ids, const size_t maxnum, const U64 max_disp_len) const {

        assert (maxnum > 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        auto find_result = find(input_ids);
        if (find_result.cnt == 0) {
            return SearchDocsResult{ .cnt = 0, .approx = false, .idxs = {}, .docs = {}, };
        }

        //sample up to maxnum documents
        auto &segment_by_shard = find_result.segment_by_shard;
        vector<U64> cnt_by_shard;
        for (const auto &segment : segment_by_shard) {
            cnt_by_shard.push_back(segment.second - segment.first);
        }
        vector<U64> idxs;
        vector<DocResult> docs;
        for (size_t d = 0; d < maxnum; d++) {
            size_t s = discrete_distribution<size_t>(cnt_by_shard.begin(), cnt_by_shard.end())(gen);
            auto &[start, end] = segment_by_shard[s];
            U64 rank = uniform_int_distribution<U64>(start, end - 1)(gen); // left inclusive, right inclusive
            U64 ptr = _convert_rank_to_ptr(_shards[s], rank);
            U64 idx = accumulate(cnt_by_shard.begin(), cnt_by_shard.begin() + s, (U64)0) + (rank - start);
            DocResult doc = get_doc_by_ptr(s, ptr, max_disp_len);
            idxs.push_back(idx);
            docs.push_back(doc);
        }

        return SearchDocsResult{ .cnt = find_result.cnt, .approx = false, .idxs = idxs, .docs = docs, };
    }

    SearchDocsResult search_docs_cnf(const vector<vector<vector<U16>>> &cnf, const size_t maxnum, const U64 max_disp_len, const U64 max_clause_freq, const U64 max_diff_tokens) const {

        assert (cnf.size() > 0);
        assert (maxnum > 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        auto find_cnf_result = find_cnf(cnf, max_clause_freq, max_diff_tokens);
        if (find_cnf_result.cnt == 0) {
            return SearchDocsResult{ .cnt = 0, .approx = false, .idxs = {}, .docs = {}, };
        }

        // sample up to maxnum documents
        auto &ptrs_by_shard = find_cnf_result.ptrs_by_shard;
        vector<U64> ptr_cnt_by_shard;
        for (const auto &ptrs : ptrs_by_shard) {
            ptr_cnt_by_shard.push_back(ptrs.size());
        }
        U64 ptr_cnt = accumulate(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.end(), (U64)0);
        vector<U64> idxs;
        vector<DocResult> docs;
        for (size_t d = 0; d < maxnum; d++) {
            size_t s = discrete_distribution<size_t>(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.end())(gen);
            auto &ptrs = ptrs_by_shard[s];
            U64 i = uniform_int_distribution<U64>(0, ptrs.size() - 1)(gen); // left inclusive, right inclusive
            auto &ptr = ptrs[i];
            double percentile = (double)(accumulate(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.begin() + s, (U64)0) + i) / ptr_cnt;
            U64 idx = (U64)(percentile * find_cnf_result.cnt);
            DocResult doc = get_doc_by_ptr(s, ptr, max_disp_len);
            idxs.push_back(idx);
            docs.push_back(doc);
        }

        return SearchDocsResult{ .cnt = find_cnf_result.cnt, .approx = find_cnf_result.approx, .idxs = idxs, .docs = docs, };
    }

    DocResult get_doc_by_rank(const size_t s, const U64 rank, const U64 max_disp_len) const {

        assert (s < _num_shards);
        const auto &shard = _shards[s];
        assert (rank < shard.tok_cnt);

        U64 ptr = _convert_rank_to_ptr(shard, rank);
        return get_doc_by_ptr(s, ptr, max_disp_len);
    }

    DocResult get_doc_by_ptr(const size_t s, const U64 ptr, const U64 max_disp_len) const {

        assert (s < _num_shards);
        const auto &shard = _shards[s];
        assert (ptr < shard.ds_size);

        U64 max_prepend_tokens = max_disp_len / 2;
        U64 max_append_tokens = (max_disp_len + 1) / 2;

        const vector<U8> doc_sep = {0xff, 0xff};
        U64 lo = 0, hi = shard.doc_cnt;
        while (hi - lo > 1) {
            _prefetch_doc(shard, lo, hi);
            U64 mi = (lo + hi) >> 1;
            U64 p = _convert_doc_ix_to_ptr(shard, mi);
            if (p <= ptr) {
                lo = mi;
            } else {
                hi = mi;
            }
        }

        U64 doc_ix = 0; for (auto _ = 0; _ < s; _++) doc_ix += _shards[_].doc_cnt; doc_ix += lo;

        U64 doc_start_ptr = _convert_doc_ix_to_ptr(shard, lo) + sizeof(U16); // +2 because we want to skip the document separator
        U64 doc_end_ptr = _convert_doc_ix_to_ptr(shard, lo + 1);
        U64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(U16);

        U64 disp_start_ptr = max(doc_start_ptr, ptr - sizeof(U16) * max_prepend_tokens);
        U64 disp_end_ptr = min(doc_end_ptr, ptr + sizeof(U16) * max_append_tokens);
        U64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(U16);

        string metadata = "";
        if (shard.mt) {
            U64 meta_start_ptr = _convert_doc_ix_to_meta_ptr(shard, lo);
            U64 meta_end_ptr = _convert_doc_ix_to_meta_ptr(shard, lo + 1);
            vector<U8> meta_chars(shard.mt + meta_start_ptr, shard.mt + meta_end_ptr);
            metadata = string(meta_chars.begin(), meta_chars.end());
        }

        vector<U16> token_ids(reinterpret_cast<U16*>(shard.ds + disp_start_ptr), reinterpret_cast<U16*>(shard.ds + disp_end_ptr));

        return DocResult{ .doc_ix = doc_ix, .doc_len = doc_len, .disp_len = disp_len, .metadata = metadata, .token_ids = token_ids, };
    }

private:

    void _prefetch_find(const DatastoreShard &shard, const U64 num_bytes, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.tok_cnt) return; // This might happen when lo = -1 and hi = 0
        if (depth == _ds_prefetch_depth) { // fetch ds
            U64 ptr = _convert_rank_to_ptr(shard, mi);
            madvise(shard.ds + ptr - ptr % PAGESIZE, num_bytes + ptr % PAGESIZE, MADV_WILLNEED);
        } else if (depth == _sa_prefetch_depth) { // fetch sa
            madvise(shard.sa + mi * shard.ptr_size - mi * shard.ptr_size % PAGESIZE, shard.ptr_size + mi * shard.ptr_size % PAGESIZE, MADV_WILLNEED);
            return;
        }
        _prefetch_find(shard, num_bytes, lo, mi, depth + 1);
        _prefetch_find(shard, num_bytes, mi, hi, depth + 1);
    }

    void _prefetch_ntd(const DatastoreShard &shard, const U64 num_bytes, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.tok_cnt) return; // This might happen when lo = -1 and hi = 0
        if (depth == _ds_prefetch_depth) { // fetch ds
            U64 ptr = _convert_rank_to_ptr(shard, mi-1) + num_bytes;
            madvise(shard.ds + ptr - ptr % PAGESIZE, sizeof(U16) + ptr % PAGESIZE, MADV_WILLNEED);
            ptr = _convert_rank_to_ptr(shard, mi) + num_bytes;
            madvise(shard.ds + ptr - ptr % PAGESIZE, sizeof(U16) + ptr % PAGESIZE, MADV_WILLNEED);
        } else if (depth == _sa_prefetch_depth) { // fetch sa
            madvise(shard.sa + (mi-1) * shard.ptr_size - (mi-1) * shard.ptr_size % PAGESIZE, 2 * shard.ptr_size + (mi-1) * shard.ptr_size % PAGESIZE, MADV_WILLNEED); // since we need both mi-1 and mi
            return;
        }
        _prefetch_ntd(shard, num_bytes, lo, mi, depth + 1);
        _prefetch_ntd(shard, num_bytes, mi, hi, depth + 1);
    }

    void _prefetch_doc(const DatastoreShard &shard, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.doc_cnt) return; // This might happen when lo = -1 and hi = 0
        if (depth == _od_prefetch_depth) { // fetch od
            madvise(shard.od + mi * sizeof(U64) - mi * sizeof(U64) % PAGESIZE, sizeof(U64) + mi * sizeof(U64) % PAGESIZE, MADV_WILLNEED);
            return;
        }
        _prefetch_doc(shard, lo, mi, depth + 1);
        _prefetch_doc(shard, mi, hi, depth + 1);
    }

    inline U16 _convert_ptr_to_token_id(const DatastoreShard &shard, const U64 ptr) const {
        assert (ptr % sizeof(U16) == 0);
        assert (ptr <= shard.ds_size);
        if (ptr == shard.ds_size) {
            // This happens when we matched the very end of the ds.
            return _eos_token_id;
        }
        U16 token_id; // no need to initialize
        memcpy(&token_id, shard.ds + ptr, sizeof(U16));
        // If you see \xff\xff, this actually means we're at the very end of a document.
        if (token_id == 65535) token_id = _eos_token_id;
        return token_id;
    }

    inline U64 _convert_rank_to_ptr(const DatastoreShard &shard, const U64 rank) const {
        assert (rank < shard.tok_cnt);
        U64 ptr = 0; // need to zero-initialize such that all 8 bytes are filled
        memcpy(&ptr, shard.sa + rank * shard.ptr_size, shard.ptr_size);
        return ptr;
    }

    inline vector<U64> _convert_ranks_to_ptrs(const DatastoreShard &shard, const U64 rank_start, const U64 rank_end) const {
        assert (rank_start <= rank_end);
        assert (rank_end <= shard.tok_cnt);
        vector<U64> ptrs(rank_end - rank_start);
        U64 ptr = 0; // need to zero-initialize such that all 8 bytes are filled
        for (U64 rank = rank_start; rank < rank_end; rank++) {
            memcpy(&ptr, shard.sa + rank * shard.ptr_size, shard.ptr_size);
            ptrs[rank - rank_start] = ptr;
        }
        return ptrs;
    }

    inline U64 _convert_doc_ix_to_ptr(const DatastoreShard &shard, const U64 doc_ix) const {
        assert (doc_ix <= shard.doc_cnt);
        if (doc_ix == shard.doc_cnt) {
            return shard.ds_size;
        }
        U64 ptr = 0;
        memcpy(&ptr, shard.od + doc_ix * sizeof(U64), sizeof(U64));
        return ptr;
    }

    inline U64 _convert_doc_ix_to_meta_ptr(const DatastoreShard &shard, const U64 doc_ix) const {
        assert (doc_ix <= shard.doc_cnt);
        if (doc_ix == shard.doc_cnt) {
            return shard.mt_size;
        }
        U64 ptr = 0;
        memcpy(&ptr, shard.om + doc_ix * sizeof(U64), sizeof(U64));
        return ptr;
    }

    inline pair<U64, U64> _bin_search(const vector<U64> &arr, U64 val) const {
        U64 lo = -1, hi = arr.size(); // lo is always < val, hi is always >= val
        while (hi - lo > 1) {
            U64 mi = (lo + hi) >> 1;
            if (arr[mi] < val) {
                lo = mi;
            } else {
                hi = mi;
            }
        }
        return {lo, hi};
    }

private:

    U16 _eos_token_id;
    bool _load_to_ram;
    size_t _ds_prefetch_depth;
    size_t _sa_prefetch_depth;
    size_t _od_prefetch_depth;
    size_t _num_shards;
    vector<DatastoreShard> _shards;
};
