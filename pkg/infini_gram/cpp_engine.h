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
#include <algorithm> // for sort
#include <random>
#include <thread>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>
#include <cmath>

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

struct Config {
    U64 MAX_CNT_FOR_NTD;
    U64 MAX_OUTPUT_DOC_TOKENS;
    U64 MAX_CLAUSE_FREQ_PER_SHARD;
    U64 MAX_DIFF_TOKENS;
    size_t ds_prefetch_depth;
    size_t sa_prefetch_depth;
    size_t od_prefetch_depth;
};
struct DatastoreShard {
    U8* ds;
    U8* sa;
    U64 tok_cnt;
    U64 ds_size;
    U8 ptr_size;
    U8* od;
    U64 doc_cnt;
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
    vector<U64> cnt_by_clause;
    U64 cnt;
    vector<U64> cnt_by_shard;
    vector<vector<pair<U64, U64>>> valid_ptr_ranges_by_shard;
    vector<double> subsampling_factor_by_shard;
    bool approx;
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
struct SearchDocResult {
    U64 doc_ix;
    U64 doc_len;
    U64 disp_len;
    vector<U16> token_ids;
};
struct SearchDocsResult {
    U64 cnt;
    bool approx;
    vector<U64> cnt_by_clause;
    vector<U64> idxs;
    vector<SearchDocResult> documents;
};

class NGramLanguageModeling {

public:

    NGramLanguageModeling() {}

    NGramLanguageModeling(const string index_dir, const U16 eos_token_id, const Config config)
        : _index_dir(index_dir), _eos_token_id(eos_token_id), _config(config) {

        assert_little_endian();
        assert (fs::exists(index_dir));

        vector<string> ds_paths, sa_paths, od_paths;
        for (const auto & entry : fs::directory_iterator(index_dir)) {
            if (entry.path().string().find("tokenized") != string::npos) {
                ds_paths.push_back(entry.path());
            } else if (entry.path().string().find("table") != string::npos) {
                sa_paths.push_back(entry.path());
            } else if (entry.path().string().find("offset") != string::npos) {
                od_paths.push_back(entry.path());
            }
        }
        sort(ds_paths.begin(), ds_paths.end());
        sort(sa_paths.begin(), sa_paths.end());
        sort(od_paths.begin(), od_paths.end());
        assert (ds_paths.size() == sa_paths.size());
        assert (ds_paths.size() == od_paths.size());
        _num_shards = ds_paths.size();
        assert (_num_shards > 0);

        for (auto s = 0; s < _num_shards; s++) {
            auto ds_path = ds_paths[s];
            int f_ds = open(ds_path.c_str(), O_RDONLY);
            assert (f_ds != -1);
            struct stat s_ds;
            auto fstat_ret = fstat(f_ds, &s_ds);
            assert (fstat_ret != -1);
            U8 *ds = static_cast<U8*>(mmap(NULL, s_ds.st_size, PROT_READ, MAP_PRIVATE, f_ds, 0));
            assert (ds != MAP_FAILED);
            madvise(ds, s_ds.st_size, MADV_RANDOM);

            auto sa_path = sa_paths[s];
            int f_sa = open(sa_path.c_str(), O_RDONLY);
            assert (f_sa != -1);
            struct stat s_sa;
            fstat_ret = fstat(f_sa, &s_sa);
            assert (fstat_ret != -1);
            U8 *sa = static_cast<U8*>(mmap(NULL, s_sa.st_size, PROT_READ, MAP_PRIVATE, f_sa, 0));
            assert (sa != MAP_FAILED);
            madvise(sa, s_sa.st_size, MADV_RANDOM);

            auto od_path = od_paths[s];
            int f_od = open(od_path.c_str(), O_RDONLY);
            assert (f_od != -1);
            struct stat s_od;
            fstat_ret = fstat(f_od, &s_od);
            assert (fstat_ret != -1);
            U8 *od = static_cast<U8*>(mmap(NULL, s_od.st_size, PROT_READ, MAP_PRIVATE, f_od, 0));
            assert (od != MAP_FAILED);
            madvise(od, s_od.st_size, MADV_RANDOM);

            U64 ds_size = s_ds.st_size;
            U64 sa_size = s_sa.st_size;
            U64 od_size = s_od.st_size;
            assert (ds_size % 2 == 0);
            U64 tok_cnt = ds_size / sizeof(U16);
            assert (sa_size % tok_cnt == 0);
            U8 ptr_size = (U8)(sa_size / tok_cnt);
            assert (od_size % 8 == 0);
            U64 doc_cnt = od_size / 8;

            auto shard = DatastoreShard{ds, sa, tok_cnt, ds_size, ptr_size, od, doc_cnt};
            _shards.push_back(shard);
        }

        cout << "Loaded index from " << index_dir << endl;
    }

    ~NGramLanguageModeling() {
        for (auto shard : _shards) {
            munmap(shard.ds, shard.ds_size);
            munmap(shard.sa, shard.tok_cnt * shard.ptr_size);
            munmap(shard.od, shard.doc_cnt * 8);
        }
    }

    // Version of methods which can be invoked as thread
    virtual void find_inplace(const vector<U16>* const input_ids, FindResult* const thread_output) const {
        *thread_output = find(*input_ids);
    }
    virtual void count_inplace(const vector<U16>* const input_ids, CountResult* const thread_output) const {
        *thread_output = count(*input_ids);
    }
    virtual void prob_inplace(const vector<U16>* const prompt_ids, const U16 cont_id, ProbResult* const thread_output) const {
        *thread_output = prob(*prompt_ids, cont_id);
    }
    virtual void ntd_inplace(const vector<U16>* const prompt_ids, DistResult* const thread_output) const {
        *thread_output = ntd(*prompt_ids);
    }
    virtual void find_disj_inplace(const vector<vector<U16>>* const disj_clause, const bool approx, FindDisjResult* const thread_output) const {
        *thread_output = find_disj(*disj_clause, approx);
    }
    virtual void find_cnf_inplace(const vector<vector<vector<U16>>>* const cnf, FindCnfResult* const thread_output) const {
        *thread_output = find_cnf(*cnf);
    }

    virtual FindResult find(const vector<U16> &input_ids, const vector<pair<U64, U64>> &hint_segment_by_shard = {}) const {

        assert (hint_segment_by_shard.empty() || hint_segment_by_shard.size() == _num_shards);

        const U8 *input_buf = reinterpret_cast<const U8*>(input_ids.data());
        U64 num_bytes = input_ids.size() * sizeof(U16);

        vector<U64> start_thread_outputs(_num_shards);
        vector<U64> end_thread_outputs(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.push_back(thread(&NGramLanguageModeling::_find_thread, this, s,
                &input_ids, input_buf, num_bytes, hint_segment_by_shard.empty() ? nullptr : &hint_segment_by_shard[s], true, &start_thread_outputs[s]));
            threads.push_back(thread(&NGramLanguageModeling::_find_thread, this, s,
                &input_ids, input_buf, num_bytes, hint_segment_by_shard.empty() ? nullptr : &hint_segment_by_shard[s], false, &end_thread_outputs[s]));
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

        return FindResult{cnt, segment_by_shard};
    }

    void _find_thread(
        const size_t s,
        const vector<U16>* const input_ids,
        const U8* const input_buf,
        const U64 num_bytes,
        const pair<U64, U64>* const hint_segment,
        const bool finding_start,
        U64* const thread_output) const {

        const auto &shard = _shards[s];

        if (input_ids->empty()) {
            *thread_output = finding_start ? 0 : shard.tok_cnt;
            return;
        }

        U64 lo, hi;
        if (!hint_segment) {
            lo = -1; hi = shard.tok_cnt;
        } else {
            lo = hint_segment->first - 1; hi = hint_segment->second;
        }
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

    FindDisjResult find_disj(const vector<vector<U16>> &disj_clause, const bool approx = false) const {

        vector<FindResult> find_result_by_term;
        for (const auto &term : disj_clause) {
            find_result_by_term.push_back(find(term));
        }
        vector<U64> cnt_by_shard(_num_shards);
        vector<vector<pair<U64, U64>>> segment_by_term_by_shard(_num_shards);
        vector<double> subsampling_factor_by_shard(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.push_back(thread(&NGramLanguageModeling::_find_disj_thread, this, s,
                &find_result_by_term, approx, &cnt_by_shard[s], &segment_by_term_by_shard[s], &subsampling_factor_by_shard[s]));
        }
        for (auto &thread : threads) {
            thread.join();
        }

        U64 cnt = std::accumulate(cnt_by_shard.begin(), cnt_by_shard.end(), (U64)0);
        return FindDisjResult{cnt, cnt_by_shard, segment_by_term_by_shard, subsampling_factor_by_shard};
    }

    void _find_disj_thread(
        const size_t s,
        const vector<FindResult>* const find_result_by_term,
        const bool approx,
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
        if (approx && cnt > _config.MAX_CLAUSE_FREQ_PER_SHARD) {
            // TODO: This subsampling might not be uniform
            U64 new_cnt = 0;
            vector<pair<U64, U64>> new_segment_by_term;
            for (const auto &[start, end] : segment_by_term) {
                U64 length = end - start;
                U64 new_length = (U64)((length * _config.MAX_CLAUSE_FREQ_PER_SHARD + (cnt - 1)) / cnt); // ceil div
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

    virtual FindCnfResult find_cnf(const vector<vector<vector<U16>>> &cnf) const {

        assert (cnf.size() > 0);

        vector<FindDisjResult> find_disj_result_by_clause;
        vector<U64> cnt_by_clause;
        for (const auto &clause : cnf) {
            auto find_disj_result = find_disj(clause, true);
            find_disj_result_by_clause.push_back(find_disj_result);
            cnt_by_clause.push_back(find_disj_result.cnt);
        }
        if (std::find(cnt_by_clause.begin(), cnt_by_clause.end(), 0) != cnt_by_clause.end()) {
            return FindCnfResult{cnt_by_clause, 0, {}, {}, {}, false};
        }

        vector<U64> cnt_by_shard(_num_shards);
        vector<vector<pair<U64, U64>>> valid_ptr_ranges_by_shard(_num_shards);
        vector<double> subsampling_factor_by_shard(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.push_back(thread(&NGramLanguageModeling::_find_cnf_thread, this, s,
                &find_disj_result_by_clause, &cnt_by_shard[s], &valid_ptr_ranges_by_shard[s], &subsampling_factor_by_shard[s]));
        }
        for (auto &thread : threads) {
            thread.join();
        }

        U64 cnt = accumulate(cnt_by_shard.begin(), cnt_by_shard.end(), (U64)0);
        bool approx = any_of(subsampling_factor_by_shard.begin(), subsampling_factor_by_shard.end(), [](double f) { return f != 1.0; });
        return FindCnfResult{cnt_by_clause, cnt, cnt_by_shard, valid_ptr_ranges_by_shard, subsampling_factor_by_shard, approx};
    }

    void _find_cnf_thread(
        const size_t s,
        const vector<FindDisjResult>* const _find_disj_result_by_clause,
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
                if (new_l != (U64)-1 && new_l + _config.MAX_DIFF_TOKENS * sizeof(U16) >= l && new_r != (U64)-1 && new_r <= r + _config.MAX_DIFF_TOKENS * sizeof(U16)) { // +- MAX_DIFF_TOKENS tokens
                    new_valid_ptr_ranges.push_back({new_l, new_r});
                } else {
                    if (new_l != (U64)-1 && new_l +  _config.MAX_DIFF_TOKENS * sizeof(U16) >= l) {
                        new_valid_ptr_ranges.push_back({new_l, l});
                    }
                    if (new_r != (U64)-1 && new_r <= r +  _config.MAX_DIFF_TOKENS * sizeof(U16)) {
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

    virtual CountResult count(const vector<U16> &input_ids) const {
        auto find_result = find(input_ids);
        return CountResult{find_result.cnt, false};
    }

    virtual CountResult count_cnf(const vector<vector<vector<U16>>> &cnf) const {
        auto find_cnf_result = find_cnf(cnf);
        return CountResult{find_cnf_result.cnt, find_cnf_result.approx};
    }

    virtual ProbResult prob(const vector<U16> &prompt_ids, const U16 cont_id) const {

        auto prompt_find_result = find(prompt_ids);
        U64 prompt_cnt = prompt_find_result.cnt;
        if (prompt_cnt == 0) {
            return ProbResult{0, 0, -1.0};
        }
        vector<U16> ngram = {prompt_ids.begin(), prompt_ids.end()};
        ngram.push_back(cont_id);
        auto cont_find_result = find(ngram, prompt_find_result.segment_by_shard);
        U64 cont_cnt = cont_find_result.cnt;
        double prob = (double)cont_cnt / prompt_cnt;

        return ProbResult{prompt_cnt, cont_cnt, prob};
    }

    virtual DistResult ntd(const vector<U16> &prompt_ids) const {

        auto prompt_find_result = find(prompt_ids);
        if (prompt_find_result.cnt == 0) {
            return DistResult{0, {}, false};
        }
        U64 unit = 1;
        while (prompt_find_result.cnt > unit * _config.MAX_CNT_FOR_NTD) {
            unit <<= 1;
        }
        bool approx = (unit > 1);

        vector<map<U16, U64>> thread_outputs(_num_shards);
        vector<thread> threads;
        for (auto s = 0; s < _num_shards; s++) {
            threads.push_back(thread(&NGramLanguageModeling::_get_freq_by_token_id_approx, this,
                s, prompt_ids.size() * sizeof(U16), &prompt_find_result.segment_by_shard[s], unit, nullptr, nullptr, &thread_outputs[s]));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        map<U16, U64> freq_by_token_id = {};
        for (auto s = 0; s < _num_shards; s++) {
            for (auto& [token_id, freq] : thread_outputs[s]) {
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

        return DistResult{prompt_cnt, result_by_token_id, approx};
    }

    void _get_freq_by_token_id_approx(
        const size_t s,
        const U64 num_bytes,
        const pair<U64, U64>* const segment,
        const U64 unit,
        const U16* const token_start,
        const U16* const token_end,
        map<U16, U64>* const thread_output) const {

        const auto& shard = _shards[s];
        U64 start = segment->first, end = segment->second;

        _prefetch_ntd(shard, num_bytes, start, end);

        if (end - start < 4 * unit) {
            for (U64 rank = start; rank < end; rank += unit) {
                U64 rank_mid = (rank + unit <= end) ? (rank + (unit >> 1)) : ((rank + end) >> 1);
                U64 ptr = _convert_rank_to_ptr(shard, rank_mid);
                U64 offset = ptr + num_bytes;
                U16 token_id = _convert_offset_to_token_id(shard, offset);
                (*thread_output)[token_id] += (rank + unit <= end) ? unit : (end - rank);
            }
            return;
        }

        // If start and end-1 has the same token, then we know this segment is all the same token
        U16 new_token_start = 0, new_token_end = 0;
        if (token_start) {
            new_token_start = *token_start;
        } else {
            U64 ptr_start = _convert_rank_to_ptr(shard, start);
            new_token_start = _convert_offset_to_token_id(shard, ptr_start + num_bytes);
        }
        if (token_end) {
            new_token_end = *token_end;
        } else {
            U64 ptr_end = _convert_rank_to_ptr(shard, end - 1);
            new_token_end = _convert_offset_to_token_id(shard, ptr_end + num_bytes);
        }
        if (new_token_start == new_token_end) {
            (*thread_output)[new_token_start] = end - start;
            return;
        }

        // Otherwise, we do divide and conquer
        U64 mi = (start + end) >> 1;
        pair<U64, U64> left_segment = {start, mi}, right_segment = {mi, end};
        // this->_get_freq_by_token_id_approx(s, num_bytes, &left_segment, unit, &new_token_start, nullptr, &left_thread_output);
        // this->_get_freq_by_token_id_approx(s, num_bytes, &right_segment, unit, nullptr, &new_token_end, &right_thread_output);
        map<U16, U64> left_thread_output = {}, right_thread_output = {};
        auto left_thread = thread(&NGramLanguageModeling::_get_freq_by_token_id_approx, this,
            s, num_bytes, &left_segment, unit, &new_token_start, nullptr, &left_thread_output);
        auto right_thread = thread(&NGramLanguageModeling::_get_freq_by_token_id_approx, this,
            s, num_bytes, &right_segment, unit, nullptr, &new_token_end, &right_thread_output);
        left_thread.join();
        right_thread.join();
        // TODO: This map merge is not efficient. Need to hack into the endianness of token_ids.
        for (auto& [token_id, freq] : left_thread_output) {
            (*thread_output)[token_id] += freq;
        }
        for (auto& [token_id, freq] : right_thread_output) {
            (*thread_output)[token_id] += freq;
        }
        // if (left_thread_output.rbegin()->first != right_thread_output.begin()->first) {
        //     *thread_output = left_thread_output;
        //     thread_output->insert(right_thread_output.begin(), right_thread_output.end());
        // } else {
        //     *thread_output = left_thread_output;
        //     (*thread_output)[right_thread_output.begin()->first] += right_thread_output.begin()->second;
        //     thread_output->insert(right_thread_output.begin(), right_thread_output.end()); // the first key in right_thread_output will be ignored
        // }
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

        return InfgramProbResult{result.prompt_cnt, result.cont_cnt, result.prob, suffix_len};
    }

    InfgramDistResult infgram_ntd(const vector<U16> &prompt_ids) const {

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
        auto result = ntd(prompt_suffix_ids);

        return InfgramDistResult{result.prompt_cnt, result.result_by_token_id, result.approx, suffix_len};
    }

    virtual SearchDocsResult search_docs(const vector<vector<vector<U16>>> &cnf, const size_t maxnum) const {

        assert (cnf.size() > 0);
        assert (maxnum > 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        if (cnf.size() == 1) {
            auto disj_clause = cnf[0];
            auto find_result = find_disj(disj_clause, false);
            if (find_result.cnt == 0) {
                return SearchDocsResult{0, false, {0}, {}, {}};
            }

            // sample up to maxnum documents
            auto &cnt = find_result.cnt;
            auto &cnt_by_shard = find_result.cnt_by_shard;
            auto &segment_by_term_by_shard = find_result.segment_by_term_by_shard;
            vector<SearchDocResult> documents;
            vector<U64> idxs;
            for (auto d = 0; d < maxnum; d++) {
                size_t s = discrete_distribution<size_t>(cnt_by_shard.begin(), cnt_by_shard.end())(gen);
                auto &segment_by_term = segment_by_term_by_shard[s];
                vector<U64> cnt_by_term;
                for (const auto &[start, end] : segment_by_term) {
                    cnt_by_term.push_back(end - start);
                }
                size_t ss = discrete_distribution<size_t>(cnt_by_term.begin(), cnt_by_term.end())(gen);
                auto &[start, end] = segment_by_term[ss];
                U64 rank = uniform_int_distribution<U64>(start, end - 1)(gen); // left inclusive, right inclusive
                U64 ptr = _convert_rank_to_ptr(_shards[s], rank);
                SearchDocResult document = get_document(s, ptr, ptr, _config.MAX_OUTPUT_DOC_TOKENS / maxnum);
                U64 idx = accumulate(cnt_by_shard.begin(), cnt_by_shard.begin() + s, (U64)0) + accumulate(cnt_by_term.begin(), cnt_by_term.begin() + ss, (U64)0) + (rank - start);
                documents.push_back(document);
                idxs.push_back(idx);
            }
            return SearchDocsResult{cnt, false, {find_result.cnt}, idxs, documents};
        }

        auto find_cnf_result = find_cnf(cnf);
        if (find_cnf_result.cnt == 0) {
            return SearchDocsResult{0, false, find_cnf_result.cnt_by_clause, {}, {}};
        }

        // sample up to maxnum documents
        auto &valid_ptr_ranges_by_shard = find_cnf_result.valid_ptr_ranges_by_shard;
        vector<U64> valid_ptr_cnt_by_shard;
        for (const auto &valid_ptr_ranges : valid_ptr_ranges_by_shard) {
            valid_ptr_cnt_by_shard.push_back(valid_ptr_ranges.size());
        }
        U64 valid_ptr_cnt = accumulate(valid_ptr_cnt_by_shard.begin(), valid_ptr_cnt_by_shard.end(), (U64)0);
        vector<SearchDocResult> documents;
        vector<U64> idxs;
        for (auto d = 0; d < maxnum; d++) {
            size_t s = discrete_distribution<size_t>(valid_ptr_cnt_by_shard.begin(), valid_ptr_cnt_by_shard.end())(gen);
            auto &valid_ptr_ranges = valid_ptr_ranges_by_shard[s];
            U64 i = uniform_int_distribution<U64>(0, valid_ptr_ranges.size() - 1)(gen); // left inclusive, right inclusive
            auto &ptr_range = valid_ptr_ranges[i];
            double percentile = (double)(accumulate(valid_ptr_cnt_by_shard.begin(), valid_ptr_cnt_by_shard.begin() + s, (U64)0) + i) / valid_ptr_cnt;
            U64 idx = (U64)(percentile * find_cnf_result.cnt);
            auto &[l, r] = ptr_range;
            SearchDocResult document = get_document(s, l, r, _config.MAX_OUTPUT_DOC_TOKENS / maxnum);
            documents.push_back(document);
            idxs.push_back(idx);
        }

        return SearchDocsResult{find_cnf_result.cnt, find_cnf_result.approx, find_cnf_result.cnt_by_clause, idxs, documents};
    }

    SearchDocResult get_document(
        const size_t s,
        const U64 left_ptr,
        const U64 right_ptr,
        const U64 max_output_doc_tokens) const {

        assert (s < _num_shards);
        assert (left_ptr <= right_ptr);
        const auto &shard = _shards[s];
        assert (left_ptr < shard.ds_size);
        assert (right_ptr <= shard.ds_size);

        U64 max_prepend_tokens, max_append_tokens;
        if (max_output_doc_tokens < (right_ptr - left_ptr) / sizeof(U16)) {
            max_prepend_tokens = 0;
            max_append_tokens = 0;
        } else {
            max_prepend_tokens = (max_output_doc_tokens - (right_ptr - left_ptr) / sizeof(U16)) / 2;
            max_append_tokens = (max_output_doc_tokens - (right_ptr - left_ptr) / sizeof(U16) + 1) / 2;
        }

        const vector<U8> doc_sep = {0xff, 0xff};
        U64 lo = 0, hi = shard.doc_cnt;
        while (hi - lo > 1) {
            _prefetch_doc(shard, lo, hi);
            U64 mi = (lo + hi) >> 1;
            U64 ptr = _convert_doc_rank_to_ptr(shard, mi);
            if (ptr <= left_ptr) {
                lo = mi;
            } else {
                hi = mi;
            }
        }

        U64 doc_ix = 0; for (auto _ = 0; _ < s; _++) doc_ix += _shards[_].doc_cnt; doc_ix += lo;

        U64 doc_start_ptr = _convert_doc_rank_to_ptr(shard, lo) + 2; // +2 because we want to skip the document separator
        U64 doc_end_ptr = _convert_doc_rank_to_ptr(shard, lo + 1);
        U64 doc_len = (doc_end_ptr - doc_start_ptr) >> 1;

        U64 start_ptr = max(doc_start_ptr, left_ptr - 2 * max_prepend_tokens);
        U64 end_ptr = min(doc_end_ptr, right_ptr + 2 * max_append_tokens);
        U64 disp_len = (end_ptr - start_ptr) >> 1;

        vector<U16> token_ids(reinterpret_cast<U16*>(shard.ds + start_ptr), reinterpret_cast<U16*>(shard.ds + end_ptr));
        U64 token_offset = (left_ptr - start_ptr) / sizeof(U16);

        return SearchDocResult{doc_ix, doc_len, disp_len, token_ids};
    }

public:

    void _prefetch_find(const DatastoreShard &shard, const U64 num_bytes, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.tok_cnt) return; // This might happen when lo = -1 and hi = 0
        if (depth == _config.ds_prefetch_depth) { // fetch ds
            U64 ptr = _convert_rank_to_ptr(shard, mi);
            madvise(shard.ds + ptr - ptr % PAGESIZE, num_bytes + ptr % PAGESIZE, MADV_WILLNEED);
        } else if (depth == _config.sa_prefetch_depth) { // fetch sa
            madvise(shard.sa + mi * shard.ptr_size - mi * shard.ptr_size % PAGESIZE, shard.ptr_size + mi * shard.ptr_size % PAGESIZE, MADV_WILLNEED);
            return;
        }
        _prefetch_find(shard, num_bytes, lo, mi, depth + 1);
        _prefetch_find(shard, num_bytes, mi, hi, depth + 1);
    }

    void _prefetch_ntd(const DatastoreShard &shard, const U64 num_bytes, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.tok_cnt) return; // This might happen when lo = -1 and hi = 0
        if (depth == _config.ds_prefetch_depth) { // fetch ds
            U64 ptr = _convert_rank_to_ptr(shard, mi-1);
            U64 offset = ptr + num_bytes;
            madvise(shard.ds + offset - offset % PAGESIZE, sizeof(U16) + offset % PAGESIZE, MADV_WILLNEED);
            ptr = _convert_rank_to_ptr(shard, mi);
            offset = ptr + num_bytes;
            madvise(shard.ds + offset - offset % PAGESIZE, sizeof(U16) + offset % PAGESIZE, MADV_WILLNEED);
        } else if (depth == _config.sa_prefetch_depth) { // fetch sa
            madvise(shard.sa + (mi-1) * shard.ptr_size - (mi-1) * shard.ptr_size % PAGESIZE, 2 * shard.ptr_size + (mi-1) * shard.ptr_size % PAGESIZE, MADV_WILLNEED); // since we need both mi-1 and mi
            return;
        }
        _prefetch_ntd(shard, num_bytes, lo, mi, depth + 1);
        _prefetch_ntd(shard, num_bytes, mi, hi, depth + 1);
    }

    void _prefetch_doc(const DatastoreShard &shard, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.doc_cnt) return; // This might happen when lo = -1 and hi = 0
        if (depth == _config.od_prefetch_depth) { // fetch od
            madvise(shard.od + mi * 8 - mi * 8 % PAGESIZE, 2 * PAGESIZE, MADV_WILLNEED);
            return;
        }
        _prefetch_doc(shard, lo, mi, depth + 1);
        _prefetch_doc(shard, mi, hi, depth + 1);
    }

    inline U16 _convert_offset_to_token_id(const DatastoreShard &shard, const U64 offset) const {
        assert (offset % 2 == 0);
        assert (offset <= shard.ds_size);
        if (offset == shard.ds_size) {
            // This happens when we matched the very end of the ds.
            return _eos_token_id;
        }
        U16 token_id; // no need to initialize
        memcpy(&token_id, shard.ds + offset, sizeof(U16));
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

    inline U64 _convert_doc_rank_to_ptr(const DatastoreShard &shard, const U64 doc_rank) const {
        assert (doc_rank <= shard.doc_cnt);
        if (doc_rank == shard.doc_cnt) {
            return shard.ds_size;
        }
        U64 ptr = 0;
        memcpy(&ptr, shard.od + doc_rank * 8, 8);
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

public:
    string _index_dir;
    U16 _eos_token_id;
    Config _config;
    size_t _num_shards;
    vector<DatastoreShard> _shards;
};


class NGramLanguageModelingUnion : public NGramLanguageModeling {

public:

    NGramLanguageModelingUnion(const vector<string> index_dirs, const U16 eos_token_id, const Config config) {
        for (const auto &index_dir : index_dirs) {
            _lms.emplace_back(make_unique<NGramLanguageModeling>(index_dir, eos_token_id, config));
        }
        _num_lms = _lms.size();
    }

    FindResult find(const vector<U16> &input_ids, const vector<pair<U64, U64>> &hint_segment_by_shard = {}) const override {
        vector<FindResult> results(_num_lms);
        vector<thread> threads;
        for (auto l = 0; l < _num_lms; l++) {
            threads.push_back(thread(&NGramLanguageModeling::find_inplace, _lms[l].get(), &input_ids, &results[l]));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        U64 cnt = accumulate(results.begin(), results.end(), (U64)0, [](U64 a, const FindResult &b) { return a + b.cnt; });
        return FindResult{cnt, {}};
    }

    FindCnfResult find_cnf(const vector<vector<vector<U16>>> &cnf) const override {
        vector<FindCnfResult> results(_num_lms);
        vector<thread> threads;
        for (auto l = 0; l < _num_lms; l++) {
            threads.push_back(thread(&NGramLanguageModeling::find_cnf_inplace, _lms[l].get(), &cnf, &results[l]));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        U64 cnt = 0;
        vector<U64> cnt_by_clause(cnf.size(), 0);
        for (const auto &result : results) {
            cnt += result.cnt;
            for (auto c = 0; c < cnf.size(); c++) {
                cnt_by_clause[c] += result.cnt_by_clause[c];
            }
        }
        bool approx = any_of(results.begin(), results.end(), [](const FindCnfResult &result) { return result.approx; });
        return FindCnfResult{cnt_by_clause, cnt, {}, {}, {}, approx};
    }

    CountResult count(const vector<U16> &input_ids) const override {
        vector<CountResult> results(_num_lms);
        vector<thread> threads;
        for (auto l = 0; l < _num_lms; l++) {
            threads.push_back(thread(&NGramLanguageModeling::count_inplace, _lms[l].get(), &input_ids, &results[l]));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        U64 cnt = accumulate(results.begin(), results.end(), (U64)0, [](U64 a, const CountResult &b) { return a + b.count; });
        bool approx = any_of(results.begin(), results.end(), [](const CountResult &result) { return result.approx; });
        return CountResult{cnt, approx};
    }

    ProbResult prob(const vector<U16> &prompt_ids, const U16 cont_id) const {
        vector<ProbResult> results(_num_lms);
        vector<thread> threads;
        for (auto l = 0; l < _num_lms; l++) {
            threads.push_back(thread(&NGramLanguageModeling::prob_inplace, _lms[l].get(), &prompt_ids, cont_id, &results[l]));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        U64 prompt_cnt = accumulate(results.begin(), results.end(), (U64)0, [](U64 a, const ProbResult &b) { return a + b.prompt_cnt; });
        U64 cont_cnt = accumulate(results.begin(), results.end(), (U64)0, [](U64 a, const ProbResult &b) { return a + b.cont_cnt; });
        if (prompt_cnt == 0) {
            return ProbResult{0, 0, -1.0};
        }
        double prob = (double)cont_cnt / prompt_cnt;
        return ProbResult{prompt_cnt, cont_cnt, prob};
    }

    DistResult ntd(const vector<U16> &prompt_ids) const override {
        vector<DistResult> results(_num_lms);
        vector<thread> threads;
        for (auto l = 0; l < _num_lms; l++) {
            threads.push_back(thread(&NGramLanguageModeling::ntd_inplace, _lms[l].get(), &prompt_ids, &results[l]));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        U64 prompt_cnt = accumulate(results.begin(), results.end(), (U64)0, [](U64 a, const DistResult &b) { return a + b.prompt_cnt; });
        map<U16, U64> freq_by_token_id = {};
        for (const auto &result : results) {
            for (const auto& [token_id, token_result] : result.result_by_token_id) {
                freq_by_token_id[token_id] += token_result.cont_cnt;
            }
        }
        map<U16, DistTokenResult> result_by_token_id = {};
        for (const auto& [token_id, freq] : freq_by_token_id) {
            result_by_token_id[token_id] = DistTokenResult{freq, (double)freq / prompt_cnt};
        }
        bool approx = any_of(results.begin(), results.end(), [](const DistResult &result) { return result.approx; });
        return DistResult{prompt_cnt, result_by_token_id, approx};
    }

    SearchDocsResult search_docs(const vector<vector<vector<U16>>> &cnf, const size_t maxnum) const override {

        assert (cnf.size() > 0);
        assert (maxnum > 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        if (cnf.size() == 1) {
            auto disj_clause = cnf[0];
            vector<FindDisjResult> find_results(_num_lms);
            vector<thread> threads;
            for (auto l = 0; l < _num_lms; l++) {
                threads.push_back(thread(&NGramLanguageModeling::find_disj_inplace, _lms[l].get(), &disj_clause, false, &find_results[l]));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            vector<U64> cnt_by_lm;
            for (const auto &find_result : find_results) {
                cnt_by_lm.push_back(find_result.cnt);
            }
            U64 cnt_total = accumulate(cnt_by_lm.begin(), cnt_by_lm.end(), (U64)0);
            if (cnt_total == 0) {
                return SearchDocsResult{0, false, {0}, {}, {}};
            }

            // sample up to maxnum documents
            vector<SearchDocResult> documents;
            vector<U64> idxs;
            for (auto d = 0; d < maxnum; d++) {
                size_t l = discrete_distribution<size_t>(cnt_by_lm.begin(), cnt_by_lm.end())(gen);
                const auto &lm = _lms[l];
                auto &find_result = find_results[l];
                auto &cnt = find_result.cnt;
                auto &cnt_by_shard = find_result.cnt_by_shard;
                auto &segment_by_term_by_shard = find_result.segment_by_term_by_shard;
                size_t s = discrete_distribution<size_t>(cnt_by_shard.begin(), cnt_by_shard.end())(gen);
                auto &segment_by_term = segment_by_term_by_shard[s];
                vector<U64> cnt_by_term;
                for (const auto &[start, end] : segment_by_term) {
                    cnt_by_term.push_back(end - start);
                }
                size_t ss = discrete_distribution<size_t>(cnt_by_term.begin(), cnt_by_term.end())(gen);
                auto &[start, end] = segment_by_term[ss];
                U64 rank = uniform_int_distribution<U64>(start, end - 1)(gen); // left inclusive, right inclusive
                U64 ptr = lm->_convert_rank_to_ptr(lm->_shards[s], rank);
                SearchDocResult document = lm->get_document(s, ptr, ptr, lm->_config.MAX_OUTPUT_DOC_TOKENS / maxnum);
                U64 idx = accumulate(cnt_by_shard.begin(), cnt_by_shard.begin() + s, (U64)0) + accumulate(cnt_by_term.begin(), cnt_by_term.begin() + ss, (U64)0) + (rank - start);
                documents.push_back(document);
                idxs.push_back(idx);
            }
            return SearchDocsResult{cnt_total, false, {cnt_total}, idxs, documents};
        }

        vector<FindCnfResult> find_cnf_results(_num_lms);
        vector<thread> threads;
        for (auto l = 0; l < _num_lms; l++) {
            threads.push_back(thread(&NGramLanguageModeling::find_cnf_inplace, _lms[l].get(), &cnf, &find_cnf_results[l]));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        vector<U64> cnt_by_lm;
        for (const auto &find_cnf_result : find_cnf_results) {
            cnt_by_lm.push_back(find_cnf_result.cnt);
        }
        U64 cnt_total = accumulate(cnt_by_lm.begin(), cnt_by_lm.end(), (U64)0);
        bool approx = any_of(find_cnf_results.begin(), find_cnf_results.end(), [](const FindCnfResult &find_cnf_result) { return find_cnf_result.approx; });
        if (cnt_total == 0) {
            return SearchDocsResult{0, false, find_cnf_results[0].cnt_by_clause, {}, {}}; // NOTE: The cnt_by_clause here is a placeholder, since we do not use it in the union case
        }

        // sample up to maxnum documents
        vector<SearchDocResult> documents;
        vector<U64> idxs;
        for (auto d = 0; d < maxnum; d++) {
            size_t ll = discrete_distribution<size_t>(cnt_by_lm.begin(), cnt_by_lm.end())(gen);
            const auto &lm = _lms[ll];
            auto &find_cnf_result = find_cnf_results[ll];
            auto &valid_ptr_ranges_by_shard = find_cnf_result.valid_ptr_ranges_by_shard;
            vector<U64> valid_ptr_cnt_by_shard;
            for (const auto &valid_ptr_ranges : valid_ptr_ranges_by_shard) {
                valid_ptr_cnt_by_shard.push_back(valid_ptr_ranges.size());
            }
            U64 valid_ptr_cnt = accumulate(valid_ptr_cnt_by_shard.begin(), valid_ptr_cnt_by_shard.end(), (U64)0);
            size_t s = discrete_distribution<size_t>(valid_ptr_cnt_by_shard.begin(), valid_ptr_cnt_by_shard.end())(gen);
            auto &valid_ptr_ranges = valid_ptr_ranges_by_shard[s];
            U64 i = uniform_int_distribution<U64>(0, valid_ptr_ranges.size() - 1)(gen); // left inclusive, right inclusive
            auto &ptr_range = valid_ptr_ranges[i];
            double percentile = (double)(accumulate(valid_ptr_cnt_by_shard.begin(), valid_ptr_cnt_by_shard.begin() + s, (U64)0) + i) / valid_ptr_cnt;
            U64 idx = (U64)(percentile * find_cnf_result.cnt);
            auto &[l, r] = ptr_range;
            SearchDocResult document = lm->get_document(s, l, r, lm->_config.MAX_OUTPUT_DOC_TOKENS / maxnum);
            documents.push_back(document);
            idxs.push_back(idx);
        }

        return SearchDocsResult{cnt_total, approx, find_cnf_results[0].cnt_by_clause, idxs, documents}; // NOTE: The cnt_by_clause here is a placeholder, since we do not use it in the union case
    }

public:
    vector<unique_ptr<NGramLanguageModeling>> _lms;
    size_t _num_lms;
};
