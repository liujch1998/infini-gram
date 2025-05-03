#include <cassert>
#include <cstdint> // for uint64_t
#include <cstring> // for memcpy
#include <iostream>
#include <map>
#include <set>
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
#include <deque>

#define U64 uint64_t
#define U32 uint32_t
#define U16 uint16_t
#define U8 uint8_t
#define PSS pair<size_t, size_t>

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
template<typename T=U16>
struct DistResult {
    U64 prompt_cnt;
    map<T, DistTokenResult> result_by_token_id;
    bool approx;
};
struct InfgramProbResult {
    U64 prompt_cnt;
    U64 cont_cnt;
    double prob;
    U64 suffix_len;
};
template<typename T=U16>
struct InfgramDistResult {
    U64 prompt_cnt;
    map<T, DistTokenResult> result_by_token_id;
    bool approx;
    U64 suffix_len;
};
template<typename T=U16>
struct DocResult {
    U64 doc_ix;
    U64 doc_len;
    U64 disp_len;
    U64 needle_offset; // token offset of the search term
    string metadata;
    vector<T> token_ids;
    bool blocked = false;
};
template<typename T=U16>
struct SearchDocsResult {
    U64 cnt;
    bool approx;
    vector<U64> idxs;
    vector<DocResult<T>> docs;
};
struct CreativityResult {
    vector<size_t> rs;
};
struct AttributionDoc {
    size_t s;
    U64 ptr;
};
struct AttributionSpan {
    size_t l;
    size_t r;
    size_t length;
    U64 count;
    double unigram_logprob_sum;
    vector<AttributionDoc> docs;
};
struct AttributionResult {
    vector<AttributionSpan> spans;
};

template<typename T=U16>
class EngineDiff;

template<typename T=U16>
class Engine {

public:

    Engine(
        const vector<string> index_dirs, const T eos_token_id, const T vocab_size, const size_t version,
        const bool load_to_ram, const size_t ds_prefetch_depth, const size_t sa_prefetch_depth, const size_t od_prefetch_depth,
        const set<T> bow_ids, const size_t attribution_block_size, const bool precompute_unigram_logprobs,
        map<string, vector<DatastoreShard>> prev_shards_by_index_dir)
        : _eos_token_id(eos_token_id), _vocab_size(vocab_size), _version(version),
          _load_to_ram(load_to_ram), _ds_prefetch_depth(ds_prefetch_depth), _sa_prefetch_depth(sa_prefetch_depth), _od_prefetch_depth(od_prefetch_depth),
          _bow_ids(bow_ids), _attribution_block_size(attribution_block_size),
          _doc_sep_id((T)(-1)), _doc_sep(vector<U8>(sizeof(T), 0xff))
    {

        assert_little_endian();

        map<T, U64> unigram_counts;

        for (const auto &index_dir : index_dirs) {
            if (prev_shards_by_index_dir.find(index_dir) != prev_shards_by_index_dir.end()) {
                _shards.insert(_shards.end(), prev_shards_by_index_dir[index_dir].begin(), prev_shards_by_index_dir[index_dir].end());
                continue;
            }

            assert (fs::exists(index_dir));

            vector<string> ds_paths, sa_paths, od_paths, mt_paths, om_paths, ug_paths;
            for (const auto &entry : fs::directory_iterator(index_dir)) {
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
                } else if (entry.path().string().find("unigram") != string::npos) {
                    ug_paths.push_back(entry.path());
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
            assert (ug_paths.size() == 0 || ug_paths.size() == ds_paths.size());

            for (size_t s = 0; s < ds_paths.size(); s++) {
                auto [ds, ds_size] = load_file(ds_paths[s]);
                auto [sa, sa_size] = load_file(sa_paths[s]);
                auto [od, od_size] = load_file(od_paths[s]);

                assert (ds_size % sizeof(T) == 0);
                U64 tok_cnt = ds_size / sizeof(T);
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

                _new_shards_by_index_dir[index_dir].push_back(_shards.back());

                if (precompute_unigram_logprobs) {
                    map<T, U64> shard_unigram_counts;
                    if (ug_paths.size() != 0) { // use cached result
                        ifstream f(ug_paths[s]);
                        string line;
                        for (T linenum = 0; getline(f, line); linenum++) {
                            T token_id;
                            U64 count;
                            if (line.find(" ") != string::npos) {
                                istringstream iss(line);
                                iss >> token_id >> count;
                            } else {
                                token_id = linenum;
                                istringstream iss(line);
                                iss >> count;
                            }
                            shard_unigram_counts[token_id] = count;
                        }
                    } else { // compute result
                        shard_unigram_counts = compute_unigram_counts(_shards.size() - 1);
                    }
                    for (const auto &[token_id, count] : shard_unigram_counts) {
                        unigram_counts[token_id] += count;
                    }
                }
            }
        }

        _num_shards = _shards.size();

        if (precompute_unigram_logprobs) {
            U64 total_tok_cnt = 0;
            for (const auto &[token_id, count] : unigram_counts) {
                total_tok_cnt += count;
            }
            assert (total_tok_cnt == get_total_tok_cnt());
            for (const auto &[token_id, count] : unigram_counts) {
                if (count > 0) {
                    _unigram_logprobs[token_id] = log(count) - log(total_tok_cnt);
                }
            }
        }
    }

    ~Engine() {
        for (auto &[index_dir, shards] : _new_shards_by_index_dir) {
            for (auto &shard : shards) {
                unload_file(shard.ds, shard.ds_size);
                unload_file(shard.sa, shard.tok_cnt * shard.ptr_size);
                unload_file(shard.od, shard.doc_cnt * sizeof(U64));
                if (shard.mt) {
                    unload_file(shard.mt, shard.mt_size);
                    unload_file(shard.om, shard.doc_cnt * sizeof(U64));
                }
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

    map<string, vector<DatastoreShard>> get_new_shards_by_index_dir() const {
        return _new_shards_by_index_dir;
    }

    map<T, U64> compute_unigram_counts(const size_t s) const {
        vector<T> input_ids((size_t)_vocab_size + 1);
        for (T i = 0; i < _vocab_size; i++) {
            input_ids[i] = i;
        }
        input_ids[_vocab_size] = _doc_sep_id;
        const U8 *input_buf = reinterpret_cast<const U8*>(input_ids.data());
        U64 num_bytes = sizeof(T);
        pair<U64, U64> hint_segment = {0, _shards[s].tok_cnt};

        map<T, U64> unigram_counts;
        vector<pair<U64, U64>> segment_by_token((size_t)_vocab_size + 1);
        for (size_t i_start = 0; i_start < (size_t)_vocab_size + 1; i_start += 256) {
            size_t i_end = min(i_start + 256, (size_t)_vocab_size + 1);
            vector<thread> threads;
            for (size_t i = i_start; i < i_end; i++) {
                threads.emplace_back(&Engine::_find_thread, this, s,
                    input_buf + i * num_bytes, num_bytes, hint_segment, &segment_by_token[i]);
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }
        for (size_t i = 0; i < (size_t)_vocab_size + 1; i++) {
            assert (segment_by_token[i].first <= segment_by_token[i].second);
            unigram_counts[input_ids[i]] = segment_by_token[i].second - segment_by_token[i].first;
        }

        return unigram_counts;
    }

    FindResult find(const vector<T> input_ids) const {

        vector<pair<U64, U64>> hint_segment_by_shard;
        for (const auto &shard : _shards) {
            hint_segment_by_shard.push_back({0, shard.tok_cnt});
        }
        return _find(input_ids, hint_segment_by_shard);
    }

    FindResult _find(const vector<T> &input_ids, const vector<pair<U64, U64>> &hint_segment_by_shard) const {

        assert (hint_segment_by_shard.size() == _num_shards);

        vector<T> reversed_input_ids;
        const U8 *input_buf;
        if (_version == 4) {
            input_buf = reinterpret_cast<const U8*>(input_ids.data());
        } else if (_version == 5) {
            reversed_input_ids = input_ids;
            reverse(reversed_input_ids.begin(), reversed_input_ids.end());
            input_buf = reinterpret_cast<const U8*>(reversed_input_ids.data());
        }
        U64 num_bytes = input_ids.size() * sizeof(T);

        vector<pair<U64, U64>> segment_by_shard(_num_shards);
        vector<thread> threads;
        for (size_t s = 0; s < _num_shards; s++) {
            threads.emplace_back(&Engine::_find_thread, this, s,
                input_buf, num_bytes, hint_segment_by_shard[s], &segment_by_shard[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        U64 cnt = 0;
        for (size_t s = 0; s < _num_shards; s++) {
            assert (segment_by_shard[s].first <= segment_by_shard[s].second);
            cnt += segment_by_shard[s].second - segment_by_shard[s].first;
        }

        return FindResult{ .cnt = cnt, .segment_by_shard = segment_by_shard, };
    }

    void _find_thread(
        const size_t s,
        const U8* const input_buf,
        const U64 num_bytes,
        const pair<U64, U64> hint_segment,
        pair<U64, U64>* const out_segment) const {

        const auto &shard = _shards[s];

        if (num_bytes == 0) {
            *out_segment = {0, shard.tok_cnt};
            return;
        }

        U64 lo = hint_segment.first, hi = hint_segment.second;
        U64 mi;
        while (lo < hi) {
            _prefetch_find_2(shard, num_bytes, lo, hi);
            mi = (lo + hi - 1) >> 1;
            U64 ptr = _convert_rank_to_ptr(shard, mi);
            auto o = lexicographical_compare_three_way(
                shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size),
                input_buf, input_buf + num_bytes);
            if (o == strong_ordering::less) {
                lo = mi + 1;
            } else if (o == strong_ordering::greater) {
                hi = mi;
            } else { // o == strong_ordering::equal
                break;
            }
        }
        if (lo == hi) {
            out_segment->first = lo;
            out_segment->second = lo;
            return;
        }

        // search left boundary in (lo-1, mi], which should be >= query
        U64 l = lo - 1, r = mi; // l is always < query, r is always >= query
        while (r - l > 1) {
            _prefetch_find(shard, num_bytes, l, r);
            U64 m = (l + r) >> 1;
            U64 ptr = _convert_rank_to_ptr(shard, m);
            bool less = lexicographical_compare(
                shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size),
                input_buf, input_buf + num_bytes);
            if (less) {
                l = m;
            } else {
                r = m;
            }
        }
        out_segment->first = r;

        // search right boundary in (mi, hi], which should be > query
        l = mi, r = hi; // l is always <= query, r is always > query
        while (r - l > 1) {
            _prefetch_find(shard, num_bytes, l, r);
            U64 m = (l + r) >> 1;
            U64 ptr = _convert_rank_to_ptr(shard, m);
            bool less = lexicographical_compare(
                input_buf, input_buf + num_bytes,
                shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size));
            if (less) {
                r = m;
            } else {
                l = m;
            }
        }
        out_segment->second = r;
    }

    void find_inplace(const vector<T>* const input_ids, FindResult* const thread_output) const {
        *thread_output = find(*input_ids);
    }

    FindDisjResult find_disj(const vector<vector<T>> &disj_clause, const U64 max_clause_freq) const {

        vector<FindResult> find_result_by_term(disj_clause.size());
        vector<thread> find_threads;
        for (size_t t = 0; t < disj_clause.size(); t++) {
            find_threads.emplace_back(&Engine::find_inplace, this, &disj_clause[t], &find_result_by_term[t]);
        }
        for (auto &thread : find_threads) {
            thread.join();
        }

        vector<U64> cnt_by_shard(_num_shards);
        vector<vector<pair<U64, U64>>> segment_by_term_by_shard(_num_shards);
        vector<double> subsampling_factor_by_shard(_num_shards);
        vector<thread> threads;
        U64 max_clause_freq_per_shard = (U64)((max_clause_freq + (_num_shards - 1)) / _num_shards); // ceil div
        for (size_t s = 0; s < _num_shards; s++) {
            threads.emplace_back(&Engine::_find_disj_thread, this, s,
                &find_result_by_term, max_clause_freq_per_shard, &cnt_by_shard[s], &segment_by_term_by_shard[s], &subsampling_factor_by_shard[s]);
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

    void find_disj_inplace(const vector<vector<T>>* const disj_clause, const U64 max_clause_freq, FindDisjResult* const thread_output) const {
        *thread_output = find_disj(*disj_clause, max_clause_freq);
    }

    FindCnfResult find_cnf(const vector<vector<vector<T>>> &cnf, const U64 max_clause_freq, const U64 max_diff_tokens) const {

        assert (cnf.size() > 0);

        vector<FindDisjResult> find_disj_result_by_clause(cnf.size());
        vector<thread> find_threads;
        for (size_t c = 0; c < cnf.size(); c++) {
            find_threads.emplace_back(&Engine::find_disj_inplace, this, &cnf[c], max_clause_freq, &find_disj_result_by_clause[c]);
        }
        for (auto &thread : find_threads) {
            thread.join();
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
        for (size_t s = 0; s < _num_shards; s++) {
            threads.emplace_back(&Engine::_find_cnf_thread, this, s,
                &find_disj_result_by_clause, max_diff_tokens, &cnt_by_shard[s], &valid_ptr_ranges_by_shard[s], &subsampling_factor_by_shard[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        U64 cnt = accumulate(cnt_by_shard.begin(), cnt_by_shard.end(), (U64)0);
        bool approx = any_of(subsampling_factor_by_shard.begin(), subsampling_factor_by_shard.end(), [](double f) { return f != 1.0; });
        vector<vector<U64>> ptrs_by_shard(_num_shards);
        for (size_t s = 0; s < _num_shards; s++) {
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
        for (size_t d = 1; d < find_disj_result_by_clause.size(); d++) {
            auto &find_disj_result = find_disj_result_by_clause[d];
            vector<U64> ptrs;
            for (const auto &[start, end] : find_disj_result.segment_by_term_by_shard[s]) {
                vector<U64> new_ptrs = _convert_ranks_to_ptrs(shard, start, end);
                ptrs.insert(ptrs.end(), new_ptrs.begin(), new_ptrs.end());
            }
            sort(ptrs.begin(), ptrs.end());
            vector<pair<U64, U64>> new_valid_ptr_ranges;
            for (const auto &[l, r] : valid_ptr_ranges) {
                auto lo = _bin_search(ptrs, r).first; // find the last match that is < r
                U64 new_l = lo == (U64)-1 ? -1 : min(l, ptrs[lo]);
                auto hi = _bin_search(ptrs, l).second; // find the first match that is >= l
                U64 new_r = hi == ptrs.size() ? -1 : max(r, ptrs[hi]);
                if (new_l != (U64)-1 && new_l + max_diff_tokens * sizeof(T) >= l && new_r != (U64)-1 && new_r <= r + max_diff_tokens * sizeof(T)) { // +- MAX_DIFF_TOKENS tokens
                    new_valid_ptr_ranges.push_back({new_l, new_r});
                } else {
                    if (new_l != (U64)-1 && new_l + max_diff_tokens * sizeof(T) >= l) {
                        new_valid_ptr_ranges.push_back({new_l, r});
                    }
                    if (new_r != (U64)-1 && new_r <= r + max_diff_tokens * sizeof(T)) {
                        new_valid_ptr_ranges.push_back({l, new_r});
                    }
                }
            }
            valid_ptr_ranges = new_valid_ptr_ranges;
            subsampling_factor *= find_disj_result.subsampling_factor_by_shard[s];
        }

        // remove ptr ranges that cross document boundary
        vector<pair<U64, U64>> new_valid_ptr_ranges;
        for (const auto &[l, r] : valid_ptr_ranges) {
            auto it = search(shard.ds + l, shard.ds + r, _doc_sep.begin(), _doc_sep.end());
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

    CountResult count(const vector<T> input_ids) const {
        auto find_result = find(input_ids);
        return CountResult{ .count = find_result.cnt, .approx = false, };
    }

    void count_inplace(const vector<T>* const input_ids, CountResult* const thread_output) const {
        *thread_output = count(*input_ids);
    }

    CountResult count_cnf(const vector<vector<vector<T>>> cnf, const U64 max_clause_freq, const U64 max_diff_tokens) const {
        auto find_cnf_result = find_cnf(cnf, max_clause_freq, max_diff_tokens);
        return CountResult{ .count = find_cnf_result.cnt, .approx = find_cnf_result.approx, };
    }

    ProbResult prob(const vector<T> prompt_ids, const T cont_id) const {

        auto prompt_find_result = find(prompt_ids);
        U64 prompt_cnt = prompt_find_result.cnt;
        if (prompt_cnt == 0) {
            return ProbResult{ .prompt_cnt = 0, .cont_cnt = 0, .prob = -1.0 };
        }
        vector<T> input_ids = {prompt_ids.begin(), prompt_ids.end()};
        input_ids.push_back(cont_id);
        FindResult cont_find_result;
        if (_version == 4) {
            cont_find_result = _find(input_ids, prompt_find_result.segment_by_shard);
        } else if (_version == 5) {
            cont_find_result = find(input_ids);
        }
        U64 cont_cnt = cont_find_result.cnt;
        double prob = (double)cont_cnt / prompt_cnt;

        return ProbResult{ .prompt_cnt = prompt_cnt, .cont_cnt = cont_cnt, .prob = prob };
    }

    DistResult<T> ntd(const vector<T> prompt_ids, const U64 max_support) const {

        auto prompt_find_result = find(prompt_ids);
        if (prompt_find_result.cnt == 0) {
            return DistResult<T>{ .prompt_cnt = 0, .result_by_token_id = {}, .approx = false, };
        }
        U64 unit = 1;
        while (prompt_find_result.cnt > unit * max_support) {
            unit <<= 1;
        }
        bool approx = (unit > 1);

        vector<map<T, U64>> freq_by_token_id_by_shard(_num_shards);
        vector<thread> threads;
        for (size_t s = 0; s < _num_shards; s++) {
            threads.emplace_back(&Engine::_get_freq_by_token_id_approx, this,
                s, prompt_ids.size() * sizeof(T), prompt_find_result.segment_by_shard[s], unit, nullptr, nullptr, &freq_by_token_id_by_shard[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        map<T, U64> freq_by_token_id = {};
        for (size_t s = 0; s < _num_shards; s++) {
            for (const auto &[token_id, freq] : freq_by_token_id_by_shard[s]) {
                freq_by_token_id[token_id] += freq;
            }
        }

        U64 prompt_cnt = 0;
        for (const auto &[token_id, freq] : freq_by_token_id) {
            prompt_cnt += freq;
        }
        assert (prompt_cnt == prompt_find_result.cnt);
        map<T, DistTokenResult> result_by_token_id = {};
        for (const auto &[token_id, freq] : freq_by_token_id) {
            result_by_token_id[token_id] = DistTokenResult{freq, (double)freq / prompt_cnt};
        }

        return DistResult<T>{ .prompt_cnt = prompt_cnt, .result_by_token_id = result_by_token_id, .approx = approx, };
    }

    void _get_freq_by_token_id_approx(
        const size_t s,
        const U64 num_bytes,
        const pair<U64, U64> segment,
        const U64 unit,
        const T* const token_start,
        const T* const token_end,
        map<T, U64>* const out_freq_by_token_id) const {

        const auto& shard = _shards[s];
        U64 start = segment.first, end = segment.second;

        _prefetch_ntd(shard, num_bytes, start, end);

        if (end - start < 4 * unit) {
            for (U64 rank = start; rank < end; rank += unit) {
                U64 rank_mid = (rank + unit <= end) ? (rank + (unit >> 1)) : ((rank + end) >> 1);
                U64 ptr = _convert_rank_to_ptr(shard, rank_mid);
                T token_id = _convert_ptr_to_token_id(shard, _version == 4 ? ptr + num_bytes : ptr - sizeof(T));
                (*out_freq_by_token_id)[token_id] += (rank + unit <= end) ? unit : (end - rank);
            }
            return;
        }

        // If start and end-1 has the same token, then we know this segment is all the same token
        T new_token_start = 0, new_token_end = 0;
        if (token_start) {
            new_token_start = *token_start;
        } else {
            U64 ptr_start = _convert_rank_to_ptr(shard, start);
            new_token_start = _convert_ptr_to_token_id(shard, _version == 4 ? ptr_start + num_bytes : ptr_start - sizeof(T));
        }
        if (token_end) {
            new_token_end = *token_end;
        } else {
            U64 ptr_end = _convert_rank_to_ptr(shard, end - 1);
            new_token_end = _convert_ptr_to_token_id(shard, _version == 4 ? ptr_end + num_bytes : ptr_end - sizeof(T));
        }
        if (new_token_start == new_token_end) {
            (*out_freq_by_token_id)[new_token_start] = end - start;
            return;
        }

        // Otherwise, we do divide and conquer
        U64 mi = (start + end) >> 1;
        pair<U64, U64> left_segment = {start, mi}, right_segment = {mi, end};
        map<T, U64> left_thread_output = {}, right_thread_output = {};
        auto left_thread = thread(&Engine::_get_freq_by_token_id_approx, this,
            s, num_bytes, left_segment, unit, &new_token_start, nullptr, &left_thread_output);
        auto right_thread = thread(&Engine::_get_freq_by_token_id_approx, this,
            s, num_bytes, right_segment, unit, nullptr, &new_token_end, &right_thread_output);
        left_thread.join();
        right_thread.join();
        // TODO: This map merge is not efficient. Need to hack into the endianness of token_ids.
        for (const auto &[token_id, freq] : left_thread_output) {
            (*out_freq_by_token_id)[token_id] += freq;
        }
        for (const auto &[token_id, freq] : right_thread_output) {
            (*out_freq_by_token_id)[token_id] += freq;
        }
    }

    InfgramProbResult infgram_prob(const vector<T> prompt_ids, const T cont_id) const {

        size_t L = prompt_ids.size();
        // binary lifting
        size_t l_lo = 0, l_hi = 1;
        while (true) {
            if (l_hi > L) { l_hi = L + 1; break; }
            const vector<T> prompt_suffix_ids{prompt_ids.begin() + L - l_hi, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) break;
            l_lo = l_hi;
            l_hi <<= 1;
        }
        // binary search within [l_lo, l_hi)
        while (l_hi - l_lo > 1) {
            size_t l_mid = (l_lo + l_hi) >> 1;
            const vector<T> prompt_suffix_ids{prompt_ids.begin() + L - l_mid, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) {
                l_hi = l_mid;
            } else {
                l_lo = l_mid;
            }
        }

        size_t suffix_len = l_lo;
        const vector<T> prompt_suffix_ids{prompt_ids.begin() + L - suffix_len, prompt_ids.end()};
        auto result = prob(prompt_suffix_ids, cont_id);

        return InfgramProbResult{
            .prompt_cnt = result.prompt_cnt,
            .cont_cnt = result.cont_cnt,
            .prob = result.prob,
            .suffix_len = suffix_len,
        };
    }

    InfgramDistResult<T> infgram_ntd(const vector<T> prompt_ids, const U64 max_support) const {

        size_t L = prompt_ids.size();
        // binary lifting
        size_t l_lo = 0, l_hi = 1;
        while (true) {
            if (l_hi > L) { l_hi = L + 1; break; }
            const vector<T> prompt_suffix_ids{prompt_ids.begin() + L - l_hi, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) break;
            l_lo = l_hi;
            l_hi <<= 1;
        }
        // binary search within [l_lo, l_hi)
        while (l_hi - l_lo > 1) {
            size_t l_mid = (l_lo + l_hi) >> 1;
            const vector<T> prompt_suffix_ids{prompt_ids.begin() + L - l_mid, prompt_ids.end()};
            auto result = find(prompt_suffix_ids);
            if (result.cnt == 0) {
                l_hi = l_mid;
            } else {
                l_lo = l_mid;
            }
        }

        size_t suffix_len = l_lo;
        const vector<T> prompt_suffix_ids{prompt_ids.begin() + L - suffix_len, prompt_ids.end()};
        auto result = ntd(prompt_suffix_ids, max_support);

        return InfgramDistResult<T>{
            .prompt_cnt = result.prompt_cnt,
            .result_by_token_id = result.result_by_token_id,
            .approx = result.approx,
            .suffix_len = suffix_len,
        };
    }

    SearchDocsResult<T> search_docs(const vector<T> input_ids, const size_t maxnum, const U64 max_disp_len) const {

        assert (maxnum > 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        auto find_result = find(input_ids);
        if (find_result.cnt == 0) {
            return SearchDocsResult<T>{ .cnt = 0, .approx = false, .idxs = {}, .docs = {}, };
        }

        // sample up to maxnum documents
        auto &segment_by_shard = find_result.segment_by_shard;
        vector<U64> cnt_by_shard;
        for (const auto &segment : segment_by_shard) {
            cnt_by_shard.push_back(segment.second - segment.first);
        }
        vector<U64> idxs;
        vector<DocResult<T>> docs;
        for (size_t d = 0; d < maxnum; d++) {
            size_t s = discrete_distribution<size_t>(cnt_by_shard.begin(), cnt_by_shard.end())(gen);
            auto &[start, end] = segment_by_shard[s];
            U64 rank = uniform_int_distribution<U64>(start, end - 1)(gen); // left inclusive, right inclusive
            U64 ptr = _convert_rank_to_ptr(_shards[s], rank);
            U64 idx = accumulate(cnt_by_shard.begin(), cnt_by_shard.begin() + s, (U64)0) + (rank - start);
            auto doc = get_doc_by_ptr(s, ptr, max_disp_len);
            idxs.push_back(idx);
            docs.push_back(doc);
        }

        return SearchDocsResult<T>{ .cnt = find_result.cnt, .approx = false, .idxs = idxs, .docs = docs, };
    }

    SearchDocsResult<T> search_docs_cnf(const vector<vector<vector<T>>> cnf, const size_t maxnum, const U64 max_disp_len, const U64 max_clause_freq, const U64 max_diff_tokens) const {

        assert (cnf.size() > 0);
        assert (maxnum > 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        auto find_cnf_result = find_cnf(cnf, max_clause_freq, max_diff_tokens);
        if (find_cnf_result.cnt == 0) {
            return SearchDocsResult<T>{ .cnt = 0, .approx = false, .idxs = {}, .docs = {}, };
        }

        // sample up to maxnum documents
        auto &ptrs_by_shard = find_cnf_result.ptrs_by_shard;
        vector<U64> ptr_cnt_by_shard;
        for (const auto &ptrs : ptrs_by_shard) {
            ptr_cnt_by_shard.push_back(ptrs.size());
        }
        U64 ptr_cnt = accumulate(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.end(), (U64)0);
        vector<U64> idxs;
        vector<DocResult<T>> docs;
        for (size_t d = 0; d < maxnum; d++) {
            size_t s = discrete_distribution<size_t>(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.end())(gen);
            auto &ptrs = ptrs_by_shard[s];
            U64 i = uniform_int_distribution<U64>(0, ptrs.size() - 1)(gen); // left inclusive, right inclusive
            auto &ptr = ptrs[i];
            double percentile = (double)(accumulate(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.begin() + s, (U64)0) + i) / ptr_cnt;
            U64 idx = (U64)(percentile * find_cnf_result.cnt);
            auto doc = get_doc_by_ptr(s, ptr, max_disp_len);
            idxs.push_back(idx);
            docs.push_back(doc);
        }

        return SearchDocsResult<T>{ .cnt = find_cnf_result.cnt, .approx = find_cnf_result.approx, .idxs = idxs, .docs = docs, };
    }

    DocResult<T> get_doc_by_rank(const size_t s, const U64 rank, const U64 max_disp_len) const {

        assert (s < _num_shards);
        const auto &shard = _shards[s];
        assert (rank < shard.tok_cnt);

        U64 ptr = _convert_rank_to_ptr(shard, rank);
        return get_doc_by_ptr(s, ptr, max_disp_len);
    }

    void get_doc_by_rank_inplace(const size_t s, const U64 rank, const U64 max_disp_len, DocResult<T>* const thread_output) const {
        *thread_output = get_doc_by_rank(s, rank, max_disp_len);
    }

    vector<DocResult<T>> get_docs_by_ranks(const vector<pair<size_t, U64>> list_of_s_and_rank, const U64 max_disp_len) const {

        vector<DocResult<T>> docs(list_of_s_and_rank.size());
        vector<thread> threads;
        for (size_t i = 0; i < list_of_s_and_rank.size(); i++) {
            threads.emplace_back(&Engine::get_doc_by_rank_inplace, this, list_of_s_and_rank[i].first, list_of_s_and_rank[i].second, max_disp_len, &docs[i]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        return docs;
    }

    DocResult<T> get_doc_by_ptr(const size_t s, const U64 ptr, const U64 max_disp_len) const {

        assert (s < _num_shards);
        const auto &shard = _shards[s];
        assert (ptr < shard.ds_size);
        assert (ptr % sizeof(T) == 0);

        U64 max_prepend_tokens = max_disp_len / 2;
        U64 max_append_tokens = (max_disp_len + 1) / 2;

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

        U64 local_doc_ix = lo;
        U64 doc_ix = 0; for (size_t _ = 0; _ < s; _++) doc_ix += _shards[_].doc_cnt; doc_ix += local_doc_ix;

        U64 doc_start_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix) + sizeof(T); // because we want to skip the document separator
        U64 doc_end_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix + 1);
        U64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(T);

        U64 disp_start_ptr = max(doc_start_ptr, ptr < sizeof(T) * max_prepend_tokens ? (U64)0 : (ptr - sizeof(T) * max_prepend_tokens));
        U64 disp_end_ptr = min(doc_end_ptr, ptr + sizeof(T) * max_append_tokens);
        U64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(T);
        U64 needle_offset = (ptr - disp_start_ptr) / sizeof(T);

        string metadata = "";
        if (shard.mt) {
            U64 meta_start_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix);
            U64 meta_end_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix + 1);
            vector<U8> meta_chars(shard.mt + meta_start_ptr, shard.mt + meta_end_ptr);
            metadata = string(meta_chars.begin(), meta_chars.end());
        }

        vector<T> token_ids(reinterpret_cast<T*>(shard.ds + disp_start_ptr), reinterpret_cast<T*>(shard.ds + disp_end_ptr));
        if (_version == 5) {
            reverse(token_ids.begin(), token_ids.end());
        }

        return DocResult<T>{ .doc_ix = doc_ix, .doc_len = doc_len, .disp_len = disp_len, .needle_offset = needle_offset, .metadata = metadata, .token_ids = token_ids, };
    }

    void get_doc_by_ptr_inplace(const size_t s, const U64 ptr, const U64 max_disp_len, DocResult<T>* const thread_output) const {
        *thread_output = get_doc_by_ptr(s, ptr, max_disp_len);
    }

    vector<DocResult<T>> get_docs_by_ptrs(const vector<pair<size_t, U64>> list_of_s_and_ptr, const U64 max_disp_len) const {

        vector<DocResult<T>> docs(list_of_s_and_ptr.size());
        vector<thread> threads;
        for (size_t i = 0; i < list_of_s_and_ptr.size(); i++) {
            threads.emplace_back(&Engine::get_doc_by_ptr_inplace, this, list_of_s_and_ptr[i].first, list_of_s_and_ptr[i].second, max_disp_len, &docs[i]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        return docs;
    }

    DocResult<T> get_doc_by_ix(const U64 doc_ix, const U64 max_disp_len) const {

        assert (doc_ix < get_total_doc_cnt());

        size_t s = 0;
        U64 local_doc_ix = doc_ix;
        while (local_doc_ix >= _shards[s].doc_cnt) {
            local_doc_ix -= _shards[s].doc_cnt;
            s++;
        }
        const auto &shard = _shards[s];

        U64 doc_start_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix) + sizeof(T); // because we want to skip the document separator
        U64 doc_end_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix + 1);
        U64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(T);

        U64 disp_start_ptr = doc_start_ptr;
        U64 disp_end_ptr = min(doc_end_ptr, doc_start_ptr + sizeof(T) * max_disp_len);
        U64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(T);

        string metadata = "";
        if (shard.mt) {
            U64 meta_start_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix);
            U64 meta_end_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix + 1);
            vector<U8> meta_chars(shard.mt + meta_start_ptr, shard.mt + meta_end_ptr);
            metadata = string(meta_chars.begin(), meta_chars.end());
        }

        vector<T> token_ids(reinterpret_cast<T*>(shard.ds + disp_start_ptr), reinterpret_cast<T*>(shard.ds + disp_end_ptr));
        if (_version == 5) {
            reverse(token_ids.begin(), token_ids.end());
        }

        return DocResult<T>{ .doc_ix = doc_ix, .doc_len = doc_len, .disp_len = disp_len, .needle_offset = 0, .metadata = metadata, .token_ids = token_ids, };
    }

    void get_doc_by_ix_inplace(const U64 doc_ix, const U64 max_disp_len, DocResult<T>* const thread_output) const {
        *thread_output = get_doc_by_ix(doc_ix, max_disp_len);
    }

    vector<DocResult<T>> get_docs_by_ixs(const vector<U64> list_of_doc_ix, const U64 max_disp_len) const {

        vector<DocResult<T>> docs(list_of_doc_ix.size());
        vector<thread> threads;
        for (size_t i = 0; i < list_of_doc_ix.size(); i++) {
            threads.emplace_back(&Engine::get_doc_by_ix_inplace, this, list_of_doc_ix[i], max_disp_len, &docs[i]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        return docs;
    }

    DocResult<T> get_doc_by_rank_2(const size_t s, const U64 rank, const U64 needle_len, const U64 max_ctx_len) const {

        assert (s < _num_shards);
        const auto &shard = _shards[s];
        assert (rank < shard.tok_cnt);

        U64 ptr = _convert_rank_to_ptr(shard, rank);
        return get_doc_by_ptr_2(s, ptr, needle_len, max_ctx_len);
    }

    void get_doc_by_rank_inplace_2(const size_t s, const U64 rank, const U64 needle_len, const U64 max_ctx_len, DocResult<T>* const thread_output) const {
        *thread_output = get_doc_by_rank_2(s, rank, needle_len, max_ctx_len);
    }

    vector<DocResult<T>> get_docs_by_ranks_2(const vector<tuple<size_t, U64, U64, U64>> requests) const {

        vector<DocResult<T>> docs(requests.size());
        vector<thread> threads;
        for (size_t i = 0; i < requests.size(); i++) {
            threads.emplace_back(&Engine::get_doc_by_rank_inplace_2, this, get<0>(requests[i]), get<1>(requests[i]), get<2>(requests[i]), get<3>(requests[i]), &docs[i]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        return docs;
    }

    void get_docs_by_ranks_inplace_2(const vector<tuple<size_t, U64, U64, U64>> requests, vector<DocResult<T>>* const thread_output) const {
        *thread_output = get_docs_by_ranks_2(requests);
    }

    DocResult<T> get_doc_by_ptr_2(const size_t s, const U64 ptr, const U64 needle_len, const U64 max_ctx_len) const {

        assert (s < _num_shards);
        const auto &shard = _shards[s];
        assert (ptr < shard.ds_size);
        assert (ptr % sizeof(T) == 0);

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

        U64 local_doc_ix = lo;
        U64 doc_ix = 0; for (size_t _ = 0; _ < s; _++) doc_ix += _shards[_].doc_cnt; doc_ix += local_doc_ix;

        U64 doc_start_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix) + sizeof(T); // because we want to skip the document separator
        U64 doc_end_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix + 1);
        U64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(T);

        U64 disp_start_ptr = max(doc_start_ptr, ptr < sizeof(T) * max_ctx_len ? (U64)0 : (ptr - sizeof(T) * max_ctx_len));
        U64 disp_end_ptr = min(doc_end_ptr, ptr + sizeof(T) * (needle_len + max_ctx_len));
        U64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(T);
        U64 needle_offset = (ptr - disp_start_ptr) / sizeof(T);

        string metadata = "";
        if (shard.mt) {
            U64 meta_start_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix);
            U64 meta_end_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix + 1);
            vector<U8> meta_chars(shard.mt + meta_start_ptr, shard.mt + meta_end_ptr);
            metadata = string(meta_chars.begin(), meta_chars.end());
        }

        vector<T> token_ids(reinterpret_cast<T*>(shard.ds + disp_start_ptr), reinterpret_cast<T*>(shard.ds + disp_end_ptr));
        if (_version == 5) {
            reverse(token_ids.begin(), token_ids.end());
        }

        return DocResult<T>{ .doc_ix = doc_ix, .doc_len = doc_len, .disp_len = disp_len, .needle_offset = needle_offset, .metadata = metadata, .token_ids = token_ids, };
    }

    void get_doc_by_ptr_inplace_2(const size_t s, const U64 ptr, const U64 needle_len, const U64 max_ctx_len, DocResult<T>* const thread_output) const {
        *thread_output = get_doc_by_ptr_2(s, ptr, needle_len, max_ctx_len);
    }

    vector<DocResult<T>> get_docs_by_ptrs_2(const vector<tuple<size_t, U64, U64, U64>> requests) const {

        vector<DocResult<T>> docs(requests.size());
        vector<thread> threads;
        for (size_t i = 0; i < requests.size(); i++) {
            threads.emplace_back(&Engine::get_doc_by_ptr_inplace_2, this, get<0>(requests[i]), get<1>(requests[i]), get<2>(requests[i]), get<3>(requests[i]), &docs[i]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        return docs;
    }

    void get_docs_by_ptrs_inplace_2(const vector<tuple<size_t, U64, U64, U64>> requests, vector<DocResult<T>>* const thread_output) const {
        *thread_output = get_docs_by_ptrs_2(requests);
    }

    DocResult<T> get_doc_by_ix_2(const U64 doc_ix, const U64 max_disp_len) const {

        assert (doc_ix < get_total_doc_cnt());

        size_t s = 0;
        U64 local_doc_ix = doc_ix;
        while (local_doc_ix >= _shards[s].doc_cnt) {
            local_doc_ix -= _shards[s].doc_cnt;
            s++;
        }
        const auto &shard = _shards[s];

        U64 doc_start_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix) + sizeof(T); // because we want to skip the document separator
        U64 doc_end_ptr = _convert_doc_ix_to_ptr(shard, local_doc_ix + 1);
        U64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(T);

        U64 disp_start_ptr = doc_start_ptr;
        U64 disp_end_ptr = min(doc_end_ptr, doc_start_ptr + sizeof(T) * max_disp_len);
        U64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(T);

        string metadata = "";
        if (shard.mt) {
            U64 meta_start_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix);
            U64 meta_end_ptr = _convert_doc_ix_to_meta_ptr(shard, local_doc_ix + 1);
            vector<U8> meta_chars(shard.mt + meta_start_ptr, shard.mt + meta_end_ptr);
            metadata = string(meta_chars.begin(), meta_chars.end());
        }

        vector<T> token_ids(reinterpret_cast<T*>(shard.ds + disp_start_ptr), reinterpret_cast<T*>(shard.ds + disp_end_ptr));
        if (_version == 5) {
            reverse(token_ids.begin(), token_ids.end());
        }

        return DocResult<T>{ .doc_ix = doc_ix, .doc_len = doc_len, .disp_len = disp_len, .needle_offset = 0, .metadata = metadata, .token_ids = token_ids, };
    }

    void get_doc_by_ix_inplace_2(const U64 doc_ix, const U64 max_disp_len, DocResult<T>* const thread_output) const {
        *thread_output = get_doc_by_ix_2(doc_ix, max_disp_len);
    }

    vector<DocResult<T>> get_docs_by_ixs_2(const vector<tuple<U64, U64>> requests) const {

        vector<DocResult<T>> docs(requests.size());
        vector<thread> threads;
        for (size_t i = 0; i < requests.size(); i++) {
            threads.emplace_back(&Engine::get_doc_by_ix_inplace_2, this, get<0>(requests[i]), get<1>(requests[i]), &docs[i]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        return docs;
    }

    size_t get_num_shards() const {
        return _num_shards;
    }

    U64 get_tok_cnt(const size_t s) const {
        assert (s < _num_shards);
        return _shards[s].tok_cnt;
    }

    U64 get_ds_size(const size_t s) const {
        assert (s < _num_shards);
        return _shards[s].ds_size;
    }

    U64 get_total_tok_cnt() const {
        U64 total_tok_cnt = 0;
        for (const auto &shard : _shards) {
            total_tok_cnt += shard.tok_cnt;
        }
        return total_tok_cnt;
    }

    U64 get_total_doc_cnt() const {
        U64 total_doc_cnt = 0;
        for (const auto &shard : _shards) {
            total_doc_cnt += shard.doc_cnt;
        }
        return total_doc_cnt;
    }

    // get the length (in bytes) of the longest common prefix of two byte arrays
    size_t get_lcp_len(const U8* const a, const size_t len_a, const U8* const b, const size_t len_b) const {
        size_t i = 0;
        while (i < len_a && i < len_b && a[i] == b[i]) {
            i++;
        }
        return i;
    }

    void compute_longest_prefix_len_thread(const vector<T>* const input_ids, const size_t s, size_t* const out_longest_prefix_len) const {

        const auto &shard = _shards[s];

        const U8 *input_buf = reinterpret_cast<const U8*>(input_ids->data());
        U64 num_bytes = input_ids->size() * sizeof(T);
        pair<U64, U64> segment;
        pair<U64, U64> hint_segment = {0, shard.tok_cnt};
        _find_thread(s, input_buf, num_bytes, hint_segment, &segment);
        U64 start = segment.first, end = segment.second;

        if (start != end) {
            *out_longest_prefix_len = input_ids->size();
            return;
        }

        *out_longest_prefix_len = 0;
        // Inspect start-1 and start, but skip start-1 if start == 0, and skip start if start == shard.tok_cnt
        for (U64 rank = max((U64)0, start - 1); rank < min(shard.tok_cnt, start + 1); rank++) {
            U64 ptr = _convert_rank_to_ptr(shard, rank);
            size_t prefix_len = get_lcp_len(
                shard.ds + ptr, shard.ds_size - ptr,
                input_buf, input_ids->size() * sizeof(T)) / sizeof(T);
            *out_longest_prefix_len = max(*out_longest_prefix_len, prefix_len);
        }
    }

    virtual size_t compute_longest_prefix_len(const vector<T> &input_ids, const vector<T> &delim_ids, const bool enforce_bow = false) const {

        vector<thread> threads;
        vector<size_t> longest_prefix_len_by_shard(_num_shards);
        for (size_t s = 0; s < _num_shards; s++) {
            threads.emplace_back(&Engine::compute_longest_prefix_len_thread, this,
                &input_ids, s, &longest_prefix_len_by_shard[s]);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        size_t longest_prefix_len = 0;
        for (size_t s = 0; s < _num_shards; s++) {
            longest_prefix_len = max(longest_prefix_len, longest_prefix_len_by_shard[s]);
        }

        // cut off at the first occurrence of a delimiter
        if (!delim_ids.empty()) {
            for (size_t pos = 0; pos + 1 < longest_prefix_len; pos++) {
                if (std::find(delim_ids.begin(), delim_ids.end(), input_ids[pos]) != delim_ids.end()) {
                    longest_prefix_len = pos + 1;
                    break;
                }
            }
        }

        // back off until the next token is a beginning-of-word
        if (enforce_bow) {
            while (longest_prefix_len > 0) {
                if (longest_prefix_len == input_ids.size() || _bow_ids.find(input_ids[longest_prefix_len]) != _bow_ids.end()) {
                    break;
                }
                longest_prefix_len--;
            }
        }

        return longest_prefix_len;
    }

    void creativity_thread(const vector<T>* const input_ids, const size_t l, size_t* const out_r) const {

        vector<T> delim_ids;
        vector<T> suffix_ids(input_ids->begin() + l, input_ids->end());
        size_t len = compute_longest_prefix_len(suffix_ids, delim_ids, false);
        *out_r = l + len;
    }

    CreativityResult creativity(const vector<T> input_ids) const {

        vector<size_t> rs(input_ids.size());
        vector<thread> threads;
        for (size_t l = 0; l < input_ids.size(); l++) {
            threads.emplace_back(&Engine::creativity_thread, this,
                &input_ids, l, &rs[l]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        return CreativityResult{ .rs = rs, };
    }

    void compute_interesting_spans_thread(const vector<T>* const input_ids, const size_t l, const vector<T>* const delim_ids, const size_t min_len, const size_t max_cnt, const bool enforce_bow, vector<pair<PSS, FindResult>>* const out_span_find_pairs) const {

        vector<T> suffix_ids(input_ids->begin() + l, input_ids->end());
        size_t len = compute_longest_prefix_len(suffix_ids, *delim_ids, enforce_bow);
        if (len < min_len) return;
        vector<T> span_ids(input_ids->begin() + l, input_ids->begin() + l + len);
        auto find_result = find(span_ids);
        if (find_result.cnt > max_cnt) return;
        out_span_find_pairs->push_back({{l, l + len}, find_result});
    }

    vector<pair<PSS, FindResult>> compute_interesting_spans(const vector<T> &input_ids, const vector<T> &delim_ids, const size_t min_len, const size_t max_cnt, const bool enforce_bow) const {

        vector<vector<pair<PSS, FindResult>>> span_find_pairs_by_l(input_ids.size());
        for (size_t l_block = 0; l_block < input_ids.size(); l_block += _attribution_block_size) {
            vector<thread> threads;
            for (size_t l = l_block; l < min(l_block + _attribution_block_size, input_ids.size()); l++) {
                // skip if this word is not a beginning-of-word
                if (enforce_bow && _bow_ids.find(input_ids[l]) == _bow_ids.end()) continue;

                threads.emplace_back(&Engine::compute_interesting_spans_thread, this,
                    &input_ids, l, &delim_ids, min_len, max_cnt, enforce_bow, &span_find_pairs_by_l[l]);
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        vector<pair<PSS, FindResult>> flattened_span_find_pairs;
        for (const auto &span_find_pairs : span_find_pairs_by_l) {
            flattened_span_find_pairs.insert(flattened_span_find_pairs.end(), span_find_pairs.begin(), span_find_pairs.end());
        }

        vector<pair<PSS, FindResult>> filtered_span_find_pairs;
        size_t last_r = 0;
        for (const auto &span_find_pair : flattened_span_find_pairs) {
            auto &[span, find_result] = span_find_pair;
            size_t l = span.first, r = span.second;
            if (r > last_r) {
                last_r = r;
                filtered_span_find_pairs.push_back(span_find_pair);
            }
        }

        return filtered_span_find_pairs;
    }

    void get_attribution_span_thread(const pair<PSS, FindResult>* const span_find_pair, AttributionSpan* const out_attribution_span) const {

        const auto &[span, find_result] = *span_find_pair;
        vector<vector<U64>> ptrs_by_shard(_num_shards);
        for (size_t s = 0; s < _num_shards; s++) {
            const auto &shard = _shards[s];
            auto &[start, end] = find_result.segment_by_shard[s];
            ptrs_by_shard[s] = _convert_ranks_to_ptrs(shard, start, end);
        }
        vector<AttributionDoc> docs;
        for (size_t s = 0; s < _num_shards; s++) {
            for (size_t i = 0; i < ptrs_by_shard[s].size(); i++) {
                AttributionDoc doc{ .s = s, .ptr = ptrs_by_shard[s][i], };
                docs.push_back(doc);
            }
        }
        out_attribution_span->l = span.first;
        out_attribution_span->r = span.second;
        out_attribution_span->length = span.second - span.first;
        out_attribution_span->count = find_result.cnt;
        out_attribution_span->docs = docs;
    }

    AttributionResult attribute(const vector<T> input_ids, const vector<T> delim_ids, const size_t min_len, const size_t max_cnt, const bool enforce_bow) const {

        vector<pair<PSS, FindResult>> span_find_pairs = compute_interesting_spans(input_ids, delim_ids, min_len, max_cnt, enforce_bow);

        vector<AttributionSpan> attribution_spans(span_find_pairs.size());
        vector<thread> threads;
        for (size_t i = 0; i < span_find_pairs.size(); i++) {
            threads.emplace_back(&Engine::get_attribution_span_thread, this, &span_find_pairs[i], &attribution_spans[i]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        // populate the unigram_logprob_sum for each attribution_span
        for (auto &attribution_span : attribution_spans) {
            double unigram_logprob_sum = 0.0f;
            for (auto i = attribution_span.l; i < attribution_span.r; i++) {
                unigram_logprob_sum +=
                    _unigram_logprobs.find(input_ids[i]) != _unigram_logprobs.end()
                    ? _unigram_logprobs.at(input_ids[i])
                    : -10000.0;
            }
            attribution_span.unigram_logprob_sum = unigram_logprob_sum;
        }

        return AttributionResult{ .spans = attribution_spans, };
    }

private:

    void _prefetch_find(const DatastoreShard &shard, const U64 num_bytes, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.tok_cnt) return; // This might happen when lo = -1 and hi = 0
        if (_ds_prefetch_depth != 0 && depth == _ds_prefetch_depth) { // fetch ds
            U64 ptr = _convert_rank_to_ptr(shard, mi);
            madvise(shard.ds + ptr - ptr % PAGESIZE, num_bytes + ptr % PAGESIZE, MADV_WILLNEED);
        }
        if (_ds_prefetch_depth != _sa_prefetch_depth && depth == _sa_prefetch_depth) { // fetch sa
            madvise(shard.sa + mi * shard.ptr_size - mi * shard.ptr_size % PAGESIZE, shard.ptr_size + mi * shard.ptr_size % PAGESIZE, MADV_WILLNEED);
        }
        if (depth == _sa_prefetch_depth) return;
        _prefetch_find(shard, num_bytes, lo, mi, depth + 1);
        _prefetch_find(shard, num_bytes, mi, hi, depth + 1);
    }

    void _prefetch_find_2(const DatastoreShard &shard, const U64 num_bytes, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi - 1) >> 1;
        if (mi >= shard.tok_cnt) return; // This might happen when lo = 0 and hi = 0
        if (_ds_prefetch_depth != 0 && depth == _ds_prefetch_depth) { // fetch ds
            U64 ptr = _convert_rank_to_ptr(shard, mi);
            madvise(shard.ds + ptr - ptr % PAGESIZE, num_bytes + ptr % PAGESIZE, MADV_WILLNEED);
        }
        if (_ds_prefetch_depth != _sa_prefetch_depth && depth == _sa_prefetch_depth) { // fetch sa
            madvise(shard.sa + mi * shard.ptr_size - mi * shard.ptr_size % PAGESIZE, shard.ptr_size + mi * shard.ptr_size % PAGESIZE, MADV_WILLNEED);
        }
        if (depth == _sa_prefetch_depth) return;
        _prefetch_find_2(shard, num_bytes, lo, mi, depth + 1);
        _prefetch_find_2(shard, num_bytes, mi + 1, hi, depth + 1);
    }

    void _prefetch_ntd(const DatastoreShard &shard, const U64 num_bytes, const U64 lo, const U64 hi, const size_t depth = 0) const {
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.tok_cnt) return; // This might happen when lo = -1 and hi = 0
        if (_ds_prefetch_depth != 0 && depth == _ds_prefetch_depth) { // fetch ds
            U64 ptr = _version == 4 ? (_convert_rank_to_ptr(shard, mi-1) + num_bytes) : (_convert_rank_to_ptr(shard, mi-1) - sizeof(T));
            madvise(shard.ds + ptr - ptr % PAGESIZE, sizeof(T) + ptr % PAGESIZE, MADV_WILLNEED);
            ptr = _version == 4 ? (_convert_rank_to_ptr(shard, mi) + num_bytes) : (_convert_rank_to_ptr(shard, mi) - sizeof(T));
            madvise(shard.ds + ptr - ptr % PAGESIZE, sizeof(T) + ptr % PAGESIZE, MADV_WILLNEED);
        }
        if (_ds_prefetch_depth != _sa_prefetch_depth && depth == _sa_prefetch_depth) { // fetch sa
            madvise(shard.sa + (mi-1) * shard.ptr_size - (mi-1) * shard.ptr_size % PAGESIZE, 2 * shard.ptr_size + (mi-1) * shard.ptr_size % PAGESIZE, MADV_WILLNEED); // since we need both mi-1 and mi
        }
        if (depth == _sa_prefetch_depth) return;
        _prefetch_ntd(shard, num_bytes, lo, mi, depth + 1);
        _prefetch_ntd(shard, num_bytes, mi, hi, depth + 1);
    }

    void _prefetch_doc(const DatastoreShard &shard, const U64 lo, const U64 hi, const size_t depth = 0) const {
        if (_od_prefetch_depth == 0) return;
        U64 mi = (lo + hi) >> 1;
        if (mi >= shard.doc_cnt) return; // This might happen when lo = -1 and hi = 0
        if (depth == _od_prefetch_depth) { // fetch od
            madvise(shard.od + mi * sizeof(U64) - mi * sizeof(U64) % PAGESIZE, sizeof(U64) + mi * sizeof(U64) % PAGESIZE, MADV_WILLNEED);
            return;
        }
        _prefetch_doc(shard, lo, mi, depth + 1);
        _prefetch_doc(shard, mi, hi, depth + 1);
    }

    inline T _convert_ptr_to_token_id(const DatastoreShard &shard, const U64 ptr) const {
        assert (ptr % sizeof(T) == 0);
        assert (ptr <= shard.ds_size);
        if (ptr == shard.ds_size) {
            // This happens when we matched the very end of the ds.
            return _eos_token_id;
        }
        T token_id; // no need to initialize
        memcpy(&token_id, shard.ds + ptr, sizeof(T));
        // If you see \xff\xff, this actually means we're at the very end of a document.
        if (token_id == _doc_sep_id) token_id = _eos_token_id;
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

    T _eos_token_id;
    T _vocab_size;
    size_t _version;
    bool _load_to_ram;
    size_t _ds_prefetch_depth;
    size_t _sa_prefetch_depth;
    size_t _od_prefetch_depth;
    set<T> _bow_ids;
    size_t _attribution_block_size;
    T _doc_sep_id;
    vector<U8> _doc_sep;
    size_t _num_shards;
    vector<DatastoreShard> _shards;
    map<string, vector<DatastoreShard>> _new_shards_by_index_dir;
    map<T, double> _unigram_logprobs;

    friend class EngineDiff<T>;
};

template <typename T>
class EngineDiff : public Engine<T> {

public:

    EngineDiff(
        const vector<string> index_dirs, const vector<string> index_dirs_diff, const T eos_token_id, const T vocab_size, const size_t version,
        const bool load_to_ram, const size_t ds_prefetch_depth, const size_t sa_prefetch_depth, const size_t od_prefetch_depth,
        const set<T> bow_ids, const size_t attribution_block_size, const bool precompute_unigram_logprobs,
        map<string, vector<DatastoreShard>> prev_shards_by_index_dir)
        : Engine<T>(index_dirs, eos_token_id, vocab_size, version, load_to_ram, ds_prefetch_depth, sa_prefetch_depth, od_prefetch_depth, bow_ids, attribution_block_size, precompute_unigram_logprobs, prev_shards_by_index_dir),
          _engine_diff(make_unique<Engine<T>>(index_dirs_diff, eos_token_id, vocab_size, version, load_to_ram, ds_prefetch_depth, sa_prefetch_depth, od_prefetch_depth, bow_ids, attribution_block_size, precompute_unigram_logprobs, prev_shards_by_index_dir)) {}

    // The shape of returned document results is identical to the shape of input requests. Blocked documents are marked and have an empty token_ids.
    vector<vector<DocResult<T>>> get_docs_by_ptrs_2_grouped(const vector<tuple<vector<pair<size_t, U64>>, vector<T>, U64, U64>> requests) const {

        vector<vector<DocResult<T>>> docs_main_by_request(requests.size());
        vector<vector<DocResult<T>>> docs_diff_by_request(requests.size());
        vector<thread> threads;
        for (size_t r = 0; r < requests.size(); r++) {
            const auto &request = requests[r];
            vector<tuple<size_t, U64, U64, U64>> requests_main;
            for (const auto &[s, ptr] : get<0>(request)) {
                requests_main.emplace_back(s, ptr, get<2>(request), get<3>(request));
            }
            threads.emplace_back(&Engine<T>::get_docs_by_ptrs_inplace_2, this, requests_main, &docs_main_by_request[r]);

            const auto &span_ids = get<1>(request);
            auto find_result_diff = _engine_diff->find(span_ids);
            vector<tuple<size_t, U64, U64, U64>> requests_diff;
            for (size_t s = 0; s < _engine_diff->get_num_shards(); s++) {
                auto [start, end] = find_result_diff.segment_by_shard[s];
                for (U64 rank = start; rank < end; rank++) {
                    requests_diff.emplace_back(s, rank, get<2>(request), get<3>(request));
                }
            }
            threads.emplace_back(&Engine<T>::get_docs_by_ranks_inplace_2, _engine_diff.get(), requests_diff, &docs_diff_by_request[r]);
        }
        for (auto &thread : threads) {
            thread.join();
        }

        for (size_t r = 0; r < requests.size(); r++) {
            auto &docs_main = docs_main_by_request[r];
            auto &docs_diff = docs_diff_by_request[r];
            for (auto &doc_main : docs_main) {
                // if doc_main.token_ids match any in the diff index, block this document
                if (any_of(docs_diff.begin(), docs_diff.end(), [&](const DocResult<T> &doc_diff) {
                    return doc_main.token_ids == doc_diff.token_ids;
                })) {
                    doc_main.token_ids.clear();
                    doc_main.blocked = true;
                }
            }
        }

        return docs_main_by_request;
    }

    size_t compute_longest_prefix_len(const vector<T> &input_ids, const vector<T> &delim_ids, const bool enforce_bow = false) const override {

        size_t longest_prefix_len = Engine<T>::compute_longest_prefix_len(input_ids, delim_ids, enforce_bow);

        // consider docs in the diff index
        while (longest_prefix_len > 0) {
            auto count_main = Engine<T>::count({input_ids.begin(), input_ids.begin() + longest_prefix_len}).count;
            auto count_diff = _engine_diff->count({input_ids.begin(), input_ids.begin() + longest_prefix_len}).count;
            if (count_main > count_diff) break;
            longest_prefix_len--;
        }

        return longest_prefix_len;
    }

private:

    unique_ptr<Engine<T>> _engine_diff;
};
