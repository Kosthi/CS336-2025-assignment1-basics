#include "bpe.hpp"

#ifdef BUILD_PYTHON_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <assert.h>
#include <queue>
#include <omp.h>
#include <chrono>

namespace bpe {
    // 自定义哈希函数，用于将std::pair<int, int>作为unordered_map的键
    struct PairHash {
        std::size_t operator()(const std::pair<int, int> &p) const {
            // 将两个int的哈希值组合成一个哈希值
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    // BPE训练器主类
    class BPETrainer {
    private:
        // 词汇表：从token ID到字节串的映射
        std::map<int, Bytes> vocab;

        // 单词列表：每个单词表示为token ID的序列
        std::vector<std::vector<int> > words;

        // 每个单词的频率计数
        std::vector<int> word_counts;

        // 下一个可用的token ID
        int next_id = 0;

        // 记录每个pair的频率
        std::unordered_map<std::pair<int, int>, int, PairHash> pair_counts;

        // 优先级队列中使用的pair信息结构
        struct PairInfo {
            std::pair<int, int> pair; // token对
            int count; // 该pair出现的频率
        };

        // 优先级队列的比较器，用于确定哪个pair应该优先合并
        struct PairComparator {
            const std::map<int, Bytes> *vocab; // 指向词汇表的指针

            explicit PairComparator(const std::map<int, Bytes> *v) : vocab(v) {
            }

            bool operator()(const PairInfo &a, const PairInfo &b) const {
                // 优先频率高的pair（注意：这是最大堆，返回true表示a的优先级低于b）
                if (a.count != b.count) return a.count < b.count;

                // 频率相同时，按照pair对应的字符串的字典序比较
                // 先比较第一个token的字符串
                if (a.pair.first != b.pair.first)
                    return vocab->at(a.pair.first) < vocab->at(b.pair.first);

                // 第一个token相同时，比较第二个token的字符串
                return vocab->at(a.pair.second) < vocab->at(b.pair.second);
            }
        };

        // 优先级队列，用于快速找到频率最高的pair
        std::priority_queue<PairInfo, std::vector<PairInfo>, PairComparator> pair_queue;

        // 索引结构：记录每个pair在哪些单词的什么位置出现
        // 键：pair
        // 值：向量，每个元素是(单词索引, 在该单词中的位置)
        // 注意：这里记录的是pair中第一个token的位置
        std::unordered_map<std::pair<int, int>,
            std::vector<std::pair<size_t, size_t> >, // (word_idx, position)
            PairHash> pair_positions;

        struct ThreadLocal {
            std::unordered_map<std::pair<int, int>, int, PairHash> local_counts;
            std::unordered_map<std::pair<int, int>,
                std::vector<std::pair<size_t, size_t> >,
                PairHash> local_positions;
        };

    public:
        // 构造函数：初始化BPE训练器
        BPETrainer(const std::vector<std::string> &distinct_words,
                   const std::vector<int> &counts)
            : next_id(256), pair_queue(PairComparator(&vocab)) {
            // 初始化词汇表：前256个ID对应单个字节
            for (int i = 0; i < 256; ++i) {
                vocab[i] = Bytes(1, static_cast<char>(i));
            }

            std::cout << "BPETrainer: distinct_words size = " << distinct_words.size() << std::endl;

            words.resize(distinct_words.size());
            word_counts = counts;

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < distinct_words.size(); ++i) {
                const auto &word = distinct_words[i];
                auto &token_ids = words[i];
                token_ids.reserve(word.size());
                for (char ch: word) {
                    token_ids.emplace_back(static_cast<unsigned char>(ch));
                }
            }

            std::cout << "BPETrainer: converted words to token ids" << std::endl;

            std::cout << "BPETrainer: start build_initial_counts" << std::endl;
            build_initial_counts();
        }

    private:
        // 辅助函数：在单词中从指定位置向后查找下一个有效token的位置
        static int get_next_pos(const std::vector<int> &word, int pos) {
            for (int i = pos + 1; i < word.size(); ++i) {
                if (word[i] != -1) return i; // -1表示该token已被合并
            }
            return -1; // 没有找到有效token
        }

        // 辅助函数：在单词中从指定位置向前查找上一个有效token的位置
        static int get_prev_pos(const std::vector<int> &word, int pos) {
            for (int i = pos - 1; i >= 0; --i) {
                if (word[i] != -1) return i; // -1表示该token已被合并
            }
            return -1; // 没有找到有效token
        }

        // build_initial_counts 串行版本
        void single_build_initial_counts() {
            // 清空现有数据
            pair_counts.clear();
            pair_positions.clear();

            std::priority_queue<PairInfo, std::vector<PairInfo>, PairComparator>
                    empty_queue{PairComparator(&vocab)};
            pair_queue.swap(empty_queue);

            // 直接串行处理
            for (size_t i = 0; i < words.size(); ++i) {
                const auto &word = words[i];
                int count = word_counts[i];

                if (word.empty()) continue;

                // 直接遍历所有有效位置
                for (size_t curr = 0; curr < word.size(); ++curr) {
                    if (word[curr] == -1) continue;

                    // 找到下一个有效位置
                    size_t next = curr + 1;
                    while (next < word.size() && word[next] == -1) {
                        ++next;
                    }

                    if (next >= word.size()) break;

                    std::pair<int, int> p = {word[curr], word[next]};
                    pair_counts[p] += count;
                    pair_positions[p].emplace_back(i, curr);

                    curr = next - 1; // 循环会自增
                }
            }

            // 将所有pair添加到优先级队列中
            for (const auto &entry: pair_counts) {
                pair_queue.push({entry.first, entry.second});
            }
        }

        // 构建初始的pair统计信息 并行版本
        void build_initial_counts() {
            // 清空现有数据
            pair_counts.clear();
            pair_positions.clear();

            // 清空优先级队列（通过与空队列交换）
            std::priority_queue<PairInfo, std::vector<PairInfo>, PairComparator>
                    empty_queue{PairComparator(&vocab)};
            pair_queue.swap(empty_queue);

            int hw_threads = omp_get_num_procs();
            int desired_threads = hw_threads;
            if (desired_threads > 24) desired_threads = 24;
            if (desired_threads < 1) desired_threads = 1;
            omp_set_num_threads(desired_threads);

            std::vector<ThreadLocal> thread_locals;

#pragma omp parallel
            {
#pragma omp single
                {
                    int actual_threads = omp_get_num_threads();
                    std::cout << "build_initial_counts using " << actual_threads << " threads" << std::endl;
                    thread_locals.resize(actual_threads);
                }

                ThreadLocal &local = thread_locals[omp_get_thread_num()];
                local.local_counts.clear();
                local.local_positions.clear();

#pragma omp for schedule(static)
                for (size_t i = 0; i < words.size(); ++i) {
                    const auto &word = words[i];
                    int count = word_counts[i];

                    if (word.empty()) continue;

                    int curr = -1;
                    for (size_t k = 0; k < word.size(); ++k) {
                        if (word[k] != -1) {
                            curr = k;
                            break;
                        }
                    }

                    if (curr == -1) continue;

                    while (true) {
                        int next = get_next_pos(word, curr);
                        if (next == -1) break;

                        std::pair<int, int> p = {word[curr], word[next]};

                        local.local_counts[p] += count;
                        local.local_positions[p].emplace_back(i, curr);

                        curr = next;
                    }
                }

#pragma omp critical
                {
                    for (const auto &entry: local.local_counts) {
                        pair_counts[entry.first] += entry.second;
                    }
                    for (auto &entry: local.local_positions) {
                        auto &vec = pair_positions[entry.first];
                        auto &local_vec = entry.second;
                        vec.insert(vec.end(), local_vec.begin(), local_vec.end());
                    }
                }
            }

            // 将所有pair添加到优先级队列中
            for (const auto &entry: pair_counts) {
                pair_queue.push({entry.first, entry.second});
            }
        }

        // 合并一对token并更新相关统计信息
        void merge_and_update(size_t word_idx, size_t pos, int new_id, const std::pair<int, int> &old_pair) {
            auto &word = words[word_idx];
            int count = word_counts[word_idx]; // 当前单词的频率

            // 获取相邻token的位置
            int prev_pos = get_prev_pos(word, pos); // 前一个token的位置
            int next_pos = get_next_pos(word, pos); // 当前pair中第二个token的位置

            // 1. 删除左相邻pair（prev_token, old_first_token）
            if (prev_pos != -1) {
                std::pair<int, int> left_pair = {word[prev_pos], old_pair.first};
                pair_counts[left_pair] -= count; // 减少频率
                if (pair_counts[left_pair] <= 0) {
                    pair_counts.erase(left_pair); // 如果频率为0，删除该pair
                } else {
                    // 更新后的频率推入队列（旧条目通过惰性删除处理）
                    pair_queue.push({left_pair, pair_counts[left_pair]});
                }
            }

            // 找到第三个token的位置（跳过被合并的第二个token）
            int third_pos = get_next_pos(word, next_pos);

            // 2. 删除右相邻pair（old_second_token, third_token）
            if (third_pos != -1) {
                std::pair<int, int> right_pair = {old_pair.second, word[third_pos]};
                pair_counts[right_pair] -= count;
                if (pair_counts[right_pair] <= 0) {
                    pair_counts.erase(right_pair);
                } else {
                    pair_queue.push({right_pair, pair_counts[right_pair]});
                }
            }

            // 3. 合并tokens：用new_id替换第一个token，将第二个token标记为已删除(-1)
            word[pos] = new_id;
            word[next_pos] = -1;

            // 4. 添加新的左相邻pair（prev_token, new_token）
            if (prev_pos != -1) {
                std::pair<int, int> new_left = {word[prev_pos], new_id};
                pair_counts[new_left] += count; // 增加频率
                pair_positions[new_left].emplace_back(word_idx, prev_pos); // 记录位置
                pair_queue.push({new_left, pair_counts[new_left]}); // 推入队列
            }

            // 5. 添加新的右相邻pair（new_token, third_token）
            if (third_pos != -1) {
                std::pair<int, int> new_right = {new_id, word[third_pos]};
                pair_counts[new_right] += count;
                pair_positions[new_right].emplace_back(word_idx, pos); // 注意：这里记录的是新token的位置
                pair_queue.push({new_right, pair_counts[new_right]});
            }
        }

        // 合并指定的pair
        void merge_pair(const std::pair<int, int> &pair, int new_id,
                        std::vector<std::pair<Bytes, Bytes> > &merges) {
            // 1. 更新词汇表：为新token创建字符串表示
            vocab[new_id] = vocab[pair.first] + vocab[pair.second];
            // 记录这次合并（用于最终输出）
            merges.emplace_back(vocab[pair.first], vocab[pair.second]);

            // 2. 获取该pair在所有单词中出现的位置
            auto positions = pair_positions[pair];

            // 3. 遍历所有出现位置并合并
            for (const auto &[word_idx, pos]: positions) {
                auto &word = words[word_idx];

                // 验证当前位置的pair是否仍然是我们要合并的pair
                // （因为之前的合并可能已经改变了这个位置）
                if (word[pos] != pair.first) continue;
                int next_pos = get_next_pos(word, pos);
                if (next_pos == -1 || word[next_pos] != pair.second) continue;

                // 合并并更新统计信息
                merge_and_update(word_idx, pos, new_id, pair);
            }

            // 4. 清理已合并pair的记录
            pair_positions.erase(pair);
            pair_counts.erase(pair);
        }

    public:
        // 主训练函数
        Result train(int vocab_size, const std::vector<std::string> &special_tokens) {
            std::vector<std::pair<Bytes, Bytes> > merges; // 记录所有的合并操作

            // 计算目标词汇表大小（减去特殊token）
            // 注意：需要确保vocab_size大于特殊token的数量
            size_t target_base_vocab_size = vocab_size - special_tokens.size();

            auto start_time = std::chrono::steady_clock::now();
            int merge_count = 0;

            std::cout << "BPETrainer::train enter, target_base_vocab_size=" << target_base_vocab_size
                      << " current_vocab_size=" << vocab.size() << std::endl;

            // 持续合并直到达到目标词汇表大小
            while (vocab.size() < target_base_vocab_size) {
                // 如果优先级队列为空，重新构建统计信息
                if (pair_queue.empty()) {
                    // single_build_initial_counts();
                    build_initial_counts();

                    if (pair_queue.empty()) break; // 如果没有可合并的pair，结束
                }

                PairInfo best_info;
                bool found_valid = false;

                // 从优先级队列中查找有效的、频率最高的pair
                // 由于惰性删除，队列中可能包含过期的条目，相比重建队列只需要 O(KlogN)
                while (!pair_queue.empty()) {
                    best_info = pair_queue.top();
                    pair_queue.pop();

                    // 检查该pair是否仍然有效（频率与当前统计匹配）
                    auto it = pair_counts.find(best_info.pair);
                    if (it != pair_counts.end() && it->second == best_info.count) {
                        found_valid = true;
                        break;
                    }
                }

                // 如果找不到有效pair，可以尝试重新构建统计信息
                if (!found_valid) {
                    // single_build_initial_counts();
                    build_initial_counts();
                    if (pair_queue.empty()) break; // 如果没有可合并的pair，结束
                }

                // 合并最佳的pair
                int new_id = next_id++; // 分配新的token ID
                merge_pair(best_info.pair, new_id, merges);

                ++merge_count;
                if (merge_count % 1000 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                    std::cout << "[bpe] merges=" << merge_count
                              << " vocab_size=" << vocab.size()
                              << " elapsed=" << elapsed << "s" << std::endl;
                }
            }

            // 添加特殊tokens到词汇表
            for (const auto &token: special_tokens) {
                vocab[next_id++] = token;
            }

            // 返回训练结果：词汇表和合并记录
            return {vocab, merges};
        }
    };

    // 对外的训练接口函数
    Result train(
        const std::vector<std::string> &distinct_words, // 去重后的单词列表
        const std::vector<int> &counts, // 对应单词的频率
        int vocab_size, // 目标词汇表大小
        const std::vector<std::string> &special_tokens // 特殊token列表
    ) {
        // 创建训练器实例并开始训练
        BPETrainer trainer(distinct_words, counts);
        return trainer.train(vocab_size, special_tokens);
    }

#ifdef BUILD_PYTHON_MODULE
    class BPEMerger {
    public:
        BPEMerger() = default;

        BPEMerger(
            const std::vector<std::pair<Bytes, Bytes>> &merges,
            const std::unordered_map<std::string, int> &vocab_to_id,
            const std::array<int, 256> &byte_to_vocab_id
        ) : byte_to_vocab_id_(byte_to_vocab_id) {
            pair_to_rank_.reserve(merges.size() * 2);
            pair_to_vocab_id_.reserve(merges.size() * 2);

            // 预存 a、b的哈希与合并后的映射 token id
            for (uint32_t rank = 0; rank < merges.size(); ++rank) {
                const auto &a = merges[rank].first;
                const auto &b = merges[rank].second;

                auto a_it = vocab_to_id.find(a);
                auto b_it = vocab_to_id.find(b);
                if (a_it == vocab_to_id.end() || b_it == vocab_to_id.end()) {
                    continue;
                }
                uint32_t a_id = static_cast<uint32_t>(a_it->second);
                uint32_t b_id = static_cast<uint32_t>(b_it->second);

                Bytes merged = a;
                merged.append(b);
                auto merged_it = vocab_to_id.find(merged);
                if (merged_it == vocab_to_id.end()) {
                    continue;
                }
                int merged_id = merged_it->second;

                uint64_t key = pair_key_(a_id, b_id);
                pair_to_rank_.emplace(key, static_cast<int>(rank));
                pair_to_vocab_id_.emplace(key, merged_id);
            }
        }

        std::vector<int> apply(const Bytes &token_bytes) const {
            if (token_bytes.empty()) {
                return {};
            }

            std::vector<int> tokens;
            tokens.reserve(token_bytes.size());
            // 编码为 utf-8后，必然可以用 0-255 表示一个字节
            for (unsigned char c : token_bytes) {
                int id = byte_to_vocab_id_[c];
                tokens.emplace_back(id);
            }

            while (tokens.size() > 1) {
                int best_rank = std::numeric_limits<int>::max();
                size_t best_i = static_cast<size_t>(-1);
                uint64_t best_key = 0;

                for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                    uint64_t key = pair_key_(tokens[i], tokens[i + 1]);
                    auto it = pair_to_rank_.find(key);
                    if (it == pair_to_rank_.end()) {
                        continue;
                    }
                    int rank = it->second;
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_i = i;
                        best_key = key;
                    }
                }

                if (best_i == static_cast<size_t>(-1)) {
                    break;
                }

                auto merged_it = pair_to_vocab_id_.find(best_key);
                if (merged_it == pair_to_vocab_id_.end()) {
                    break;
                }

                tokens[best_i] = static_cast<uint32_t>(merged_it->second);
                tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
            }

            return tokens;
        }

    private:
        static uint64_t pair_key_(uint32_t left, uint32_t right) {
            return (static_cast<uint64_t>(left) << 32) | static_cast<uint64_t>(right);
        }

        std::unordered_map<uint64_t, int> pair_to_rank_;
        std::unordered_map<uint64_t, int> pair_to_vocab_id_;
        std::array<int, 256> byte_to_vocab_id_{};
    };

    class BPETokenEncoder {
    public:
        BPETokenEncoder(
            const py::dict &vocab,
            const std::vector<std::pair<Bytes, Bytes>> &merges,
            const std::vector<std::string> &special_tokens
        ) {
            byte_to_vocab_id_.fill(-1);
            vocab_to_id_.reserve(static_cast<size_t>(vocab.size()) * 2);
            for (auto &item : vocab) {
                int id = py::cast<int>(item.first);
                std::string token = py::cast<py::bytes>(item.second);
                if (token.size() == 1) {
                    byte_to_vocab_id_[static_cast<unsigned char>(token[0])] = id;
                }
                vocab_to_id_.emplace(std::move(token), id);
            }
            special_set_.reserve(special_tokens.size() * 2 + 1);
            for (const auto &s : special_tokens) {
                special_set_.insert(s);
            }
            merger_ = BPEMerger(merges, vocab_to_id_, byte_to_vocab_id_);
        }

        std::vector<int> encode_tokens(const std::vector<std::string> &tokens) const {
            std::vector<int> out;
            out.reserve(tokens.size() * 2);
            for (const auto &tok_str : tokens) {
                if (special_set_.find(tok_str) != special_set_.end()) {
                    auto it = vocab_to_id_.find(tok_str);
                    if (it != vocab_to_id_.end()) {
                        out.emplace_back(it->second);
                    }
                    continue;
                }
                encode_token_bytes_(tok_str, out);
            }
            return out;
        }

    private:
        void encode_token_bytes_(const std::string &token_bytes, std::vector<int> &out) const {
            auto ids = merger_.apply(token_bytes);
            out.insert(out.end(), ids.begin(), ids.end());
        }

        BPEMerger merger_;
        std::array<int, 256> byte_to_vocab_id_{};
        std::unordered_map<std::string, int> vocab_to_id_;
        std::unordered_set<std::string> special_set_;
    };
#endif
}

// Example usage
int main() {
    std::vector<std::string> words = {"low", "lower", "widest", "newest"};
    std::vector<int> counts = {5, 2, 3, 6}; // Frequencies of distinct words

    std::vector<std::string> special_tokens = {"<pad>"};

    // Train with optimized version
    auto result = bpe::train(words, counts, 263, special_tokens);

    std::cout << "Vocabulary size: " << result.vocab.size() << std::endl;
    std::cout << "Number of merges: " << result.merges.size() << std::endl;

    // Print first 10 merges
    for (int i = 0; i < std::min(10, (int) result.merges.size()); ++i) {
        std::cout << "Merge " << i << ": "
                << result.merges[i].first << " + "
                << result.merges[i].second << std::endl;
    }

    return 0;
}

#ifdef BUILD_PYTHON_MODULE
PYBIND11_MODULE(bpe, m) {
    m.doc() = "BPE training module";

    py::class_<bpe::Result>(m, "Result")
        .def_property_readonly("vocab", [](const bpe::Result &r) {
            py::dict d;
            for (const auto &kv : r.vocab) {
                d[py::int_(kv.first)] = py::bytes(kv.second);
            }
            return d;
        })
        .def_property_readonly("merges", [](const bpe::Result &r) {
            py::list l;
            for (const auto &p : r.merges) {
                l.append(py::make_tuple(py::bytes(p.first), py::bytes(p.second)));
            }
            return l;
        });

    m.def("train", &bpe::train, "Train BPE",
          py::arg("distinct_words"),
          py::arg("counts"),
          py::arg("vocab_size"),
          py::arg("special_tokens"));

    py::class_<bpe::BPETokenEncoder>(m, "BPETokenEncoder")
        .def(py::init<const py::dict &, const std::vector<std::pair<bpe::Bytes, bpe::Bytes>> &, const std::vector<std::string> &>(),
             py::arg("vocab"),
             py::arg("merges"),
             py::arg("special_tokens"))
        .def("encode_tokens", &bpe::BPETokenEncoder::encode_tokens, py::arg("tokens"));
}
#endif
