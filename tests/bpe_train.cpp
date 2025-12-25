#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>

namespace bpe {
    // 把 string 当成字节容器
    // Using string to store bytes
    using Bytes = std::string;

    struct PairHash {
        std::size_t operator()(const std::pair<int, int> &p) const {
            return std::hash<int>()(p.first) ^ std::hash<int>()(p.second);
        }
    };

    // Return type: vocab map and merges list
    struct Result {
        std::map<int, Bytes> vocab;
        std::vector<std::pair<Bytes, Bytes> > merges;
    };

    Result train(
        const std::vector<std::string> &distinct_words,
        const std::vector<int> &counts,
        int vocab_size,
        const std::vector<std::string> &special_tokens
    ) {
        std::map<int, Bytes> vocab;
        std::vector<std::pair<Bytes, Bytes> > merges;

        // 1. 初始化词汇表 vocab[97] = "a"
        for (int i = 0; i < 256; ++i) {
            vocab[i] = Bytes(1, (char) i);
        }

        int next_id = 256;

        // 2. 创建词到词汇表的 id 映射，如 word: 'abc', token_ids: 97,98,99
        std::vector<std::vector<int> > words;
        words.reserve(distinct_words.size());

        for (const auto &word: distinct_words) {
            std::vector<int> token_ids;
            token_ids.reserve(word.size());
            for (auto &ch: word) {
                // windows 中字符默认是 signed char
                token_ids.emplace_back(static_cast<unsigned char>(ch));
            }
            words.emplace_back(token_ids);
        }

        // 3. 循环合并
        while (vocab.size() + special_tokens.size() < (size_t) vocab_size) {
            std::map<std::pair<int, int>, int> pair_counts;

            for (size_t i = 0; i < words.size(); ++i) {
                const auto &word = words[i];
                int count = counts[i];
                // 字节大小小于2，可直接忽略
                if (word.size() < 2) continue;

                // 统计相邻字节出现次数
                for (size_t j = 0; j < word.size() - 1; ++j) {
                    pair_counts[{word[j], word[j + 1]}] += count;
                }
            }

            if (pair_counts.empty()) break;

            // Find best pair (max count, break ties by lexicographically largest (STR1, STR2) tuple)
            int max_val = -1;
            std::pair<int, int> best_pair = {-1, -1};

            // 找到出现最高频率的 pair 对
            for (auto &pair_count: pair_counts) {
                int count = pair_count.second;
                std::pair<int, int> p = pair_count.first;

                if (count > max_val) {
                    max_val = count;
                    best_pair = p;
                } else if (count == max_val) {
                    // 比较字典序，不能用 token id 比较，因为加入的新词 token id 是自增分配的，无规律
                    // 如果按照 ASCII 码简单相加的值来分配tokenid，是不是可以直接比较了
                    // 不能，有可能相加的和正好相同，哈希也有可能碰撞
                    const Bytes &s1_first = vocab[best_pair.first];
                    const Bytes &s1_second = vocab[best_pair.second];

                    const Bytes &s2_first = vocab[p.first];
                    const Bytes &s2_second = vocab[p.second];

                    if (s2_first > s1_first || (s2_first == s1_first && s2_second > s1_second)) {
                        best_pair = p;
                    }
                }
            }

            if (max_val < 0) break;

            // 添加新词进词汇表
            int new_id = next_id++; // 自增 id 作为新词 id
            Bytes new_token = vocab[best_pair.first] + vocab[best_pair.second];
            vocab[new_id] = new_token;
            // 添加新词对
            merges.emplace_back(vocab[best_pair.first], vocab[best_pair.second]);

            // 更新词库
            for (auto &word: words) {
                if (word.size() < 2) continue;

                std::vector<int> new_word;
                new_word.reserve(word.size());

                for (size_t j = 0; j < word.size(); ++j) {
                    if (j < word.size() - 1 && word[j] == best_pair.first && word[j + 1] == best_pair.second) {
                        new_word.emplace_back(new_id);
                        ++j; // 匹配一个字节对，跳到下下一个
                    } else {
                        new_word.emplace_back(word[j]);
                    }
                }

                word = std::move(new_word);
            }
        }

        // 4. 特殊标记也加入词汇表
        for (const auto &st: special_tokens) {
            vocab[next_id++] = st;
        }

        return {vocab, merges};
    }
}
