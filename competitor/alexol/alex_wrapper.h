#ifndef ALEX_WRAPPER_H
#define ALEX_WRAPPER_H
#include "src/alex.h"
#include"../indexInterface.h"
#include <unordered_set>
#include <cmath>

inline int counter = 0;


template<class KEY_TYPE, class PAYLOAD_TYPE>
class alexolInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    void init(Param *param = nullptr) {}

    void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool remove(KEY_TYPE key, Param *param = nullptr);

    size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr);

    long long memory_consumption() {
        long long bytes = index.model_size() + index.data_size();
#if ENABLE_CONTROLLER
        bytes += static_cast<long long>(
            alexol::g_alex_controller.metadata_memory_bytes());
#endif
        return bytes;
    }

    ControllerStatistics controller_statistics() const override {
        ControllerStatistics result;
#if ENABLE_CONTROLLER
        const auto stats = alexol::g_alex_controller.statistics();
        result.hash_decisions = stats.hash_decisions;
        result.cooling_decisions = stats.cooling_decisions;
        result.hash_completions = stats.hash_completions;
        result.cooling_completions = stats.cooling_completions;
        result.cooling_bytes_reclaimed = stats.cooling_bytes_reclaimed;
        result.inspection_cycles = stats.inspection_cycles;
        result.hot_node_inspections = stats.hot_node_inspections;
        result.cold_node_inspections = stats.cold_node_inspections;
        result.registration_attempts = stats.registration_attempts;
        result.registered_nodes = stats.registered_nodes;
        result.rejected_by_filter = stats.rejected_by_filter;
        result.monitored_nodes = stats.monitored_nodes;
        result.hash_nodes = stats.hash_nodes;
#endif
        return result;
    }
#if ENABLE_CONTROLLER
    void print_controller_stats() override {
        alexol::g_alex_controller.print_statistics();
    }
    void start_controller() override {
        alexol::g_alex_controller.start();
    }
    void stop_controller() override {
        alexol::g_alex_controller.stop();
        index.reclaim_retired_storage();
    }
#endif

    void compress_cold_zone(KEY_TYPE end_key, const std::string& csv_path = "compression_memory.csv") {
        std::cout << "Boundry: " << end_key << std::endl;
        data_node_type* data_node = index.first_data_node();
        if (!data_node) return;

        std::ofstream csv_file(csv_path, std::ios::app);
        // 仅在文件为空时写入表头
        if (csv_file.tellp() == 0) {
            csv_file << "compressed_nodes,memory_size_bytes" << std::endl;
        }

        int total_compressed = 0;
        int batch_count = 0;
        const int BATCH_SIZE = 400;

        long long initial_size = memory_consumption();
        csv_file << "0," << initial_size << std::endl;
        std::cout << "Initial memory size: " << initial_size << " bytes" << std::endl;
#if ENABLE_CONTROLLER
        while (data_node != nullptr) {
            if (data_node->num_keys_ > 0) {
                KEY_TYPE node_last = data_node->last_key();

                if (node_last < end_key) {
                    data_node->compress_gaps();
                    total_compressed++;
                    batch_count++;

                    // if (batch_count >= BATCH_SIZE) {
                    //     long long current_size = memory_consumption();
                    //     csv_file << total_compressed << "," << current_size << std::endl;
                    //     batch_count = 0;
                    // }
                }
            }
            data_node = data_node->next_leaf_;
        }

        long long final_size = memory_consumption();
        csv_file << total_compressed << "," << final_size << std::endl;

        csv_file.close();

        std::cout << "Compressed " << total_compressed << " cold nodes" << std::endl;
        std::cout << "Final memory: " << final_size << " bytes" << std::endl;
        std::cout << "Memory reduced: " << (initial_size - final_size) << " bytes" << std::endl;
        std::cout << "Compression log saved to " << csv_path << std::endl;
#endif
    }


    void optimize_targeted_zones(KEY_TYPE start_key, KEY_TYPE end_key, std::string file_name = "compression_memory.csv",
        bool adjust_density = false) {
            if (adjust_density) return;
#if ENABLE_CONTROLLER
        // if (!adjust_density) {
        data_node_type* data_node = index.first_data_node();
        if (!data_node) {
            std::cout << "Warning: first datanode is null" << std::endl;
            return;
        }

        TSCNS tn;
        tn.init();

        int converted_count = 0;
        int skipped_count = 0;
        long long converted_size = 0;
        double total_convert_time = 0.0;
        size_t total_keys = 0;

        std::cout << "Scanning nodes in range [" << start_key << ", " << end_key << "]..." << std::endl;

        // 遍历所有叶子节点
        while (data_node != nullptr) {
            bool in_range = false;

            if (data_node->num_keys_ > 0) {
                KEY_TYPE node_first_key = data_node->first_key();
                KEY_TYPE node_last_key = data_node->last_key();

                in_range = (node_first_key >= start_key) && (node_last_key <= end_key);
            }

            if (in_range) {
                if (data_node->is_hash_node_) {
                    skipped_count++;
                    data_node = data_node->next_leaf_;
                    continue;
                }
                // data_node->save_stats_to_csv(file_name);
                auto convert_start = tn.rdtsc();
                auto converted = data_node->convert_to_hash_node();
                if (converted) converted_count++;
                auto convert_end = tn.rdtsc();

                auto convert_time = tn.tsc2ns(convert_end) - tn.tsc2ns(convert_start);
                total_convert_time += convert_time;
                total_keys += data_node->num_keys_;
                converted_size += data_node->data_size();
            }

            data_node = data_node->next_leaf_;
        }

        std::cout << "\n=== Zone Optimization Summary ===" << std::endl;
        std::cout << "Range: [" << start_key << ", " << end_key << "]" << std::endl;
        std::cout << "Converted nodes: " << converted_count << std::endl;
        std::cout << "Skipped nodes: " << skipped_count << std::endl;
        std::cout << "Total keys in converted nodes: " << total_keys << std::endl;
        std::cout << "Total converted size: " << converted_size << " bytes" << std::endl;
        std::cout << "Total conversion time: " << total_convert_time / 1e6 << " ms" << std::endl;
        if (converted_count > 0) {
            std::cout << "Average conversion time per node: " << total_convert_time / converted_count << " ns" << std::endl;
        }
        std::cout << "==================================\n" << std::endl;
        // }
#endif
        // std::ofstream csv_file(file_name, std::ios::app);
        // if (csv_file.tellp() == 0) {
        //     csv_file << "compressed_nodes,memory_size_bytes" << std::endl;
        // }
        // long long current_size = memory_consumption();
        // csv_file << 0 << "," << current_size << std::endl;
    }
// #endif

   void generate_attack_keys(
    std::vector<KEY_TYPE>& attack_keys,
    KEY_TYPE original_min,
    KEY_TYPE original_max,
    size_t original_size,
    size_t num_attack_keys = 200000000
) {
    attack_keys.clear();
    attack_keys.reserve(num_attack_keys);
    std::mt19937_64 gen(42);
    std::cout << "\n=== Generating RIGHT-BIASED, DUPLICATE-FREE Attack Keys ===\n";

    std::unordered_set<KEY_TYPE> unique_keys;

    double ood_fraction = 0.02;  // 少部分OOD ~2%
    size_t root_budget = static_cast<size_t>(num_attack_keys * ood_fraction);
    size_t generated = 0;
    KEY_TYPE base = original_max + 1;

    std::uniform_real_distribution<double> u(0.0, 1.0);
    double pareto_alpha = 1.5;
    KEY_TYPE pareto_min = 1LL << 10;
    for (size_t i = 0; i < root_budget * 2; ++i) {
        double pareto_val = pareto_min / std::pow(1.0 - u(gen), 1.0 / pareto_alpha);
        KEY_TYPE key = base + static_cast<KEY_TYPE>(pareto_val) + (gen() % (1LL << 10));
        if (unique_keys.insert(key).second) {
            generated++;
            if (generated >= root_budget) break;
        }
    }
    std::cout << "Root OOD keys: " << generated << "\n";

    std::vector<data_node_type*> dns;
    for (auto* n = index.first_data_node(); n; n = n->next_leaf_) {
        dns.push_back(n);
    }
    if (dns.empty()) return;
    std::sort(dns.begin(), dns.end(), [](auto* a, auto* b) {
        if (a->data_capacity_ == 0 || b->data_capacity_ == 0) return false;
        double density_a = static_cast<double>(a->num_keys_) / a->data_capacity_;
        double density_b = static_cast<double>(b->num_keys_) / b->data_capacity_;
        return density_a > density_b;
    });

    size_t hot_nodes = std::min<size_t>(500, dns.size());
    size_t remaining = num_attack_keys - unique_keys.size();
    size_t base_per_node = remaining / hot_nodes / 2;  // 减少quota，poison ~15%
    size_t max_nodes_focus = std::min<size_t>(5, hot_nodes);

    std::cout << "Num data nodes: " << dns.size() << "\n";

    size_t leftover_quota = 0;

    for (size_t ni = 0; ni < hot_nodes; ++ni) {
        size_t current_node_quota = base_per_node + leftover_quota;
        leftover_quota = 0;
        if (ni < max_nodes_focus) {
            current_node_quota *= 8;
        }
        auto* node = dns[ni];
        auto keys = node->get_all_keys();
        if (keys.size() < 2) continue;
        std::sort(keys.begin(), keys.end());
        KEY_TYPE node_min = keys.front();
        KEY_TYPE node_max = keys.back();
        KEY_TYPE range_val = node_max - node_min;
        if (range_val < 3) continue;

        size_t estimated_free = (static_cast<size_t>(node_max) - static_cast<size_t>(node_min) + 1) - keys.size();
        current_node_quota = std::min(current_node_quota, estimated_free);

        if (node->data_capacity_ == 0) continue;

        std::vector<std::pair<size_t, size_t>> dense_regions = get_multiple_dense_regions(node, 20);
        size_t inserted = 0;
        double poison_ratio = 0.7;

        std::normal_distribution<double> offset_dis(0.0, range_val / 8.0);

        for (const auto& region : dense_regions) {
            size_t max_cont_keys = region.first;
            size_t start_idx = region.second;

            if (max_cont_keys <= 0) continue;
            KEY_TYPE dense_min = 0;
            KEY_TYPE dense_max = 0;
            size_t found = 0;

            for (size_t i = start_idx; i < node->data_capacity_ && found < max_cont_keys; ++i) {
                if (node->check_exists(i)) {
                    KEY_TYPE key = node->get_key(i);
                    if (found == 0) {
                        dense_min = key;
                    }
                    dense_max = key;
                    found++;
                }
            }

            if (found == 0 || dense_max - dense_min < 1) continue;
            size_t num_poison = static_cast<size_t>(max_cont_keys * poison_ratio);

            std::vector<KEY_TYPE> candidate_poison;
            for (size_t i = 0; i < num_poison * 3; ++i) {
                KEY_TYPE mid = dense_min + (dense_max - dense_min) / 2;
                KEY_TYPE perturb = static_cast<KEY_TYPE>(offset_dis(gen));
                KEY_TYPE poison_k = mid + perturb;
                if (poison_k >= node_min && poison_k <= node_max &&
                    !std::binary_search(keys.begin(), keys.end(), poison_k)) {
                    candidate_poison.push_back(poison_k);
                }
            }

            long long mid = dense_min + (dense_max - dense_min) / 2;
            std::sort(candidate_poison.begin(), candidate_poison.end(), [&](KEY_TYPE a, KEY_TYPE b) {
                long long dev_a = std::abs(static_cast<long long>(a) - mid);
                long long dev_b = std::abs(static_cast<long long>(b) - mid);
                return dev_a < dev_b;
            });

            for (auto k : candidate_poison) {
                if (inserted >= current_node_quota) break;
                if (unique_keys.size() >= num_attack_keys) break;
                if (unique_keys.insert(k).second) {
                    inserted++;
                }
            }

            if (inserted < current_node_quota) {
                size_t cluster_size = std::min<size_t>(20000, current_node_quota - inserted);
                KEY_TYPE dense_span = dense_max - dense_min;
                if (dense_span <= cluster_size) continue;
                KEY_TYPE cluster_start = dense_min + (gen() % (dense_span - cluster_size));
                for (size_t j = 0; j < cluster_size; ++j) {
                    KEY_TYPE k = cluster_start + static_cast<KEY_TYPE>(std::pow(static_cast<double>(j), 1.2));
                    if (!std::binary_search(keys.begin(), keys.end(), k) && unique_keys.insert(k).second) {
                        inserted++;
                    }
                    if (inserted >= current_node_quota || unique_keys.size() >= num_attack_keys) break;
                }
            }

            if (inserted < current_node_quota) {
                for (size_t sub = 0; sub < 20; ++sub) {
                    KEY_TYPE dense_span = dense_max - dense_min;
                    if (dense_span <= 200) continue;
                    KEY_TYPE sub_start = dense_min + (gen() % (dense_span - 200));
                    for (size_t j = 0; j < 200; ++j) {
                        KEY_TYPE k = sub_start + j + (gen() % 10);
                        if (!std::binary_search(keys.begin(), keys.end(), k) && unique_keys.insert(k).second) {
                            inserted++;
                        }
                        if (inserted >= current_node_quota || unique_keys.size() >= num_attack_keys) break;
                    }
                    if (inserted >= current_node_quota) break;
                }
            }
        }

        size_t bands[] = {10000, 5000, 2000, 1000, 500, 200};
        size_t band_idx = 0;
        while (inserted < current_node_quota && band_idx < sizeof(bands)/sizeof(bands[0])) {
            if (bands[band_idx] == 0) { band_idx++; continue; }
            KEY_TYPE band_min = node_max - std::max<KEY_TYPE>(1, range_val / bands[band_idx]);
            KEY_TYPE k = band_min;
            while (k < node_max && inserted < current_node_quota) {
                if (std::binary_search(keys.begin(), keys.end(), k)) {
                    k++;
                    continue;
                }
                if (unique_keys.size() >= num_attack_keys) break;
                KEY_TYPE jitter_key = k + (gen() % 20 - 10);
                if (unique_keys.insert(jitter_key).second) {
                    inserted++;
                }
                k += 1 + (gen() % 5);
            }
            band_idx++;
        }

        size_t sample_attempts = 0;
        const size_t max_attempts = 100000000;
        std::uniform_int_distribution<KEY_TYPE> range_dis(node_min, node_max);
        while (inserted < current_node_quota && unique_keys.size() < num_attack_keys && sample_attempts < max_attempts) {
            KEY_TYPE k = range_dis(gen);
            if (!std::binary_search(keys.begin(), keys.end(), k) && unique_keys.insert(k).second) {
                inserted++;
            }
            sample_attempts++;
        }

        if (inserted < current_node_quota / 2) {
            leftover_quota += current_node_quota - inserted;
        }
    }

    // Filler: 剩余大部分，使用非线性扭曲生成（多簇 + 幂律 + 正弦 + 间隙）
    KEY_TYPE filler_base = original_min;
    size_t filler_count = unique_keys.size();
    size_t num_remaining = num_attack_keys - filler_count;

    size_t num_clusters = 20 + (gen() % 30);  // 20-50簇
    std::vector<KEY_TYPE> cluster_centers;
    std::uniform_int_distribution<KEY_TYPE> center_dis(original_min, original_max);
    for (size_t i = 0; i < num_clusters; ++i) {
        cluster_centers.push_back(center_dis(gen));
    }
    std::sort(cluster_centers.begin(), cluster_centers.end());

    size_t per_cluster = num_remaining / num_clusters;
    for (size_t c = 0; c < num_clusters; ++c) {
        KEY_TYPE center = cluster_centers[c];
        KEY_TYPE cluster_min = std::max(original_min, center - (1L << 20));
        KEY_TYPE cluster_max = std::min(original_max, center + (1L << 20));
        std::normal_distribution<double> cluster_dis(0.0, (cluster_max - cluster_min) / 10.0);

        for (size_t j = 0; j < per_cluster; ++j) {
            if (unique_keys.size() >= num_attack_keys) break;
            double distort = std::pow(static_cast<double>(j), 1.5) + 5000 * std::sin(2 * M_PI * j / per_cluster);
            KEY_TYPE k = center + static_cast<KEY_TYPE>(distort + cluster_dis(gen));
            if (k < original_min || k > original_max) continue;

            if (unique_keys.insert(k).second) {
                // 成功插入，无duplicate
            } else {
                k += (gen() % 100) + 1;  // 轻微调整避免duplicate
                if (k <= original_max) unique_keys.insert(k);
            }
        }

        if (c < num_clusters - 1) {
            KEY_TYPE gap_size = (gen() % (1LL << 25)) + (1LL << 20);  // 大随机间隙，创建CDF平台
            filler_base = cluster_centers[c + 1] + gap_size;
        }
    }

    // 如果仍不足，稀疏填充 + 扭曲
    while (unique_keys.size() < num_attack_keys) {
        KEY_TYPE k = filler_base + static_cast<KEY_TYPE>(std::pow(static_cast<double>(filler_count % 10000), 1.3));
        if (unique_keys.insert(k).second) {
            // 成功
        } else {
            k += (gen() % 100) + 1;
            unique_keys.insert(k);
        }
        filler_base = k + 1 + (gen() % 10000);
        filler_count++;
    }

    attack_keys.assign(unique_keys.begin(), unique_keys.end());

    std::cout << "\n=== Attack Summary ===\n";
    std::cout << "Total keys : " << attack_keys.size() << "\n";
    std::cout << "Right OOD : "
              << std::count_if(attack_keys.begin(), attack_keys.end(),
                               [&](KEY_TYPE k) { return k > original_max; })
              << "\n";

    std::sort(attack_keys.begin(), attack_keys.end());
    std::cout << "CDF Quantiles (for nonlinearity check):\n";
    for (double q : {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}) {
        size_t idx = static_cast<size_t>(q * (attack_keys.size() - 1));
        std::cout << q << ": " << attack_keys[idx] << "\n";
    }

    // Kurtosis计算（简单版，评估扭曲度）
    double mean = 0.0;
    for (auto k : attack_keys) mean += static_cast<double>(k);
    mean /= attack_keys.size();
    double var = 0.0, kurt = 0.0;
    for (auto k : attack_keys) {
        double dev = static_cast<double>(k) - mean;
        var += dev * dev;
        kurt += dev * dev * dev * dev;
    }
    var /= attack_keys.size();
    kurt = (kurt / attack_keys.size()) / (var * var);
    std::cout << "Kurtosis (higher = more distorted/peaked): " << kurt << "\n";
    std::cout << "=============================================\n";
}

private:
    using AlexType =  alexol::Alex <KEY_TYPE, PAYLOAD_TYPE, alexol::AlexCompare, std::allocator<
            std::pair < KEY_TYPE, PAYLOAD_TYPE>>, false>;
    AlexType index;
    typedef typename AlexType::model_node_type model_node_type;
    typedef typename AlexType::data_node_type data_node_type;

    std::pair<int, int> get_max_continuous_key_region(data_node_type* node) {
        if (!node) return {-1, -1};

        int cur_max = -1;
        int start_index = -1;
        int keys = 0;

        for (int i = 0; i < node->data_capacity_; i++) {
            if (node->check_exists(i)) {
                keys++;
            } else {
                if (keys > cur_max) {
                    start_index = i - keys;
                    cur_max = keys;
                }
                keys = 0;
            }
        }

        // Handle corner case: largest segment spans to end
        if (keys != 0) {
            start_index = node->data_capacity_ - keys;
            cur_max = keys;
        }

        return {cur_max, start_index};
    }

    std::vector<std::pair<size_t, size_t>> get_multiple_dense_regions(data_node_type* node, size_t top_n) {
        std::vector<std::pair<size_t, size_t>> regions;
        if (!node || top_n == 0) return regions;

        size_t current_length = 0;
        size_t current_start = static_cast<size_t>(-1);  // Invalid start

        for (size_t i = 0; i < node->data_capacity_; ++i) {
            if (node->check_exists(i)) {
                if (current_length == 0) {
                    current_start = i;  // Start of a new region
                }
                current_length++;
            } else {
                if (current_length > 0) {
                    regions.emplace_back(current_length, current_start);
                    current_length = 0;
                    current_start = static_cast<size_t>(-1);
                }
            }
        }

        // Handle the case where the last region extends to the end
        if (current_length > 0) {
            regions.emplace_back(current_length, current_start);
        }

        // Sort regions by length descending (largest first)
        std::sort(regions.begin(), regions.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        // Return only the top_n regions, or all if fewer
        if (regions.size() > top_n) {
            regions.resize(top_n);
        }

        return regions;
    }

};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                        Param *param) {
    index.set_max_model_node_size(
            1 << 25); // (22,15,66M) (22,16,89M) (22,17,91M) (22,18,81M) (23,17,92M) (21,17,92M) (20,17,92M)
    index.set_max_data_node_size(1 << 19);
    std::cout << "start alex bulkload" << std::endl;
    index.bulk_load(key_value, (int) num);
    std::cout << "end alex bulkload" << std::endl;
    auto memory = memory_consumption();
    std::cout << "Size after bulkload: " << memory << std::endl;
#if EXTREME
    data_node_type* data_node = index.first_data_node();
    if (!data_node) {
        std::cout << "Warning: first datanode is null" << std::endl;
        return;
    }

    TSCNS tn;
    tn.init();

    int converted_count = 0;
    int skipped_count = 0;
    long long converted_size = 0;
    double total_convert_time = 0.0;
    size_t total_keys = 0;

    // 遍历所有叶子节点
    while (data_node != nullptr) {
        // data_node->save_stats_to_csv(file_name);
        auto convert_start = tn.rdtsc();
        bool converted = false;
        #if ENABLE_CONTROLLER
        converted = data_node->convert_to_hash_node();
        #else
        data_node->resize(0.97);
        converted = true;
        #endif
        if (converted) converted_count++;
        auto convert_end = tn.rdtsc();

        auto convert_time = tn.tsc2ns(convert_end) - tn.tsc2ns(convert_start);
        total_convert_time += convert_time;
        total_keys += data_node->num_keys_;
        converted_size += data_node->data_size();

        data_node = data_node->next_leaf_;
    }

    std::cout << "\n=== Zone Optimization Summary ===" << std::endl;
    std::cout << "Converted nodes: " << converted_count << std::endl;
    std::cout << "Skipped nodes: " << skipped_count << std::endl;
    std::cout << "Total keys in converted nodes: " << total_keys << std::endl;
    std::cout << "Total converted size: " << converted_size << " bytes" << std::endl;
    std::cout << "Total conversion time: " << total_convert_time / 1e6 << " ms" << std::endl;
    if (converted_count > 0) {
        std::cout << "Average conversion time per node: " << total_convert_time / converted_count << " ns" << std::endl;
    }
    std::cout << "==================================\n" << std::endl;
#endif
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
    auto ret = index.get_payload(key, &val);
    return ret;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return index.insert(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return index.update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
    int num = index.erase(key);
    if(num) return true; else return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                     std::pair <KEY_TYPE, PAYLOAD_TYPE> *result,
                                                     Param *param) {
    auto scan_size = index.range_scan_by_size(key_low_bound, static_cast<uint32_t>(key_num), result);
    return scan_size;
}
#endif
