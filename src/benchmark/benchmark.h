#pragma once
#ifndef ENABLE_CONTROLLER
#define ENABLE_CONTROLLER 1
#endif

#ifndef ENABLE_MANUAL_OPTIMIZATION
#define ENABLE_MANUAL_OPTIMIZATION 0
#endif
#define EXTREME 0
#define EXTRA 0
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <ctime>
#include "../tscns.h"
#include "omp.h"
#include "tbb/parallel_sort.h"
#include "flags.h"
#include "utils.h"
#include "../competitor/competitor.h"
#include "../competitor/indexInterface.h"

template<typename KEY_TYPE, typename PAYLOAD_TYPE>
class Benchmark {
    typedef indexInterface <KEY_TYPE, PAYLOAD_TYPE> index_t;
    enum Operation {
        READ = 0, INSERT, DELETE, SCAN, UPDATE
    };

    struct OpUnit {
        Operation type;
        KEY_TYPE key;
        bool should_sample;
    };

    struct Zone {
        KEY_TYPE min_key;
        KEY_TYPE max_key;

        std::vector<KEY_TYPE> existing_keys;
        std::vector<KEY_TYPE> original_to_insert;
        std::vector<KEY_TYPE> attack_to_insert;

        std::vector<OpUnit> ops;
    };

    struct SubZone {
        size_t begin_index;
        size_t end_index;
        KEY_TYPE min_key;
        KEY_TYPE max_key;
    };

    std::vector<SubZone> attack_sub_zones;

    // parameters
    double read_ratio = 1;
    double insert_ratio = 0;
    double delete_ratio = 0;
    double update_ratio = 0;
    double scan_ratio = 0;
    double skew_alpha = 1.6;
    size_t scan_num = 100;
    size_t operations_num;
    long long table_size = -1;
    size_t init_table_size;
    double init_table_ratio;
    double del_table_ratio;
    size_t thread_num = 1;
    std::vector <std::string> all_index_type;
    std::vector <std::string> all_thread_num;
    std::string index_type;
    std::string keys_file_path;
    std::string attack_keys_file_path;
    std::string keys_file_type;
    std::string sample_distribution;
    bool latency_sample = false;
    double latency_sample_ratio = 0.1;
    int error_bound;
    std::string output_path;
    std::string latency_output_path;
    size_t random_seed;
    bool memory_record;
    bool dataset_statistic;
    bool data_shift = false;
    bool is_read_only = false;
    int num_repeats = 1;
    bool adjust_density = false;
    bool ban_compress = false;
    bool save_generated_dataset = false;
    bool allow_attack_generation = false;
    bool controller_runtime_enabled = true;

    double throughput_stddev = 0.0;
    double throughput_ci95 = 0.0;
    double memory_stddev = 0.0;

    int num_optimizations = 10;
    double cold_zone_ratio = 0.3;
    double interval_variance = 0.1;
    std::vector<size_t> optimization_trigger_points;
    std::vector<std::pair<KEY_TYPE, KEY_TYPE>> optimization_ranges;

    // Data structures
    std::vector<Zone> zones;
    std::vector <KEY_TYPE> init_keys;
    std::vector <KEY_TYPE> all_sorted_keys;
    KEY_TYPE *keys;
    std::pair <KEY_TYPE, PAYLOAD_TYPE> *init_key_values;
    std::vector<KEY_TYPE> stored_original_keys;
    KEY_TYPE stored_global_min;
    KEY_TYPE stored_global_max;

    std::mt19937 gen;

    struct Stat {
        std::vector<double> latency;
        uint64_t throughput = 0;
        size_t fitness_of_dataset = 0;
        long long memory_before = 0;
        long long memory_consumption = 0;
        std::uint64_t elapsed_ns = 0;
        ControllerStatistics controller;
        uint64_t success_insert = 0;
        uint64_t success_read = 0;
        uint64_t success_update = 0;
        uint64_t success_remove = 0;
        uint64_t scan_not_enough = 0;
        void clear() {
            latency.clear();
            throughput = 0;
            fitness_of_dataset = 0;
            memory_before = 0;
            memory_consumption = 0;
            elapsed_ns = 0;
            controller = {};
            success_insert = 0;
            success_read = 0;
            success_update = 0;
            success_remove = 0;
            scan_not_enough = 0;
        }
        void add(const Stat& other) {
            throughput += other.throughput;
            fitness_of_dataset += other.fitness_of_dataset;
            memory_before += other.memory_before;
            memory_consumption += other.memory_consumption;
            elapsed_ns += other.elapsed_ns;
            controller.add(other.controller);
            success_insert += other.success_insert;
            success_read += other.success_read;
            success_update += other.success_update;
            success_remove += other.success_remove;
            scan_not_enough += other.scan_not_enough;
        }
        void divide(int n) {
            throughput /= n;
            fitness_of_dataset /= n;
            memory_before /= n;
            memory_consumption /= n;
            elapsed_ns /= n;
            controller.divide(static_cast<std::uint64_t>(n));
            success_insert /= n;
            success_read /= n;
            success_update /= n;
            success_remove /= n;
            scan_not_enough /= n;
        }
    } stat;

    struct alignas(CACHELINE_SIZE)
    ThreadParam {
        std::vector<std::pair<uint64_t, uint64_t>> latency;
        uint64_t success_insert = 0;
        uint64_t success_read = 0;
        uint64_t success_update = 0;
        uint64_t success_remove = 0;
        uint64_t scan_not_enough = 0;
    };
    typedef ThreadParam param_t;

public:
    Benchmark() {
    }

    void save_final_keys(const std::vector<KEY_TYPE>& sorted_keys, const std::string& filename) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file for saving keys: " << filename << std::endl;
            return;
        }
        outfile << "key\n";
        for (const auto& k : sorted_keys) {
            outfile << k << "\n";
        }
        outfile.close();
        std::cout << "Successfully saved " << sorted_keys.size() << " keys to " << filename << std::endl;
    }

    KEY_TYPE *load_keys() {
        COUT_THIS("Reading dataset from file: " << keys_file_path);

        if (table_size > 0) keys = new KEY_TYPE[table_size];

        if (keys_file_type == "binary") {
            table_size = load_binary_data(keys, table_size, keys_file_path);
        } else if (keys_file_type == "text") {
            table_size = load_text_data(keys, table_size, keys_file_path);
        } else {
            COUT_THIS("Error: Invalid keys_file_type.");
            exit(1);
        }

        tbb::parallel_sort(keys, keys + table_size);
        auto last = std::unique(keys, keys + table_size);
        table_size = last - keys;

        std::vector<KEY_TYPE> original_keys(keys, keys + table_size);
        KEY_TYPE global_min = original_keys.front();
        KEY_TYPE global_max = original_keys.back();

        COUT_THIS("Original dataset: " << table_size << " keys, range=["
                << global_min << ", " << global_max << "]");

        std::string init_path = keys_file_path + "_init_keys.bin";
        const size_t expected_init_size =
            static_cast<size_t>(init_table_ratio * table_size);
        if (file_exists(init_path) && !is_read_only) {
            KEY_TYPE cached_size = 0;
            std::ifstream init_header(init_path, std::ios::binary);
            init_header.read(reinterpret_cast<char*>(&cached_size),
                             sizeof(cached_size));
            if (init_header &&
                static_cast<size_t>(cached_size) == expected_init_size) {
                init_keys = load_binary_keys<KEY_TYPE>(init_path);
            } else {
                COUT_THIS("Ignoring cached init keys with " << cached_size
                          << " entries; expected " << expected_init_size);
                init_keys.clear();
            }
        }
        if (init_keys.empty()) {
            init_table_size = expected_init_size;
            init_keys.clear();
            init_keys.resize(init_table_size);

            std::shuffle(keys, keys + table_size, gen);

            for (size_t i = 0; i < init_table_size; ++i) {
                init_keys[i] = keys[i];
            }

            tbb::parallel_sort(init_keys.begin(), init_keys.end());
            // save_keys_to_binary(init_keys, init_path);
        } else {
            init_table_size = init_keys.size();
        }

        COUT_THIS("Init keys prepared: " << init_table_size
                << " (" << (init_table_ratio * 100) << "%)");

        init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_table_size];
        #pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < init_table_size; i++) {
            init_key_values[i].first = init_keys[i];
            init_key_values[i].second = 123456789;
        }

        stored_original_keys = original_keys;
        stored_global_min = global_min;
        stored_global_max = global_max;

        delete[] keys;
        table_size = original_keys.size();
        keys = new KEY_TYPE[table_size];
        std::copy(original_keys.begin(), original_keys.end(), keys);

        COUT_THIS("Ready for bulkload.");

        return keys;
    }

    void generate_attack_keys_post_bulkload(index_t *index)
    {
        COUT_THIS("Generating progressive attack workload...");

        std::vector<KEY_TYPE> attack_keys;
        const bool explicit_attack_path = !attack_keys_file_path.empty();
        std::string attack_path = explicit_attack_path
            ? attack_keys_file_path
            : keys_file_path + "+" + index_type + "_attack_keys.bin";

        if (file_exists(attack_path)) {
            attack_keys = load_binary_keys<KEY_TYPE>(attack_path);
            if (attack_keys.empty()) {
                throw std::runtime_error("Attack-key file is empty: " + attack_path);
            }
            auto max_it = std::max_element(attack_keys.begin(), attack_keys.end());
            stored_global_max= *max_it;
            COUT_THIS("✓ Loaded existing attack keys: " << attack_keys.size() << " keys");
        } else {
            if (explicit_attack_path && !allow_attack_generation) {
                throw std::runtime_error("Attack-key file does not exist: " + attack_path);
            }
            COUT_THIS("Generating new attack keys using index...");
            index->generate_attack_keys(
                attack_keys,
                stored_global_min,
                stored_global_max,
                table_size,
                operations_num
            );
            save_keys_to_binary(attack_keys, attack_path);
            COUT_THIS("✓ Generated and saved attack keys: " << attack_keys.size() << " keys");
        }

        tbb::parallel_sort(attack_keys.begin(), attack_keys.end());

        setup_simplified_zones(stored_original_keys, attack_keys);
    }

    void setup_simplified_zones(
        const std::vector<KEY_TYPE>& original_keys,
        const std::vector<KEY_TYPE>& attack_keys)
    {
        COUT_THIS("=== Setting up simplified zones (Global Unique Keys) ===");

        zones.clear();
        zones.resize(2);

        double warmup_ops_ratio = 0.1;
        size_t warmup_ops = static_cast<size_t>(operations_num * warmup_ops_ratio);
        size_t zone1_ops = operations_num - warmup_ops;
        const size_t zone1_insert_margin =
            insert_ratio > 0.0 ? std::max<size_t>(1024, zone1_ops / 100) : 0;
        size_t zone1_total_inserts_needed =
            static_cast<size_t>(zone1_ops * insert_ratio) +
            zone1_insert_margin;

        KEY_TYPE global_min = std::min({original_keys.front(), attack_keys.front()});
        KEY_TYPE global_max = std::max({original_keys.back(), attack_keys.back()});
        const double bounded_cold_ratio =
            std::clamp(cold_zone_ratio, 0.0, 1.0);
        const size_t cold_boundary_index = std::min(
            original_keys.size() - 1,
            static_cast<size_t>(bounded_cold_ratio *
                                static_cast<double>(original_keys.size() - 1)));
        const KEY_TYPE cold_boundary = original_keys[cold_boundary_index];

        std::vector<KEY_TYPE> pool_original_not_in_init;
        std::vector<KEY_TYPE> pool_existing_total;
        pool_original_not_in_init.reserve(
            original_keys.size() - init_keys.size());
        pool_existing_total.reserve(init_keys.size());
        size_t init_index = 0;
        for (const auto& k : original_keys) {
            while (init_index < init_keys.size() &&
                   init_keys[init_index] < k) {
                ++init_index;
            }
            if (init_index < init_keys.size() && init_keys[init_index] == k) {
                pool_existing_total.push_back(k);
                ++init_index;
            } else {
                pool_original_not_in_init.push_back(k);
            }
        }
        // 打乱候选池以保证随机分配
        std::shuffle(pool_original_not_in_init.begin(), pool_original_not_in_init.end(), gen);

        zones[0].min_key = global_min;
        zones[0].max_key = global_max;

        zones[0].existing_keys = pool_existing_total;

        const size_t zone0_insert_margin =
            insert_ratio > 0.0 ? std::max<size_t>(128, warmup_ops / 100) : 0;
        const size_t zone0_inserts_needed =
            static_cast<size_t>(warmup_ops * insert_ratio) +
            zone0_insert_margin;
        size_t z0_take =
            std::min(zone0_inserts_needed, pool_original_not_in_init.size());
        zones[0].original_to_insert.assign(pool_original_not_in_init.begin(),
                                        pool_original_not_in_init.begin() + z0_take);

        // 5. === Zone 1: Attack 分配 ===
        zones[1].min_key = cold_boundary;
        zones[1].max_key = global_max;

        for (const auto& k : pool_existing_total) {
            if (k > cold_boundary) {
                zones[1].existing_keys.push_back(k);
            }
        }

        std::vector<KEY_TYPE> attack_candidates;
        attack_candidates.reserve(
            std::min(zone1_total_inserts_needed, attack_keys.size()));
        std::vector<KEY_TYPE> zone0_insert_keys = zones[0].original_to_insert;
        tbb::parallel_sort(zone0_insert_keys.begin(), zone0_insert_keys.end());
        init_index = 0;
        size_t zone0_insert_index = 0;
        size_t eligible_attack_count = 0;
        for (const auto& k : attack_keys) {
            while (init_index < init_keys.size() &&
                   init_keys[init_index] < k) {
                ++init_index;
            }
            const bool in_initial_keys =
                init_index < init_keys.size() && init_keys[init_index] == k;
            while (zone0_insert_index < zone0_insert_keys.size() &&
                   zone0_insert_keys[zone0_insert_index] < k) {
                ++zone0_insert_index;
            }
            const bool inserted_in_warmup =
                zone0_insert_index < zone0_insert_keys.size() &&
                zone0_insert_keys[zone0_insert_index] == k;
            if (in_initial_keys || inserted_in_warmup) {
                continue;
            }
            ++eligible_attack_count;
            if (attack_candidates.size() < zone1_total_inserts_needed) {
                attack_candidates.push_back(k);
            } else if (zone1_total_inserts_needed != 0) {
                std::uniform_int_distribution<size_t> replacement(
                    0, eligible_attack_count - 1);
                const size_t position = replacement(gen);
                if (position < zone1_total_inserts_needed) {
                    attack_candidates[position] = k;
                }
            }
        }
        std::shuffle(attack_candidates.begin(), attack_candidates.end(), gen);

        size_t attack_to_take = std::min(zone1_total_inserts_needed, attack_candidates.size());
        zones[1].attack_to_insert.assign(attack_candidates.begin(),
                                        attack_candidates.begin() + attack_to_take);

        size_t remaining_gap = zone1_total_inserts_needed - zones[1].attack_to_insert.size();
        if (remaining_gap > 0) {
            std::vector<KEY_TYPE> original_candidates_zone1;
            original_candidates_zone1.reserve(
                pool_original_not_in_init.size() - z0_take);
            for (size_t i = z0_take; i < pool_original_not_in_init.size(); ++i) {
                const auto k = pool_original_not_in_init[i];
                if (k > cold_boundary) {
                    original_candidates_zone1.push_back(k);
                }
            }
            size_t original_to_take = std::min(remaining_gap, original_candidates_zone1.size());
            zones[1].original_to_insert.assign(original_candidates_zone1.begin(),
                                            original_candidates_zone1.begin() + original_to_take);
        }

        bool zone1_has_keys = false;
        auto include_bounds = [&zone = zones[1], &zone1_has_keys](const auto& values) {
            if (values.empty()) return;
            const auto bounds = std::minmax_element(values.begin(), values.end());
            if (!zone1_has_keys) {
                zone.min_key = *bounds.first;
                zone.max_key = *bounds.second;
                zone1_has_keys = true;
            } else {
                zone.min_key = std::min(zone.min_key, *bounds.first);
                zone.max_key = std::max(zone.max_key, *bounds.second);
            }
        };
        include_bounds(zones[1].existing_keys);
        include_bounds(zones[1].attack_to_insert);
        include_bounds(zones[1].original_to_insert);

        COUT_THIS("Zone 1" << " summary:"
                << "\n  range=[" << zones[1].min_key << ", " << zones[1].max_key << "]"
                << "\n  existing=" << zones[1].existing_keys.size()
                << "\n  original_to_insert=" << zones[1].original_to_insert.size()
                << "\n  attack_to_insert=" << zones[1].attack_to_insert.size()
                << "\n  total_keys="
                << zones[1].existing_keys.size() +
                       zones[1].attack_to_insert.size() +
                       zones[1].original_to_insert.size());
    }

    void calculate_optimization_triggers() {
        COUT_THIS("Calculating optimization triggers ...");

        optimization_trigger_points.clear();
        optimization_ranges.clear();
        attack_sub_zones.clear();

        if (num_optimizations <= 0 || zones.size() <= 1) {
            COUT_THIS("No optimizations needed.");
            return;
        }

        auto& attack_zone = zones[1];
        size_t total_ops = attack_zone.ops.size();

        KEY_TYPE attack_min = attack_zone.min_key;
        KEY_TYPE attack_max = attack_zone.max_key;
        double attack_range_dbl = static_cast<double>(attack_max) - static_cast<double>(attack_min);

        // === 1. 生成触发点 ===
        double coverage_ratio = 0.6;

        for (int i = 0; i < num_optimizations; ++i) {
            double ratio = (static_cast<double>(i + 1) / (num_optimizations + 1)) * coverage_ratio;

            std::uniform_real_distribution<> jitter_dist(-0.1, 0.1);
            double jitter = jitter_dist(gen) * coverage_ratio / (num_optimizations + 1);
            ratio += jitter;

            size_t trigger_pos = static_cast<size_t>(total_ops * ratio);

            if (!optimization_trigger_points.empty()) {
                size_t min_pos = optimization_trigger_points.back() + 100;
                trigger_pos = std::max(trigger_pos, min_pos);
            }

            if (trigger_pos >= total_ops) {
                trigger_pos = total_ops - 1;
            }

            optimization_trigger_points.push_back(trigger_pos);
        }

        // === 2. 生成优化范围 ===
        std::vector<double> ratios;
        double ratio_sum = 0.0;

        for (int i = 0; i < num_optimizations; ++i) {
            double weight = static_cast<double>(i + 1);
            ratios.push_back(weight);
            ratio_sum += weight;
        }

        for (auto& r : ratios) {
            r /= ratio_sum;
        }

        double cumulative_range = 0.0;
        for (int i = 0; i < num_optimizations; ++i) {
            double start_ratio = cumulative_range;
            cumulative_range += ratios[i];
            double end_ratio = cumulative_range;

            KEY_TYPE opt_min = attack_min + static_cast<KEY_TYPE>(attack_range_dbl * start_ratio);
            KEY_TYPE opt_max = attack_min + static_cast<KEY_TYPE>(attack_range_dbl * end_ratio);

            if (i == num_optimizations - 1) {
                opt_max = attack_max;
            }

            optimization_ranges.push_back({opt_min, opt_max});
        }

        std::reverse(optimization_ranges.begin(), optimization_ranges.end());

        // === 3. 拆分 Zone 1 的 ops 为子 zones（只复制 ops，不复制键集合）===
        std::vector<size_t> split_points = {0};
        split_points.insert(split_points.end(), optimization_trigger_points.begin(), optimization_trigger_points.end());
        split_points.push_back(total_ops);

        for (size_t i = 0; i < split_points.size() - 1; ++i) {
            SubZone sub_zone;
            size_t start_idx = split_points[i];
            size_t end_idx = split_points[i + 1];
            sub_zone.begin_index = start_idx;
            sub_zone.end_index = end_idx;

            // 复制元数据
            sub_zone.min_key = attack_min;
            sub_zone.max_key = attack_max;

            attack_sub_zones.push_back(std::move(sub_zone));

            COUT_THIS("Sub-zone " << i << ": ops [" << start_idx << ", " << end_idx
                    << "), total " << (end_idx - start_idx) << " operations");
        }

        // === 打印日志 ===
        COUT_THIS("Optimization triggers (total_ops=" << total_ops << "):");
        for (int i = 0; i < num_optimizations; ++i) {
            double range_size = static_cast<double>(optimization_ranges[i].second) -
                            static_cast<double>(optimization_ranges[i].first);
            COUT_THIS("  Trigger " << i << ": at op # " << optimization_trigger_points[i]
                    << " (" << std::fixed << std::setprecision(1)
                    << (100.0 * optimization_trigger_points[i] / total_ops) << "%)"
                    << ", optimize [" << optimization_ranges[i].first
                    << ", " << optimization_ranges[i].second << "]"
                    << " (range size: " << range_size << ")");
        }

        COUT_THIS("Created " << attack_sub_zones.size() << " sub-zones for isolated execution");
    }

    void save_final_dataset_after_generation() {
        COUT_THIS("\n=== Saving final dataset after operation generation ===");

        std::vector<KEY_TYPE> final_keys;
        final_keys.insert(final_keys.end(), init_keys.begin(), init_keys.end());

        for (auto& zone : zones) {
            for (auto& op : zone.ops) {
                if (op.type == INSERT) {
                    final_keys.push_back(op.key);
                }
            }
        }

        tbb::parallel_sort(final_keys.begin(), final_keys.end());
        auto last = std::unique(final_keys.begin(), final_keys.end());
        final_keys.resize(last - final_keys.begin());

        COUT_THIS("Total keys in final dataset: " << final_keys.size());

        std::string output_path = keys_file_path + "_progressive_final.bin";
        save_keys_to_binary(final_keys, output_path);

        COUT_THIS("Saved to: " << output_path);

        std::string zone_info_path = keys_file_path + "_" + index_type + "_zone_info.txt";
        std::ofstream zone_file(zone_info_path);

        zone_file << "num_zones=" << zones.size() << "\n";
        zone_file << "num_optimizations=" << num_optimizations << "\n";
        zone_file << "cold_zone_ratio=" << cold_zone_ratio << "\n";
        for (size_t s = 0; s < zones.size(); ++s) {
            auto& zone = zones[s];
            zone_file << "zone_" << s
                    << " min=" << zone.min_key
                    << " max=" << zone.max_key << "\n";
        }
        zone_file.close();

        COUT_THIS("Zone info saved to: " << zone_info_path);
        COUT_THIS("===========================================\n");
    }

    std::vector<KEY_TYPE> load_progressive_dataset(const std::string& filepath) {
        COUT_THIS("Loading progressive dataset from: " << filepath);

        std::ifstream infile(filepath, std::ios::binary);
        if (!infile) {
            COUT_THIS("Error: Cannot open file: " << filepath);
            exit(1);
        }

        size_t count;
        infile.read(reinterpret_cast<char*>(&count), sizeof(count));

        std::vector<KEY_TYPE> keys(count);
        infile.read(reinterpret_cast<char*>(keys.data()), count * sizeof(KEY_TYPE));
        infile.close();

        COUT_THIS("Loaded " << count << " keys");
        return keys;
    }

    void load_zone_info(const std::string& filepath) {
        COUT_THIS("Loading zone info from: " << filepath);

        std::ifstream zone_file(filepath);
        if (!zone_file) {
            COUT_THIS("Warning: Zone info file not found");
            return;
        }

        std::string line;
        while (std::getline(zone_file, line)) {
            if (line.find("num_optimizations=") != std::string::npos) {
                sscanf(line.c_str(), "num_optimizations=%d", &num_optimizations);
            } else if (line.find("cold_zone_ratio=") != std::string::npos) {
                sscanf(line.c_str(), "cold_zone_ratio=%lf", &cold_zone_ratio);
            }
        }
        zone_file.close();

        COUT_THIS("Loaded num_optimizations=" << num_optimizations
                << ", cold_zone_ratio=" << cold_zone_ratio);
    }

    void generate_operations(KEY_TYPE *keys) {
        COUT_THIS("Generating operations for all zones...");

        std::uniform_real_distribution<> ratio_dis(0, 1);

        double warmup_ops_ratio = 0.1;
        size_t warmup_ops = static_cast<size_t>(operations_num * warmup_ops_ratio);
        size_t attack_zone_ops = operations_num - warmup_ops;

        std::vector<size_t> zone_ops(2);
        zone_ops[0] = warmup_ops;
        zone_ops[1] = attack_zone_ops;

        if (insert_ratio > 0.9) {
            size_t available_keys = zones[1].original_to_insert.size() +
                                zones[1].attack_to_insert.size();
            if (zone_ops[1] > available_keys) {
                COUT_THIS("Warning: Zone 1 ops=" << zone_ops[1]
                        << " exceeds available insert keys=" << available_keys
                        << ", adjusting...");
                zone_ops[1] = available_keys;
            }
        }

        int latency_sample_interval = std::max(1,
            static_cast<int>(operations_num / (operations_num * latency_sample_ratio)));
        if (latency_sample_interval > 1 &&
            (latency_sample_interval & 1) == 0) {
            ++latency_sample_interval;
        }
        size_t global_op_idx = 0;

        for (int s = 0; s < 2; ++s) {
            size_t this_ops = zone_ops[s];
            auto& zone = zones[s];
            const size_t zone_key_count = zone.existing_keys.size() +
                                          zone.original_to_insert.size() +
                                          zone.attack_to_insert.size();

            COUT_THIS("Zone " << s << ": generating " << this_ops << " operations"
                    << " (keys=" << zone_key_count << ")");

            const std::vector<KEY_TYPE>* read_pool = &zone.existing_keys;

            std::vector<KEY_TYPE> insert_pool;
            insert_pool.insert(insert_pool.end(),
                zone.original_to_insert.begin(), zone.original_to_insert.end());
            insert_pool.insert(insert_pool.end(),
                zone.attack_to_insert.begin(), zone.attack_to_insert.end());
            std::shuffle(insert_pool.begin(), insert_pool.end(), gen);

            if (read_pool->empty() && insert_pool.empty()) {
                COUT_THIS("Warning: Zone " << s << " has no keys! Skipping...");
                continue;
            }

            if (read_pool->empty() && !zones[0].existing_keys.empty()) {
                COUT_THIS("Zone " << s
                          << " has no resident keys; reads use the bulk-loaded pool");
                read_pool = &zones[0].existing_keys;
            }

            KEY_TYPE *read_sample_keys = nullptr;
            const bool needs_read_samples =
                read_ratio > 0.0 || update_ratio > 0.0 || scan_ratio > 0.0;
            if (needs_read_samples && !read_pool->empty()) {
                read_sample_keys = (sample_distribution == "uniform") ?
                    get_search_keys(read_pool->data(), read_pool->size(), this_ops, &random_seed) :
                    get_search_keys_zipf(read_pool->data(), read_pool->size(), this_ops, skew_alpha, &random_seed);
            }

            zone.ops.clear();
            zone.ops.reserve(this_ops);

            size_t insert_idx = 0;
            size_t read_sample_idx = 0;
            const bool track_dynamic_keys =
                delete_ratio > 0.0 ||
                (needs_read_samples && read_sample_keys == nullptr);
            std::vector<KEY_TYPE> all_inserted_keys;
            if (track_dynamic_keys) {
                all_inserted_keys = *read_pool;
                all_inserted_keys.reserve(read_pool->size() + insert_pool.size());
            }

            for (size_t i = 0; i < this_ops; ++i) {
                auto prob = ratio_dis(gen);
                Operation op_type;
                KEY_TYPE op_key;

                if (prob < read_ratio) {
                    op_type = READ;
                    if (read_sample_keys) {
                        op_key = read_sample_keys[read_sample_idx++ % this_ops];
                    } else if (!all_inserted_keys.empty()) {
                        op_key = all_inserted_keys[read_sample_idx++ % all_inserted_keys.size()];
                    } else {
                        continue;
                    }

                } else if (prob < read_ratio + insert_ratio) {
                    op_type = INSERT;
                    if (insert_idx < insert_pool.size()) {
                        op_key = insert_pool[insert_idx++];
                        if (track_dynamic_keys) all_inserted_keys.push_back(op_key);
                    } else {
                        if (insert_ratio > 0.9) {
                            op_key = insert_pool[(insert_idx++) % insert_pool.size()];
                        } else {
                            op_type = READ;
                            if (read_sample_keys) {
                                op_key = read_sample_keys[read_sample_idx++ % this_ops];
                            } else if (!all_inserted_keys.empty()) {
                                op_key = all_inserted_keys[read_sample_idx++ % all_inserted_keys.size()];
                            } else {
                                continue;
                            }
                        }
                    }

                } else if (prob < read_ratio + insert_ratio + update_ratio) {
                    op_type = UPDATE;
                    if (read_sample_keys) {
                        op_key = read_sample_keys[read_sample_idx++ % this_ops];
                    } else if (!all_inserted_keys.empty()) {
                        op_key = all_inserted_keys[read_sample_idx++ % all_inserted_keys.size()];
                    } else {
                        continue;
                    }

                } else if (prob < read_ratio + insert_ratio + update_ratio + scan_ratio) {
                    op_type = SCAN;
                    if (read_sample_keys) {
                        op_key = read_sample_keys[read_sample_idx++ % this_ops];
                    } else if (!all_inserted_keys.empty()) {
                        op_key = all_inserted_keys[read_sample_idx++ % all_inserted_keys.size()];
                    } else {
                        continue;
                    }

                } else {
                    op_type = DELETE;
                    if (!all_inserted_keys.empty()) {
                        op_key = all_inserted_keys[read_sample_idx++ % all_inserted_keys.size()];
                    } else {
                        continue;
                    }
                }

                bool should_sample = (latency_sample && global_op_idx % latency_sample_interval == 0);
                zone.ops.push_back({op_type, op_key, should_sample});
                global_op_idx++;
            }

            if (read_sample_keys) delete[] read_sample_keys;

            COUT_THIS("  Zone " << s << ": " << zone.ops.size() << " ops generated");
        }

        COUT_THIS("All operations generated. Total: " << global_op_idx);

        calculate_optimization_triggers();
        if (save_generated_dataset && !is_read_only) {
            save_final_dataset_after_generation();
        }
    }

    inline void prepare(index_t *&index, const KEY_TYPE *keys) {
        index = get_index<KEY_TYPE, PAYLOAD_TYPE>(index_type);
        Param param = Param(thread_num, 0);
        index->init(&param);
#if ENABLE_CONTROLLER
        index->stop_controller();
#endif
        thread_num = param.worker_num;
        COUT_THIS("bulk loading");
        index->bulk_load(init_key_values, init_keys.size(), &param);
    }

    inline void parse_args(int argc, char **argv) {
        auto flags = parse_flags(argc, argv);
        keys_file_path = get_required(flags, "keys_file");
        attack_keys_file_path = get_with_default(flags, "attack_keys_file", "");
        keys_file_type = get_with_default(flags, "keys_file_type", "binary");
        read_ratio = stod(get_required(flags, "read"));
        insert_ratio = stod(get_with_default(flags, "insert", "0"));
        delete_ratio = stod(get_with_default(flags, "delete", "0"));
        update_ratio = stod(get_with_default(flags, "update", "0"));
        skew_alpha = stod(get_with_default(flags, "skew_alpha", "1.6"));
        scan_ratio = stod(get_with_default(flags, "scan", "0"));
        scan_num = stoi(get_with_default(flags, "scan_num", "100"));
        operations_num = stoull(get_with_default(flags, "operations_num", "1000000000"));
        table_size = stoll(get_with_default(flags, "table_size", "-1"));
        init_table_ratio = stod(get_with_default(flags, "init_table_ratio", "0.5"));
        del_table_ratio = stod(get_with_default(flags, "del_table_ratio", "0.5"));
        init_table_size = 0;
        all_thread_num = get_comma_separated(flags, "thread_num");
        all_index_type = get_comma_separated(flags, "index");
        sample_distribution = get_with_default(flags, "sample_distribution", "uniform");
        latency_sample = get_boolean_flag(flags, "latency_sample");
        latency_sample_ratio = stod(get_with_default(flags, "latency_sample_ratio", "0.01"));
        error_bound = stoi(get_with_default(flags, "error_bound", "64"));
        output_path = get_with_default(flags, "output_path", "./out.csv");
        latency_output_path = get_with_default(flags, "latency_output_path", "./latency_timeline.csv");
        random_seed = stoul(get_with_default(flags, "seed", "1866"));
        num_optimizations = stoi(get_with_default(flags, "num_optimizations", "8"));
        cold_zone_ratio = stod(get_with_default(flags, "cold_zone_ratio", "0.7"));
        interval_variance = stod(get_with_default(flags, "interval_variance", "0.1"));
        num_repeats = stoi(get_with_default(flags, "num_repeats", "1"));
        gen.seed(random_seed);
        memory_record = get_boolean_flag(flags, "memory");
        dataset_statistic = get_boolean_flag(flags, "dataset_statistic");
        data_shift = get_boolean_flag(flags, "data_shift");
        adjust_density = get_boolean_flag(flags, "adjust_density");
        ban_compress = get_boolean_flag(flags, "ban_compress");
        save_generated_dataset = get_boolean_flag(flags, "save_generated_dataset");
        allow_attack_generation = get_boolean_flag(flags, "generate_attack_keys");
        controller_runtime_enabled = !get_boolean_flag(flags, "disable_controller");

        double ratio_sum = read_ratio + insert_ratio + delete_ratio + update_ratio + scan_ratio;
        INVARIANT(ratio_sum > 0.9999 && ratio_sum < 1.0001);
        INVARIANT(!all_thread_num.empty());
        INVARIANT(!all_index_type.empty());
        INVARIANT(num_repeats > 0);
        INVARIANT(latency_sample_ratio > 0.0 && latency_sample_ratio <= 1.0);
    }

    void run(index_t *index) {
        param_t params[thread_num];
        TSCNS tn;
        tn.init();

        printf("Begin running\n");

        uint64_t start_time = 0, end_time = 0;
        double excluded_ns = 0.0;

        std::string compress_path = output_path;
        std::string suffix = ".csv";

        if (compress_path.size() >= suffix.size() &&
            compress_path.compare(compress_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            compress_path = compress_path.substr(0, compress_path.size() - suffix.size());
        }

        compress_path += "_compress.csv";
        KEY_TYPE final_boundary = zones[1].min_key;
        KEY_TYPE total_range = final_boundary - stored_global_min;
        KEY_TYPE step = total_range / num_optimizations;
        KEY_TYPE current_compress_boundary = stored_global_min;

        // ========== WARMUP ZONE ==========
        {
            #pragma omp parallel num_threads(thread_num)
            {
                int tid = omp_get_thread_num();
                auto paramI = Param(thread_num, tid);
                param_t &thread_param = params[tid];

                uint64_t lat_start = 0, lat_end = 0;
                PAYLOAD_TYPE val;
                auto *scan_result =
                    new std::pair<KEY_TYPE, PAYLOAD_TYPE>[scan_num];

                if (latency_sample) {
                    thread_param.latency.reserve(
                        operations_num * latency_sample_ratio / thread_num * 1.2);
                }

                #pragma omp barrier
                #pragma omp master
                start_time = tn.rdtsc();

                auto &zone = zones[0];
                size_t ops = zone.ops.size();

                #pragma omp for schedule(dynamic, 10000)
                for (size_t i = 0; i < ops; ++i) {
                    auto &op_pair = zone.ops[i];

                    if (op_pair.should_sample)
                        lat_start = tn.rdtsc();

                    switch (op_pair.type) {
                        case READ:
                            thread_param.success_read +=
                                index->get(op_pair.key, val, &paramI);
                            break;
                        case INSERT:
                            thread_param.success_insert +=
                                index->put(op_pair.key, 123456789, &paramI);
                            break;
                        case UPDATE:
                            thread_param.success_update +=
                                index->update(op_pair.key, 234567891, &paramI);
                            break;
                        case DELETE:
                            thread_param.success_remove +=
                                index->remove(op_pair.key, &paramI);
                            break;
                        case SCAN: {
                            auto len =
                                index->scan(op_pair.key, scan_num, scan_result, &paramI);
                            if (len != scan_num)
                                thread_param.scan_not_enough++;
                            break;
                        }
                    }

                    if (op_pair.should_sample) {
                        lat_end = tn.rdtsc();
                        thread_param.latency.emplace_back(lat_start, lat_end);
                    }
                }

                delete[] scan_result;
            } // warmup parallel region ends
        }

        // ========== ATTACK SUB-ZONES ==========
        for (size_t sub_idx = 0; sub_idx < attack_sub_zones.size(); ++sub_idx) {
            // -------- Phase 1: operations --------
            #pragma omp parallel num_threads(thread_num)
            {
                int tid = omp_get_thread_num();
                auto paramI = Param(thread_num, tid);
                param_t &thread_param = params[tid];

                uint64_t lat_start = 0, lat_end = 0;
                PAYLOAD_TYPE val;
                auto *scan_result =
                    new std::pair<KEY_TYPE, PAYLOAD_TYPE>[scan_num];

                auto &sub_zone = attack_sub_zones[sub_idx];
                auto &ops = zones[1].ops;

                #pragma omp for schedule(dynamic, 10000)
                for (size_t i = sub_zone.begin_index;
                     i < sub_zone.end_index; ++i) {
                    auto &op_pair = ops[i];

                    if (op_pair.should_sample)
                        lat_start = tn.rdtsc();

                    switch (op_pair.type) {
                        case READ:
                            thread_param.success_read +=
                                index->get(op_pair.key, val, &paramI);
                            break;
                        case INSERT:
                            thread_param.success_insert +=
                                index->put(op_pair.key, 123456789, &paramI);
                            break;
                        case UPDATE:
                            thread_param.success_update +=
                                index->update(op_pair.key, 234567891, &paramI);
                            break;
                        case DELETE:
                            thread_param.success_remove +=
                                index->remove(op_pair.key, &paramI);
                            break;
                        case SCAN: {
                            auto len =
                                index->scan(op_pair.key, scan_num, scan_result, &paramI);
                            if (len != scan_num)
                                thread_param.scan_not_enough++;
                            break;
                        }
                    }

                    if (op_pair.should_sample) {
                        lat_end = tn.rdtsc();
                        thread_param.latency.emplace_back(lat_start, lat_end);
                    }
                }

                delete[] scan_result;
            }

#if ENABLE_MANUAL_OPTIMIZATION
            if (sub_idx < num_optimizations) {
                uint64_t opt_start = tn.rdtsc();

            #if ENABLE_CONTROLLER
                auto &range = optimization_ranges[sub_idx];
                if (!ban_compress && !adjust_density) {
                    KEY_TYPE next_boundary;
                    if (sub_idx == num_optimizations - 1) next_boundary = final_boundary;
                    else next_boundary = current_compress_boundary + step;
                    index->compress_cold_zone(next_boundary, compress_path);
                    current_compress_boundary = next_boundary;
                    // index->compress_cold_zone(final_boundary, compress_path);
                }
                if(scan_ratio==0) {
                    index->optimize_targeted_zones(
                        range.first,
                        range.second,
                        compress_path,
                        adjust_density
                    );
                }
            #endif

                uint64_t opt_end = tn.rdtsc();
                excluded_ns +=
                    (double)(tn.tsc2ns(opt_end) - tn.tsc2ns(opt_start));
            }
#endif

        }

#if EXTRA

        // if (!ban_compress) {
        //     uint64_t opt_start = tn.rdtsc();
        // // #if ENABLE_CONTROLLER
        //     index->compress_cold_zone(zones[1].max_key, compress_path);
        // // #endif
        //     uint64_t opt_end = tn.rdtsc();
        //     excluded_ns += (double)(tn.tsc2ns(opt_end) - tn.tsc2ns(opt_start));
        // }

        {
            #pragma omp parallel num_threads(thread_num)
            {
                int tid = omp_get_thread_num();
                auto paramI = Param(thread_num, tid);
                param_t &thread_param = params[tid];

                uint64_t lat_start = 0, lat_end = 0;
                PAYLOAD_TYPE val;

                if (latency_sample) {
                    thread_param.latency.reserve(
                        operations_num * latency_sample_ratio / thread_num * 1.2);
                }

                // #pragma omp barrier
                // #pragma omp master
                // start_time = tn.rdtsc();
                auto &zone = zones[1];
                size_t ops = zone.ops.size();

                #pragma omp for schedule(dynamic, 10000)
                for (size_t i = 0; i < ops; ++i) {
                    auto &op_pair = zone.ops[i];

                    if (op_pair.should_sample) lat_start = tn.rdtsc();

                    thread_param.success_read += index->get(op_pair.key, val, &paramI);

                    if (op_pair.should_sample) {
                        lat_end = tn.rdtsc();
                        thread_param.latency.emplace_back(lat_start, lat_end);
                    }
                }
            }
        }
#endif
        // if(!ban_compress && !adjust_density){
        //     uint64_t opt_start = tn.rdtsc();
        //     index->compress_cold_zone(final_boundary, compress_path);
        //     uint64_t opt_end = tn.rdtsc();
        //     excluded_ns +=
        //         (double)(tn.tsc2ns(opt_end) - tn.tsc2ns(opt_start));
        // }

        end_time = tn.rdtsc();
        printf("Finish running\n");

        // ========== MERGE LATENCY & STATS ==========
        std::vector<std::pair<uint64_t, uint64_t>> all_lat;
        for (int i = 0; i < thread_num; ++i) {
            all_lat.insert(all_lat.end(),
                        params[i].latency.begin(),
                        params[i].latency.end());

            stat.success_read   += params[i].success_read;
            stat.success_insert += params[i].success_insert;
            stat.success_update += params[i].success_update;
            stat.success_remove += params[i].success_remove;
            stat.scan_not_enough += params[i].scan_not_enough;
        }

        if (latency_sample) {
            std::sort(all_lat.begin(), all_lat.end(),
                    [](auto &a, auto &b) { return a.first < b.first; });

            for (auto &e : all_lat) {
                stat.latency.push_back(
                    tn.tsc2ns(e.second) - tn.tsc2ns(e.first));
            }
        }

        double effective_ns =
            (double)(tn.tsc2ns(end_time) - tn.tsc2ns(start_time)) - excluded_ns;
        if (effective_ns <= 0.0) {
            throw std::runtime_error("Measured workload duration is not positive");
        }
        stat.elapsed_ns = static_cast<std::uint64_t>(effective_ns);

#if EXTRA
        stat.throughput =
            static_cast<uint64_t>((operations_num+zones[1].ops.size()) / (effective_ns / 1e9));
#else
        stat.throughput =
            static_cast<uint64_t>(operations_num / (effective_ns / 1e9));
#endif

        if (memory_record)
            stat.memory_consumption = index->memory_consumption();
    }

    void print_stat(bool header = false, bool clear_flag = true) {
        (void) header;
        double avg_latency = 0;
        for (auto t : stat.latency) {
            avg_latency += t;
        }
        if (!stat.latency.empty()) {
            avg_latency /= stat.latency.size();
        }
        double latency_variance = 0;
        std::ofstream csv_file(latency_output_path);
        if (latency_sample && !stat.latency.empty()) {
            csv_file << "sample_index,latency_ns\n";
            for (size_t i = 0; i < stat.latency.size(); i++) {
                auto t = stat.latency[i];
                latency_variance += (t - avg_latency) * (t - avg_latency);
                csv_file << i << "," << t << "\n";
            }
            latency_variance /= stat.latency.size();
            std::sort(stat.latency.begin(), stat.latency.end());
            COUT_THIS("Latency Saved to: " << latency_output_path);
        }
        csv_file.close();

        std::cout << "Throughput = " << stat.throughput << '\n'
                  << "Memory: " << stat.memory_consumption << '\n'
                  << "success_read: " << stat.success_read << '\n'
                  << "success_insert: " << stat.success_insert << '\n'
                  << "success_update: " << stat.success_update << '\n'
                  << "success_remove: " << stat.success_remove << '\n'
                  << "scan_not_enough: " << stat.scan_not_enough << '\n'
                  << "Controller inspections: "
                  << stat.controller.inspection_cycles << '\n'
                  << "Hash decisions/completions: "
                  << stat.controller.hash_decisions << '/'
                  << stat.controller.hash_completions << '\n'
                  << "Rebuild decisions/completions: "
                  << stat.controller.rebuild_decisions << '/'
                  << stat.controller.rebuild_completions << '\n'
                  << "Cooling decisions/completions: "
                  << stat.controller.cooling_decisions << '/'
                  << stat.controller.cooling_completions << '\n'
                  << "Cooling bytes reclaimed: "
                  << stat.controller.cooling_bytes_reclaimed << '\n'
                  << "Index adjustments: "
                  << stat.controller.index_adjustments << '\n';

        std::string trigger_csv_path = output_path;
        std::string suffix = ".csv";

        if (trigger_csv_path.size() >= suffix.size() &&
            trigger_csv_path.compare(trigger_csv_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            trigger_csv_path = trigger_csv_path.substr(0, trigger_csv_path.size() - suffix.size());
        }

        trigger_csv_path += "_trigger.csv";
        std::ofstream trigger_file(trigger_csv_path);
        if (!optimization_trigger_points.empty()) {
            for (size_t i = 0; i < optimization_trigger_points.size(); ++i) {
                trigger_file << optimization_trigger_points[i] << (i == optimization_trigger_points.size() - 1 ? "" : ",");
            }
            COUT_THIS("Trigger points saved to: " << trigger_csv_path);
        }
        trigger_file.close();

        std::time_t t = std::time(nullptr);
        char time_str[100];
        if (!file_exists(output_path)) {
            std::ofstream ofile;
            ofile.open(output_path, std::ios::app);
            ofile << "id" << ",";
            ofile << "skew_alpha" << ",";
            ofile << "read_ratio" << "," << "insert_ratio" << "," << "update_ratio" << "," << "scan_ratio" << "," << "delete_ratio" << ",";
            ofile << "key_path" << ",";
            ofile << "index_type" << ",";
            ofile << "throughput" << ",";
            ofile << "init_table_size" << ",";
            ofile << "memory_consumption" << ",";
            ofile << "thread_num" << ",";
            ofile << "min" << ",";
            ofile << "50 percentile" << ",";
            ofile << "90 percentile" << ",";
            ofile << "99 percentile" << ",";
            ofile << "99.9 percentile" << ",";
            ofile << "99.99 percentile" << ",";
            ofile << "max" << ",";
            ofile << "avg" << ",";
            ofile << "seed" << ",";
            ofile << "num_optimizations" << ",";
            ofile << "cold_zone_ratio" << ",";
            ofile << "scan_num" << ",";
            ofile << "latency_variance" << ",";
            ofile << "latency_sample" << ",";
            ofile << "data_shift" << ",";
            ofile << "pgm" << ",";
            ofile << "error_bound" ",";
            ofile << "table_size" << ",";
            ofile << "operations_num" << ",";
            ofile << "elapsed_ns" << ",";
            ofile << "repeat_count" << ",";
            ofile << "throughput_stddev" << ",";
            ofile << "throughput_ci95" << ",";
            ofile << "memory_before" << ",";
            ofile << "memory_delta_bytes" << ",";
            ofile << "memory_stddev" << ",";
            ofile << "latency_sample_count" << ",";
            ofile << "attack_keys_path" << ",";
            ofile << "controller_inspection_cycles" << ",";
            ofile << "controller_hot_inspections" << ",";
            ofile << "controller_cold_inspections" << ",";
            ofile << "hash_decisions" << ",";
            ofile << "hash_completions" << ",";
            ofile << "hash_failures" << ",";
            ofile << "rebuild_decisions" << ",";
            ofile << "rebuild_completions" << ",";
            ofile << "rebuild_failures" << ",";
            ofile << "cooling_decisions" << ",";
            ofile << "cooling_completions" << ",";
            ofile << "cooling_bytes_reclaimed" << ",";
            ofile << "registration_attempts" << ",";
            ofile << "registered_nodes" << ",";
            ofile << "rejected_by_filter" << ",";
            ofile << "monitored_nodes" << ",";
            ofile << "hash_nodes" << ",";
            ofile << "index_adjustments" << std::endl;
        }
        std::ofstream ofile;
        ofile.open(output_path, std::ios::app);
        if (std::strftime(time_str, sizeof(time_str), "%Y%m%d%H%M%S", std::localtime(&t))) {
            ofile << time_str << ',';
        }
        ofile << skew_alpha << ",";
        ofile << read_ratio << "," << insert_ratio << "," << update_ratio << "," << scan_ratio << "," << delete_ratio << ",";
        ofile << keys_file_path << ",";
#if ENABLE_CONTROLLER
        ofile << index_type
              << (controller_runtime_enabled ? "_controlled," : "_monitored,");
#else
        ofile << index_type << ",";
#endif
        ofile << stat.throughput << ",";
        ofile << init_table_size << ",";
        ofile << stat.memory_consumption << ",";
        ofile << thread_num << ",";
        if (latency_sample && !stat.latency.empty()) {
            ofile << stat.latency[0] << ",";
            ofile << stat.latency[static_cast<size_t>(0.5 * stat.latency.size())] << ",";
            ofile << stat.latency[static_cast<size_t>(0.9 * stat.latency.size())] << ",";
            ofile << stat.latency[static_cast<size_t>(0.99 * stat.latency.size())] << ",";
            ofile << stat.latency[static_cast<size_t>(0.999 * stat.latency.size())] << ",";
            ofile << stat.latency[static_cast<size_t>(0.9999 * stat.latency.size())] << ",";
            ofile << stat.latency[stat.latency.size() - 1] << ",";
            ofile << avg_latency << ",";
        } else {
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
        }
        ofile << random_seed << ",";
        ofile << num_optimizations << ",";
        ofile << cold_zone_ratio << ",";
        ofile << scan_num << ",";
        ofile << latency_variance << ",";
        ofile << latency_sample << ",";
        ofile << data_shift << ",";
        ofile << stat.fitness_of_dataset << ",";
        ofile << error_bound << ",";
        ofile << table_size << ",";
        ofile << operations_num << ",";
        ofile << stat.elapsed_ns << ",";
        ofile << num_repeats << ",";
        ofile << throughput_stddev << ",";
        ofile << throughput_ci95 << ",";
        ofile << stat.memory_before << ",";
        ofile << (stat.memory_consumption - stat.memory_before) << ",";
        ofile << memory_stddev << ",";
        ofile << stat.latency.size() << ",";
        ofile << attack_keys_file_path << ",";
        ofile << stat.controller.inspection_cycles << ",";
        ofile << stat.controller.hot_node_inspections << ",";
        ofile << stat.controller.cold_node_inspections << ",";
        ofile << stat.controller.hash_decisions << ",";
        ofile << stat.controller.hash_completions << ",";
        ofile << stat.controller.hash_failures << ",";
        ofile << stat.controller.rebuild_decisions << ",";
        ofile << stat.controller.rebuild_completions << ",";
        ofile << stat.controller.rebuild_failures << ",";
        ofile << stat.controller.cooling_decisions << ",";
        ofile << stat.controller.cooling_completions << ",";
        ofile << stat.controller.cooling_bytes_reclaimed << ",";
        ofile << stat.controller.registration_attempts << ",";
        ofile << stat.controller.registered_nodes << ",";
        ofile << stat.controller.rejected_by_filter << ",";
        ofile << stat.controller.monitored_nodes << ",";
        ofile << stat.controller.hash_nodes << ",";
        ofile << stat.controller.index_adjustments << std::endl;
        ofile.close();
        if (clear_flag) stat.clear();
    }

    static double critical_value_95(std::size_t degrees_of_freedom) {
        static const double values[] = {
            0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
            2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
            2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069,
            2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
        if (degrees_of_freedom < sizeof(values) / sizeof(values[0])) {
            return values[degrees_of_freedom];
        }
        return 1.96;
    }

    static double sample_stddev(const std::vector<double>& values) {
        if (values.size() < 2) return 0.0;
        double sum = 0.0;
        for (double value : values) sum += value;
        const double mean = sum / static_cast<double>(values.size());
        double squared_deviation = 0.0;
        for (double value : values) {
            const double difference = value - mean;
            squared_deviation += difference * difference;
        }
        return std::sqrt(squared_deviation /
                         static_cast<double>(values.size() - 1));
    }

    void run_benchmark() {
        is_read_only = (read_ratio == 1.0);
        load_keys();
        const auto base_seed = random_seed;
        for (auto s: all_index_type) {
            for (auto t: all_thread_num) {
                thread_num = stoi(t);
                index_type = s;
                Stat avg_stat;
                avg_stat.clear();
                std::vector<std::vector<double>> all_latencies;
                std::vector<double> throughputs;
                std::vector<double> memory_measurements;
                all_latencies.resize(num_repeats);
                throughputs.reserve(num_repeats);
                memory_measurements.reserve(num_repeats);
                for (int rep = 0; rep < num_repeats; ++rep) {
                    random_seed = base_seed + static_cast<std::size_t>(rep);
                    gen.seed(random_seed);
                    index_t *index;
                    prepare(index, keys);
                    generate_attack_keys_post_bulkload(index);
                    generate_operations(keys);
                    const auto controller_before = index->controller_statistics();
                    if (memory_record) {
                        stat.memory_before = index->memory_consumption();
                    }
#if ENABLE_CONTROLLER
                    if (controller_runtime_enabled) {
                        index->start_controller();
                    }
#endif
                    run(index);
#if ENABLE_CONTROLLER
                    if (controller_runtime_enabled) {
                        index->stop_controller();
                    }
#endif
                    if (memory_record) {
                        stat.memory_consumption = index->memory_consumption();
                    }
                    const auto controller_after = index->controller_statistics();
                    stat.controller = ControllerStatistics::delta(
                        controller_after, controller_before);
                    if (latency_sample) {
                        all_latencies[rep] = stat.latency;
                    }
                    throughputs.push_back(static_cast<double>(stat.throughput));
                    if (memory_record) {
                        memory_measurements.push_back(
                            static_cast<double>(stat.memory_consumption));
                    }
                    avg_stat.add(stat);
#if ENABLE_CONTROLLER
                    index->print_controller_stats();
#endif
                    if (index != nullptr) delete index;
                    stat.clear();
                }
                avg_stat.divide(num_repeats);
                stat = avg_stat;
                if (latency_sample) {
                    stat.latency.clear();
                    std::size_t total_samples = 0;
                    for (const auto& samples : all_latencies) {
                        total_samples += samples.size();
                    }
                    stat.latency.reserve(total_samples);
                    for (const auto& samples : all_latencies) {
                        stat.latency.insert(stat.latency.end(),
                                            samples.begin(), samples.end());
                    }
                }
                throughput_stddev = sample_stddev(throughputs);
                throughput_ci95 = throughputs.size() < 2
                    ? 0.0
                    : critical_value_95(throughputs.size() - 1) *
                          throughput_stddev /
                          std::sqrt(static_cast<double>(throughputs.size()));
                memory_stddev = sample_stddev(memory_measurements);
                random_seed = base_seed;
                std::cout << "\nAverage Stats over " << num_repeats << " repeats:\n";
                print_stat(false, true);
            }
        }
    }
};
