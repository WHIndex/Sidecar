#pragma once
#include <fstream>

#include"src/core/lipp.h"
#include"../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class LIPPOLInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    void init(Param *param = nullptr) {}

    void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool remove(KEY_TYPE key, Param *param = nullptr);

    size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair <KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr);

    long long memory_consumption() {
        long long bytes = lipp.total_size();
#if ENABLE_CONTROLLER
        bytes += static_cast<long long>(
            lippolc::g_lipp_controller.metadata_memory_bytes());
#endif
        return bytes;
    }

    ControllerStatistics controller_statistics() const override {
        ControllerStatistics result;
        result.index_adjustments =
            static_cast<std::uint64_t>(lipp.successful_rebuilds());
#if ENABLE_CONTROLLER
        const auto stats = lippolc::g_lipp_controller.statistics();
        result.hash_decisions = stats.hash_decisions;
        result.rebuild_decisions = stats.rebuild_decisions;
        result.cooling_decisions = stats.cooling_decisions;
        result.hash_completions = stats.hash_completions;
        result.rebuild_completions = stats.rebuild_completions;
        result.cooling_completions = stats.cooling_completions;
        result.cooling_bytes_reclaimed = stats.cooling_bytes_reclaimed;
        result.hash_failures = stats.hash_failures;
        result.rebuild_failures = stats.rebuild_failures;
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
        lippolc::g_lipp_controller.print_statistics();
        std::cout << "Successful adjustments: "
                  << lipp.successful_rebuilds() << std::endl;
    }
    void start_controller() override {
        lippolc::g_lipp_controller.start();
    }
    void stop_controller() override {
        lippolc::g_lipp_controller.stop();
    }
#endif

    void optimize_targeted_zones(KEY_TYPE start_key, KEY_TYPE end_key, std::string file_name = "compression_memory.csv",
        bool adjust_density = false) {
        if (adjust_density) return;
        #if ENABLE_CONTROLLER
        lipp.optimize_targeted_zones(start_key, end_key, 2);
        #endif
        std::ofstream csv_file(file_name, std::ios::app);
        if (csv_file.tellp() == 0) {
            csv_file << "compressed_nodes,memory_size_bytes" << std::endl;
        }
        long long current_size = memory_consumption();
        csv_file << 1 << "," << current_size << std::endl;
    }

    void compress_cold_zone(KEY_TYPE end_key, const std::string& csv_path = "compression_memory.csv") {
        #if ENABLE_CONTROLLER
        lipp.compress_cold_zone(end_key);
        #endif
        std::ofstream csv_file(csv_path, std::ios::app);
        if (csv_file.tellp() == 0) {
            csv_file << "compressed_nodes,memory_size_bytes" << std::endl;
        }
        long long current_size = memory_consumption();
        csv_file << 0 << "," << current_size << std::endl;
    }
 void generate_attack_keys(
        std::vector<KEY_TYPE>& attack_keys,
        KEY_TYPE original_min,
        KEY_TYPE original_max,
        size_t original_size,
        size_t num_attack_keys = 200000000
    ) {
        lipp.generate_attack_keys(attack_keys, original_min, original_max, original_size, num_attack_keys);
    }

private:
    lippolc::LIPP<KEY_TYPE, PAYLOAD_TYPE> lipp;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                      Param *param) {
    lipp.bulk_load(key_value, static_cast<int>(num));
    auto memory = memory_consumption();
    std::cout << "Size after bulkload: " << memory << std::endl;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
    return lipp.at(key, val);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    lipp.insert(key, value);
    return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return lipp.update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
    return lipp.remove(key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair <KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
    return lipp.range_query_len(result, key_low_bound, key_num);
}
