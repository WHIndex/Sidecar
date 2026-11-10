#include <iomanip>
#include <cstdint>

#pragma once

struct Param {
  size_t worker_num;
  uint32_t thread_id;

  Param(size_t worker_num, uint32_t thread_id) : worker_num(worker_num), thread_id(thread_id) {}
};

struct ControllerStatistics {
  std::uint64_t hash_decisions{0};
  std::uint64_t rebuild_decisions{0};
  std::uint64_t cooling_decisions{0};
  std::uint64_t hash_completions{0};
  std::uint64_t rebuild_completions{0};
  std::uint64_t cooling_completions{0};
  std::uint64_t cooling_bytes_reclaimed{0};
  std::uint64_t hash_failures{0};
  std::uint64_t rebuild_failures{0};
  std::uint64_t inspection_cycles{0};
  std::uint64_t hot_node_inspections{0};
  std::uint64_t cold_node_inspections{0};
  std::uint64_t registration_attempts{0};
  std::uint64_t registered_nodes{0};
  std::uint64_t rejected_by_filter{0};
  std::uint64_t monitored_nodes{0};
  std::uint64_t hash_nodes{0};
  std::uint64_t index_adjustments{0};

  static std::uint64_t counter_delta(std::uint64_t after,
                                     std::uint64_t before) noexcept {
    return after >= before ? after - before : 0;
  }

  static ControllerStatistics delta(const ControllerStatistics& after,
                                    const ControllerStatistics& before) {
    ControllerStatistics result;
    result.hash_decisions = counter_delta(after.hash_decisions, before.hash_decisions);
    result.rebuild_decisions = counter_delta(after.rebuild_decisions, before.rebuild_decisions);
    result.cooling_decisions = counter_delta(after.cooling_decisions, before.cooling_decisions);
    result.hash_completions = counter_delta(after.hash_completions, before.hash_completions);
    result.rebuild_completions = counter_delta(after.rebuild_completions, before.rebuild_completions);
    result.cooling_completions = counter_delta(after.cooling_completions, before.cooling_completions);
    result.cooling_bytes_reclaimed = counter_delta(
        after.cooling_bytes_reclaimed, before.cooling_bytes_reclaimed);
    result.hash_failures = counter_delta(after.hash_failures, before.hash_failures);
    result.rebuild_failures = counter_delta(after.rebuild_failures, before.rebuild_failures);
    result.inspection_cycles = counter_delta(after.inspection_cycles, before.inspection_cycles);
    result.hot_node_inspections = counter_delta(after.hot_node_inspections, before.hot_node_inspections);
    result.cold_node_inspections = counter_delta(after.cold_node_inspections, before.cold_node_inspections);
    result.registration_attempts = counter_delta(after.registration_attempts, before.registration_attempts);
    result.registered_nodes = counter_delta(after.registered_nodes, before.registered_nodes);
    result.rejected_by_filter = counter_delta(after.rejected_by_filter, before.rejected_by_filter);
    result.monitored_nodes = after.monitored_nodes;
    result.hash_nodes = after.hash_nodes;
    result.index_adjustments =
        counter_delta(after.index_adjustments, before.index_adjustments);
    return result;
  }

  void add(const ControllerStatistics& other) {
    hash_decisions += other.hash_decisions;
    rebuild_decisions += other.rebuild_decisions;
    cooling_decisions += other.cooling_decisions;
    hash_completions += other.hash_completions;
    rebuild_completions += other.rebuild_completions;
    cooling_completions += other.cooling_completions;
    cooling_bytes_reclaimed += other.cooling_bytes_reclaimed;
    hash_failures += other.hash_failures;
    rebuild_failures += other.rebuild_failures;
    inspection_cycles += other.inspection_cycles;
    hot_node_inspections += other.hot_node_inspections;
    cold_node_inspections += other.cold_node_inspections;
    registration_attempts += other.registration_attempts;
    registered_nodes += other.registered_nodes;
    rejected_by_filter += other.rejected_by_filter;
    monitored_nodes += other.monitored_nodes;
    hash_nodes += other.hash_nodes;
    index_adjustments += other.index_adjustments;
  }

  void divide(std::uint64_t divisor) {
    if (divisor == 0) return;
    hash_decisions /= divisor;
    rebuild_decisions /= divisor;
    cooling_decisions /= divisor;
    hash_completions /= divisor;
    rebuild_completions /= divisor;
    cooling_completions /= divisor;
    cooling_bytes_reclaimed /= divisor;
    hash_failures /= divisor;
    rebuild_failures /= divisor;
    inspection_cycles /= divisor;
    hot_node_inspections /= divisor;
    cold_node_inspections /= divisor;
    registration_attempts /= divisor;
    registered_nodes /= divisor;
    rejected_by_filter /= divisor;
    monitored_nodes /= divisor;
    hash_nodes /= divisor;
    index_adjustments /= divisor;
  }
};

enum ZoneType {
    COLD,
    NONLINEAR,
    LINEAR_INSERT,
    NORMAL
};

struct BaseCompare {
  template<class T1, class T2>
  bool operator()(const T1 &x, const T2 &y) const {
    static_assert(
      std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
      "Comparison types must be numeric.");
    return x < y;
  }
};

template<class KEY_TYPE, class PAYLOAD_TYPE, class KeyComparator=BaseCompare>
class indexInterface {
public:
  virtual ~indexInterface() = default;

  virtual void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) = 0;

  virtual bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) = 0;

  virtual bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool remove(KEY_TYPE key, Param *param = nullptr) = 0;

  virtual size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                      Param *param = nullptr) = 0;

  virtual void init(Param *param = nullptr) = 0;

  virtual long long memory_consumption() = 0; // bytes
  virtual ControllerStatistics controller_statistics() const { return {}; }
#if ENABLE_CONTROLLER
  virtual void print_controller_stats() { return; }
  virtual void start_controller() { return; }
  virtual void stop_controller() { return; }
#endif
  virtual void optimize_targeted_zones(KEY_TYPE start_key, KEY_TYPE end_key, std::string file_name, bool adjust_density) {
    return;
  }
  virtual void compress_cold_zone(KEY_TYPE end_key, const std::string& csv_path = "compression_memory.csv") { return; }

  virtual void generate_attack_keys(
      std::vector<KEY_TYPE>& attack_keys,
        KEY_TYPE original_min,
        KEY_TYPE original_max,
        size_t original_size,
        size_t num_attack_keys = 200000000)
  {
    attack_keys.clear();
    std::cerr << "Warning: generate_attack_keys not implemented for this index type" << std::endl;
  }
};
