#ifndef ALEX_CONTROLLER_H
#define ALEX_CONTROLLER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>

#include "runtime_metadata.h"

namespace alex_controller {

using NodeMetadata = index_controller::RuntimeMetadata;
using NodeState = index_controller::NodeState;

class ALEXController {
 public:
  struct StatisticsSnapshot {
    std::uint64_t hash_decisions{0};
    std::uint64_t cooling_decisions{0};
    std::uint64_t hash_completions{0};
    std::uint64_t cooling_completions{0};
    std::uint64_t cooling_bytes_reclaimed{0};
    std::uint64_t inspection_cycles{0};
    std::uint64_t hot_node_inspections{0};
    std::uint64_t cold_node_inspections{0};
    std::uint64_t registration_attempts{0};
    std::uint64_t registered_nodes{0};
    std::uint64_t rejected_by_filter{0};
    std::uint64_t monitored_nodes{0};
    std::uint64_t hash_nodes{0};
  };

  struct Config {
    std::size_t min_monitor_size{1};
    std::size_t inspection_budget{8};
    std::size_t cooling_batch_size{64};
    std::uint64_t check_interval_ms{25};
    std::uint64_t cooling_interval_ms{1000};

    double conflict_weight{5.0};
    double shift_weight{1.0};
    double read_weight{0.05};

    std::uint64_t fast_smo_interval_ms{1000};
    std::uint64_t min_post_smo_inserts{1};
    double high_shift_per_insert{25.0};
    double minimum_gap_ratio{0.10};

    std::uint64_t cold_time_ms{10000};
    std::uint64_t cold_access_threshold{4};

    std::uint64_t hash_horizon_ms{5000};
    double maximum_hash_growth{2.0};
    bool enable_debug_output{false};
  };

  static ALEXController& instance() {
    static ALEXController controller;
    return controller;
  }

  const Config& get_config() const noexcept { return config_; }

  StatisticsSnapshot statistics() const noexcept {
    return StatisticsSnapshot{
        stats_.hash_decisions.load(std::memory_order_relaxed),
        stats_.cooling_decisions.load(std::memory_order_relaxed),
        stats_.hash_completions.load(std::memory_order_relaxed),
        stats_.cooling_completions.load(std::memory_order_relaxed),
        stats_.cooling_bytes_reclaimed.load(std::memory_order_relaxed),
        stats_.inspection_cycles.load(std::memory_order_relaxed),
        stats_.hot_node_inspections.load(std::memory_order_relaxed),
        stats_.cold_node_inspections.load(std::memory_order_relaxed),
        stats_.registration_attempts.load(std::memory_order_relaxed),
        stats_.registered_nodes.load(std::memory_order_relaxed),
        stats_.rejected_by_filter.load(std::memory_order_relaxed),
        live_metadata_count_.load(std::memory_order_relaxed),
        live_hash_node_count_.load(std::memory_order_relaxed)};
  }

  std::size_t metadata_memory_bytes() const noexcept {
    return live_metadata_count_.load(std::memory_order_acquire) *
               sizeof(NodeMetadata) +
           monitored_nodes_.size() * sizeof(RegistrySlot);
  }

  bool has_pending_foreground_tasks() const noexcept {
    return pending_foreground_tasks_.load(std::memory_order_relaxed) != 0;
  }

  void complete_foreground_task() noexcept {
    auto pending = pending_foreground_tasks_.load(std::memory_order_relaxed);
    while (pending != 0 &&
           !pending_foreground_tasks_.compare_exchange_weak(
               pending, pending - 1, std::memory_order_release,
               std::memory_order_relaxed)) {
    }
  }

  void record_hash_execution(std::size_t size,
                             std::uint64_t elapsed_ns) noexcept {
    stats_.hash_completions.fetch_add(1, std::memory_order_relaxed);
    live_hash_node_count_.fetch_add(1, std::memory_order_relaxed);
    stats_.hash_execution_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    auto maximum_size = stats_.maximum_hash_size.load(std::memory_order_relaxed);
    while (maximum_size < size &&
           !stats_.maximum_hash_size.compare_exchange_weak(
               maximum_size, size, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    auto maximum_ns =
        stats_.maximum_hash_execution_ns.load(std::memory_order_relaxed);
    while (maximum_ns < elapsed_ns &&
           !stats_.maximum_hash_execution_ns.compare_exchange_weak(
               maximum_ns, elapsed_ns, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }

  void record_cooling_savings(std::size_t bytes) noexcept {
    stats_.cooling_bytes_reclaimed.fetch_add(bytes,
                                              std::memory_order_relaxed);
  }

  void enable_debug_output(bool enabled) noexcept {
    config_.enable_debug_output = enabled;
  }

  void set_check_interval(int milliseconds) noexcept {
    if (milliseconds > 0) {
      config_.check_interval_ms = static_cast<std::uint64_t>(milliseconds);
    }
  }

  void set_hash_conflict_ratio(double) noexcept {}

  void set_worker_threads(int) noexcept {}

  NodeMetadata* register_node(
      std::size_t size, std::size_t capacity,
      const NodeMetadata* previous = nullptr, void* cooling_context = nullptr,
      NodeMetadata::CoolingCallback cooling_callback = nullptr,
      bool hash_node = false) {
    stats_.registration_attempts.fetch_add(1, std::memory_order_relaxed);
    if (size < config_.min_monitor_size) {
      stats_.rejected_by_filter.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    auto* metadata = new NodeMetadata();
    metadata->init(size, capacity, 0, hash_node);
    if (previous != nullptr) {
      metadata->inherit_smo_history(previous);
    }
    if (cooling_callback != nullptr) {
      auto installed = cooling_callback_.load(std::memory_order_acquire);
      while (installed == nullptr &&
             !cooling_callback_.compare_exchange_weak(
                 installed, cooling_callback, std::memory_order_release,
                 std::memory_order_acquire)) {
      }
    }
    metadata->is_monitored.store(true, std::memory_order_release);
    {
      std::unique_lock<std::shared_mutex> lock(registry_mutex_);
      const auto position = monitored_nodes_.grow_by(1);
      metadata->registry_handle = static_cast<std::uint32_t>(
          position - monitored_nodes_.begin() + 1);
      position->metadata = metadata;
      position->cooling_context = cooling_context;
      position->clear_snapshots();
      live_metadata_count_.fetch_add(1, std::memory_order_release);
      if (hash_node) {
        live_hash_node_count_.fetch_add(1, std::memory_order_release);
      }
    }
    stats_.registered_nodes.fetch_add(1, std::memory_order_relaxed);
    return metadata;
  }

  void retire_node(NodeMetadata* metadata) noexcept {
    if (metadata == nullptr) {
      return;
    }
    const auto previous_state = metadata->mark_deleted();
    if (previous_state == NodeState::NEED_HASH) {
      complete_foreground_task();
    }
  }

  void unregister_node(NodeMetadata* metadata) noexcept {
    if (metadata == nullptr) {
      return;
    }
    const bool was_hash = metadata->is_hash();
    retire_node(metadata);
    const auto handle = metadata->registry_handle;
    {
      std::unique_lock<std::shared_mutex> lock(registry_mutex_);
      if (handle != 0 && handle <= monitored_nodes_.size() &&
          monitored_nodes_[handle - 1].metadata == metadata) {
        auto& slot = monitored_nodes_[handle - 1];
        slot.metadata = nullptr;
        slot.cooling_context = nullptr;
        slot.clear_snapshots();
      }
    }
    metadata->wait_for_callbacks();
    delete metadata;
    live_metadata_count_.fetch_sub(1, std::memory_order_release);
    if (was_hash) {
      live_hash_node_count_.fetch_sub(1, std::memory_order_release);
    }
  }

  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
      return;
    }
    {
      std::shared_lock<std::shared_mutex> lock(registry_mutex_);
      const auto total = monitored_nodes_.size();
      for (std::size_t index = 0; index < total; ++index) {
        auto& slot = monitored_nodes_[index];
        auto* metadata = slot.metadata;
        if (metadata != nullptr) {
          slot.snapshot_inserts.store(
              metadata->total_inserts.load(std::memory_order_relaxed),
              std::memory_order_relaxed);
          slot.snapshot_lookups.store(
              metadata->total_lookups.load(std::memory_order_relaxed),
              std::memory_order_relaxed);
          slot.snapshot_structural_work.store(
              metadata->total_structural_work.load(
                  std::memory_order_relaxed),
              std::memory_order_relaxed);
        }
      }
    }
    last_inspection_ns_ = index_controller::steady_now_ns();
    decision_thread_ = std::thread(&ALEXController::decision_loop, this);
    cooling_thread_ = std::thread(&ALEXController::cooling_loop, this);
  }

  void stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
      return;
    }
    if (decision_thread_.joinable()) {
      decision_thread_.join();
    }
    if (cooling_thread_.joinable()) {
      cooling_thread_.join();
    }
    drain_cooling_queue();
  }

  void print_statistics() const {
    const auto hash_completions =
        stats_.hash_completions.load(std::memory_order_relaxed);
    const auto total_hash_ns =
        stats_.hash_execution_ns.load(std::memory_order_relaxed);
    std::cout << "\n=== ALEX Controller Statistics ===\n"
              << "Hash decisions:        "
              << stats_.hash_decisions.load(std::memory_order_relaxed) << '\n'
              << "Hash completions:      "
              << hash_completions << '\n'
              << "Average hash time:     "
              << (hash_completions == 0
                      ? 0.0
                      : static_cast<double>(total_hash_ns) /
                            static_cast<double>(hash_completions) / 1000.0)
              << " us\n"
              << "Largest hash node:     "
              << stats_.maximum_hash_size.load(std::memory_order_relaxed)
              << '\n'
              << "Longest hash time:     "
              << stats_.maximum_hash_execution_ns.load(
                     std::memory_order_relaxed) /
                     1000.0
              << " us\n"
              << "Cooling decisions:     "
              << stats_.cooling_decisions.load(std::memory_order_relaxed)
              << '\n'
              << "Cooling completions:   "
              << stats_.cooling_completions.load(std::memory_order_relaxed)
              << '\n'
              << "Cooling bytes saved:   "
              << stats_.cooling_bytes_reclaimed.load(
                     std::memory_order_relaxed)
              << '\n'
              << "Inspection cycles:     "
              << stats_.inspection_cycles.load(std::memory_order_relaxed)
              << '\n'
              << "Registration attempts: "
              << stats_.registration_attempts.load(std::memory_order_relaxed)
              << '\n'
              << "Registered nodes:      "
              << stats_.registered_nodes.load(std::memory_order_relaxed) << '\n'
              << "Rejected by filter:    "
              << stats_.rejected_by_filter.load(std::memory_order_relaxed)
              << "\n==================================\n";
  }

  void print_top_hot_nodes() const {
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    std::vector<NodeMetadata*> nodes;
    nodes.reserve(monitored_nodes_.size());
    const auto total = monitored_nodes_.size();
    for (std::size_t index = 0; index < total; ++index) {
      auto* metadata = monitored_nodes_[index].metadata;
      if (metadata != nullptr && metadata->is_node_valid()) {
        nodes.push_back(metadata);
      }
    }
    const auto limit = std::min<std::size_t>(10, nodes.size());
    std::partial_sort(
        nodes.begin(), nodes.begin() + limit, nodes.end(),
        [](const NodeMetadata* left, const NodeMetadata* right) {
          return left->access_count() > right->access_count();
        });
    for (std::size_t index = 0; index < limit; ++index) {
      const auto* metadata = nodes[index];
      std::cout << "Node[" << index
                << "] size="
                << metadata->current_size.load(std::memory_order_relaxed)
                << " inserts="
                << metadata->total_inserts.load(std::memory_order_relaxed)
                << " shifts="
                << metadata->total_structural_work.load(
                       std::memory_order_relaxed)
                << " hash=" << metadata->is_hash() << '\n';
    }
  }

 private:
  struct Candidate {
    NodeMetadata* metadata;
    std::uint32_t registry_handle;
    void* cooling_context;
    index_controller::RuntimeSample sample;
  };

  struct CoolingTask {
    NodeMetadata* metadata{nullptr};
    void* cooling_context{nullptr};
  };

  struct RegistrySlot {
    NodeMetadata* metadata{nullptr};
    void* cooling_context{nullptr};
    std::atomic<std::uint32_t> snapshot_inserts{0};
    std::atomic<std::uint32_t> snapshot_lookups{0};
    std::atomic<std::uint32_t> snapshot_structural_work{0};

    RegistrySlot() = default;
    RegistrySlot(const RegistrySlot&) noexcept {}
    RegistrySlot& operator=(const RegistrySlot&) noexcept {
      metadata = nullptr;
      cooling_context = nullptr;
      clear_snapshots();
      return *this;
    }

    void clear_snapshots() noexcept {
      snapshot_inserts.store(0, std::memory_order_relaxed);
      snapshot_lookups.store(0, std::memory_order_relaxed);
      snapshot_structural_work.store(0, std::memory_order_relaxed);
    }
  };

  struct Stats {
    std::atomic<std::uint64_t> hash_decisions{0};
    std::atomic<std::uint64_t> hash_completions{0};
    std::atomic<std::uint64_t> hash_execution_ns{0};
    std::atomic<std::size_t> maximum_hash_size{0};
    std::atomic<std::uint64_t> maximum_hash_execution_ns{0};
    std::atomic<std::uint64_t> cooling_decisions{0};
    std::atomic<std::uint64_t> cooling_completions{0};
    std::atomic<std::uint64_t> cooling_bytes_reclaimed{0};
    std::atomic<std::uint64_t> inspection_cycles{0};
    std::atomic<std::uint64_t> hot_node_inspections{0};
    std::atomic<std::uint64_t> cold_node_inspections{0};
    std::atomic<std::uint64_t> registration_attempts{0};
    std::atomic<std::uint64_t> registered_nodes{0};
    std::atomic<std::uint64_t> rejected_by_filter{0};
  };

  ALEXController() = default;

  ~ALEXController() {
    stop();
    const auto total = monitored_nodes_.size();
    for (std::size_t index = 0; index < total; ++index) {
      auto* metadata = monitored_nodes_[index].metadata;
      delete metadata;
    }
  }

  ALEXController(const ALEXController&) = delete;
  ALEXController& operator=(const ALEXController&) = delete;

  static std::uint64_t milliseconds_to_ns(std::uint64_t milliseconds) {
    return milliseconds * 1000000ULL;
  }

  void decision_loop() {
    while (running_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(config_.check_interval_ms));
      if (!running_.load(std::memory_order_acquire)) {
        break;
      }
      inspect_nodes();
    }
  }

  void inspect_nodes() {
    stats_.inspection_cycles.fetch_add(1, std::memory_order_relaxed);
    const auto inspection_ns = index_controller::steady_now_ns();
    auto period_ns = inspection_ns - last_inspection_ns_;
    last_inspection_ns_ = inspection_ns;
    if (period_ns == 0) {
      period_ns = milliseconds_to_ns(config_.check_interval_ms);
    }
    std::shared_lock<std::shared_mutex> registry_lock(registry_mutex_);
    std::vector<Candidate> candidates;
    candidates.reserve(monitored_nodes_.size());
    const auto total = monitored_nodes_.size();
    for (std::size_t index = 0; index < total; ++index) {
      auto& slot = monitored_nodes_[index];
      auto* metadata = slot.metadata;
      if (metadata == nullptr || !metadata->is_node_valid()) {
        continue;
      }
      auto sample = metadata->take_sample(
          config_.conflict_weight, config_.shift_weight,
          config_.read_weight, true, slot.snapshot_inserts,
          slot.snapshot_lookups, slot.snapshot_structural_work);
      if (metadata->state.load(std::memory_order_acquire) ==
          NodeState::NORMAL) {
        candidates.push_back(Candidate{metadata, metadata->registry_handle,
                                       slot.cooling_context, sample});
      }
    }
    registry_lock.unlock();
    if (candidates.empty()) {
      return;
    }

    const auto hot_budget = std::min<std::size_t>(
        config_.inspection_budget, (candidates.size() + 1) / 2);
    stats_.hot_node_inspections.fetch_add(hot_budget,
                                          std::memory_order_relaxed);
    auto descending = [](const Candidate& left, const Candidate& right) {
      return left.sample.priority > right.sample.priority;
    };
    std::sort(candidates.begin(), candidates.end(), descending);
    for (std::size_t index = 0; index < hot_budget; ++index) {
      auto* metadata = pin_candidate(candidates[index]);
      if (metadata != nullptr) {
        inspect_hot_node(candidates[index], period_ns);
        metadata->unpin_callback();
      }
    }

    const auto cold_budget = std::min<std::size_t>(
        config_.inspection_budget, candidates.size() - hot_budget);
    stats_.cold_node_inspections.fetch_add(cold_budget,
                                           std::memory_order_relaxed);
    const auto now = index_controller::steady_now_ms();
    for (std::size_t index = 0; index < cold_budget; ++index) {
      auto& candidate = candidates[candidates.size() - 1 - index];
      auto* metadata = pin_candidate(candidate);
      if (metadata != nullptr) {
        inspect_cold_node(candidate, now);
        metadata->unpin_callback();
      }
    }
  }

  NodeMetadata* pin_candidate(const Candidate& candidate) noexcept {
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    const auto handle = candidate.registry_handle;
    if (handle == 0 || handle > monitored_nodes_.size()) {
      return nullptr;
    }
    auto* metadata = monitored_nodes_[handle - 1].metadata;
    if (metadata != candidate.metadata || !metadata->try_pin_callback()) {
      return nullptr;
    }
    return metadata;
  }

  void inspect_hot_node(const Candidate& candidate,
                        std::uint64_t period_ns) {
    auto* metadata = candidate.metadata;
    if (metadata->is_hash()) {
      return;
    }

    const auto previous_smo =
        metadata->previous_smo_ms.load(std::memory_order_relaxed);
    const auto last_smo =
        metadata->last_smo_ms.load(std::memory_order_relaxed);
    if (previous_smo == 0 ||
        index_controller::elapsed_ms(last_smo, previous_smo) >
            config_.fast_smo_interval_ms) {
      return;
    }

    const auto total_inserts =
        metadata->total_inserts.load(std::memory_order_relaxed);
    const auto post_smo_inserts =
        total_inserts -
        metadata->smo_insert_baseline.load(std::memory_order_relaxed);
    if (post_smo_inserts < config_.min_post_smo_inserts) {
      return;
    }
    const auto post_smo_shifts =
        metadata->total_structural_work.load(std::memory_order_relaxed) -
        metadata->smo_structural_baseline.load(std::memory_order_relaxed);
    const auto shift_intensity =
        static_cast<double>(post_smo_shifts) /
        static_cast<double>(post_smo_inserts);
    const auto size = metadata->current_size.load(std::memory_order_relaxed);
    const auto capacity =
        metadata->current_capacity.load(std::memory_order_relaxed);
    const auto gap_ratio = capacity == 0 || size >= capacity
                               ? 0.0
                               : static_cast<double>(capacity - size) /
                                     static_cast<double>(capacity);
    if (shift_intensity < config_.high_shift_per_insert ||
        gap_ratio < config_.minimum_gap_ratio) {
      return;
    }

    const auto insertion_rate =
        period_ns == 0
            ? 0.0
            : static_cast<double>(candidate.sample.inserts) /
                  static_cast<double>(period_ns);
    metadata->estimate_hash_capacity(
        insertion_rate, milliseconds_to_ns(config_.hash_horizon_ms),
        config_.maximum_hash_growth);
    if (metadata->try_schedule(NodeState::NEED_HASH)) {
      pending_foreground_tasks_.fetch_add(1, std::memory_order_release);
      stats_.hash_decisions.fetch_add(1, std::memory_order_relaxed);
      if (config_.enable_debug_output) {
        std::cout << "[ALEX controller] hash size=" << size
                  << " shift/insert=" << shift_intensity
                  << " gap=" << gap_ratio << '\n';
      }
    }
  }

  void inspect_cold_node(const Candidate& candidate, std::uint32_t now) {
    auto* metadata = candidate.metadata;
    const auto last_access =
        metadata->last_access_ms.load(std::memory_order_relaxed);
    if (last_access == 0 ||
        index_controller::elapsed_ms(now, last_access) <
            config_.cold_time_ms ||
        candidate.sample.accesses > config_.cold_access_threshold ||
        metadata->is_compacted_layout() ||
        cooling_callback_.load(std::memory_order_acquire) == nullptr) {
      return;
    }

    metadata->record_cooling_baseline();
    if (metadata->try_schedule(NodeState::IN_COOLING)) {
      if (metadata->try_pin_callback()) {
        cooling_queue_.push(CoolingTask{metadata,
                                        candidate.cooling_context});
        stats_.cooling_decisions.fetch_add(1, std::memory_order_relaxed);
      } else {
        metadata->cancel_task();
      }
    }
  }

  void drain_cooling_queue() noexcept {
    CoolingTask task;
    while (cooling_queue_.try_pop(task)) {
      if (task.metadata != nullptr) {
        task.metadata->cancel_task();
        task.metadata->unpin_callback();
      }
    }
  }

  void cooling_loop() {
    while (running_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(config_.cooling_interval_ms));
      for (std::size_t count = 0; count < config_.cooling_batch_size;
           ++count) {
        CoolingTask task;
        if (!cooling_queue_.try_pop(task)) {
          break;
        }
        execute_cooling(task);
      }
    }
  }

  void execute_cooling(const CoolingTask& task) {
    auto* metadata = task.metadata;
    if (metadata == nullptr || !metadata->is_node_valid()) {
      if (metadata != nullptr) {
        metadata->unpin_callback();
      }
      return;
    }
    if (!metadata->cooling_baseline_unchanged()) {
      metadata->cancel_task();
      metadata->unpin_callback();
      return;
    }
    if (!metadata->try_begin(NodeState::IN_COOLING)) {
      metadata->unpin_callback();
      return;
    }
    const auto callback = cooling_callback_.load(std::memory_order_acquire);
    const bool compacted =
        callback != nullptr && callback(task.cooling_context, metadata);
    if (metadata->is_node_valid()) {
      metadata->finish_execution(metadata->is_hash());
    }
    if (compacted) {
      stats_.cooling_completions.fetch_add(1, std::memory_order_relaxed);
    }
    metadata->unpin_callback();
  }

  Config config_;
  Stats stats_;
  tbb::concurrent_vector<RegistrySlot> monitored_nodes_;
  tbb::concurrent_queue<CoolingTask> cooling_queue_;
  mutable std::shared_mutex registry_mutex_;
  std::atomic<std::size_t> live_metadata_count_{0};
  std::atomic<std::size_t> live_hash_node_count_{0};
  std::atomic<std::uint64_t> pending_foreground_tasks_{0};
  std::atomic<NodeMetadata::CoolingCallback> cooling_callback_{nullptr};
  std::atomic<bool> running_{false};
  std::uint64_t last_inspection_ns_{0};
  std::thread decision_thread_;
  std::thread cooling_thread_;
};

}  // namespace alex_controller

namespace alexol {
static auto& g_alex_controller = alex_controller::ALEXController::instance();
}

#endif  // ALEX_CONTROLLER_H
