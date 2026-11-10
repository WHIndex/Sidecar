#ifndef LIPP_CONTROLLER_H
#define LIPP_CONTROLLER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>

#include "runtime_metadata.h"

namespace lipp_controller {

using LIPP_Node_Header = index_controller::RuntimeMetadata;
using NodeState = index_controller::NodeState;

class LIPPController {
 public:
  using ForegroundTaskCallback = void (*)(void*, bool);

  struct StatisticsSnapshot {
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
  };

  struct RuntimeStatus {
    static constexpr std::uint64_t kHashNodeFlag = 1ULL << 63;
    static constexpr std::uint64_t kMaintenanceFlag = 1ULL << 62;
    static constexpr std::uint64_t kCoolingFlag = 1ULL << 61;
    static constexpr std::uint64_t kCoolingExecutionFlag = 1ULL << 60;
    static constexpr std::uint64_t kPendingTaskMask =
        ~(kHashNodeFlag | kMaintenanceFlag | kCoolingFlag |
          kCoolingExecutionFlag);

    std::uint64_t state{0};

    bool has_pending_tasks() const noexcept {
      return (state & kPendingTaskMask) != 0;
    }

    bool has_hash_nodes() const noexcept {
      return (state & kHashNodeFlag) != 0;
    }

    bool maintenance_active() const noexcept {
      return (state & kMaintenanceFlag) != 0;
    }

    bool cooling_active() const noexcept {
      return (state & kCoolingFlag) != 0;
    }

    bool cooling_execution_active() const noexcept {
      return (state & kCoolingExecutionFlag) != 0;
    }

  };

  struct Config {
    std::size_t min_monitor_size{1};
    std::size_t eager_monitor_size{1};
    std::uint32_t min_monitor_depth{0};
    std::size_t inspection_budget{64};
    std::size_t cooling_batch_size{1};
    std::uint64_t check_interval_ms{50};
    std::uint64_t cooling_interval_ms{5000};

    double conflict_weight{5.0};
    double insert_weight{1.0};
    double read_weight{0.05};

    double rebuild_size_multiplier{2.0};
    double rebuild_size_tolerance{0.05};
    std::size_t rebuild_min_size{64};
    double rebuild_conflict_ratio{0.10};
    double rebuild_conflict_tolerance{0.02};

    std::uint64_t fast_smo_interval_ms{2000};
    std::uint64_t min_post_smo_inserts{1};
    double hash_conflict_ratio{0.30};
    std::uint32_t minimum_hash_depth{4};
    double minimum_depth_ratio{0.50};

    std::uint64_t cold_time_ms{10000};
    std::uint64_t cold_access_threshold{4};
    std::uint64_t hash_horizon_ms{5000};
    double maximum_hash_growth{2.0};
    bool enable_debug_output{false};
  };

  static LIPPController& instance() {
    static LIPPController controller;
    return controller;
  }

  const Config& get_config() const noexcept { return config_; }

  StatisticsSnapshot statistics() const noexcept {
    return StatisticsSnapshot{
        stats_.hash_decisions.load(std::memory_order_relaxed),
        stats_.rebuild_decisions.load(std::memory_order_relaxed),
        stats_.cooling_decisions.load(std::memory_order_relaxed),
        stats_.hash_completions.load(std::memory_order_relaxed),
        stats_.rebuild_completions.load(std::memory_order_relaxed),
        stats_.cooling_completions.load(std::memory_order_relaxed),
        stats_.cooling_bytes_reclaimed.load(std::memory_order_relaxed),
        stats_.hash_failures.load(std::memory_order_relaxed),
        stats_.rebuild_failures.load(std::memory_order_relaxed),
        stats_.inspection_cycles.load(std::memory_order_relaxed),
        stats_.hot_node_inspections.load(std::memory_order_relaxed),
        stats_.cold_node_inspections.load(std::memory_order_relaxed),
        stats_.registration_attempts.load(std::memory_order_relaxed),
        stats_.registered_nodes.load(std::memory_order_relaxed),
        stats_.rejected_by_filter.load(std::memory_order_relaxed),
        live_monitored_count_.load(std::memory_order_relaxed),
        live_hash_node_count_.load(std::memory_order_relaxed)};
  }

  void record_foreground_result(NodeState requested, bool success) noexcept {
    if (requested == NodeState::NEED_HASH) {
      auto& counter = success ? stats_.hash_completions : stats_.hash_failures;
      counter.fetch_add(1, std::memory_order_relaxed);
    } else if (requested == NodeState::NEED_REBUILD) {
      auto& counter =
          success ? stats_.rebuild_completions : stats_.rebuild_failures;
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void record_foreground_work(NodeState requested,
                              std::size_t key_count) noexcept {
    auto* total = requested == NodeState::NEED_HASH
                      ? &stats_.hash_keys_processed
                      : &stats_.rebuild_keys_processed;
    auto* maximum = requested == NodeState::NEED_HASH
                        ? &stats_.hash_max_keys
                        : &stats_.rebuild_max_keys;
    total->fetch_add(key_count, std::memory_order_relaxed);
    auto observed = maximum->load(std::memory_order_relaxed);
    while (observed < key_count &&
           !maximum->compare_exchange_weak(
               observed, key_count, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }

  void record_cooling_savings(std::size_t bytes) noexcept {
    stats_.cooling_bytes_reclaimed.fetch_add(bytes,
                                              std::memory_order_relaxed);
  }

  std::size_t metadata_memory_bytes() const noexcept {
    return live_metadata_count_.load(std::memory_order_relaxed) *
               sizeof(LIPP_Node_Header) +
           registry_.size() * sizeof(RegistrySlot);
  }

  bool has_pending_foreground_tasks() const noexcept {
    return (runtime_state_.load(std::memory_order_relaxed) &
            kPendingTaskMask) != 0;
  }

  std::size_t pending_foreground_tasks() const noexcept {
    return static_cast<std::size_t>(
        runtime_state_.load(std::memory_order_relaxed) & kPendingTaskMask);
  }

  bool has_hash_nodes() const noexcept {
    return (runtime_state_.load(std::memory_order_acquire) &
            kHashNodeFlag) != 0;
  }

  RuntimeStatus runtime_status() const noexcept {
    return RuntimeStatus{runtime_state_.load(std::memory_order_acquire)};
  }

  void begin_maintenance() noexcept {
    std::lock_guard<std::mutex> lock(maintenance_state_mutex_);
    if (maintenance_count_++ == 0) {
      runtime_state_.fetch_or(kMaintenanceFlag, std::memory_order_release);
    }
  }

  void end_maintenance() noexcept {
    std::lock_guard<std::mutex> lock(maintenance_state_mutex_);
    if (maintenance_count_ != 0 && --maintenance_count_ == 0) {
      runtime_state_.fetch_and(~kMaintenanceFlag,
                               std::memory_order_release);
    }
  }

  void notify_hash_node_created() noexcept {
    live_hash_node_count_.fetch_add(1, std::memory_order_acq_rel);
    runtime_state_.fetch_or(kHashNodeFlag, std::memory_order_release);
  }

  void notify_hash_node_deleted() noexcept {
    auto count = live_hash_node_count_.load(std::memory_order_relaxed);
    while (count != 0 &&
           !live_hash_node_count_.compare_exchange_weak(
               count, count - 1, std::memory_order_acq_rel,
               std::memory_order_relaxed)) {
    }
    if (count != 1) {
      return;
    }
    runtime_state_.fetch_and(~kHashNodeFlag, std::memory_order_acq_rel);
    if (live_hash_node_count_.load(std::memory_order_acquire) != 0) {
      runtime_state_.fetch_or(kHashNodeFlag, std::memory_order_release);
    }
  }

  void schedule_foreground_task() noexcept {
    runtime_state_.fetch_add(1, std::memory_order_release);
  }

  void complete_foreground_task() noexcept {
    auto state = runtime_state_.load(std::memory_order_relaxed);
    while ((state & kPendingTaskMask) != 0 &&
           !runtime_state_.compare_exchange_weak(
               state, state - 1, std::memory_order_release,
               std::memory_order_relaxed)) {
    }
  }

  void schedule_cooling_task() noexcept {
    live_cooling_task_count_.fetch_add(1, std::memory_order_acq_rel);
    runtime_state_.fetch_or(kCoolingFlag, std::memory_order_release);
  }

  void complete_cooling_task() noexcept {
    auto count = live_cooling_task_count_.load(std::memory_order_relaxed);
    while (count != 0 &&
           !live_cooling_task_count_.compare_exchange_weak(
               count, count - 1, std::memory_order_acq_rel,
               std::memory_order_relaxed)) {
    }
    if (count != 1) {
      return;
    }
    runtime_state_.fetch_and(~kCoolingFlag, std::memory_order_acq_rel);
    if (live_cooling_task_count_.load(std::memory_order_acquire) != 0) {
      runtime_state_.fetch_or(kCoolingFlag, std::memory_order_release);
    }
  }

  void begin_cooling_execution() noexcept {
    if (active_cooling_count_.fetch_add(1, std::memory_order_acq_rel) == 0) {
      runtime_state_.fetch_or(kCoolingExecutionFlag, std::memory_order_release);
    }
  }

  void end_cooling_execution() noexcept {
    auto count = active_cooling_count_.load(std::memory_order_relaxed);
    while (count != 0 &&
           !active_cooling_count_.compare_exchange_weak(
               count, count - 1, std::memory_order_acq_rel,
               std::memory_order_relaxed)) {
    }
    if (count != 1) {
      return;
    }
    runtime_state_.fetch_and(~kCoolingExecutionFlag, std::memory_order_acq_rel);
    if (active_cooling_count_.load(std::memory_order_acquire) != 0) {
      runtime_state_.fetch_or(kCoolingExecutionFlag,
                              std::memory_order_release);
    }
  }

  void enable_debug_output(bool enabled) noexcept {
    config_.enable_debug_output = enabled;
  }

  LIPP_Node_Header* create_metadata(
      std::size_t size, std::size_t capacity, std::uint32_t depth,
      const LIPP_Node_Header* previous = nullptr,
      void* cooling_context = nullptr,
      LIPP_Node_Header::CoolingCallback cooling_callback = nullptr,
      ForegroundTaskCallback foreground_task_callback = nullptr,
      bool hash_node = false, std::uint32_t depth_before = 0,
      std::uint32_t depth_after = 0, bool register_immediately = true,
      bool record_structural_change = true) {
    auto* metadata = new LIPP_Node_Header();
    metadata->init(size, capacity, depth, hash_node);
    if (previous != nullptr) {
      if (record_structural_change) {
        metadata->inherit_smo_history(previous, depth_before, depth_after);
      } else {
        metadata->copy_smo_history(previous);
      }
    }
    if (cooling_callback != nullptr) {
      auto installed = cooling_callback_.load(std::memory_order_acquire);
      while (installed == nullptr &&
             !cooling_callback_.compare_exchange_weak(
                 installed, cooling_callback, std::memory_order_release,
                 std::memory_order_acquire)) {
      }
    }
    if (foreground_task_callback != nullptr) {
      auto installed =
          foreground_task_callback_.load(std::memory_order_acquire);
      while (installed == nullptr &&
             !foreground_task_callback_.compare_exchange_weak(
                 installed, foreground_task_callback,
                 std::memory_order_release, std::memory_order_acquire)) {
      }
    }
    std::uint32_t handle = 0;
    if (!free_handles_.try_pop(handle)) {
      const auto position = registry_.grow_by(1);
      handle = static_cast<std::uint32_t>(position - registry_.begin() + 1);
    }
    metadata->registry_handle = handle;
    auto& slot = registry_[handle - 1];
    slot.monitored.store(false, std::memory_order_relaxed);
    slot.cooling_context.store(cooling_context, std::memory_order_relaxed);
    slot.owner_context.store(nullptr, std::memory_order_relaxed);
    slot.parent_handle.store(0, std::memory_order_relaxed);
    slot.clear_snapshots();
    slot.metadata.store(metadata, std::memory_order_release);
    live_metadata_count_.fetch_add(1, std::memory_order_relaxed);
    if (register_immediately) {
      register_node(metadata);
    }
    return metadata;
  }

  LIPP_Node_Header* metadata_from_handle(
      std::uint32_t handle) const noexcept {
    if (handle == 0) {
      return nullptr;
    }
    const auto index = static_cast<std::size_t>(handle - 1);
    if (index >= registry_.size()) {
      return nullptr;
    }
    return registry_[index].metadata.load(std::memory_order_acquire);
  }

  void update_context(LIPP_Node_Header* metadata, void* cooling_context,
                      void* owner_context,
                      std::uint32_t parent_handle) noexcept {
    if (metadata == nullptr || metadata->registry_handle == 0) {
      return;
    }
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    const auto index =
        static_cast<std::size_t>(metadata->registry_handle - 1);
    if (index >= registry_.size()) {
      return;
    }
    auto& slot = registry_[index];
    if (slot.metadata.load(std::memory_order_acquire) != metadata) {
      return;
    }
    slot.cooling_context.store(cooling_context, std::memory_order_release);
    slot.owner_context.store(owner_context, std::memory_order_release);
    slot.parent_handle.store(parent_handle, std::memory_order_release);
  }

  void* cooling_context_from_handle(std::uint32_t handle) const noexcept {
    const auto index = static_cast<std::size_t>(handle - 1);
    return handle == 0 || index >= registry_.size()
               ? nullptr
               : registry_[index].cooling_context.load(
                     std::memory_order_acquire);
  }

  void* owner_context_from_handle(std::uint32_t handle) const noexcept {
    const auto index = static_cast<std::size_t>(handle - 1);
    return handle == 0 || index >= registry_.size()
               ? nullptr
               : registry_[index].owner_context.load(
                     std::memory_order_acquire);
  }

  std::uint32_t parent_handle_from_handle(
      std::uint32_t handle) const noexcept {
    const auto index = static_cast<std::size_t>(handle - 1);
    return handle == 0 || index >= registry_.size()
               ? 0
               : registry_[index].parent_handle.load(
                     std::memory_order_acquire);
  }

  void release_metadata(LIPP_Node_Header* metadata) noexcept {
    if (metadata == nullptr) {
      return;
    }
    const auto handle = metadata->registry_handle;
    if (handle == 0) {
      delete metadata;
      return;
    }
    std::unique_lock<std::shared_mutex> lock(registry_mutex_);
    const auto index = static_cast<std::size_t>(handle - 1);
    if (index >= registry_.size() ||
        registry_[index].metadata.load(std::memory_order_acquire) !=
            metadata) {
      return;
    }
    auto& slot = registry_[index];
    slot.metadata.store(nullptr, std::memory_order_release);
    if (slot.monitored.exchange(false, std::memory_order_acq_rel)) {
      live_monitored_count_.fetch_sub(1, std::memory_order_relaxed);
    }
    slot.cooling_context.store(nullptr, std::memory_order_relaxed);
    slot.owner_context.store(nullptr, std::memory_order_relaxed);
    slot.parent_handle.store(0, std::memory_order_relaxed);
    slot.clear_snapshots();
    live_metadata_count_.fetch_sub(1, std::memory_order_relaxed);
    free_handles_.push(handle);
    delete metadata;
  }

  void register_node(LIPP_Node_Header* metadata) {
    if (metadata == nullptr) {
      return;
    }
    stats_.registration_attempts.fetch_add(1, std::memory_order_relaxed);
    if (metadata->current_size.load(std::memory_order_relaxed) <
            config_.min_monitor_size ||
        metadata->layer_depth_value() < config_.min_monitor_depth) {
      stats_.rejected_by_filter.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    const auto handle = metadata->registry_handle;
    if (handle == 0 || handle > registry_.size()) {
      stats_.rejected_by_filter.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    auto& slot = registry_[handle - 1];
    slot.snapshot_inserts =
        metadata->total_inserts.load(std::memory_order_relaxed);
    slot.snapshot_lookups =
        metadata->total_lookups.load(std::memory_order_relaxed);
    slot.snapshot_structural_work =
        metadata->total_structural_work.load(std::memory_order_relaxed);
    bool expected = false;
    if (!metadata->is_monitored.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return;
    }
    slot.monitored.store(true, std::memory_order_release);
    live_monitored_count_.fetch_add(1, std::memory_order_relaxed);
    stats_.registered_nodes.fetch_add(1, std::memory_order_relaxed);
  }

  void unregister_node(LIPP_Node_Header* metadata) {
    if (metadata != nullptr) {
      metadata->mark_deleted();
    }
  }

  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
      return;
    }
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    const auto total = registry_.size();
    for (std::size_t index = 0; index < total; ++index) {
      auto* metadata =
          registry_[index].metadata.load(std::memory_order_acquire);
      if (metadata == nullptr) {
        continue;
      }
      if (metadata->is_monitored.load(std::memory_order_acquire)) {
        registry_[index].snapshot_inserts =
            metadata->total_inserts.load(std::memory_order_relaxed);
        registry_[index].snapshot_lookups =
            metadata->total_lookups.load(std::memory_order_relaxed);
        registry_[index].snapshot_structural_work =
            metadata->total_structural_work.load(std::memory_order_relaxed);
      }
    }
    last_inspection_ns_ = index_controller::steady_now_ns();
    decision_thread_ = std::thread(&LIPPController::decision_loop, this);
    cooling_thread_ = std::thread(&LIPPController::cooling_loop, this);
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
    std::cout << "\n=== LIPP Controller Statistics ===\n"
              << "Hash decisions:        "
              << stats_.hash_decisions.load(std::memory_order_relaxed) << '\n'
              << "Rebuild decisions:     "
              << stats_.rebuild_decisions.load(std::memory_order_relaxed)
              << '\n'
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
              << "Hash completions:      "
              << stats_.hash_completions.load(std::memory_order_relaxed)
              << '\n'
              << "Rebuild completions:   "
              << stats_.rebuild_completions.load(std::memory_order_relaxed)
              << '\n'
              << "Hash failures:         "
              << stats_.hash_failures.load(std::memory_order_relaxed) << '\n'
              << "Rebuild failures:      "
              << stats_.rebuild_failures.load(std::memory_order_relaxed)
              << '\n'
              << "Hash keys processed:   "
              << stats_.hash_keys_processed.load(std::memory_order_relaxed)
              << " (max "
              << stats_.hash_max_keys.load(std::memory_order_relaxed)
              << ")\n"
              << "Rebuild keys processed:"
              << stats_.rebuild_keys_processed.load(std::memory_order_relaxed)
              << " (max "
              << stats_.rebuild_max_keys.load(std::memory_order_relaxed)
              << ")\n"
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
    std::vector<const LIPP_Node_Header*> nodes;
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    nodes.reserve(registry_.size());
    const auto total = registry_.size();
    for (std::size_t index = 0; index < total; ++index) {
      auto* metadata =
          registry_[index].metadata.load(std::memory_order_acquire);
      if (metadata == nullptr) {
        continue;
      }
      if (metadata->is_monitored.load(std::memory_order_acquire) &&
          metadata->is_node_valid()) {
        nodes.push_back(metadata);
      }
    }
    const auto limit = std::min<std::size_t>(10, nodes.size());
    std::partial_sort(
        nodes.begin(), nodes.begin() + limit, nodes.end(),
        [](const LIPP_Node_Header* left, const LIPP_Node_Header* right) {
          return left->access_count() > right->access_count();
        });
    for (std::size_t index = 0; index < limit; ++index) {
      const auto* metadata = nodes[index];
      std::cout << "Node[" << index
                << "] size="
                << metadata->current_size.load(std::memory_order_relaxed)
                << " inserts="
                << metadata->total_inserts.load(std::memory_order_relaxed)
                << " conflicts="
                << metadata->total_structural_work.load(
                       std::memory_order_relaxed)
                << " depth="
                << metadata->layer_depth_value()
                << " hash=" << metadata->is_hash() << '\n';
    }
  }

 private:
  struct RegistrySlot {
    std::atomic<LIPP_Node_Header*> metadata{nullptr};
    std::atomic<bool> monitored{false};
    std::atomic<void*> cooling_context{nullptr};
    std::atomic<void*> owner_context{nullptr};
    std::atomic<std::uint32_t> parent_handle{0};
    std::uint32_t snapshot_inserts{0};
    std::uint32_t snapshot_lookups{0};
    std::uint32_t snapshot_structural_work{0};

    RegistrySlot() = default;
    RegistrySlot(const RegistrySlot&) noexcept {}
    RegistrySlot& operator=(const RegistrySlot&) noexcept {
      metadata.store(nullptr, std::memory_order_relaxed);
      monitored.store(false, std::memory_order_relaxed);
      cooling_context.store(nullptr, std::memory_order_relaxed);
      owner_context.store(nullptr, std::memory_order_relaxed);
      parent_handle.store(0, std::memory_order_relaxed);
      clear_snapshots();
      return *this;
    }

    void clear_snapshots() noexcept {
      snapshot_inserts = 0;
      snapshot_lookups = 0;
      snapshot_structural_work = 0;
    }
  };

  struct Candidate {
    LIPP_Node_Header* metadata;
    std::uint32_t registry_handle;
    void* cooling_context;
    index_controller::RuntimeSample sample;
  };

  struct CoolingTask {
    LIPP_Node_Header* metadata{nullptr};
    void* cooling_context{nullptr};
  };

  struct Stats {
    std::atomic<std::uint64_t> hash_decisions{0};
    std::atomic<std::uint64_t> rebuild_decisions{0};
    std::atomic<std::uint64_t> cooling_decisions{0};
    std::atomic<std::uint64_t> hash_completions{0};
    std::atomic<std::uint64_t> rebuild_completions{0};
    std::atomic<std::uint64_t> cooling_completions{0};
    std::atomic<std::uint64_t> cooling_bytes_reclaimed{0};
    std::atomic<std::uint64_t> hash_failures{0};
    std::atomic<std::uint64_t> rebuild_failures{0};
    std::atomic<std::uint64_t> hash_keys_processed{0};
    std::atomic<std::uint64_t> rebuild_keys_processed{0};
    std::atomic<std::size_t> hash_max_keys{0};
    std::atomic<std::size_t> rebuild_max_keys{0};
    std::atomic<std::uint64_t> inspection_cycles{0};
    std::atomic<std::uint64_t> hot_node_inspections{0};
    std::atomic<std::uint64_t> cold_node_inspections{0};
    std::atomic<std::uint64_t> registration_attempts{0};
    std::atomic<std::uint64_t> registered_nodes{0};
    std::atomic<std::uint64_t> rejected_by_filter{0};
  };

  LIPPController() = default;
  ~LIPPController() {
    stop();
    for (auto& slot : registry_) {
      delete slot.metadata.exchange(nullptr, std::memory_order_acq_rel);
    }
  }
  LIPPController(const LIPPController&) = delete;
  LIPPController& operator=(const LIPPController&) = delete;

  static std::uint64_t milliseconds_to_ns(std::uint64_t milliseconds) {
    return milliseconds * 1000000ULL;
  }

  void decision_loop() {
    while (running_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(
          config_.check_interval_ms));
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
    auto& candidates = candidate_buffer_;
    candidates.clear();
    {
      std::shared_lock<std::shared_mutex> lock(registry_mutex_);
      candidates.reserve(registry_.size());
      const auto total = registry_.size();
      for (std::size_t index = 0; index < total; ++index) {
        auto* metadata =
            registry_[index].metadata.load(std::memory_order_acquire);
        if (metadata == nullptr ||
            !metadata->is_monitored.load(std::memory_order_acquire) ||
            !metadata->is_node_valid()) {
          continue;
        }
        const auto state = metadata->state.load(std::memory_order_acquire);
        auto& slot = registry_[index];
        auto sample = metadata->take_sample(
            config_.conflict_weight, config_.insert_weight,
            config_.read_weight, false, slot.snapshot_inserts,
            slot.snapshot_lookups, slot.snapshot_structural_work);
        if (state == NodeState::NORMAL) {
          candidates.push_back(Candidate{
              metadata, metadata->registry_handle,
              slot.cooling_context.load(std::memory_order_acquire), sample});
        }
      }
    }
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
      inspect_hot_node(candidates[index], period_ns);
    }

    const auto cold_budget = std::min<std::size_t>(
        config_.inspection_budget, candidates.size() - hot_budget);
    stats_.cold_node_inspections.fetch_add(cold_budget,
                                           std::memory_order_relaxed);
    const auto now = index_controller::steady_now_ms();
    for (std::size_t index = 0; index < cold_budget; ++index) {
      inspect_cold_node(candidates[candidates.size() - 1 - index], now);
    }
  }

  LIPP_Node_Header* pin_candidate(const Candidate& candidate) noexcept {
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    const auto handle = candidate.registry_handle;
    if (handle == 0 || handle > registry_.size()) {
      return nullptr;
    }
    auto* metadata = registry_[handle - 1].metadata.load(
        std::memory_order_acquire);
    if (metadata != candidate.metadata || !metadata->try_pin_callback()) {
      return nullptr;
    }
    return metadata;
  }

  void inspect_hot_node(const Candidate& candidate,
                        std::uint64_t period_ns) {
    auto* metadata = pin_candidate(candidate);
    if (metadata == nullptr) {
      return;
    }
    if (!metadata->is_hash() && should_hash(metadata)) {
      const auto insertion_rate =
          period_ns == 0
              ? 0.0
              : static_cast<double>(candidate.sample.inserts) /
                    static_cast<double>(period_ns);
      metadata->estimate_hash_capacity(
          insertion_rate, milliseconds_to_ns(config_.hash_horizon_ms),
          config_.maximum_hash_growth);
      if (metadata->try_schedule(NodeState::NEED_HASH)) {
        publish_foreground_task(candidate.cooling_context, true);
        schedule_foreground_task();
        stats_.hash_decisions.fetch_add(1, std::memory_order_relaxed);
      }
      metadata->unpin_callback();
      return;
    }
    if (!metadata->is_hash() && should_rebuild(metadata, candidate.sample) &&
        metadata->try_schedule(NodeState::NEED_REBUILD)) {
      publish_foreground_task(candidate.cooling_context, true);
      schedule_foreground_task();
      stats_.rebuild_decisions.fetch_add(1, std::memory_order_relaxed);
    }
    metadata->unpin_callback();
  }

  bool should_hash(const LIPP_Node_Header* metadata) const {
    const auto depth = metadata->layer_depth_value();
    if (depth < config_.minimum_hash_depth) {
      return false;
    }
    const auto previous_smo =
        metadata->previous_smo_ms.load(std::memory_order_relaxed);
    const auto last_smo =
        metadata->last_smo_ms.load(std::memory_order_relaxed);
    if (previous_smo == 0 ||
        index_controller::elapsed_ms(last_smo, previous_smo) >
            config_.fast_smo_interval_ms) {
      return false;
    }
    const auto post_smo_inserts =
        metadata->total_inserts.load(std::memory_order_relaxed) -
        metadata->smo_insert_baseline.load(std::memory_order_relaxed);
    if (post_smo_inserts < config_.min_post_smo_inserts) {
      return false;
    }
    const auto post_smo_conflicts =
        metadata->total_structural_work.load(std::memory_order_relaxed) -
        metadata->smo_structural_baseline.load(std::memory_order_relaxed);
    const auto conflict_ratio =
        static_cast<double>(post_smo_conflicts) /
        static_cast<double>(post_smo_inserts);
    const auto depth_before = metadata->depth_before_smo_value();
    const auto depth_after = metadata->depth_after_smo_value();
    if (depth_before == 0 || depth_after == 0) {
      return false;
    }
    const auto depth_ratio = static_cast<double>(depth_after) /
                             static_cast<double>(depth_before);
    return conflict_ratio >= config_.hash_conflict_ratio &&
           depth_ratio >= config_.minimum_depth_ratio;
  }

  bool should_rebuild(
      const LIPP_Node_Header* metadata,
      const index_controller::RuntimeSample& sample) const {
    const auto size = metadata->current_size.load(std::memory_order_relaxed);
    const auto built = metadata->build_size.load(std::memory_order_relaxed);
    if (built == 0 || size < config_.rebuild_min_size) {
      return false;
    }
    const auto inserts = metadata->total_inserts.load(std::memory_order_relaxed);
    const auto build_baseline =
        metadata->smo_insert_baseline.load(std::memory_order_relaxed);
    const auto inserts_since_build = inserts - build_baseline;
    const auto required_growth =
        (config_.rebuild_size_multiplier - 1.0 -
         config_.rebuild_size_tolerance) *
        static_cast<double>(built);
    if (static_cast<double>(inserts_since_build) < required_growth) {
      return false;
    }
    if (sample.inserts == 0) {
      return false;
    }
    const auto conflict_ratio =
        static_cast<double>(sample.conflicts) /
        static_cast<double>(sample.inserts);
    return conflict_ratio >= config_.rebuild_conflict_ratio -
                                 config_.rebuild_conflict_tolerance;
  }

  void inspect_cold_node(const Candidate& candidate, std::uint32_t now) {
    auto* metadata = pin_candidate(candidate);
    if (metadata == nullptr) {
      return;
    }
    const auto last_access =
        metadata->last_access_ms.load(std::memory_order_relaxed);
    if (last_access == 0 ||
        index_controller::elapsed_ms(now, last_access) <
            config_.cold_time_ms ||
        candidate.sample.accesses > config_.cold_access_threshold ||
        metadata->is_compacted_layout() ||
        cooling_callback_.load(std::memory_order_acquire) == nullptr) {
      metadata->unpin_callback();
      return;
    }
    metadata->record_cooling_baseline();
    if (metadata->try_schedule(NodeState::IN_COOLING)) {
      schedule_cooling_task();
      cooling_queue_.push(CoolingTask{metadata, candidate.cooling_context});
      stats_.cooling_decisions.fetch_add(1, std::memory_order_relaxed);
    } else {
      metadata->unpin_callback();
    }
  }

  void drain_cooling_queue() noexcept {
    CoolingTask task;
    while (cooling_queue_.try_pop(task)) {
      if (task.metadata != nullptr) {
        task.metadata->cancel_task();
        task.metadata->unpin_callback();
        complete_cooling_task();
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
      complete_cooling_task();
      return;
    }
    if (!metadata->cooling_baseline_unchanged()) {
      metadata->cancel_task();
      metadata->unpin_callback();
      complete_cooling_task();
      return;
    }
    if (!metadata->try_begin(NodeState::IN_COOLING)) {
      metadata->unpin_callback();
      complete_cooling_task();
      return;
    }
    begin_cooling_execution();
    bool compacted = false;
    const auto callback = cooling_callback_.load(std::memory_order_acquire);
    compacted = callback != nullptr &&
                callback(task.cooling_context, metadata);
    if (metadata->is_node_valid()) {
      metadata->finish_execution(metadata->is_hash());
    }
    end_cooling_execution();
    if (compacted) {
      stats_.cooling_completions.fetch_add(1, std::memory_order_relaxed);
    }
    metadata->unpin_callback();
    complete_cooling_task();
  }

  void publish_foreground_task(void* context, bool pending) noexcept {
    const auto callback =
        foreground_task_callback_.load(std::memory_order_acquire);
    if (callback != nullptr) {
      callback(context, pending);
    }
  }

  Config config_;
  Stats stats_;
  tbb::concurrent_vector<RegistrySlot> registry_;
  tbb::concurrent_queue<std::uint32_t> free_handles_;
  tbb::concurrent_queue<CoolingTask> cooling_queue_;
  mutable std::shared_mutex registry_mutex_;
  static constexpr std::uint64_t kHashNodeFlag = 1ULL << 63;
  static constexpr std::uint64_t kMaintenanceFlag = 1ULL << 62;
  static constexpr std::uint64_t kCoolingFlag = 1ULL << 61;
  static constexpr std::uint64_t kCoolingExecutionFlag = 1ULL << 60;
  static constexpr std::uint64_t kPendingTaskMask =
      ~(kHashNodeFlag | kMaintenanceFlag | kCoolingFlag |
        kCoolingExecutionFlag);
  std::atomic<std::uint64_t> runtime_state_{0};
  std::mutex maintenance_state_mutex_;
  std::size_t maintenance_count_{0};
  std::atomic<LIPP_Node_Header::CoolingCallback> cooling_callback_{nullptr};
  std::atomic<ForegroundTaskCallback> foreground_task_callback_{nullptr};
  std::atomic<std::size_t> live_metadata_count_{0};
  std::atomic<std::size_t> live_monitored_count_{0};
  std::atomic<std::size_t> live_hash_node_count_{0};
  std::atomic<std::size_t> live_cooling_task_count_{0};
  std::atomic<std::size_t> active_cooling_count_{0};
  std::atomic<bool> running_{false};
  std::uint64_t last_inspection_ns_{0};
  std::vector<Candidate> candidate_buffer_;
  std::thread decision_thread_;
  std::thread cooling_thread_;
};

}  // namespace lipp_controller

namespace lippolc {
static auto& g_lipp_controller = lipp_controller::LIPPController::instance();
}

#endif  // LIPP_CONTROLLER_H
