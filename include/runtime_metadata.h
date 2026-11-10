#ifndef INDEX_CONTROLLER_RUNTIME_METADATA_H
#define INDEX_CONTROLLER_RUNTIME_METADATA_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace index_controller {

inline std::uint64_t steady_now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline std::uint32_t steady_now_ms() noexcept {
  return static_cast<std::uint32_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline std::uint32_t elapsed_ms(std::uint32_t now,
                                std::uint32_t then) noexcept {
  return static_cast<std::uint32_t>(now - then);
}

enum class NodeState : std::uint8_t {
  NORMAL = 0,
  NEED_REBUILD,
  NEED_HASH,
  IN_COOLING,
  IS_EXECUTING,
  DELETED,
};

struct RuntimeSample {
  std::uint64_t inserts{0};
  std::uint64_t lookups{0};
  std::uint64_t conflicts{0};
  std::uint64_t shifts{0};
  std::uint64_t accesses{0};
  double priority{0.0};
};

struct alignas(64) RuntimeMetadata {
  using CoolingCallback = bool (*)(void*, RuntimeMetadata*);

  static constexpr std::uint32_t kDepthBits = 10;
  static constexpr std::uint32_t kDepthMask = (1U << kDepthBits) - 1U;
  static constexpr std::uint32_t kBeforeDepthShift = kDepthBits;
  static constexpr std::uint32_t kAfterDepthShift = kDepthBits * 2;
  static constexpr std::uint32_t kCompactedMask = 1U << 31;

  std::atomic<NodeState> state{NodeState::NORMAL};
  std::atomic<bool> is_valid{true};
  std::atomic<bool> is_monitored{false};
  std::atomic<bool> is_hash_node{false};
  std::atomic<std::uint32_t> active_callbacks{0};

  std::atomic<std::uint32_t> total_inserts{0};
  std::atomic<std::uint32_t> total_lookups{0};
  std::atomic<std::uint32_t> total_structural_work{0};

  std::atomic<std::uint32_t> last_access_ms{0};
  std::atomic<std::uint32_t> previous_smo_ms{0};
  std::atomic<std::uint32_t> last_smo_ms{0};
  std::atomic<std::uint32_t> smo_insert_baseline{0};
  std::atomic<std::uint32_t> smo_structural_baseline{0};

  std::atomic<std::uint32_t> current_size{0};
  std::atomic<std::uint32_t> current_capacity{0};
  std::atomic<std::uint32_t> build_size{0};
  std::atomic<std::uint32_t> reserved_hash_capacity{0};
  std::atomic<std::uint32_t> packed_depths{0};
  std::uint32_t registry_handle{0};

  static std::uint32_t narrow_size(std::size_t value) noexcept {
    constexpr auto limit =
        static_cast<std::size_t>(UINT32_MAX);
    return static_cast<std::uint32_t>(std::min(value, limit));
  }

  static std::uint32_t narrow_depth(std::uint32_t value) noexcept {
    return std::min(value, kDepthMask);
  }

  std::uint32_t layer_depth_value() const noexcept {
    return packed_depths.load(std::memory_order_relaxed) & kDepthMask;
  }

  std::uint32_t depth_before_smo_value() const noexcept {
    return (packed_depths.load(std::memory_order_relaxed) >>
            kBeforeDepthShift) & kDepthMask;
  }

  std::uint32_t depth_after_smo_value() const noexcept {
    return (packed_depths.load(std::memory_order_relaxed) >>
            kAfterDepthShift) & kDepthMask;
  }

  bool is_compacted_layout() const noexcept {
    return (packed_depths.load(std::memory_order_relaxed) &
            kCompactedMask) != 0;
  }

  void set_compacted_layout(bool compacted) noexcept {
    if (compacted) {
      packed_depths.fetch_or(kCompactedMask, std::memory_order_relaxed);
    } else {
      packed_depths.fetch_and(~kCompactedMask, std::memory_order_relaxed);
    }
  }

  void set_layer_depth(std::uint32_t depth) noexcept {
    auto current = packed_depths.load(std::memory_order_relaxed);
    const auto value = narrow_depth(depth);
    do {
      const auto desired = (current & ~kDepthMask) | value;
      if (packed_depths.compare_exchange_weak(
              current, desired, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return;
      }
    } while (true);
  }

  void set_smo_depths(std::uint32_t before, std::uint32_t after) noexcept {
    auto current = packed_depths.load(std::memory_order_relaxed);
    const auto values =
        (narrow_depth(before) << kBeforeDepthShift) |
        (narrow_depth(after) << kAfterDepthShift);
    constexpr auto smo_mask =
        (kDepthMask << kBeforeDepthShift) |
        (kDepthMask << kAfterDepthShift);
    do {
      const auto desired = (current & ~smo_mask) | values;
      if (packed_depths.compare_exchange_weak(
              current, desired, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return;
      }
    } while (true);
  }

  void init(std::size_t size, std::size_t capacity,
            std::uint32_t depth = 0, bool hash_node = false) noexcept {
    state.store(NodeState::NORMAL, std::memory_order_relaxed);
    is_valid.store(true, std::memory_order_relaxed);
    is_monitored.store(false, std::memory_order_relaxed);
    is_hash_node.store(hash_node, std::memory_order_relaxed);

    total_inserts.store(0, std::memory_order_relaxed);
    total_lookups.store(0, std::memory_order_relaxed);
    total_structural_work.store(0, std::memory_order_relaxed);

    const auto now = steady_now_ms();
    last_access_ms.store(now, std::memory_order_relaxed);
    previous_smo_ms.store(0, std::memory_order_relaxed);
    last_smo_ms.store(0, std::memory_order_relaxed);
    smo_insert_baseline.store(0, std::memory_order_relaxed);
    smo_structural_baseline.store(0, std::memory_order_relaxed);

    current_size.store(narrow_size(size), std::memory_order_relaxed);
    current_capacity.store(narrow_size(capacity), std::memory_order_relaxed);
    build_size.store(narrow_size(size), std::memory_order_relaxed);
    reserved_hash_capacity.store(0, std::memory_order_relaxed);
    packed_depths.store(narrow_depth(depth), std::memory_order_relaxed);
    active_callbacks.store(0, std::memory_order_relaxed);
    registry_handle = 0;
  }

  void inherit_smo_history(const RuntimeMetadata* previous,
                           std::uint32_t depth_before = 0,
                           std::uint32_t depth_after = 0) noexcept {
    const auto now = steady_now_ms();
    const auto previous_smo = previous
                                  ? previous->last_smo_ms.load(
                                        std::memory_order_acquire)
                                  : 0;
    previous_smo_ms.store(previous_smo, std::memory_order_relaxed);
    last_smo_ms.store(now, std::memory_order_release);
    set_smo_depths(depth_before, depth_after);
    reset_smo_baselines();
  }

  void copy_smo_history(const RuntimeMetadata* previous) noexcept {
    if (previous == nullptr) {
      return;
    }
    previous_smo_ms.store(
        previous->previous_smo_ms.load(std::memory_order_acquire),
        std::memory_order_relaxed);
    last_smo_ms.store(previous->last_smo_ms.load(std::memory_order_acquire),
                      std::memory_order_relaxed);
    set_smo_depths(previous->depth_before_smo_value(),
                   previous->depth_after_smo_value());
    reset_smo_baselines();
  }

  void record_smo(std::uint32_t depth_before = 0,
                  std::uint32_t depth_after = 0) noexcept {
    const auto now = steady_now_ms();
    const auto old = last_smo_ms.exchange(now, std::memory_order_acq_rel);
    previous_smo_ms.store(old, std::memory_order_release);
    set_smo_depths(depth_before, depth_after);
    build_size.store(current_size.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    reset_smo_baselines();
  }

  void reset_smo_baselines() noexcept {
    smo_insert_baseline.store(total_inserts.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
    smo_structural_baseline.store(
        total_structural_work.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
  }

  void record_lookup_at(std::size_t size, std::uint32_t access_time_ms,
                        std::uint64_t weight = 1) noexcept {
    total_lookups.fetch_add(static_cast<std::uint32_t>(weight),
                            std::memory_order_relaxed);
    if (size != 0) {
      current_size.store(narrow_size(size), std::memory_order_relaxed);
    }
    if (last_access_ms.load(std::memory_order_relaxed) != access_time_ms) {
      last_access_ms.store(access_time_ms, std::memory_order_relaxed);
    }
  }

  void record_lookup(std::size_t size = 0,
                     std::uint64_t weight = 1) noexcept {
    record_lookup_at(size, steady_now_ms(), weight);
  }

  void record_insert(std::uint64_t shifts, bool conflict,
                     std::size_t size, std::size_t capacity,
                     std::uint64_t weight = 1) noexcept {
    total_inserts.fetch_add(static_cast<std::uint32_t>(weight),
                            std::memory_order_relaxed);
    if (conflict) {
      total_structural_work.fetch_add(static_cast<std::uint32_t>(weight),
                                      std::memory_order_relaxed);
    }
    if (shifts != 0) {
      total_structural_work.fetch_add(static_cast<std::uint32_t>(shifts),
                                      std::memory_order_relaxed);
    }
    current_size.store(narrow_size(size), std::memory_order_relaxed);
    current_capacity.store(narrow_size(capacity), std::memory_order_relaxed);
    if (is_compacted_layout()) {
      const auto built = build_size.load(std::memory_order_relaxed);
      if (built != 0 && size >= static_cast<std::size_t>(built) +
                                    std::max<std::size_t>(4, built / 20)) {
        set_compacted_layout(false);
      }
    }
    const auto access_time = steady_now_ms();
    if (last_access_ms.load(std::memory_order_relaxed) != access_time) {
      last_access_ms.store(access_time, std::memory_order_relaxed);
    }
  }

  void record_insert_access(std::size_t size,
                            std::uint32_t access_time_ms) noexcept {
    total_inserts.fetch_add(1, std::memory_order_relaxed);
    current_size.store(narrow_size(size), std::memory_order_relaxed);
    if (is_compacted_layout()) {
      const auto built = build_size.load(std::memory_order_relaxed);
      if (built != 0 && size >= static_cast<std::size_t>(built) +
                                    std::max<std::size_t>(4, built / 20)) {
        set_compacted_layout(false);
      }
    }
    if (last_access_ms.load(std::memory_order_relaxed) != access_time_ms) {
      last_access_ms.store(access_time_ms, std::memory_order_relaxed);
    }
  }

  void record_conflict() noexcept {
    total_structural_work.fetch_add(1, std::memory_order_relaxed);
  }

  static std::uint32_t exchange_snapshot(
      std::atomic<std::uint32_t>& snapshot,
      std::uint32_t current) noexcept {
    return snapshot.exchange(current, std::memory_order_relaxed);
  }

  static std::uint32_t exchange_snapshot(
      std::uint32_t& snapshot, std::uint32_t current) noexcept {
    const auto previous = snapshot;
    snapshot = current;
    return previous;
  }

  template <class Snapshot>
  RuntimeSample take_sample(double conflict_weight, double insert_weight,
                            double read_weight,
                            bool shifts_are_conflicts,
                            Snapshot& snapshot_inserts,
                            Snapshot& snapshot_lookups,
                            Snapshot& snapshot_structural_work) noexcept {
    RuntimeSample sample;
    const auto inserts = total_inserts.load(std::memory_order_relaxed);
    const auto lookups = total_lookups.load(std::memory_order_relaxed);
    const auto structural_work =
        total_structural_work.load(std::memory_order_relaxed);
    const auto current_inserts = static_cast<std::uint32_t>(inserts);
    const auto current_lookups = static_cast<std::uint32_t>(lookups);
    const auto current_structural =
        static_cast<std::uint32_t>(structural_work);

    sample.inserts = static_cast<std::uint32_t>(
        current_inserts -
        exchange_snapshot(snapshot_inserts, current_inserts));
    sample.lookups = static_cast<std::uint32_t>(
        current_lookups -
        exchange_snapshot(snapshot_lookups, current_lookups));
    const auto structural_delta = static_cast<std::uint32_t>(
        current_structural - exchange_snapshot(snapshot_structural_work,
                                               current_structural));
    if (shifts_are_conflicts) {
      sample.shifts = structural_delta;
    } else {
      sample.conflicts = structural_delta;
    }
    sample.accesses = sample.inserts + sample.lookups;
    sample.priority = conflict_weight * static_cast<double>(structural_delta) +
                      insert_weight * static_cast<double>(sample.inserts) +
                      read_weight * static_cast<double>(sample.lookups);
    return sample;
  }

  std::uint64_t access_count() const noexcept {
    return total_inserts.load(std::memory_order_relaxed) +
           total_lookups.load(std::memory_order_relaxed);
  }

  void record_cooling_baseline() noexcept {
    reserved_hash_capacity.store(
        static_cast<std::uint32_t>(access_count()),
        std::memory_order_relaxed);
  }

  bool cooling_baseline_unchanged() const noexcept {
    return static_cast<std::uint32_t>(access_count()) ==
           reserved_hash_capacity.load(std::memory_order_relaxed);
  }

  bool try_schedule(NodeState requested) noexcept {
    NodeState expected = NodeState::NORMAL;
    return is_valid.load(std::memory_order_acquire) &&
           state.compare_exchange_strong(expected, requested,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed);
  }

  bool try_begin(NodeState requested) noexcept {
    return state.compare_exchange_strong(requested, NodeState::IS_EXECUTING,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed);
  }

  void finish_execution(bool hash_node = false) noexcept {
    if (hash_node) {
      is_hash_node.store(true, std::memory_order_release);
    }
    if (is_valid.load(std::memory_order_acquire)) {
      state.store(NodeState::NORMAL, std::memory_order_release);
    }
  }

  void cancel_task() noexcept {
    NodeState current = state.load(std::memory_order_relaxed);
    while (current != NodeState::DELETED && current != NodeState::NORMAL &&
           !state.compare_exchange_weak(current, NodeState::NORMAL,
                                        std::memory_order_release,
                                        std::memory_order_relaxed)) {
    }
  }

  NodeState mark_deleted() noexcept {
    is_valid.store(false, std::memory_order_release);
    is_monitored.store(false, std::memory_order_release);
    return state.exchange(NodeState::DELETED, std::memory_order_acq_rel);
  }

  void wait_for_callbacks() const noexcept {
    while (active_callbacks.load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }
  }

  bool try_pin_callback() noexcept {
    if (!is_node_valid()) {
      return false;
    }
    active_callbacks.fetch_add(1, std::memory_order_acq_rel);
    if (!is_node_valid()) {
      active_callbacks.fetch_sub(1, std::memory_order_release);
      return false;
    }
    return true;
  }

  void unpin_callback() noexcept {
    active_callbacks.fetch_sub(1, std::memory_order_release);
  }

  bool is_node_valid() const noexcept {
    return is_valid.load(std::memory_order_acquire) &&
           state.load(std::memory_order_acquire) != NodeState::DELETED;
  }

  bool is_hash() const noexcept {
    return is_hash_node.load(std::memory_order_acquire);
  }

  bool has_task() const noexcept {
    if (!is_valid.load(std::memory_order_acquire)) {
      return false;
    }
    const auto current = state.load(std::memory_order_acquire);
    return current == NodeState::NEED_REBUILD ||
           current == NodeState::NEED_HASH;
  }

  bool needs_rebuild() const noexcept {
    return state.load(std::memory_order_acquire) == NodeState::NEED_REBUILD;
  }

  bool needs_hash() const noexcept {
    return state.load(std::memory_order_acquire) == NodeState::NEED_HASH;
  }

  bool needs_compress() const noexcept {
    return state.load(std::memory_order_acquire) == NodeState::IN_COOLING;
  }

  bool cancel_cooling_on_access() noexcept {
    NodeState expected = NodeState::IN_COOLING;
    return state.compare_exchange_strong(expected, NodeState::NORMAL,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed);
  }

  void clear_needs_hash() noexcept { cancel_task(); }
  void clear_needs_compress() noexcept { cancel_task(); }
  void clear_flags() noexcept { cancel_task(); }

  std::size_t estimate_hash_capacity(double insertion_rate_per_ns,
                                     std::uint64_t horizon_ns,
                                     double maximum_growth) noexcept {
    const auto size = static_cast<std::size_t>(
        current_size.load(std::memory_order_relaxed));
    const auto predicted = static_cast<std::size_t>(
        std::ceil(insertion_rate_per_ns * static_cast<double>(horizon_ns)));
    const auto hard_limit = static_cast<std::size_t>(
        static_cast<double>(std::max<std::size_t>(size, 1)) *
        maximum_growth);
    const auto capacity = std::min(size + predicted, hard_limit);
    reserved_hash_capacity.store(narrow_size(std::max(capacity, size)),
                                 std::memory_order_release);
    return std::max(capacity, size);
  }
};

static_assert(alignof(RuntimeMetadata) == 64,
              "runtime metadata must remain cache-line aligned");
static_assert(sizeof(RuntimeMetadata) == 64,
              "runtime metadata layout changed unexpectedly");

}  // namespace index_controller

#endif  // INDEX_CONTROLLER_RUNTIME_METADATA_H
