#ifndef __LIPPOL_H__
#define __LIPPOL_H__

#include "concurrency.h"
#include "lipp_base.h"
#include "omp.h"
#include "tbb/combinable.h"
#include "tbb/enumerable_thread_specific.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <math.h>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <stdint.h>
#include <thread>
#include <vector>
#include <tbb/concurrent_queue.h>
#include <unordered_set>
#if ENABLE_CONTROLLER
#include "lipp_controller.h"
#include "phmap/phmap.h"

#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#endif

namespace lippolc {

// runtime assert
#define RT_ASSERT(expr)                                                        \
  {                                                                            \
    if (!(expr)) {                                                             \
      fprintf(stderr, "Thread %d: RT_ASSERT Error at %s:%d, `%s` not hold!\n", \
              omp_get_thread_num(), __FILE__, __LINE__, #expr);                \
      exit(0);                                                                 \
    }                                                                          \
  }

typedef void (*dealloc_func)(void *ptr);

// runtime debug
#define LIPP_PRINT_DEBUG 0

#if LIPP_PRINT_DEBUG
#define RESET "\033[0m"
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[37m"   /* White */

#define RT_DEBUG(msg, ...)                                                     \
  if (omp_get_thread_num() == 0) {                                             \
    printf(GREEN "T%d: " msg RESET "\n", omp_get_thread_num(), __VA_ARGS__);   \
  } else if (omp_get_thread_num() == 1) {                                      \
    printf(YELLOW "\t\t\tT%d: " msg RESET "\n", omp_get_thread_num(),          \
           __VA_ARGS__);                                                       \
  } else {                                                                     \
    printf(BLUE "\t\t\t\t\t\tT%d: " msg RESET "\n", omp_get_thread_num(),      \
           __VA_ARGS__);                                                       \
  }
#else
#define RT_DEBUG(msg, ...)
#endif

#define COLLECT_TIME 0

#if COLLECT_TIME
#include <chrono>
#endif
#if ENABLE_CONTROLLER
template <class T, class P>
class HashNodeWrapper {
    using MapType = phmap::parallel_flat_hash_map<
        T, P, std::hash<T>, std::equal_to<T>,
        std::allocator<std::pair<const T, P>>,
        4, std::mutex>;

public:
    bool insert(const T& k, const P& v) {
        bool inserted = false;
        map.lazy_emplace_l(
            k,
            [&](auto&) {},
            [&](auto f) {
                f(k, v);
                inserted = true;
            });
        return inserted;
    }

    void upsert(const T& k, const P& v) {
        map.lazy_emplace_l(
            k,
            [&](auto& val) { val.second = v; },
            [&](auto f)    { f(k, v); }
        );
    }

    bool find(const T& k, P& v) {
        return map.if_contains(k, [&](const auto& p) { v = p.second; });
    }

    bool update(const T& k, const P& v) {
        return map.if_contains(k, [&](auto& p) {
            p.second = v;
            return true;
        });
    }

    bool remove(const T& k) {
        return map.erase(k) > 0;
    }

    bool contains(const T& k) const {
        return map.contains(k);
    }

    void reserve(size_t n) {
        map.reserve(n);
    }

    size_t size() const {
        return map.size();
    }

    size_t bucket_count() const {
        return map.bucket_count();
    }

    bool empty() const {
        return map.empty();
    }

    void get_all_items(std::vector<T>& keys, std::vector<P>& values) const {
        map.for_each([&](const auto& pair) {
            keys.push_back(pair.first);
            values.push_back(pair.second);
        });
    }

    void batch_insert(const std::vector<T>& keys, const std::vector<P>& values) {
        assert(keys.size() == values.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            upsert(keys[i], values[i]);
        }
    }

    std::size_t compress() {
        const auto old_bucket_count = map.bucket_count();
        map.max_load_factor(0.95f);
        map.rehash(0);
        const auto new_bucket_count = map.bucket_count();
        return old_bucket_count > new_bucket_count
                   ? old_bucket_count - new_bucket_count
                   : 0;
    }

    void clear() {
        map.clear();
    }

private:
    MapType map;
};
#endif

template <class T, class P, bool USE_FMCD = false> class LIPP {
  static_assert(std::is_arithmetic<T>::value, "LIPP key type must be numeric.");

  struct Node;
  static constexpr int kInlinePathCapacity = 32;

  class NodePath {
    Node* inline_nodes_[kInlinePathCapacity];
    std::unique_ptr<std::vector<Node*>> overflow_;
    int size_{0};

   public:
    void push(Node* node) {
      if (size_ < kInlinePathCapacity) {
        inline_nodes_[size_] = node;
      } else {
        if (overflow_ == nullptr) {
          overflow_ = std::make_unique<std::vector<Node*>>();
        }
        overflow_->push_back(node);
      }
      ++size_;
    }

    Node* operator[](int index) const noexcept {
      return index < kInlinePathCapacity
                 ? inline_nodes_[index]
                 : (*overflow_)[static_cast<std::size_t>(
                       index - kInlinePathCapacity)];
    }

    int size() const noexcept { return size_; }
  };

  inline int compute_gap_count(int size) {
    if (size >= 1000000)
      return 1;
    if (size >= 100000)
      return 2;
    return 5;
  }

  inline int PREDICT_POS(Node *node, T key) const {
    double v = node->model.predict_double(key);
    if (v > std::numeric_limits<int>::max() / 2) {
      return node->num_items - 1;
    }
    if (v < 0) {
      return 0;
    }
    return std::min(node->num_items - 1, static_cast<int>(v));
  }


  const double BUILD_LR_REMAIN;
  const bool QUIET;

  struct {
    long long fmcd_success_times = 0;
    long long fmcd_broken_times = 0;
#if COLLECT_TIME
    double time_scan_and_destory_tree = 0;
    double time_build_tree_bulk = 0;
#endif
  } stats;

public:
  class FastConcurrentStack {
    tbb::concurrent_bounded_queue<Node*> queue_;
  public:
    FastConcurrentStack() { queue_.set_capacity(100000); }
    void push(Node* n) { queue_.try_push(n); }
    Node* pop() { Node* n = nullptr; queue_.try_pop(n); return n; }
    [[nodiscard]] bool empty() const { return queue_.empty(); }
    [[nodiscard]] size_t size() const { return queue_.size(); }

  };
  // Epoch based Memory Reclaim
  class ThreadSpecificEpochBasedReclamationInformation {

    std::array<std::vector<Node *>, 3> mFreeLists;
    std::atomic<uint32_t> mLocalEpoch;
    uint32_t mPreviouslyAccessedEpoch;
    bool mThreadWantsToAdvance;
    LIPP<T, P> *tree;

  public:
    ThreadSpecificEpochBasedReclamationInformation(LIPP<T, P> *index)
        : mFreeLists(), mLocalEpoch(3), mPreviouslyAccessedEpoch(3),
          mThreadWantsToAdvance(false), tree(index) {}

    ThreadSpecificEpochBasedReclamationInformation(
        ThreadSpecificEpochBasedReclamationInformation const &other) = delete;

    ThreadSpecificEpochBasedReclamationInformation(
        ThreadSpecificEpochBasedReclamationInformation &&other) = delete;

    ~ThreadSpecificEpochBasedReclamationInformation() {
      for (uint32_t i = 0; i < 3; ++i) {
        freeForEpoch(i);
      }
    }

    void scheduleForDeletion(Node *pointer) {
      assert(mLocalEpoch != 3);
      std::vector<Node *> &currentFreeList =
          mFreeLists[mLocalEpoch];
      currentFreeList.emplace_back(pointer);
      mThreadWantsToAdvance = (currentFreeList.size() % 64u) == 0;
    }

    uint32_t getLocalEpoch() const {
      return mLocalEpoch.load(std::memory_order_acquire);
    }

    void enter(uint32_t newEpoch) {
      assert(mLocalEpoch == 3);
      if (mPreviouslyAccessedEpoch != newEpoch) {
        freeForEpoch(newEpoch);
        mThreadWantsToAdvance = false;
        mPreviouslyAccessedEpoch = newEpoch;
      }
      mLocalEpoch.store(newEpoch, std::memory_order_release);
    }

    void leave() { mLocalEpoch.store(3, std::memory_order_release); }

    bool doesThreadWantToAdvanceEpoch() { return (mThreadWantsToAdvance); }

    void reclaim_all() {
      RT_ASSERT(mLocalEpoch.load(std::memory_order_acquire) == 3);
      for (std::uint32_t epoch = 0; epoch < 3; ++epoch) {
        freeForEpoch(epoch);
      }
      mThreadWantsToAdvance = false;
    }

  private:
    void freeForEpoch(uint32_t epoch) {
      std::vector<Node *> &previousFreeList =
          mFreeLists[epoch];
      for (Node *node : previousFreeList) {
#if ENABLE_CONTROLLER
        if (tree->node_is_hash(node)) {
          tree->delete_nodes(node, 1);
          continue;
        }
#endif
        if (node->is_two) {
          node->size = 2;
          node->num_inserts = node->num_insert_to_data = 0;
          for(int i = 0; i < node->num_items; i++) node->items[i].typeVersionLockObsolete.store(0b100);
          for(int i = 0; i < node->num_items; i++) node->items[i].entry_type = 0;
          tree->delete_items(node->items, node->num_items);
          tree->delete_nodes(node, 1);
        } else {
          tree->delete_items(node->items, node->num_items);
          tree->delete_nodes(node, 1);
        }
      }
      previousFreeList.resize(0u);
    }
  };

  class EpochBasedMemoryReclamationStrategy {
  public:
    uint32_t NEXT_EPOCH[3] = {1, 2, 0};
    uint32_t PREVIOUS_EPOCH[3] = {2, 0, 1};

    std::atomic<uint32_t> mCurrentEpoch;
    tbb::enumerable_thread_specific<
        ThreadSpecificEpochBasedReclamationInformation,
        tbb::cache_aligned_allocator<
            ThreadSpecificEpochBasedReclamationInformation>,
        tbb::ets_key_per_instance>
        mThreadSpecificInformations;

  private:
    EpochBasedMemoryReclamationStrategy(LIPP<T, P> *index)
        : mCurrentEpoch(0), mThreadSpecificInformations(index) {}

  public:
    static EpochBasedMemoryReclamationStrategy *getInstance(LIPP<T, P> *index) {
      static EpochBasedMemoryReclamationStrategy instance(index);
      return &instance;
    }

    ThreadSpecificEpochBasedReclamationInformation* enterCriticalSection() {
      ThreadSpecificEpochBasedReclamationInformation &currentMemoryInformation =
          mThreadSpecificInformations.local();
      uint32_t currentEpoch = mCurrentEpoch.load(std::memory_order_acquire);
      currentMemoryInformation.enter(currentEpoch);
      if (currentMemoryInformation.doesThreadWantToAdvanceEpoch() &&
          canAdvance(currentEpoch)) {
        mCurrentEpoch.compare_exchange_strong(currentEpoch,
                                              NEXT_EPOCH[currentEpoch]);
      }
      return &currentMemoryInformation;
    }

    bool canAdvance(uint32_t currentEpoch) {
      uint32_t previousEpoch = PREVIOUS_EPOCH[currentEpoch];
      return !std::any_of(
          mThreadSpecificInformations.begin(),
          mThreadSpecificInformations.end(),
          [previousEpoch](ThreadSpecificEpochBasedReclamationInformation const
                              &threadInformation) {
            return (threadInformation.getLocalEpoch() == previousEpoch);
          });
    }

    void leaveCriticalSection(
        ThreadSpecificEpochBasedReclamationInformation* information) {
      information->leave();
    }

    void scheduleForDeletion(Node *pointer) {
      mThreadSpecificInformations.local().scheduleForDeletion(pointer);
    }

    void reclaim_all() {
      for (auto& information : mThreadSpecificInformations) {
        information.reclaim_all();
      }
    }

#if ENABLE_CONTROLLER
    void wait_for_other_threads() {
      auto* current = &mThreadSpecificInformations.local();
      for (;;) {
        bool quiescent = true;
        for (auto& information : mThreadSpecificInformations) {
          if (&information != current && information.getLocalEpoch() != 3) {
            quiescent = false;
            break;
          }
        }
        if (quiescent) {
          return;
        }
        std::this_thread::yield();
      }
    }
#endif
  };

  class EpochGuard {
    EpochBasedMemoryReclamationStrategy *instance;
    ThreadSpecificEpochBasedReclamationInformation* local_epoch_;
#if ENABLE_CONTROLLER
    lipp_controller::LIPPController::RuntimeStatus runtime_status_;

    [[gnu::noinline]] void wait_for_maintenance() {
      do {
        instance->leaveCriticalSection(local_epoch_);
        do {
          std::this_thread::yield();
          runtime_status_ = g_lipp_controller.runtime_status();
        } while (runtime_status_.maintenance_active());
        local_epoch_ = instance->enterCriticalSection();
        runtime_status_ = g_lipp_controller.runtime_status();
      } while (runtime_status_.maintenance_active());
    }
#endif

  public:
    [[gnu::always_inline]] explicit EpochGuard(LIPP<T, P> *index) {
      instance = EpochBasedMemoryReclamationStrategy::getInstance(index);
      local_epoch_ = instance->enterCriticalSection();
#if ENABLE_CONTROLLER
      runtime_status_ = g_lipp_controller.runtime_status();
      if (UNLIKELY(runtime_status_.maintenance_active())) {
        wait_for_maintenance();
      }
#endif
    }

    [[gnu::always_inline]] ~EpochGuard() {
      instance->leaveCriticalSection(local_epoch_);
    }

#if ENABLE_CONTROLLER
    const lipp_controller::LIPPController::RuntimeStatus& runtime_status()
        const noexcept {
      return runtime_status_;
    }
#endif
  };

#if ENABLE_CONTROLLER
  class MaintenanceBarrierGuard {
    LIPP<T, P>* tree;

   public:
    explicit MaintenanceBarrierGuard(LIPP<T, P>* index) : tree(index) {
      g_lipp_controller.begin_maintenance();
      tree->ebr->wait_for_other_threads();
    }

    ~MaintenanceBarrierGuard() {
      g_lipp_controller.end_maintenance();
    }

    MaintenanceBarrierGuard(const MaintenanceBarrierGuard&) = delete;
    MaintenanceBarrierGuard& operator=(const MaintenanceBarrierGuard&) =
        delete;
  };
#endif

  EpochBasedMemoryReclamationStrategy *ebr;

  typedef std::pair<T, P> V;

  LIPP(double BUILD_LR_REMAIN = 0, bool QUIET = true)
      : BUILD_LR_REMAIN(BUILD_LR_REMAIN), QUIET(QUIET) {
    {
      std::vector<Node *> nodes;
      constexpr int initial_pool_nodes = 4096;
      nodes.reserve(initial_pool_nodes);
      for (int _ = 0; _ < initial_pool_nodes; _++) {
        Node *node = build_tree_two(T(0), P(), T(1), P());
        nodes.push_back(node);
      }
      for (auto node : nodes) {
        destroy_tree(node);
      }
      if (!QUIET) {
        printf("initial memory pool size = %lu\n",
               pending_two[omp_get_thread_num()].size());
      }
    }
    if (USE_FMCD && !QUIET) {
      printf("enable FMCD\n");
    }

    root.store(build_tree_none(), std::memory_order_release);
    ebr = EpochBasedMemoryReclamationStrategy::getInstance(this);
#if ENABLE_CONTROLLER
    g_lipp_controller.start();
#endif
  }

  ~LIPP() {
#if ENABLE_CONTROLLER
    g_lipp_controller.stop();
#endif
    ebr->reclaim_all();
    destroy_tree(root.exchange(nullptr, std::memory_order_acq_rel));
    destory_pending();
  }

  void insert(const V &v) { insert(v.first, v.second); }
  void insert(const T &key, const P &value) {
    EpochGuard guard(this);
#if ENABLE_CONTROLLER
    bool state = insert_tree(key, value, guard.runtime_status());
#else
    bool state = insert_tree(key, value);
#endif
    RT_DEBUG("Insert_tree(%d): success/fail? %d", key, state);
  }

  [[gnu::always_inline]] bool at(const T &key, P &value) {
    EpochGuard guard(this);
#if ENABLE_CONTROLLER
    const auto controller_status = guard.runtime_status();
    return at_with_runtime(key, value, controller_status);
#else
    int restartCount = 0;
  restart:
    if (restartCount++) {
      yield(restartCount);
    }
    bool needRestart = false;

    // for lock coupling
    uint64_t versionItem;
    Node *parent = nullptr;
    int parent_pos = -1;
    int current_depth = 0;

    for (Node *node = root.load(std::memory_order_acquire);;) {
      int pos = PREDICT_POS(node, key);
      versionItem = node->items[pos].readLockOrRestart(needRestart);
      if (needRestart)
        goto restart;

      if (node->items[pos].entry_type == 1) { // 1 means child
        parent = node;
        parent_pos = pos;
        node = node->items[pos].comp.child;

        parent->items[pos].readUnlockOrRestart(versionItem, needRestart);
        if (needRestart)
          goto restart;
      } else { // the entry is a data or empty
        if (node->items[pos].entry_type == 0) { // 0 means empty
          return false;
        } else { // 2 means data
          node->items[pos].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart)
            goto restart;

          value = node->items[pos].comp.data.value;
          return true;
        }
      }
      ++current_depth;
    }
#endif
  }

  void reclaim_retired_storage() { ebr->reclaim_all(); }

  bool exists(const T &key) const {
    Node *node = root.load(std::memory_order_acquire);
    while (true) {
#if ENABLE_CONTROLLER
      if (node_is_hash(node)) {
        auto* hash_map = node_hash_map(node);
        return hash_map ? hash_map->contains(key) : false;
      }
#endif
      int pos = PREDICT_POS(node, key);
      if (node->items[pos].entry_type == 0) {
        return false;
      } else if (node->items[pos].entry_type == 2) {
        return node->items[pos].comp.data.key == key;
      } else {
        node = node->items[pos].comp.child;
      }
    }
  }

  void yield(int count) {
    if (count > 3)
      sched_yield();
    else
      _mm_pause();
  }

  void bulk_load(const V *vs, int num_keys) {
    if (num_keys == 0) {
      destroy_tree(root.exchange(build_tree_none(),
                                 std::memory_order_acq_rel));
      return;
    }
    if (num_keys == 1) {
      destroy_tree(root.exchange(build_tree_none(),
                                 std::memory_order_acq_rel));
      insert(vs[0]);
      return;
    }
    if (num_keys == 2) {
      auto* replacement =
          build_tree_two(vs[0].first, vs[0].second, vs[1].first, vs[1].second);
      destroy_tree(root.exchange(replacement, std::memory_order_acq_rel));
#if ENABLE_CONTROLLER
      register_subtree_runtime(replacement, nullptr, -1, 0);
#endif
      return;
    }

    RT_ASSERT(num_keys > 2);
    for (int i = 1; i < num_keys; i++) {
      RT_ASSERT(vs[i].first > vs[i - 1].first);
    }

    T *keys = new T[num_keys];
    P *values = new P[num_keys];
    for (int i = 0; i < num_keys; i++) {
      keys[i] = vs[i].first;
      values[i] = vs[i].second;
    }
    auto* replacement = build_tree_bulk(keys, values, num_keys);
    destroy_tree(root.exchange(replacement, std::memory_order_acq_rel));
#if ENABLE_CONTROLLER
    register_subtree_runtime(replacement, nullptr, -1, 0);
#endif

    delete[] keys;
    delete[] values;
  }

  bool remove(const T &key) {
    EpochGuard guard(this);
#if ENABLE_CONTROLLER
    auto controller_status = guard.runtime_status();
    bool check_maintenance = controller_status.has_pending_tasks();
    bool check_cooling = controller_status.cooling_execution_active();
#endif
    int restartCount = 0;
  restart:
    if (restartCount++) {
      yield(restartCount);
#if ENABLE_CONTROLLER
      controller_status = g_lipp_controller.runtime_status();
      check_maintenance = controller_status.has_pending_tasks();
      check_cooling = controller_status.cooling_execution_active();
#endif
    }
    bool needRestart = false;
#if ENABLE_CONTROLLER
    const bool has_hash_nodes = controller_status.has_hash_nodes();
#endif

    NodePath path;

    // for lock coupling
    uint64_t versionItem;
    Node *parent = nullptr;
    int parent_pos = -1;
    Node *node = root.load(std::memory_order_acquire);

    while (true) {
      // R-lock this node
#if ENABLE_CONTROLLER
      auto* runtime =
          runtime_for_control(node, check_maintenance, check_cooling);
      if (check_cooling && runtime != nullptr) {
        runtime->cancel_cooling_on_access();
      }
      if (check_maintenance && runtime != nullptr &&
          UNLIKELY(runtime->has_task()) &&
          process_maintenance(node, parent, parent_pos,
                              path.size())) {
        goto restart;
      }
      if (UNLIKELY(has_hash_nodes && node_is_hash(node))) {
        auto* hash_map = node_hash_map(node);
        if (!hash_map) return false;
        path.push(node);
        bool removed = hash_map->remove(key);
        if (removed) {
          for (int index = 0; index < path.size(); ++index) {
            const auto new_size =
                path[index]->size.fetch_sub(1, std::memory_order_relaxed) - 1;
            auto* path_runtime = node_metadata(path[index]);
            if (path_runtime != nullptr) {
              path_runtime->current_size.store(
                  static_cast<std::size_t>(std::max(0, new_size)),
                  std::memory_order_relaxed);
            }
          }
        }
        record_runtime_lookup(path, node, hash_map->size());
        return removed;
      }
#endif
      path.push(node);

      int pos = PREDICT_POS(node, key);
      versionItem = node->items[pos].readLockOrRestart(needRestart);
      if (needRestart)
        goto restart;
      if (node->items[pos].entry_type == 0) // 0 means empty entry
      {
#if ENABLE_CONTROLLER
        record_runtime_lookup(
            path, node,
            static_cast<std::size_t>(
                std::max(0, node->size.load(std::memory_order_relaxed))));
#endif
        return false;
      } else if (node->items[pos].entry_type == 2) // 2 means existing entry has data already
      {
        RT_DEBUG("Existed %p pos %d, locking.", node, pos);
        node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
        if (needRestart) {
          goto restart;
        }

        node->items[pos].entry_type = 0;

        node->items[pos].writeUnlock();
        for (int i = 0; i < path.size(); i++) {
          path[i]->size--;
#if ENABLE_CONTROLLER
          auto* path_runtime = node_metadata(path[i]);
          if (path_runtime != nullptr) {
            path_runtime->current_size.store(
                static_cast<std::size_t>(
                    std::max(0, path[i]->size.load(std::memory_order_relaxed))),
                std::memory_order_relaxed);
          }
#endif
        }
#if ENABLE_CONTROLLER
        record_runtime_lookup(
            path, node,
            static_cast<std::size_t>(
                std::max(0, node->size.load(std::memory_order_relaxed))));
#endif
#if !ENABLE_CONTROLLER
        if(node->size == 0) {
          int parent_pos = PREDICT_POS(parent, key);
          restartCount = 0;
        deleteNodeRemove:
          bool deleteNodeRestart = false;
          if (restartCount++)
            yield(restartCount);
          parent->items[parent_pos].writeLockOrRestart(deleteNodeRestart);
          if(deleteNodeRestart) goto deleteNodeRemove;

          parent->items[parent_pos].entry_type = 0;

          parent->items[parent_pos].writeUnlock();

          safe_delete_nodes(node, 1);

        }
#endif
        return true;
      } else // 1 means has a child, need to go down and see
      {
        parent = node;
#if ENABLE_CONTROLLER
        parent_pos = pos;
#endif
        node = node->items[pos].comp.child;           // now: node is the child

        parent->items[pos].readUnlockOrRestart(versionItem, needRestart);
        if (needRestart)
          goto restart;
      }
    }

  }

  bool update(const T &key, const P& value) {
    EpochGuard guard(this);
#if ENABLE_CONTROLLER
    auto controller_status = guard.runtime_status();
    return update_tree_with_runtime(key, value, controller_status);
#else
    int restartCount = 0;
  restart:
    if (restartCount++) {
      yield(restartCount);
#if ENABLE_CONTROLLER
      controller_status = g_lipp_controller.runtime_status();
      check_maintenance = controller_status.has_pending_tasks();
#endif
    }
    bool needRestart = false;
#if ENABLE_CONTROLLER
    const bool has_hash_nodes = controller_status.has_hash_nodes();
#endif

    // for lock coupling
    uint64_t versionItem;
    Node *parent = nullptr;
#if ENABLE_CONTROLLER
    int parent_pos = -1;
    int current_depth = 0;
#endif

    for (Node *node = root.load(std::memory_order_acquire);;) {
      // R-lock this node
#if ENABLE_CONTROLLER
      auto* runtime =
          runtime_for_control(node, check_maintenance, false);
      if (check_maintenance && runtime != nullptr &&
          UNLIKELY(runtime->has_task()) &&
          process_maintenance(node, parent, parent_pos, current_depth)) {
        goto restart;
      }
      if (UNLIKELY(has_hash_nodes && node_is_hash(node))) {
        auto* hash_map = node_hash_map(node);
        if (!hash_map) return false;
        return hash_map->update(key, value);
      }
#endif
      int pos = PREDICT_POS(node, key);
      versionItem = node->items[pos].readLockOrRestart(needRestart);
      if (needRestart)
        goto restart;
      if (node->items[pos].entry_type == 0) // 0 means empty entry
      {
        return false;
      } else if (node->items[pos].entry_type == 2) // 2 means existing entry has data already
      {
        RT_DEBUG("Existed %p pos %d, locking.", node, pos);
        node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
        if (needRestart) {
          goto restart;
        }

        node->items[pos].comp.data.value = value;

        node->items[pos].writeUnlock();

        break;
      } else // 1 means has a child, need to go down and see
      {
        parent = node;
#if ENABLE_CONTROLLER
        parent_pos = pos;
#endif
        node = node->items[pos].comp.child;           // now: node is the child

        parent->items[pos].readUnlockOrRestart(versionItem, needRestart);
        if (needRestart)
          goto restart;
#if ENABLE_CONTROLLER
        ++current_depth;
#endif
      }
    }

    return true;
#endif
  }

  size_t total_size() const {
    std::stack < Node * > s;
    s.push(root.load(std::memory_order_acquire));

    size_t size = 0;
    while (!s.empty()) {
      Node *node = s.top();
      s.pop();
      size += sizeof(*node);
#if ENABLE_CONTROLLER
      if (node_is_hash(node)) {
        auto* hash_map = node_hash_map(node);
        if (hash_map != nullptr) {
          size += hash_map->bucket_count() * (sizeof(T) + sizeof(P) + 1);
        }
      } else {
          for (int i = 0; i < node->num_items; i++) {
              size += sizeof(Item);
              if (node->items[i].entry_type == 1) {
                  s.push(node->items[i].comp.child);
              }
          }
      }
#else
      for (int i = 0; i < node->num_items; i++) {
        size += sizeof(Item);
        if (node->items[i].entry_type == 1) {
          s.push(node->items[i].comp.child);
        }
      }
#endif
    }
    return size;
  }

  int successful_rebuilds() const noexcept {
    return adjustsuccess.load(std::memory_order_relaxed);
  }

private:
#if ENABLE_CONTROLLER
  static void record_runtime_lookup(
      const NodePath& path, Node* terminal,
      std::size_t terminal_size) noexcept {
    const auto access_time = index_controller::steady_now_ms();
    for (int index = 0; index < path.size(); ++index) {
      auto* runtime = node_metadata(path[index]);
      if (runtime != nullptr && runtime->is_node_valid()) {
        runtime->record_lookup_at(
            path[index] == terminal ? terminal_size : 0, access_time);
      }
    }
  }

  [[gnu::noinline]] bool update_tree_with_runtime(
      const T& key, const P& value,
      lipp_controller::LIPPController::RuntimeStatus controller_status) {
    bool check_maintenance = controller_status.has_pending_tasks();
    bool check_cooling = controller_status.cooling_execution_active();
    int restart_count = 0;
  restart:
    if (restart_count++) {
      yield(restart_count);
      controller_status = g_lipp_controller.runtime_status();
      check_maintenance = controller_status.has_pending_tasks();
      check_cooling = controller_status.cooling_execution_active();
    }
    bool need_restart = false;
    const bool has_hash_nodes = controller_status.has_hash_nodes();
    std::uint64_t version_item;
    Node* parent = nullptr;
    int parent_position = -1;
    int current_depth = 0;
    NodePath lookup_path;

    for (Node* node = root.load(std::memory_order_acquire);;) {
      auto* runtime =
          runtime_for_control(node, check_maintenance, check_cooling);
      if (check_cooling && runtime != nullptr) {
        runtime->cancel_cooling_on_access();
      }
      if (check_maintenance && runtime != nullptr &&
          UNLIKELY(runtime->has_task()) &&
          process_maintenance(node, parent, parent_position,
                              current_depth)) {
        goto restart;
      }
      lookup_path.push(node);
      if (UNLIKELY(has_hash_nodes && node_is_hash(node))) {
        auto* hash_map = node_hash_map(node);
        const bool updated =
            hash_map != nullptr && hash_map->update(key, value);
        record_runtime_lookup(
            lookup_path, node, hash_map == nullptr ? 0 : hash_map->size());
        return updated;
      }

      const int position = PREDICT_POS(node, key);
      version_item =
          node->items[position].readLockOrRestart(need_restart);
      if (need_restart) {
        goto restart;
      }
      if (node->items[position].entry_type == 0) {
        record_runtime_lookup(
            lookup_path, node,
            static_cast<std::size_t>(
                std::max(0, node->size.load(std::memory_order_relaxed))));
        return false;
      }
      if (node->items[position].entry_type == 2) {
        node->items[position].upgradeToWriteLockOrRestart(version_item,
                                                          need_restart);
        if (need_restart) {
          goto restart;
        }
        node->items[position].comp.data.value = value;
        node->items[position].writeUnlock();
        record_runtime_lookup(
            lookup_path, node,
            static_cast<std::size_t>(
                std::max(0, node->size.load(std::memory_order_relaxed))));
        return true;
      }

      parent = node;
      parent_position = position;
      node = node->items[position].comp.child;
      parent->items[position].readUnlockOrRestart(version_item,
                                                  need_restart);
      if (need_restart) {
        goto restart;
      }
      ++current_depth;
    }
  }
#endif

  struct Node;
  struct Item : OptLock {
    union {
      struct {
        T key;
        P value;
      } data;
      Node *child;
    } comp;
    uint8_t entry_type; //0 means empty, 1 means child, 2 means data
  };
  struct Node {
    int is_two;            // is special node for only two keys
    int build_size;        // tree size (include sub nodes) when node created
#if ENABLE_CONTROLLER
    std::atomic<std::uint32_t> controller_word{0};
#else
    int fixed; // fixed node will not trigger rebuild
#endif
    std::atomic<int> num_inserts, num_insert_to_data;
    std::atomic<int> size; // current subtree size
     // number of slots
    std::atomic<int> num_items;
    LinearModel<T> model;
    Item *items;
  };

#if ENABLE_CONTROLLER
  static constexpr std::uint32_t kFixedNodeMask = 0x80000000u;
  static constexpr std::uint32_t kHashNodeMask = 0x40000000u;
  static constexpr std::uint32_t kMaintenanceNodeMask = 0x20000000u;
  static constexpr std::uint32_t kNodeFlagMask =
      kFixedNodeMask | kHashNodeMask | kMaintenanceNodeMask;
  static constexpr std::uint32_t kMetadataHandleMask = 0x1fffffffu;

  static lipp_controller::LIPP_Node_Header* node_metadata(
      const Node* node) noexcept {
    if (node == nullptr) {
      return nullptr;
    }
    const auto word = node->controller_word.load(std::memory_order_acquire);
    return g_lipp_controller.metadata_from_handle(word & kMetadataHandleMask);
  }

  static bool node_is_hash(const Node* node) noexcept {
    return node != nullptr &&
           (node->controller_word.load(std::memory_order_acquire) &
            kHashNodeMask) != 0;
  }

  static bool node_has_maintenance(const Node* node) noexcept {
    return node != nullptr &&
           (node->controller_word.load(std::memory_order_acquire) &
            kMaintenanceNodeMask) != 0;
  }

  static void set_node_maintenance(Node* node, bool pending) noexcept {
    if (node == nullptr) {
      return;
    }
    if (pending) {
      node->controller_word.fetch_or(kMaintenanceNodeMask,
                                     std::memory_order_release);
    } else {
      node->controller_word.fetch_and(~kMaintenanceNodeMask,
                                      std::memory_order_release);
    }
  }

  [[gnu::always_inline]] static lipp_controller::LIPP_Node_Header*
  runtime_for_control(Node* node, bool check_maintenance,
                      bool check_cooling) noexcept {
    if (!check_cooling &&
        (!check_maintenance || !node_has_maintenance(node))) {
      return nullptr;
    }
    return node_metadata(node);
  }

  static HashNodeWrapper<T, P>* node_hash_map(const Node* node) noexcept {
    return node_is_hash(node)
               ? reinterpret_cast<HashNodeWrapper<T, P>*>(node->items)
               : nullptr;
  }

  static void set_node_hash(Node* node) noexcept {
    node->controller_word.fetch_or(kHashNodeMask,
                                   std::memory_order_release);
  }
#endif

  static bool node_fixed(const Node* node) noexcept {
#if ENABLE_CONTROLLER
    return (node->controller_word.load(std::memory_order_relaxed) &
            kFixedNodeMask) != 0;
#else
    return node->fixed != 0;
#endif
  }

  static void set_node_fixed(Node* node, bool fixed) noexcept {
#if ENABLE_CONTROLLER
    if (fixed) {
      node->controller_word.fetch_or(kFixedNodeMask,
                                     std::memory_order_relaxed);
    } else {
      node->controller_word.fetch_and(~kFixedNodeMask,
                                      std::memory_order_relaxed);
    }
#else
    node->fixed = fixed ? 1 : 0;
#endif
  }

#if ENABLE_CONTROLLER

  [[gnu::noinline]] bool at_with_runtime(
      const T& key, P& value,
      lipp_controller::LIPPController::RuntimeStatus controller_status) {
    bool check_maintenance = controller_status.has_pending_tasks();
    bool check_cooling = controller_status.cooling_execution_active();
    int restart_count = 0;
  restart:
    if (restart_count++) {
      yield(restart_count);
      controller_status = g_lipp_controller.runtime_status();
      check_maintenance = controller_status.has_pending_tasks();
      check_cooling = controller_status.cooling_execution_active();
    }
    bool need_restart = false;
    const bool has_hash_nodes = controller_status.has_hash_nodes();
    NodePath lookup_path;
    std::uint64_t version_item;
    Node* parent = nullptr;
    int parent_position = -1;
    int current_depth = 0;

    for (Node* node = root.load(std::memory_order_acquire);;) {
      auto* runtime =
          runtime_for_control(node, check_maintenance, check_cooling);
      if (check_cooling && runtime != nullptr) {
        runtime->cancel_cooling_on_access();
      }
      if (check_maintenance && runtime != nullptr &&
          UNLIKELY(runtime->has_task()) &&
          process_maintenance(node, parent, parent_position,
                              current_depth)) {
        goto restart;
      }
      lookup_path.push(node);
      if (UNLIKELY(has_hash_nodes && node_is_hash(node))) {
        auto* hash_map = node_hash_map(node);
        RT_ASSERT(hash_map != nullptr);
        const bool found = hash_map->find(key, value);
        record_runtime_lookup(lookup_path, node, hash_map->size());
        return found;
      }

      const int position = PREDICT_POS(node, key);
      version_item =
          node->items[position].readLockOrRestart(need_restart);
      if (need_restart) {
        goto restart;
      }
      if (node->items[position].entry_type == 1) {
        parent = node;
        parent_position = position;
        node = node->items[position].comp.child;
        parent->items[position].readUnlockOrRestart(version_item,
                                                    need_restart);
        if (need_restart) {
          goto restart;
        }
      } else {
        if (node->items[position].entry_type == 0) {
          node->items[position].readUnlockOrRestart(version_item,
                                                    need_restart);
          if (need_restart) {
            goto restart;
          }
          record_runtime_lookup(
              lookup_path, node,
              static_cast<std::size_t>(
                  node->size.load(std::memory_order_relaxed)));
          return false;
        }
        node->items[position].readUnlockOrRestart(version_item,
                                                  need_restart);
        if (need_restart) {
          goto restart;
        }
        value = node->items[position].comp.data.value;
        record_runtime_lookup(
            lookup_path, node,
            static_cast<std::size_t>(
                node->size.load(std::memory_order_relaxed)));
        return true;
      }
      ++current_depth;
    }
  }

#endif

  std::atomic<Node*> root{nullptr};
  std::atomic<int> adjustsuccess{0};
  FastConcurrentStack pending_two[1024];
#if ENABLE_CONTROLLER
  std::mutex maintenance_mutex_;
  std::vector<T> maintenance_keys_;
  std::vector<P> maintenance_values_;
  std::vector<Item*> maintenance_locked_items_;
#endif

  std::allocator<Node> node_allocator;
  Node *new_nodes(int n) {
    Node *p = node_allocator.allocate(n);
    RT_ASSERT(p != NULL && p != (Node *)(-1));
#if ENABLE_CONTROLLER
    for (int i = 0; i < n; ++i) {
      p[i].controller_word.store(0, std::memory_order_relaxed);
    }
#endif
    return p;
  }

  void delete_nodes(Node *p, int n) {
#if ENABLE_CONTROLLER
    for (int i = 0; i < n; ++i) {
      Node* node = p + i;
      auto* metadata = node_metadata(node);
      if (metadata != nullptr) {
        auto* hash_map = node_hash_map(node);
        if (metadata->needs_rebuild() || metadata->needs_hash()) {
          g_lipp_controller.complete_foreground_task();
        }
        metadata->mark_deleted();
        metadata->wait_for_callbacks();
        node->controller_word.store(0, std::memory_order_release);
        if (hash_map != nullptr) {
          delete hash_map;
          g_lipp_controller.notify_hash_node_deleted();
        }
        g_lipp_controller.release_metadata(metadata);
      }
    }
#endif
    node_allocator.deallocate(p, n);
  }

  void safe_delete_nodes(Node *p, int n) {
    for (int i = 0; i < n; ++i) {
      Node* node = p + i;
#if ENABLE_CONTROLLER
      auto* metadata = node_metadata(node);
      if (metadata != nullptr) {
        if (metadata->needs_rebuild() || metadata->needs_hash()) {
          g_lipp_controller.complete_foreground_task();
        }
        metadata->mark_deleted();
      }
#endif
      ebr->scheduleForDeletion(node);
    }
  }

  std::allocator<Item> item_allocator;
  Item *new_items(int n) {
    Item *p = item_allocator.allocate(n);
    for (int i = 0; i < n; ++i) {
      p[i].typeVersionLockObsolete.store(0b100);
      p[i].entry_type = 0;
    }
    RT_ASSERT(p != NULL && p != (Item *)(-1));
    return p;
  }
  void delete_items(Item *p, int n) { item_allocator.deallocate(p, n); }

  /// build an empty tree
  Node *build_tree_none() {
    Node *node = new_nodes(1);
    node->is_two = 0;
    node->build_size = 0;
    node->size = 0;
    node->num_inserts = node->num_insert_to_data = 0;
    set_node_fixed(node, false);
    node->num_items = 1;
    node->model.a = node->model.b = 0;
    node->items = new_items(1);
    node->items[0].entry_type = 0;
    return node;
  }
  /// build a tree with two keys
  Node *build_tree_two(T key1, P value1, T key2, P value2) {
    if (key1 > key2) {
      std::swap(key1, key2);
      std::swap(value1, value2);
    }
    RT_ASSERT(key1 < key2);

    Node *node = pending_two[omp_get_thread_num()].pop();
    if (node == nullptr) {
      node = new_nodes(1);
      node->is_two = 1;
      node->build_size = 2;
      node->size = 2;
      node->num_inserts = node->num_insert_to_data = 0;
      set_node_fixed(node, false);

      node->num_items = 8;
      node->items = new_items(node->num_items);
    } else {
      node->size = 2;
      node->num_inserts = node->num_insert_to_data = 0;
#if ENABLE_CONTROLLER
      node->controller_word.store(0, std::memory_order_relaxed);
#endif
    }

    const long double mid1_key = key1;
    const long double mid2_key = key2;

    const double mid1_target = node->num_items / 3;
    const double mid2_target = node->num_items * 2 / 3;

    node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
    node->model.b = mid1_target - node->model.a * mid1_key;
    RT_ASSERT(isfinite(node->model.a));
    RT_ASSERT(isfinite(node->model.b));

    { // insert key1&value1
      int pos = PREDICT_POS(node, key1);
      RT_ASSERT(node->items[pos].entry_type == 0);
      node->items[pos].entry_type = 2;
      node->items[pos].comp.data.key = key1;
      node->items[pos].comp.data.value = value1;
    }
    { // insert key2&value2
      int pos = PREDICT_POS(node, key2);
      RT_ASSERT(node->items[pos].entry_type == 0);
      node->items[pos].entry_type = 2;
      node->items[pos].comp.data.key = key2;
      node->items[pos].comp.data.value = value2;
    }

    return node;
  }
  /// bulk build, _keys must be sorted in asc order.
  Node *build_tree_bulk(T *_keys, P *_values, int _size,
                        bool compact_layout = false) {
    if (USE_FMCD && !compact_layout) {
      return build_tree_bulk_fmcd(_keys, _values, _size);
    } else {
      return build_tree_bulk_fast(_keys, _values, _size, compact_layout);
    }
  }
  /// bulk build, _keys must be sorted in asc order.
  /// split keys into three parts at each node.
  Node *build_tree_bulk_fast(T *_keys, P *_values, int _size,
                             bool compact_layout = false) {
    RT_ASSERT(_size > 1);

    typedef struct {
      int begin;
      int end;
      int level; // top level = 1
      Node *node;
    } Segment;
    std::stack<Segment> s;

    Node *ret = new_nodes(1);
    s.push((Segment){0, _size, 1, ret});

    while (!s.empty()) {
      const int begin = s.top().begin;
      const int end = s.top().end;
      const int level = s.top().level;
      Node *node = s.top().node;
      s.pop();

      RT_ASSERT(end - begin >= 2);
      if (end - begin == 2) {
        Node *_ = build_tree_two(_keys[begin], _values[begin], _keys[begin + 1],
                                 _values[begin + 1]);
        node->is_two = _->is_two;
        node->build_size = _->build_size;
        set_node_fixed(node, node_fixed(_));
        node->size.store(_->size.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        node->num_inserts.store(0, std::memory_order_relaxed);
        node->num_insert_to_data.store(0, std::memory_order_relaxed);
        node->num_items.store(_->num_items.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
        node->model = _->model;
        node->items = _->items;
        delete_nodes(_, 1);
      } else {
        T *keys = _keys + begin;
        P *values = _values + begin;
        const int size = end - begin;
        const int BUILD_GAP_CNT =
            compact_layout ? 0 : compute_gap_count(size);

        node->is_two = 0;
        node->build_size = size;
        node->size = size;
        node->num_inserts = node->num_insert_to_data = 0;
        set_node_fixed(node, false);

        int mid1_pos = (size - 1) / 3;
        int mid2_pos = (size - 1) * 2 / 3;

        RT_ASSERT(0 <= mid1_pos);
        RT_ASSERT(mid1_pos < mid2_pos);
        RT_ASSERT(mid2_pos < size - 1);

        const long double mid1_key =
            (static_cast<long double>(keys[mid1_pos]) +
             static_cast<long double>(keys[mid1_pos + 1])) /
            2;
        const long double mid2_key =
            (static_cast<long double>(keys[mid2_pos]) +
             static_cast<long double>(keys[mid2_pos + 1])) /
            2;

        node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
        const double mid1_target =
            mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
            static_cast<int>(BUILD_GAP_CNT + 1) / 2;
        const double mid2_target =
            mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
            static_cast<int>(BUILD_GAP_CNT + 1) / 2;

        node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
        node->model.b = mid1_target - node->model.a * mid1_key;
        RT_ASSERT(isfinite(node->model.a));
        RT_ASSERT(isfinite(node->model.b));

        const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
        node->model.b += lr_remains;
        node->num_items += lr_remains * 2;

        if (size > 1e6) {
          set_node_fixed(node, true);
        }

        node->items = new_items(node->num_items);

        for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
             offset < size;) {
          int next = offset + 1, next_i = -1;
          while (next < size) {
            next_i = PREDICT_POS(node, keys[next]);
            if (next_i == item_i) {
              next++;
            } else {
              break;
            }
          }
          if (next == offset + 1) {
            node->items[item_i].entry_type = 2;
            node->items[item_i].comp.data.key = keys[offset];
            node->items[item_i].comp.data.value = values[offset];
          } else {
            node->items[item_i].entry_type = 1;
            node->items[item_i].comp.child = new_nodes(1);
            s.push((Segment){begin + offset, begin + next, level + 1,
                             node->items[item_i].comp.child});
          }
          if (next >= size) {
            break;
          } else {
            item_i = next_i;
            offset = next;
          }
        }
      }
    }

    return ret;
  }
  /// bulk build, _keys must be sorted in asc order.
  /// FMCD method.
  Node *build_tree_bulk_fmcd(T *_keys, P *_values, int _size) {
    RT_ASSERT(_size > 1);

    typedef struct {
      int begin;
      int end;
      int level; // top level = 1
      Node *node;
    } Segment;
    std::stack<Segment> s;

    Node *ret = new_nodes(1);
    s.push((Segment){0, _size, 1, ret});

    while (!s.empty()) {
      const int begin = s.top().begin;
      const int end = s.top().end;
      const int level = s.top().level;
      Node *node = s.top().node;
      s.pop();

      RT_ASSERT(end - begin >= 2);
      if (end - begin == 2) {
        Node *_ = build_tree_two(_keys[begin], _values[begin], _keys[begin + 1],
                                 _values[begin + 1]);
        node->is_two = _->is_two;
        node->build_size = _->build_size;
        set_node_fixed(node, node_fixed(_));
        node->size.store(_->size.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        node->num_inserts.store(0, std::memory_order_relaxed);
        node->num_insert_to_data.store(0, std::memory_order_relaxed);
        node->num_items.store(_->num_items.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
        node->model = _->model;
        node->items = _->items;
        delete_nodes(_, 1);
      } else {
        T *keys = _keys + begin;
        P *values = _values + begin;
        const int size = end - begin;
        const int BUILD_GAP_CNT = compute_gap_count(size);

        node->is_two = 0;
        node->build_size = size;
        node->size = size;
        node->num_inserts = node->num_insert_to_data = 0;
        set_node_fixed(node, false);
        // FMCD method
        // Here the implementation is a little different with Algorithm 1 in our
        // paper. In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L
        // - 2). But according to the derivation described in our paper, M.A
        // should be less than 1 / U_T. So we added a small number (1e-6) to
        // U_T. In fact, it has only a negligible impact of the performance.
        {
          const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
          int i = 0;
          int D = 1;
          RT_ASSERT(D <= size - 1 - D);
          double Ut = (static_cast<long double>(keys[size - 1 - D]) -
                       static_cast<long double>(keys[D])) /
                          (static_cast<double>(L - 2)) +
                      1e-6;
          while (i < size - 1 - D) {
            while (i + D < size && keys[i + D] - keys[i] >= Ut) {
              i++;
            }
            if (i + D >= size) {
              break;
            }
            D = D + 1;
            if (D * 3 > size)
              break;
            RT_ASSERT(D <= size - 1 - D);
            Ut = (static_cast<long double>(keys[size - 1 - D]) -
                  static_cast<long double>(keys[D])) /
                     (static_cast<double>(L - 2)) +
                 1e-6;
          }
          if (D * 3 <= size) {
            stats.fmcd_success_times++;

            node->model.a = 1.0 / Ut;
            node->model.b =
                (L -
                 node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
                                  static_cast<long double>(keys[D]))) /
                2;
            RT_ASSERT(isfinite(node->model.a));
            RT_ASSERT(isfinite(node->model.b));
            node->num_items = L;
          } else {
            stats.fmcd_broken_times++;

            int mid1_pos = (size - 1) / 3;
            int mid2_pos = (size - 1) * 2 / 3;

            RT_ASSERT(0 <= mid1_pos);
            RT_ASSERT(mid1_pos < mid2_pos);
            RT_ASSERT(mid2_pos < size - 1);

            const long double mid1_key =
                (static_cast<long double>(keys[mid1_pos]) +
                 static_cast<long double>(keys[mid1_pos + 1])) /
                2;
            const long double mid2_key =
                (static_cast<long double>(keys[mid2_pos]) +
                 static_cast<long double>(keys[mid2_pos + 1])) /
                2;

            node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
            const double mid1_target =
                mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
                static_cast<int>(BUILD_GAP_CNT + 1) / 2;
            const double mid2_target =
                mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
                static_cast<int>(BUILD_GAP_CNT + 1) / 2;

            node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
            node->model.b = mid1_target - node->model.a * mid1_key;
            RT_ASSERT(isfinite(node->model.a));
            RT_ASSERT(isfinite(node->model.b));
          }
        }
        RT_ASSERT(node->model.a >= 0);
        const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
        node->model.b += lr_remains;
        node->num_items += lr_remains * 2;

        if (size > 1e6) {
          set_node_fixed(node, true);
        }

        node->items = new_items(node->num_items);

        for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
             offset < size;) {
          int next = offset + 1, next_i = -1;
          while (next < size) {
            next_i = PREDICT_POS(node, keys[next]);
            if (next_i == item_i) {
              next++;
            } else {
              break;
            }
          }
          if (next == offset + 1) {
            node->items[item_i].entry_type = 2;
            node->items[item_i].comp.data.key = keys[offset];
            node->items[item_i].comp.data.value = values[offset];
          } else {
            node->items[item_i].entry_type = 1;
            node->items[item_i].comp.child = new_nodes(1);
            s.push((Segment){begin + offset, begin + next, level + 1,
                             node->items[item_i].comp.child});
          }
          if (next >= size) {
            break;
          } else {
            item_i = next_i;
            offset = next;
          }
        }
      }
    }

    return ret;
  }

  void destory_pending() {
    for (int i = 0; i < 1024; ++i) {
      while (Node *node = pending_two[i].pop()) {
        delete_items(node->items, node->num_items);
        delete_nodes(node, 1);
      }
    }
  }

  void destroy_tree(Node *root) {
    if (root == nullptr) {
      return;
    }
    std::stack<Node *> s;
    s.push(root);
    while (!s.empty()) {
      Node *node = s.top();
      s.pop();

#if ENABLE_CONTROLLER
      if (node_is_hash(node)) {
        delete_nodes(node, 1);
        continue;
      }
#endif

      for (int i = 0; i < node->num_items; i++) {
        if (node->items[i].entry_type == 1) {
          s.push(node->items[i].comp.child);
        }
      }

      if (node->is_two) {
        RT_ASSERT(node->build_size == 2);
        RT_ASSERT(node->num_items == 8);
#if ENABLE_CONTROLLER
        auto* metadata = node_metadata(node);
        if (metadata != nullptr) {
          metadata->mark_deleted();
          metadata->wait_for_callbacks();
          node->controller_word.store(0, std::memory_order_release);
          g_lipp_controller.release_metadata(metadata);
        }
#endif
        node->size = 2;
        node->num_inserts = node->num_insert_to_data = 0;
        for(int i = 0; i < node->num_items; i++) node->items[i].typeVersionLockObsolete.store(0b100);;
        for(int i = 0; i < node->num_items; i++) node->items[i].entry_type = 0;
        pending_two[omp_get_thread_num()].push(node);
      } else {
        delete_items(node->items, node->num_items);
        delete_nodes(node, 1);
      }
    }
  }
#if !ENABLE_CONTROLLER
  static void unlock_rebuild_items(std::vector<Item*>& locked_items,
                                   bool obsolete) {
    for (auto iterator = locked_items.rbegin(); iterator != locked_items.rend();
         ++iterator) {
      if (obsolete) {
        (*iterator)->writeUnlockObsolete();
      } else {
        (*iterator)->writeUnlock();
      }
    }
    locked_items.clear();
  }

  bool collect_rebuild_data(Node* subroot, std::vector<T>& keys,
                            std::vector<P>& values,
                            std::vector<Node*>& nodes,
                            std::vector<Item*>& locked_items) {
    nodes.push_back(subroot);
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      Node* node = nodes[node_index];
      const int slots = node->num_items.load(std::memory_order_acquire);
      for (int slot = 0; slot < slots; ++slot) {
        bool restart = false;
        node->items[slot].writeLockOrRestart(restart);
        if (restart) {
          return false;
        }
        locked_items.push_back(&node->items[slot]);
        if (node->items[slot].entry_type == 1) {
          nodes.push_back(node->items[slot].comp.child);
        }
      }
    }

    keys.reserve(static_cast<std::size_t>(
        std::max(0, subroot->size.load(std::memory_order_relaxed))));
    values.reserve(keys.capacity());
    std::vector<Node*> pending;
    pending.push_back(subroot);
    while (!pending.empty()) {
      Node* node = pending.back();
      pending.pop_back();
      const int slots = node->num_items.load(std::memory_order_relaxed);
      for (int slot = slots - 1; slot >= 0; --slot) {
        if (node->items[slot].entry_type == 1) {
          pending.push_back(node->items[slot].comp.child);
        }
      }
      for (int slot = 0; slot < slots; ++slot) {
        if (node->items[slot].entry_type == 2) {
          keys.push_back(node->items[slot].comp.data.key);
          values.push_back(node->items[slot].comp.data.value);
        }
      }
    }
    return true;
  }

  bool rebuild_foreground(const NodePath& path, int path_index, T key) {
    Node* node = path[path_index];
    static thread_local std::vector<T> keys;
    static thread_local std::vector<P> values;
    static thread_local std::vector<Node*> nodes;
    static thread_local std::vector<Item*> locked_items;
    keys.clear();
    values.clear();
    nodes.clear();
    locked_items.clear();

    if (!collect_rebuild_data(node, keys, values, nodes, locked_items)) {
      unlock_rebuild_items(locked_items, false);
      return false;
    }
    if (keys.size() < 4) {
      unlock_rebuild_items(locked_items, false);
      return false;
    }

    std::vector<std::pair<T, P>> sorted(keys.size());
    for (std::size_t index = 0; index < keys.size(); ++index) {
      sorted[index] = {keys[index], values[index]};
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& left, const auto& right) {
                return left.first < right.first;
              });
    for (std::size_t index = 0; index < sorted.size(); ++index) {
      keys[index] = sorted[index].first;
      values[index] = sorted[index].second;
    }

    Node* replacement = build_tree_bulk(keys.data(), values.data(), keys.size());
    bool replaced = replacement != nullptr;
    if (replaced && path_index == 0) {
      Node* expected = node;
      replaced = root.compare_exchange_strong(
          expected, replacement, std::memory_order_acq_rel,
          std::memory_order_acquire);
    } else if (replaced) {
      Node* parent = path[path_index - 1];
      const int slot = PREDICT_POS(parent, key);
      bool restart = false;
      parent->items[slot].writeLockOrRestart(restart);
      if (restart) {
        replaced = false;
      } else {
        if (parent->items[slot].entry_type != 1 ||
            parent->items[slot].comp.child != node) {
          replaced = false;
        } else {
          parent->items[slot].comp.child = replacement;
        }
        parent->items[slot].writeUnlock();
      }
    }

    if (!replaced) {
      if (replacement != nullptr) {
        destroy_tree(replacement);
      }
      unlock_rebuild_items(locked_items, false);
      return false;
    }

    unlock_rebuild_items(locked_items, true);
    for (Node* retired : nodes) {
      safe_delete_nodes(retired, 1);
    }
    adjustsuccess.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
#endif

#if ENABLE_CONTROLLER
  static bool cooling_compaction(
      void* context, lipp_controller::LIPP_Node_Header* metadata) {
    auto* node = static_cast<Node*>(context);
    auto* owner = metadata == nullptr
                      ? nullptr
                      : static_cast<LIPP*>(
                            g_lipp_controller.owner_context_from_handle(
                                metadata->registry_handle));
    if (node == nullptr || owner == nullptr ||
        node_metadata(node) != metadata) {
      return false;
    }
    return owner->compact_cold_node(node, metadata);
  }

  static void publish_foreground_task(void* context, bool pending) {
    set_node_maintenance(static_cast<Node*>(context), pending);
  }

  lipp_controller::LIPP_Node_Header* attach_runtime_metadata(
      Node* node, std::uint32_t depth,
      const lipp_controller::LIPP_Node_Header* previous = nullptr,
      std::uint32_t depth_before = 0, std::uint32_t depth_after = 0,
      bool record_structural_change = true, Node* parent = nullptr,
      int parent_slot = -1) {
    if (node == nullptr) {
      return nullptr;
    }
    auto* current = node_metadata(node);
    if (current != nullptr) {
      current->set_layer_depth(depth);
      const auto* parent_metadata = node_metadata(parent);
      g_lipp_controller.update_context(
          current, node, this,
          parent_metadata == nullptr ? 0 : parent_metadata->registry_handle);
      return current;
    }
    const auto size = static_cast<std::size_t>(
        std::max(0, node->size.load(std::memory_order_relaxed)));
    const auto& config = g_lipp_controller.get_config();
    const auto admission_size = previous != nullptr
                                    ? config.min_monitor_size
                                    : config.eager_monitor_size;
    if (size < admission_size) {
      return nullptr;
    }
    const auto capacity = static_cast<std::size_t>(
        node->num_items.load(std::memory_order_relaxed));
    auto* created = g_lipp_controller.create_metadata(
        size, capacity, depth, previous, node, &LIPP::cooling_compaction,
        &LIPP::publish_foreground_task, false, depth_before, depth_after, false,
        record_structural_change);
    if (previous == nullptr) {
      const auto build_size = static_cast<std::size_t>(
          std::max(0, node->build_size));
      const auto bounded_build_size = std::min(build_size, size);
      created->build_size.store(
          lipp_controller::LIPP_Node_Header::narrow_size(
              bounded_build_size),
          std::memory_order_relaxed);
      created->total_inserts.store(
          lipp_controller::LIPP_Node_Header::narrow_size(
              size - bounded_build_size),
          std::memory_order_relaxed);
    }
    const auto* parent_metadata = node_metadata(parent);
    g_lipp_controller.update_context(
        created, node, this,
        parent_metadata == nullptr ? 0 : parent_metadata->registry_handle);
    RT_ASSERT(created->registry_handle <= kMetadataHandleMask);
    auto expected = node->controller_word.load(std::memory_order_acquire);
    while ((expected & kMetadataHandleMask) == 0) {
      const auto desired = (expected & kNodeFlagMask) |
                           created->registry_handle;
      if (node->controller_word.compare_exchange_weak(
              expected, desired, std::memory_order_release,
              std::memory_order_acquire)) {
        g_lipp_controller.register_node(created);
        return created;
      }
    }
    created->mark_deleted();
    created->wait_for_callbacks();
    g_lipp_controller.release_metadata(created);
    current = g_lipp_controller.metadata_from_handle(
        expected & kMetadataHandleMask);
    if (current != nullptr) {
      g_lipp_controller.update_context(
          current, node, this,
          parent_metadata == nullptr ? 0 : parent_metadata->registry_handle);
    }
    return current;
  }

  void register_subtree_runtime(Node* subtree, Node* parent, int parent_slot,
                                std::uint32_t depth,
                                bool compacted = false) {
    if (subtree == nullptr) {
      return;
    }
    auto* metadata = attach_runtime_metadata(
        subtree, depth, nullptr, 0, 0, true, parent, parent_slot);
    if (metadata != nullptr && compacted) {
      metadata->set_compacted_layout(true);
    }
    if (node_is_hash(subtree)) {
      return;
    }
    const auto slots = subtree->num_items.load(std::memory_order_acquire);
    for (int slot = 0; slot < slots; ++slot) {
      if (subtree->items[slot].entry_type == 1) {
        register_subtree_runtime(subtree->items[slot].comp.child, subtree,
                                 slot, depth + 1, compacted);
      }
    }
  }

  [[gnu::always_inline]] void record_insert_path(
      const NodePath& path, const T& key, std::uint32_t access_time) {
    const auto monitor_size =
        g_lipp_controller.get_config().eager_monitor_size;
    for (int index = 0; index < path.size(); ++index) {
      Node* node = path[index];
      const auto new_size =
          node->size.fetch_add(1, std::memory_order_relaxed) + 1;
      auto* runtime = node_metadata(node);
      bool attached = false;
      if (runtime == nullptr &&
          static_cast<std::size_t>(new_size) >= monitor_size) {
        Node* parent = index == 0 ? nullptr : path[index - 1];
        runtime = attach_runtime_metadata(
            node, static_cast<std::uint32_t>(index), nullptr, 0, 0, true,
            parent, parent == nullptr ? -1 : PREDICT_POS(parent, key));
        attached = runtime != nullptr;
      }
      if (runtime != nullptr && !attached) {
        runtime->record_insert_access(static_cast<std::size_t>(new_size),
                                      access_time);
      }
    }
  }

  [[gnu::always_inline]] void record_insert_conflict(
      Node* conflict_node) {
    auto* runtime = node_metadata(conflict_node);
    if (runtime != nullptr) {
      runtime->record_conflict();
    }
  }
#endif

#if ENABLE_CONTROLLER
  template <bool CheckMaintenance>
  bool insert_tree_fast(const T& key, const P& value) {
    int restart_count = 0;
  restart:
    if (restart_count++) {
      yield(restart_count);
    }
    bool need_restart = false;
    NodePath path;
    Node* conflict_node = nullptr;
    Node* parent = nullptr;
    int parent_position = -1;
    int current_depth = 0;
    std::uint64_t version_item;

    for (Node* node = root.load(std::memory_order_acquire);;) {
      path.push(node);
      if constexpr (CheckMaintenance) {
        const auto controller_word =
            node->controller_word.load(std::memory_order_acquire);
        if (UNLIKELY((controller_word & kMaintenanceNodeMask) != 0) &&
            process_maintenance(node, parent, parent_position,
                                current_depth)) {
          goto restart;
        }
        if (UNLIKELY((controller_word & kHashNodeMask) != 0)) {
          auto* hash_map = node_hash_map(node);
          RT_ASSERT(hash_map != nullptr);
          if (!hash_map->insert(key, value)) {
            return false;
          }
          record_insert_path(path, key, index_controller::steady_now_ms());
          return true;
        }
      }
      const int position = PREDICT_POS(node, key);
      version_item =
          node->items[position].readLockOrRestart(need_restart);
      if (need_restart) {
        goto restart;
      }

      if (node->items[position].entry_type == 0) {
        node->items[position].upgradeToWriteLockOrRestart(version_item,
                                                          need_restart);
        if (need_restart) {
          goto restart;
        }
        node->items[position].entry_type = 2;
        node->items[position].comp.data.key = key;
        node->items[position].comp.data.value = value;
        node->items[position].writeUnlock();
        break;
      }

      if (node->items[position].entry_type == 2) {
        if (key == node->items[position].comp.data.key) {
          return false;
        }
        node->items[position].upgradeToWriteLockOrRestart(version_item,
                                                          need_restart);
        if (need_restart) {
          goto restart;
        }
        auto* new_node = build_tree_two(
            key, value, node->items[position].comp.data.key,
            node->items[position].comp.data.value);
        attach_runtime_metadata(
            new_node, static_cast<std::uint32_t>(current_depth + 1),
            nullptr, 0, 0, true, node, position);
        node->items[position].comp.child = new_node;
        node->items[position].entry_type = 1;
        node->items[position].writeUnlock();
        conflict_node = node;
        break;
      }

      parent = node;
      parent_position = position;
      node = node->items[position].comp.child;
      parent->items[position].readUnlockOrRestart(version_item,
                                                  need_restart);
      if (need_restart) {
        goto restart;
      }
      ++current_depth;
    }

    record_insert_path(path, key, index_controller::steady_now_ms());
    if (conflict_node != nullptr) {
      record_insert_conflict(conflict_node);
    }
    return true;
  }

#endif

  bool insert_tree(
      const T &key, const P &value
#if ENABLE_CONTROLLER
      , lipp_controller::LIPPController::RuntimeStatus controller_status
#endif
  ) {
#if ENABLE_CONTROLLER
    bool check_maintenance = controller_status.has_pending_tasks();
    bool check_cooling = controller_status.cooling_execution_active();
    if (!check_cooling && !controller_status.has_hash_nodes()) {
      return check_maintenance ? insert_tree_fast<true>(key, value)
                               : insert_tree_fast<false>(key, value);
    }
#endif
    int restartCount = 0;
  restart:
    if (restartCount++) {
      yield(restartCount);
#if ENABLE_CONTROLLER
      controller_status = g_lipp_controller.runtime_status();
      check_maintenance = controller_status.has_pending_tasks();
      check_cooling = controller_status.cooling_execution_active();
#endif
    }
    bool needRestart = false;
#if ENABLE_CONTROLLER
    const bool has_hash_nodes = controller_status.has_hash_nodes();
#endif

    NodePath path;
    int insert_to_data = 0;
    Node* conflict_node = nullptr;

    // for lock coupling
    uint64_t versionItem;
    Node *parent = nullptr;
    int parent_pos = -1;
    int current_depth = 0;

    for (Node *node = root.load(std::memory_order_acquire);;) {
#if ENABLE_CONTROLLER
      auto* runtime =
          runtime_for_control(node, check_maintenance, check_cooling);
      if (runtime != nullptr) {
        if (check_cooling) {
          runtime->cancel_cooling_on_access();
        }
        if (check_maintenance && UNLIKELY(runtime->has_task()) &&
            process_maintenance(node, parent, parent_pos,
                                current_depth)) {
          goto restart;
        }
      }
      if (UNLIKELY(has_hash_nodes && node_is_hash(node))) {
        auto* hash_map = node_hash_map(node);
        RT_ASSERT(hash_map != nullptr);
        path.push(node);
        if (!hash_map->insert(key, value)) {
          return false;
        }
        record_insert_path(path, key, index_controller::steady_now_ms());
        return true;
      }
#endif
      // R-lock this node
      path.push(node);
      int pos = PREDICT_POS(node, key);
      versionItem = node->items[pos].readLockOrRestart(needRestart);
      if (needRestart)
        goto restart;
      if (node->items[pos].entry_type == 0) // 0 means empty entry
      {
        RT_DEBUG("Empty %p pos %d, locking.", node, pos);
        node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
        if (needRestart) {
          goto restart;
        }

        node->items[pos].entry_type = 2;
        node->items[pos].comp.data.key = key;
        node->items[pos].comp.data.value = value;

        RT_DEBUG("Key %d inserted into node %p.  Unlock", key, node);

        node->items[pos].writeUnlock();
        break;
      } else if (node->items[pos].entry_type == 2) // 2 means existing entry has data already
      {
        if (key == node->items[pos].comp.data.key) return false;
        RT_DEBUG("Existed %p pos %d, locking.", node, pos);

        node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
        if (needRestart) {
          goto restart;
        }

        node->items[pos].entry_type = 1;
        Node* new_node =
            build_tree_two(key, value, node->items[pos].comp.data.key,
                           node->items[pos].comp.data.value);
#if ENABLE_CONTROLLER
        attach_runtime_metadata(
            new_node, static_cast<std::uint32_t>(current_depth + 1),
            nullptr, 0, 0, true, node, pos);
#endif
        insert_to_data = 1;
        conflict_node = node;

        node->items[pos].comp.child = new_node;
        node->items[pos].entry_type = 1;

        RT_DEBUG("Key %d inserted into node %p.  Unlock", key, node);

        node->items[pos].writeUnlock();
        break;
      } else // 1 means has a child, need to go down and see
      {
        parent = node;
        parent_pos = pos;
        node = node->items[pos].comp.child;           // now: node is the child

        parent->items[pos].readUnlockOrRestart(versionItem, needRestart);

        if (needRestart)
          goto restart;

        current_depth++;
      }
    }

    //***** so, when reaching here, no node is locked.
#if ENABLE_CONTROLLER
    const auto access_time = index_controller::steady_now_ms();
    record_insert_path(path, key, access_time);
#else
    for (int i = 0; i < path.size(); i++) {
      path[i]->num_insert_to_data.fetch_add(insert_to_data,
                                            std::memory_order_relaxed);
      path[i]->num_inserts.fetch_add(1, std::memory_order_relaxed);
      path[i]->size.fetch_add(1, std::memory_order_relaxed);
    }
#endif

#if ENABLE_CONTROLLER
    if (conflict_node != nullptr) {
      record_insert_conflict(conflict_node);
    }

#else
    for (int i = 0; i < path.size(); i++) {
      Node *node = path[i];
      const int num_inserts =
          node->num_inserts.load(std::memory_order_relaxed);
      const int num_insert_to_data =
          node->num_insert_to_data.load(std::memory_order_relaxed);
      const int node_size = node->size.load(std::memory_order_relaxed);
      const bool need_rebuild =
          node->fixed == 0 && node_size >= node->build_size * 2 &&
          node_size >= 64 && num_insert_to_data * 10 >= num_inserts;

      if (need_rebuild) {
        rebuild_foreground(path, i, key);
        break; // break out for the for loop
      }        // end REBUILD
    }          // end for
#endif

    return true;
  } // end of insert_tree

#if ENABLE_CONTROLLER
  bool collect_node_data_optimistic(Node* node, std::vector<T>& keys, std::vector<P>& values) {
    if (!node) return false;

    if (node_is_hash(node)) {
      auto* hash_map = node_hash_map(node);
      if (!hash_map) return false;
      hash_map->get_all_items(keys, values);
      return true;
    }

    for (int i = 0; i < node->num_items; ++i) {
      bool needRestart = false;
      uint64_t version = node->items[i].readLockOrRestart(needRestart);
      if (needRestart) return false;

      uint8_t entry_type = node->items[i].entry_type;

      if (entry_type == 2) {  // data
        T key = node->items[i].comp.data.key;
        P value = node->items[i].comp.data.value;

        node->items[i].readUnlockOrRestart(version, needRestart);
        if (needRestart) return false;

        keys.push_back(key);
        values.push_back(value);

      } else if (entry_type == 1) {  // child
        Node* child = node->items[i].comp.child;
        node->items[i].readUnlockOrRestart(version, needRestart);
        if (needRestart) return false;

        if (!collect_node_data_optimistic(child, keys, values)) {
          return false;
        }
      } else {
        node->items[i].readUnlockOrRestart(version, needRestart);
        if (needRestart) return false;
      }
    }
    return true;
  }

  bool collect_node_data(Node* node, std::vector<T>& keys, std::vector<P>& values) {
    keys.clear();
    values.clear();

    const int MAX_RETRIES = 5;
    for (int retry = 0; retry < MAX_RETRIES; ++retry) {
      if (retry > 0) {
        keys.clear();
        values.clear();
        std::this_thread::yield();
      }

      if (collect_node_data_optimistic(node, keys, values)) {
        return true;
      }
    }
    return false;
  }

  bool convert_node_to_hash(Node* node, Node* parent, int slot_idx) {
    if (!node || !parent || node_is_hash(node)) return false;

    std::vector<T> keys;
    std::vector<P> values;

    if (!collect_node_data(node, keys, values)) {
      std::cerr << "Failed to collect node data for hash conversion" << std::endl;
      return false;
    }

    if (keys.empty()) {
      std::cerr << "No keys to convert to hash" << std::endl;
      return false;
    }
    if (keys.size() < g_lipp_controller.get_config().min_monitor_size) {
      return false;
    }

    Node* new_node = new_nodes(1);
    auto* hash_map = new HashNodeWrapper<T, P>();
    hash_map->reserve(keys.size() * 1.5);
    hash_map->batch_insert(keys, values);

    new_node->build_size = keys.size();
    set_node_fixed(new_node, true);
    new_node->is_two = 0;
    new_node->num_items = 0;
    new_node->size = static_cast<int>(keys.size());
    new_node->num_inserts = 0;
    new_node->num_insert_to_data = 0;
    new_node->items = reinterpret_cast<Item*>(hash_map);
    new_node->model.a = new_node->model.b = 0;
    auto* previous = node_metadata(node);
    const auto depth = previous == nullptr
                           ? 0U
                           : previous->layer_depth_value();
    auto* new_metadata = attach_runtime_metadata(
        new_node, depth, previous, 0, 0, false, parent, slot_idx);
    new_metadata->is_hash_node.store(true, std::memory_order_release);
    new_metadata->current_capacity.store(hash_map->bucket_count(),
                                         std::memory_order_relaxed);
    set_node_hash(new_node);
    g_lipp_controller.notify_hash_node_created();

    int retryLockCount = 0;
retryLock:
    if (retryLockCount++)
      yield(retryLockCount);

    bool needRetry = false;
    parent->items[slot_idx].writeLockOrRestart(needRetry);
    if (needRetry) {
      goto retryLock;
    }

    if (parent->items[slot_idx].entry_type != 1 ||
        parent->items[slot_idx].comp.child != node) {
      parent->items[slot_idx].writeUnlock();
      delete_nodes(new_node, 1);
      return false;
    }
    if (node_is_hash(node)) {
        parent->items[slot_idx].writeUnlock();
        delete_nodes(new_node, 1);
        return false;
    }

    parent->items[slot_idx].comp.child = new_node;
    parent->items[slot_idx].writeUnlock();

    safe_delete_nodes(node, 1);

    return true;
  }

   bool compress_node(Node* node) {
    if (!node || node->is_two) {
      return false;
    }
    if (node_is_hash(node)) {
      auto* hash_map = node_hash_map(node);
      if (!hash_map) return false;

      size_t size_before = hash_map->size();
      if (size_before == 0) return false;

      const auto reclaimed_buckets = hash_map->compress();
      if (reclaimed_buckets == 0) return false;

      if (!QUIET) {
        std::cout << "Compressed hash node with " << size_before << " keys" << std::endl;
      }
      return true;
    }

    std::vector<T> keys;
    std::vector<P> values;
    std::vector<Node*> children;
    std::vector<int> types;

    int non_empty_count = 0;
    for (int i = 0; i < node->num_items; ++i) {
      if (node->items[i].entry_type != 0) {
        non_empty_count++;
      }
    }

    if (non_empty_count * 2 > node->num_items) {
      return false;
    }

    keys.reserve(non_empty_count);
    values.reserve(non_empty_count);
    children.reserve(non_empty_count);
    types.reserve(non_empty_count);

    for (int i = 0; i < node->num_items; ++i) {
      bool needRestart = false;
      uint64_t version = node->items[i].readLockOrRestart(needRestart);
      if (needRestart) return false;

      uint8_t entry_type = node->items[i].entry_type;

      if (entry_type == 2) {
        keys.push_back(node->items[i].comp.data.key);
        values.push_back(node->items[i].comp.data.value);
        children.push_back(nullptr);
        types.push_back(2);

        node->items[i].readUnlockOrRestart(version, needRestart);
        if (needRestart) return false;

      } else if (entry_type == 1) {
        Node* child = node->items[i].comp.child;
        keys.push_back(T());
        values.push_back(P());
        children.push_back(child);
        types.push_back(1);

        node->items[i].readUnlockOrRestart(version, needRestart);
        if (needRestart) return false;
      } else {
        node->items[i].readUnlockOrRestart(version, needRestart);
        if (needRestart) return false;
      }
    }

    if (keys.empty()) return false;

    int new_num_items = keys.size();
    Item* new_item = new_items(new_num_items);

    for (int i = 0; i < new_num_items; ++i) {
      new_item[i].entry_type = types[i];
      if (types[i] == 2) {
        new_item[i].comp.data.key = keys[i];
        new_item[i].comp.data.value = values[i];
      } else {
        new_item[i].comp.child = children[i];
      }
    }

    if (new_num_items > 1) {
      node->model.a = static_cast<double>(new_num_items - 1) /
                      static_cast<double>(new_num_items - 1);
      node->model.b = 0;
    }

    Item* old_items = node->items;
    int old_num_items = node->num_items;

    node->items = new_item;
    node->num_items.store(new_num_items, std::memory_order_release);

    delete_items(old_items, old_num_items);

    if (!QUIET) {
      std::cout << "Compressed regular node from " << old_num_items
               << " to " << new_num_items << " slots" << std::endl;
    }

    return true;
  }

  struct NodeInfo {
    Node* node;
    Node* parent;
    int slot_idx;
    int level;
    T min_key;
    T max_key;
    int num_keys;

    NodeInfo() : node(nullptr), parent(nullptr), slot_idx(-1), level(0),
                 min_key(T()), max_key(T()), num_keys(0) {}
  };

  bool get_node_key_range(Node* node, T& min_key, T& max_key, int& num_keys) {
    if (!node) return false;

    if (node_is_hash(node)) {
      auto* hash_map = node_hash_map(node);
      if (!hash_map) return false;
      num_keys = hash_map->size();
      std::vector<T> temp_keys;
      std::vector<P> temp_values;
      hash_map->get_all_items(temp_keys, temp_values);
      if (temp_keys.empty()) return false;
      min_key = max_key = temp_keys[0];
      for (const auto& key : temp_keys) {
        min_key = std::min(min_key, key);
        max_key = std::max(max_key, key);
      }
      return true;
    }

    bool found = false;
    min_key = std::numeric_limits<T>::max();
    max_key = std::numeric_limits<T>::min();
    num_keys = 0;

    for (int i = 0; i < node->num_items; ++i) {
      if (node->items[i].entry_type == 2) {
        T key = node->items[i].comp.data.key;
        min_key = std::min(min_key, key);
        max_key = std::max(max_key, key);
        num_keys++;
        found = true;
      } else if (node->items[i].entry_type == 1) {
        T child_min, child_max;
        int child_keys;
        if (get_node_key_range(node->items[i].comp.child, child_min, child_max, child_keys)) {
          if (!found) {
            min_key = child_min;
            max_key = child_max;
            found = true;
          } else {
            min_key = std::min(min_key, child_min);
            max_key = std::max(max_key, child_max);
          }
          num_keys += child_keys;
        }
      }
    }

    return found;
  }

  void collect_nodes_in_range(Node* node, Node* parent, int slot_idx, int current_level,
                               T start_key, T end_key, int target_level,
                               std::vector<NodeInfo>& target_nodes) {
    if (!node) return;
    if (node_is_hash(node)) return;

    T node_min, node_max;
    int num_keys;
    if (!get_node_key_range(node, node_min, node_max, num_keys)) {
      return;
    }

    bool overlaps = !(node_max < start_key || node_min > end_key);

    bool fully_inside = (node_min >= start_key && node_max <= end_key);

    if (!overlaps || num_keys < 128) {
      return;
    }

    if (current_level == target_level && fully_inside && parent != nullptr) {
      NodeInfo info;
      info.node = node;
      info.parent = parent;
      info.slot_idx = slot_idx;
      info.level = current_level;
      info.min_key = node_min;
      info.max_key = node_max;
      info.num_keys = num_keys;
      target_nodes.push_back(info);
      return;
    }

    if (current_level < target_level || !overlaps) {
      for (int i = 0; i < node->num_items; ++i) {
        if (node->items[i].entry_type == 1) {
          collect_nodes_in_range(node->items[i].comp.child, node, i,
                                current_level + 1, start_key, end_key,
                                target_level, target_nodes);
        }
      }
    }
  }
public:
  void compress_cold_zone(T end_key) {
    int compressed_count = 0;
    compress_nodes_outside_range(root.load(std::memory_order_acquire), end_key,
                                 compressed_count);

    std::cout << "Compressed nodes: " << compressed_count << std::endl;
    std::cout << "==================================\n" << std::endl;
  }

  void compress_nodes_outside_range(Node* node, T end_key, int& count) {
    if (!node) return;

    T node_min, node_max;
    int num_keys;
    if (!get_node_key_range(node, node_min, node_max, num_keys)) {
      return;
    }

    if (node_max < end_key) {
      if (compress_node(node)) {
        count++;
      }
    }

    if (node_is_hash(node)) {
      return;
    }

    for (int i = 0; i < node->num_items; ++i) {
      if (node->items[i].entry_type == 1) {
        compress_nodes_outside_range(node->items[i].comp.child,
                                     end_key, count);
      }
    }
  }

public:
  void optimize_targeted_zones(T start_key, T end_key, int target_level = 1) {
    EpochGuard guard(this);
    std::unique_lock<std::mutex> maintenance_lock(maintenance_mutex_);
    MaintenanceBarrierGuard maintenance_barrier(this);
    std::cout << "\n=== Starting Zone Optimization ===" << std::endl;
    std::cout << "Range: [" << start_key << ", " << end_key << "]" << std::endl;
    std::cout << "Target level: " << target_level << std::endl;

    std::vector<NodeInfo> target_nodes;
    collect_nodes_in_range(root.load(std::memory_order_acquire), nullptr, 0, 0,
                          start_key, end_key,
                          target_level, target_nodes);

    std::cout << "Found " << target_nodes.size() << " nodes to convert" << std::endl;

    int converted_count = 0;
    int failed_count = 0;
    int skipped_count = 0;
    size_t total_keys = 0;

    for (const auto& info : target_nodes) {
      total_keys += info.num_keys;

      if (convert_node_to_hash(info.node, info.parent, info.slot_idx)) {
        converted_count++;
      } else {
        failed_count++;
      }
    }

    std::cout << "\n=== Zone Optimization Summary ===" << std::endl;
    std::cout << "Total nodes found: " << target_nodes.size() << std::endl;
    std::cout << "Successfully converted: " << converted_count << std::endl;
    std::cout << "Skipped (too small): " << skipped_count << std::endl;
    std::cout << "Failed conversions: " << failed_count << std::endl;
    std::cout << "Total keys in converted nodes: " << total_keys << std::endl;
    std::cout << "==================================\n" << std::endl;
  }

private:
  bool collect_node_data_locked(Node* node, std::vector<T>& keys,
                                std::vector<P>& values,
                                std::vector<Item*>& locked_items) {
    if (node == nullptr) {
      return false;
    }
    if (node_is_hash(node)) {
      auto* hash_map = node_hash_map(node);
      if (hash_map == nullptr) {
        return false;
      }
      hash_map->get_all_items(keys, values);
      return true;
    }

    const auto slots = node->num_items.load(std::memory_order_acquire);
    for (int slot = 0; slot < slots; ++slot) {
      bool restart = false;
      node->items[slot].writeLockOrRestart(restart);
      if (restart) {
        return false;
      }
      locked_items.push_back(&node->items[slot]);
    }
    for (int slot = 0; slot < slots; ++slot) {
      const auto type = node->items[slot].entry_type;
      if (type == 2) {
        keys.push_back(node->items[slot].comp.data.key);
        values.push_back(node->items[slot].comp.data.value);
      } else if (type == 1 &&
                 !collect_node_data_locked(node->items[slot].comp.child, keys,
                                           values, locked_items)) {
        return false;
      }
    }
    return true;
  }

  static void unlock_items(std::vector<Item*>& locked_items,
                           bool obsolete = false) {
    for (auto iterator = locked_items.rbegin(); iterator != locked_items.rend();
         ++iterator) {
      if (obsolete) {
        (*iterator)->writeUnlockObsolete();
      } else {
        (*iterator)->writeUnlock();
      }
    }
    locked_items.clear();
  }

  bool scan_and_collect_fast(Node* node, std::vector<T>& keys_out, std::vector<P>& values_out) {
    keys_out.clear();
    values_out.clear();

    const int MAX_RETRIES = 5;
    for (int retry = 0; retry < MAX_RETRIES; ++retry) {
      if (retry > 0) {
        keys_out.clear();
        values_out.clear();
        std::this_thread::yield();
      }

      if (collect_node_data_optimistic(node, keys_out, values_out)) {
        return true;
      }
    }
    return false;
  }

  bool convert_to_hash_with_data(Node* node, std::vector<T>& keys, std::vector<P>& values,
                                 Node* parent, int slot_idx) {
    auto* metadata = node_metadata(node);
    if (!node || !metadata || !parent || slot_idx < 0) return false;
    const int num_items = static_cast<int>(keys.size());
    if (num_items < 64) {
      RT_DEBUG("Node too small for hash (%d items)", num_items);
      return false;
    }

    Node* new_node = new_nodes(1);
    auto* hash_map = new HashNodeWrapper<T, P>();
    const auto requested_capacity =
        metadata->reserved_hash_capacity.load(std::memory_order_acquire);
    hash_map->reserve(
        std::max<std::size_t>(keys.size(), requested_capacity));
    hash_map->batch_insert(keys, values);

    new_node->build_size = num_items;
    set_node_fixed(new_node, true);
    new_node->is_two = 0;
    new_node->num_items = 0;
    new_node->size = num_items;
    new_node->num_inserts = 0;
    new_node->num_insert_to_data = 0;
    new_node->items = reinterpret_cast<Item*>(hash_map);
    new_node->model.a = new_node->model.b = 0;
    const auto depth_before = get_subtree_depth(node);
    auto* new_metadata = attach_runtime_metadata(
        new_node,
        metadata->layer_depth_value(),
        metadata, static_cast<std::uint32_t>(depth_before), 1, true,
        parent, slot_idx);
    new_metadata->is_hash_node.store(true, std::memory_order_release);
    new_metadata->current_capacity.store(hash_map->bucket_count(),
                                         std::memory_order_relaxed);
    set_node_hash(new_node);
    g_lipp_controller.notify_hash_node_created();

    int retryLockCount = 0;
  retryLock:
    if (retryLockCount++)
      yield(retryLockCount);

    bool needRetry = false;
    parent->items[slot_idx].writeLockOrRestart(needRetry);
    if (needRetry) {
      RT_DEBUG("Final step of adjust, obtain parent %p lock FAIL, retry", parent);
      goto retryLock;
    }

    if (parent->items[slot_idx].entry_type != 1 ||
        parent->items[slot_idx].comp.child != node) {
      parent->items[slot_idx].writeUnlock();
      delete_nodes(new_node, 1);
      return false;
    }

    parent->items[slot_idx].comp.child = new_node;
    parent->items[slot_idx].writeUnlock();

    metadata->mark_deleted();
    adjustsuccess.fetch_add(1, std::memory_order_relaxed);
    RT_DEBUG("Adjusted success=%d",
             adjustsuccess.load(std::memory_order_relaxed));

    return true;
  }

  std::size_t subtree_storage_size(Node* subtree) const {
    if (subtree == nullptr) return 0;
    std::size_t bytes = 0;
    std::vector<Node*> pending{subtree};
    while (!pending.empty()) {
      Node* current = pending.back();
      pending.pop_back();
      bytes += sizeof(*current);
      if (node_is_hash(current)) {
        const auto* hash_map = node_hash_map(current);
        if (hash_map != nullptr) {
          bytes += hash_map->bucket_count() * (sizeof(T) + sizeof(P) + 1);
        }
        continue;
      }
      const int slots = current->num_items.load(std::memory_order_relaxed);
      bytes += static_cast<std::size_t>(slots) * sizeof(Item);
      for (int slot = 0; slot < slots; ++slot) {
        if (current->items[slot].entry_type == 1) {
          pending.push_back(current->items[slot].comp.child);
        }
      }
    }
    return bytes;
  }

  bool try_rebuild_with_data(Node* node, std::vector<T>& keys,
                             std::vector<P>& values, Node* parent,
                             int slot_idx, int current_depth,
                             bool record_structural_change = true,
                             bool require_space_reduction = false) {
    if (keys.empty()) {
      return false;
    }
    if (keys.size() < 4) {
      return false;
    }

    if (!std::is_sorted(keys.begin(), keys.end())) {
      std::vector<std::pair<T, P>> pack(keys.size());
      for(size_t i = 0; i < keys.size(); ++i) pack[i] = {keys[i], values[i]};
      std::sort(pack.begin(), pack.end());
      for(size_t i = 0; i < keys.size(); ++i) {
        keys[i] = pack[i].first;
        values[i] = pack[i].second;
      }
    }

    int depth_before = get_subtree_depth(node);
    Node *new_node = build_tree_bulk(keys.data(), values.data(), keys.size(),
                                     require_space_reduction);

    if (!new_node) {
      return false;
    }

    std::size_t reclaimed_bytes = 0;
    if (require_space_reduction) {
      const auto old_bytes = subtree_storage_size(node);
      const auto new_bytes = subtree_storage_size(new_node);
      if (old_bytes == 0 || new_bytes >= old_bytes) {
        destroy_tree(new_node);
        return false;
      }
      reclaimed_bytes = old_bytes - new_bytes;
    }

    int depth_after = get_subtree_depth(new_node);
    auto* previous = node_metadata(node);
    auto* replacement_metadata = attach_runtime_metadata(
        new_node, static_cast<std::uint32_t>(current_depth), previous,
        static_cast<std::uint32_t>(depth_before),
        static_cast<std::uint32_t>(depth_after), record_structural_change,
        parent, slot_idx);
    if (replacement_metadata != nullptr && require_space_reduction) {
      replacement_metadata->set_compacted_layout(true);
    }
    const auto slots = new_node->num_items.load(std::memory_order_acquire);
    for (int slot = 0; slot < slots; ++slot) {
      if (new_node->items[slot].entry_type == 1) {
        register_subtree_runtime(new_node->items[slot].comp.child, new_node,
                                 slot,
                                 static_cast<std::uint32_t>(current_depth + 1),
                                 require_space_reduction);
      }
    }

    if (parent == nullptr) {
      Node* expected_root = node;
      if (!root.compare_exchange_strong(expected_root, new_node,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        destroy_tree(new_node);
        return false;
      }
    } else {
      if (slot_idx < 0) {
        destroy_tree(new_node);
        return false;
      }
      int retryLockCount = 0;
    retryLock:
      if (retryLockCount++)
        yield(retryLockCount);
      bool needRetry = false;
      parent->items[slot_idx].writeLockOrRestart(needRetry);
      if (needRetry) {
        goto retryLock;
      }
      if (parent->items[slot_idx].entry_type != 1 ||
          parent->items[slot_idx].comp.child != node) {
        parent->items[slot_idx].writeUnlock();
        destroy_tree(new_node);
        return false;
      }
      parent->items[slot_idx].comp.child = new_node;
      parent->items[slot_idx].writeUnlock();
    }

    if (previous != nullptr) {
      previous->mark_deleted();
    }
    if (reclaimed_bytes != 0) {
      g_lipp_controller.record_cooling_savings(reclaimed_bytes);
    }
    adjustsuccess.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  int get_subtree_depth(Node* node) {
    if (!node || node_is_hash(node)) return 1;
    int max_d = 0;
    for (int i = 0; i < node->num_items; ++i) {
      if (node->items[i].entry_type == 1) {
        int d = get_subtree_depth(node->items[i].comp.child);
        if (d > max_d) max_d = d;
      }
    }
    return 1 + max_d;
  }

  bool process_maintenance(Node* node, Node* parent, int slot_idx, int current_depth) {
    auto* metadata = node_metadata(node);
    if (!node || !metadata || !metadata->is_node_valid()) {
      return false;
    }

    std::unique_lock<std::mutex> maintenance_lock(maintenance_mutex_,
                                                   std::try_to_lock);
    if (!maintenance_lock.owns_lock()) {
      return false;
    }

    const auto requested =
        metadata->state.load(std::memory_order_acquire);
    if (requested != index_controller::NodeState::NEED_REBUILD &&
        requested != index_controller::NodeState::NEED_HASH) {
      return false;
    }
    if (!metadata->try_begin(requested)) {
      return false;
    }

    auto& keys = maintenance_keys_;
    auto& values = maintenance_values_;
    auto& locked_items = maintenance_locked_items_;
    keys.clear();
    values.clear();
    locked_items.clear();
    const auto expected_size = static_cast<std::size_t>(
        metadata->current_size.load(std::memory_order_relaxed));
    if (keys.capacity() < expected_size) {
      keys.reserve(expected_size);
    }
    if (values.capacity() < expected_size) {
      values.reserve(expected_size);
    }
    if (locked_items.capacity() < expected_size) {
      locked_items.reserve(expected_size);
    }

    if (!collect_node_data_locked(node, keys, values, locked_items)) {
      unlock_items(locked_items);
      if (metadata->is_node_valid()) {
        metadata->state.store(requested, std::memory_order_release);
      }
      return false;
    }
    g_lipp_controller.record_foreground_work(requested, keys.size());

    bool success = false;
    if (requested == index_controller::NodeState::NEED_HASH) {
      success = convert_to_hash_with_data(node, keys, values, parent, slot_idx);
      if (!QUIET) {
        if (success) std::cout << "Convert to hash succeeded." << std::endl;
        else std::cout << "Convert to hash failed." << std::endl;
      }
    } else {
      success = try_rebuild_with_data(node, keys, values, parent, slot_idx,
                                      current_depth);
    }

    unlock_items(locked_items, success);
    if (success) {
      safe_delete_tree(node);
    }

    if (!success && metadata->is_node_valid()) {
      metadata->finish_execution(node_is_hash(node));
    }
    set_node_maintenance(node, false);
    g_lipp_controller.record_foreground_result(requested, success);
    g_lipp_controller.complete_foreground_task();

    return success;
  }

  bool compact_cold_node(Node* node,
                         lipp_controller::LIPP_Node_Header* metadata) {
    if (node == nullptr || metadata == nullptr ||
        node_metadata(node) != metadata) {
      return false;
    }
    EpochGuard epoch_guard(this);
    if (node_is_hash(node)) {
      auto* hash_map = node_hash_map(node);
      std::unique_lock<std::mutex> maintenance_lock(maintenance_mutex_,
                                                     std::try_to_lock);
      if (!maintenance_lock.owns_lock()) {
        return false;
      }
      MaintenanceBarrierGuard maintenance_barrier(this);
      if (hash_map == nullptr) {
        return false;
      }
      const auto reclaimed_buckets = hash_map->compress();
      if (reclaimed_buckets == 0) {
        return false;
      }
      metadata->current_capacity.store(hash_map->bucket_count(),
                                       std::memory_order_relaxed);
      g_lipp_controller.record_cooling_savings(
          reclaimed_buckets * (sizeof(T) + sizeof(P) + 1));
      metadata->build_size.store(
          index_controller::RuntimeMetadata::narrow_size(hash_map->size()),
          std::memory_order_relaxed);
      metadata->set_compacted_layout(true);
      return true;
    }
    static thread_local std::vector<T> keys;
    static thread_local std::vector<P> values;
    static thread_local std::vector<Item*> locked_items;
    std::unique_lock<std::mutex> maintenance_lock(maintenance_mutex_,
                                                   std::try_to_lock);
    if (!maintenance_lock.owns_lock()) {
      return false;
    }
    keys.clear();
    values.clear();
    locked_items.clear();
    if (!collect_node_data_locked(node, keys, values, locked_items)) {
      unlock_items(locked_items);
      return false;
    }
    Node* parent = nullptr;
    const auto parent_handle =
        g_lipp_controller.parent_handle_from_handle(
            metadata->registry_handle);
    if (parent_handle != 0) {
      auto* parent_metadata =
          g_lipp_controller.metadata_from_handle(parent_handle);
      if (parent_metadata == nullptr || !parent_metadata->is_node_valid()) {
        unlock_items(locked_items);
        return false;
      }
      parent = static_cast<Node*>(
          g_lipp_controller.cooling_context_from_handle(parent_handle));
    } else if (root.load(std::memory_order_acquire) != node) {
      unlock_items(locked_items);
      return false;
    }
    const int parent_slot = find_child_slot(parent, node);
    if (parent != nullptr && parent_slot < 0) {
      unlock_items(locked_items);
      return false;
    }
    const bool compacted = try_rebuild_with_data(
        node, keys, values, parent, parent_slot,
        static_cast<int>(metadata->layer_depth_value()), false, true);
    unlock_items(locked_items, compacted);
    if (compacted) {
      safe_delete_tree(node);
    }
    return compacted;
  }

  int find_child_slot(Node* parent, Node* child) {
    if (parent == nullptr) {
      return -1;
    }
    for (int attempt = 0; attempt < 8; ++attempt) {
      const int slots = parent->num_items.load(std::memory_order_acquire);
      bool restart = false;
      for (int slot = 0; slot < slots; ++slot) {
        auto& item = parent->items[slot];
        const auto version = item.readLockOrRestart(restart);
        if (restart) {
          break;
        }
        const bool match = item.entry_type == 1 && item.comp.child == child;
        item.readUnlockOrRestart(version, restart);
        if (restart) {
          break;
        }
        if (match) {
          return slot;
        }
      }
      if (!restart) {
        return -1;
      }
      std::this_thread::yield();
    }
    return -1;
  }

  void safe_delete_tree(Node* subtree) {
    if (subtree == nullptr) {
      return;
    }
    std::stack<Node*> nodes;
    nodes.push(subtree);
    while (!nodes.empty()) {
      Node* current = nodes.top();
      nodes.pop();
      if (current == nullptr) {
        continue;
      }
      if (!node_is_hash(current)) {
        const auto slots = current->num_items.load(std::memory_order_relaxed);
        for (int slot = 0; slot < slots; ++slot) {
          if (current->items[slot].entry_type == 1) {
            nodes.push(current->items[slot].comp.child);
          }
        }
      }
      safe_delete_nodes(current, 1);
    }
  }

#endif
  size_t count_data_items(Node* node) const {
    size_t count = 0;
    int num_slots = node->num_items.load();
    for (int i = 0; i < num_slots; ++i) {
      if (node->items[i].entry_type == 2) {
        ++count;
      }
    }
    return count;
  }

// Recursive helper to collect nodes with data items
  void collect_data_nodes(Node* node, std::vector<Node*>& dns) const {
    if (!node) return;
    if (count_data_items(node) > 0) {
      dns.push_back(node);
    }
    int num_slots = node->num_items.load();
    for (int i = 0; i < num_slots; ++i) {
      if (node->items[i].entry_type == 1) {  // child
          collect_data_nodes(node->items[i].comp.child, dns);
      }
    }
  }

  std::vector<T> get_all_keys(Node* node) const {
    std::vector<T> keys;
    int num_slots = node->num_items.load();
    keys.reserve(count_data_items(node));
    for (int i = 0; i < num_slots; ++i) {
      if (node->items[i].entry_type == 2) {
        keys.push_back(node->items[i].comp.data.key);
      }
    }
    return keys;
  }

  std::pair<size_t, size_t> get_largest_dense_region(Node* node) const {
    std::pair<size_t, size_t> max_region = {0, 0};
    size_t current_length = 0;
    size_t start_idx = 0;
    int num_slots = node->num_items.load();
    for (int i = 0; i < num_slots; ++i) {
        if (node->items[i].entry_type == 2) {
            if (current_length == 0) {
                start_idx = i;
            }
            current_length++;
        } else {
            if (current_length > max_region.first) {
                max_region = {current_length, start_idx};
            }
            current_length = 0;
        }
    }
    if (current_length > max_region.first) {
        max_region = {current_length, start_idx};
    }
    return max_region;
  }

  size_t predict_slot(Node* node, T key) const {
        double v = node->model.predict_double(key);
        if (v > std::numeric_limits<int>::max() / 2) {
            return node->num_items.load() - 1;
        }
        if (v < 0) {
            return 0;
        }
        return std::min(static_cast<size_t>(node->num_items.load() - 1), static_cast<size_t>(v));
    }

    // Direct access to model params
    double get_model_slope(Node* node) const {
        return node->model.a;
    }

    double get_model_intercept(Node* node) const {
        return node->model.b;
    }

    // Helper to generate colliding key for target slot
    T generate_colliding_key(Node* node, size_t target_slot, T min_key, T max_key, std::mt19937_64& gen) const {
        double slope = get_model_slope(node);
        double intercept = get_model_intercept(node);
        if (std::abs(slope) < 1e-9) {
            return min_key + (gen() % (max_key - min_key + 1));
        }
        double approx_key = (static_cast<double>(target_slot) - intercept) / slope;
        T base_key = static_cast<T>(std::round(approx_key));
        T perturb = static_cast<T>(gen() % 100) - 50;
        T key = base_key + perturb;
        if (key < min_key) key = min_key;
        if (key > max_key) key = max_key;
        return key;
    }

    void generate_right_biased_keys(std::set<T>& unique_keys, T min_key, size_t num_keys, size_t max_depth, std::mt19937_64& gen) const {
        if (num_keys == 0) return;
        if (max_depth == 0 || num_keys < 100) {
            T k = min_key;
            for (size_t i = 0; i < num_keys; ++i) {
                while (unique_keys.count(k)) ++k;
                unique_keys.insert(k);
                k += 1 + (gen() % 10);  // Small jitter for variety
            }
            return;
        }
        // Skew: 10% left (sparse), 90% right (dense)
        size_t left_keys = num_keys / 10;
        size_t right_keys = num_keys - left_keys;
        // Left: sparse over larger range
        uint64_t left_range = static_cast<uint64_t>(left_keys) * 100ULL;
        T left_max = min_key + static_cast<T>(left_range) - 1;
        generate_right_biased_keys(unique_keys, min_key, left_keys, max_depth - 1, gen);
        // Right: dense over smaller range
        uint64_t right_range = static_cast<uint64_t>(right_keys) * 1ULL;
        T right_min = left_max + static_cast<T>(gen() % 10000ULL) + 1;  // Small random gap
        generate_right_biased_keys(unique_keys, right_min, right_keys, max_depth - 1, gen);
    }

public:
  void generate_attack_keys(
    std::vector<T>& attack_keys,
    T original_min,
    T original_max,
    size_t original_size,
    size_t num_attack_keys = 200000000
  ) const {
    attack_keys.clear();
    attack_keys.reserve(num_attack_keys);
    std::mt19937_64 gen(42);
    std::cout << "\n=== Generating RIGHT-BIASED, DUPLICATE-FREE Attack Keys for LIPP (Collision-Focused) ===\n";

    std::set<T> unique_keys;

    size_t root_budget = 80000;
    size_t generated = 0;
    T base = original_max + 1;
    for (size_t e = 5; generated < root_budget && e < 63; ++e) {
        T key = base + (static_cast<T>(1) << e) + (gen() % (1LL << (e / 2))) + (e * e * e % 1024);
        if (unique_keys.insert(key).second) {
            generated++;
        }
    }
    T tail = base + (static_cast<T>(1) << 63);
    while (generated < root_budget) {
        T key = tail + (gen() % 10000) * (generated % 2 == 0 ? 1 : -1);
        if (key < tail) key = tail + 1;
        if (unique_keys.insert(key).second) {
            generated++;
        }
        tail = key + 1;
    }
    std::cout << "Root OOD keys: " << generated << "\n";

    std::vector<Node*> dns;
    collect_data_nodes(root.load(std::memory_order_acquire), dns);
    if (dns.empty()) return;
    std::sort(dns.begin(), dns.end(), [this](Node* a, Node* b) {
        double density_a = static_cast<double>(this->count_data_items(a)) / a->num_items.load();
        double density_b = static_cast<double>(this->count_data_items(b)) / b->num_items.load();
        return density_a > density_b;
    });

    size_t hot_nodes = std::min<size_t>(8, dns.size());
    size_t remaining = num_attack_keys - unique_keys.size();
    size_t base_per_node = remaining / hot_nodes;
    size_t max_nodes_focus = std::min<size_t>(10, hot_nodes);

    for (size_t ni = 0; ni < hot_nodes; ++ni) {
        size_t current_node_quota = base_per_node;
        if (ni < max_nodes_focus) {
            current_node_quota *= 4;
        }
        auto* node = dns[ni];
        auto keys = get_all_keys(node);
        if (keys.size() < 2) continue;
        std::sort(keys.begin(), keys.end());
        T node_min = keys.front();
        T node_max = keys.back();
        T range = node_max - node_min;

        auto region = get_largest_dense_region(node);
        size_t max_cont_keys = region.first;
        size_t start_idx = region.second;

        if (max_cont_keys <= 0) continue;
        T dense_min = 0;
        T dense_max = 0;
        size_t found = 0;
        for (size_t i = start_idx; i < static_cast<size_t>(node->num_items.load()) && found < max_cont_keys; ++i) {
            if (node->items[i].entry_type == 2) {
                T key = node->items[i].comp.data.key;
                if (found == 0) {
                    dense_min = key;
                }
                dense_max = key;
                found++;
            }
        }
        if (found == 0) continue;

        size_t inserted = 0;
        double poison_ratio = 0.8;
        size_t num_poison = static_cast<size_t>(current_node_quota * poison_ratio);
        std::vector<size_t> target_slots;
        for (size_t s = 0; s < 5; ++s) {
            target_slots.push_back(start_idx + (s * (max_cont_keys / 5)));
        }

        std::vector<T> candidate_poison;
        candidate_poison.reserve(num_poison * 10);
        for (size_t i = 0; i < num_poison * 10; ++i) {
            size_t tgt_slot = target_slots[gen() % target_slots.size()];
            T poison_k = generate_colliding_key(node, tgt_slot, dense_min, dense_max, gen);
            if (poison_k >= node_min && poison_k <= node_max &&
                !std::binary_search(keys.begin(), keys.end(), poison_k)) {
                candidate_poison.push_back(poison_k);
            }
        }

        long long mid = static_cast<long long>(dense_min) + (static_cast<long long>(dense_max) - dense_min) / 2;
        std::sort(candidate_poison.begin(), candidate_poison.end(), [&](T a, T b) {
            long long dev_a = std::abs(static_cast<long long>(a) - mid);
            long long dev_b = std::abs(static_cast<long long>(b) - mid);
            return dev_a > dev_b || (dev_a == dev_b && (static_cast<long long>(a) * a * a % 1000 > static_cast<long long>(b) * b * b % 1000));
        });

        for (auto k : candidate_poison) {
            if (inserted >= current_node_quota) break;
            if (unique_keys.size() >= num_attack_keys) break;
            if (unique_keys.insert(k).second) {
                inserted++;
            }
        }

        if (inserted < current_node_quota) {
            size_t cluster_size = std::min<size_t>(100, current_node_quota - inserted);
            size_t tgt_slot = target_slots[gen() % target_slots.size()];
            for (size_t j = 0; j < cluster_size; ++j) {
                T k = generate_colliding_key(node, tgt_slot, dense_min, dense_max, gen);
                if (!std::binary_search(keys.begin(), keys.end(), k)) {
                    if (unique_keys.insert(k).second) {
                        inserted++;
                    }
                }
                if (inserted >= current_node_quota || unique_keys.size() >= num_attack_keys) break;
            }
        }

        size_t bands[] = {20000, 10000, 5000, 2000, 1000, 500};
        size_t band_idx = 0;
        while (inserted < current_node_quota && band_idx < sizeof(bands)/sizeof(bands[0])) {
            T band_min = node_max - std::max<T>(1, range / bands[band_idx]);
            T k = band_min;
            while (k < node_max && inserted < current_node_quota) {
                if (std::binary_search(keys.begin(), keys.end(), k)) {
                    k++;
                    continue;
                }
                if (unique_keys.size() >= num_attack_keys) break;
                T jitter_key = k + (gen() % 10 - 5);
                if (unique_keys.insert(jitter_key).second) {
                    inserted++;
                }
                k += 1 + (gen() % 3);
            }
            band_idx++;
        }

        if (inserted < current_node_quota && !keys.empty()) {
            for (int gi = static_cast<int>(keys.size()) - 2; gi >= 0; --gi) {  // Start from rightmost gap
                T gap_start = keys[gi] + 1;
                T gap_end = keys[gi + 1] - 1;
                if (gap_start > gap_end) continue;
                uint64_t gap_size = static_cast<uint64_t>(gap_end - gap_start + 1);
                size_t to_add = std::min(static_cast<size_t>(gap_size), current_node_quota - inserted);
                T current_k = gap_end;  // Fill from right end of gap
                for (size_t j = 0; j < to_add; ++j) {
                    if (unique_keys.size() >= num_attack_keys) break;
                    if (unique_keys.insert(current_k).second) {
                        inserted++;
                    } else {
                        // Rare collision; skip
                    }
                    if (current_k == gap_start) break;  // Avoid underflow
                    --current_k;
                    if (inserted >= current_node_quota) break;
                }
                if (inserted >= current_node_quota || unique_keys.size() >= num_attack_keys) break;
            }
        }
    }

    if (unique_keys.size() < num_attack_keys) {
        T filler_base = unique_keys.empty() ? (original_max + 1) : (*std::prev(unique_keys.end()) + 1);  // Start after max so far
        size_t remaining = num_attack_keys - unique_keys.size();
        generate_right_biased_keys(unique_keys, filler_base, remaining, 50, gen);  // Depth 50 for deep potential skew
    }

    attack_keys.assign(unique_keys.begin(), unique_keys.end());

    std::cout << "\n=== Attack Summary for LIPP ===\n";
    std::cout << "Total keys : " << attack_keys.size() << "\n";
    std::cout << "Right OOD : "
            << std::count_if(attack_keys.begin(), attack_keys.end(),
                [&](T k) { return k > original_max; })
            << "\n";
    std::cout << "=============================================\n";
  }


};

}

#endif
