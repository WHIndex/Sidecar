# Sidecar design

## Monitoring model

Every monitored node owns an independent `RuntimeMetadata` header aligned to a
64-byte cache line. The index operation paths update only cumulative atomic
counters and timestamps. The controllers read those counters periodically and
keep the previous values in registry slots, so a cycle obtains:

```text
delta_conflict = conflict_total - conflict_snapshot
delta_write    = insert_total   - insert_snapshot
delta_read     = lookup_total   - lookup_snapshot
priority       = wc * delta_conflict
               + ww * delta_write
               + wr * delta_read
```

For ALEX, `delta_conflict` is the exact number of keys moved by shifts and only
leaf nodes are registered. For LIPP, it is the number of prediction conflicts
and every nonempty node on an operation path is registered. SMO counter
baselines are separate from periodic snapshots.

Each decision cycle fully sorts all `NORMAL` candidates in descending priority.
It selects at most `K` nodes from the high-priority end and at most `K` disjoint
nodes from the low-priority end. The resulting cycle complexity is
`O(N log N + K)` time and `O(N + |P|)` auxiliary space, including the cooling
pool `P`.

## Degradation decisions

An ALEX leaf is scheduled as `NEED_HASH` only when all conditions hold:

```text
last_smo - previous_smo <= fast_smo_interval
post_smo_shifts / post_smo_inserts >= high_shift_per_insert
(capacity - size) / capacity >= minimum_gap_ratio
post_smo_inserts > 0
```

A LIPP node is scheduled as `NEED_HASH` only when all conditions hold:

```text
last_smo - previous_smo <= fast_smo_interval
post_smo_conflicts / post_smo_inserts >= hash_conflict_ratio
depth_after / depth_before >= minimum_depth_ratio
node_depth >= minimum_hash_depth
post_smo_inserts > 0
```

If the LIPP hash conditions are false, regular rebuilding is checked using:

```text
current_size >= rebuild_min_size
inserts_since_build >=
    (rebuild_size_multiplier - 1 - rebuild_size_tolerance) * build_size
cycle_conflicts / cycle_inserts >=
    rebuild_conflict_ratio - rebuild_conflict_tolerance
cycle_inserts > 0
```

The reserved hash capacity is:

```text
min(ceil(current_size + cycle_insert_rate * horizon),
    maximum_hash_growth * current_size)
```

Controllers publish decisions only. The next foreground operation reaching the
target changes `NEED_HASH` or `NEED_REBUILD` to `IS_EXECUTING`, performs the
replacement, retires the old node as `DELETED`, and continues on the new
structure. LIPP hash conversion is restricted to sufficiently deep subtrees;
ALEX conversion changes only the target leaf.

## Cold-node compaction

A low-priority node enters `IN_COOLING` only when both the inactivity threshold
and the current-cycle sparse-access threshold hold. Cooling uses a separate
thread and queue, processing at most `B` nodes per cooling cycle. The worker
verifies that the access baseline is unchanged before it transitions the node
to `IS_EXECUTING`; the LIPP path also cancels queued cooling when an access
reaches the node.

Compaction reports success only when storage capacity actually decreases. The
node then returns to `NORMAL`, stays registered, and participates in later
snapshots and priority ordering.

## State and lifetime rules

The state machine is:

```text
NORMAL -> NEED_HASH    -> IS_EXECUTING -> NORMAL or DELETED
NORMAL -> NEED_REBUILD -> IS_EXECUTING -> NORMAL or DELETED
NORMAL -> IN_COOLING   -> IS_EXECUTING -> NORMAL
```

Registry handles remain stable while their metadata is live. Controller
callbacks pin metadata before releasing registry protection. Retirement marks
metadata invalid before reclamation and waits for all pinned callbacks, so a
background decision cannot dereference a reclaimed header.
