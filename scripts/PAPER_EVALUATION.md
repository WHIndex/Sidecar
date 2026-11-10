# Sidecar paper evaluation

The evaluation uses the GRE Book and Covid datasets with paired Sidecar and
baseline ALEX and LIPP configurations. Controlled and baseline runs for the same seed consume
the same dataset, attack-key file, generated operation distribution, thread
count, and CPU set.

## Quick validation

```bash
./scripts/run_paper_evaluation.sh --smoke
```

Smoke mode reads 100,000 keys from each GRE dataset and runs five million
operations at 1 and 4 threads. It validates the result schema, latency and
memory collection, controller counters, paired execution, and aggregation.

## Full evaluation

```bash
./scripts/run_paper_evaluation.sh --full
```

Full mode verifies and reads all 200,000,000 Book and Covid keys and their
200,000,000 GRE LIPP attack keys. The default matrix is:

- Threads: 1, 2, 4, 8, 16, 32, 64
- Indexes: controlled ALEX and LIPP
- Read/write ratios: 97/3, 50/50, 0/100
- Seeds: 1866, 1867, 1868, 1869, 1870
- Variants: controlled and baseline
- Operations per run: 200,000,000

An optional `monitored` variant uses the controller-enabled index and records
the same runtime metadata while leaving both controller worker threads stopped.
It separates instrumentation and structural-policy effects from background
decision and compaction costs:

```bash
VARIANTS=controlled,monitored,baseline ./scripts/run_paper_evaluation.sh --full
```

Existing nonempty result files are skipped, so the experiment can be resumed.
Set `FORCE=1` to rerun completed cells; previous CSV files are retained with a
timestamped suffix.

The matrix can be narrowed or extended with comma-separated environment
variables:

```bash
THREADS=8,16,32 SEEDS=1866,1867 DATASETS=books INDEXES=lippol \
  ./scripts/run_paper_evaluation.sh --full
```

Set `PIN_CPUS=0` only when CPU affinity is managed externally. The default pins
each run to CPUs `0..threads-1` and enables OpenMP core binding.

## Outputs

Each run stores `stats.csv`, sampled `latency.csv`, and `run.log` below
`results/paper_evaluation/<profile>/raw/<dataset>/<index>`. Aggregation produces:

- `summary.csv`: means, standard deviations, 95% confidence intervals,
  latency percentiles, memory, controller decisions/completions, and exact
  bytes reclaimed by successful cold-node compaction.
- `comparisons.csv`: per-seed paired controlled/baseline speedups, memory
  overhead, p99 ratio, and 95% confidence intervals.
- `validation.csv`: checks that operations, latency, memory, and controller
  inspection data were recorded.

The runner exits unsuccessfully when any requested run fails or any validation
check is missing. A controlled run must last long enough to include at least
one controller inspection cycle.

Existing results can be summarized again without running benchmarks:

```bash
./scripts/run_paper_evaluation.sh --summarize \
  results/paper_evaluation/full
```
