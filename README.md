# Sidecar

## Build

Requirements are a C++17 compiler, CMake 3.14 or newer, Intel oneTBB, and
OpenMP.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target sidecar_benchmark sidecar_baseline
```

The main executable accepts `--key=value` arguments. A typical controlled LIPP
run is:

```bash
./build/sidecar_benchmark \
  --keys_file=/path/to/books \
  --attack_keys_file=/path/to/books_lippol_attack_keys.bin \
  --keys_file_type=binary \
  --index=lippol \
  --read=0.97 --insert=0.03 --delete=0 --update=0 --scan=0 \
  --table_size=1000000 --init_table_ratio=0.5 \
  --operations_num=5000000 --thread_num=16 \
  --sample_distribution=zipf --seed=1866 --num_repeats=1 \
  --latency_sample --latency_sample_ratio=0.001 --memory \
  --output_path=results/stats.csv \
  --latency_output_path=results/latency.csv
```

Use `--index=alexol` for the shift-based implementation. The original ALEX
implementation remains available as `--index=alexorigin` for comparison.

## GRE evaluation

The repository does not duplicate the GRE datasets. By default, the evaluation
runner reads Book and Covid from a sibling `GRE/datasets` directory as the files
`books` and `covid`. Override `DATASET_DIR` when the datasets are stored
elsewhere.

```bash
./scripts/run_paper_evaluation.sh --smoke
./scripts/run_paper_evaluation.sh --full
```

Full mode validates that each dataset and attack-key file contains 200 million
keys, then evaluates ALEX and LIPP across 1--64 threads, three read/write mixes,
and five seeds. It records throughput, sampled latency, memory, controller
decisions, completed adjustments, and 95% confidence intervals. See
[scripts/PAPER_EVALUATION.md](scripts/PAPER_EVALUATION.md) for the complete
matrix and output schema.

## Implementation map

- `include/runtime_metadata.h`: cache-line-aligned shared node header, atomic
  cumulative counters, snapshots, and node state transitions.
- `include/alex_controller.h`: priority scheduling, degradation decisions, and
  cooling for shift-based leaves.
- `include/lipp_controller.h`: priority scheduling, rebuild/hash decisions, and
  cooling for conflict-based subtrees.
- `competitor/alexol`: ALEX integration and local hash-leaf implementation.
- `competitor/lippolc`: LIPP integration, subtree maintenance, and hash nodes.
- `src/benchmark`: common controlled/baseline benchmark driver.

The formula-to-code mapping and concurrency invariants are documented in
[docs/DESIGN.md](docs/DESIGN.md).

Third-party index implementations retain their original notices and licenses
in their respective directories.
