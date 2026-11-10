#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


T_CRITICAL_95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def as_float(row, name):
    value = row.get(name, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values):
    return statistics.fmean(values) if values else 0.0


def confidence(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    average = mean(values)
    if len(values) == 1:
        return average, 0.0, average, average
    deviation = statistics.stdev(values)
    degrees = len(values) - 1
    critical = T_CRITICAL_95[degrees] if degrees < len(T_CRITICAL_95) else 1.96
    half_width = critical * deviation / math.sqrt(len(values))
    return average, deviation, average - half_width, average + half_width


def parse_location(stats_path, raw_root):
    relative = stats_path.relative_to(raw_root)
    parts = relative.parts
    if parts[-1] != "stats.csv":
        raise ValueError(f"Unexpected result path: {stats_path}")
    if len(parts) == 7:
        dataset, index, workload, thread_dir, seed_dir, variant, _ = parts
    elif len(parts) == 6:
        dataset, workload, thread_dir, seed_dir, variant, _ = parts
        index = "lippol"
    else:
        raise ValueError(f"Unexpected result path: {stats_path}")
    return {
        "dataset": dataset,
        "index": index,
        "workload": workload,
        "threads": int(thread_dir.removeprefix("t")),
        "seed": int(seed_dir.removeprefix("seed")),
        "variant": variant,
    }


def load_rows(result_root):
    raw_root = result_root / "raw"
    rows = []
    for stats_path in sorted(raw_root.rglob("stats.csv")):
        with stats_path.open(newline="") as handle:
            parsed = list(csv.DictReader(handle))
        if not parsed:
            continue
        row = parsed[-1]
        row.update(parse_location(stats_path, raw_root))
        row["stats_path"] = str(stats_path)
        rows.append(row)
    return rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["index"], row["workload"],
            row["threads"], row["variant"]
        )
        grouped[key].append(row)

    output = []
    for key in sorted(grouped):
        dataset, index, workload, threads, variant = key
        group = grouped[key]
        throughputs = [as_float(row, "throughput") for row in group]
        memories = [as_float(row, "memory_consumption") for row in group]
        p50_values = [as_float(row, "50 percentile") for row in group]
        p99_values = [as_float(row, "99 percentile") for row in group]
        p999_values = [as_float(row, "99.9 percentile") for row in group]
        throughput_mean, throughput_sd, throughput_low, throughput_high = confidence(throughputs)
        memory_mean, memory_sd, memory_low, memory_high = confidence(memories)
        adaptation_runs = sum(
            as_float(row, "hash_completions")
            + as_float(row, "rebuild_completions")
            + as_float(row, "cooling_completions")
            > 0
            for row in group
        )
        output.append(
            {
                "dataset": dataset,
                "index": index,
                "workload": workload,
                "threads": threads,
                "variant": variant,
                "runs": len(group),
                "table_size": int(as_float(group[0], "table_size")),
                "operations_num": int(as_float(group[0], "operations_num")),
                "throughput_mean": throughput_mean,
                "throughput_stddev": throughput_sd,
                "throughput_ci95_low": throughput_low,
                "throughput_ci95_high": throughput_high,
                "memory_mean": memory_mean,
                "memory_stddev": memory_sd,
                "memory_ci95_low": memory_low,
                "memory_ci95_high": memory_high,
                "latency_p50_mean_ns": mean(p50_values),
                "latency_p99_mean_ns": mean(p99_values),
                "latency_p999_mean_ns": mean(p999_values),
                "adaptation_runs": adaptation_runs,
                "inspection_cycles_total": int(sum(as_float(row, "controller_inspection_cycles") for row in group)),
                "hash_decisions_total": int(sum(as_float(row, "hash_decisions") for row in group)),
                "hash_completions_total": int(sum(as_float(row, "hash_completions") for row in group)),
                "rebuild_decisions_total": int(sum(as_float(row, "rebuild_decisions") for row in group)),
                "rebuild_completions_total": int(sum(as_float(row, "rebuild_completions") for row in group)),
                "cooling_decisions_total": int(sum(as_float(row, "cooling_decisions") for row in group)),
                "cooling_completions_total": int(sum(as_float(row, "cooling_completions") for row in group)),
                "cooling_bytes_reclaimed_total": int(sum(as_float(row, "cooling_bytes_reclaimed") for row in group)),
                "index_adjustments_total": int(sum(as_float(row, "index_adjustments") for row in group)),
            }
        )
    return output


def comparisons(rows):
    paired = defaultdict(dict)
    for row in rows:
        key = (
            row["dataset"], row["index"], row["workload"],
            row["threads"], row["seed"]
        )
        paired[key][row["variant"]] = row

    grouped = defaultdict(list)
    requested_pairs = (
        ("controlled_vs_baseline", "controlled", "baseline"),
        ("controlled_vs_monitored", "controlled", "monitored"),
        ("monitored_vs_baseline", "monitored", "baseline"),
    )
    for key, variants in paired.items():
        dataset, index, workload, threads, seed = key
        for comparison, numerator_name, denominator_name in requested_pairs:
            if numerator_name not in variants or denominator_name not in variants:
                continue
            numerator = variants[numerator_name]
            denominator = variants[denominator_name]
            denominator_throughput = as_float(denominator, "throughput")
            denominator_memory = as_float(denominator, "memory_consumption")
            if denominator_throughput <= 0:
                continue
            grouped[(dataset, index, workload, threads, comparison)].append(
                {
                    "seed": seed,
                    "speedup": as_float(numerator, "throughput") / denominator_throughput,
                    "memory_overhead": (
                        as_float(numerator, "memory_consumption") / denominator_memory - 1.0
                        if denominator_memory > 0
                        else 0.0
                    ),
                    "p99_ratio": (
                        as_float(numerator, "99 percentile")
                        / as_float(denominator, "99 percentile")
                        if as_float(denominator, "99 percentile") > 0
                        else 0.0
                    ),
                    "adapted": (
                        as_float(numerator, "hash_completions")
                        + as_float(numerator, "rebuild_completions")
                        + as_float(numerator, "cooling_completions")
                        > 0
                    ),
                }
            )

    output = []
    for key in sorted(grouped):
        dataset, index, workload, threads, comparison = key
        pairs = grouped[key]
        speedups = [pair["speedup"] for pair in pairs]
        average, deviation, low, high = confidence(speedups)
        output.append(
            {
                "dataset": dataset,
                "index": index,
                "workload": workload,
                "threads": threads,
                "comparison": comparison,
                "paired_seeds": len(pairs),
                "speedup_mean": average,
                "speedup_stddev": deviation,
                "speedup_ci95_low": low,
                "speedup_ci95_high": high,
                "throughput_change_percent": (average - 1.0) * 100.0,
                "memory_overhead_percent_mean": mean(
                    [pair["memory_overhead"] * 100.0 for pair in pairs]
                ),
                "p99_ratio_mean": mean([pair["p99_ratio"] for pair in pairs]),
                "adaptation_pairs": sum(pair["adapted"] for pair in pairs),
            }
        )
    return output


def validate(rows):
    output = []
    for row in rows:
        controlled = row["variant"] == "controlled"
        checks = {
            "table_loaded": as_float(row, "table_size") > 0,
            "operations_executed": as_float(row, "elapsed_ns") > 0,
            "memory_recorded": as_float(row, "memory_consumption") > 0,
            "latency_recorded": as_float(row, "latency_sample_count") > 0,
            "controller_inspected": (
                as_float(row, "controller_inspection_cycles") > 0 if controlled else True
            ),
        }
        output.append(
            {
                "dataset": row["dataset"],
                "index": row["index"],
                "workload": row["workload"],
                "threads": row["threads"],
                "seed": row["seed"],
                "variant": row["variant"],
                **{name: int(value) for name, value in checks.items()},
                "all_checks_pass": int(all(checks.values())),
                "stats_path": row["stats_path"],
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.result_root)
    if not rows:
        raise SystemExit(f"No result files found below {args.result_root / 'raw'}")

    summary_rows = summarize(rows)
    comparison_rows = comparisons(rows)
    validation_rows = validate(rows)
    write_csv(
        args.result_root / "summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )
    if comparison_rows:
        write_csv(
            args.result_root / "comparisons.csv",
            list(comparison_rows[0].keys()),
            comparison_rows,
        )
    write_csv(
        args.result_root / "validation.csv",
        list(validation_rows[0].keys()),
        validation_rows,
    )

    failed = sum(not int(row["all_checks_pass"]) for row in validation_rows)
    triggered = sum(int(row["adaptation_runs"]) for row in summary_rows)
    print(f"Loaded {len(rows)} runs")
    print(f"Wrote {args.result_root / 'summary.csv'}")
    print(f"Wrote {args.result_root / 'comparisons.csv'}")
    print(f"Validation failures: {failed}")
    print(f"Groups with completed adaptive maintenance: {triggered}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
