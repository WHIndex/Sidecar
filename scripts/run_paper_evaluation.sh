#!/usr/bin/env bash

set -u
set -o pipefail

usage() {
    echo "Usage: $0 --smoke | --full | --summarize RESULT_DIR"
    echo "Environment overrides: DATASETS INDEXES THREADS SEEDS WORKLOADS VARIANTS OPERATIONS_NUM"
    echo "                       RESULT_DIR BUILD_DIR DATASET_DIR FORCE PIN_CPUS"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODE=$1

if [[ "${MODE}" == "--summarize" ]]; then
    if [[ $# -ne 2 ]]; then
        usage
        exit 2
    fi
    exec python3 "${SCRIPT_DIR}/summarize_paper_results.py" "$2"
fi

if [[ "${MODE}" != "--smoke" && "${MODE}" != "--full" ]]; then
    usage
    exit 2
fi

DATASET_DIR=${DATASET_DIR:-${PROJECT_DIR}/../GRE/datasets}
BUILD_DIR=${BUILD_DIR:-/tmp/sidecar-paper-build}

if [[ "${MODE}" == "--full" ]]; then
    PROFILE=full
    TABLE_SIZE=-1
    OPERATIONS_NUM=${OPERATIONS_NUM:-200000000}
    DATASETS=${DATASETS:-books,covid}
    INDEXES=${INDEXES:-lippol,alexol}
    THREADS=${THREADS:-1,2,4,8,16,32,64}
    SEEDS=${SEEDS:-1866,1867,1868,1869,1870}
    WORKLOADS=${WORKLOADS:-read97_write03,read50_write50,read00_write100}
    LATENCY_SAMPLE_RATIO=${LATENCY_SAMPLE_RATIO:-0.0001}
else
    PROFILE=smoke
    TABLE_SIZE=${TABLE_SIZE:-100000}
    OPERATIONS_NUM=${OPERATIONS_NUM:-5000000}
    DATASETS=${DATASETS:-books,covid}
    INDEXES=${INDEXES:-lippol,alexol}
    THREADS=${THREADS:-1,4}
    SEEDS=${SEEDS:-1866}
    WORKLOADS=${WORKLOADS:-read97_write03}
    LATENCY_SAMPLE_RATIO=${LATENCY_SAMPLE_RATIO:-0.001}
fi

VARIANTS=${VARIANTS:-controlled,baseline}
RESULT_DIR=${RESULT_DIR:-${PROJECT_DIR}/results/paper_evaluation/${PROFILE}}
FORCE=${FORCE:-0}
PIN_CPUS=${PIN_CPUS:-1}

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
IFS=',' read -r -a INDEX_LIST <<< "${INDEXES}"
IFS=',' read -r -a THREAD_LIST <<< "${THREADS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a WORKLOAD_LIST <<< "${WORKLOADS}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"

mkdir -p "${RESULT_DIR}/raw" "${RESULT_DIR}/cache" "${BUILD_DIR}"
MAX_THREADS=$(getconf _NPROCESSORS_ONLN)

for dataset in "${DATASET_LIST[@]}"; do
    data_path="${DATASET_DIR}/${dataset}"
    if [[ ! -r "${data_path}" ]]; then
        echo "Dataset is not readable: ${data_path}" >&2
        exit 1
    fi
    if [[ "${MODE}" == "--full" ]]; then
        attack_path="${DATASET_DIR}/${dataset}_lippol_attack_keys.bin"
        if [[ ! -r "${attack_path}" ]]; then
            echo "Attack-key file is not readable: ${attack_path}" >&2
            exit 1
        fi
        data_count=$(od -An -tu8 -N8 "${data_path}" | tr -d ' ')
        attack_count=$(od -An -tu8 -N8 "${attack_path}" | tr -d ' ')
        if [[ "${data_count}" != "200000000" || "${attack_count}" != "200000000" ]]; then
            echo "Full mode requires 200000000 dataset and attack keys for ${dataset}" >&2
            exit 1
        fi
    fi
done

cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release || exit 1
cmake --build "${BUILD_DIR}" -j"${MAX_THREADS}" \
    --target sidecar_benchmark sidecar_baseline || exit 1

CONTROLLED_BIN="${BUILD_DIR}/sidecar_benchmark"
BASELINE_BIN="${BUILD_DIR}/sidecar_baseline"

{
    echo "profile=${PROFILE}"
    echo "datasets=${DATASETS}"
    echo "indexes=${INDEXES}"
    echo "threads=${THREADS}"
    echo "seeds=${SEEDS}"
    echo "workloads=${WORKLOADS}"
    echo "variants=${VARIANTS}"
    echo "table_size=${TABLE_SIZE}"
    echo "operations_num=${OPERATIONS_NUM}"
    echo "latency_sample_ratio=${LATENCY_SAMPLE_RATIO}"
    echo "pin_cpus=${PIN_CPUS}"
    uname -a
    lscpu
} > "${RESULT_DIR}/environment.txt"

workload_ratios() {
    case "$1" in
        read97_write03) echo "0.97 0.03 0.5" ;;
        read50_write50) echo "0.50 0.50 0.5" ;;
        read00_write100) echo "0.00 1.00 0.5" ;;
        read100_write00) echo "1.00 0.00 1.0" ;;
        *)
            echo "Unknown workload: $1" >&2
            return 1
            ;;
    esac
}

variant_order() {
    local seed=$1
    if [[ ${#VARIANT_LIST[@]} -eq 2 && "${VARIANTS}" == "controlled,baseline" ]]; then
        if (( seed % 2 == 0 )); then
            echo "controlled baseline"
        else
            echo "baseline controlled"
        fi
    else
        echo "${VARIANT_LIST[*]}"
    fi
}

failed_runs=0
completed_runs=0
skipped_runs=0

for dataset in "${DATASET_LIST[@]}"; do
    keys_file="${DATASET_DIR}/${dataset}"

    for index in "${INDEX_LIST[@]}"; do
        if [[ "${index}" != "lippol" && "${index}" != "alexol" ]]; then
            echo "Unknown controlled index: ${index}" >&2
            exit 1
        fi
        if [[ "${MODE}" == "--full" ]]; then
            attack_keys_file="${DATASET_DIR}/${dataset}_lippol_attack_keys.bin"
        else
            attack_keys_file="${RESULT_DIR}/cache/${dataset}_${index}_t${TABLE_SIZE}_o${OPERATIONS_NUM}.bin"
        fi
    for workload in "${WORKLOAD_LIST[@]}"; do
        read -r read_ratio insert_ratio init_ratio <<< "$(workload_ratios "${workload}")"
        for threads in "${THREAD_LIST[@]}"; do
            if (( threads < 1 || threads > MAX_THREADS )); then
                echo "Skipping unsupported thread count ${threads}"
                continue
            fi
            for seed in "${SEED_LIST[@]}"; do
                read -r -a ordered_variants <<< "$(variant_order "${seed}")"
                for variant in "${ordered_variants[@]}"; do
                    run_dir="${RESULT_DIR}/raw/${dataset}/${index}/${workload}/t${threads}/seed${seed}/${variant}"
                    stats_file="${run_dir}/stats.csv"
                    latency_file="${run_dir}/latency.csv"
                    log_file="${run_dir}/run.log"
                    mkdir -p "${run_dir}"

                    if [[ -s "${stats_file}" && "${FORCE}" != "1" ]]; then
                        echo "[skip] ${dataset} ${index} ${workload} t=${threads} seed=${seed} ${variant}"
                        skipped_runs=$((skipped_runs + 1))
                        continue
                    fi

                    if [[ "${variant}" == "controlled" || "${variant}" == "monitored" ]]; then
                        executable="${CONTROLLED_BIN}"
                    elif [[ "${variant}" == "baseline" ]]; then
                        executable="${BASELINE_BIN}"
                    else
                        echo "Unknown variant: ${variant}" >&2
                        exit 1
                    fi

                    command=(
                        "${executable}"
                        "--keys_file=${keys_file}"
                        "--attack_keys_file=${attack_keys_file}"
                        "--keys_file_type=binary"
                        "--read=${read_ratio}"
                        "--insert=${insert_ratio}"
                        "--delete=0"
                        "--update=0"
                        "--scan=0"
                        "--operations_num=${OPERATIONS_NUM}"
                        "--table_size=${TABLE_SIZE}"
                        "--init_table_ratio=${init_ratio}"
                        "--thread_num=${threads}"
                        "--index=${index}"
                        "--sample_distribution=zipf"
                        "--seed=${seed}"
                        "--num_repeats=1"
                        "--latency_sample"
                        "--latency_sample_ratio=${LATENCY_SAMPLE_RATIO}"
                        "--memory"
                        "--output_path=${stats_file}"
                        "--latency_output_path=${latency_file}"
                    )
                    if [[ "${MODE}" == "--smoke" && ! -s "${attack_keys_file}" ]]; then
                        command+=("--generate_attack_keys")
                    fi
                    if [[ "${variant}" == "monitored" ]]; then
                        command+=("--disable_controller")
                    fi

                    echo "[run]  ${dataset} ${index} ${workload} t=${threads} seed=${seed} ${variant}"
                    if [[ -e "${stats_file}" ]]; then
                        mv "${stats_file}" "${stats_file}.previous.$(date +%s)"
                    fi
                    if [[ -e "${latency_file}" ]]; then
                        mv "${latency_file}" "${latency_file}.previous.$(date +%s)"
                    fi

                    export OMP_NUM_THREADS=${threads}
                    export OMP_PROC_BIND=close
                    export OMP_PLACES=cores
                    if [[ "${PIN_CPUS}" == "1" ]]; then
                        cpu_end=$((threads - 1))
                        taskset -c "0-${cpu_end}" "${command[@]}" > "${log_file}" 2>&1
                    else
                        "${command[@]}" > "${log_file}" 2>&1
                    fi
                    status=$?
                    if [[ ${status} -ne 0 || ! -s "${stats_file}" ]]; then
                        echo "[fail] ${dataset} ${index} ${workload} t=${threads} seed=${seed} ${variant}; log=${log_file}" >&2
                        failed_runs=$((failed_runs + 1))
                    else
                        completed_runs=$((completed_runs + 1))
                    fi
                done
            done
        done
    done
    done
done

python3 "${SCRIPT_DIR}/summarize_paper_results.py" "${RESULT_DIR}"
summary_status=$?

echo "Completed runs: ${completed_runs}"
echo "Skipped runs:   ${skipped_runs}"
echo "Failed runs:    ${failed_runs}"
echo "Results:        ${RESULT_DIR}"

if [[ ${failed_runs} -ne 0 || ${summary_status} -ne 0 ]]; then
    exit 1
fi
