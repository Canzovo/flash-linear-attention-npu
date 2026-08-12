#!/usr/bin/env bash
set -euo pipefail

ATK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-all}"
CASES="${ATK_DIR}/all_recurrent_kda.json"
PERFORMANCE_DATA="${PERFORMANCE_DATA:-5,5,5}"
ATK_TIMEOUT="${ATK_TIMEOUT:-2000}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

cd "${ATK_DIR}"

run_accuracy() {
  atk task \
    -c "${CASES}" \
    -n nodes_accuracy.yaml \
    --task accuracy \
    --bm_device cpu \
    -p executor_recurrent_kda_gpu.py \
    -to "${ATK_TIMEOUT}"
}

run_npu_performance() {
  atk task \
    -c "${CASES}" \
    -n nodes_performance_npu.yaml \
    --task performance_device \
    --performance_data "${PERFORMANCE_DATA}" \
    --enable_custom_data \
    -p executor_recurrent_kda_gpu.py \
    -to "${ATK_TIMEOUT}"
}

run_gpu_performance() {
  atk task \
    -c "${CASES}" \
    -n nodes_performance_gpu.yaml \
    --task performance_device \
    --performance_data "${PERFORMANCE_DATA}" \
    --enable_custom_data \
    -p executor_recurrent_kda_gpu.py \
    -to "${ATK_TIMEOUT}"
}

run_cpu_performance() {
  atk task \
    -c "${CASES}" \
    -n nodes_performance_cpu.yaml \
    --task performance_device \
    --performance_data "${PERFORMANCE_DATA}" \
    --enable_custom_data \
    -p executor_recurrent_kda_gpu.py \
    -to "${ATK_TIMEOUT}"
}

case "${TASK}" in
  accuracy)
    run_accuracy
    ;;
  performance_npu)
    run_npu_performance
    ;;
  performance_gpu)
    run_gpu_performance
    ;;
  performance_cpu)
    run_cpu_performance
    ;;
  performance)
    run_npu_performance
    run_gpu_performance
    run_cpu_performance
    ;;
  all)
    run_accuracy
    run_npu_performance
    run_gpu_performance
    run_cpu_performance
    ;;
  *)
    echo "usage: $0 [accuracy|performance_npu|performance_gpu|performance_cpu|performance|all]" >&2
    exit 2
    ;;
esac
