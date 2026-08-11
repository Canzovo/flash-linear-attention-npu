#!/usr/bin/env bash
set -euo pipefail

ATK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ATK_DIR}/../../../../../../.." && pwd)"
TASK="${1:-all}"

export RECURRENT_KDA_METRICS_FILE="${ATK_DIR}/results/accuracy_metrics.jsonl"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

mkdir -p "${ATK_DIR}/results"
cd "${ATK_DIR}"


CASES="${ATK_DIR}/all_recurrent_kda.json"

run_accuracy() {
  rm -f "${RECURRENT_KDA_METRICS_FILE}"
  local started_at
  started_at="$(date +%s)"
  atk task \
    -c "${CASES}" \
    -n nodes_accuracy.yaml \
    --task accuracy \
    -p executor_recurrent_kda.py \
    -sp \
    -to 2000
  python verify_results.py accuracy --not-before "${started_at}"
}

run_performance() {
  local started_at
  started_at="$(date +%s)"
  atk task \
    -c "${CASES}" \
    -n nodes_performance.yaml \
    --task performance_device \
    --performance_data 20,20,10 \
    --enable_custom_data \
    -p executor_recurrent_kda.py \
    -sp \
    -to 2000
  python verify_results.py performance --not-before "${started_at}"
}

case "${TASK}" in
  accuracy)
    run_accuracy
    ;;
  performance)
    run_performance
    ;;
  all)
    run_accuracy
    run_performance
    ;;
  *)
    echo "usage: $0 [accuracy|performance|all]" >&2
    exit 2
    ;;
esac

