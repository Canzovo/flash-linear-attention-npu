#!/usr/bin/env bash
set -euo pipefail

ATK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ATK_DIR}/../../../../../../.." && pwd)"
TASK="${1:-all}"

source /home/npu_user6/ziqiyang/CANN0713/ascend-toolkit/set_env.sh
source /home/npu_user6/miniforge3/etc/profile.d/conda.sh
conda activate yzq
export PYTHONPATH="/home/npu_user6/miniforge3/pkgs/requests-2.34.2-pyhcf101f3_0/site-packages:/home/npu_user6/miniforge3/pkgs/urllib3-2.7.0-pyhd8ed1ab_0/site-packages:/home/npu_user6/miniforge3/pkgs/certifi-2026.5.20-pyhd8ed1ab_0/site-packages:/home/npu_user6/miniforge3/pkgs/charset-normalizer-3.4.7-pyhd8ed1ab_0/site-packages:${PYTHONPATH:-}"

export PYTHONPATH="${REPO_ROOT}/tests/reference:/home/npu_user6/ziqiyang/ATK-dev:${PYTHONPATH:-}"
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

