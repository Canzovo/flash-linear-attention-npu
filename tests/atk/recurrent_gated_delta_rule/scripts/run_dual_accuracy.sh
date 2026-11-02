#!/usr/bin/env bash
set -euo pipefail
OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-0}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$OP_DIR"
OUTPUT_ROOT="${ATK_OUTPUT_ROOT:-./atk_output}"
OUTPUT="$OUTPUT_ROOT/accuracy"
mkdir -p "$OUTPUT"
STARTED="$(date +%s)"
TASK="${ATK_TASK:-accuracy}"
CASE_FILE="${ATK_CASE_FILE:-}"
if [[ -z "$CASE_FILE" ]]; then
    if [[ "$TASK" == accuracy ]]; then
        CASE_FILE="$OUTPUT/atk_recurrent_gated_delta_rule_random20.json"
        SAMPLE_ARGS=()
        if [[ -n "${ATK_SAMPLE_SEED:-}" ]]; then
            SAMPLE_ARGS=(--seed "$ATK_SAMPLE_SEED")
        fi
        python "$OP_DIR/scripts/sample_accuracy_cases.py" \
            --source "$OP_DIR/atk_recurrent_gated_delta_rule.json" \
            --output "$CASE_FILE" --count 20 "${SAMPLE_ARGS[@]}"
    else
        # accuracy_lt uses the explicitly selected failure IDs in the full pool.
        CASE_FILE="$OP_DIR/atk_recurrent_gated_delta_rule.json"
    fi
fi
# ATK schedules the FP64 golden invocation automatically on bm_device.
# The ordinary CPU invocation below is the independent FP32 benchmark.
atk node --name npu_dut --backend npu --devices "$DEVICE" --is_bm False \
      --output_path "$OUTPUT" \
    node --name cpu_bench --backend cpu --is_bm True \
      --output_path "$OUTPUT" \
    task -c "$CASE_FILE" \
      -p ./executor_recurrent_gated_delta_rule.py \
      --task "$TASK" --bm_device cpu -to "${ATK_TIMEOUT:-2000}" "$@"
python "$OP_DIR/scripts/check_accuracy_report.py" \
    --output-root "$OUTPUT_ROOT" --since "$STARTED"
