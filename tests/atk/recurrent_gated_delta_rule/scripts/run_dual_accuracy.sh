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
# ATK schedules the FP64 golden invocation automatically on bm_device.
# The ordinary CPU invocation below is the independent FP32 benchmark.
atk node --name npu_dut --backend npu --devices "$DEVICE" --is_bm False \
      --output_path "$OUTPUT" \
    node --name cpu_bench --backend cpu --is_bm True \
      --output_path "$OUTPUT" \
    task -c ./atk_recurrent_gated_delta_rule.json \
      -p ./executor_recurrent_gated_delta_rule.py \
      --task "$TASK" --bm_device cpu -to "${ATK_TIMEOUT:-2000}" "$@"
python "$OP_DIR/scripts/check_accuracy_report.py" \
    --output-root "$OUTPUT_ROOT" --since "$STARTED"
