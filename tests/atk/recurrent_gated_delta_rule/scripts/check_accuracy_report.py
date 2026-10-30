"""Fail the shell command when the native ATK accuracy report has failures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from check_atk_result import check_accuracy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--op", default="recurrent_gated_delta_rule")
    parser.add_argument("--since", type=float, default=0)
    args = parser.parse_args()
    result = check_accuracy(args.output_root, args.op)
    report = result.get("xlsx")
    fresh = bool(report) and Path(report).stat().st_mtime >= args.since
    passed = bool(result.get("found")) and fresh and bool(result.get("all_pass"))
    print(json.dumps(dict(report_found=bool(result.get("found")), fresh=fresh,
                          total=result.get("total", 0),
                          executed=result.get("exec_pass", 0),
                          execution_errors=result.get("exec_fail", 0),
                          accuracy_pass=result.get("check_pass", 0),
                          all_pass=passed), ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
