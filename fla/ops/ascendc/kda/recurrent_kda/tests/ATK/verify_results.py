#!/usr/bin/env python3
"""Validate ATK outputs and materialize stable result files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ATK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ATK_DIR / "results"
CASE_JSON = ATK_DIR / "all_recurrent_kda.json"


def _latest_report(require_custom: bool, not_before: float) -> Path:
    candidates = []
    for path in (ATK_DIR / "atk_output").glob("*/report/*.xlsx"):
        if path.stat().st_mtime < not_before:
            continue
        sheets = pd.ExcelFile(path).sheet_names
        if ("npu_0_custom_data" in sheets) == require_custom:
            candidates.append(path)
    if not candidates:
        raise RuntimeError("no matching ATK XLSX report produced by this run")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _assert_success(report: Path) -> pd.DataFrame:
    statistic = pd.read_excel(report, sheet_name="statistic")
    if len(statistic) != 8:
        raise RuntimeError(f"expected 8 ATK cases, got {len(statistic)}")
    if not statistic["运行结果"].eq("SUCCESS").all():
        failed = statistic.loc[statistic["运行结果"] != "SUCCESS", ["编号", "运行结果", "失败原因"]]
        raise RuntimeError(f"ATK has failed cases: {failed.to_dict(orient='records')}")
    failed_cases = pd.read_excel(report, sheet_name="failed cases")
    if not failed_cases.empty:
        raise RuntimeError(f"ATK failed-cases sheet is not empty: {len(failed_cases)}")
    return statistic


def verify_accuracy(not_before: float) -> None:
    metrics_path = RESULTS_DIR / "accuracy_metrics.jsonl"
    if not metrics_path.exists():
        raise RuntimeError("accuracy_metrics.jsonl was not generated")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(rows) != 8 or {row["case_id"] for row in rows} != set(range(8)):
        raise RuntimeError("accuracy metrics must contain exactly case ids 0..7")
    if not all(row["passed"] for row in rows):
        raise RuntimeError("one or more recurrent_kda accuracy cases failed")
    report = _latest_report(require_custom=False, not_before=not_before)
    statistic = _assert_success(report)
    if not statistic["cpu_0_精度通过"].eq(True).all():
        raise RuntimeError("ATK accuracy column contains a failed case")
    shutil.copy2(report, RESULTS_DIR / "atk_accuracy_report.xlsx")
    print("verified accuracy: 8/8 PASS")


def _case_attributes() -> dict[int, tuple[int, str, int]]:
    cases = json.loads(CASE_JSON.read_text())
    result = {}
    for case in cases:
        values = {item["name"]: item["range_values"] for item in case["inputs"]}
        result[int(case["id"])] = (int(values["batch"]), str(values["mode"]), int(values["seed"]))
    return result


def verify_performance(not_before: float) -> None:
    report = _latest_report(require_custom=True, not_before=not_before)
    statistic = _assert_success(report)
    if not statistic["npu_0_custom_data"].notna().all():
        raise RuntimeError("ATK custom performance data is incomplete")
    custom = pd.read_excel(report, sheet_name="npu_0_custom_data").sort_values("编号")
    if len(custom) != 8 or set(custom["编号"].astype(int)) != set(range(8)):
        raise RuntimeError("performance report must contain exactly case ids 0..7")
    attrs = _case_attributes()
    rows = []
    for _, row in custom.iterrows():
        case_id = int(row["编号"])
        batch, mode, seed = attrs[case_id]
        rows.append({
            "case_id": case_id,
            "batch": batch,
            "mode": mode,
            "seed": seed,
            "latency_us": float(row["latency(us)"]),
            "mfu_percent": float(row["calc_utilization(%)"]),
            "mbu_percent": float(row["mem_utilization(%)"]),
            "read_mib": float(row["read_bytes(MB)"]),
            "write_mib": float(row["write_bytes(MB)"]),
            "logical_flops": float(row["calc_flops_power(FLOPs)"]),
        })
    (RESULTS_DIR / "performance_metrics.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )
    shutil.copy2(report, RESULTS_DIR / "atk_performance_report.xlsx")
    print("verified performance: 8/8 SUCCESS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("accuracy", "performance"))
    parser.add_argument("--not-before", type=float, default=0.0)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.task == "accuracy":
        verify_accuracy(args.not_before)
    else:
        verify_performance(args.not_before)


if __name__ == "__main__":
    main()
