#!/usr/bin/env python3
"""Compare the legacy and V2 NPU kernels with all four fusion flags False.

Uses the same 36 shapes and both State dtypes as benchmark_recurrent_gdn.py.
Every pair shares its input tensors and seed. Each call restores initial State
outside the target kernel; timings come only from msprof Task Duration(us).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from decimal import Decimal
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import benchmark_recurrent_gdn as benchmark


def comparison_cases() -> list[dict]:
    return [dict(case, comparison_id=index) for index, case in enumerate(
        case for case in benchmark.matrix("both")
        if not any(case[name] for name in benchmark.FLAG_NAMES))]


def run_worker(args, cases: list[dict]) -> None:
    import torch
    import torch_npu
    from fla_npu.ops import ascendc
    from fla_npu.ops.ascendc import _aclnn_ctypes as legacy

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    operator = ascendc.recurrent_gated_delta_rule
    source_hash = lambda obj: hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest()
    benchmark.write_json(args.output / "environment.json", dict(
        torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
        device_name=torch.npu.get_device_name(args.device),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_generator_sha256=hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
        v1_bridge_sha256=source_hash(legacy._call_aclnn), v2_wrapper_sha256=source_hash(operator),
        flags={name: False for name in benchmark.FLAG_NAMES}, warmups=args.warmups, samples=1,
        timing_source="msprof op_summary Task Duration(us)",
        pair_order="v1,v2 for even comparison_id; v2,v1 for odd comparison_id",
        inputs="BF16 Q/K/V/beta, FP32 log g, BF16/FP32 initial State; no CPU fusion preprocessing",
        state_pool="T slots; indices=range(T); restore initial slots=b*TOKEN_PER_BATCH before each call",
        both_versions_write_every_token_checkpoint=True,
        initial_state_reset_and_input_generation_excluded=True, formal_accuracy_test=False))
    records = []
    with torch.no_grad(), (args.output / "launches.jsonl").open("x", encoding="utf-8") as progress:
        for case in cases:
            data, hashes = benchmark.make_inputs(torch, case, torch.device("npu", args.device))
            hashes = benchmark.input_hashes(hashes, case)
            hashes["actual_seq_lengths"] = benchmark.tensor_hash(
                torch.tensor([0] + [case["TOKEN_PER_BATCH"]] * case["BS"], dtype=torch.int32))
            hashes["ssm_state_indices"] = benchmark.tensor_hash(torch.arange(case["T"], dtype=torch.int32))
            order = ("v1", "v2") if case["comparison_id"] % 2 == 0 else ("v2", "v1")
            for version in order:
                for _ in range(args.warmups + 1):
                    data["state"][::case["TOKEN_PER_BATCH"]].copy_(data["initial_state"])
                    # NaNs make a legacy empty return observable; TPB is at most 8.
                    output = torch.full_like(data["v"], float("nan")) if version == "v1" else None
                    torch.npu.synchronize()
                    if version == "v1":
                        def build(ctx):
                            def desc(tensor, name):
                                return ctx.tensor(tensor, name, acl_format_override=legacy.ACL_FORMAT_ND,
                                                  storage_shape_override=tuple(tensor.shape))
                            return [desc(data["q"], "query"), desc(data["k"], "key"),
                                desc(data["v"], "value"), desc(data["beta_post"], "beta"),
                                desc(data["state"], "state"), desc(data["lengths"], "actual_seq_lengths"),
                                desc(data["indices"], "ssm_state_indices"), desc(data["g_log"], "g"),
                                None, None, ctypes.c_float(case["scale"]), desc(output, "out")]
                        legacy._call_aclnn("aclnnRecurrentGatedDeltaRule", build, output)
                    else:
                        output = operator(data["q"], data["k"], data["v"], data["state"],
                            beta=data["beta_post"], g=data["g_log"], scale=case["scale"],
                            actual_seq_lengths=data["lengths"], ssm_state_indices=data["indices"],
                            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                            use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False)
                        if ascendc.BACKENDS.get("npu_recurrent_gated_delta_rule") != "stable":
                            raise RuntimeError("The V2 Stable ABI was not selected.")
                    torch.npu.synchronize()
                    if tuple(output.shape) != (case["T"], case["HV"], case["V"]) or output.dtype != torch.bfloat16:
                        raise RuntimeError(f"Unexpected {version} output for case {case['case_id']}")
                if not torch.isfinite(output).all().item():
                    raise RuntimeError(f"Non-finite or unwritten {version} output for case {case['case_id']}")
                record = dict(case, version=version, status="PASS", warmups=args.warmups, samples=1,
                              input_sha256=hashes, finite_output=True)
                records.append(record)
                progress.write(json.dumps(record) + "\n")
                progress.flush()
                print(f"[{len(records)}/{len(cases) * 2}] case={case['case_id']} {version} "
                      f"State={case['state_dtype']} BS={case['BS']} HV={case['HV']} "
                      f"TPB={case['TOKEN_PER_BATCH']}", flush=True)
                del output
            del data
    benchmark.write_json(args.output / "launches.json", records)


def parse_results(output: Path, cases: list[dict], warmups: int) -> None:
    launches = json.loads((output / "launches.json").read_text(encoding="utf-8"))
    expected = [(case["case_id"], version) for case in cases for version in
        (("v1", "v2") if case["comparison_id"] % 2 == 0 else ("v2", "v1"))]
    if [(row["case_id"], row["version"]) for row in launches] != expected:
        raise RuntimeError("Incomplete or unexpected launch order.")
    targets = []
    for path in sorted((output / "profile").rglob("op_summary_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line, row in enumerate(csv.DictReader(stream), start=2):
                if row.get("OP Type") == "RecurrentGatedDeltaRule":
                    row.update(source_csv=path.relative_to(output).as_posix(), source_line=line)
                    targets.append(row)
    targets.sort(key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
    per_launch = warmups + 1
    if len(targets) != len(launches) * per_launch:
        raise RuntimeError(f"Expected {len(launches) * per_launch} target tasks, got {len(targets)}")
    for index, record in enumerate(launches):
        group = targets[index * per_launch:(index + 1) * per_launch]
        prefix = "aclnnRecurrentGatedDeltaRuleV2_" if record["version"] == "v2" else "aclnnRecurrentGatedDeltaRule_"
        for row in group:
            if not row["Op Name"].startswith(prefix):
                raise RuntimeError(f"Kernel version/order mismatch: {record['case_id']}, {row['Op Name']}")
            shape = row["Output Shapes"].strip('"').split(';')[0].replace(' ', '')
            if shape != f"{record['T']},{record['HV']},{record['V']}":
                raise RuntimeError(f"Output shape mismatch: {record['case_id']}, {shape}")
            dtype = "DT_BF16" if record["state_dtype"] == "bf16" else "FLOAT"
            if row["Input Data Types"].split(';')[4] != dtype:
                raise RuntimeError(f"State dtype mismatch: {record['case_id']}")
        measured = group[-1]
        record.update(duration_us=float(measured["Task Duration(us)"]), msprof_row=measured)
        if not record["duration_us"] > 0:
            raise RuntimeError("Invalid target duration.")
        for field in benchmark.PIPE_FIELDS:
            record[field] = float(measured.get(field) or 0)
        record["pipe_fields_complete"] = all(record[field] > 0 for field in benchmark.PIPE_FIELDS)
    by_id = {(row["case_id"], row["version"]): row for row in launches}
    comparisons = []
    for case in cases:
        v1, v2 = (by_id[(case["case_id"], version)] for version in ("v1", "v2"))
        if v1["input_sha256"] != v2["input_sha256"]:
            raise RuntimeError("The two versions received different inputs.")
        pair = dict(case, v1_us=v1["duration_us"], v2_us=v2["duration_us"],
            speedup=v1["duration_us"] / v2["duration_us"],
            reduction_percent=(1 - v2["duration_us"] / v1["duration_us"]) * 100,
            input_sha256=v1["input_sha256"], finite_outputs=True,
            pipe_fields_complete=v1["pipe_fields_complete"] and v2["pipe_fields_complete"])
        for version, record in (("v1", v1), ("v2", v2)):
            pair.update({version + "_" + field: record[field] for field in benchmark.PIPE_FIELDS})
        comparisons.append(pair)
    benchmark.write_json(output / "measurements.json", launches)
    benchmark.write_json(output / "comparison.json", comparisons)
    fields = [name for name in comparisons[0] if name != "input_sha256"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)
    summary = dict(pairs=len(comparisons), measurements=len(launches), target_kernel_calls=len(targets),
        flags={name: False for name in benchmark.FLAG_NAMES}, timing_source="msprof op_summary Task Duration(us)",
        warmups=warmups, samples=1, identical_inputs=True, all_outputs_finite=True,
        incomplete_pipe_case_ids=[row["case_id"] for row in comparisons if not row["pipe_fields_complete"]],
        by_state_dtype={dtype: dict(count=len(rows), v2_faster=sum(row["speedup"] > 1 for row in rows),
            speedup_min=min(row["speedup"] for row in rows), speedup_max=max(row["speedup"] for row in rows),
            speedup_median=statistics.median(row["speedup"] for row in rows),
            v1_min_us=min(row["v1_us"] for row in rows), v1_max_us=max(row["v1_us"] for row in rows),
            v2_min_us=min(row["v2_us"] for row in rows), v2_max_us=max(row["v2_us"] for row in rows))
            for dtype in ("bf16", "fp32") if (rows := [row for row in comparisons if row["state_dtype"] == dtype])})
    benchmark.write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("recurrent_gdn_v1_v2_false"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--case-ids", help="Optional comma-separated original benchmark case IDs")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("warmups must be nonnegative")
    cases = comparison_cases()
    if args.case_ids:
        ids = {int(value) for value in args.case_ids.split(",")}
        cases = [case for case in cases if case["case_id"] in ids]
        if {case["case_id"] for case in cases} != ids:
            parser.error("case-ids must identify existing all-False cases")
    if args.list_cases:
        print(json.dumps(cases, indent=2))
        return
    args.output = args.output.resolve()
    if args.parse_only:
        parse_results(args.output, cases, args.warmups)
        return
    if args.worker:
        run_worker(args, cases)
        return
    if args.output.exists():
        parser.error("output must be a new directory")
    if not shutil.which("msprof"):
        parser.error("msprof is required")
    args.output.mkdir(parents=True)
    benchmark.write_json(args.output / "cases.json", cases)
    command = ["msprof", "--output=" + str(args.output / "profile"), sys.executable, str(Path(__file__).resolve()),
               "--worker", "--output", str(args.output), "--device", str(args.device), "--warmups", str(args.warmups)]
    if args.case_ids:
        command += ["--case-ids", args.case_ids]
    with (args.output / "msprof.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    parse_results(args.output, cases, args.warmups)


if __name__ == "__main__":
    main()
