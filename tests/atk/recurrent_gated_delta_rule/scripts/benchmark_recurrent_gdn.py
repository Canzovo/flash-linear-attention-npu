#!/usr/bin/env python3
"""Standalone RecurrentGDN performance matrix for Ascend NPU and NVIDIA GPU.

Examples (no ATK or case JSON dependency):
  python benchmark_recurrent_gdn.py --backend npu --output npu_results
  python benchmark_recurrent_gdn.py --backend gpu --output gpu_results
  python benchmark_recurrent_gdn.py --backend gpu --warmup 5 --repeat 20 --output gpu_results
  python benchmark_recurrent_gdn.py --list-cases > cases.json

NPU: invokes msprof automatically and reads target Task Duration(us).
GPU: CUDA events, warmup 5/repeat 20; mean/median/min/max and every sample.
Both: BF16 Q/K/V/beta, FP32 g/A_log/dt_bias, BF16 and FP32 initial State.
NPU writes every token checkpoint; upstream GPU writes only final FP32 State.
The results describe those public interfaces, with different State write volumes.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


BATCH_SIZES = (32, 64, 72)
KEY_HEADS = 16
VALUE_HEADS = (32, 48, 128)
TOKENS_PER_BATCH = (1, 2, 4, 8)
HEAD_DIM = 128
STATE_DTYPES = ("bf16", "fp32")
FLAG_NAMES = ("use_qk_l2norm_in_kernel", "use_gate_in_kernel",
              "use_beta_sigmoid_in_kernel", "allow_neg_eigval")
BASE_SEED = 20261022
GPU_REFERENCE = "https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py"
PIPE_FIELDS = ("aiv_time(us)", "aiv_vec_time(us)", "aiv_vec_ratio", "aiv_scalar_time(us)",
               "aiv_mte2_time(us)", "aiv_mte3_time(us)")


def matrix(state_dtype: str) -> list[dict]:
    cases = []
    dtypes = STATE_DTYPES if state_dtype == "both" else (state_dtype,)
    for dtype in dtypes:
        for shape_id, (bs, hv, length) in enumerate(itertools.product(
                BATCH_SIZES, VALUE_HEADS, TOKENS_PER_BATCH)):
            for flags in itertools.product((False, True), repeat=len(FLAG_NAMES)):
                norm, gate, sigmoid, neg = flags
                # The upstream public GPU function rejects this combination.
                if neg and not sigmoid:
                    continue
                cases.append(dict(case_id=len(cases), shape_id=shape_id, BS=bs,
                    HK=KEY_HEADS, HV=hv, TOKEN_PER_BATCH=length, T=bs * length,
                    K=HEAD_DIM, V=HEAD_DIM, state_dtype=dtype, qkv_dtype="bf16",
                    beta_dtype="bf16", g_dtype="fp32", scale=HEAD_DIM ** -0.5,
                    seed=BASE_SEED + shape_id, **dict(zip(FLAG_NAMES, flags))))
    return cases


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tensor_hash(tensor) -> str:
    import torch

    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()


def make_inputs(torch, case: dict, device) -> tuple[dict, dict]:
    """CPU generation is identical on both backends; seed offsets match the ATK ranges."""
    bs, hv, length = case["BS"], case["HV"], case["TOKEN_PER_BATCH"]
    total, hk, dim, seed = case["T"], case["HK"], case["K"], case["seed"]
    state_dtype = torch.bfloat16 if case["state_dtype"] == "bf16" else torch.float32

    def normal(shape, offset, scale, dtype):
        gen = torch.Generator(device="cpu").manual_seed(seed + offset)
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(dtype)

    def uniform(shape, offset, low, high, dtype):
        gen = torch.Generator(device="cpu").manual_seed(seed + offset)
        return (torch.rand(shape, generator=gen, dtype=torch.float32) * (high - low) + low).to(dtype)

    cpu = dict(q=normal((total, hk, dim), 1, 0.05, torch.bfloat16),
               k=normal((total, hk, dim), 2, 0.05, torch.bfloat16),
               v=normal((total, hv, dim), 3, 0.05, torch.bfloat16),
               initial_state=normal((bs, hv, dim, dim), 4, 0.01, state_dtype),
               beta_post=uniform((total, hv), 5, 0.1, 0.9, torch.bfloat16),
               beta_raw=normal((total, hv), 5, 3.0, torch.bfloat16),
               g_log=-uniform((total, hv), 6, 0.001, 0.02, torch.float32),
               g_raw=normal((total, hv), 6, 2.0, torch.float32),
               A_log=normal((hv,), 8, 0.5, torch.float32),
               dt_bias=normal((hv,), 9, 0.2, torch.float32))
    hashes = {name: tensor_hash(tensor) for name, tensor in cpu.items()}
    # Hash the logical initial states; unused checkpoint slots are never read.
    data = {name: tensor.to(device) for name, tensor in cpu.items()}
    if device.type == "npu":
        data["lengths"] = torch.tensor([0] + [length] * bs, dtype=torch.int32, device=device)
        data["indices"] = torch.arange(total, dtype=torch.int32, device=device)
        data["state"] = torch.empty((total, hv, dim, dim), dtype=state_dtype, device=device)
    else:
        # Keep upstream's default [K,V] layout, including its Hopper dispatch.
        # Hashes above always describe the same logical [V,K] initial states.
        data["initial_state"] = data["initial_state"].transpose(-1, -2).contiguous()
    return data, hashes


def input_hashes(hashes: dict, case: dict) -> dict:
    names = ["q", "k", "v", "initial_state",
             "beta_raw" if case["use_beta_sigmoid_in_kernel"] else "beta_post",
             "g_raw" if case["use_gate_in_kernel"] else "g_log"]
    if case["use_gate_in_kernel"]:
        names += ["A_log", "dt_bias"]
    return {name: hashes[name] for name in names}


def time_gpu_calls(torch, fn, warmup: int, repeat: int) -> list[float]:
    """Match the supplied KDA benchmark's time_calls; samples are milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def summarize_gpu_times(samples_ms: list[float]) -> dict:
    result = dict(mean_ms=statistics.mean(samples_ms), median_ms=statistics.median(samples_ms),
                  min_ms=min(samples_ms), max_ms=max(samples_ms), samples_ms=samples_ms)
    result.update({name.replace("_ms", "_us"): value * 1000.0
                   for name, value in result.copy().items() if name != "samples_ms"})
    result.update(duration_us=result["mean_us"], samples_us=[value * 1000.0 for value in samples_ms])
    return result


def run_worker(args, cases: list[dict]) -> None:
    import torch

    torch.set_num_threads(args.threads)
    if args.backend == "npu":
        import torch_npu  # noqa: F401
        from fla_npu.ops import ascendc

        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        accelerator = torch.npu
        operator = ascendc.recurrent_gated_delta_rule
        timing_source = "msprof op_summary Task Duration(us)"
    else:
        from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU and CUDA-enabled PyTorch are required.")
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        accelerator = torch.cuda
        operator = fused_recurrent_gated_delta_rule
        required = set(FLAG_NAMES) | {"state_v_first", "output_final_state", "initial_state"}
        if not required.issubset(inspect.signature(operator).parameters):
            raise RuntimeError("Installed FLA does not expose the required upstream API; update FLA.")
        timing_source = "CUDA event per-call samples; duration_us=mean_ms*1000"

    source = inspect.getsourcefile(operator)
    package_file = Path(sys.modules[operator.__module__].__file__)
    version = dict(backend=args.backend, torch_version=torch.__version__,
                  device_name=accelerator.get_device_name(args.device),
                  operator_module=operator.__module__, operator_function=operator.__name__,
                  operator_source_sha256=hashlib.sha256(Path(source or package_file).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  timing_source=timing_source, gpu_reference=GPU_REFERENCE,
                  warmups=args.warmups, samples=args.samples,
                  npu_checkpoint_slots="T; indices=range(T); initial read slots=b*TOKEN_PER_BATCH",
                  state_v_first=args.backend == "npu",
                  initial_state_hash_layout="canonical [BS,HV,V,K] on both backends",
                  npu_state_writes="every token checkpoint, in state_dtype",
                  gpu_state_writes="one final FP32 State per sequence",
                  preprocessing_and_initial_state_reset_excluded=True,
                  formal_accuracy_test=False)
    if args.backend == "gpu":
        import triton
        version.update(cuda_version=torch.version.cuda, triton_version=triton.__version__,
                       warmup=args.warmups, repeat=args.samples, untimed_prologue_calls=1,
                       duration_statistic="mean", raw_sample_unit="ms")
    else:
        version["torch_npu_version"] = torch_npu.__version__
    write_json(args.output / "environment.json", version)
    write_json(args.output / "cases.json", cases)
    data, previous = None, None
    records = []
    with (args.output / "launches.jsonl").open("x", encoding="utf-8") as progress:
        with torch.no_grad():
            for index, case in enumerate(cases):
                shape = (case["state_dtype"], case["shape_id"])
                if shape != previous:
                    data = None
                    accelerator.empty_cache()
                    data, hashes = make_inputs(torch, case, device)
                    previous = shape
                flags = {name: case[name] for name in FLAG_NAMES}
                beta = data["beta_raw" if case["use_beta_sigmoid_in_kernel"] else "beta_post"]
                gate = data["g_raw" if case["use_gate_in_kernel"] else "g_log"]
                alog = data["A_log"] if case["use_gate_in_kernel"] else None
                bias = data["dt_bias"] if case["use_gate_in_kernel"] else None
                bs, length, hv, dim = case["BS"], case["TOKEN_PER_BATCH"], case["HV"], case["K"]
                if args.backend == "gpu":
                    # Prepare views and kwargs outside the event interval.
                    gpu_inputs = dict(q=data["q"].view(bs, length, case["HK"], dim),
                        k=data["k"].view(bs, length, case["HK"], dim),
                        v=data["v"].view(bs, length, hv, dim), beta=beta.view(bs, length, hv),
                        g=gate.view(bs, length, hv), A_log=alog, dt_bias=bias,
                        scale=case["scale"], initial_state=data["initial_state"],
                        output_final_state=True, state_v_first=False, **flags)

                def launch():
                    if args.backend == "npu":
                        return operator(data["q"], data["k"], data["v"], data["state"],
                            beta=beta, g=gate, A_log=alog, dt_bias=bias, scale=case["scale"],
                            actual_seq_lengths=data["lengths"], ssm_state_indices=data["indices"], **flags)
                    return operator(**gpu_inputs)

                timing = dict(duration_us=None, samples_us=[])
                if args.backend == "gpu":
                    result = launch()  # Untimed API/output check, as in the reference prologue.
                    if tuple(result[0].shape) != (bs, length, hv, dim) or result[0].dtype != torch.bfloat16:
                        raise RuntimeError(f"Unexpected public output for case {case['case_id']}")
                    del result
                    timing = summarize_gpu_times(time_gpu_calls(torch, launch, args.warmups, args.samples))
                    timing["repeat"] = args.samples
                else:
                    for _ in range(args.warmups + args.samples):
                        # Restore all read slots. All other slots are written by
                        # this invocation and never used as its initial states.
                        data["state"][::length].copy_(data["initial_state"])
                        accelerator.synchronize()
                        result = launch()
                        accelerator.synchronize()
                        if tuple(result.shape) != (case["T"], hv, dim) or result.dtype != torch.bfloat16:
                            raise RuntimeError(f"Unexpected public output for case {case['case_id']}")
                        del result
                if args.backend == "npu" and ascendc.BACKENDS.get("npu_recurrent_gated_delta_rule") != "stable":
                    raise RuntimeError("NPU benchmark must use the installed Stable ABI V2 operator.")
                record = dict(case, backend=args.backend, status="PASS", warmups=args.warmups,
                              samples=args.samples, input_sha256=input_hashes(hashes, case),
                              timing_source=timing_source, **timing)
                records.append(record)
                progress.write(json.dumps(record) + "\n")
                progress.flush()
                label = (f"[{index + 1}/{len(cases)}] case={case['case_id']} "
                         f"State={case['state_dtype']} BS={bs} HV={hv} TPB={length}")
                if args.backend == "gpu":
                    label += (f" mean={record['mean_ms']:.6f} ms median={record['median_ms']:.6f} ms"
                              f" min={record['min_ms']:.6f} ms max={record['max_ms']:.6f} ms")
                print(label, flush=True)
    write_json(args.output / "launches.json", records)
    if args.backend == "gpu":
        save_results(args.output, records)


def parse_npu(output: Path, expected_cases: list[dict], warmups: int, samples: int) -> None:
    records = json.loads((output / "launches.json").read_text(encoding="utf-8"))
    if [r["case_id"] for r in records] != [c["case_id"] for c in expected_cases]:
        raise RuntimeError("Incomplete or unexpected case launch manifest.")
    targets = []
    for path in sorted((output / "profile").rglob("op_summary_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line, row in enumerate(csv.DictReader(stream), start=2):
                if row.get("OP Type") == "RecurrentGatedDeltaRule":
                    if not row["Op Name"].startswith("aclnnRecurrentGatedDeltaRuleV2_"):
                        raise RuntimeError("A legacy kernel appeared in the V2 profile.")
                    row["source_csv"] = path.relative_to(output).as_posix()
                    row["source_line"] = line
                    targets.append(row)
    targets.sort(key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
    per_case = warmups + samples
    if len(targets) != len(records) * per_case:
        raise RuntimeError(f"Expected {len(records) * per_case} target tasks, got {len(targets)}; retain raw profiles.")
    for index, record in enumerate(records):
        measured = targets[index * per_case + warmups:(index + 1) * per_case]
        values = [float(row["Task Duration(us)"]) for row in measured]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise RuntimeError("Invalid target task duration.")
        record.update(duration_us=statistics.mean(values), samples_us=values,
                      msprof_rows=measured, timing_source="msprof op_summary Task Duration(us)")
        for field in PIPE_FIELDS:
            record[field] = statistics.mean(float(row.get(field) or 0) for row in measured)
        record["pipe_fields_complete"] = all(record[field] > 0 for field in PIPE_FIELDS)
        record["tokens_per_second"] = record["T"] * 1e6 / record["duration_us"]
    save_results(output, records)


def save_results(output: Path, records: list[dict]) -> None:
    write_json(output / "results.json", records)
    fields = ["case_id", "backend", "status", "BS", "HK", "HV", "TOKEN_PER_BATCH", "T", "K", "V",
              "state_dtype", "qkv_dtype", "beta_dtype", "g_dtype", *FLAG_NAMES, "scale", "seed",
              "warmups", "samples", "duration_us", "timing_source", *PIPE_FIELDS, "pipe_fields_complete"]
    if records[0]["backend"] == "gpu":
        fields += ["repeat", "mean_ms", "median_ms", "min_ms", "max_ms", "samples_ms",
                   "mean_us", "median_us", "min_us", "max_us"]
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "summary.json", dict(total=len(records), passed=len(records),
        backend=records[0]["backend"], timing_source=records[0]["timing_source"],
        min_us=min(r["duration_us"] for r in records), max_us=max(r["duration_us"] for r in records),
        warmups=records[0]["warmups"], samples=records[0]["samples"],
        incomplete_pipe_case_ids=[r["case_id"] for r in records if r.get("pipe_fields_complete") is False],
        npu_state_writes="every token checkpoint", gpu_state_writes="final FP32 State only",
        gpu_executed_by_this_npu_run=False if records[0]["backend"] == "npu" else True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("npu", "gpu"))
    parser.add_argument("--output", type=Path, default=Path("recurrent_gdn_benchmark"))
    parser.add_argument("--device", type=int, default=0, help="Process-visible device index")
    parser.add_argument("--state-dtype", choices=(*STATE_DTYPES, "both"), default="both")
    parser.add_argument("--warmups", "--warmup", dest="warmups", type=int, default=5)
    parser.add_argument("--samples", "--repeat", dest="samples", type=int,
                        help="Timed calls per case (default: GPU 20, NPU 1)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--case-ids", help="Optional comma-separated IDs for smoke or diagnosis")
    parser.add_argument("--list-cases", action="store_true", help="Print embedded matrix without importing torch")
    parser.add_argument("--parse-only", action="store_true", help="Reparse an existing NPU profile")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = matrix(args.state_dtype)
    if args.case_ids:
        ids = {int(value) for value in args.case_ids.split(",")}
        if not ids.issubset({case["case_id"] for case in cases}):
            parser.error("case ID outside embedded matrix")
        cases = [case for case in cases if case["case_id"] in ids]
    if args.list_cases:
        print(json.dumps(cases, indent=2))
        return
    if args.backend is None:
        parser.error("--backend npu or --backend gpu is required")
    if args.samples is None:
        args.samples = 20 if args.backend == "gpu" else 1
    if args.warmups < 1 or args.samples < 1 or args.threads < 1:
        parser.error("warmups, samples and threads must be positive")
    args.output = args.output.resolve()
    if args.parse_only:
        if args.backend != "npu":
            parser.error("--parse-only applies to NPU msprof data")
        parse_npu(args.output, cases, args.warmups, args.samples)
        return
    if args.worker:
        run_worker(args, cases)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if args.backend == "gpu":
        run_worker(args, cases)
        return
    msprof = shutil.which("msprof")
    if not msprof:
        raise RuntimeError("msprof was not found; load the CANN environment before running.")
    command = [msprof, f"--output={args.output / 'profile'}", "--aic-metrics=PipeUtilization",
               sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    print("Collecting NPU target kernel timings with msprof.", flush=True)
    with (args.output / "msprof.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"msprof/worker exited {result.returncode}; inspect msprof.log")
    parse_npu(args.output, cases, args.warmups, args.samples)
    print(f"Saved {len(cases)} cases to results.csv and results.json.", flush=True)


if __name__ == "__main__":
    main()
