#!/usr/bin/env python3
"""Isolate A5 KDA tail/final-state hangs with short, reproducible cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SHORT_CASES = (
    ("t64_final", 64, True, False),
    ("t65_minimal", 65, False, False),
    ("t65_final", 65, True, False),
    ("t65_saved", 65, True, True),
)
LONG_CASES = (
    ("h96_t8k", 8192, True, False),
    ("h96_t16k", 16384, True, False),
)
ADAPTER_CASES = (("bf16_gate_params", 64, True, False),)
OUTPUT_NAMES = (
    "attn_out", "final_state", "gk", "Aqk", "Akk", "w",
    "u", "qg", "kg", "v_new", "h", "initial_state",
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--long-seq", action="store_true")
    parser.add_argument("--bf16-gate-params", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tokens", type=int, default=65, help=argparse.SUPPRESS)
    parser.add_argument("--final-state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--saved", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--constant-inputs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--layout", choices=("NTD", "BSND"), default="NTD", help=argparse.SUPPRESS)
    parser.add_argument("--adapter", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gate-params-bf16", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=2, help=argparse.SUPPRESS)
    return parser.parse_args()


def _fingerprint(torch, tensor):
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1)
    stride = max(1, (flat.numel() + 4095) // 4096)
    value = flat[::stride][:4096].float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "finite": bool(torch.isfinite(flat).all().item()),
        "sum": float(value.sum().item()),
        "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
        "sample_numel": value.numel(),
    }


def _cpu_snapshot(outputs):
    return tuple(
        None if value is None else value.detach().cpu().contiguous()
        for value in outputs
    )


def _compare_snapshots(current, baseline):
    equal_by_output = {}
    differences = []
    for name, value, expected in zip(OUTPUT_NAMES, current, baseline):
        if value is None or expected is None:
            is_equal = value is expected
        else:
            is_equal = value.shape == expected.shape and value.equal(expected)
        equal_by_output[name] = is_equal
        if is_equal:
            continue
        detail = {"output": name}
        if value is None or expected is None:
            detail["optional_output_mismatch"] = True
        elif value.shape != expected.shape:
            detail["shape"] = list(value.shape)
            detail["baseline_shape"] = list(expected.shape)
        else:
            unequal = value != expected
            first = unequal.nonzero(as_tuple=False)[0]
            index = tuple(int(item) for item in first.tolist())
            detail.update({
                "mismatched_elements": int(unequal.sum().item()),
                "max_abs": float(
                    (value.float() - expected.float()).abs().max().item()
                ),
                "first_index": list(index),
                "actual": float(value[index].float().item()),
                "baseline": float(expected[index].float().item()),
            })
        differences.append(detail)
    return all(equal_by_output.values()), equal_by_output, differences


def _run_child(args):
    import torch
    import torch_npu  # noqa: F401

    from fla_npu.ops.ascendc import chunk_kda_fwd

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.manual_seed(20260806 + args.heads + args.tokens)
    h, t, dim = args.heads, args.tokens, 128
    shape = (1, t, h, dim) if args.layout == "BSND" else (h, t, dim)
    beta_shape = (1, t, h) if args.layout == "BSND" else (h, t)
    if args.constant_inputs:
        q = torch.full(shape, dim**-0.5, dtype=torch.bfloat16, device=device)
        k = torch.full_like(q, dim**-0.5)
        v = torch.zeros_like(q)
        beta = torch.full(beta_shape, 0.5, dtype=torch.bfloat16, device=device)
        g = torch.full(shape, -1.0, dtype=torch.float32, device=device)
        a_log = torch.zeros(h, dtype=torch.float32, device=device)
        dt_bias = torch.zeros(h * dim, dtype=torch.float32, device=device)
    else:
        q = (torch.randn(shape) * 0.02).to(torch.bfloat16).to(device)
        k = (torch.randn(shape) * 0.02).to(torch.bfloat16).to(device)
        v = (torch.randn(shape) * 0.02).to(torch.bfloat16).to(device)
        beta = torch.sigmoid(torch.randn(beta_shape)).to(torch.bfloat16).to(device)
        g = (torch.randn(shape) * 0.2).to(device)
        a_log = (torch.randn(h) * 0.2 - 0.5).to(device)
        dt_bias = (torch.randn(h * dim) * 0.1).to(device)
    if args.gate_params_bf16:
        a_log = a_log.to(torch.bfloat16)
        dt_bias = dt_bias.to(torch.bfloat16)

    baseline = None
    deterministic = None
    deterministic_by_output = None
    binary_differences = []
    fingerprints = None
    started = time.perf_counter()
    for _ in range(args.repeats):
        common = {
            "cu_seqlens": None if args.layout == "BSND" else (0, t),
            "output_final_state": args.final_state,
            "safe_gate": True,
            "lower_bound": -5.0,
            "use_gate_in_kernel": True,
            "A_log": a_log,
            "dt_bias": dt_bias,
            "disable_recompute": args.saved,
            "return_intermediate_states": args.saved,
        }
        if args.adapter:
            from fla_npu.adapters.triton_ascend_kda import (
                triton_ascend_chunk_kda_fwd,
            )

            adapter_common = dict(common)
            adapter_common.pop("output_final_state")
            outputs = triton_ascend_chunk_kda_fwd(
                q, k, v, g, beta, dim**-0.5, None, args.final_state,
                chunk_size=64,
                transpose_state_layout=False,
                **adapter_common,
            )
        else:
            outputs = chunk_kda_fwd(
                q, k, v, g, beta, dim**-0.5, 64,
                layout=args.layout,
                **common,
            )
        torch.npu.synchronize()
        fingerprints = {
            name: _fingerprint(torch, value)
            for name, value in zip(OUTPUT_NAMES, outputs)
        }
        snapshot = _cpu_snapshot(outputs)
        if baseline is not None:
            repeat_equal, equal_by_output, differences = _compare_snapshots(
                snapshot, baseline
            )
            deterministic = (
                repeat_equal if deterministic is None
                else deterministic and repeat_equal
            )
            deterministic_by_output = equal_by_output
            binary_differences.extend(differences)
        elif args.repeats > 1:
            baseline = snapshot

    memory_scale = 1024**3
    print(json.dumps({
        "output_count": len(outputs),
        "elapsed_ms": (time.perf_counter() - started) * 1e3,
        "memory_allocated_gib": torch.npu.memory_allocated(device) / memory_scale,
        "memory_reserved_gib": torch.npu.memory_reserved(device) / memory_scale,
        "deterministic": deterministic,
        "deterministic_by_output": deterministic_by_output,
        "binary_differences": binary_differences,
        "outputs": fingerprints,
    }))
    return 0 if deterministic is not False and all(
        value is None or value["finite"] for value in fingerprints.values()
    ) else 1


def _run_parent(args):
    if args.long_seq and args.bf16_gate_params:
        raise ValueError("--long-seq and --bf16-gate-params are mutually exclusive")
    long_seq = args.long_seq
    adapter = args.bf16_gate_params
    cases = LONG_CASES if long_seq else ADAPTER_CASES if adapter else SHORT_CASES
    heads = 96 if long_seq else args.heads
    timeout = args.timeout or (180 if long_seq else 30)
    repeats = 1 if long_seq else 2
    root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    print(json.dumps({
        "commit": commit, "device": args.device, "heads": heads,
        "long_seq": long_seq, "bf16_gate_params": adapter, "timeout": timeout,
    }))

    for name, tokens, final_state, saved in cases:
        command = [
            sys.executable, "-u", __file__, "--child",
            "--device", str(args.device), "--heads", str(heads),
            "--tokens", str(tokens), "--repeats", str(repeats),
        ]
        if long_seq:
            command.extend(("--constant-inputs", "--layout", "BSND"))
        if adapter:
            command.extend(
                ("--adapter", "--gate-params-bf16", "--layout", "BSND")
            )
        if final_state:
            command.append("--final-state")
        if saved:
            command.append("--saved")
        print(f"[RUN] {name}", flush=True)
        try:
            result = subprocess.run(
                command,
                env={**os.environ, "ASCEND_LAUNCH_BLOCKING": "1"},
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            print(error.stdout or "", end="")
            print(f"[TIMEOUT] {name} after {timeout}s; stop and reset the device")
            return 124
        print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, end="", file=sys.stderr)
            print(f"[FAIL] {name}: returncode={result.returncode}")
            return result.returncode
        print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    parsed = _parse_args()
    raise SystemExit(_run_child(parsed) if parsed.child else _run_parent(parsed))
