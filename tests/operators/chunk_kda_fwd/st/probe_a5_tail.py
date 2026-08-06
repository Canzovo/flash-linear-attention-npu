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
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tokens", type=int, default=65, help=argparse.SUPPRESS)
    parser.add_argument("--final-state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--saved", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--constant-inputs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--layout", choices=("NTD", "BSND"), default="NTD", help=argparse.SUPPRESS)
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

    baseline = None
    deterministic = None
    fingerprints = None
    started = time.perf_counter()
    for _ in range(args.repeats):
        outputs = chunk_kda_fwd(
            q, k, v, g, beta, dim**-0.5, 64,
            layout=args.layout,
            cu_seqlens=None if args.layout == "BSND" else (0, t),
            output_final_state=args.final_state,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            disable_recompute=args.saved,
            return_intermediate_states=args.saved,
        )
        torch.npu.synchronize()
        fingerprints = {
            name: _fingerprint(torch, value)
            for name, value in zip(OUTPUT_NAMES, outputs)
        }
        if baseline is not None:
            deterministic = all(
                left is right or (
                    left is not None and right is not None and torch.equal(left, right)
                )
                for left, right in zip(outputs, baseline)
            )
        elif args.repeats > 1:
            baseline = outputs

    memory_scale = 1024**3
    print(json.dumps({
        "output_count": len(outputs),
        "elapsed_ms": (time.perf_counter() - started) * 1e3,
        "memory_allocated_gib": torch.npu.memory_allocated(device) / memory_scale,
        "memory_reserved_gib": torch.npu.memory_reserved(device) / memory_scale,
        "deterministic": deterministic,
        "outputs": fingerprints,
    }))
    return 0 if deterministic is not False and all(
        value is None or value["finite"] for value in fingerprints.values()
    ) else 1


def _run_parent(args):
    long_seq = args.long_seq
    cases = LONG_CASES if long_seq else SHORT_CASES
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
        "long_seq": long_seq, "timeout": timeout,
    }))

    for name, tokens, final_state, saved in cases:
        command = [
            sys.executable, "-u", __file__, "--child",
            "--device", str(args.device), "--heads", str(heads),
            "--tokens", str(tokens), "--repeats", str(repeats),
        ]
        if long_seq:
            command.extend(("--constant-inputs", "--layout", "BSND"))
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
