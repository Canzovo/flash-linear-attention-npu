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


CASES = (
    ("t64_final", 64, True, False),
    ("t65_minimal", 65, False, False),
    ("t65_final", 65, True, False),
    ("t65_saved", 65, True, True),
)
OUTPUT_NAMES = (
    "attn_out", "final_state", "gk", "Aqk", "Akk", "w",
    "u", "qg", "kg", "v_new", "h", "initial_state",
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tokens", type=int, default=65, help=argparse.SUPPRESS)
    parser.add_argument("--final-state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--saved", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _fingerprint(torch, tensor):
    if tensor is None:
        return None
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "finite": bool(torch.isfinite(value).all().item()),
        "sum": float(value.sum().item()),
        "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
    }


def _run_child(args):
    import torch
    import torch_npu  # noqa: F401

    from fla_npu.ops.ascendc import chunk_kda_fwd

    torch.npu.set_device(args.device)
    torch.manual_seed(20260806 + args.heads + args.tokens)
    h, t, dim = args.heads, args.tokens, 128
    q = (torch.randn(h, t, dim) * 0.02).to(torch.bfloat16).npu()
    k = (torch.randn(h, t, dim) * 0.02).to(torch.bfloat16).npu()
    v = (torch.randn(h, t, dim) * 0.02).to(torch.bfloat16).npu()
    beta = torch.sigmoid(torch.randn(h, t)).to(torch.bfloat16).npu()
    g = (torch.randn(h, t, dim) * 0.2).npu()
    a_log = (torch.randn(h) * 0.2 - 0.5).npu()
    dt_bias = (torch.randn(h * dim) * 0.1).npu()

    snapshots = []
    fingerprints = None
    started = time.perf_counter()
    for _ in range(2):
        outputs = chunk_kda_fwd(
            q, k, v, g, beta, dim**-0.5, 64,
            layout="NTD",
            cu_seqlens=(0, t),
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
        snapshots.append(tuple(None if x is None else x.detach().cpu() for x in outputs))
        fingerprints = {
            name: _fingerprint(torch, value)
            for name, value in zip(OUTPUT_NAMES, outputs)
        }

    deterministic = all(
        left is right or (
            left is not None and right is not None and torch.equal(left, right)
        )
        for left, right in zip(*snapshots)
    )
    print(json.dumps({
        "output_count": len(outputs),
        "elapsed_ms": (time.perf_counter() - started) * 1e3,
        "deterministic": deterministic,
        "outputs": fingerprints,
    }))
    return 0 if deterministic and all(
        value is None or value["finite"] for value in fingerprints.values()
    ) else 1


def _run_parent(args):
    root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    print(json.dumps({"commit": commit, "device": args.device, "heads": args.heads}))

    for name, tokens, final_state, saved in CASES:
        command = [
            sys.executable, "-u", __file__, "--child",
            "--device", str(args.device), "--heads", str(args.heads),
            "--tokens", str(tokens),
        ]
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
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            print(error.stdout or "", end="")
            print(f"[TIMEOUT] {name} after {args.timeout}s; stop and reset the device")
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
