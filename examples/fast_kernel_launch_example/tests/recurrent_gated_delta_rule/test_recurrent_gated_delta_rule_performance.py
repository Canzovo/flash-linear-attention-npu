#!/usr/bin/env python3
"""Profile RGDR state layouts through the direct ``<<<>>>`` entry point.

The cases, deterministic input generation, state-layout construction, and
workload markers are loaded from the operator PTA state-layout workload.  This
keeps the formal ``fla_npu.ops`` path and this fast-launch path on exactly the
same case set.

This script deliberately does not calculate latency or enforce a performance
threshold.  Run continuous and non-contiguous layouts in separate ``msprof``
processes and compare the profiled ``recurrent_gated_delta_rule_kernel`` task
duration.  The mutable API is the default because it matches the API measured
by the original PTA workload; the functional API can be selected explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import ascend_ops  # noqa: F401  # Import _C and register torch.ops.ascend_ops.
import torch


_REPO_ROOT = Path(__file__).resolve().parents[4]
_PTA_WORKLOAD_FILE = (
    _REPO_ROOT
    / "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/tests/pta"
    / "test_state_layout_performance.py"
)


def load_pta_workload() -> ModuleType:
    """Load the shared workload under a non-pytest module name."""

    module_name = "_rgdr_state_layout_performance_workload"
    spec = importlib.util.spec_from_file_location(module_name, _PTA_WORKLOAD_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load PTA workload: {_PTA_WORKLOAD_FILE}")

    module = importlib.util.module_from_spec(spec)
    # Dataclass processing expects the module to be visible in sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PTA_WORKLOAD = load_pta_workload()


def resolve_fast_launch_operator(api_mode: str) -> tuple[Callable[..., Any], str]:
    """Return the requested torch.ops direct-launch entry point."""

    op_name = "recurrent_gated_delta_rule"
    if api_mode == "functional":
        op_name += "_functional"

    namespace = torch.ops.ascend_ops
    if not hasattr(namespace, op_name):
        raise RuntimeError(
            f"torch.ops.ascend_ops.{op_name} is not registered; build and install "
            "the recurrent_gated_delta_rule fast-launch wheel first"
        )
    return getattr(namespace, op_name), f"torch.ops.ascend_ops.{op_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--state-layout",
        choices=("all", "continuous", "noncontiguous"),
        default="all",
        help="Layout to execute; profile layouts separately for comparison",
    )
    parser.add_argument(
        "--api-mode",
        choices=("mutable", "functional"),
        default="mutable",
        help="Direct-launch API to profile (default: mutable)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Operator invocations per case and layout (default: 1)",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only this named shared PTA case; may be repeated",
    )
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0")

    selected_cases = list(PTA_WORKLOAD.CASES)
    if args.case:
        requested_cases = set(args.case)
        selected_cases = [
            case for case in selected_cases if case.name in requested_cases
        ]
        missing_cases = requested_cases - {case.name for case in selected_cases}
        if missing_cases:
            raise ValueError(f"Unknown cases: {sorted(missing_cases)}")

    if args.list_cases:
        for case in selected_cases:
            print(case.name)
        return

    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError(
            "This workload must run in an environment with torch_npu"
        ) from error

    operator, operator_entrypoint = resolve_fast_launch_operator(args.api_mode)
    layouts = PTA_WORKLOAD.selected_layouts(args.state_layout)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must be an NPU device such as npu:0")
    torch_npu.npu.set_device(device)

    print(
        f"device={args.device} operator={operator_entrypoint} "
        f"api_mode={args.api_mode} layouts={','.join(layouts)} "
        f"repeat={args.repeat} cases={len(selected_cases)}",
        flush=True,
    )
    for case in selected_cases:
        print(
            f"CASE case={case.name} B={case.batch_size} T={case.token_count} "
            f"Nk={case.nk} Nv={case.nv} Dk={case.dk} Dv={case.dv} "
            f"gate={'g' if case.use_g else 'gk'} seed={case.seed}",
            flush=True,
        )
        PTA_WORKLOAD.run_case(
            case,
            layouts,
            operator,
            torch_npu,
            device,
            args.repeat,
        )

    print("All requested fast-launch workloads completed.", flush=True)


if __name__ == "__main__":
    main()
