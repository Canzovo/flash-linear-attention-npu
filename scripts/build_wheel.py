"""Build the root wheel and print its exact installation command."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fla_npu_artifacts import get_wheel_filename  # noqa: E402


def _resolve_output_dir(value: str) -> Path:
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    return output_dir.resolve()


def _install_command(wheel_path: Path) -> str:
    return (
        f"{shlex.quote(sys.executable)} -m pip install "
        "--force-reinstall --no-cache-dir --no-deps "
        f"{shlex.quote(str(wheel_path))}"
    )


def _collect_build_args(args: argparse.Namespace) -> str:
    parts = list(args.build_args)
    env_args = os.getenv("FLA_NPU_BUILD_ARGS", "").strip()
    if env_args:
        parts.insert(0, env_args)
    return " ".join(part.strip() for part in parts if part.strip())


def _strip_op_debug_config(build_args: str) -> str:
    """从透传参数里去掉已有的 --op_debug_config（含其值），
    避免与原生 -g / --sanitizer 生成的配置重复。"""
    tokens = build_args.split()
    out = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--op_debug_config":
            skip_next = True
            continue
        if token.startswith("--op_debug_config="):
            continue
        out.append(token)
    return " ".join(out)


def _op_debug_config_args(args: argparse.Namespace) -> str:
    """Map 一键编包的原生 -g / --sanitizer 选项到 build.sh 的 --op_debug_config。

    build.sh 解析 `--op_debug_config <value>`（空格分隔，不用 '='），
    value 用逗号分隔多个配置项。add_opc_config 会把
    ccec_g -> -g、sanitizer -> -sanitizer。
    """
    configs = []
    if args.debug:
        configs.append("ccec_g")
    if args.sanitizer:
        configs.append("sanitizer")
    if not configs:
        return ""
    return f"--op_debug_config {','.join(configs)}"


def _assemble_build_args(args: argparse.Namespace) -> str:
    build_args = _collect_build_args(args)
    native = _op_debug_config_args(args)
    if not native:
        return build_args
    build_args = _strip_op_debug_config(build_args)
    return f"{build_args} {native}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-dir",
        default="dist",
        help="wheel output directory relative to the repository root (default: dist)",
    )
    parser.add_argument(
        "-g",
        "--debug",
        action="store_true",
        help=(
            "add kernel debug info (asc_opc -g). Equivalent to passing "
            "--op_debug_config ccec_g to build.sh."
        ),
    )
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help=(
            "enable Ascend kernel sanitizer instrumentation (asc_opc "
            "--op_debug_config=sanitizer). Equivalent to passing "
            "--op_debug_config sanitizer to build.sh."
        ),
    )
    parser.add_argument(
        "--build-args",
        action="append",
        default=[],
        metavar="ARGS",
        help=(
            "extra arguments forwarded to build.sh (e.g. "
            "--build-args='-O3'). build.sh parses option values "
            "space-separated, so do not use '=' between an option and its "
            "value. May be repeated or space-separated within one value. "
            "Also honored via the FLA_NPU_BUILD_ARGS environment variable."
        ),
    )
    args = parser.parse_args()

    wheel_dir = _resolve_output_dir(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / get_wheel_filename(REPO_ROOT)

    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        ".",
        "-w",
        str(wheel_dir),
    ]

    env = os.environ.copy()
    build_args = _assemble_build_args(args)
    if build_args:
        env["FLA_NPU_BUILD_ARGS"] = build_args
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)

    if not wheel_path.is_file():
        raise RuntimeError(f"Expected wheel was not produced: {wheel_path}")

    print(f"[fla-npu build] Wheel: {wheel_path}", flush=True)
    print(f"[fla-npu build] Install command:", flush=True)
    print(_install_command(wheel_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
