#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_URL="https://github.com/flashserve/flash-linear-attention-npu.git"
DEFAULT_REF="refs/pull/264/head"

repo_url="$DEFAULT_REPO_URL"
ref="$DEFAULT_REF"
soc="ascend950"
device="0"
work_root="${PWD}/outputs/kda-a5-acceptance"
cann_env=""
conda_init=""
conda_env=""
model_source_root=""
case_filter="smoke"
case_timeout="900"
profile_launch_count="20"
profile_warm_up="5"
clone_retries="3"
ops="chunk_kda_fwd,kda_gate_cumsum,chunk_gated_delta_rule_fwd_h"

usage() {
    cat <<'EOF'
Usage: bash scripts/validate_kda_a5.sh [options]

Fetch PR264, build and install an isolated A5 wheel, then run the KDA tail,
BF16 gate-parameter and H96/T8K/T16K acceptance cases. Results are written to
results.json, results.md and per-case logs.

Options:
  --work-root DIR          Parent directory for source, wheel, logs and results
  --repo-url URL           Source repository URL
  --ref REF                Git ref (default: refs/pull/264/head)
  --soc SOC                Build SOC (default: ascend950)
  --device ID              Physical NPU exposed to the run (default: 0)
  --cann-env FILE          CANN set_env.sh to source
  --conda-init FILE        Conda profile script; required with --conda-env
  --conda-env NAME         Conda environment to activate
  --model-source-root DIR  Optional pinned Triton-Ascend kernels source root
  --cases IDS              smoke, all, or comma-separated case IDs
  --case-timeout SEC       Timeout for one case (default: 900)
  --profile-launch-count N msopprof launch count (default: 20)
  --profile-warm-up N      msopprof warm-up count (default: 5)
  --ops IDS                Comma-separated operators for the scoped wheel build
  --clone-retries N        Source fetch attempts (default: 3)
  -h, --help               Show this help

Cases:
  tail_sync, bf16_gate_params, h96_t8k_t16k,
  profile_h96_t8k, profile_h96_t16k

Example:
  bash scripts/validate_kda_a5.sh \
    --cann-env /path/to/Ascend/ascend-toolkit/set_env.sh \
    --device 0 --work-root "$PWD/outputs/kda-a5"
EOF
}

source_env_file() {
    local path="$1"
    set +u
    # shellcheck disable=SC1090
    source "$path"
    set -u
}

while (($#)); do
    case "$1" in
        --work-root) work_root="$2"; shift 2 ;;
        --repo-url) repo_url="$2"; shift 2 ;;
        --ref) ref="$2"; shift 2 ;;
        --soc) soc="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --cann-env) cann_env="$2"; shift 2 ;;
        --conda-init) conda_init="$2"; shift 2 ;;
        --conda-env) conda_env="$2"; shift 2 ;;
        --model-source-root) model_source_root="$2"; shift 2 ;;
        --cases) case_filter="$2"; shift 2 ;;
        --case-timeout) case_timeout="$2"; shift 2 ;;
        --profile-launch-count) profile_launch_count="$2"; shift 2 ;;
        --profile-warm-up) profile_warm_up="$2"; shift 2 ;;
        --ops) ops="$2"; shift 2 ;;
        --clone-retries) clone_retries="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$soc" == "ascend950" ]] || {
    echo "This acceptance entry targets A5; --soc must be ascend950" >&2
    exit 2
}
[[ "$device" =~ ^[0-9]+$ ]] || { echo "--device must be non-negative" >&2; exit 2; }
[[ "$case_timeout" =~ ^[1-9][0-9]*$ ]] || { echo "--case-timeout must be positive" >&2; exit 2; }
[[ "$profile_launch_count" =~ ^[1-9][0-9]*$ ]] || { echo "--profile-launch-count must be positive" >&2; exit 2; }
[[ "$profile_warm_up" =~ ^[0-9]+$ ]] || { echo "--profile-warm-up must be non-negative" >&2; exit 2; }
[[ "$clone_retries" =~ ^[1-9][0-9]*$ ]] || { echo "--clone-retries must be positive" >&2; exit 2; }
[[ -n "$ops" ]] || { echo "--ops must not be empty" >&2; exit 2; }

if [[ -n "$conda_env" ]]; then
    [[ -f "$conda_init" ]] || { echo "Invalid --conda-init: $conda_init" >&2; exit 2; }
    source_env_file "$conda_init"
    conda activate "$conda_env"
fi
if [[ -n "$cann_env" ]]; then
    [[ -f "$cann_env" ]] || { echo "Invalid --cann-env: $cann_env" >&2; exit 2; }
    source_env_file "$cann_env"
fi

for command_name in git python3 npu-smi realpath; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Required command not found: $command_name" >&2
        exit 1
    }
done
[[ -n "${ASCEND_HOME_PATH:-}" || -n "${ASCEND_OPP_PATH:-}" ]] || {
    echo "CANN is not active. Source set_env.sh or pass --cann-env." >&2
    exit 1
}

mkdir -p -- "$work_root"
work_root="$(realpath "$work_root")"
[[ "$work_root" != "/" ]] || { echo "Refusing to use / as --work-root" >&2; exit 2; }
if [[ -n "${FLA_NPU_ALLOWED_ROOT:-}" ]]; then
    allowed_root="$(realpath "${FLA_NPU_ALLOWED_ROOT}")"
    case "$work_root/" in
        "$allowed_root"/*) ;;
        *) echo "Work root is outside FLA_NPU_ALLOWED_ROOT: $work_root" >&2; exit 2 ;;
    esac
fi
if [[ -n "$model_source_root" ]]; then
    model_source_root="$(realpath "$model_source_root")"
    [[ -d "$model_source_root" ]] || {
        echo "Invalid --model-source-root: $model_source_root" >&2
        exit 2
    }
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$work_root/run_${timestamp}_$$"
source_dir="$run_dir/source"
wheel_dir="$run_dir/wheel"
venv_dir="$run_dir/venv"
tmp_dir="$run_dir/tmp"
result_dir="$run_dir/results"
mkdir -p -- "$run_dir" "$wheel_dir" "$tmp_dir"

on_error() {
    status=$?
    echo "Validation failed with status $status. Outputs: $run_dir" >&2
    exit "$status"
}
trap on_error ERR

echo "[1/6] Checking NPU $device"
npu-smi info -t board -i "$device" >/dev/null

echo "[2/6] Fetching $repo_url at $ref"
git init "$source_dir"
git -C "$source_dir" remote add origin "$repo_url"
fetch_succeeded="0"
for ((attempt = 1; attempt <= clone_retries; attempt++)); do
    if git -C "$source_dir" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
        fetch --depth 1 --filter=blob:none origin "$ref"; then
        fetch_succeeded="1"
        break
    fi
    ((attempt == clone_retries)) || sleep 5
done
[[ "$fetch_succeeded" == "1" ]] || { echo "Unable to fetch source" >&2; exit 1; }
git -C "$source_dir" checkout --detach FETCH_HEAD
commit="$(git -C "$source_dir" rev-parse HEAD)"
runner_script="$source_dir/scripts/validate_kda_a5.py"
[[ -f "$runner_script" ]] || { echo "Runner missing at commit $commit" >&2; exit 1; }
echo "Source commit: $commit"

echo "[3/6] Creating isolated Python environment"
python3 -m venv --system-site-packages "$venv_dir"
venv_python="$venv_dir/bin/python"
export PATH="$venv_dir/bin:$PATH"
"$venv_python" -c 'import torch, torch_npu; print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")'

echo "[4/6] Installing build dependencies"
build_requirements=(
    "setuptools>=70.1"
    wheel
    packaging
    psutil
    "cmake>=3.16,<4"
)
if ! command -v patch >/dev/null 2>&1; then
    build_requirements+=("patch-ng==1.19.1")
fi
"$venv_python" -m pip install --upgrade "${build_requirements[@]}"

if ! command -v patch >/dev/null 2>&1; then
    export FLA_NPU_PATCH_PYTHON="$venv_python"
    patch_wrapper="$venv_dir/bin/patch"
    cat > "$patch_wrapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

: "${FLA_NPU_PATCH_PYTHON:?FLA_NPU_PATCH_PYTHON is required}"
case "${1:-}" in
    --version|-h|--help)
        exec "$FLA_NPU_PATCH_PYTHON" -m patch_ng "$@"
        ;;
    -p1)
        shift
        set -- -p0 "$@"
        ;;
esac
patch_input="$(mktemp "${TMPDIR:-/tmp}/fla-npu-patch.XXXXXX")"
trap 'rm -f -- "$patch_input"' EXIT
cat > "$patch_input"
"$FLA_NPU_PATCH_PYTHON" -m patch_ng "$@" "$patch_input"
EOF
    chmod +x "$patch_wrapper"
fi
for build_command in cmake make patch; do
    command -v "$build_command" >/dev/null 2>&1 || {
        echo "Required build command not found: $build_command" >&2
        exit 1
    }
done

export FLA_NPU_SOC="$soc"
export FLA_NPU_OPS="$ops"
export FLA_NPU_BUILD_LEGACY_EXTENSION="FALSE"
export TMPDIR="$tmp_dir"
export TORCH_EXTENSIONS_DIR="$run_dir/torch_extensions"
export ASCEND_RT_VISIBLE_DEVICES="$device"
[[ -z "$model_source_root" ]] || export FLA_NPU_MODEL_SOURCE_ROOT="$model_source_root"
mkdir -p -- "$TORCH_EXTENSIONS_DIR"
"$venv_python" "$source_dir/scripts/check_npu_env.py" --build-only

echo "[5/6] Building and installing the scoped wheel"
(
    cd "$source_dir"
    "$venv_python" -m pip wheel --no-build-isolation --no-deps . -w "$wheel_dir"
)
mapfile -t wheels < <(find "$wheel_dir" -maxdepth 1 -type f -name 'flash_linear_attention_npu-*.whl' -print)
if ((${#wheels[@]} != 1)); then
    echo "Expected one wheel, found ${#wheels[@]}" >&2
    exit 1
fi
"$venv_python" -m pip install --force-reinstall --no-deps "${wheels[0]}"
"$venv_python" "$source_dir/scripts/check_packaged_wheel_api.py"

echo "[6/6] Running A5 KDA acceptance cases"
runner=(
    "$runner_script"
    --repo-dir "$source_dir"
    --output-dir "$result_dir"
    --repo-commit "$commit"
    --soc "$soc"
    --device-visible-id 0
    --cases "$case_filter"
    --case-timeout "$case_timeout"
    --profile-launch-count "$profile_launch_count"
    --profile-warm-up "$profile_warm_up"
)
"$venv_python" "${runner[@]}"

trap - ERR
echo "Completed. Results: $result_dir/results.md"
