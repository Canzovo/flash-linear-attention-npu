"""ATK executor for the recurrent_gated_delta_rule Ascend C operator."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _int_tensor,
    _marker_device,
    _orig_dtype,
    _rand,
    _randn,
)
from reference_recurrent_gated_delta_rule import (
    recurrent_gated_delta_rule_bench,
    recurrent_gated_delta_rule_golden,
)


OP_NAME = "recurrent_gated_delta_rule"


def _flat_storage(tensor: torch.Tensor) -> torch.Tensor:
    count = tensor.untyped_storage().nbytes() // tensor.element_size()
    return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (count,), (1,))


def _move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    storages = {}
    moved = {}
    for name, tensor in inputs.items():
        if not isinstance(tensor, torch.Tensor):
            moved[name] = tensor
            continue
        identity = (tensor.untyped_storage().data_ptr(), tensor.dtype)
        if identity not in storages:
            storages[identity] = _flat_storage(tensor).to(device)
        moved[name] = storages[identity].as_strided(
            tensor.shape, tensor.stride(), tensor.storage_offset())
    return moved


def _operator_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {name: tensor for name, tensor in inputs.items() if not name.startswith("_")}


def _int_list(spec: dict[str, Any], name: str) -> list[int]:
    values = spec.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{OP_NAME}: {name} must be a non-empty integer list.")
    return [int(value) for value in values]


def _build_state(
    spec: dict[str, Any],
    shape: tuple[int, int, int, int],
    dtype_name: str,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    dense_state = _randn(
        shape,
        dtype_name,
        dtype,
        device,
        seed,
        scale=0.01,
    )
    state_layout = str(spec.get("state_layout", "contiguous"))
    if state_layout == "contiguous":
        return dense_state
    block_num, value_heads, value_dim, key_dim = shape
    if state_layout == "noncontiguous":
        storage = torch.empty(
            (value_heads, block_num, value_dim, key_dim),
            dtype=dense_state.dtype,
            device=device,
        )
        state = storage.permute(1, 0, 2, 3)
        state.copy_(dense_state)
        return state

    if state_layout == "interleaved":
        # Slot and head axes interleave while each 128x128 matrix is dense.
        storage = torch.full((value_heads, block_num, value_dim, key_dim),
                             0.625, dtype=dtype, device=device)
        state = storage.permute(1, 0, 2, 3)
        state.copy_(dense_state)
        return state

    if state_layout not in {
        "head_padded",
        "block_padded",
        "head_block_padded",
        "row_padded",
    }:
        raise ValueError(f"{OP_NAME}: unsupported state_layout {state_layout!r}.")

    head_padding = key_dim if state_layout in {"head_padded", "head_block_padded"} else 0
    block_padding = (
        value_dim * key_dim
        if state_layout in {"block_padded", "head_block_padded"}
        else 0
    )
    row_stride = key_dim + (16 if state_layout == "row_padded" else 0)
    head_stride = value_dim * row_stride + head_padding
    block_stride = value_heads * head_stride + block_padding
    storage_offset = key_dim + 1 if state_layout == "head_block_padded" else 0
    storage_size = (
        storage_offset
        + (block_num - 1) * block_stride
        + (value_heads - 1) * head_stride
        + (value_dim - 1) * row_stride + key_dim
    )
    storage = torch.full((storage_size,), 0.625, dtype=dense_state.dtype, device=device)
    state = torch.as_strided(
        storage,
        shape,
        (block_stride, head_stride, row_stride, 1),
        storage_offset=storage_offset,
    )
    state.copy_(dense_state)
    return state


def _make_noncontiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or tensor.ndim == 0:
        return tensor
    storage_shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
    storage = torch.empty(storage_shape, dtype=tensor.dtype, device=tensor.device)
    view = storage[..., ::2]
    view.copy_(tensor)
    return view


def _apply_input_layout(
    inputs: dict[str, Any], layout: str
) -> dict[str, Any]:
    layout_names = {
        "contiguous": (),
        "token_head_transposed": ("query", "key", "value"),
        "overlapping_qkv": ("query", "key", "value"),
        "mixed_qkv": ("query", "key", "value"),
        "noncontiguous_qkv_beta_g": ("query", "key", "value", "beta", "g"),
        "noncontiguous_gk_metadata": (
            "gk",
            "actual_seq_lengths",
            "ssm_state_indices",
            "num_accepted_tokens",
        ),
        "noncontiguous_qk": ("query", "key"),
        "noncontiguous_v_gates": ("value", "beta", "g", "gk"),
        "noncontiguous_all": (
            "query",
            "key",
            "value",
            "beta",
            "g",
            "gk",
            "actual_seq_lengths",
            "ssm_state_indices",
            "num_accepted_tokens",
            "A_log",
            "dt_bias",
        ),
    }
    if layout not in layout_names:
        raise ValueError(f"{OP_NAME}: unsupported input_layout {layout!r}.")
    if layout == "mixed_qkv":
        query, key, value = (inputs[name] for name in ("query", "key", "value"))
        tokens, key_heads, dim = query.shape
        value_heads = value.shape[1]
        storage = torch.full((tokens + 2, 2 * key_heads + value_heads + 1, dim),
                             0.625, dtype=query.dtype, device=query.device)
        for name, begin, count in (("query", 0, key_heads), ("key", key_heads, key_heads),
                                   ("value", 2 * key_heads, value_heads)):
            view = storage[1:tokens + 1, begin:begin + count]
            view.copy_(inputs[name])
            inputs[name] = view
        return inputs
    if layout == "overlapping_qkv":
        tokens = inputs["query"].shape[0]
        heads = max(inputs[name].shape[1] for name in ("query", "key", "value"))
        storage = _randn((tokens + heads + 2, 128), "bf16", torch.bfloat16,
                         inputs["query"].device, int(inputs["_seed"]) + 10)
        for offset, name in enumerate(("query", "key", "value")):
            inputs[name] = storage.as_strided(inputs[name].shape, (128, 128, 1), offset * 128)
        return inputs
    for name in layout_names[layout]:
        if layout == "token_head_transposed":
            inputs[name] = inputs[name].permute(1, 0, 2).contiguous().permute(1, 0, 2)
        elif name in {"query", "key", "value"}:
            # QKV preserve a contiguous feature axis. Only outer strides and
            # offsets vary; a feature-stride-2 view is a negative case.
            tensor = inputs[name]
            storage = torch.full((tensor.shape[0] * 2 + 2, tensor.shape[1] + 1, 128),
                                 0.625, dtype=tensor.dtype, device=tensor.device)
            view = storage[1:2 * tensor.shape[0] + 1:2, :tensor.shape[1]]
            view.copy_(tensor)
            inputs[name] = view
        else:
            inputs[name] = _make_noncontiguous(inputs[name])
    return inputs


def _add_head_offsets(
    tensor: torch.Tensor, scale: float, dtype_name: str
) -> None:
    """Add zero-centered head offsets with magnitude bounded by scale."""
    storage_dtype = _orig_dtype(dtype_name)
    quantized = tensor.to(storage_dtype)
    head_count = tensor.shape[1]
    if head_count == 1:
        offsets = torch.zeros(1, device=tensor.device, dtype=storage_dtype)
    else:
        offsets = torch.linspace(
            -scale,
            scale,
            steps=head_count,
            dtype=torch.float32,
        ).to(storage_dtype).to(tensor.device)
    quantized.add_(
        offsets.view(1, -1, *([1] * (tensor.ndim - 2)))
    )
    tensor.copy_(quantized.to(tensor.dtype))


def _quantize_input_(tensor: torch.Tensor, dtype_name: str) -> None:
    tensor.copy_(tensor.to(_orig_dtype(dtype_name)).to(tensor.dtype))


def build_inputs(
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    # Construct every number on CPU for all three roles. Transfer complete
    # backing storages afterwards so legal views, offsets and QKV aliases
    # remain identical on CPU and DUT.
    if device.type != "cpu":
        return _move_inputs(build_inputs(spec, torch.device("cpu")), device)
    seed = int(spec.get("seed", 20260817))
    seq_lengths = _int_list(spec, "seq_lengths")
    prefix_tokens = int(spec.get("prefix_tokens", 0))
    total_tokens = prefix_tokens + sum(seq_lengths)
    if total_tokens != int(spec["T"]):
        raise ValueError(
            f"{OP_NAME}: T must equal prefix_tokens + sum(seq_lengths), "
            f"got T={spec['T']} and computed {total_tokens}."
        )

    key_heads = int(spec["HK"])
    value_heads = int(spec["HV"])
    key_dim = int(spec["K"])
    value_dim = int(spec["V"])
    block_num = int(spec["block_num"])
    state_dtype_name = str(spec.get("state_dtype", "bf16"))
    state_dtype = _orig_dtype(state_dtype_name)
    low_precision_dtype = torch.bfloat16
    state_calc_dtype = state_dtype
    beta_dtype_name = str(spec.get("beta_dtype", "bf16"))
    gate_dtype_name = str(spec.get("g_dtype", "fp32"))
    beta_dtype = _orig_dtype(beta_dtype_name)
    gate_calc_dtype = _orig_dtype(gate_dtype_name)

    table_width = int(spec.get("state_indices_width", 0))
    if table_width:
        if "state_indices" in spec and isinstance(spec["state_indices"][0], list):
            state_indices = spec["state_indices"]
        else:
            state_indices = [list(range(b * table_width, (b + 1) * table_width))
                             for b in range(len(seq_lengths))]
    else:
        state_indices = [int(value) for value in spec.get("state_indices", list(range(total_tokens)))]
    if not table_width and len(state_indices) != total_tokens:
        raise ValueError(
            f"{OP_NAME}: state_indices must contain T entries, "
            f"got {len(state_indices)} for T={total_tokens}."
        )

    gate_mode = str(spec.get("gate_mode", "g"))
    if gate_mode not in {"g", "gk", "both", "none"}:
        raise ValueError(f"{OP_NAME}: unsupported gate_mode {gate_mode!r}.")

    gate_profile = str(spec.get("gate_profile", "random"))
    gate_min, gate_max = 0.001, 0.02
    if gate_profile == "near_zero":
        gate_min, gate_max = 0.000001, 0.0001
    elif gate_profile == "strong_decay":
        gate_min, gate_max = 0.5, 2.0

    g = None
    if gate_mode in {"g", "both"}:
        g = -_rand(
            (total_tokens, value_heads),
            gate_dtype_name,
            gate_calc_dtype,
            device,
            seed + 6,
            gate_min,
            gate_max,
        )
        if gate_profile == "per_head":
            values = torch.linspace(
                -0.001,
                -0.02,
                value_heads,
                dtype=torch.float32,
                device=device,
            ).to(gate_calc_dtype)
            g.copy_(values.view(1, -1).expand_as(g))
            _quantize_input_(g, "fp32")
        if spec.get("use_gate_in_kernel", False):
            g = _randn((total_tokens, value_heads), gate_dtype_name, gate_calc_dtype,
                       device, seed + 6, scale=2.0)

    gk = None
    if gate_mode in {"gk", "both"}:
        gk = -_rand(
            (total_tokens, value_heads, key_dim),
            "fp32",
            gate_calc_dtype,
            device,
            seed + 7,
            gate_min,
            gate_max,
        )
        if gate_profile == "column_pulse":
            gk.fill_(-0.001)
            for value_head in range(value_heads):
                gk[:, value_head, (value_head * 17) % key_dim] = -0.05
            _quantize_input_(gk, "fp32")

    accepted_tokens = spec.get("accepted_tokens")
    if accepted_tokens is not None:
        accepted_tokens = [int(value) for value in accepted_tokens]
        if len(accepted_tokens) != len(seq_lengths):
            raise ValueError(
                f"{OP_NAME}: accepted_tokens must contain B entries, "
                f"got {len(accepted_tokens)} for B={len(seq_lengths)}."
            )

    inputs = {
        "query": _randn(
            (total_tokens, key_heads, key_dim),
            "bf16",
            low_precision_dtype,
            device,
            seed + 1,
        ),
        "key": _randn(
            (total_tokens, key_heads, key_dim),
            "bf16",
            low_precision_dtype,
            device,
            seed + 2,
        ),
        "value": _randn(
            (total_tokens, value_heads, value_dim),
            "bf16",
            low_precision_dtype,
            device,
            seed + 3,
        ),
        "state": _build_state(
            spec,
            (block_num, value_heads, value_dim, key_dim),
            state_dtype_name,
            state_calc_dtype,
            device,
            seed + 4,
        ),
        "beta": _rand(
            (total_tokens, value_heads),
            beta_dtype_name,
            beta_dtype,
            device,
            seed + 5,
            0.1,
            0.9,
        ),
        "g": g,
        "gk": gk,
        "actual_seq_lengths": _int_tensor(
            [prefix_tokens, *seq_lengths], device, torch.int32
        ),
        "ssm_state_indices": _int_tensor(state_indices, device, torch.int32),
        "num_accepted_tokens": (
            None
            if accepted_tokens is None
            else _int_tensor(accepted_tokens, device, torch.int32)
        ),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(key_dim))),
        "_seed": seed,
        "use_qk_l2norm_in_kernel": bool(spec.get("use_qk_l2norm_in_kernel", False)),
        "use_gate_in_kernel": bool(spec.get("use_gate_in_kernel", False)),
        "use_beta_sigmoid_in_kernel": bool(spec.get("use_beta_sigmoid_in_kernel", False)),
        "allow_neg_eigval": bool(spec.get("allow_neg_eigval", False)),
        "A_log": (_randn((value_heads,), "fp32", torch.float32, device, seed + 8, scale=0.5)
                  if spec.get("use_gate_in_kernel", False) else None),
        "dt_bias": (_randn((value_heads,), "fp32", torch.float32, device, seed + 9, scale=0.2)
                    if spec.get("use_gate_in_kernel", False) and spec.get("use_dt_bias", True) else None),
    }

    if str(spec.get("data_profile", "random")) == "traceable_gva":
        _add_head_offsets(inputs["query"], 0.01, "bf16")
        _add_head_offsets(inputs["key"], 0.02, "bf16")
        _add_head_offsets(inputs["value"], 0.005, "bf16")

    if str(spec.get("beta_profile", "random")) == "per_head":
        values = torch.linspace(
            0.1,
            0.9,
            value_heads,
            dtype=torch.float32,
            device=device,
        ).to(torch.bfloat16).to(low_precision_dtype)
        inputs["beta"].copy_(values.view(1, -1).expand_as(inputs["beta"]))
        _quantize_input_(inputs["beta"], beta_dtype_name)
    if spec.get("use_beta_sigmoid_in_kernel", False):
        inputs["beta"] = _randn((total_tokens, value_heads), beta_dtype_name, beta_dtype,
                                device, seed + 5, scale=3.0)

    if str(spec.get("beta_profile", "random")) == "extreme":
        inputs["beta"].copy_(torch.linspace(-100,100,value_heads,device=device).bfloat16().expand_as(inputs["beta"]))
    if str(spec.get("beta_profile", "random")) == "zero":
        inputs["beta"].zero_()
    if gate_profile == "cancelling_tail":
        magnitudes = torch.linspace(12, 25, value_heads, device=device)
        inputs["g"].copy_((-magnitudes).expand_as(inputs["g"]))
        inputs["A_log"].copy_(magnitudes)
        if inputs["dt_bias"] is not None:
            inputs["dt_bias"].zero_()
    if gate_profile == "extreme" and inputs["g"] is not None:
        inputs["g"].copy_(torch.linspace(-100,100,value_heads,device=device).expand_as(inputs["g"]))
    if spec.get("data_profile") == "zero_qk":
        inputs["query"].zero_()
        inputs["key"].zero_()

    state_profile = str(spec.get("state_profile", "random"))
    if state_profile.startswith("pulse_hv"):
        pulse_head = int(state_profile.removeprefix("pulse_hv"))
        inputs["state"].zero_()
        inputs["state"][:, pulse_head, 0, 0] = 0.25
    elif state_profile == "traceable":
        _add_head_offsets(inputs["state"], 0.0001, state_dtype_name)
    if state_profile != "random":
        _quantize_input_(inputs["state"], state_dtype_name)

    return _apply_input_layout(
        inputs, str(spec.get("input_layout", "contiguous"))
    )


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    previous_threads = torch.get_num_threads()
    try:
        # ATK runs several CPU workers concurrently. One Torch thread per worker
        # avoids severe nested-parallelism overhead for the per-head reference.
        torch.set_num_threads(1)
        inputs = build_inputs(spec, torch.device("cpu"))
        outputs = None
        repeat_calls = int(spec.get("repeat_calls", 1))
        reference = (recurrent_gated_delta_rule_golden if high_precision
                     else recurrent_gated_delta_rule_bench)
        for call_index in range(repeat_calls):
            outputs = reference(
                query=inputs["query"],
                key=inputs["key"],
                value=inputs["value"],
                state=inputs["state"],
                beta=inputs["beta"],
                scale=inputs["scale"],
                actual_seq_lengths=inputs["actual_seq_lengths"],
                ssm_state_indices=inputs["ssm_state_indices"],
                num_accepted_tokens=inputs["num_accepted_tokens"],
                g=inputs["g"],
                gk=inputs["gk"],
                use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
                use_gate_in_kernel=inputs["use_gate_in_kernel"],
                use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
                allow_neg_eigval=inputs["allow_neg_eigval"],
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
            )
            if call_index + 1 < repeat_calls:
                inputs["state"].copy_(outputs[1])
        if outputs is None:
            raise RuntimeError(f"{OP_NAME}: repeat_calls must be positive.")
        return outputs
    finally:
        torch.set_num_threads(previous_threads)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))
    before_storage = _flat_storage(inputs["state"]).detach().cpu().clone()
    from fla_npu.ops import ascendc

    out = None
    for _ in range(int(spec.get("repeat_calls", 1))):
        out = ascendc.recurrent_gated_delta_rule(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["state"],
            beta=inputs["beta"],
            scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"],
            num_accepted_tokens=inputs["num_accepted_tokens"],
            g=inputs["g"],
            gk=inputs["gk"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
            use_gate_in_kernel=inputs["use_gate_in_kernel"],
            use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=inputs["allow_neg_eigval"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
        )
    if out is None:
        raise RuntimeError(f"{OP_NAME}: repeat_calls must be positive.")
    backend = getattr(ascendc, "BACKENDS", {}).get("npu_recurrent_gated_delta_rule")
    if backend != "stable":
        raise RuntimeError(f"{OP_NAME}: Stable backend required, got {backend!r}.")
    prefix_tokens = int(inputs["actual_seq_lengths"][0].item())
    if prefix_tokens > 0:
        out[:prefix_tokens].zero_()
    state = inputs["state"]
    actual_storage = _flat_storage(state).detach().cpu()
    expected_storage = before_storage.clone()
    expected_view = expected_storage.as_strided(state.shape, state.stride(), state.storage_offset())
    expected_view.copy_(state.detach().cpu())
    if not torch.equal(actual_storage, expected_storage):
        raise RuntimeError("State storage padding or offset guard changed")
    return out, inputs["state"]


@register("executor_recurrent_gated_delta_rule")
class FunctionApi(BaseApi):
    """ATK execution entry point."""

    def __init__(self, task_result):
        super().__init__(task_result)
        # ATK automatically schedules a separate CPU golden invocation when
        # --bm_device cpu is set. The regular CPU node is the FP32 benchmark.
        self.high_precision = self.device == "cpu" and bool(task_result.is_benchmark_task)
        self.recheck_case_id = (
            int(task_result.case_config.id)
            if bool(getattr(task_result, "disable_id_seed", False))
            and int(getattr(task_result, "loop_nums", 1)) > 1 else None)

    def _run_spec(self, input_data: InputDataset):
        spec = _case_spec(input_data, OP_NAME)
        if self.recheck_case_id is not None:
            # ATK gives all three roles the same runtime id for each LT loop.
            # Its process-global torch seed does not change between loops.
            # Only the numeric materialization seed changes; all structural
            # metadata and frozen input ranges remain in the original spec.
            spec = dict(spec, seed=(int(spec["seed"]) +
                                   1000000007 * self.recheck_case_id) % (2**63 - 1))
        return spec

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        spec = self._run_spec(input_data)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, high_precision=self.high_precision)
        else:
            raise RuntimeError(
                f"{OP_NAME} requires an NPU DUT node and a CPU reference node, "
                f"got {self.device!r}."
            )
        return _finite_tuple(outputs, golden=self.device == "cpu")

    def export_custom_data(self, input_data: InputDataset):
        spec = _case_spec(input_data, OP_NAME)
        return {
            "design_id": str(spec["design_id"]),
            "case_name": str(spec["name"]),
            "B": int(spec["B"]),
            "T": int(spec["T"]),
            "HK": int(spec["HK"]),
            "HV": int(spec["HV"]),
            "K": int(spec["K"]),
            "V": int(spec["V"]),
            "state_dtype": str(spec["state_dtype"]),
            "state_layout": str(spec.get("state_layout", "contiguous")),
            "input_layout": str(spec.get("input_layout", "contiguous")),
            "gate_mode": str(spec["gate_mode"]),
        }


@register("executor_recurrent_gated_delta_rule_perf")
class PerformanceApi(FunctionApi):
    """Profile a fixed input while restoring State before every invocation."""

    def __init__(self, task_result):
        super().__init__(task_result)
        self.performance_spec = None
        self.performance_inputs = None
        self.initial_state = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        if self.device not in {"npu", "pyaclnn"}:
            raise RuntimeError("The performance entry requires an NPU node.")
        spec = self._run_spec(input_data)
        if spec != self.performance_spec:
            self.performance_inputs = build_inputs(spec, _marker_device(input_data))
            self.initial_state = self.performance_inputs["state"].clone()
            self.performance_spec = spec
        self.performance_inputs["state"].copy_(self.initial_state)
        from fla_npu.ops import ascendc

        out = ascendc.recurrent_gated_delta_rule(**_operator_kwargs(self.performance_inputs))
        backend = getattr(ascendc, "BACKENDS", {}).get("npu_recurrent_gated_delta_rule")
        if backend != "stable":
            raise RuntimeError(f"{OP_NAME}: Stable backend required, got {backend!r}.")
        # Accuracy/storage guards belong to FunctionApi. Returning only the
        # small attention output avoids profiling or exporting the full State.
        return (out,)


def _negative_inputs(spec):
    inputs = build_inputs(spec, torch.device("cpu"))
    kind = spec["negative_kind"]
    if kind == "gk":
        inputs["gk"] = torch.zeros(spec["T"], spec["HV"], 128, dtype=torch.float32)
    elif kind == "qkv_last_stride":
        inputs["query"] = _make_noncontiguous(inputs["query"])
    elif kind == "state_overlap":
        state = inputs["state"]
        inputs["state"] = state.as_strided(state.shape, (128, 128, 128, 1))
    elif kind == "state_input_alias":
        inputs["state"] = inputs["state"].bfloat16()
        inputs["query"] = inputs["state"].reshape(-1)[:spec["T"] * spec["HK"] * 128].view(spec["T"], spec["HK"], 128)
    elif kind in {"cross_request_write", "cross_request_read"}:
        inputs["actual_seq_lengths"] = torch.tensor([0, 5, 4], dtype=torch.int32)
        if kind == "cross_request_write":
            inputs["ssm_state_indices"][5] = inputs["ssm_state_indices"][0]
        else:
            inputs["ssm_state_indices"] = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 8, 8]], dtype=torch.int32)
            inputs["num_accepted_tokens"] = torch.tensor([6, 1], dtype=torch.int32)
    elif kind == "slot_range":
        inputs["ssm_state_indices"][0] = inputs["state"].shape[0]
    elif kind == "accepted_range":
        inputs["num_accepted_tokens"] = torch.tensor([spec["T"] + 1], dtype=torch.int32)
    elif kind == "bool_attr":
        inputs["use_qk_l2norm_in_kernel"] = 1
    elif kind == "scale_nan":
        inputs["scale"] = float("nan")
    elif kind == "raw_gate_no_g":
        inputs["use_gate_in_kernel"] = True
        inputs["g"] = None
        inputs["A_log"] = torch.zeros(spec["HV"], dtype=torch.float32)
    elif kind == "raw_gate_no_A_log":
        inputs["use_gate_in_kernel"] = True
        inputs["A_log"] = None
    elif kind == "query_dtype":
        inputs["query"] = inputs["query"].float()
    elif kind == "beta_dtype":
        inputs["beta"] = inputs["beta"].half()
    elif kind == "metadata_dtype":
        inputs["actual_seq_lengths"] = inputs["actual_seq_lengths"].long()
    return inputs


@register("executor_recurrent_gated_delta_rule_negative")
class NegativeFunctionApi(BaseApi):
    """ATK run task: verify rejection before State mutation."""

    def __call__(self, input_data, with_output=False):
        del with_output
        spec = _case_spec(input_data, OP_NAME)
        inputs = _negative_inputs(spec)
        if self.device in {"npu", "pyaclnn"}:
            inputs = _move_inputs(inputs, _marker_device(input_data))
            from fla_npu.ops.ascendc import recurrent_gated_delta_rule as operator
        elif self.device == "cpu":
            operator = recurrent_gated_delta_rule_bench
        else:
            raise RuntimeError("negative checks require CPU or NPU")
        state = inputs["state"]
        before = _flat_storage(state).detach().cpu().clone()
        version = state._version
        try:
            operator(**_operator_kwargs(inputs))
        except (ValueError, TypeError):
            if state._version != version or not torch.equal(_flat_storage(state).detach().cpu(), before):
                raise RuntimeError("rejected input modified State")
            return (torch.ones(1),)
        raise RuntimeError(f"{spec['negative_kind']}: invalid input was accepted")
