"""ATK CPU/NPU/GPU executor for recurrent_kda decode accuracy and performance."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch

try:
    import torch.distributed.tensor as _torch_dtensor

    if not hasattr(_torch_dtensor, "DTensor"):
        class DTensor:
            pass

        _torch_dtensor.DTensor = DTensor
except Exception:
    pass

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


REPO_ROOT = Path(__file__).resolve().parents[7]
REFERENCE_DIR = REPO_ROOT / "tests" / "reference"
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

from recurrent_kda_reference import recurrent_kda_reference


HEADS = 96
KEY_DIM = 128
VALUE_DIM = 128
DECODE_STEP = 1
FIXED_SEED = 20260811
DEFAULT_GPU_CALLABLE = (
    "fla.ops.kda.fused_recurrent:fused_recurrent_kda_fwd"
)


def _randn(generator, shape, dtype, scale=1.0):
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return value.to(dtype).contiguous()


def _build_inputs(batch: int, mode: str, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q_shape = (batch, DECODE_STEP, HEADS, KEY_DIM)
    v_shape = (batch, DECODE_STEP, HEADS, VALUE_DIM)

    inputs = {
        "q": _randn(generator, q_shape, torch.bfloat16),
        "k": _randn(generator, q_shape, torch.bfloat16),
        "v": _randn(generator, v_shape, torch.bfloat16),
        "g": _randn(generator, q_shape, torch.float32, scale=0.5),
        "beta": _randn(
            generator,
            (batch, DECODE_STEP, HEADS),
            torch.float32,
        ),
        "initial_state": _randn(
            generator,
            (batch, HEADS, VALUE_DIM, KEY_DIM),
            torch.bfloat16,
            scale=0.02,
        ),
        "cu_seqlens": (
            torch.arange(batch + 1, dtype=torch.int64) * DECODE_STEP
        ),
        "A_log": _randn(generator, (HEADS,), torch.float32, scale=0.1),
        "dt_bias": _randn(
            generator,
            (HEADS, KEY_DIM),
            torch.float32,
            scale=0.1,
        ),
        "ssm_state_indices": None,
        "num_accepted_tokens": None,
    }
    if mode == "cb_mtp":
        inputs["ssm_state_indices"] = torch.arange(
            batch - 1,
            -1,
            -1,
            dtype=torch.int32,
        ).reshape(batch, 1)
        inputs["num_accepted_tokens"] = torch.ones(
            batch,
            dtype=torch.int32,
        )
    elif mode != "base":
        raise ValueError(f"unsupported mode: {mode}")

    return inputs


def _move_inputs(inputs, device):
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def _load_gpu_callable():
    target = os.environ.get(
        "RECURRENT_KDA_GPU_CALLABLE",
        DEFAULT_GPU_CALLABLE,
    ).strip()
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(
            "RECURRENT_KDA_GPU_CALLABLE must use "
            "'<python_module>:<callable>' syntax"
        )
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attribute, None)
    if not callable(callable_obj):
        raise RuntimeError(f"configured GPU target is not callable: {target}")
    return callable_obj


def _reference_kwargs(inputs):
    return {
        "cu_seqlens": inputs["cu_seqlens"],
        "ssm_state_indices": inputs["ssm_state_indices"],
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "num_accepted_tokens": inputs["num_accepted_tokens"],
        "layout": "BSND",
        "output_final_state": False,
        "inplace_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": False,
        "safe_gate": True,
        "lower_bound": -5.0,
        "state_v_first": True,
    }


def _run_gpu(inputs, gpu_callable):
    return gpu_callable(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        initial_state=inputs["initial_state"],
        scale=None,
        output_final_state=False,
        inplace_final_state=True,
        state_v_first=True,
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        lower_bound=-5.0,
    )


def _load_npu_callable():
    from fla_npu.ops.ascendc import recurrent_kda

    return recurrent_kda


def _run_npu(inputs, npu_callable):
    return npu_callable(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        **_reference_kwargs(inputs),
    )


@register("recurrent_kda_gpu_atk")
class RecurrentKdaGpuAtkApi(BaseApi):
    """Run the NPU DUT, GPU control, or canonical CPU benchmark."""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.inputs = None
        self.gpu_callable = None
        self.npu_callable = None
        self.batch = 0
        self.mode = ""
        task_names = {
            str(getattr(task_type, "value", task_type))
            for task_type in (task_result.task_type or [])
        }
        self.accuracy_task = bool({"accuracy", "accuracy_lt"} & task_names)

    def init_by_input_data(self, input_data: InputDataset):
        self.batch = int(input_data.kwargs["batch"])
        self.mode = str(input_data.kwargs["mode"])
        seed = int(input_data.kwargs.get("seed", FIXED_SEED))
        inputs = _build_inputs(self.batch, self.mode, seed)
        if self.device == "npu":
            import torch_npu

            device = torch.device(f"npu:{self.device_id}")
            torch_npu.npu.set_device(device)
            inputs["cu_seqlens"] = inputs["cu_seqlens"].to(torch.int32)
            inputs = _move_inputs(inputs, device)
            self.npu_callable = _load_npu_callable()
        elif self.device == "gpu":
            device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(device)
            inputs = _move_inputs(inputs, device)
            self.gpu_callable = _load_gpu_callable()
        self.inputs = inputs

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del input_data, with_output
        if self.device == "npu":
            output, _ = _run_npu(self.inputs, self.npu_callable)
        elif self.device == "gpu":
            output, _ = _run_gpu(self.inputs, self.gpu_callable)
        elif self.device == "cpu":
            output, _ = recurrent_kda_reference(
                self.inputs["q"],
                self.inputs["k"],
                self.inputs["v"],
                self.inputs["g"],
                self.inputs["beta"],
                self.inputs["initial_state"],
                **_reference_kwargs(self.inputs),
            )
        else:
            raise RuntimeError(
                "recurrent_kda ATK supports only cpu, npu, and gpu backends; "
                f"got {self.device!r}"
            )
        if self.accuracy_task:
            if not torch.isfinite(output).all().item():
                raise RuntimeError("recurrent_kda output contains NaN or Inf")
            return output.to(torch.float32)
        return output

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        inputs = self.inputs
        tensor_inputs = [
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["initial_state"],
            inputs["cu_seqlens"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["ssm_state_indices"],
            inputs["num_accepted_tokens"],
        ]
        read_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensor_inputs
            if isinstance(tensor, torch.Tensor)
        )
        output_bytes = (
            inputs["v"].numel() * inputs["v"].element_size()
        )
        state_write_bytes = (
            inputs["initial_state"].numel()
            * inputs["initial_state"].element_size()
        )
        calc_flops = float(
            5 * self.batch * HEADS * DECODE_STEP * KEY_DIM * VALUE_DIM
        )
        mib = 1024.0 * 1024.0
        return {
            "read_bytes": read_bytes / mib,
            "write_bytes": (output_bytes + state_write_bytes) / mib,
            "calc_flops_power": calc_flops,
        }
