"""ATK CPU/NPU executor and accuracy metrics for recurrent_kda decode."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from fla_npu.ops.ascendc import recurrent_kda as npu_recurrent_kda

try:
    import torch.distributed.tensor as _torch_dtensor
    if not hasattr(_torch_dtensor, "DTensor"):
        class DTensor:
            pass

        _torch_dtensor.DTensor = DTensor
except Exception:
    pass

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import AccuracyConfig, TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare


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
RTOL = 0.02
ATOL = 0.01


def _case_attr(case_config, name):
    for input_config in case_config.inputs:
        if input_config.name == name:
            return input_config.range_values
    raise KeyError(name)


@ACCURACY_REGISTRY.register("recurrent_kda_metrics")
class RecurrentKdaMetrics(BaseAccuracyCompare):
    """Compare NPU output with CPU reference and persist requested metrics."""

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        actual = local_output.detach().cpu().to(torch.float32)
        reference = remote_output.detach().cpu().to(torch.float32)
        if actual.shape != reference.shape:
            return AccuracyConfig(
                filename=data_file,
                result=False,
                error_info=(
                    f"shape mismatch: actual={tuple(actual.shape)}, "
                    f"reference={tuple(reference.shape)}"
                ),
            )

        diff = (actual - reference).abs()
        relative = diff / reference.abs().clamp_min(1e-6)
        max_relative_error = relative.max().item() if relative.numel() else 0.0
        mean_relative_error = relative.mean().item() if relative.numel() else 0.0
        rmse = torch.sqrt(torch.mean((actual - reference) ** 2)).item() if diff.numel() else 0.0
        max_absolute_error = diff.max().item() if diff.numel() else 0.0
        passed = bool(torch.allclose(actual, reference, rtol=RTOL, atol=ATOL, equal_nan=True))

        record = {
            "case_id": int(self.case_config.id),
            "batch": int(_case_attr(self.case_config, "batch")),
            "mode": str(_case_attr(self.case_config, "mode")),
            "seed": int(_case_attr(self.case_config, "seed")),
            "max_relative_error": max_relative_error,
            "mean_relative_error": mean_relative_error,
            "rmse": rmse,
            "max_absolute_error": max_absolute_error,
            "passed": passed,
        }
        metrics_path = Path(
            os.environ.get(
                "RECURRENT_KDA_METRICS_FILE",
                Path(__file__).resolve().parent / "results" / "accuracy_metrics.jsonl",
            )
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, sort_keys=True) + chr(10))

        return AccuracyConfig(
            filename=data_file,
            result=passed,
            error_info=(
                f"allclose={passed}, rtol={RTOL}, atol={ATOL}, "
                f"max_relative_error={max_relative_error:.8e}, "
                f"mean_relative_error={mean_relative_error:.8e}, "
                f"rmse={rmse:.8e}, max_absolute_error={max_absolute_error:.8e}"
            ),
        )


def _randn(generator, shape, dtype, scale=1.0):
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return value.to(dtype).contiguous()


def _build_cpu_inputs(batch: int, mode: str, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q_shape = (batch, DECODE_STEP, HEADS, KEY_DIM)
    v_shape = (batch, DECODE_STEP, HEADS, VALUE_DIM)
    gate_shape = (batch, DECODE_STEP, HEADS, KEY_DIM)
    beta_shape = (batch, DECODE_STEP, HEADS)

    inputs = {
        "q": _randn(generator, q_shape, torch.bfloat16),
        "k": _randn(generator, q_shape, torch.bfloat16),
        "v": _randn(generator, v_shape, torch.bfloat16),
        "g": _randn(generator, gate_shape, torch.float32, scale=0.5),
        "beta": _randn(generator, beta_shape, torch.float32),
        "initial_state": _randn(
            generator,
            (batch, HEADS, VALUE_DIM, KEY_DIM),
            torch.bfloat16,
            scale=0.02,
        ),
        "cu_seqlens": torch.arange(batch + 1, dtype=torch.int32) * DECODE_STEP,
        "A_log": _randn(generator, (HEADS,), torch.float32, scale=0.1),
        "dt_bias": _randn(generator, (HEADS, KEY_DIM), torch.float32, scale=0.1),
        "ssm_state_indices": None,
        "num_accepted_tokens": None,
    }
    if mode == "cb_mtp":
        inputs["ssm_state_indices"] = torch.arange(
            batch - 1, -1, -1, dtype=torch.int32
        ).reshape(batch, 1)
        inputs["num_accepted_tokens"] = torch.ones(batch, dtype=torch.int32)
    elif mode != "base":
        raise ValueError(f"unsupported mode: {mode}")
    return inputs


def _move_inputs(inputs, device):
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


@register("recurrent_kda_atk")
class RecurrentKdaAtkApi(BaseApi):
    """Execute the same deterministic case on CPU reference or NPU wrapper."""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.inputs = None
        self.batch = 0
        self.mode = ""

    def init_by_input_data(self, input_data: InputDataset):
        self.batch = int(input_data.kwargs["batch"])
        self.mode = str(input_data.kwargs["mode"])
        seed = int(input_data.kwargs.get("seed", FIXED_SEED))
        inputs = _build_cpu_inputs(self.batch, self.mode, seed)
        if self.device == "npu":
            import torch_npu

            device = torch.device(f"npu:{self.device_id}")
            torch_npu.npu.set_device(device)
            self._warmup_npu(self.batch, self.mode, device)
            inputs = _move_inputs(inputs, device)
        self.inputs = inputs

    @classmethod
    def _warmup_npu(cls, batch, mode, device):
        import torch_npu

        q_shape = (batch, DECODE_STEP, HEADS, KEY_DIM)
        warmup = {
            "q": torch.ones(q_shape, dtype=torch.bfloat16, device=device),
            "k": torch.zeros(q_shape, dtype=torch.bfloat16, device=device),
            "v": torch.zeros(
                (batch, DECODE_STEP, HEADS, VALUE_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
            "g": torch.zeros(q_shape, dtype=torch.float32, device=device),
            "beta": torch.zeros(
                (batch, DECODE_STEP, HEADS), dtype=torch.float32, device=device
            ),
            "initial_state": torch.zeros(
                (batch, HEADS, VALUE_DIM, KEY_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
            "cu_seqlens": torch.arange(
                batch + 1, dtype=torch.int32, device=device
            ) * DECODE_STEP,
            "A_log": torch.zeros((HEADS,), dtype=torch.float32, device=device),
            "dt_bias": torch.zeros(
                (HEADS, KEY_DIM), dtype=torch.float32, device=device
            ),
            "ssm_state_indices": None,
            "num_accepted_tokens": None,
        }
        if mode == "cb_mtp":
            warmup["ssm_state_indices"] = torch.arange(
                batch - 1, -1, -1, dtype=torch.int32, device=device
            ).reshape(batch, 1)
            warmup["num_accepted_tokens"] = torch.ones(
                batch, dtype=torch.int32, device=device
            )
        npu_recurrent_kda(
            warmup["q"],
            warmup["k"],
            warmup["v"],
            warmup["g"],
            warmup["beta"],
            warmup["initial_state"],
            **cls._common_kwargs(warmup),
        )
        torch_npu.npu.synchronize()

    @staticmethod
    def _common_kwargs(inputs):
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

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del input_data, with_output
        inputs = self.inputs
        kwargs = self._common_kwargs(inputs)
        if self.device == "npu":
            output, _ = npu_recurrent_kda(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                inputs["initial_state"],
                **kwargs,
            )
            return output

        output, _ = recurrent_kda_reference(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["initial_state"],
            **kwargs,
        )
        return output

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        inputs = self.inputs
        tensor_inputs = [
            inputs["q"], inputs["k"], inputs["v"], inputs["g"],
            inputs["beta"], inputs["initial_state"], inputs["cu_seqlens"],
            inputs["A_log"], inputs["dt_bias"], inputs["ssm_state_indices"],
            inputs["num_accepted_tokens"],
        ]
        read_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensor_inputs
            if isinstance(tensor, torch.Tensor)
        )
        output_bytes = inputs["v"].numel() * inputs["v"].element_size()
        state_write_bytes = (
            inputs["initial_state"].numel() * inputs["initial_state"].element_size()
        )

        # Lower-bound arithmetic model for one decode token per sequence.
        # It counts recurrent state decay, two state-vector products, delta,
        # outer-product update, q/k normalization and query scaling. Exp/sigmoid
        # transcendental instructions are intentionally excluded.
        flops_per_head = (
            7 * KEY_DIM * VALUE_DIM + 2 * VALUE_DIM + 7 * KEY_DIM
        )
        calc_flops = float(self.batch * HEADS * flops_per_head)
        mib = 1024.0 * 1024.0
        return {
            "read_bytes": read_bytes / mib,
            "write_bytes": (output_bytes + state_write_bytes) / mib,
            "calc_flops_power": calc_flops,
        }

