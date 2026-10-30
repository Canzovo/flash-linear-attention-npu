"""Trace ATK roles, comparison direction and seeds without operator calls."""
from __future__ import annotations

import json
import os
import hashlib
import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.executors.compare_excutor import CompareExecutor


_get_compare_executor = CompareExecutor._get_compare_executor


def _trace_compare_executor(self):
    comparator = _get_compare_executor(self)
    print("RGDN_ATK_COMPARATOR " + type(comparator).__name__, flush=True)
    return comparator


CompareExecutor._get_compare_executor = _trace_compare_executor


@register("rgdn_dual_role_probe")
class FunctionApi(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        self.role = dict(name=task_result.name, backend=task_result.backend,
                         device=self.device,
                         is_benchmark_task=task_result.is_benchmark_task,
                         benchmark_device=task_result.benchmark_device,
                         case_id=task_result.case_config.id)
        self.materializer = None
        if os.environ.get("RGDN_PROBE_NUMERIC_SEEDS"):
            sys.path.insert(0, os.environ.get("RGDN_EXECUTOR_DIR", str(Path(__file__).resolve().parents[1])))
            from executor_recurrent_gated_delta_rule import FunctionApi as NumericApi
            self.materializer = NumericApi(task_result)
        print("RGDN_ATK_ROLE " + json.dumps(self.role, sort_keys=True), flush=True)

    def __call__(self, input_data, with_output=False):
        del with_output
        if self.materializer is not None:
            from executor_recurrent_gated_delta_rule import build_inputs
            spec = self.materializer._run_spec(input_data)
            inputs = build_inputs(spec, torch.device("cpu"))
            meta = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    meta[name] = dict(dtype=str(value.dtype), shape=list(value.shape),
                                      stride=list(value.stride()), offset=value.storage_offset())
                    if name in {"actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens"}:
                        meta[name]["values"] = value.tolist()
                elif name != "_seed":
                    meta[name] = value
            digest = lambda data: hashlib.sha256(data).hexdigest()
            print("RGDN_ATK_NUMERIC " + json.dumps(dict(self.role, seed=spec["seed"],
                  metadata=digest(json.dumps(meta, sort_keys=True).encode()),
                  query=digest(inputs["query"].contiguous().view(torch.uint8).numpy().tobytes()))), flush=True)
        # Distinct traceable values prove which node is evaluated against the
        # other node and the automatically inserted high precision golden.
        dut_value, bench_value = ((2.0, 1.01) if self.role["case_id"] == 0 else (1.01, 2.0))
        value = (1.0 if self.role["is_benchmark_task"] else
                 dut_value if "dut" in self.role["name"] else bench_value)
        print("RGDN_ATK_VALUE " + json.dumps(dict(self.role, value=value,
              torch_seed=torch.initial_seed())), flush=True)
        return (torch.full((128,), value, dtype=torch.float32),)


if __name__ == "__main__":
    case = dict(id=0, name="rgdn_dual_role_probe", api="pytorch",
                api_type="rgdn_dual_role_probe", version="v2.1", backward=False,
                standard={"acc": {"cv_fused_double_benchmark": {
                    "max_re_ratio": 5, "avg_re_ratio": 1.5,
                    "root_mean_squared_ratio": 1.5}}, "perf": "not_key"},
                inputs=[dict(name="marker", type="tensor", required=True,
                             dtype="fp32", shape=[1], range_values=[0, 0],
                             backward=False)])
    output = Path(os.environ.get("RGDN_PROBE_CASE", "probe_atk_dual_roles.json"))
    other = dict(case, id=1)
    if os.environ.get("RGDN_PROBE_NUMERIC_SEEDS"):
        sys.path.insert(0, os.environ.get("RGDN_EXECUTOR_DIR", str(Path(__file__).resolve().parents[1])))
        from gen_recurrent_gated_delta_rule import build_specs
        frozen = build_specs()[864]
        for payload in (case, other):
            payload["inputs"] = payload["inputs"] + [dict(name="case_spec", type="attr", required=True,
                  dtype="string", range_values=json.dumps(frozen), backward=False)]
    output.write_text(json.dumps([case, other], indent=2) + "\n", encoding="utf-8")
