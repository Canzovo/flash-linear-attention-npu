"""Validate ATK roles, deterministic inputs and contract without NPU work."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import executor_recurrent_gated_delta_rule as executor
import gen_recurrent_gated_delta_rule as generator
import reference
from atk.configs.results_config import TaskResult
from atk.configs.case_config import CaseConfig
from atk.configs.dataset_config import InputDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-dual", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    specs = generator.build_specs()
    for spec in specs:
        inputs = executor.build_inputs(spec, torch.device("cpu"))
        reference._validate_inputs(**executor._operator_kwargs(inputs))
    for spec in generator.build_negative_specs():
        inputs = executor._negative_inputs(spec)
        state = inputs["state"]
        before = executor._flat_storage(state).clone()
        try:
            reference.recurrent_gated_delta_rule_bench(**executor._operator_kwargs(inputs))
        except (ValueError, TypeError):
            assert torch.equal(before, executor._flat_storage(state)), spec["negative_kind"]
        else:
            raise AssertionError("CPU accepted negative: " + spec["negative_kind"])
    roles = {}
    for golden in (False, True):
        task_result = TaskResult(name="cpu_bench", backend="cpu", device="cpu",
                                 is_benchmark_task=golden)
        api = executor.FunctionApi(task_result)
        assert api.high_precision is golden
        roles[str(golden)] = "FP64 golden" if api.high_precision else "FP32 bench"
    frozen = specs[864]
    dataset = InputDataset(kwargs={"case_spec": json.dumps(frozen)})
    loop_inputs = []
    for loop_id in (0, 815192147):
        role_inputs = []
        for golden in (False, True):
            payload = generator._case_payload(loop_id, frozen)
            task_result = TaskResult(name="cpu_bench", backend="cpu", device="cpu",
                                     is_benchmark_task=golden, disable_id_seed=True,
                                     loop_nums=2, case_config=CaseConfig(**payload))
            api = executor.FunctionApi(task_result)
            resampled = api._run_spec(dataset)
            assert {k: v for k, v in resampled.items() if k != "seed"} == {
                k: v for k, v in frozen.items() if k != "seed"}
            role_inputs.append(executor.build_inputs(resampled, torch.device("cpu")))
        for name, value in role_inputs[0].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, role_inputs[1][name]), name
        loop_inputs.append(role_inputs[0])
    for name, value in loop_inputs[0].items():
        other = loop_inputs[1][name]
        if isinstance(value, torch.Tensor):
            assert value.dtype == other.dtype and value.shape == other.shape, name
            assert value.stride() == other.stride(), name
            assert value.storage_offset() == other.storage_offset(), name
            if name in {"actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens"}:
                assert torch.equal(value, other), name
        elif name != "_seed":
            assert value == other, name
    assert not torch.equal(loop_inputs[0]["query"], loop_inputs[1]["query"])
    ids = list(range(len(specs))) if args.all_dual else [
        0, 657, 663, 807, 822, 825, 828, 831, 834, 837, 840, 843,
        855, 858, 864, 870, 885, 888, 894, 903,
    ]
    for index in ids:
        spec = specs[index]
        first = executor.build_inputs(spec, torch.device("cpu"))
        second = executor.build_inputs(spec, torch.device("cpu"))
        for name, tensor in first.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, second[name]), (index, name)
                assert tensor.stride() == second[name].stride(), (index, name)
                assert tensor.storage_offset() == second[name].storage_offset(), (index, name)
        for golden in (False, True):
            out, state = executor.run_cpu(spec, high_precision=golden)
            assert out.dtype == torch.bfloat16
            assert state.dtype == first["state"].dtype
            assert torch.isfinite(out).all() and torch.isfinite(state).all()
            assert not torch.count_nonzero(out[:int(spec.get("prefix_tokens", 0))])
    print(json.dumps(dict(positive_contract=len(specs),
                          negative_contract=len(generator.build_negative_specs()),
                          dual_executions=2 * len(ids), roles=roles,
                          recheck_metadata_fixed=True, recheck_values_resampled=True), sort_keys=True))


if __name__ == "__main__":
    main()
