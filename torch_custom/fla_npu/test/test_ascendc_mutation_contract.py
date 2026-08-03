# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_INIT_PATH = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc" / "__init__.py"


class FakeTensor:
    def __init__(self, *, requires_grad: bool = False):
        self.requires_grad = requires_grad


def fake_torch(incremented):
    module = types.ModuleType("torch")
    module.Tensor = FakeTensor
    module.autograd = types.SimpleNamespace(
        graph=types.SimpleNamespace(increment_version=lambda tensors: incremented.extend(tensors))
    )
    return module


def load_ascendc_module(raw_calls):
    fake_fla_npu = types.ModuleType("fla_npu")
    fake_fla_npu.__path__ = []
    fake_fla_npu.load_ascendc_opapi_libraries = lambda: None

    fake_ops = types.ModuleType("fla_npu.ops")
    fake_ops.__path__ = []

    ctypes_module = types.ModuleType("fla_npu.ops.ascendc._aclnn_ctypes")

    def npu_causal_conv1d(
        x,
        weight,
        bias=None,
        conv_states=None,
        *,
        query_start_loc=None,
        cache_indices=None,
        initial_state_mode=None,
        num_accepted_tokens=None,
        activation_mode=0,
        pad_slot_id=-1,
        run_mode=0,
        head_num=0,
    ):
        raw_calls.append(conv_states)
        return "output"

    def npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        *,
        beta,
        scale=1.0,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens=None,
        g=None,
        gk=None,
    ):
        raw_calls.append(state)
        return "output"

    ctypes_module.ASCENDC_CTYPES_OPS = {
        "npu_causal_conv1d": npu_causal_conv1d,
        "npu_recurrent_gated_delta_rule": npu_recurrent_gated_delta_rule,
    }
    modules = {
        "fla_npu": fake_fla_npu,
        "fla_npu.ops": fake_ops,
        "fla_npu.ops.ascendc._aclnn_ctypes": ctypes_module,
    }

    spec = importlib.util.spec_from_file_location(
        "fla_npu.ops.ascendc",
        ASCENDC_INIT_PATH,
        submodule_search_locations=[str(ASCENDC_INIT_PATH.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    modules["fla_npu.ops.ascendc"] = module
    return module, spec, modules


class AscendCMutationContractTest(unittest.TestCase):
    def test_mutable_raw_op_increments_state_version_after_launch(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor()
            result = module.npu_causal_conv1d(FakeTensor(), FakeTensor(), conv_states=state)

        self.assertEqual(result, "output")
        self.assertEqual(raw_calls, [state])
        self.assertEqual(incremented, [state])
        self.assertEqual(module.MUTATED_ARGUMENTS["npu_causal_conv1d"], ("conv_states",))

    def test_mutable_state_requiring_grad_is_rejected_before_launch(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor(requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, r"conv_states.*must not require gradients"):
                module.npu_causal_conv1d(FakeTensor(), FakeTensor(), conv_states=state)

        self.assertEqual(raw_calls, [])
        self.assertEqual(incremented, [])

    def test_recurrent_gdn_increments_state_version_after_launch(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertIs(
                module.recurrent_gated_delta_rule,
                module.npu_recurrent_gated_delta_rule,
            )
            state = FakeTensor()
            result = module.npu_recurrent_gated_delta_rule(
                FakeTensor(),
                FakeTensor(),
                FakeTensor(),
                state,
                beta=FakeTensor(),
                actual_seq_lengths=FakeTensor(),
                ssm_state_indices=FakeTensor(),
            )

        self.assertEqual(result, "output")
        self.assertEqual(raw_calls, [state])
        self.assertEqual(incremented, [state])
        self.assertEqual(
            module.MUTATED_ARGUMENTS["npu_recurrent_gated_delta_rule"],
            ("state",),
        )

    def test_recurrent_gdn_state_requiring_grad_is_rejected_before_launch(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor(requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, r"state.*must not require gradients"):
                module.npu_recurrent_gated_delta_rule(
                    FakeTensor(),
                    FakeTensor(),
                    FakeTensor(),
                    state,
                    beta=FakeTensor(),
                    actual_seq_lengths=FakeTensor(),
                    ssm_state_indices=FakeTensor(),
                )

        self.assertEqual(raw_calls, [])
        self.assertEqual(incremented, [])

    def test_direct_runtime_error_preserves_root_cause(self):
        raw_calls = []
        module, spec, modules = load_ascendc_module(raw_calls)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module._DIRECT_RUNTIME_READY = False

            def fail_to_load_opapi():
                raise RuntimeError("custom OPP contains libopapi.so")

            modules["fla_npu"].load_ascendc_opapi_libraries = fail_to_load_opapi
            with self.assertRaisesRegex(
                RuntimeError,
                r"Root cause: custom OPP contains libopapi\.so",
            ):
                module._prepare_direct_runtime()


if __name__ == "__main__":
    unittest.main()
