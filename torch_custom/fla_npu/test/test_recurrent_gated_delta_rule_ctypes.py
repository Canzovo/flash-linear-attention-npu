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

import ctypes
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


CTYPES_PATH = (
    Path(__file__).resolve().parents[1]
    / "fla_npu"
    / "ops"
    / "ascendc"
    / "_aclnn_ctypes.py"
)


class FakeTensor:
    def __init__(self, name):
        self.name = name


class FakeContext:
    def tensor(self, value, name):
        return (name, value)


def load_ctypes_module():
    package = types.ModuleType("fla_npu.ops.ascendc")
    package.__path__ = []
    policy = types.ModuleType("fla_npu.ops.ascendc._kda_policy")
    policy.kda_fwd_optional_output_mask = lambda **kwargs: ()
    runtime = types.ModuleType("fla_npu.ops.ascendc._runtime")
    runtime.call_aclnn = lambda *args, **kwargs: None
    runtime.chunk_num = lambda *args, **kwargs: 0
    runtime.empty = lambda *args, **kwargs: FakeTensor("empty")
    runtime.empty_like = lambda tensor, **kwargs: FakeTensor(
        f"empty_like:{tensor.name}"
    )
    runtime.optional_bool = lambda value, default: default if value is None else bool(value)
    runtime.optional_float = (
        lambda value, default: default if value is None else float(value)
    )
    runtime.optional_int = lambda value, default: default if value is None else int(value)
    runtime.shape = lambda tensor: ()
    runtime.zeros = lambda *args, **kwargs: FakeTensor("zeros")

    modules = {
        "fla_npu.ops.ascendc": package,
        "fla_npu.ops.ascendc._kda_policy": policy,
        "fla_npu.ops.ascendc._runtime": runtime,
    }
    spec = importlib.util.spec_from_file_location(
        "fla_npu.ops.ascendc._aclnn_ctypes",
        CTYPES_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    modules["fla_npu.ops.ascendc._aclnn_ctypes"] = module
    with mock.patch.dict(sys.modules, modules):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class RecurrentGatedDeltaRuleCtypesTest(unittest.TestCase):
    def test_wrapper_matches_aclnn_header_order_and_allocates_value_shaped_output(self):
        module = load_ctypes_module()
        tensors = {
            name: FakeTensor(name)
            for name in (
                "query",
                "key",
                "value",
                "state",
                "beta",
                "actual_seq_lengths",
                "ssm_state_indices",
                "g",
                "gk",
                "num_accepted_tokens",
            )
        }
        captured = {}

        def fake_call(name, build_args, output):
            captured["name"] = name
            captured["args"] = build_args(FakeContext())
            captured["output"] = output
            return output

        module._call_aclnn = fake_call
        result = module.npu_recurrent_gated_delta_rule(
            tensors["query"],
            tensors["key"],
            tensors["value"],
            tensors["state"],
            beta=tensors["beta"],
            scale=0.125,
            actual_seq_lengths=tensors["actual_seq_lengths"],
            ssm_state_indices=tensors["ssm_state_indices"],
            num_accepted_tokens=tensors["num_accepted_tokens"],
            g=tensors["g"],
            gk=tensors["gk"],
        )

        self.assertEqual(captured["name"], "aclnnRecurrentGatedDeltaRule")
        self.assertIs(result, captured["output"])
        self.assertEqual(result.name, "empty_like:value")
        self.assertEqual(
            [arg[0] for arg in captured["args"][:10]],
            [
                "query",
                "key",
                "value",
                "beta",
                "state",
                "actual_seq_lengths",
                "ssm_state_indices",
                "g",
                "gk",
                "num_accepted_tokens",
            ],
        )
        self.assertIsInstance(captured["args"][10], ctypes.c_float)
        self.assertAlmostEqual(captured["args"][10].value, 0.125)
        self.assertIs(captured["args"][8][1], tensors["gk"])
        self.assertEqual(captured["args"][11], ("out", result))


if __name__ == "__main__":
    unittest.main()
