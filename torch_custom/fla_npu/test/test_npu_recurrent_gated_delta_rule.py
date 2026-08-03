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

import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops


torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def recurrent_gated_delta_rule_ref(
    query,
    key,
    value,
    state,
    beta,
    actual_seq_lengths,
    ssm_state_indices,
    *,
    scale=1.0,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    query_f = query.float() * float(scale)
    key_f = key.float()
    value_f = value.float()
    beta_f = beta.float()
    final_state = state.float().clone()
    out = torch.zeros_like(value_f)
    alpha = None if g is None else torch.exp(g.float())
    alpha_k = None if gk is None else torch.exp(gk.float())

    seq_start = int(actual_seq_lengths[0].item())
    for batch_idx, seq_len_tensor in enumerate(actual_seq_lengths[1:]):
        seq_len = int(seq_len_tensor.item())
        seq_end = seq_start + seq_len
        state_token_idx = seq_start
        if num_accepted_tokens is not None:
            state_token_idx += int(num_accepted_tokens[batch_idx].item()) - 1
        state_offset = int(ssm_state_indices[state_token_idx].item())

        for value_head in range(value.shape[1]):
            key_head = value_head // (value.shape[1] // key.shape[1])
            recurrent_state = final_state[state_offset, value_head].clone()
            for token_idx in range(seq_start, seq_end):
                if alpha is not None:
                    recurrent_state *= alpha[token_idx, value_head]
                if alpha_k is not None:
                    recurrent_state *= alpha_k[token_idx, value_head].unsqueeze(0)

                delta = value_f[token_idx, value_head] - torch.mv(
                    recurrent_state,
                    key_f[token_idx, key_head],
                )
                recurrent_state += torch.outer(
                    delta * beta_f[token_idx, value_head],
                    key_f[token_idx, key_head],
                )
                out[token_idx, value_head] = torch.mv(
                    recurrent_state,
                    query_f[token_idx, key_head],
                )
                final_state[
                    int(ssm_state_indices[token_idx].item()),
                    value_head,
                ] = recurrent_state
        seq_start = seq_end

    return out.to(value.dtype), final_state.to(state.dtype)


def make_inputs(*, use_g, use_gk, use_accepted_tokens):
    torch.manual_seed(42)
    actual_seq_lengths = torch.tensor([1, 2], dtype=torch.int32)
    total_tokens = int(actual_seq_lengths.sum().item())
    key_heads, value_heads, key_dim, value_dim = 2, 4, 128, 128

    query = torch.nn.functional.normalize(
        torch.rand(total_tokens, key_heads, key_dim, dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    key = torch.nn.functional.normalize(
        torch.rand(total_tokens, key_heads, key_dim, dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    value = torch.rand(total_tokens, value_heads, value_dim, dtype=torch.bfloat16)
    state = torch.rand(
        total_tokens,
        value_heads,
        value_dim,
        key_dim,
        dtype=torch.bfloat16,
    )
    beta = torch.rand(total_tokens, value_heads, dtype=torch.bfloat16)
    ssm_state_indices = torch.arange(total_tokens, dtype=torch.int32)
    g = -torch.rand(total_tokens, value_heads, dtype=torch.float32) if use_g else None
    gk = (
        -torch.rand(total_tokens, value_heads, key_dim, dtype=torch.float32)
        if use_gk
        else None
    )
    num_accepted_tokens = (
        torch.tensor([1], dtype=torch.int32) if use_accepted_tokens else None
    )
    return {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "g": g,
        "gk": gk,
        "scale": key_dim**-0.5,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestRecurrentGatedDeltaRule(unittest.TestCase):
    def test_output_and_inplace_state_match_cpu_reference(self):
        cases = (
            (True, False, False),
            (False, True, True),
            (True, True, False),
        )
        for use_g, use_gk, use_accepted_tokens in cases:
            with self.subTest(
                use_g=use_g,
                use_gk=use_gk,
                use_accepted_tokens=use_accepted_tokens,
            ):
                inputs = make_inputs(
                    use_g=use_g,
                    use_gk=use_gk,
                    use_accepted_tokens=use_accepted_tokens,
                )
                expected_out, expected_state = recurrent_gated_delta_rule_ref(**inputs)
                npu_inputs = {
                    name: value.npu() if isinstance(value, torch.Tensor) else value
                    for name, value in inputs.items()
                }
                state_version = npu_inputs["state"]._version

                actual_out = ascendc_ops.npu_recurrent_gated_delta_rule(**npu_inputs)
                torch.npu.synchronize()

                self.assertIsInstance(actual_out, torch.Tensor)
                self.assertEqual(npu_inputs["state"]._version, state_version + 1)
                torch.testing.assert_close(
                    actual_out.cpu().float(),
                    expected_out.float(),
                    rtol=2e-2,
                    atol=5e-2,
                )
                torch.testing.assert_close(
                    npu_inputs["state"].cpu().float(),
                    expected_state.float(),
                    rtol=2e-2,
                    atol=5e-2,
                )


if __name__ == "__main__":
    unittest.main()
