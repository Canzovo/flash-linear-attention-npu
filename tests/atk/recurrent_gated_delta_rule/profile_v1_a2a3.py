"""Single-case A2/A3 msprof workload for the legacy recurrent GDN ABI."""

import os

import torch

from fla_npu.ops import ascendc


def main() -> None:
    torch.manual_seed(20261030)
    device = torch.device("npu")
    total_tokens = int(os.environ.get("RGDR_PROFILE_T", "36"))
    key_heads, value_heads = 2, 8
    dim = 128
    query = torch.randn(total_tokens, key_heads, dim, dtype=torch.bfloat16, device=device)
    key = torch.randn_like(query)
    value = torch.randn(total_tokens, value_heads, dim, dtype=torch.bfloat16, device=device)
    beta = torch.rand(total_tokens, value_heads, dtype=torch.bfloat16, device=device)
    gate = -torch.rand(total_tokens, value_heads, dtype=torch.float32, device=device)
    state = torch.randn(36, value_heads, dim, dim, dtype=torch.bfloat16, device=device)
    actual_seq_lengths = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
    state_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)

    for _ in range(5):
        ascendc.recurrent_gated_delta_rule(
            query,
            key,
            value,
            state,
            beta=beta,
            scale=dim**-0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=state_indices,
            g=gate,
        )
    torch.npu.synchronize()
    ascendc.recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta=beta,
        scale=dim**-0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=state_indices,
        g=gate,
    )
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
