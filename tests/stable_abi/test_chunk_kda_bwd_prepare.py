#!/usr/bin/env python3
"""Verify the public Stable-ABI entry against an exact, independent CPU reference.

Run after installing the current wheel:
    python tests/stable_abi/test_chunk_kda_bwd_prepare.py --device 0

Integer BF16 inputs and a dyadic scale make all FP32 intermediates exact. This
checks indexing, tails and state orientation; ATK remains the accuracy gate for
the original random input ranges and native accuracy criteria.
"""
import argparse
import itertools

import torch
import torch_npu  # noqa: F401

from fla_npu.ops import ascendc


def make_case(layout, tokens, state_v_first=False, *, batch=1, heads=3):
    packed = layout in {"NTD", "TND"}
    generator = torch.Generator().manual_seed(1000 + tokens + batch + heads)

    def values(shape):
        return torch.randint(-2, 3, shape, generator=generator).to(torch.bfloat16)

    prefix = (heads, tokens) if packed else (batch, heads, tokens)
    aqk, v_new, d_o = (values(prefix + (width,)) for width in (64, 128, 128))
    if packed:
        # An empty middle sequence, a one-token sequence and a possible tail.
        cu = [0, 1, 1, tokens] if tokens > 1 else [0, 0, 1]
        chunks = [(seq, chunk) for seq in range(len(cu) - 1)
                  for chunk in range((cu[seq + 1] - cu[seq] + 63) // 64)]
        h = values((len(chunks), heads, 128, 128))
    else:
        cu = None
        chunks = [(b, chunk) for b in range(batch)
                  for chunk in range((tokens + 63) // 64)]
        h = values((batch, (tokens + 63) // 64, heads, 128, 128))

    expected = [torch.zeros(prefix + (width,), dtype=dtype)
                for width, dtype in ((64, torch.float32), (128, torch.bfloat16),
                                     (128, torch.float32))]
    for index, (seq, chunk) in enumerate(chunks):
        begin = (cu[seq] if packed else 0) + chunk * 64
        end = min(begin + 64, cu[seq + 1] if packed else tokens)
        for head in range(heads):
            pos = (head, slice(begin, end)) if packed else (seq, head, slice(begin, end))
            do = d_o[pos].double()
            vn = v_new[pos].double()
            a = aqk[pos][..., :end - begin].double()
            state = (h[index, head] if packed else h[seq, chunk, head]).double()
            expected[0][pos][..., :end - begin] = (do @ vn.T).tril().float() * 0.125
            expected[1][pos] = (a.T @ do).bfloat16()
            expected[2][pos] = (do @ (state if state_v_first else state.T)).float()

    if layout == "BSND":
        aqk, v_new, d_o = (x.transpose(1, 2).contiguous() for x in (aqk, v_new, d_o))
    elif layout == "TND":
        aqk, v_new, d_o = (x.transpose(0, 1).contiguous() for x in (aqk, v_new, d_o))
    kwargs = dict(layout=layout, scale=0.125, chunk_size=64,
                  state_v_first=state_v_first, cu_seqlens=cu,
                  chunk_indices=[item for pair in chunks for item in pair] if packed else None)
    return (aqk, v_new, d_o, h), kwargs, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    count = 0
    matrix = list(itertools.product(("BNSD", "BSND", "NTD", "TND"),
                                    (1, 64, 65), (False, True)))
    matrix += [(layout, 129, flag) for layout in ("BNSD", "BSND", "NTD", "TND")
               for flag in (False, True)]
    for layout, tokens, flag in matrix:
        inputs, kwargs, expected = make_case(layout, tokens, flag,
            batch=2 if tokens == 129 else 1, heads=5 if tokens == 129 else 3)
        actual = ascendc.chunk_kda_bwd_prepare(*(x.npu() for x in inputs), **kwargs)
        for got, want in zip(actual, expected):
            assert got.is_contiguous()
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
        assert ascendc.BACKENDS.get("npu_chunk_kda_bwd_prepare") == "stable", ascendc.BACKENDS
        assert not ascendc.FALLBACKS.get("npu_chunk_kda_bwd_prepare", 0), ascendc.FALLBACKS
        count += 1
        print(f"PASS {layout} T={tokens} state_v_first={flag}", flush=True)
    print(f"PASS {count} cases, {3 * count} exact output comparisons; backend=stable, fallback=0")


if __name__ == "__main__":
    main()
