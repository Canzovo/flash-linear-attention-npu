#!/usr/bin/env python3
"""Validate fused GDN forward input errors without launching ACLNN."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import torch

from fla_npu.ops.ascendc import _aclnn_ctypes as target


def _canonical_indices(cu_seqlens, chunk_size=64):
    indices = []
    for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_chunk in range((end - begin + chunk_size - 1) // chunk_size):
            indices.extend((sequence, local_chunk))
    return tuple(indices)


def _inputs(
    *,
    layout="BNSD",
    batch=1,
    dtype=torch.bfloat16,
    gate_dtype=torch.float32,
    beta_dtype=None,
    value_dim=128,
):
    tokens, key_heads, value_heads, key_dim = 65, 2, 4, 128
    if layout in ("BSND", "TND"):
        q_shape = (batch, tokens, key_heads, key_dim)
        v_shape = (batch, tokens, value_heads, value_dim)
    else:
        q_shape = (batch, key_heads, tokens, key_dim)
        v_shape = (batch, value_heads, tokens, value_dim)
    return {
        "q": torch.empty(q_shape, dtype=dtype),
        "k": torch.empty(q_shape, dtype=dtype),
        "v": torch.empty(v_shape, dtype=dtype),
        "g": torch.empty((batch, tokens, value_heads), dtype=gate_dtype),
        "beta": torch.empty(
            (batch, tokens, value_heads), dtype=beta_dtype or dtype
        ),
    }


def _state(data, sequences, dtype, *, state_v_first=False):
    value_heads = data["g"].shape[-1]
    key_dim = data["q"].shape[-1]
    value_dim = data["v"].shape[-1]
    tail = (value_dim, key_dim) if state_v_first else (key_dim, value_dim)
    return torch.empty((sequences, value_heads, *tail), dtype=dtype)


def _invoke(data, **options):
    return target.npu_chunk_gated_delta_rule_fwd(
        data["q"], data["k"], data["v"], data["g"], data["beta"],
        **options,
    )


def _expect_rejected(name, message, invoke, launch_count):
    before = launch_count[0]
    try:
        invoke()
    except RuntimeError as error:
        if message not in str(error):
            raise AssertionError(
                f"{name}: expected error containing {message!r}, got {error!r}"
            ) from error
    else:
        raise AssertionError(f"{name}: invalid input was accepted")
    if launch_count[0] != before:
        raise AssertionError(f"{name}: invalid input reached ACLNN launch")


def main():
    invalid_cases = []

    data = _inputs(dtype=torch.float32)
    invalid_cases.append(("q_dtype", "same torch.float16 or torch.bfloat16", lambda data=data: _invoke(data)))

    data = _inputs()
    data["k"] = torch.empty(data["k"].shape, dtype=torch.float16)
    invalid_cases.append(("k_dtype", "same torch.float16 or torch.bfloat16", lambda data=data: _invoke(data)))

    data = _inputs()
    data["v"] = torch.empty(data["v"].shape, dtype=torch.float16)
    invalid_cases.append(("v_dtype", "same torch.float16 or torch.bfloat16", lambda data=data: _invoke(data)))

    data = _inputs(gate_dtype=torch.float64)
    invalid_cases.append(("g_dtype", "g and beta must use torch.float32", lambda data=data: _invoke(data)))

    data = _inputs(beta_dtype=torch.int32)
    invalid_cases.append(("beta_dtype", "g and beta must use torch.float32", lambda data=data: _invoke(data)))

    data = _inputs()
    state = _state(data, 1, torch.float64)
    invalid_cases.append((
        "initial_state_dtype",
        "initial_state must use torch.float32",
        lambda data=data, state=state: _invoke(data, initial_state=state),
    ))

    data = _inputs(batch=2)
    state = _state(data, 1, torch.float32)
    invalid_cases.append((
        "initial_state_dense_shape",
        "initial_state must have shape (2, 4, 128, 128)",
        lambda data=data, state=state: _invoke(data, initial_state=state),
    ))

    data = _inputs(value_dim=256)
    state = _state(data, 1, torch.float32, state_v_first=False)
    invalid_cases.append((
        "initial_state_state_v_first_shape",
        "when state_v_first=True",
        lambda data=data, state=state: _invoke(
            data, initial_state=state, state_v_first=True
        ),
    ))

    data = _inputs()
    cu = (0, 32, 65)
    state = _state(data, 1, torch.float32)
    invalid_cases.append((
        "initial_state_varlen_shape",
        "initial_state must have shape (2, 4, 128, 128)",
        lambda data=data, state=state, cu=cu: _invoke(
            data,
            initial_state=state,
            cu_seqlens=cu,
            chunk_indices=_canonical_indices(cu),
        ),
    ))

    for value, label, message in (
        (1, "scale_integer", "scale must be a floating-point number"),
        (True, "scale_bool", "scale must be a floating-point number"),
        (float("nan"), "scale_nan", "scale must be finite"),
        (float("inf"), "scale_pos_inf", "scale must be finite"),
        (float("-inf"), "scale_neg_inf", "scale must be finite"),
    ):
        data = _inputs()
        invalid_cases.append((
            label,
            message,
            lambda data=data, value=value: _invoke(data, scale=value),
        ))

    for layout in ("NTD", "TND"):
        data = _inputs(layout=layout, batch=2)
        invalid_cases.append((
            f"{layout.lower()}_batch",
            "NTD/TND input requires physical B=1",
            lambda data=data, layout=layout: _invoke(data, layout=layout),
        ))

    data = _inputs(batch=2)
    cu = (0, 65)
    invalid_cases.append((
        "varlen_batch",
        "varlen BNSD input requires physical B=1",
        lambda data=data, cu=cu: _invoke(
            data,
            cu_seqlens=cu,
            chunk_indices=_canonical_indices(cu),
        ),
    ))

    data = _inputs()
    data["q"] = torch.empty((1, 0, 65, 128), dtype=torch.bfloat16)
    data["k"] = torch.empty_like(data["q"])
    invalid_cases.append((
        "zero_key_heads",
        "B, Hk, Hv and T must be positive",
        lambda data=data: _invoke(data),
    ))

    launch_count = [0]

    def fake_empty(shape, like, **kwargs):
        return torch.empty(shape, dtype=kwargs.get("dtype", like.dtype))

    def fake_call(_name, _build_args, outputs):
        launch_count[0] += 1
        return outputs

    with mock.patch.object(target, "_empty", side_effect=fake_empty), mock.patch.object(
        target, "_call_aclnn", side_effect=fake_call
    ):
        for name, message, invoke in invalid_cases:
            _expect_rejected(name, message, invoke, launch_count)

        valid_cases = []
        data = _inputs()
        valid_cases.append((data, {"initial_state": _state(data, 1, torch.float32)}))

        data = _inputs(layout="BSND", dtype=torch.float16, gate_dtype=torch.float16,
                       beta_dtype=torch.float32, value_dim=256)
        valid_cases.append((
            data,
            {
                "layout": "BSND",
                "state_v_first": True,
                "initial_state": _state(data, 1, torch.float16, state_v_first=True),
                "scale": 0.125,
            },
        ))

        data = _inputs(layout="NTD")
        valid_cases.append((data, {"layout": "NTD"}))

        data = _inputs(layout="TND")
        cu = (0, 32, 65)
        valid_cases.append((
            data,
            {
                "layout": "TND",
                "cu_seqlens": cu,
                "chunk_indices": _canonical_indices(cu),
                "initial_state": _state(data, 2, torch.bfloat16),
            },
        ))

        for data, options in valid_cases:
            outputs = _invoke(data, **options)
            if len(outputs) != 10:
                raise AssertionError("valid input did not preserve the ten-output contract")

    expected_launches = len(valid_cases)
    if launch_count[0] != expected_launches:
        raise AssertionError(
            f"expected {expected_launches} valid launches, got {launch_count[0]}"
        )
    print(json.dumps({
        "schema": "chunk-gated-delta-rule-fwd-invalid-inputs/v1",
        "result": "passed",
        "module": str(Path(target.__file__).resolve()),
        "invalid_cases": len(invalid_cases),
        "valid_cases": len(valid_cases),
        "aclnn_launches": launch_count[0],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
