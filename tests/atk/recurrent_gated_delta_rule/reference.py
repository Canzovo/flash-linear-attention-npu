"""Readable CPU reference and dual CPU benchmarks for ``recurrent_gated_delta_rule``.

The implementation intentionally follows the mathematical recurrence one
token at a time. It is not optimized: clarity and independence from the NPU
kernel are more important than CPU speed.

The contract supports scalar gates, optional fused preprocessing and both
packed ``ssm_state_indices[T]`` and request-row ``[B,W]`` checkpoint tables.

``recurrent_gated_delta_rule_golden`` (and the historical non-mutating
``*_reference`` entry) evaluates the recurrence in FP64 before casting the
public outputs. ``recurrent_gated_delta_rule_bench`` evaluates it in FP32 and
is the CPU path used to approximate the product arithmetic. The public
``recurrent_gated_delta_rule`` entry updates the State pool in place and uses
the FP32 bench path. The CPU formula does not emulate UB banks, DMA or the
8-token input pipeline.
"""

from __future__ import annotations

import math
import numbers
from itertools import product

import torch


_METADATA_DTYPE = torch.int32
QK_L2_NORM_EPS = 1.0e-6


def l2_normalize_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    eps: float = QK_L2_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize Q/K along ``Dk`` with the GDN epsilon convention.

    Results preserve the argument dtype. The caller converts Q/K to its
    selected recurrence dtype before invoking this helper, so normalization
    uses that same dtype.
    """
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")

    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        calc_dtype = (
            torch.float64 if tensor.dtype == torch.float64 else torch.float32
        )
        tensor_float = tensor.to(calc_dtype)
        inverse_norm = torch.rsqrt(
            tensor_float.square().sum(dim=-1, keepdim=True) + float(eps)
        )
        return (tensor_float * inverse_norm).to(tensor.dtype)

    return _normalize(query), _normalize(key)


def _require_cpu(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None and tensor.device.type != "cpu":
        raise ValueError(
            f"recurrent_gated_delta_rule CPU golden requires {name} on CPU, "
            f"got device={tensor.device}."
        )


def _require_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} shape must be {expected}, got {actual}.")


def _require_dtype(name: str, tensor: torch.Tensor, allowed: tuple) -> None:
    if tensor.dtype not in allowed:
        raise TypeError(f"{name} dtype must be one of {allowed}, got {tensor.dtype}.")


def _storage_bounds(name: str, tensor: torch.Tensor) -> None:
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use strided (ND) layout.")
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} strides must be non-negative.")
    if tensor.numel() == 0:
        return
    last = tensor.storage_offset() + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    end = (last + 1) * tensor.element_size()
    if tensor.storage_offset() < 0 or end > min(tensor.untyped_storage().nbytes(), (1 << 63) - 1):
        raise ValueError(f"{name} view exceeds storage or 64-bit address bounds.")


def _byte_intervals(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """Exact occupied byte intervals, including holes and storage offsets.

    Used only for ambiguous State overlap or shared-storage alias checks.
    Dense trailing axes are coalesced to avoid enumerating every element.
    """
    if tensor.numel() == 0:
        return []
    axes = sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride()) if size > 1)
    span, count = 1, 0
    for stride, size in axes:
        if stride != span:
            break
        span *= size
        count += 1
    rest = axes[count:]
    itemsize = tensor.element_size()
    base = tensor.data_ptr()  # Already includes storage_offset.
    starts = sorted(
        base + sum(index * stride for index, (stride, _) in zip(indices, rest)) * itemsize
        for indices in product(*(range(size) for _, size in rest))
    )
    return [(start, start + span * itemsize) for start in starts]


def _validate_state_layout(state: torch.Tensor, others: dict) -> None:
    if any(stride <= 0 for stride in state.stride()):
        raise ValueError("State strides must be positive.")
    if state.stride(-1) != 1 or state.stride(-2) < 128:
        raise ValueError("State requires stride_K=1 and stride_V>=128.")
    if (state.stride(-2) - 128) * state.element_size() > (1 << 32) - 1:
        raise ValueError("State DMA row gap exceeds UINT32_MAX bytes.")
    # This quick proof covers ordinary padding, axis permutations and offsets.
    span, proven_disjoint = 1, True
    for stride, size in sorted((s, n) for s, n in zip(state.stride(), state.shape) if n > 1):
        if stride < span:
            proven_disjoint = False
            break
        span += (size - 1) * stride
    state_intervals = None
    if not proven_disjoint:
        state_intervals = _byte_intervals(state)
        if any(b > c for (_, b), (c, _) in zip(state_intervals, state_intervals[1:])):
            raise ValueError("State view must not overlap itself.")

    def extent(tensor):
        start = tensor.data_ptr()
        end = start + (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
        return start, end

    state_start, state_end = extent(state)
    for name, tensor in others.items():
        if tensor is None or tensor.numel() == 0:
            continue
        other_start, other_end = extent(tensor)
        if state_end <= other_start or other_end <= state_start:
            continue
        if state_intervals is None:
            state_intervals = _byte_intervals(state)
        a, b, i, j = state_intervals, _byte_intervals(tensor), 0, 0
        while i < len(a) and j < len(b):
            if max(a[i][0], b[j][0]) < min(a[i][1], b[j][1]):
                raise ValueError(f"State must not overlap input {name}.")
            if a[i][1] <= b[j][1]:
                i += 1
            else:
                j += 1


def _validate_state_ownership(lengths, state_indices, accepted, table_mode):
    writers, readers = {}, []
    start = lengths[0]
    for batch, length in enumerate(lengths[1:]):
        if length:
            row = state_indices[batch] if table_mode else state_indices[start:start + length]
            initial_position = 0 if accepted is None else accepted[batch] - 1
            readers.append((row[initial_position], batch))
            for slot in row[:length]:
                if writers.setdefault(slot, batch) != batch:
                    raise ValueError("State writes overlap across requests.")
        start += length
    for slot, batch in readers:
        if slot in writers and writers[slot] != batch:
            raise ValueError("Initial State is written by another request.")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    use_gate_in_kernel: bool,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
) -> tuple[list[int], list[int] | list[list[int]], list[int] | None]:
    """Validate the current design contract before any numerical/state update."""
    flags = (use_qk_l2norm_in_kernel, use_gate_in_kernel,
             use_beta_sigmoid_in_kernel, allow_neg_eigval)
    if any(not isinstance(flag, bool) for flag in flags):
        raise TypeError("All operator switches must be bool.")
    if not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(float(scale)):
        raise ValueError("scale must be a finite real number (not bool).")
    if gk is not None:
        raise ValueError("gk is reserved; only scalar gate g is supported.")
    named_tensors = {
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
        "A_log": A_log,
        "dt_bias": dt_bias,
    }
    optional = {"num_accepted_tokens", "g", "gk", "A_log", "dt_bias"}
    for name, tensor in named_tensors.items():
        if tensor is None and name in optional:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        _require_cpu(name, tensor)
        _storage_bounds(name, tensor)
    for name in ("query", "key", "value"):
        _require_dtype(name, named_tensors[name], (torch.bfloat16,))
    for name in ("state", "beta", "g"):
        if named_tensors[name] is not None:
            _require_dtype(name, named_tensors[name], (torch.bfloat16, torch.float32))
    for name in ("A_log", "dt_bias"):
        if named_tensors[name] is not None:
            _require_dtype(name, named_tensors[name], (torch.float32,))

    if query.ndim != 3:
        raise ValueError(f"query must be rank 3, got shape={tuple(query.shape)}.")
    total_tokens, key_heads, key_dim = query.shape
    _require_shape("key", key, (total_tokens, key_heads, key_dim))

    if value.ndim != 3:
        raise ValueError(f"value must be rank 3, got shape={tuple(value.shape)}.")
    value_tokens, value_heads, value_dim = value.shape
    if value_tokens != total_tokens:
        raise ValueError(
            f"query/value token counts must match, got {total_tokens} and "
            f"{value_tokens}."
        )
    if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads != 0:
        raise ValueError(
            "value head count must be a positive integer multiple of key head "
            f"count, got Nk={key_heads}, Nv={value_heads}."
        )

    if total_tokens <= 0 or key_dim != 128 or value_dim != 128:
        raise ValueError("T must be positive and K=V=128.")
    if value_heads // key_heads > 16:
        raise ValueError("HV/HK must be an integer in [1,16].")
    for name in ("query", "key", "value"):
        tensor = named_tensors[name]
        if any(s <= 0 for s in tensor.stride()) or tensor.stride(-1) != 1 or tensor.stride(0) < 128:
            raise ValueError(f"{name} requires stride_D=1 and token stride>=128.")
        if (tensor.stride(0) - 128) * tensor.element_size() > (1 << 32) - 1:
            raise ValueError(f"{name} DMA row gap exceeds UINT32_MAX bytes.")

    if state.ndim != 4:
        raise ValueError(f"state must be rank 4, got shape={tuple(state.shape)}.")
    state_blocks = state.shape[0]
    _require_shape(
        "state",
        state,
        (state_blocks, value_heads, value_dim, key_dim),
    )
    if state_blocks <= 0:
        raise ValueError("state must contain at least one State pool slot.")
    _require_shape("beta", beta, (total_tokens, value_heads))
    if g is not None:
        _require_shape("g", g, (total_tokens, value_heads))
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        if tensor is not None:
            _require_shape(name, tensor, (value_heads,))
    _validate_state_layout(state, {k: v for k, v in named_tensors.items() if k != "state"})
    if use_gate_in_kernel:
        if g is None:
            raise ValueError(
                "g is required when use_gate_in_kernel=True because g is "
                "the raw scalar gate input."
            )
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True.")

    if actual_seq_lengths.ndim != 1 or actual_seq_lengths.numel() < 2:
        raise ValueError(
            "actual_seq_lengths must be [prefix, length_0, ..., length_B-1]."
        )
    if actual_seq_lengths.dtype != _METADATA_DTYPE:
        raise TypeError(
            "actual_seq_lengths must use INT32, got "
            f"{actual_seq_lengths.dtype}."
        )
    lengths = [int(item) for item in actual_seq_lengths.tolist()]
    prefix_tokens = lengths[0]
    seq_lengths = lengths[1:]
    if prefix_tokens < 0 or any(length < 0 for length in seq_lengths):
        raise ValueError(f"sequence lengths must be non-negative, got {lengths}.")
    if sum(lengths) != total_tokens:
        raise ValueError(
            "actual_seq_lengths must sum to T, got "
            f"sum={sum(lengths)} and T={total_tokens}."
        )

    if ssm_state_indices.dtype != _METADATA_DTYPE:
        raise TypeError(
            "ssm_state_indices must use INT32, got "
            f"{ssm_state_indices.dtype}."
        )
    batch_size = len(seq_lengths)
    if ssm_state_indices.ndim == 1:
        _require_shape("ssm_state_indices", ssm_state_indices, (total_tokens,))
        state_widths = seq_lengths
    elif ssm_state_indices.ndim == 2:
        if ssm_state_indices.shape[0] != batch_size:
            raise ValueError(
                "2-D ssm_state_indices row count must equal B, got "
                f"shape={tuple(ssm_state_indices.shape)} and B={batch_size}."
            )
        state_width = int(ssm_state_indices.shape[1])
        if state_width <= 0:
            raise ValueError("2-D ssm_state_indices must have a positive W.")
        if any(length > state_width for length in seq_lengths):
            raise ValueError(
                f"every sequence length must be <= W={state_width}, got "
                f"{seq_lengths}."
            )
        state_widths = [state_width] * batch_size
    else:
        raise ValueError(
            "ssm_state_indices must be rank 1 [T] or rank 2 [B, W], got "
            f"shape={tuple(ssm_state_indices.shape)}."
        )

    flat_indices = ssm_state_indices.reshape(-1).to(torch.int64)
    if torch.any(flat_indices < 0) or torch.any(flat_indices >= state_blocks):
        raise ValueError(
            "ssm_state_indices values must be in [0, BlockNum), got "
            f"BlockNum={state_blocks}."
        )

    accepted: list[int] | None = None
    if num_accepted_tokens is not None:
        if num_accepted_tokens.dtype != _METADATA_DTYPE:
            raise TypeError(
                "num_accepted_tokens must use INT32, got "
                f"{num_accepted_tokens.dtype}."
            )
        _require_shape(
            "num_accepted_tokens",
            num_accepted_tokens,
            (batch_size,),
        )
        accepted = [int(item) for item in num_accepted_tokens.tolist()]
        for batch_index, seq_len in enumerate(seq_lengths):
            # Empty rows are skipped before the kernel reads accepted[b].
            if seq_len == 0:
                continue
            if not 1 <= accepted[batch_index] <= state_widths[batch_index]:
                raise ValueError(
                    f"num_accepted_tokens[{batch_index}] must be in "
                    f"[1, {state_widths[batch_index]}], got "
                    f"{accepted[batch_index]}."
                )

    state_indices = ssm_state_indices.tolist()
    _validate_state_ownership(lengths, state_indices, accepted, ssm_state_indices.ndim == 2)
    return lengths, state_indices, accepted


def _state_slot(
    state_indices: list[int] | list[list[int]],
    *,
    table_mode: bool,
    batch_index: int,
    sequence_start: int,
    relative_token: int,
) -> int:
    """Map a request-local checkpoint/token position to a State pool slot."""
    if table_mode:
        return int(state_indices[batch_index][relative_token])
    return int(state_indices[sequence_start + relative_token])


@torch.no_grad()
def recurrent_gated_delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    *,
    calc_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the RecurrentGDN recurrence on CPU.

    For every value head and token, the implementation computes::

        q_eff, k_eff = q, k
        if use_qk_l2norm_in_kernel:
            q_eff = q / sqrt(sum(q * q) + 1e-6)
            k_eff = k / sqrt(sum(k * k) + 1e-6)

        g_eff = g
        if use_gate_in_kernel:
            g_eff = -exp(A_log) * softplus(g + dt_bias)

        beta_eff = sigmoid(beta) if use_beta_sigmoid_in_kernel else beta
        if use_beta_sigmoid_in_kernel and allow_neg_eigval:
            beta_eff *= 2
        decayed_state = exp(g_eff) * state  # no g means decay = 1
        delta = beta_eff * (value - decayed_state @ k_eff)
        state = decayed_state + outer(delta, k_eff)
        out = state @ (scale * q_eff)

    ``state`` is treated as a State pool. The returned ``final_state`` is a
    clone containing all per-token writebacks; the input ``state`` is not
    mutated. ``calc_dtype`` selects the internal recurrence precision and is
    restricted to FP32 or FP64. Each written checkpoint is cast to the public
    State dtype, while the next token keeps the unrounded recurrence state.
    Untouched State slots are cloned in their original dtype. Prefix output is
    filled with zero for comparison convenience, not as a product guarantee.
    """
    lengths, state_indices, accepted_tokens = _validate_inputs(
        query,
        key,
        value,
        state,
        beta,
        scale,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens,
        g,
        gk,
        A_log,
        dt_bias,
        use_gate_in_kernel,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
    )
    if calc_dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "calc_dtype must be torch.float32 or torch.float64, "
            f"got {calc_dtype}."
        )
    # Preserve original State dtype/untouched slots and never read a rounded
    # checkpoint as the next token's recurrent state.
    final_state = state.clone()
    total_tokens, key_heads, _ = query.shape
    value_heads, value_dim = value.shape[1:]
    value_heads_per_key_head = value_heads // key_heads
    out = torch.zeros(
        (total_tokens, value_heads, value_dim),
        dtype=calc_dtype,
        device="cpu",
    )

    table_mode = ssm_state_indices.ndim == 2
    sequence_start = lengths[0]
    for batch_index, sequence_length in enumerate(lengths[1:]):
        sequence_end = sequence_start + sequence_length
        if sequence_length == 0:
            sequence_start = sequence_end
            continue

        accepted_count = (
            1 if accepted_tokens is None else accepted_tokens[batch_index]
        )
        initial_state_slot = _state_slot(
            state_indices,
            table_mode=table_mode,
            batch_index=batch_index,
            sequence_start=sequence_start,
            relative_token=accepted_count - 1,
        )

        for value_head in range(value_heads):
            key_head = value_head // value_heads_per_key_head
            recurrent_state = state[initial_state_slot, value_head].to(calc_dtype).clone()

            for token_index in range(sequence_start, sequence_end):
                relative_token = token_index - sequence_start

                query_vector = query[token_index, key_head].to(calc_dtype)
                key_vector = key[token_index, key_head].to(calc_dtype)
                if use_qk_l2norm_in_kernel:
                    query_vector, key_vector = l2_normalize_qk(query_vector, key_vector)
                scaled_query = query_vector * float(scale)

                beta_eff = beta[token_index, value_head].to(calc_dtype)
                if use_beta_sigmoid_in_kernel:
                    beta_eff = torch.sigmoid(beta_eff)
                    if allow_neg_eigval:
                        beta_eff = beta_eff * 2.0

                # Process only valid tokens. Prefix/empty requests do not
                # participate in norm, gate or State arithmetic.
                alpha = torch.tensor(1.0, dtype=calc_dtype)
                if g is not None:
                    gate = g[token_index, value_head].to(calc_dtype)
                    if use_gate_in_kernel:
                        z = gate
                        if dt_bias is not None:
                            z = z + dt_bias[value_head]
                        # Stable mathematical softplus without the default
                        # F.softplus linear threshold; no kernel polynomial.
                        softplus = torch.clamp_min(z, 0.0) + torch.log1p(torch.exp(-torch.abs(z)))
                        gate = -torch.exp(A_log[value_head]) * softplus
                    alpha = torch.exp(gate)
                decayed_state = alpha * recurrent_state
                prediction = torch.matmul(decayed_state, key_vector)
                delta = beta_eff * (value[token_index, value_head].to(calc_dtype) - prediction)
                recurrent_state = decayed_state + torch.outer(delta, key_vector)
                out[token_index, value_head] = torch.matmul(recurrent_state, scaled_query)

                writeback_slot = _state_slot(
                    state_indices,
                    table_mode=table_mode,
                    batch_index=batch_index,
                    sequence_start=sequence_start,
                    relative_token=relative_token,
                )
                final_state[writeback_slot, value_head] = recurrent_state.to(state.dtype)

        sequence_start = sequence_end

    return out.to(value.dtype), final_state


@torch.no_grad()
def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    *,
    beta: torch.Tensor,
    scale: float = 1.0,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
) -> torch.Tensor:
    """CPU counterpart of the public API: update State in place, return out.

    All validation and numerical work precede State commit. Copy only written
    slots so untouched slots and padding remain untouched in the input view.
    """
    out, final_state = recurrent_gated_delta_rule_bench(
        query, key, value, state, beta, scale, actual_seq_lengths,
        ssm_state_indices, num_accepted_tokens, g, gk,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, A_log, dt_bias,
        use_beta_sigmoid_in_kernel, allow_neg_eigval,
    )
    lengths = actual_seq_lengths.tolist()
    start = lengths[0]
    table_mode = ssm_state_indices.ndim == 2
    slots = set()
    for batch, length in enumerate(lengths[1:]):
        if length:
            row = ssm_state_indices[batch, :length] if table_mode else ssm_state_indices[start:start + length]
            slots.update(row.tolist())
        start += length
    for slot in sorted(slots):
        state[slot].copy_(final_state[slot])
    return out


def recurrent_gated_delta_rule_golden(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 CPU golden; returns non-mutating ``(out, final_state)``."""
    kwargs["calc_dtype"] = torch.float64
    return recurrent_gated_delta_rule_reference(*args, **kwargs)


def recurrent_gated_delta_rule_bench(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 CPU benchmark; returns non-mutating ``(out, final_state)``."""
    kwargs["calc_dtype"] = torch.float32
    return recurrent_gated_delta_rule_reference(*args, **kwargs)


# Historical aliases remain available, with names now making the precision
# contract explicit. The old reference entry itself defaults to FP64 golden.
recurrent_gated_delta_rule_cpu_golden = recurrent_gated_delta_rule_golden
recurrent_gated_delta_rule_cpu_bench = recurrent_gated_delta_rule_bench
