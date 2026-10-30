"""CPU contract checks; these do not claim NPU precision/performance acceptance."""
import importlib.util
import itertools
import os
from pathlib import Path
import unittest

import torch

from reference import recurrent_gated_delta_rule as inplace
from reference import recurrent_gated_delta_rule_bench as bench
from reference import recurrent_gated_delta_rule_golden as golden


torch.set_num_threads(1)


def inputs(lengths=(9,), prefix=0, hk=1, hv=2, state_dtype=torch.float32,
           beta_dtype=torch.bfloat16, gate_dtype=torch.float32):
    """Traceable one-hot inputs; no recurrent growth/range assumptions."""
    total = prefix + sum(lengths)
    pool = max(4, sum(lengths) + 3)
    q = torch.zeros(total, hk, 128, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros(total, hv, 128, dtype=torch.bfloat16)
    for t in range(total):
        q[t, :, 0] = 1
        k[t, :, 0] = 0.5
        for h in range(hv):
            v[t, h, h % 128] = 1 + (t % 3) * 0.25
    state = torch.zeros(pool, hv, 128, 128, dtype=state_dtype)
    for s in range(pool):
        for h in range(hv):
            state[s, h, h % 128, 0] = (s + 1) * 0.125
    indices = torch.zeros(total, dtype=torch.int32)
    indices[prefix:] = torch.arange(sum(lengths), dtype=torch.int32)
    return dict(query=q, key=k, value=v, state=state,
                beta=torch.full((total, hv), 0.5, dtype=beta_dtype), scale=1.0,
                actual_seq_lengths=torch.tensor((prefix, *lengths), dtype=torch.int32),
                ssm_state_indices=indices,
                g=torch.full((total, hv), -0.25, dtype=gate_dtype))


def old_reference():
    # Optional path lets a reviewer reproduce comparison with the exact upload.
    path = os.environ.get('RGDN_ORIGINAL_REFERENCE')
    if path is None:
        return None
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader('original_upload', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.recurrent_gated_delta_rule_reference


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = staticmethod(old_reference())

    def assert_dual_close(self, golden_result, bench_result):
        """Check the FP64 golden against the FP32 CPU benchmark after casts."""
        golden_out, golden_state = golden_result
        bench_out, bench_state = bench_result
        torch.testing.assert_close(
            golden_out, bench_out, rtol=2e-2, atol=2e-2
        )
        state_tol = (2e-2, 2e-2) if bench_state.dtype == torch.bfloat16 else (5e-4, 5e-4)
        torch.testing.assert_close(
            golden_state, bench_state, rtol=state_tol[0], atol=state_tol[1]
        )

    def test_switches_and_dtypes(self):
        for state_dt, beta_dt, gate_dt, norm, sig, gate_mode, neg in itertools.product(
                (torch.bfloat16, torch.float32), (torch.bfloat16, torch.float32),
                (torch.bfloat16, torch.float32), (False, True), (False, True),
                ('none', 'log', 'raw'), (False, True)):
            with self.subTest(state=state_dt, beta=beta_dt, g=gate_dt,
                              norm=norm, sig=sig, gate=gate_mode, neg=neg):
                x = inputs(state_dtype=state_dt, beta_dtype=beta_dt, gate_dtype=gate_dt)
                x.update(use_qk_l2norm_in_kernel=norm,
                         use_beta_sigmoid_in_kernel=sig, allow_neg_eigval=neg,
                         use_gate_in_kernel=gate_mode == 'raw')
                if gate_mode == 'none':
                    x['g'] = None
                if gate_mode == 'raw':
                    x['A_log'] = torch.tensor([-0.5, 0.25])
                    x['dt_bias'] = torch.tensor([0.5, -0.25])
                initial = x['state'].clone()
                out, state = golden(**x)
                bench_out, bench_state = bench(**x)
                self.assert_dual_close((out, state), (bench_out, bench_state))
                torch.testing.assert_close(x['state'], initial, rtol=0, atol=0)
                actual = inplace(**x)
                torch.testing.assert_close(actual, bench_out, rtol=0, atol=0)
                torch.testing.assert_close(x['state'], bench_state, rtol=0, atol=0)
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertEqual(state.dtype, state_dt)
                if self.old is not None:
                    x['state'] = initial
                    old_out, old_state = self.old(**x)
                    torch.testing.assert_close(bench_out, old_out, rtol=0, atol=0)
                    torch.testing.assert_close(bench_state, old_state, rtol=0, atol=0)

    def test_analytical_scalar_recurrence_and_repeated_slot(self):
        for length in (1, 7, 8, 9, 16, 17, 32, 33, 65):
            with self.subTest(length=length):
                x = inputs(lengths=(length,), hv=1)
                x['g'] = None
                x['state'].zero_()
                x['state'][0, 0, 0, 0] = 0.25
                x['ssm_state_indices'].zero_()
                x['query'][:, :, 0] = 2
                x['key'][:, :, 0] = 1
                x['value'].zero_()
                x['value'][:, :, 0] = 1
                x['scale'] = 0.5
                out, state = golden(**x)
                # s_new=s+0.5*(1-s), q*scale=1. Exactly representable scalar oracle.
                s = torch.tensor(0.25)
                expected = torch.zeros_like(out)
                for t in range(length):
                    s = s + 0.5 * (1 - s)
                    expected[t, 0, 0] = s.to(torch.bfloat16)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                self.assertEqual(state[0, 0, 0, 0], s)
                torch.testing.assert_close(state[1:], x['state'][1:], rtol=0, atol=0)

    def test_bf16_checkpoint_does_not_round_recurrence(self):
        x = inputs(lengths=(9,), hv=1, state_dtype=torch.bfloat16, beta_dtype=torch.float32)
        x['g'] = None
        x['ssm_state_indices'].zero_()
        x['query'].zero_()
        x['query'][:, :, 0] = 1
        x['key'].zero_()  # All state entries persist exactly before decay.
        x['g'] = torch.full((9, 1), -0.01)
        x['state'].zero_()
        x['state'][0, 0, 0, 0] = 1
        out, state = golden(**x)
        exact = torch.tensor(1.0)
        rounded = torch.tensor(1.0)
        alpha = torch.exp(torch.tensor(-0.01))
        for t in range(9):
            exact = exact * alpha
            rounded = (rounded * alpha).to(torch.bfloat16).float()
            self.assertEqual(out[t, 0, 0], exact.to(torch.bfloat16))
        self.assertNotEqual(exact.to(torch.bfloat16), rounded.to(torch.bfloat16))
        self.assertEqual(state[0, 0, 0, 0], exact.to(torch.bfloat16))

    def test_checkpoint_table_accepted_beyond_sequence_length(self):
        x = inputs(lengths=(2, 0, 3), prefix=1)
        x['ssm_state_indices'] = torch.tensor([[0, 1, 6, 7], [7, 7, 7, 7], [2, 3, 4, 5]], dtype=torch.int32)
        x['num_accepted_tokens'] = torch.tensor([4, -999, 4], dtype=torch.int32)
        out, state = golden(**x)
        # Independent requests must equal isolated calls with selected checkpoints.
        start = 1
        for b, length in enumerate((2, 0, 3)):
            if not length:
                continue
            single = dict(x)
            for name in ('query', 'key', 'value', 'beta', 'g'):
                single[name] = x[name][start:start + length]
            single['actual_seq_lengths'] = torch.tensor([0, length], dtype=torch.int32)
            single['ssm_state_indices'] = x['ssm_state_indices'][b:b + 1]
            single['num_accepted_tokens'] = torch.tensor([4], dtype=torch.int32)
            expected_out, expected_state = golden(**single)
            torch.testing.assert_close(out[start:start + length], expected_out, rtol=0, atol=0)
            for slot in x['ssm_state_indices'][b, :length]:
                torch.testing.assert_close(state[slot], expected_state[slot], rtol=0, atol=0)
            start += length

    def test_mixed_qkv_views_and_head_first(self):
        for head_first in (False, True):
            with self.subTest(head_first=head_first):
                x = inputs(lengths=(9,), hk=2, hv=6)
                total = 9
                shape = (10, total, 128) if head_first else (total, 10, 128)
                mixed = torch.empty(shape, dtype=torch.bfloat16)
                for name, start, end in (('query', 0, 2), ('key', 2, 4), ('value', 4, 10)):
                    view = mixed[start:end].transpose(0, 1) if head_first else mixed[:, start:end]
                    view.copy_(x[name])
                    x[name] = view
                    self.assertEqual(view.stride(-1), 1)
                    self.assertFalse(view.is_contiguous())
                expected = golden(**{k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in x.items()})
                actual = golden(**x)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_state_padding_offset_and_axis_permutation(self):
        x = inputs(lengths=(9,))
        golden_result = golden(**x)
        expected_out, expected_state = bench(**x)
        self.assert_dual_close(golden_result, (expected_out, expected_state))
        slots, heads = x['state'].shape[:2]
        base = torch.full((heads + 1, slots + 1, 128, 133), -777.0)
        view = base[:heads, 1:, :, 2:130].transpose(0, 1)
        view.copy_(x['state'])
        x['state'] = view
        before = base.clone()
        returned = inplace(**x)
        torch.testing.assert_close(returned, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(view, expected_state, rtol=0, atol=0)
        mask = torch.ones_like(base, dtype=torch.bool)
        mask[:heads, 1:, :, 2:130] = False
        torch.testing.assert_close(base[mask], before[mask], rtol=0, atol=0)

    def test_empty_requests_prefix_and_no_head_cap(self):
        for heads in (1, 257):
            x = inputs(lengths=(0, 0), prefix=2, hk=heads, hv=heads)
            x['num_accepted_tokens'] = torch.tensor([-9, 0], dtype=torch.int32)
            x['query'].fill_(float('nan'))
            x['g'].fill_(float('nan'))
            original = x['state'].clone()
            out = inplace(**x)
            torch.testing.assert_close(x['state'], original, rtol=0, atol=0)
            self.assertEqual(out.shape, (2, heads, 128))

    def test_ratio_sixteen_and_read_only_qkv_alias(self):
        x = inputs(lengths=(1,), hk=1, hv=16)
        x['key'] = x['query']
        out, state = golden(**x)
        self.assertEqual(out.shape, (1, 16, 128))
        self.assertEqual(state.dtype, torch.float32)

    def test_dense_random_regression_and_head_feature_mapping(self):
        if self.old is not None:
            generator = torch.Generator().manual_seed(20261030)
            for dtype in (torch.bfloat16, torch.float32):
                x = inputs(lengths=(9, 2), prefix=1, hk=2, hv=4, state_dtype=dtype)
                for name in ('query', 'key', 'value', 'state', 'beta', 'g'):
                    x[name].copy_(torch.randn(x[name].shape, generator=generator))
                for norm in (False, True):
                    x['use_qk_l2norm_in_kernel'] = norm
                    a, b = bench(**x), self.old(**x)
                    for u, v in zip(a, b):
                        torch.testing.assert_close(u, v, rtol=0, atol=0)
        x = inputs(lengths=(1,), hk=2, hv=4)
        x['beta'].zero_()
        x['g'] = None
        x['query'].zero_()
        x['query'][0, 0, 3], x['query'][0, 1, 17] = 1, 1
        x['state'].zero_()
        for h in range(4):
            x['state'][0, h, 5, 3] = h + 1
            x['state'][0, h, 5, 17] = (h + 1) * 10
        out, _ = golden(**x)
        self.assertTrue(torch.equal(out[0, :, 5], torch.tensor([1, 2, 30, 40], dtype=torch.bfloat16)))

    def test_unused_state_nan_payload_preserved(self):
        for dtype in (torch.bfloat16, torch.float32):
            x = inputs(lengths=(1,), state_dtype=dtype)
            int_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
            # Signaling NaN payload in an untouched slot must not be converted.
            payload = 0x7f81 if dtype == torch.bfloat16 else 0x7f800001
            x['state'].view(int_dtype)[-1].fill_(payload)
            before = x['state'].view(int_dtype)[-1].clone()
            _, final = golden(**x)
            self.assertTrue(torch.equal(final.view(int_dtype)[-1], before))
            inplace(**x)
            self.assertTrue(torch.equal(x['state'].view(int_dtype)[-1], before))

    def assert_invalid(self, x):
        saved = x['state'].clone()
        with self.assertRaises((ValueError, TypeError)):
            inplace(**x)
        torch.testing.assert_close(x['state'], saved, rtol=0, atol=0)

    def test_cross_request_write_and_initial_read_conflicts(self):
        for indices in ([[0, 1], [1, 2]], [[0, 1, 4], [2, 3, 0]]):
            x = inputs(lengths=(2, 2))
            x['ssm_state_indices'] = torch.tensor(indices, dtype=torch.int32)
            if len(indices[0]) == 3:
                x['num_accepted_tokens'] = torch.tensor([1, 3], dtype=torch.int32)
            self.assert_invalid(x)

    def test_reject_shape_dtype_flags_scale_and_metadata(self):
        for change in (
                lambda x: x.update(query=x['query'].float()),
                lambda x: x.update(beta=x['beta'].half()),
                lambda x: x.update(g=x['g'].double()),
                lambda x: x.update(query=x['query'][:, :, :64], key=x['key'][:, :, :64], state=x['state'][:, :, :, :64]),
                lambda x: x.update(use_qk_l2norm_in_kernel=1),
                lambda x: x.update(use_gate_in_kernel='false'),
                lambda x: x.update(scale=True),
                lambda x: x.update(scale=float('nan')),
                lambda x: x.update(gk=torch.zeros(9, 2, 128)),
                lambda x: x.update(use_gate_in_kernel=True, A_log=None),
                lambda x: x.update(A_log=torch.zeros(3)),
                lambda x: x.update(dt_bias=torch.zeros(2, dtype=torch.bfloat16)),
                lambda x: x.update(actual_seq_lengths=torch.tensor([0, 9], dtype=torch.int64)),
                lambda x: x.update(actual_seq_lengths=torch.tensor([1, 9], dtype=torch.int32)),
                lambda x: x.update(ssm_state_indices=torch.full((9,), 999, dtype=torch.int32)),
                lambda x: x.update(num_accepted_tokens=torch.tensor([10], dtype=torch.int32)),
                lambda x: x.update(ssm_state_indices=torch.zeros(1, 8, dtype=torch.int32))):
            x = inputs()
            change(x)
            self.assert_invalid(x)
        self.assert_invalid(inputs(lengths=(1,), hv=17))
        self.assert_invalid(inputs(lengths=(0,)))

    def test_reject_layout_overlap_and_state_input_alias(self):
        x = inputs()
        x['query'] = torch.zeros(9, 1, 256, dtype=torch.bfloat16)[:, :, ::2]
        self.assert_invalid(x)
        x = inputs()
        x['state'] = x['state'].as_strided(x['state'].shape, (128, 128, 128, 1))
        self.assert_invalid(x)
        x = inputs()
        x['query'] = x['query'].expand(9, 1, 128).as_strided((9, 1, 128), (0, 128, 1))
        self.assert_invalid(x)
        x = inputs(state_dtype=torch.bfloat16)
        x['query'] = x['state'].view(-1)[:9 * 128].view(9, 1, 128)
        self.assert_invalid(x)

    def test_disjoint_storage_alias_and_interleaved_state(self):
        x = inputs(lengths=(1,), hv=2)
        # Same backing storage with disjoint logical intervals is valid.
        count = x['state'].numel()
        storage = torch.empty(count * 2 + 128, dtype=torch.bfloat16)
        state = storage[:count * 2].view(x['state'].shape[0], 2, 128, 256)[..., :128]
        state.copy_(x['state'])
        q = storage[count * 2:].view(1, 1, 128)
        q.copy_(x['query'])
        x.update(state=state, query=q)
        golden(**x)
        # A legal interleaving that the fast sorted-stride span proof cannot prove.
        x = inputs(lengths=(1,), hv=2)
        x['state'] = torch.zeros(128 * 2048).as_strided((3, 2, 128, 128), (256, 384, 2048, 1))
        golden(**x)

    def test_stable_gate_extremes_and_ignored_head_params(self):
        for raw in (-1000.0, -80.0, 0.0, 21.0, 1000.0):
            x = inputs(lengths=(1,), hv=1)
            x.update(g=torch.full((1, 1), raw), use_gate_in_kernel=True,
                     A_log=torch.tensor([0.0]), dt_bias=None)
            out, state = golden(**x)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(state).all())
        x = inputs(lengths=(1,))
        a = golden(**x)
        x.update(A_log=torch.full((2,), float('nan')), dt_bias=torch.full((2,), float('nan')))
        b = golden(**x)
        for u, v in zip(a, b):
            torch.testing.assert_close(u, v, rtol=0, atol=0)

    def test_small_tensor_expanded_views(self):
        x = inputs(lengths=(9,))
        x['beta'] = torch.tensor([0.5]).expand(9, 2)
        x['g'] = torch.tensor([-0.25]).expand(9, 2)
        x['ssm_state_indices'] = torch.tensor([0], dtype=torch.int32).expand(9)
        a = golden(**x)
        b = golden(**{k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in x.items()})
        for u, v in zip(a, b):
            torch.testing.assert_close(u, v, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
