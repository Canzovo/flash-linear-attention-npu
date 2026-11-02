"""Profile the same ten fixed cases through the legacy and V2 interfaces.

CPU preprocessing is shared by both versions. Q/K and beta are materialized
in the legacy BF16 input dtype; this measures their common kernel capability,
not the full-precision fused V2 path or preprocessing latency.

The unchanged legacy kernel returns immediately for a sequence longer than
MAX_MTP=8. Keep those raw traces, but report TPB=16/32 as unsupported instead
of treating an empty return as valid computation performance.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops import ascendc
    from fla_npu.ops.ascendc import _aclnn_ctypes as legacy

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from executor_recurrent_gated_delta_rule import build_inputs

    torch.set_num_threads(1)
    source = args.cases.read_bytes()
    cases = json.loads(source)
    if [c["id"] for c in cases] != list(range(10)):
        raise ValueError("Use the fixed first ten performance cases in their original order.")
    records = []
    for case in cases:
        spec = json.loads(next(x["range_values"] for x in case["inputs"] if x["name"] == "case_spec"))
        if spec["block_num"] != spec["T"] or spec["state_indices"] != list(range(spec["T"])):
            raise ValueError("State allocation and indices must cover every token.")
        cpu = build_inputs(spec, torch.device("cpu"))
        # Freeze one legacy-compatible preprocessing result for either kernel.
        for name in ("query", "key"):
            value = cpu[name].float()
            cpu[name] = (value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
        cpu["beta"] = (cpu["beta"].float().sigmoid() * 2.0).bfloat16()
        z = cpu["g"].float() + cpu["dt_bias"]
        cpu["g"] = -cpu["A_log"].exp() * (z.clamp_min(0) + torch.log1p(torch.exp(-z.abs())))
        names = ("query", "key", "value", "beta", "state", "g", "actual_seq_lengths", "ssm_state_indices")
        hashes = {name: hashlib.sha256(memoryview(cpu[name].contiguous().view(torch.uint8).numpy())).hexdigest()
                  for name in names}
        npu = {name: cpu[name].to("npu") for name in names}
        state = npu["state"].clone()

        def launch() -> None:
            state.copy_(npu["state"])
            if args.version == "v1":
                output = torch.empty_like(npu["value"])

                def build(ctx):
                    def desc(tensor, name):
                        return ctx.tensor(tensor, name, acl_format_override=legacy.ACL_FORMAT_ND,
                                          storage_shape_override=tuple(tensor.shape))
                    return [desc(npu["query"], "query"), desc(npu["key"], "key"),
                            desc(npu["value"], "value"), desc(npu["beta"], "beta"),
                            desc(state, "state"), desc(npu["actual_seq_lengths"], "actual_seq_lengths"),
                            desc(npu["ssm_state_indices"], "ssm_state_indices"), desc(npu["g"], "g"),
                            None, None, ctypes.c_float(cpu["scale"]), desc(output, "out")]
                legacy._call_aclnn("aclnnRecurrentGatedDeltaRule", build, output)
            else:
                ascendc.recurrent_gated_delta_rule(
                    npu["query"], npu["key"], npu["value"], state, beta=npu["beta"], scale=cpu["scale"],
                    actual_seq_lengths=npu["actual_seq_lengths"], ssm_state_indices=npu["ssm_state_indices"],
                    g=npu["g"], use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False)
                if ascendc.BACKENDS.get("npu_recurrent_gated_delta_rule") != "stable":
                    raise RuntimeError("The V2 Stable ABI was not selected.")

        for _ in range(5):
            launch()
        torch.npu.synchronize()
        launch()
        torch.npu.synchronize()
        record = dict(case_id=case["id"], BS=spec["B"], HK=spec["HK"], HV=spec["HV"],
                      TOKEN_PER_BATCH=spec["token_per_batch"], T=spec["T"], K=spec["K"], V=spec["V"],
                      block_num=spec["block_num"], seed=spec["seed"], warmups=5, samples=1,
                      state_reset_before_each_call=True, cpu_input_sha256=hashes)
        records.append(record)
        print(json.dumps(record), flush=True)
        del npu, state, cpu
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(dict(
        schema="rgdn_fixed10_common_performance_v1", version=args.version,
        input_json_sha256=hashlib.sha256(source).hexdigest(),
        scope="common kernel capability; CPU preprocessing excluded; Q/K/beta quantized to legacy BF16 for both",
        cases=records), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
