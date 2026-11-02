#!/usr/bin/env python3
"""Verify real ACLNN Error Messages, rejection before launch and valid controls.

This bypasses the Python wrapper's guards intentionally. It checks the C API's
return value AND aclGetRecentErrMsg, with ordinary logs disabled. It does not
replace ATK accuracy or memory checks. Run with the installed current package.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path


TENSOR_NAMES = ("query", "key", "value", "beta", "state", "actual_seq_lengths",
                "ssm_state_indices", "g", "gk", "num_accepted_tokens", "A_log", "dt_bias")
REQUIRED = TENSOR_NAMES[:7] + ("out",)
INVALID_PARAMETER = 161002
FORMAT_ND = 2
FORMAT_FRACTAL_NZ = 29


def fixture(torch):
    def zeros(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device="npu")
    return dict(query=zeros((2, 2, 128)), key=zeros((2, 2, 128)),
                value=zeros((2, 4, 128)), beta=zeros((2, 4)),
                state=zeros((2, 4, 128, 128)), out=zeros((2, 4, 128)),
                actual_seq_lengths=torch.tensor([0, 2], dtype=torch.int32, device="npu"),
                ssm_state_indices=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
                g=zeros((2, 4), torch.float32), gk=None,
                num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device="npu"),
                A_log=zeros((4,), torch.float32), dt_bias=zeros((4,), torch.float32),
                scaleValue=0.25, use_qk_l2norm_in_kernel=False,
                use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False)


def negative_cases(torch):
    cases = []

    def add(name, mutate, code, parameter, details=(), version="v2", **options):
        cases.append((name, mutate, code, parameter, details, version, options))

    def replace(key, shape, dtype=torch.bfloat16):
        return lambda case: case.update({key: torch.empty(shape, dtype=dtype, device="npu")})

    for version in ("v1", "v2"):
        for key in REQUIRED:
            add(f"{version}_null_{key}", lambda c, k=key: c.update({k: None}),
                "EZ0004", key, version=version)
        for key in ("workspaceSize", "executor"):
            add(f"{version}_null_{key}", lambda c: None, "EZ0004", key,
                version=version, null_pointer=key, expected_status=161001)
    wrong_types = {"query": torch.float32, "key": torch.float16, "value": torch.float32,
                   "out": torch.float32, "state": torch.float16, "beta": torch.float16,
                   "g": torch.int64, "A_log": torch.bfloat16, "dt_bias": torch.bfloat16,
                   "actual_seq_lengths": torch.int64, "ssm_state_indices": torch.int64,
                   "num_accepted_tokens": torch.int64}
    for key, dtype in wrong_types.items():
        add("v2_dtype_" + key,
            lambda c, k=key, d=dtype: c.update({k: c[k].to(d)}),
            "EZ0007", key, ("dtype", "should"))
    ranks = {"query": (2, 256), "key": (2, 256), "value": (2, 512), "out": (2, 512),
             "state": (2, 4, 16384), "beta": (8,), "g": (8,), "A_log": (1, 4),
             "dt_bias": (1, 4), "actual_seq_lengths": (1, 2), "ssm_state_indices": (1, 1, 2),
             "num_accepted_tokens": (1, 1)}
    for key, shape in ranks.items():
        dtype = (torch.int32 if key in TENSOR_NAMES[5:7] + ("num_accepted_tokens",)
                 else torch.float32 if key in ("g", "A_log", "dt_bias") else torch.bfloat16)
        add("v2_rank_" + key, replace(key, shape, dtype), "EZ0011", key, ("dimension",))
    shapes = [("query", (2, 2, 64)), ("key", (2, 2, 64)), ("key", (3, 2, 128)),
              ("value", (2, 4, 64)), ("beta", (2, 3)), ("state", (2, 3, 128, 128)),
              ("state", (2, 4, 128, 64)), ("out", (3, 4, 128)), ("ssm_state_indices", (1,)),
              ("g", (2, 5)), ("A_log", (3,)), ("dt_bias", (3,)), ("num_accepted_tokens", (2,))]
    for index, (key, shape) in enumerate(shapes):
        dtype = (torch.int32 if key in ("ssm_state_indices", "num_accepted_tokens")
                 else torch.float32 if key in ("g", "A_log", "dt_bias") else torch.bfloat16)
        add(f"v2_shape_{key}_{index}", replace(key, shape, dtype), "EZ0008", key, ("shape", "should"))
    for shape in ((2, 1), (1, 0)):
        add(f"v2_indices_2d_{shape}", replace("ssm_state_indices", shape, torch.int32),
            "EZ0009", "ssm_state_indices", ("Ckpt",))
    for value in (float("nan"), float("inf")):
        add(f"v2_scale_{value}", lambda c, x=value: c.update(scaleValue=x),
            "EZ0024", "scaleValue", ("finite float32",))
    for hv in (5, 34):
        add(f"v2_ratio_{hv}", replace("value", (2, hv, 128)),
            "EZ0027", "query.HK,value.HV", ("multiple", "16"))
    for key in ("g", "A_log"):
        add("v2_raw_gate_missing_" + key,
            lambda c, k=key: c.update({k: None, "use_gate_in_kernel": True}),
            "EZ0037", "use_gate_in_kernel", ("requires", "null"))
    add("v2_gk_unsupported", replace("gk", (2, 2, 128), torch.float32),
        "EZ0037", "gk", ("V1", "V2"))
    add("v2_query_stride", lambda c: c.update(query=torch.empty_strided(
        (2, 2, 128), (512, 256, 2), dtype=torch.bfloat16, device="npu")),
        "EZ0028", "query", ("stride", "last stride=1"))
    add("v2_state_stride", lambda c: c.update(state=torch.empty_strided(
        (2, 4, 128, 128), (131072, 32768, 256, 2), dtype=torch.bfloat16, device="npu")),
        "EZ0028", "state", ("stride", "K stride=1"))
    add("v2_out_stride", lambda c: c.update(out=torch.empty_strided(
        (2, 4, 128), (1024, 256, 2), dtype=torch.bfloat16, device="npu")),
        "EZ0028", "out", ("contiguous",))
    add("v2_state_self_overlap", lambda c: c.update(state=c["state"].as_strided(
        (2, 4, 128, 128), (65536, 128, 128, 1))),
        "EZ0037", "state", ("overlap",))
    add("v2_out_is_value", lambda c: c.update(out=c["value"]),
        "EZ0037", "value", ("overlaps writable out",))
    add("v2_out_descriptor_is_value", lambda c: c.update(out=c["value"]),
        "EZ0037", "value", ("overlaps writable out",), reuse_descriptor={"out": "value"})
    add("v2_query_alias_state", lambda c: c.update(query=c["state"].view(-1)[:512].view(2, 2, 128)),
        "EZ0037", "query", ("overlaps writable state",))
    add("v2_out_alias_state", lambda c: c.update(out=c["state"].view(-1)[:1024].view(2, 4, 128)),
        "EZ0037", "state,out", ("overlap",))
    add("v2_query_storage_bounds", lambda c: None, "EZ0037", "query",
        ("storage bounds", "storage=[1]"), descriptor={"query": {"storage_shape_override": (1,)}})
    add("v2_query_format", lambda c: None, "EZ0006", "query", ("FRACTAL_NZ",),
        descriptor={"query": {"acl_format_override": FORMAT_FRACTAL_NZ}})
    for key, dtype in (("beta", torch.float32), ("g", torch.bfloat16),
                       ("state", torch.float16), ("actual_seq_lengths", torch.int64),
                       ("ssm_state_indices", torch.int64), ("num_accepted_tokens", torch.int64)):
        add("v1_dtype_" + key, lambda c, k=key, d=dtype: c.update({k: c[k].to(d)}),
            "EZ0007", key, ("dtype",), version="v1")
    # V1 keeps its existing inference/tiling rejection stage and status. Check
    # that the specific report survives propagation through the L0 launcher.
    for key, shape in (("state", (2, 4, 16384)), ("value", (2, 512)), ("query", (2, 256))):
        add("v1_rank_" + key, replace(key, shape), "EZ0011", key, ("shape dim",),
            version="v1", expected_status=561103)
    add("v1_key_width", replace("key", (2, 2, 64)), "EZ0037", "input shapes",
        ("inconsistent dimensions", "key=[2,2,64]"), version="v1", expected_status=561103)
    return cases


def invoke(torch, runtime_module, case, version, *, options=None, launch=False):
    options = options or {}
    runtime = runtime_module.runtime()
    context = runtime_module._CallContext(runtime, torch.device("npu", torch.npu.current_device()))
    endpoint = "aclnnRecurrentGatedDeltaRule" + ("V2" if version == "v2" else "")
    api = runtime.symbol(endpoint + "GetWorkspaceSize")
    api.argtypes = ([ctypes.c_void_p] * (12 if version == "v2" else 10) + [ctypes.c_float]
                    + ([ctypes.c_bool] * 4 if version == "v2" else [])
                    + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
    api.restype = ctypes.c_int
    workspace, executor = ctypes.c_uint64(), ctypes.c_void_p()
    try:
        descriptors = {}

        def tensor(key):
            reused = options.get("reuse_descriptor", {}).get(key)
            if reused:
                return descriptors[reused]
            descriptor = {"acl_format_override": FORMAT_ND,
                          **options.get("descriptor", {}).get(key, {})}
            descriptors[key] = context.tensor(case[key], key, **descriptor)
            return descriptors[key]
        names = TENSOR_NAMES if version == "v2" else TENSOR_NAMES[:10]
        args = [tensor(key) for key in names] + [case["scaleValue"]]
        if version == "v2":
            args += [case[key] for key in ("use_qk_l2norm_in_kernel", "use_gate_in_kernel",
                                          "use_beta_sigmoid_in_kernel", "allow_neg_eigval")]
        args += [tensor("out"),
                 None if options.get("null_pointer") == "workspaceSize" else ctypes.byref(workspace),
                 None if options.get("null_pointer") == "executor" else ctypes.byref(executor)]
        runtime._recent_error_message()
        status = api(*args)
        message = runtime._recent_error_message()
        if launch and status == 0:
            buffer = torch.empty(workspace.value, dtype=torch.uint8, device="npu")
            run = runtime.symbol(endpoint)
            run.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
            run.restype = ctypes.c_int
            run_status = run(buffer.data_ptr() if workspace.value else None, workspace.value,
                             executor, torch.npu.current_stream().npu_stream)
            assert run_status == 0, (endpoint, run_status, runtime._recent_error_message())
            torch.npu.synchronize()
        return status, message, workspace.value, bool(executor.value), endpoint
    finally:
        context.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Process-visible NPU ordinal")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "4"
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import _runtime
    torch.npu.set_device(args.device)
    records = []
    for version in ("v1", "v2"):
        case = fixture(torch)
        # A successful no-op must not pass as a valid compute control.
        case["out"].fill_(float("nan"))
        status, message, _, executor, _ = invoke(torch, _runtime, case, version, launch=True)
        assert status == 0 and executor, (version, status, message)
        assert not torch.count_nonzero(case["out"]).item() and not torch.count_nonzero(case["state"]).item()
        records.append(dict(name=version + "_valid_control", passed=True, status=status))
        print("PASS", version + "_valid_control", flush=True)
    for name, mutate, code, parameter, details, version, options in negative_cases(torch):
        case = fixture(torch)
        mutate(case)
        state = case["state"]
        before = state.cpu().clone() if state is not None else None
        state_version = state._version if state is not None else None
        record = dict(name=name, expected_error_code=code, parameter=parameter, passed=False)
        try:
            status, message, workspace, executor, endpoint = invoke(torch, _runtime, case, version, options=options)
            record.update(status=status, message=message)
            assert status == options.get("expected_status", INVALID_PARAMETER), (status, message)
            assert code in message and parameter in message and endpoint in message, message
            assert all(token.lower() in message.lower() for token in details), (details, message)
            assert not executor and workspace == 0, (workspace, executor)
            assert state is None or (state._version == state_version and torch.equal(state.cpu(), before)), "State changed"
            record["passed"] = True
            print(f"PASS {name}: aclnnStatus={status}; {message.strip()}", flush=True)
        except Exception as error:
            record["failure"] = str(error)
            print(f"FAIL {name}: {error}", flush=True)
        records.append(record)
    summary = dict(total=len(records), passed=sum(r["passed"] for r in records),
                   failed=sum(not r["passed"] for r in records), cases=records)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Error Message checks: {summary['passed']}/{summary['total']} PASS", flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
