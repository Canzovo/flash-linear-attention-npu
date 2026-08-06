"""Static op_host contract for chunk_kda_fwd; device execution lives in accuracy/routes."""

from tests.operators.chunk_kda_fwd.common.case_matrix import manifest


def test_host_contract_has_platform_and_negative_matrix():
    data = manifest()
    assert set(data["capability"]["soc"]) >= {"ascend910b", "ascend910_93", "ascend950"}
    negatives = [case for case in data["cases"] if "negative" in case["tags"]]
    assert negatives
    for case in negatives:
        assert case["expect"]["return_code"] != "ACLNN_SUCCESS"
        assert case["expect"].get("message_contains")
        assert "aclnn" in case["run_on"] or case["expect"]["return_code"] == "RuntimeError"


def test_route_case_uses_one_shape_definition():
    data = manifest()
    route_cases = [case for case in data["cases"] if "route" in case["tags"]]
    assert route_cases
    assert any({"ascendc", "aclnn", "direct_launch"} <= set(case["run_on"]) for case in route_cases)


def test_a5_h96_model_performance_cases_keep_full_preprocess_contract():
    data = manifest()
    cases = {case["id"]: case for case in data["cases"]}
    for suffix, tokens, chunks in (("t8k", 8192, 128), ("t16k", 16384, 256)):
        case = cases[f"chunk_kda_fwd_h96_{suffix}_model_performance"]
        assert case["soc"] == ["ascend950"]
        assert case["layout"] == "BSND"
        assert case["shape"] == {
            "B": 1,
            "H_k": 96,
            "H_v": 96,
            "T": tokens,
            "K": 128,
            "V": 128,
            "chunk_size": 64,
            "N_c": chunks,
        }
        assert case["dtype"]["q_k_v"] == "bfloat16"
        assert case["dtype"]["beta"] == "bfloat16"
        assert case["attrs"]["use_gate_in_kernel"] is True
        assert case["attrs"]["use_qk_l2norm_in_kernel"] is True
        assert case["attrs"]["use_beta_sigmoid_in_kernel"] is True
        assert case["attrs"]["safe_gate"] is True
        assert case["attrs"]["lower_bound"] == -5.0
