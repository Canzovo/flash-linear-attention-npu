"""Verify the native atk case output preserves the frozen full matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_recurrent_gated_delta_rule import STANDARD, build_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()
    actual = json.loads(args.json_path.read_text(encoding="utf-8"))
    expected = build_specs()
    assert len(actual) == len(expected) == 906
    for index, (case, spec) in enumerate(zip(actual, expected)):
        assert case["id"] == index
        assert case["default_seed"] == spec["seed"]
        inputs = {item["name"]: item for item in case["inputs"]}
        assert json.loads(inputs["case_spec"]["range_values"]) == spec, index
        assert case["standard"]["acc"] == STANDARD["acc"]
        assert case["api_type"] == "executor_recurrent_gated_delta_rule"
    print(json.dumps(dict(cases=len(actual), ids=[actual[0]["id"], actual[-1]["id"]],
                          frozen_specs_equal=True, native_standard_equal=True)))


if __name__ == "__main__":
    main()
