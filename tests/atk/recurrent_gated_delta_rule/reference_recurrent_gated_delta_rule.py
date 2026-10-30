"""Compatibility imports; the unique CPU implementation lives in reference.py."""

from reference import (
    QK_L2_NORM_EPS,
    l2_normalize_qk,
    recurrent_gated_delta_rule_reference,
    recurrent_gated_delta_rule_bench,
    recurrent_gated_delta_rule_golden,
)

__all__ = ["QK_L2_NORM_EPS", "l2_normalize_qk", "recurrent_gated_delta_rule_reference",
           "recurrent_gated_delta_rule_bench", "recurrent_gated_delta_rule_golden"]
