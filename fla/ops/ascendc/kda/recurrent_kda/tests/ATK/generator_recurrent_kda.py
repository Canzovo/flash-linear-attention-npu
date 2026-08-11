"""Deterministic ATK generator for the eight recurrent_kda decode cases."""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


FIXED_SEED = 20260811
BATCHES = (1, 4, 16, 64)
MODES = ("base", "cb_mtp")
CASE_MATRIX = tuple((batch, mode) for batch in BATCHES for mode in MODES)


@GENERATOR_REGISTRY.register("generator_recurrent_kda")
class RecurrentKdaGenerator(CaseGenerator):
    """Force the generated cases to cover the requested Cartesian product."""

    def __init__(self, config):
        super().__init__(config)
        self._case_index = 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        batch, mode = CASE_MATRIX[self._case_index % len(CASE_MATRIX)]
        self._case_index += 1
        case_config.inputs[0].range_values = batch
        case_config.inputs[1].range_values = mode
        case_config.inputs[2].range_values = FIXED_SEED
        case_config.default_seed = FIXED_SEED
        return case_config

