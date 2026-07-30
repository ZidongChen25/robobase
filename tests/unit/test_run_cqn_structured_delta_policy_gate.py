import argparse

import pytest

from scripts.run_cqn_structured_delta_policy_gate import (
    _default_variants,
    _variant,
)


def test_structured_policy_variants_parse_and_have_ordered_coverage():
    parsed = _variant("test:0.01:3:0.5")

    assert parsed.label == "test"
    assert parsed.min_value_margin == 0.01
    assert parsed.max_state_rms == 3.0
    assert parsed.max_bc_logprob_drop == 0.5

    defaults = _default_variants()
    assert [item.label for item in defaults] == [
        "conservative",
        "medium",
        "wide",
    ]
    assert defaults[0].min_value_margin == defaults[-1].min_value_margin
    assert defaults[0].max_state_rms < defaults[-1].max_state_rms


@pytest.mark.parametrize(
    "value",
    [
        "missing",
        "label:0.1:2",
        "label:-0.1:2:0.5",
        "label:nan:2:0.5",
    ],
)
def test_structured_policy_variant_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _variant(value)
