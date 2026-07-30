import argparse

import pytest

from scripts.run_cqn_floq_clean_fallback_gate import (
    _default_variants,
    _variant,
)


def test_floq_fallback_variants_parse_and_order_coverage():
    parsed = _variant("test:1.2:0.3:0.4:0.75:0.01")

    assert parsed.label == "test"
    assert parsed.min_value_margin == 1.2
    assert parsed.max_bc_logprob_drop == 0.3
    assert parsed.max_best_bc_logprob_drop == 0.4
    assert parsed.min_source_win_fraction == 0.75
    assert parsed.min_source_mean_delta == 0.01

    defaults = _default_variants()
    assert [item.label for item in defaults] == [
        "conservative",
        "medium",
        "wide",
    ]
    assert defaults[0].min_value_margin > defaults[-1].min_value_margin
    assert (
        defaults[0].min_source_win_fraction
        > defaults[-1].min_source_win_fraction
    )
    assert (
        defaults[0].max_bc_logprob_drop
        < defaults[-1].max_bc_logprob_drop
    )


@pytest.mark.parametrize(
    "value",
    [
        "missing",
        "label:1:0.2:0.2:0.5",
        "label:-1:0.2:0.2:0.5:0",
        "label:1:nan:0.2:0.5:0",
        "label:1:0.2:0.2:1.1:0",
        "label:1:0.2:0.2:0.5:nan",
    ],
)
def test_floq_fallback_variant_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _variant(value)
