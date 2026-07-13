import numpy as np
import pytest

from robobase.language import (
    lang_feature_rows,
    lang_token_rows,
    tokens_to_feature_array,
    tokens_to_feature_jax,
    tokenize_text,
)

jax = pytest.importorskip("jax")


def test_jax_language_features_match_numpy_reference():
    tokens = tokenize_text("Move the plate between racks")

    expected = tokens_to_feature_array(tokens, feature_dim=32)
    actual = np.asarray(jax.device_get(tokens_to_feature_jax(tokens, feature_dim=32)))

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_language_rows_use_latest_stacked_entry():
    tokens = np.arange(2 * 3 * 1 * 77, dtype=np.int32).reshape(2, 3, 1, 77)
    features = np.arange(2 * 3 * 1 * 4, dtype=np.float32).reshape(2, 3, 1, 4)

    np.testing.assert_array_equal(
        lang_token_rows({"lang_tokens": tokens}, context="test"),
        tokens.reshape(2, -1, 77)[:, -1],
    )
    np.testing.assert_allclose(
        lang_feature_rows({"lang_features": features}, context="test"),
        features.reshape(2, -1, 4)[:, -1],
    )
