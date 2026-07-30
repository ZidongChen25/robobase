import numpy as np
import pytest

from robobase.language import (
    lang_feature_rows,
    lang_token_rows,
    load_precomputed_language_features,
    tokens_to_feature_array,
    tokens_to_feature_jax,
    tokenize_text,
)

jax = pytest.importorskip("jax")


def test_load_precomputed_language_features_returns_float32_copy(tmp_path):
    path = tmp_path / "language.npy"
    expected = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    np.save(path, expected)

    actual = load_precomputed_language_features(path, feature_dim=8)

    assert actual.shape == (1, 8)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected.astype(np.float32)[None, :])


@pytest.mark.parametrize(
    "features, message",
    [
        (np.zeros((2, 8), dtype=np.float32), "shape"),
        (np.full((1, 8), np.nan, dtype=np.float32), "finite"),
        (np.zeros((1, 8), dtype=np.complex64), "real numeric"),
    ],
)
def test_load_precomputed_language_features_rejects_invalid_arrays(
    tmp_path, features, message
):
    path = tmp_path / "invalid.npy"
    np.save(path, features)

    with pytest.raises(ValueError, match=message):
        load_precomputed_language_features(path, feature_dim=8)


def test_load_precomputed_language_features_disables_pickle(tmp_path):
    path = tmp_path / "object.npy"
    np.save(path, np.full((1, 8), "unsafe", dtype=object), allow_pickle=True)

    with pytest.raises(ValueError, match="safely load"):
        load_precomputed_language_features(path, feature_dim=8)


def test_load_precomputed_language_features_rejects_npz(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, features=np.ones((1, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="single .npy array"):
        load_precomputed_language_features(path, feature_dim=8)


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
