"""Language helpers for task text conditioning."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping

import numpy as np


LANG_TOKEN_LENGTH = 77


def load_precomputed_language_features(
    path: str | Path | None,
    *,
    feature_dim: int = 512,
) -> np.ndarray:
    """Load one fixed language feature row without enabling pickle loading."""

    feature_dim = int(feature_dim)
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    if path is None or not str(path).strip():
        raise ValueError("A precomputed language feature path is required.")

    feature_path = Path(path).expanduser()
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Precomputed language feature file does not exist: {feature_path}"
        )
    try:
        loaded = np.load(feature_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Failed to safely load precomputed language features from "
            f"{feature_path}: {exc}"
        ) from exc

    if not isinstance(loaded, np.ndarray):
        close = getattr(loaded, "close", None)
        if callable(close):
            close()
        raise ValueError(
            "Precomputed language features must be stored as a single .npy array."
        )
    if loaded.ndim == 1:
        loaded = loaded[None, :]
    expected_shape = (1, feature_dim)
    if loaded.shape != expected_shape:
        raise ValueError(
            "Precomputed language features must have shape "
            f"{expected_shape}, got {loaded.shape}."
        )
    if not np.issubdtype(loaded.dtype, np.number) or np.issubdtype(
        loaded.dtype, np.complexfloating
    ):
        raise ValueError(
            "Precomputed language features must contain real numeric values, "
            f"got dtype {loaded.dtype}."
        )

    features = np.asarray(loaded, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(
            "Precomputed language features must contain only finite values."
        )
    return np.array(features, dtype=np.float32, copy=True)


def _latest_rows(array, *, dtype, context: str) -> np.ndarray:
    rows = np.asarray(array, dtype=dtype)
    if rows.ndim == 1:
        rows = rows[None, :]
    elif rows.ndim >= 3:
        rows = rows.reshape(rows.shape[0], -1, rows.shape[-1])[:, -1, :]
    if rows.ndim != 2:
        raise ValueError(
            f"{context} expected rows with shape (B, D), got {rows.shape}."
        )
    return rows


def tokenize_text(text: str, *, context_length: int = LANG_TOKEN_LENGTH) -> np.ndarray:
    """Return stable integer tokens for a short task description.

    This is intentionally not a CLIP tokenizer. It gives the pure-JAX training
    stack task-distinguishing language tokens without pulling in PyTorch.
    """

    words = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
    tokens = np.zeros((context_length,), dtype=np.int32)
    tokens[0] = 1
    max_words = max(0, context_length - 2)
    for index, word in enumerate(words[:max_words], start=1):
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()
        tokens[index] = int.from_bytes(digest, "little", signed=False) % 32000 + 2
    end_index = min(len(words), max_words) + 1
    if end_index < context_length:
        tokens[end_index] = 2
    return tokens[None, :]


def lang_token_rows(batch_or_obs: Mapping, *, context: str) -> np.ndarray:
    if "lang_tokens" not in batch_or_obs:
        raise ValueError(f"{context} requires 'lang_tokens' observations.")
    return _latest_rows(
        batch_or_obs["lang_tokens"],
        dtype=np.int32,
        context=context,
    )


def lang_feature_rows(batch_or_obs: Mapping, *, context: str) -> np.ndarray:
    if "lang_features" not in batch_or_obs:
        raise ValueError(f"{context} requires 'lang_features' observations.")
    return _latest_rows(
        batch_or_obs["lang_features"],
        dtype=np.float32,
        context=context,
    )


def tokens_to_feature_array(
    tokens: np.ndarray, *, feature_dim: int = 512
) -> np.ndarray:
    token_rows = np.asarray(tokens, dtype=np.float32)
    if token_rows.ndim != 2:
        raise ValueError(
            f"Expected language tokens with shape (B, L), got {token_rows.shape}."
        )
    feature_dim = int(feature_dim)
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")

    positions = np.arange(token_rows.shape[1], dtype=np.float32)[:, None]
    freqs = np.arange(1, feature_dim + 1, dtype=np.float32)[None, :]
    phases = token_rows[:, :, None] * (freqs[None, :, :] * 1.0e-4)
    phases = phases + positions[None, :, :] * (freqs[None, :, :] * 1.0e-3)
    features = np.sin(phases) + np.cos(phases * 0.5)
    features = features.mean(axis=1).astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, 1.0e-6)


def tokens_to_feature_jax(tokens, *, feature_dim: int = 512):
    """JAX version of ``tokens_to_feature_array`` for JAX-only methods."""

    import jax.numpy as jnp

    token_rows = jnp.asarray(tokens, dtype=jnp.float32)
    if token_rows.ndim != 2:
        raise ValueError(
            f"Expected language tokens with shape (B, L), got {token_rows.shape}."
        )
    feature_dim = int(feature_dim)
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")

    positions = jnp.arange(token_rows.shape[1], dtype=jnp.float32)[:, None]
    freqs = jnp.arange(1, feature_dim + 1, dtype=jnp.float32)[None, :]
    phases = token_rows[:, :, None] * (freqs[None, :, :] * 1.0e-4)
    phases = phases + positions[None, :, :] * (freqs[None, :, :] * 1.0e-3)
    features = jnp.sin(phases) + jnp.cos(phases * 0.5)
    features = features.mean(axis=1)
    norms = jnp.linalg.norm(features, axis=-1, keepdims=True)
    return features / jnp.maximum(norms, jnp.asarray(1.0e-6, dtype=features.dtype))
