"""Language helpers for task text conditioning."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Mapping

import numpy as np


LANG_TOKEN_LENGTH = 77
CLIP_LANG_FEATURE_DIM = 512
CLIP_MODEL_NAME = "ViT-B/32"


def _latest_rows(array, *, dtype, context: str) -> np.ndarray:
    rows = np.asarray(array, dtype=dtype)
    if rows.ndim == 1:
        rows = rows[None, :]
    elif rows.ndim >= 3:
        rows = rows.reshape(rows.shape[0], -1, rows.shape[-1])[:, -1, :]
    if rows.ndim != 2:
        raise ValueError(f"{context} expected rows with shape (B, D), got {rows.shape}.")
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


def clip_tokenize_text(
    text: str,
    *,
    context_length: int = LANG_TOKEN_LENGTH,
) -> np.ndarray:
    import clip

    return (
        clip.tokenize([str(text)], context_length=int(context_length))
        .numpy()
        .astype(np.int32, copy=False)
    )


@lru_cache(maxsize=4)
def _clip_text_model(model_name: str, device: str):
    import clip
    import torch

    model, _ = clip.load(str(model_name), device=str(device))
    if hasattr(model, "visual"):
        del model.visual
    model.eval()
    return model, torch


def clip_tokens_to_feature_array(
    tokens: np.ndarray,
    *,
    device: str = "cpu",
    model_name: str = CLIP_MODEL_NAME,
) -> np.ndarray:
    model, torch = _clip_text_model(str(model_name), str(device))
    token_rows = _latest_rows(
        tokens,
        dtype=np.int64,
        context="CLIP language feature extraction",
    )
    with torch.no_grad():
        tks = torch.as_tensor(token_rows, dtype=torch.long, device=str(device))
        dtype = torch.float16 if tks.device.type == "cuda" else torch.float32
        x = model.token_embedding(tks).type(dtype)
        x = x + model.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x).type(dtype)
        x = x[torch.arange(x.shape[0], device=tks.device), tks.argmax(dim=-1)]
        x = x @ model.text_projection
    return x.float().cpu().numpy().astype(np.float32, copy=False)


def clip_text_feature_array(
    text: str,
    *,
    device: str = "cpu",
    model_name: str = CLIP_MODEL_NAME,
) -> np.ndarray:
    return clip_tokens_to_feature_array(
        clip_tokenize_text(text),
        device=device,
        model_name=model_name,
    )


def tokens_to_feature_array(tokens: np.ndarray, *, feature_dim: int = 512) -> np.ndarray:
    token_rows = np.asarray(tokens, dtype=np.float32)
    if token_rows.ndim != 2:
        raise ValueError(f"Expected language tokens with shape (B, L), got {token_rows.shape}.")
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
        raise ValueError(f"Expected language tokens with shape (B, L), got {token_rows.shape}.")
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
