"""Utilities for the pinned official A2A checkout.

Imports of the official PyTorch stack are intentionally lazy so this module can
be inspected and tested from the JAX-only production environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import typing


OFFICIAL_A2A_URL = "https://github.com/JIAjindou/A2A_Flow_Matching.git"
OFFICIAL_A2A_COMMIT = "a5792ecf4e7f8fa4d85fe66ea9a50618138f925c"


def validate_official_checkout(
    checkout: str | Path,
    *,
    expected_commit: str | None = OFFICIAL_A2A_COMMIT,
) -> tuple[Path, str]:
    """Validate the isolated upstream checkout and return its exact commit."""

    checkout = Path(checkout).expanduser().resolve()
    policy_file = checkout / "roboverse_learn/il/policies/a2a/a2a_policy.py"
    if not policy_file.is_file():
        raise FileNotFoundError(
            f"{checkout} is not an A2A checkout; missing {policy_file.relative_to(checkout)}."
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"Official A2A checkout is at {commit}, expected {expected_commit}. "
            "Pass --allow-unpinned-upstream only for an intentional comparison."
        )
    return checkout, commit


def add_official_checkout_to_path(checkout: str | Path) -> Path:
    checkout, _ = validate_official_checkout(checkout, expected_commit=None)
    loaded = sys.modules.get("roboverse_learn")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        loaded_paths = (
            [Path(loaded_file).resolve()]
            if loaded_file is not None
            else [Path(path).resolve() for path in getattr(loaded, "__path__", ())]
        )
        if not any(path == checkout or checkout in path.parents for path in loaded_paths):
            raise RuntimeError(
                "A different roboverse_learn package is already imported from "
                f"{loaded_paths}. Start a fresh process for the official benchmark."
            )
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    return checkout


def patch_diffusers_compat() -> None:
    """Restore typing re-exports expected by the pinned official checkout.

    The upstream scheduler imports these names from ``diffusers.optimization``.
    Recent Diffusers releases stopped re-exporting them. Supplying the same
    objects before importing the runner keeps the official scheduler code
    unchanged and works across both old and current Diffusers versions.
    """

    import diffusers.optimization as optimization
    from torch.optim import Optimizer

    compatibility_names = {
        "Optimizer": Optimizer,
        "Optional": typing.Optional,
        "Union": typing.Union,
    }
    for name, value in compatibility_names.items():
        if not hasattr(optimization, name):
            setattr(optimization, name, value)


def file_sha256(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_checkpoint(
    checkpoint: str | Path,
    checkout: str | Path,
    *,
    device: str,
):
    """Load the official runner payload and return its EMA policy and config."""

    add_official_checkout_to_path(checkout)
    patch_diffusers_compat()
    try:
        import dill
        import torch
        from roboverse_learn.il.runners.default_runner import DefaultRunner
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "The official A2A environment requires torch, torchcfm, dill, Hydra, "
            "diffusers, and the checkout's RoboVerse dependencies."
        ) from exc

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    load_kwargs = {"pickle_module": dill, "map_location": "cpu"}
    try:
        payload = torch.load(checkpoint.open("rb"), weights_only=False, **load_kwargs)
    except TypeError:  # Older PyTorch does not expose weights_only.
        payload = torch.load(checkpoint.open("rb"), **load_kwargs)
    runner = DefaultRunner(payload["cfg"], output_dir=str(checkpoint.parent.parent))
    runner.load_payload(payload, exclude_keys=None, include_keys=None)
    use_ema = bool(payload["cfg"].train_config.training_params.use_ema)
    policy = runner.ema_model if use_ema else runner.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, payload["cfg"]


def load_official_a2a_checkpoint(
    checkpoint: str | Path,
    checkout: str | Path,
    *,
    device: str,
):
    """Backward-compatible name for the shared official policy loader."""

    return load_official_checkpoint(checkpoint, checkout, device=device)


__all__ = [
    "OFFICIAL_A2A_COMMIT",
    "OFFICIAL_A2A_URL",
    "add_official_checkout_to_path",
    "file_sha256",
    "load_official_a2a_checkpoint",
    "load_official_checkpoint",
    "patch_diffusers_compat",
    "validate_official_checkout",
]
