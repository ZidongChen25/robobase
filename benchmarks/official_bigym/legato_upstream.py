"""Load the pinned official Legato Kinetix implementation without vendoring it."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Literal


UPSTREAM_COMMIT = "d302701268aa3a50ec7f07189cc3af3b31014f63"
UPSTREAM_KINETIX_COMMIT = "cf7453ea103fa0b77348af1a39f689c658161613"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = REPOSITORY_ROOT / "third_party" / "Legato_kinetix_official"
UPSTREAM_SOURCE = UPSTREAM_ROOT / "src"

UpstreamModel = Literal["vanilla", "legato"]


def checkout_commit(path: Path = UPSTREAM_ROOT) -> str:
    """Return the checked-out upstream commit without contacting the network."""
    if not (path / ".git").exists():
        raise FileNotFoundError(
            f"Official Legato checkout is missing at {path}. Clone "
            "https://github.com/lyfeng001/Legato-kinetix there."
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_upstream_checkout(path: Path = UPSTREAM_ROOT) -> None:
    """Fail closed when the adapter is pointed at a different upstream revision."""
    actual = checkout_commit(path)
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(
            "Official Legato checkout revision mismatch: "
            f"expected {UPSTREAM_COMMIT}, found {actual}."
        )
    submodule = path / "third_party" / "kinetix"
    if not submodule.exists():
        raise FileNotFoundError(
            f"Kinetix submodule is missing at {submodule}; run "
            f"git -C {path} submodule update --init."
        )
    submodule_commit = checkout_commit(submodule)
    if submodule_commit != UPSTREAM_KINETIX_COMMIT:
        raise RuntimeError(
            "Official Kinetix submodule revision mismatch: "
            f"expected {UPSTREAM_KINETIX_COMMIT}, found {submodule_commit}."
        )


def _module_path(model: UpstreamModel) -> Path:
    filename = "model.py" if model == "vanilla" else "model_legato.py"
    path = UPSTREAM_SOURCE / filename
    if not path.is_file():
        raise FileNotFoundError(f"Official Legato source file is missing: {path}")
    return path


def load_upstream_module(
    model: UpstreamModel,
    *,
    verify_commit: bool = True,
) -> ModuleType:
    """Import one official model file under a revision-specific module name."""
    if verify_commit:
        verify_upstream_checkout()
    module_name = f"_legato_official_{model}_{UPSTREAM_COMMIT[:12]}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, _module_path(model))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official Legato {model} module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


__all__ = [
    "UPSTREAM_COMMIT",
    "UPSTREAM_KINETIX_COMMIT",
    "UPSTREAM_ROOT",
    "checkout_commit",
    "load_upstream_module",
    "verify_upstream_checkout",
]
