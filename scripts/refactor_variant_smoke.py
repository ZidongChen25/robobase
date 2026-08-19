"""CPU-only routing/construct/update smoke for the 12 R2 CQN-AS variants.

Each variant is composed through hydra with its own method yaml (flags at the
yaml defaults), constructed through ``robobase.factory.create_agent`` (i.e. the
generic variant branch added in R3), and exercised with one ``act()`` pair and
one ``update()`` on the shared synthetic batch from
``scripts/refactor_equivalence_check.py``.

Run with::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        PYTHONPATH=/home/zc1525/robobase_jaxflat_refactor \
        /home/zc1525/robobase_jaxflat/.venv/bin/python \
        scripts/refactor_variant_smoke.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.refactor_equivalence_check import (  # noqa: E402
    ACTION_SEQUENCE,
    _check_agent,
)

VARIANTS = {
    "cqn_as_structured_explore": "CQNASStructuredExplore",
    "cqn_as_dense_return": "CQNASDenseReturn",
    "cqn_as_fscqn": "CQNASFrozenSupportMask",
    "cqn_as_token_split": "CQNASTokenSplit",
    "cqn_as_mc_rct": "CQNASMcRct",
    "cqn_as_progress_shaping": "CQNASProgressShaping",
    "cqn_as_awr": "CQNASAwr",
    "cqn_as_flow_policy": "CQNASFlowPolicy",
    "cqn_as_bc_policy": "CQNASBcPolicy",
    "cqn_as_twin_critic": "CQNASTwinCritic",
    "cqn_as_td_variants": "CQNASTdVariants",
    "cqn_as_guards": "CQNASGuarded",
}


def main() -> int:
    print("R3 variant routing/act/update smoke (CPU only)")
    for method_name, class_name in VARIANTS.items():
        _check_agent(
            method=method_name,
            expected_module=f"robobase.method.{method_name}",
            expected_class=class_name,
            action_sequence=ACTION_SEQUENCE,
            pixels=False,
            overrides=(f"action_sequence={ACTION_SEQUENCE}",),
        )
    print("All R3 variant routing/act/update smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
