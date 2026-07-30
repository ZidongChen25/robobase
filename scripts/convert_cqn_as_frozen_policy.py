#!/usr/bin/env python3
"""Import a validation-selected CQN-AS critic as an exact frozen behavior policy.

The output is a CQN-Flow ``flow_v_direct_a`` analysis checkpoint.  Its policy
tower has the same atoms=51 dueling architecture as the legacy CQN-AS critic,
and receives the legacy target critic plus the exact legacy image encoder.
Flow-V/direct-A remain newly initialized so branch-oracle sidecars can be
trained without changing the selected behavior.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-run-dir", required=True, type=Path)
    parser.add_argument("--legacy-snapshot", required=True, type=Path)
    parser.add_argument("--template-run-dir", required=True, type=Path)
    parser.add_argument("--output-run-dir", required=True, type=Path)
    parser.add_argument("--output-snapshot", type=Path)
    parser.add_argument("--gpu-id", type=int, default=-1)
    return parser.parse_args()


def configure_process(gpu_id: int) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    if gpu_id >= 0:
        gpu = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "agent" not in payload:
        raise ValueError(f"snapshot has no agent state: {path}")
    return payload


def _tree_signature(tree, jax) -> tuple[Any, tuple[tuple[int, ...], ...]]:
    return (
        jax.tree.structure(tree),
        tuple(
            tuple(value.shape)
            for value in jax.tree.leaves(tree)
        ),
    )


def _require_tree_compatible(name: str, source, target, jax) -> None:
    source_signature = _tree_signature(source, jax)
    target_signature = _tree_signature(target, jax)
    if source_signature != target_signature:
        raise ValueError(
            f"{name} source/target parameter structures do not match"
        )


def _tree_bitwise_equal(left, right, jax, np) -> bool:
    if jax.tree.structure(left) != jax.tree.structure(right):
        return False
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    )


def _tree_sha256(tree, jax, np) -> str:
    digest = hashlib.sha256()
    for value in jax.tree.leaves(tree):
        array = np.asarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _matching_architecture_fields(legacy_cfg, converted_cfg) -> dict[str, Any]:
    fields = {
        "action_sequence": (
            int(legacy_cfg.action_sequence),
            int(converted_cfg.action_sequence),
        ),
        "levels": (
            int(legacy_cfg.method.levels),
            int(converted_cfg.method.levels),
        ),
        "bins": (
            int(legacy_cfg.method.bins),
            int(converted_cfg.method.bins),
        ),
        "atoms": (
            int(legacy_cfg.method.atoms),
            int(converted_cfg.method.atoms),
        ),
        "gru_layers": (
            int(legacy_cfg.method.gru_layers),
            int(converted_cfg.method.gru_layers),
        ),
        "hidden_dims": (
            tuple(int(x) for x in legacy_cfg.method.model.hidden_dims),
            tuple(int(x) for x in converted_cfg.method.model.hidden_dims),
        ),
        "activation": (
            str(legacy_cfg.method.model.activation),
            str(converted_cfg.method.model.activation),
        ),
        "use_dueling": (
            bool(legacy_cfg.method.use_dueling),
            bool(converted_cfg.method.use_dueling),
        ),
    }
    mismatched = {
        name: values
        for name, values in fields.items()
        if values[0] != values[1]
    }
    if mismatched:
        raise ValueError(
            f"legacy/template policy architectures differ: {mismatched}"
        )
    return {name: values[0] for name, values in fields.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import numpy as np
    from omegaconf import OmegaConf

    from robobase.workspace import Workspace

    legacy_run_dir = _resolved(args.legacy_run_dir)
    legacy_snapshot = _resolved(args.legacy_snapshot)
    template_run_dir = _resolved(args.template_run_dir)
    output_run_dir = _resolved(args.output_run_dir)
    output_snapshot = _resolved(
        args.output_snapshot
        or output_run_dir / "snapshots" / "source_snapshot.pkl"
    )
    legacy_cfg_path = legacy_run_dir / ".hydra" / "config.yaml"
    template_cfg_path = template_run_dir / ".hydra" / "config.yaml"
    for path in (legacy_cfg_path, template_cfg_path, legacy_snapshot):
        if not path.is_file():
            raise FileNotFoundError(path)

    legacy_cfg = OmegaConf.load(legacy_cfg_path)
    cfg = OmegaConf.load(template_cfg_path)
    if str(legacy_cfg.method.name).lower() != "cqn_as":
        raise ValueError("legacy checkpoint must use method=cqn_as")
    if str(cfg.method.name).lower() != "cqn_flow":
        raise ValueError("template checkpoint must use method=cqn_flow")
    if str(cfg.method.get("critic_architecture", "")).lower() != (
        "flow_v_direct_a"
    ):
        raise ValueError("template must use critic_architecture=flow_v_direct_a")

    OmegaConf.set_struct(cfg, False)
    cfg.method.separate_bc_policy = True
    cfg.method.distinct_policy_encoder = True
    cfg.method.freeze_bc_policy = True
    cfg.method.bc_policy_mode = "legacy_c51"
    cfg.method.policy_value_beta = None
    cfg.method.causal_branch_cache = None
    cfg.method.causal_branch_weight = 0.0
    cfg.method.flow_distill_lambda = 0.0
    cfg.method.flow_distill_action_readout = False
    cfg.create_train_env = False
    cfg.num_train_envs = 0
    cfg.num_train_frames = 0
    cfg.num_eval_envs = 1
    cfg.num_eval_episodes = 1
    cfg.demo_batch_size = None
    cfg.use_self_imitation = False
    cfg.log_train_video = False
    cfg.log_eval_video = False
    cfg.save_snapshot = False
    cfg.save_csv = False
    cfg.gpu_id = None
    cfg.wandb.use = False
    cfg.tb.use = False
    cfg.replay.num_workers = 0
    cfg.replay.persist = False
    cfg.replay.reuse_saved = False
    cfg.backend.platform = "cpu" if args.gpu_id < 0 else "cuda"
    cfg.backend.replay_prefetch_size = 0
    cfg.backend.replay_device_prefetch = False
    cfg.backend.fused_update_steps = 1
    cfg.backend.update_block_every_steps = 1
    cfg.hydra = {"run": {"dir": str(output_run_dir)}}
    OmegaConf.resolve(cfg)
    architecture = _matching_architecture_fields(legacy_cfg, cfg)

    legacy_payload = _load_pickle(legacy_snapshot)
    legacy_state = legacy_payload["agent"]
    legacy_params = legacy_state["params"]
    if "critic" not in legacy_params:
        raise ValueError("legacy agent has no critic parameters")
    if "encoder" not in legacy_params:
        raise ValueError("legacy pixel agent has no encoder parameters")
    legacy_behavior = legacy_state.get(
        "target_critic_params",
        legacy_params["critic"],
    )

    output_run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="convert-cqn-as-policy-"
    ) as temporary_work_dir:
        workspace = Workspace(cfg, work_dir=temporary_work_dir)
        try:
            agent = workspace.agent
            converted_state = agent.state_dict()
            converted_params = dict(converted_state["params"])
            for required in ("policy", "encoder", "policy_encoder"):
                if required not in converted_params:
                    raise ValueError(
                        f"converted agent is missing {required} parameters"
                    )
            _require_tree_compatible(
                "legacy target critic -> frozen policy",
                legacy_behavior,
                converted_params["policy"],
                jax,
            )
            _require_tree_compatible(
                "legacy encoder -> value encoder",
                legacy_params["encoder"],
                converted_params["encoder"],
                jax,
            )
            _require_tree_compatible(
                "legacy encoder -> policy encoder",
                legacy_params["encoder"],
                converted_params["policy_encoder"],
                jax,
            )
            converted_params["policy"] = copy.deepcopy(legacy_behavior)
            converted_params["encoder"] = copy.deepcopy(
                legacy_params["encoder"]
            )
            converted_params["policy_encoder"] = copy.deepcopy(
                legacy_params["encoder"]
            )
            converted_state["params"] = converted_params
            if "encoder_state" in legacy_state:
                converted_state["encoder_state"] = copy.deepcopy(
                    legacy_state["encoder_state"]
                )
            agent.load_state_dict(converted_state)
            roundtrip = agent.state_dict()["params"]
            checks = {
                "policy_bitwise_equal_legacy_target": _tree_bitwise_equal(
                    roundtrip["policy"],
                    legacy_behavior,
                    jax,
                    np,
                ),
                "value_encoder_bitwise_equal_legacy": _tree_bitwise_equal(
                    roundtrip["encoder"],
                    legacy_params["encoder"],
                    jax,
                    np,
                ),
                "policy_encoder_bitwise_equal_legacy": _tree_bitwise_equal(
                    roundtrip["policy_encoder"],
                    legacy_params["encoder"],
                    jax,
                    np,
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"bitwise import checks failed: {checks}")

            output_payload = {
                "snapshot_version": 2,
                "agent": agent.state_dict(),
                "agent_checkpoint_state": agent.checkpoint_state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "converted_policy_metadata": {
                    "legacy_run_dir": str(legacy_run_dir),
                    "legacy_snapshot": str(legacy_snapshot),
                    "template_run_dir": str(template_run_dir),
                    "policy_source": "legacy_target_critic_params",
                    "checks": checks,
                },
            }
            output_snapshot.parent.mkdir(parents=True, exist_ok=True)
            with output_snapshot.open("wb") as handle:
                pickle.dump(output_payload, handle)
        finally:
            workspace.shutdown()

    hydra_dir = output_run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")
    result = {
        "status": "ok",
        "legacy_run_dir": str(legacy_run_dir),
        "legacy_snapshot": str(legacy_snapshot),
        "template_run_dir": str(template_run_dir),
        "output_run_dir": str(output_run_dir),
        "output_snapshot": str(output_snapshot),
        "bc_policy_mode": "legacy_c51",
        "architecture": architecture,
        "checks": checks,
        "hashes": {
            "legacy_behavior": _tree_sha256(
                legacy_behavior,
                jax,
                np,
            ),
            "imported_policy": _tree_sha256(
                roundtrip["policy"],
                jax,
                np,
            ),
            "legacy_encoder": _tree_sha256(
                legacy_params["encoder"],
                jax,
                np,
            ),
            "imported_policy_encoder": _tree_sha256(
                roundtrip["policy_encoder"],
                jax,
                np,
            ),
        },
    }
    (output_run_dir / "conversion.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    args = parse_args()
    configure_process(args.gpu_id)
    started = time.time()
    try:
        payload = run(args)
        payload["elapsed_seconds"] = time.time() - started
        output_run_dir = _resolved(args.output_run_dir)
        (output_run_dir / "conversion.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        output_run_dir = _resolved(args.output_run_dir)
        output_run_dir.mkdir(parents=True, exist_ok=True)
        (output_run_dir / "conversion.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
