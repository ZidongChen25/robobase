"""Export a Torch CleanDiffuser DiT golden fixture for JAX-only tests.

PyTorch is a development-only dependency of this exporter. The generated NPZ
contains all weights and reference values needed by the pure-JAX unit test.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np


SOURCE_COMMIT = "05f17fc9dbeae7c19a5e264632c9ae9aaac5994e"
FIXTURE_VERSION = 1
MODEL_SEED = 3907
INPUT_SEED = 7901

ACTION_DIM = 3
HORIZON = 8
EMBED_DIM = 32
D_MODEL = 32
N_HEADS = 4
DEPTH = 2
BATCH_SIZE = 2
FOURIER_SCALE = 16.0

PARAM_PREFIX = "param::"


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_clean_checkout(root: Path) -> None:
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"Expected CleanDiffuser commit {SOURCE_COMMIT}, got {commit}."
        )
    if _git_output(root, "status", "--porcelain"):
        raise RuntimeError("CleanDiffuser checkout must be clean before exporting.")


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _add_dense(payload: dict[str, np.ndarray], path: str, state, source: str) -> None:
    payload[f"{PARAM_PREFIX}{path}/kernel"] = _numpy(state[f"{source}.weight"]).T
    payload[f"{PARAM_PREFIX}{path}/bias"] = _numpy(state[f"{source}.bias"])


def _mapped_parameters(model) -> dict[str, np.ndarray]:
    state = model.state_dict()
    payload: dict[str, np.ndarray] = {
        f"{PARAM_PREFIX}time_embedding/fourier_frequencies": _numpy(
            state["map_noise.freqs"]
        )
    }
    _add_dense(payload, "x_proj", state, "x_proj")
    _add_dense(payload, "time_embedding/dense1", state, "map_noise.mlp.0")
    _add_dense(payload, "time_embedding/dense2", state, "map_noise.mlp.2")
    _add_dense(payload, "map_emb_dense1", state, "map_emb.0")
    _add_dense(payload, "map_emb_dense2", state, "map_emb.2")

    for index in range(DEPTH):
        prefix = f"blocks.{index}"
        target = f"block_{index}"
        _add_dense(
            payload,
            f"{target}/adaLN_modulation",
            state,
            f"{prefix}.adaLN_modulation.1",
        )
        payload[f"{PARAM_PREFIX}{target}/attention/in_proj/kernel"] = _numpy(
            state[f"{prefix}.attn.in_proj_weight"]
        ).T
        payload[f"{PARAM_PREFIX}{target}/attention/in_proj/bias"] = _numpy(
            state[f"{prefix}.attn.in_proj_bias"]
        )
        _add_dense(
            payload,
            f"{target}/attention/out_proj",
            state,
            f"{prefix}.attn.out_proj",
        )
        _add_dense(payload, f"{target}/mlp/fc1", state, f"{prefix}.mlp.0")
        _add_dense(payload, f"{target}/mlp/fc2", state, f"{prefix}.mlp.3")

    _add_dense(
        payload,
        "final_layer/adaLN_modulation",
        state,
        "final_layer.adaLN_modulation.1",
    )
    _add_dense(payload, "final_layer/linear", state, "final_layer.linear")
    return payload


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("/tmp/CleanDiffuser-baseline-05f17fc9"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "tests/fixtures/clean_diffuser_dit_fourier_v1.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    clean_root = args.clean_root.resolve()
    _validate_clean_checkout(clean_root)
    sys.path.insert(0, str(clean_root))

    import torch
    from cleandiffuser.nn_diffusion import DiT1d

    torch.manual_seed(MODEL_SEED)
    model = DiT1d(
        in_dim=ACTION_DIM,
        emb_dim=EMBED_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        depth=DEPTH,
        dropout=0.0,
        timestep_emb_type="fourier",
        timestep_emb_params={"scale": FOURIER_SCALE},
    )
    model.train()

    rng = np.random.default_rng(INPUT_SEED)
    warm_actions = torch.tensor(
        rng.normal(size=(BATCH_SIZE, HORIZON, ACTION_DIM)), dtype=torch.float32
    )
    warm_timesteps = torch.tensor(
        rng.uniform(0.05, 0.95, size=(BATCH_SIZE,)), dtype=torch.float32
    )
    warm_condition = torch.tensor(
        rng.normal(size=(BATCH_SIZE, EMBED_DIM)), dtype=torch.float32
    )
    warm_target = torch.tensor(
        rng.normal(size=(BATCH_SIZE, HORIZON, ACTION_DIM)), dtype=torch.float32
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # adaLN-Zero produces a zero function at initialization. A few deterministic
    # steps make all relevant input VJPs non-zero for a meaningful parity test.
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(
            model(warm_actions, warm_timesteps, warm_condition) - warm_target
        ).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    actions = torch.tensor(
        rng.normal(size=(BATCH_SIZE, HORIZON, ACTION_DIM)),
        dtype=torch.float32,
        requires_grad=True,
    )
    timesteps = torch.tensor(
        rng.uniform(0.05, 0.95, size=(BATCH_SIZE,)),
        dtype=torch.float32,
        requires_grad=True,
    )
    condition = torch.tensor(
        rng.normal(size=(BATCH_SIZE, EMBED_DIM)),
        dtype=torch.float32,
        requires_grad=True,
    )
    cotangent = torch.tensor(
        rng.normal(size=(BATCH_SIZE, HORIZON, ACTION_DIM)), dtype=torch.float32
    )
    output = model(actions, timesteps, condition)
    action_vjp, timestep_vjp, condition_vjp = torch.autograd.grad(
        output,
        (actions, timesteps, condition),
        grad_outputs=cotangent,
    )

    payload = _mapped_parameters(model)
    parameter_count = sum(value.size for value in payload.values())
    torch_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != torch_parameter_count:
        raise RuntimeError(
            f"Mapped {parameter_count} values, Torch has {torch_parameter_count}."
        )
    payload.update(
        {
            "fixture_version": np.asarray(FIXTURE_VERSION),
            "source_commit": np.asarray(SOURCE_COMMIT),
            "exporter_sha256": np.asarray(
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ),
            "action_dim": np.asarray(ACTION_DIM),
            "horizon": np.asarray(HORIZON),
            "embed_dim": np.asarray(EMBED_DIM),
            "d_model": np.asarray(D_MODEL),
            "n_heads": np.asarray(N_HEADS),
            "depth": np.asarray(DEPTH),
            "fourier_scale": np.asarray(FOURIER_SCALE, dtype=np.float32),
            "parameter_count": np.asarray(parameter_count),
            "actions": _numpy(actions),
            "timesteps": _numpy(timesteps),
            "condition": _numpy(condition),
            "cotangent": _numpy(cotangent),
            "expected_output": _numpy(output),
            "expected_action_vjp": _numpy(action_vjp),
            "expected_timestep_vjp": _numpy(timestep_vjp),
            "expected_condition_vjp": _numpy(condition_vjp),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Wrote {args.output} ({parameter_count:,} parameters).")


if __name__ == "__main__":
    main()
