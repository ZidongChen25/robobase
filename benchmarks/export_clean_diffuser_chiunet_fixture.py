"""Export a Torch-generated ChiUNet golden fixture for JAX-only tests.

This script is an external development tool. It intentionally imports PyTorch
and the pinned CleanDiffuser checkout, while the generated fixture and its test
have no Torch dependency.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


SOURCE_COMMIT = "05f17fc9dbeae7c19a5e264632c9ae9aaac5994e"
FIXTURE_VERSION = 1
MODEL_SEED = 1103
INPUT_SEED = 2207

ACTION_DIM = 2
OBS_DIM = 3
OBS_STEPS = 2
HORIZON = 8
EMBED_DIM = 32
DOWN_DIMS = (32, 64, 128)
KERNEL_SIZE = 5
N_GROUPS = 8
BATCH_SIZE = 2

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
    status = _git_output(root, "status", "--porcelain")
    if status:
        raise RuntimeError("CleanDiffuser checkout must be clean before exporting.")


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _add_dense(payload: dict[str, np.ndarray], path: str, state, source: str) -> None:
    payload[f"{PARAM_PREFIX}{path}/kernel"] = _numpy(state[f"{source}.weight"]).T
    payload[f"{PARAM_PREFIX}{path}/bias"] = _numpy(state[f"{source}.bias"])


def _add_conv1d(payload: dict[str, np.ndarray], path: str, state, source: str) -> None:
    payload[f"{PARAM_PREFIX}{path}/kernel"] = np.transpose(
        _numpy(state[f"{source}.weight"]),
        (2, 1, 0),
    )
    payload[f"{PARAM_PREFIX}{path}/bias"] = _numpy(state[f"{source}.bias"])


def _add_conv_transpose1d(
    payload: dict[str, np.ndarray], path: str, state, source: str
) -> None:
    # Torch stores ConvTranspose1d as (in, out, kernel); Flax with
    # transpose_kernel=True consumes (kernel, out, in).
    payload[f"{PARAM_PREFIX}{path}/kernel"] = np.transpose(
        _numpy(state[f"{source}.weight"]),
        (2, 1, 0),
    )
    payload[f"{PARAM_PREFIX}{path}/bias"] = _numpy(state[f"{source}.bias"])


def _add_group_norm(
    payload: dict[str, np.ndarray], path: str, state, source: str
) -> None:
    payload[f"{PARAM_PREFIX}{path}/scale"] = _numpy(state[f"{source}.weight"])
    payload[f"{PARAM_PREFIX}{path}/bias"] = _numpy(state[f"{source}.bias"])


def _add_residual_block(
    payload: dict[str, np.ndarray],
    path: str,
    state,
    source: str,
) -> None:
    _add_conv1d(payload, f"{path}/block1/conv", state, f"{source}.conv1.0")
    _add_group_norm(payload, f"{path}/block1/norm", state, f"{source}.conv1.1")
    _add_conv1d(payload, f"{path}/block2/conv", state, f"{source}.conv2.0")
    _add_group_norm(payload, f"{path}/block2/norm", state, f"{source}.conv2.1")
    _add_dense(payload, f"{path}/cond_dense", state, f"{source}.cond_encoder.1")
    residual_weight = state.get(f"{source}.residual_conv.weight")
    if residual_weight is not None:
        payload[f"{PARAM_PREFIX}{path}/residual_dense/kernel"] = _numpy(
            residual_weight
        )[:, :, 0].T
        payload[f"{PARAM_PREFIX}{path}/residual_dense/bias"] = _numpy(
            state[f"{source}.residual_conv.bias"]
        )


def _mapped_parameters(model) -> dict[str, np.ndarray]:
    state = model.state_dict()
    payload: dict[str, np.ndarray] = {}

    _add_dense(payload, "time_dense1", state, "map_emb.0")
    _add_dense(payload, "time_dense2", state, "map_emb.2")
    _add_dense(payload, "global_cond_dense", state, "global_cond_encoder")

    for index in range(len(DOWN_DIMS)):
        _add_residual_block(
            payload,
            f"down_{index}_res1",
            state,
            f"downs.{index}.0",
        )
        _add_residual_block(
            payload,
            f"down_{index}_res2",
            state,
            f"downs.{index}.1",
        )
        if index < len(DOWN_DIMS) - 1:
            _add_conv1d(
                payload,
                f"down_{index}_ds/conv",
                state,
                f"downs.{index}.2.conv",
            )

    for index in range(2):
        _add_residual_block(payload, f"mid_res{index + 1}", state, f"mids.{index}")

    for index in range(len(DOWN_DIMS) - 1):
        _add_residual_block(
            payload,
            f"up_{index}_res1",
            state,
            f"ups.{index}.0",
        )
        _add_residual_block(
            payload,
            f"up_{index}_res2",
            state,
            f"ups.{index}.1",
        )
        _add_conv_transpose1d(
            payload,
            f"up_{index}_us/conv_transpose",
            state,
            f"ups.{index}.2.conv",
        )

    _add_conv1d(payload, "final_conv", state, "final_conv.0")
    _add_group_norm(payload, "final_norm", state, "final_conv.1")
    _add_conv1d(payload, "final_out", state, "final_conv.3")
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
        default=repo_root / "tests/fixtures/clean_diffuser_chiunet_global_v1.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    clean_root = args.clean_root.resolve()
    _validate_clean_checkout(clean_root)
    sys.path.insert(0, str(clean_root))

    import cleandiffuser
    import torch
    from cleandiffuser.nn_diffusion import ChiUNet1d

    clean_package = Path(cleandiffuser.__file__).resolve().parent
    if not clean_package.is_relative_to(clean_root):
        raise RuntimeError(
            f"Loaded CleanDiffuser from unexpected path {clean_package}."
        )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(MODEL_SEED)
    model = ChiUNet1d(
        act_dim=ACTION_DIM,
        obs_dim=OBS_DIM,
        To=OBS_STEPS,
        model_dim=DOWN_DIMS[0],
        emb_dim=EMBED_DIM,
        kernel_size=KERNEL_SIZE,
        cond_predict_scale=True,
        obs_as_global_cond=True,
        dim_mult=[1, 2, 2],
        timestep_emb_type="positional",
    ).cpu()
    model.eval()

    rng = np.random.default_rng(INPUT_SEED)
    actions_np = rng.standard_normal(
        (BATCH_SIZE, HORIZON, ACTION_DIM), dtype=np.float32
    )
    timesteps_np = rng.uniform(0.0, 99.0, size=(BATCH_SIZE,)).astype(np.float32)
    condition_np = rng.standard_normal(
        (BATCH_SIZE, OBS_STEPS, OBS_DIM), dtype=np.float32
    )
    cotangent_np = rng.standard_normal(
        (BATCH_SIZE, HORIZON, ACTION_DIM), dtype=np.float32
    )

    actions = torch.tensor(actions_np, requires_grad=True)
    timesteps = torch.tensor(timesteps_np, requires_grad=True)
    condition = torch.tensor(condition_np, requires_grad=True)
    cotangent = torch.tensor(cotangent_np)
    output = model(actions, timesteps, condition)
    action_vjp, timestep_vjp, condition_vjp = torch.autograd.grad(
        (output * cotangent).sum(),
        (actions, timesteps, condition),
    )

    parameters = _mapped_parameters(model)
    torch_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    mapped_parameter_count = sum(array.size for array in parameters.values())
    if mapped_parameter_count != torch_parameter_count:
        raise RuntimeError(
            "Mapped parameter count does not match Torch: "
            f"{mapped_parameter_count} != {torch_parameter_count}."
        )

    payload: dict[str, np.ndarray] = {
        "fixture_version": np.asarray(FIXTURE_VERSION, dtype=np.int32),
        "source_commit": np.asarray(SOURCE_COMMIT),
        "source_package": np.asarray(str(clean_package)),
        "torch_version": np.asarray(torch.__version__),
        "exporter_sha256": np.asarray(
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
        "model_seed": np.asarray(MODEL_SEED, dtype=np.int64),
        "input_seed": np.asarray(INPUT_SEED, dtype=np.int64),
        "action_dim": np.asarray(ACTION_DIM, dtype=np.int32),
        "obs_dim": np.asarray(OBS_DIM, dtype=np.int32),
        "obs_steps": np.asarray(OBS_STEPS, dtype=np.int32),
        "horizon": np.asarray(HORIZON, dtype=np.int32),
        "embed_dim": np.asarray(EMBED_DIM, dtype=np.int32),
        "down_dims": np.asarray(DOWN_DIMS, dtype=np.int32),
        "kernel_size": np.asarray(KERNEL_SIZE, dtype=np.int32),
        "n_groups": np.asarray(N_GROUPS, dtype=np.int32),
        "cond_predict_scale": np.asarray(True),
        "parameter_count": np.asarray(torch_parameter_count, dtype=np.int64),
        "actions": actions_np,
        "timesteps": timesteps_np,
        "features": condition_np.reshape(BATCH_SIZE, -1),
        "cotangent": cotangent_np,
        "expected_output": _numpy(output),
        "expected_action_vjp": _numpy(action_vjp),
        "expected_timestep_vjp": _numpy(timestep_vjp),
        "expected_feature_vjp": _numpy(condition_vjp).reshape(BATCH_SIZE, -1),
        **parameters,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        np.savez_compressed(handle, **payload)
    temporary_path.replace(output_path)
    print(
        f"Wrote {output_path} ({output_path.stat().st_size} bytes, "
        f"{torch_parameter_count} parameters)."
    )


if __name__ == "__main__":
    main()
