from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.official_bigym.a2a_train import (
    build_parser,
    build_train_command,
)
from benchmarks.official_bigym.a2a_official_entrypoint import (
    patch_latent_visualization_plotter,
)
from benchmarks.official_bigym.bigym_data import (
    H1_FINE_MANIPULATION_LIMB_QPOS_INDICES,
    build_actuated_qpos_state,
    export_official_zarr,
    stack_recent_history,
)


def test_eval_source_records_episode_seed_policy_rng():
    source = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "official_bigym"
        / "a2a_eval.py"
    ).read_text(encoding="utf-8")

    assert "rollout.torch.manual_seed(seed)" in source
    assert '"policy_seed_mode": "episode_seed"' in source


def test_skip_latent_visualization_preserves_upstream_diagnostic_path():
    original = object()
    runner_module = SimpleNamespace(plot_all_latent_visualizations=original)

    patch_latent_visualization_plotter(runner_module)

    assert runner_module.plot_all_latent_visualizations is not original
    result = runner_module.plot_all_latent_visualizations(
        None, None, epoch=1, save_dir="unused"
    )
    assert math.isnan(result["avg_tsne_distance"])


def test_stack_recent_history_supports_deque_and_edge_padding():
    values = deque([np.array([1.0]), np.array([2.0])], maxlen=8)

    result = stack_recent_history(values, 4)

    np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0, 2.0])


def test_build_actuated_qpos_state_matches_bigym_actuator_order():
    qpos = np.arange(60, dtype=np.float32).reshape(2, 30)
    proprioception = np.concatenate([qpos, -qpos], axis=-1)
    floating = np.arange(8, dtype=np.float32).reshape(2, 4) + 100
    grippers = np.arange(4, dtype=np.float32).reshape(2, 2) + 200

    result = build_actuated_qpos_state(proprioception, floating, grippers)
    expected = np.concatenate(
        [
            floating,
            qpos[:, np.asarray(H1_FINE_MANIPULATION_LIMB_QPOS_INDICES)],
            grippers,
        ],
        axis=-1,
    )

    assert result.shape == (2, 16)
    np.testing.assert_array_equal(result, expected)


def test_export_official_zarr_preserves_pre_action_alignment(tmp_path: Path):
    zarr = __import__("zarr")
    save_file = __import__("safetensors.numpy", fromlist=["save_file"]).save_file
    length = 3
    qpos = np.arange(length * 30, dtype=np.float32).reshape(length, 30)
    proprioception = np.concatenate([qpos, np.zeros_like(qpos)], axis=-1)
    floating = np.arange(length * 4, dtype=np.float32).reshape(length, 4)
    grippers = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    action = np.arange(length * 16, dtype=np.float32).reshape(length, 16)
    image = np.arange(length * 3 * 4 * 4, dtype=np.uint8).reshape(
        length, 3, 4, 4
    )
    source = tmp_path / "episode.safetensors"
    save_file(
        {
            "info_demo_action": action,
            "obs_proprioception": proprioception,
            "obs_proprioception_floating_base": floating,
            "obs_proprioception_grippers": grippers,
            "obs_rgb_head": image,
            "obs_rgb_left_wrist": image + 1,
            "obs_rgb_right_wrist": image + 2,
            "reward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
        source,
        metadata={
            "environment_data": json.dumps({"env_name": "Synthetic"}),
            "seed": "7",
        },
    )
    output = tmp_path / "official.zarr"

    manifest = export_official_zarr([source], output)

    root = zarr.open(str(output), mode="r")
    np.testing.assert_array_equal(root["data/action"][:], action)
    np.testing.assert_array_equal(root["data/head_camera"][:], image)
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [length])
    assert root["data/state"].shape == action.shape
    assert manifest["observation_timing"] == "pre_action"
    assert manifest["num_episodes"] == 1

    multicam_output = tmp_path / "official_multicam.zarr"
    multicam_manifest = export_official_zarr(
        [source],
        multicam_output,
        cameras=("head", "left_wrist", "right_wrist"),
    )
    multicam_root = zarr.open(str(multicam_output), mode="r")
    assert multicam_manifest["cameras"] == [
        "head",
        "left_wrist",
        "right_wrist",
    ]
    np.testing.assert_array_equal(
        multicam_root["data/left_wrist_camera"][:], image + 1
    )
    np.testing.assert_array_equal(
        multicam_root["data/right_wrist_camera"][:], image + 2
    )


def _resolved_command(tmp_path: Path, method: str):
    dataset = tmp_path / "dataset.zarr"
    dataset.mkdir(exist_ok=True)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(tmp_path / method),
            "--method",
            method,
            "--print-command",
        ]
    )
    return build_train_command(args)


def test_official_a2a_paper_launcher_uses_reported_protocol(tmp_path: Path):
    command, manifest = _resolved_command(tmp_path, "a2a")

    assert manifest["epochs"] == 30
    assert manifest["batch_size"] == 32
    assert manifest["history_steps"] == 8
    assert manifest["action_steps"] == 8
    assert manifest["flow_steps"] == 6
    assert "policy_config.flow_matcher.num_sampling_steps=6" in command


def test_official_fm_unet_launcher_uses_ten_flow_steps(tmp_path: Path):
    command, manifest = _resolved_command(tmp_path, "fm_unet")

    assert manifest["method"] == "fm_unet"
    assert manifest["epochs"] == 30
    assert manifest["flow_steps"] == 10
    assert "policy_config.num_inference_steps=10" in command
