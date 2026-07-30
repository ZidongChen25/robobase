"""Frozen protocol facts for the public A2A RoboVerse reproduction.

The paper names five semantic tasks, but its public repository only exposes
unambiguous experiment identifiers for the first three. The paper also does
not state the per-task simulator backend; ``simulator`` below is the pinned
public-launcher reference. The two LIBERO entries are deliberately marked as
proxies, never as exact paper reproductions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


PAPER_URL = "https://arxiv.org/html/2602.07322"
PAPER_REPOSITORY = "https://github.com/JIAjindou/A2A_Flow_Matching"
PAPER_SOURCE_COMMIT = "596f6220f87734c39dd1e7598bda05b83690a3f7"
PAPER_DATA_REVISION = "1133c84a9d5624b7670a75d4043992c57d09b5cd"
DEFAULT_PAPER_CHECKOUT = "/home/zc1525/.local/share/a2a-roboverse-paper/source"

PAPER_DEMONSTRATIONS = 100
PAPER_EPOCHS = 30
PAPER_EVAL_EPISODES = 50
PAPER_MAX_EVAL_STEPS = 300
PAPER_SEED = 42
PAPER_BATCH_SIZE = 32
PAPER_HORIZON = 16
PAPER_OBSERVATION_STEPS = 8
PAPER_ACTION_STEPS = 8
PAPER_ACTION_DIM = 9
PAPER_MAX_TRAIN_STEPS_PER_EPOCH = 250
PAPER_IMAGE_SIZE = 256

# Public artifacts are insufficient to prove identity with the paper runs.
GLOBAL_EXACT_PROTOCOL_BLOCKERS = (
    "paper checkpoints and training Zarrs are unpublished",
    "paper evaluation states, rollout count, backend, and checkpoint selection "
    "are not fully specified",
)


@dataclass(frozen=True)
class TaskProtocol:
    """One row of the controlled five-task comparison."""

    key: str
    paper_name: str
    benchmark: str
    official_task_name: str
    simulator: Literal["isaacsim", "mujoco"]
    mapping_status: Literal["exact", "proxy_blocked"]
    paper_a2a_success_pct: int
    paper_fm_unet_success_pct: int
    public_unique_trajectories: int
    trajectory_relpath: str
    trajectory_sha256: str
    mapping_note: str

    @property
    def is_exact(self) -> bool:
        return self.mapping_status == "exact"


PAPER_TASKS: dict[str, TaskProtocol] = {
    "close_box": TaskProtocol(
        key="close_box",
        paper_name="Close Box",
        benchmark="RLBench",
        official_task_name="close_box",
        simulator="isaacsim",
        mapping_status="exact",
        paper_a2a_success_pct=92,
        paper_fm_unet_success_pct=82,
        public_unique_trajectories=100,
        trajectory_relpath="roboverse_data/trajs/rlbench/close_box/v2/franka_v2.pkl.gz",
        trajectory_sha256="5295860b38e64d859fd035802f6558615f05663d02b072c1c97d1a812944b55e",
        mapping_note="The public task registry and paper identify the same task.",
    ),
    "pick_cube": TaskProtocol(
        key="pick_cube",
        paper_name="Pick Cube",
        benchmark="ManiSkill",
        official_task_name="pick_cube",
        simulator="isaacsim",
        mapping_status="exact",
        paper_a2a_success_pct=92,
        paper_fm_unet_success_pct=70,
        public_unique_trajectories=1000,
        trajectory_relpath="roboverse_data/trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz",
        trajectory_sha256="edcc2d469b8fb5ea538a1bdf13b45e6c8e9f7278b5b642f5cb2f62ff849cbe63",
        mapping_note="The public task registry and paper identify the same task.",
    ),
    "stack_cube": TaskProtocol(
        key="stack_cube",
        paper_name="Stack Cube",
        benchmark="ManiSkill",
        official_task_name="stack_cube",
        simulator="isaacsim",
        mapping_status="exact",
        paper_a2a_success_pct=86,
        paper_fm_unet_success_pct=28,
        public_unique_trajectories=1000,
        trajectory_relpath="roboverse_data/trajs/maniskill/stack_cube/v2/franka_v2.pkl.gz",
        trajectory_sha256="b8c3f59fc418dc69f2aba8f2bd601211a22cccfd67d862ecb7eb762fa86acda8",
        mapping_note="The public task registry and paper identify the same task.",
    ),
    "open_drawer": TaskProtocol(
        key="open_drawer",
        paper_name="Open Drawer",
        benchmark="LIBERO",
        official_task_name="libero_90.kitchen_scene1_open_bottom_drawer",
        simulator="mujoco",
        mapping_status="proxy_blocked",
        paper_a2a_success_pct=92,
        paper_fm_unet_success_pct=34,
        public_unique_trajectories=50,
        trajectory_relpath=(
            "roboverse_data/trajs/libero90/"
            "libero_90_kitchen_scene1_open_the_bottom_drawer_of_the_cabinet_traj_v2.pkl"
        ),
        trajectory_sha256="3d34fe6059baba386b9e3484b33f236506805d8a8597e2a8707dc7f36fae4158",
        mapping_note=(
            "Inferred semantic proxy. The paper does not publish its exact CLI task ID, "
            "and the public trajectory artifact has only 50 unique demonstrations."
        ),
    ),
    "pick_place_bowl": TaskProtocol(
        key="pick_place_bowl",
        paper_name="Pick-Place Bowl",
        benchmark="LIBERO",
        official_task_name="libero_90.kitchen_scene1_open_drawer_put_bowl",
        simulator="mujoco",
        mapping_status="proxy_blocked",
        paper_a2a_success_pct=90,
        paper_fm_unet_success_pct=68,
        public_unique_trajectories=50,
        trajectory_relpath=(
            "roboverse_data/trajs/libero90/"
            "libero_90_kitchen_scene1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_traj_v2.pkl"
        ),
        trajectory_sha256="d567d141b3a5f239e85c03ea451db34917e129fb1b2b6747b23aacbc0663641b",
        mapping_note=(
            "Inferred semantic proxy. The paper does not publish its exact CLI task ID, "
            "and the public trajectory artifact has only 50 unique demonstrations."
        ),
    ),
}


def get_task(key: str) -> TaskProtocol:
    try:
        return PAPER_TASKS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task {key!r}; choose one of {', '.join(PAPER_TASKS)}."
        ) from exc


def paper_protocol_manifest() -> dict[str, object]:
    """Return a JSON-serializable declaration of the controlled comparison."""

    return {
        "paper_url": PAPER_URL,
        "repository": PAPER_REPOSITORY,
        "source_commit": PAPER_SOURCE_COMMIT,
        "data_revision": PAPER_DATA_REVISION,
        "demonstrations": PAPER_DEMONSTRATIONS,
        "paper_epochs": PAPER_EPOCHS,
        "eval_episodes": PAPER_EVAL_EPISODES,
        "max_eval_steps": PAPER_MAX_EVAL_STEPS,
        "seed": PAPER_SEED,
        "batch_size": PAPER_BATCH_SIZE,
        "horizon": PAPER_HORIZON,
        "observation_steps": PAPER_OBSERVATION_STEPS,
        "action_steps": PAPER_ACTION_STEPS,
        "action_dim": PAPER_ACTION_DIM,
        "image_size": PAPER_IMAGE_SIZE,
        "max_train_steps_per_epoch": PAPER_MAX_TRAIN_STEPS_PER_EPOCH,
        "exact_protocol_blockers": list(GLOBAL_EXACT_PROTOCOL_BLOCKERS),
        "methods": {
            "a2a": {"flow_steps": 6, "source_variant": "initial_release_ot"},
            "a2a_current": {
                "flow_steps": 6,
                "source_variant": "current_main_conditional",
                "paper_target_comparable": False,
            },
            "fm_unet": {"flow_steps": 10},
        },
        "tasks": {key: asdict(task) for key, task in PAPER_TASKS.items()},
    }


__all__ = [
    "DEFAULT_PAPER_CHECKOUT",
    "GLOBAL_EXACT_PROTOCOL_BLOCKERS",
    "PAPER_ACTION_DIM",
    "PAPER_ACTION_STEPS",
    "PAPER_BATCH_SIZE",
    "PAPER_DEMONSTRATIONS",
    "PAPER_DATA_REVISION",
    "PAPER_EPOCHS",
    "PAPER_EVAL_EPISODES",
    "PAPER_HORIZON",
    "PAPER_IMAGE_SIZE",
    "PAPER_MAX_EVAL_STEPS",
    "PAPER_MAX_TRAIN_STEPS_PER_EPOCH",
    "PAPER_OBSERVATION_STEPS",
    "PAPER_SEED",
    "PAPER_SOURCE_COMMIT",
    "PAPER_TASKS",
    "TaskProtocol",
    "get_task",
    "paper_protocol_manifest",
]
