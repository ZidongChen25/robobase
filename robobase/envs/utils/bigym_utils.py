from collections.abc import Sequence

import numpy as np

from bigym.envs.reach_target import ReachTarget, ReachTargetDual, ReachTargetSingle
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.cupboards import (
    CupboardsOpenAll,
    CupboardsCloseAll,
    WallCupboardOpen,
    WallCupboardClose,
    DrawerTopOpen,
    DrawerTopClose,
    DrawersAllOpen,
    DrawersAllClose,
)
from bigym.envs.dishwasher import (
    DishwasherOpen,
    DishwasherClose,
    DishwasherOpenTrays,
    DishwasherCloseTrays,
)
from bigym.envs.dishwasher_cups import (
    DishwasherLoadCups,
    DishwasherUnloadCups,
    DishwasherUnloadCupsLong,
)
from bigym.envs.dishwasher_cutlery import (
    DishwasherLoadCutlery,
    DishwasherUnloadCutlery,
    DishwasherUnloadCutleryLong,
)
from bigym.envs.dishwasher_plates import (
    DishwasherLoadPlates,
    DishwasherUnloadPlates,
    DishwasherUnloadPlatesLong,
)
from bigym.envs.pick_and_place import (
    PutCups,
    TakeCups,
    PickBox,
    SaucepanToHob,
    StoreKitchenware,
    ToastSandwich,
    FlipSandwich,
    RemoveSandwich,
    StoreBox,
)
from bigym.envs.manipulation import FlipCup, FlipCutlery, StackBlocks
from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper

TASK_MAP = dict(
    reach_target_single=ReachTargetSingle,  # 2000, 10, enable_all_floating_dofs=False
    reach_target_multi_modal=ReachTarget,  # 3000, 10, enable_all_floating_dofs=False
    reach_target_dual=ReachTargetDual,  # 3000, 10, enable_all_floating_dofs=False
    stack_blocks=StackBlocks,  # 28500, 25
    move_plate=MovePlate,  # 3000, 10
    move_two_plates=MoveTwoPlates,  # 5500, 10
    flip_cup=FlipCup,  # 5500, 10
    flip_cutlery=FlipCutlery,  # 12500, 25
    dishwasher_open=DishwasherOpen,  # 7500, 20
    dishwasher_close=DishwasherClose,  # 7500, 20
    dishwasher_open_trays=DishwasherOpenTrays,  # 9500, 25
    dishwasher_close_trays=DishwasherCloseTrays,  # 7500, 25
    dishwasher_load_cups=DishwasherLoadCups,  # 7500, 10
    dishwasher_unload_cups=DishwasherUnloadCups,  # 10000, 25
    dishwasher_unload_cups_long=DishwasherUnloadCupsLong,  # 18000, 25
    dishwasher_load_cutlery=DishwasherLoadCutlery,  # 7000, 10
    dishwasher_unload_cutlery=DishwasherUnloadCutlery,  # 15500, 25
    dishwasher_unload_cutlery_long=DishwasherUnloadCutleryLong,  # 18000, 25
    dishwasher_load_plates=DishwasherLoadPlates,  # 14000, 25
    dishwasher_unload_plates=DishwasherUnloadPlates,  # 20000, 25
    dishwasher_unload_plates_long=DishwasherUnloadPlatesLong,  # 26000, 25
    drawer_top_open=DrawerTopOpen,  # 5000, 10
    drawer_top_close=DrawerTopClose,  # 3000, 10
    drawers_open_all=DrawersAllOpen,  # 12000, 25
    drawers_close_all=DrawersAllClose,  # 5000, 25
    wall_cupboard_open=WallCupboardOpen,  # 6000, 20
    wall_cupboard_close=WallCupboardClose,  # 3000, 10
    cupboards_open_all=CupboardsOpenAll,  # 22500, 25
    cupboards_close_all=CupboardsCloseAll,  # 15500, 25
    take_cups=TakeCups,  # 10500, 25
    put_cups=PutCups,  # 8500, 20
    pick_box=PickBox,  # 13500, 25
    store_box=StoreBox,  # 15000, 25
    saucepan_to_hob=SaucepanToHob,  # 11000, 25
    store_kitchenware=StoreKitchenware,  # 20000, 25
    sandwich_toast=ToastSandwich,  # 16500, 25
    sandwich_flip=FlipSandwich,  # 15500, 25
    sandwich_remove=RemoveSandwich,  # 13500, 25
    store_groceries_lower=GroceriesStoreLower,  # 32000, 25
    store_groceries_upper=GroceriesStoreUpper,  # 19000, 25
)


# H1FineManipulation exposes 30 limb qpos values, while the joint-position
# controller actuates these ten entries (in actuator order).
H1_FINE_MANIPULATION_LIMB_QPOS_INDICES = (
    0,
    1,
    2,
    3,
    12,
    13,
    14,
    15,
    16,
    25,
)


def build_actuated_qpos_state(
    proprioception: np.ndarray,
    floating_base_qpos: np.ndarray,
    gripper_qpos: np.ndarray,
    *,
    limb_qpos_indices: Sequence[int] = H1_FINE_MANIPULATION_LIMB_QPOS_INDICES,
) -> np.ndarray:
    """Reconstruct H1 ``robot.qpos_actuated`` from BiGym observations."""

    proprioception = np.asarray(proprioception)
    floating_base_qpos = np.asarray(floating_base_qpos)
    gripper_qpos = np.asarray(gripper_qpos)
    if proprioception.ndim < 1 or proprioception.shape[-1] % 2:
        raise ValueError(
            "proprioception must end in [all_qpos, all_qvel]; got "
            f"{proprioception.shape}."
        )
    if not (
        proprioception.shape[:-1]
        == floating_base_qpos.shape[:-1]
        == gripper_qpos.shape[:-1]
    ):
        raise ValueError("All proprioceptive observations must share leading axes.")
    qpos = proprioception[..., : proprioception.shape[-1] // 2]
    indices = np.asarray(tuple(limb_qpos_indices), dtype=np.int64)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= qpos.shape[-1]:
        raise ValueError(
            f"Limb qpos indices {indices.tolist()} are invalid for "
            f"{qpos.shape[-1]} joints."
        )
    return np.concatenate(
        [floating_base_qpos, qpos[..., indices], gripper_qpos], axis=-1
    ).astype(np.float32, copy=False)


def build_actuated_qpos_stats(obs_stats: dict) -> dict[str, np.ndarray]:
    """Project cached BiGym observation statistics into actuator order."""

    projected = {}
    for statistic in ("mean", "std", "min", "max"):
        values = obs_stats[statistic]
        projected[statistic] = build_actuated_qpos_state(
            np.asarray(values["proprioception"])[None],
            np.asarray(values["proprioception_floating_base"])[None],
            np.asarray(values["proprioception_grippers"])[None],
        )[0]
    return projected
