"""Shared snapshot test helpers for JAX methods."""

import multiprocessing
import os
import pickle
import tempfile

import jax
import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from robobase.envs import bigym, dmc
from robobase.workspace import Workspace

dmc.UNIT_TEST = True
bigym.UNIT_TEST = True


def _copy_tree(tree):
    return jax.tree.map(lambda value: np.array(value, copy=True), tree)


def _trees_allclose(left, right) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    return left_structure == right_structure and all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def train_and_shutdown(cfg, tempdir):
    w = Workspace(cfg, work_dir=tempdir)
    w.train()
    w.shutdown()


def _train_process_helper(cfg, tempdir):
    # Initialize Workspace inside the subprocess
    workspace = Workspace(cfg, work_dir=tempdir)

    # Store the initial state_dict
    prev_state_dict = _copy_tree(workspace.agent.state_dict())

    # Perform training
    workspace.train()

    # Get the updated state_dict and save it to temp directory
    state_dict = _copy_tree(workspace.agent.state_dict())
    with open(f"{tempdir}/state_dict.pkl", "wb") as f:
        pickle.dump(state_dict, f)

    assert not _trees_allclose(state_dict, prev_state_dict)
    workspace.save_snapshot()
    workspace.shutdown()


def _load_snapshot_process_helper(cfg, tempdir):
    # Initialize Workspace inside the subprocess
    new_workspace = Workspace(cfg, work_dir=tempdir)

    # Load state_dict from previous process
    with open(f"{tempdir}/state_dict.pkl", "rb") as f:
        state_dict = pickle.load(f)

    # Check the snapshot path
    snapshot_path = os.path.join(tempdir, "snapshots", "latest_snapshot.pt")
    assert os.path.exists(snapshot_path)

    # Check whether initial parameters are different from saved parameters
    new_state_dict = new_workspace.agent.state_dict()
    assert not _trees_allclose(state_dict, new_state_dict)

    # Load snapshot
    new_workspace.load_snapshot()
    new_state_dict = new_workspace.agent.state_dict()

    # Check whether the parameters are the same after loading snapshot
    assert _trees_allclose(state_dict, new_state_dict)

    new_workspace.shutdown()


class Base:
    def test_save_load_snapshot(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + [
                "pixels=true",
                "env=dmc/acrobot_swingup",
                "save_snapshot=true",
                "snapshot_every_n=1",
            ]
            + cfg_params,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(
                target=_train_process_helper, args=(cfg, tempdir)
            )
            p.start()
            p.join()
            assert not p.exitcode

            p = multiprocessing.Process(
                target=_load_snapshot_process_helper, args=(cfg, tempdir)
            )
            p.start()
            p.join()
            assert not p.exitcode
