"""Export the packed RLPD offline dataset (as RLPDWorkspace builds it) to .npy.

Rows are ``[obs | action | reward | 1 - terminal | next_obs]`` (float32),
produced by the env factory's demo wrappers exactly as in training.

    python scripts/export_rlpd_offline_rows.py out.npy launch=rlpd_d4rl_antmaze
"""

import sys

import numpy as np
from hydra import compose, initialize_config_dir

ROOT = __file__.rsplit("/scripts/", 1)[0]


def main(out: str, overrides: list[str]) -> None:
    with initialize_config_dir(f"{ROOT}/robobase/cfgs", version_base=None):
        cfg = compose("robobase_config", overrides=overrides)
    from robobase.workspace import _create_default_envs
    from robobase.workspace_rlpd import FlatTransitionRecorder

    factory = _create_default_envs(cfg)
    factory.collect_or_fetch_demos(cfg, cfg.demos)
    factory.make_eval_env(cfg)  # sets the raw spaces used by the demo wrappers
    factory.post_collect_or_fetch_demos(cfg)
    recorder = FlatTransitionRecorder()
    factory.load_demos_into_replay(cfg, recorder)
    rows = recorder.rows()
    np.save(out, rows)
    print(f"saved {rows.shape} rows from {recorder.num_episodes} episodes to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
