"""Serve a RoboBase-wrapped training env over a Unix socket.

Lets a process in another virtualenv (the PyTorch/LeRobot RLPD benchmark)
step exactly the environment RLPDWorkspace trains on: same factory, wrappers
and observation/action layout.  One client; pickled request/response pairs:

    ("reset", seed)  -> obs
    ("step", action) -> (obs, reward, terminated, truncated, success)
    ("spec", None)   -> {"obs_dim": int, "action_dim": int}

Observations are flat float32 ``low_dim_state`` vectors; ``action`` is a flat
float32 vector in [-1, 1].

    python scripts/rlpd_env_server.py /tmp/env.sock launch=rlpd_d4rl_antmaze
"""

import sys
from multiprocessing.connection import Listener

import numpy as np
from hydra import compose, initialize_config_dir

ROOT = __file__.rsplit("/scripts/", 1)[0]


def main(address: str, overrides: list[str]) -> None:
    with initialize_config_dir(f"{ROOT}/robobase/cfgs", version_base=None):
        cfg = compose("robobase_config", overrides=overrides)
    from robobase.workspace import _create_default_envs
    from robobase.workspace_rlpd import use_private_tempdir

    use_private_tempdir()
    factory = _create_default_envs(cfg)
    env = factory.make_eval_env(cfg)  # train-env wrappers, single process
    shape = env.action_space.shape

    def flat(obs):
        return np.asarray(obs["low_dim_state"], np.float32).reshape(-1)

    with Listener(address, family="AF_UNIX") as listener:
        print("ready", flush=True)
        with listener.accept() as conn:
            while True:
                try:
                    command, payload = conn.recv()
                except EOFError:
                    return
                if command == "reset":
                    obs, _ = env.reset(seed=payload)
                    conn.send(flat(obs))
                elif command == "step":
                    action = np.asarray(payload, np.float32).reshape(shape)
                    obs, reward, terminated, truncated, info = env.step(action)
                    success = info.get("task_success")
                    conn.send(
                        (
                            flat(obs),
                            float(reward),
                            bool(terminated),
                            bool(truncated),
                            None if success is None else float(success),
                        )
                    )
                elif command == "spec":
                    obs, _ = env.reset(seed=0)
                    conn.send(
                        {"obs_dim": flat(obs).size, "action_dim": int(np.prod(shape))}
                    )
                elif command == "close":
                    return


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
