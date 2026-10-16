"""Throughput of HuggingFace LeRobot's PyTorch SAC configured as RLPD.

Runs in a LeRobot virtualenv (Python >= 3.12), NOT the RoboBase one.  Uses
LeRobot's own ``GaussianActorPolicy``, ``SACAlgorithm``, ``RLTrainer``,
``OnlineOfflineMixer`` and ``ReplayBuffer`` unmodified, in the synchronous
RLPD loop (one environment step, then one ``update`` = ``utd_ratio`` critic
steps + actor + temperature), with the RLPD AntMaze settings:

    num_critics=10, num_subsample_critics=1, utd_ratio=20, 3x256 critic and
    actor MLPs, use_backup_entropy=False, batch 256 at online_ratio 0.5,
    rewards - 1 (stored that way; LeRobot has no reward transform).

The environment is the exact RoboBase-wrapped AntMaze served by
``scripts/rlpd_env_server.py`` from the RoboBase venv; offline rows come from
``scripts/export_rlpd_offline_rows.py``.  LeRobot-specific defaults that differ
from RLPD (SiLU, LayerNorm also in the actor, shared Linear+LayerNorm+Tanh
state encoder, actor loss on the ensemble min, summed critic loss, grad clip
40) are left as shipped: this measures LeRobot's implementation, not ours.

Setup (LeRobot commit 146d8474932c8b566717eb2738ba459337ab71c4, Python 3.12):

    git clone https://github.com/huggingface/lerobot
    git -C lerobot checkout 146d8474932c8b566717eb2738ba459337ab71c4
    uv venv --python 3.12 lerobot-venv
    uv pip install --python lerobot-venv/bin/python --torch-backend cu128 \
        "torch==2.9.0" -e "./lerobot[dataset]"

Run (the exporter and the env server use the RoboBase venv):

    python scripts/export_rlpd_offline_rows.py rows.npy launch=rlpd_d4rl_antmaze
    lerobot-venv/bin/python scripts/benchmark_lerobot_rlpd.py \
        --offline-rows rows.npy --env-socket /tmp/rlpd_env.sock \
        --env-server-cmd "python scripts/rlpd_env_server.py /tmp/rlpd_env.sock \
            launch=rlpd_d4rl_antmaze" \
        --storage-device cuda --train-steps 1000 --out lerobot.json
"""

import argparse
import json
import subprocess
import sys
import time
from multiprocessing.connection import Client

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    ActorNetworkConfig,
    CriticNetworkConfig,
    GaussianActorConfig,
)
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.rl.algorithms.factory import make_algorithm
from lerobot.rl.algorithms.sac.configuration_sac import SACAlgorithmConfig
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer
from lerobot.rl.trainer import RLTrainer

STATE = "observation.state"


def fill_offline_buffer(
    buffer: ReplayBuffer, rows: np.ndarray, obs_dim: int, act_dim: int
):
    """Bulk equivalent of ``buffer.add`` for every offline row."""

    n = rows.shape[0]
    r0 = obs_dim + act_dim
    to = lambda a: torch.as_tensor(np.ascontiguousarray(a))  # noqa: E731
    buffer._initialize_storage(
        state={STATE: to(rows[:1, :obs_dim])}, action=to(rows[:1, obs_dim:r0])
    )
    dev = buffer.storage_device
    buffer.states[STATE][:n] = to(rows[:, :obs_dim]).to(dev)
    buffer.next_states[STATE][:n] = to(rows[:, r0 + 2 :]).to(dev)
    buffer.actions[:n] = to(rows[:, obs_dim:r0]).to(dev)
    buffer.rewards[:n] = to(rows[:, r0] - 1.0).to(dev)  # RLPD AntMaze: r - 1
    buffer.dones[:n] = to(rows[:, r0 + 1] == 0.0).to(dev)
    buffer.truncateds[:n] = False
    buffer.size = n
    buffer.position = n % buffer.capacity


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--offline-rows", required=True)
    p.add_argument("--env-socket", required=True)
    p.add_argument(
        "--env-server-cmd", required=True, help="shell command starting the env server"
    )
    p.add_argument("--storage-device", default="cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--start-training", type=int, default=5000)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--warmup-updates", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    server = subprocess.Popen(
        args.env_server_cmd, shell=True, stdout=subprocess.PIPE, text=True
    )
    assert server.stdout.readline().strip() == "ready"
    env = Client(args.env_socket, family="AF_UNIX")
    env.send(("spec", None))
    spec = env.recv()
    obs_dim, act_dim = spec["obs_dim"], spec["action_dim"]

    policy_cfg = GaussianActorConfig(
        input_features={STATE: PolicyFeature(type=FeatureType.STATE, shape=(obs_dim,))},
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(act_dim,))
        },
        device="cuda",
        storage_device=args.storage_device,
        push_to_hub=False,
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[256, 256, 256]),
    )
    algo_cfg = SACAlgorithmConfig.from_policy_config(policy_cfg)
    algo_cfg.num_critics = 10
    algo_cfg.num_subsample_critics = 1
    algo_cfg.utd_ratio = 20
    algo_cfg.use_backup_entropy = False
    algo_cfg.critic_network_kwargs = CriticNetworkConfig(hidden_dims=[256, 256, 256])
    algo_cfg.use_torch_compile = args.compile
    policy = GaussianActorPolicy(policy_cfg)
    policy.train()
    algorithm = make_algorithm(cfg=algo_cfg, policy=policy)

    total_steps = args.start_training + args.train_steps
    online = ReplayBuffer(
        total_steps,
        device="cuda",
        state_keys=[STATE],
        storage_device=args.storage_device,
    )
    rows = np.load(args.offline_rows, mmap_mode="r")
    offline = ReplayBuffer(
        rows.shape[0],
        device="cuda",
        state_keys=[STATE],
        storage_device=args.storage_device,
    )
    fill_offline_buffer(offline, np.asarray(rows), obs_dim, act_dim)
    mixer = OnlineOfflineMixer(online, offline, online_ratio=0.5)
    trainer = RLTrainer(algorithm=algorithm, data_mixer=mixer, batch_size=256)

    env.send(("reset", args.seed))
    obs = env.recv()
    timing = {"act": 0.0, "env": 0.0, "add": 0.0, "update": 0.0}
    measured, t_measure = 0, None
    returns, ep_ret = [], 0.0
    for step in range(total_steps):
        t0 = time.perf_counter()
        if step < args.start_training:
            action = rng.uniform(-1.0, 1.0, act_dim).astype(np.float32)
        else:
            action = (
                policy.select_action(
                    {STATE: torch.as_tensor(obs, device="cuda")[None]}
                )[0]
                .cpu()
                .numpy()
            )
        t1 = time.perf_counter()
        env.send(("step", action))
        next_obs, reward, terminated, truncated, _ = env.recv()
        t2 = time.perf_counter()
        online.add(
            state={STATE: torch.as_tensor(obs)[None]},
            action=torch.as_tensor(action)[None],
            reward=reward - 1.0,  # RLPD AntMaze: r - 1
            next_state={STATE: torch.as_tensor(next_obs)[None]},
            done=terminated,
            truncated=truncated,
        )
        t3 = time.perf_counter()
        if step >= args.start_training:
            trainer.training_step()
        t4 = time.perf_counter()
        ep_ret += reward
        if terminated or truncated:
            returns.append(ep_ret)
            ep_ret = 0.0
            env.send(("reset", None))
            next_obs = env.recv()
        obs = next_obs
        updates_done = step - args.start_training + 1
        if updates_done == args.warmup_updates:
            torch.cuda.synchronize()
            t_measure, measured = time.perf_counter(), 0
            timing = dict.fromkeys(timing, 0.0)
        elif updates_done > args.warmup_updates:
            measured += 1
            for key, dt in zip(timing, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                timing[key] += dt
        if updates_done > 0 and updates_done % 500 == 0:
            rate = measured / (time.perf_counter() - t_measure)
            print(f"step {step + 1}: {rate:.1f} steps/s", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_measure
    result = {
        "implementation": "lerobot SACAlgorithm (PyTorch)",
        "torch": torch.__version__,
        "storage_device": args.storage_device,
        "torch_compile": args.compile,
        "measured_env_steps": measured,
        "seconds": elapsed,
        "env_steps_per_second": measured / elapsed,
        "ms_per_step": {k: 1e3 * v / measured for k, v in timing.items()},
        "episodes": len(returns),
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    env.send(("close", None))
    server.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main())
