"""Compare low-dimensional dynamics predictability across replay datasets.

The metrics are normalized per task so state spaces with different dimensions
and units can be compared more fairly:

- delta_z_rmse predicts standardized one-step deltas.
- rollout_state_z_rmse compares multi-step state error in standardized state
  units, using ground-truth future actions.
- *_ratio divides the model error by a persistence/mean-delta baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULT_TASKS = {
    "adroit_pen": "exp_local/adroit_fm_transformer_202606211318_bs1024_adroit_fm_transformer/pen/replay",
    "adroit_door": "exp_local/adroit_fm_transformer_202606211318_bs1024_adroit_fm_transformer/door/replay",
    "adroit_hammer": "exp_local/adroit_fm_transformer_202606211318_bs1024_adroit_fm_transformer/hammer/replay",
    "adroit_relocate": "exp_local/adroit_fm_transformer_202606211318_bs1024_adroit_fm_transformer/relocate/replay",
    "bigym_move_plate": "exp_local/bigym_move_plate_dp_transformer_trainable_lang_ddpm_500e_gpu2_20260526_161755/replay",
    "pusht_lerobot_agent_pos": "hf://lerobot/pusht",
    "pusht_expert_zarr": "third_party_datasets/diffusion_policy/pusht/pusht_cchi_v7_replay.zarr",
}


@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray


class DynamicsMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_episodes(replay_dir: Path, max_episodes: int | None) -> list[Episode]:
    files = sorted(replay_dir.glob("*.npz"))
    if max_episodes is not None:
        files = files[:max_episodes]
    episodes: list[Episode] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "low_dim_state" not in data or "action" not in data:
                continue
            states = np.asarray(data["low_dim_state"], dtype=np.float32)
            actions = np.asarray(data["action"], dtype=np.float32)
        n = min(len(states), len(actions))
        if n < 3:
            continue
        states = states[:n]
        actions = actions[:n]
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            continue
        episodes.append(Episode(states=states, actions=actions))
    if not episodes:
        raise RuntimeError(f"No usable replay episodes in {replay_dir}")
    return episodes


def find_hf_snapshot(repo_id: str) -> Path | None:
    repo_cache = Path.home() / ".cache" / "huggingface" / "hub" / (
        "datasets--" + repo_id.replace("/", "--")
    )
    ref_path = repo_cache / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text().strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.exists():
            return snapshot
    snapshots = sorted((repo_cache / "snapshots").glob("*")) if repo_cache.exists() else []
    return snapshots[-1] if snapshots else None


def load_pusht_episodes(dataset_root: Path, max_episodes: int | None) -> list[Episode]:
    from robobase.envs.pusht import _load_lerobot_frame_table

    data = _load_lerobot_frame_table(dataset_root)
    required = {"episode_index", "observation.state", "action"}
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"PushT dataset is missing columns: {missing}")
    rows_by_episode: dict[int, list[int]] = {}
    for row, episode_index in enumerate(data["episode_index"]):
        rows_by_episode.setdefault(int(episode_index), []).append(row)
    episode_indices = sorted(rows_by_episode)
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

    episodes = []
    for episode_index in episode_indices:
        rows = rows_by_episode[episode_index]
        states = np.asarray(
            [data["observation.state"][row] for row in rows],
            dtype=np.float32,
        )
        actions = np.asarray([data["action"][row] for row in rows], dtype=np.float32)
        if len(states) < 3:
            continue
        episodes.append(Episode(states=states, actions=actions))
    if not episodes:
        raise RuntimeError(f"No usable PushT episodes in {dataset_root}")
    return episodes


def load_zarr_episodes(dataset_root: Path, max_episodes: int | None) -> list[Episode]:
    import zarr

    root = zarr.open(str(dataset_root), mode="r")
    if "data" not in root or "meta" not in root:
        raise KeyError(f"Expected zarr ReplayBuffer layout at {dataset_root}")
    data = root["data"]
    meta = root["meta"]
    if "state" not in data or "action" not in data or "episode_ends" not in meta:
        raise KeyError(
            f"Expected data/state, data/action, meta/episode_ends in {dataset_root}"
        )

    states_all = np.asarray(data["state"], dtype=np.float32)
    actions_all = np.asarray(data["action"], dtype=np.float32)
    episode_ends = np.asarray(meta["episode_ends"], dtype=np.int64)
    episodes = []
    start = 0
    for episode_index, end in enumerate(episode_ends):
        if max_episodes is not None and episode_index >= max_episodes:
            break
        states = states_all[start:end].copy()
        actions = actions_all[start:end].copy()
        start = int(end)
        if len(states) < 3:
            continue
        # PushT stores block angle as a wrapped scalar in state[..., 4].
        # Unwrapping keeps one-step and rollout errors from being dominated by
        # a harmless 2*pi boundary crossing.
        if states.shape[1] == 5:
            states[:, 4] = np.unwrap(states[:, 4])
        episodes.append(Episode(states=states, actions=actions))
    if not episodes:
        raise RuntimeError(f"No usable zarr episodes in {dataset_root}")
    return episodes


def load_task_episodes(source: str, max_episodes: int | None) -> tuple[list[Episode], str]:
    if source == "hf://lerobot/pusht":
        dataset_root = find_hf_snapshot("lerobot/pusht")
        if dataset_root is None:
            from huggingface_hub import snapshot_download

            dataset_root = Path(
                snapshot_download(
                    repo_id="lerobot/pusht",
                    repo_type="dataset",
                    allow_patterns=["meta/*", "data/**"],
                )
            )
        return load_pusht_episodes(dataset_root, max_episodes), str(dataset_root)

    path = Path(source)
    if path.suffix == ".zarr" or (path / ".zgroup").exists():
        return load_zarr_episodes(path, max_episodes), str(path)
    if (path / "meta" / "info.json").exists() and (path / "data").exists():
        return load_pusht_episodes(path, max_episodes), str(path)
    return load_episodes(path, max_episodes), str(path)


def split_episodes(
    episodes: list[Episode], train_frac: float, seed: int
) -> tuple[list[Episode], list[Episode]]:
    rng = random.Random(seed)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_train = max(1, int(len(idx) * train_frac))
    train = [episodes[i] for i in idx[:n_train]]
    test = [episodes[i] for i in idx[n_train:]]
    if not test:
        test = train[-1:]
        train = train[:-1] or train
    return train, test


def transitions_from_episodes(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = []
    actions = []
    next_states = []
    for ep in episodes:
        states.append(ep.states[:-1])
        actions.append(ep.actions[:-1])
        next_states.append(ep.states[1:])
    s = np.concatenate(states, axis=0)
    a = np.concatenate(actions, axis=0)
    ns = np.concatenate(next_states, axis=0)
    return s, a, ns


def sample_rows(*arrays: np.ndarray, max_rows: int | None, seed: int) -> tuple[np.ndarray, ...]:
    if max_rows is None or len(arrays[0]) <= max_rows:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arrays[0]), size=max_rows, replace=False)
    return tuple(a[idx] for a in arrays)


def safe_std(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return std


def z_rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> tuple[DynamicsMLP, dict[str, float]]:
    torch.manual_seed(seed)
    model = DynamicsMLP(x_train.shape[1], y_train.shape[1], hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_train).float()
    x_v = torch.from_numpy(x_val).float().to(device)
    y_v = torch.from_numpy(y_val).float().to(device)
    best_state = None
    best_val = math.inf
    final_train = math.inf
    final_val = math.inf
    rng = np.random.default_rng(seed)

    for _epoch in range(epochs):
        train_losses = []
        order = rng.permutation(len(x_train))
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val = torch.mean((model(x_v) - y_v) ** 2).item()
        final_train = float(np.mean(train_losses))
        final_val = float(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_val_mse": float(best_val),
        "final_train_mse": float(final_train),
        "final_val_mse": float(final_val),
    }


def predict(model: DynamicsMLP, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def eval_rollout(
    model: DynamicsMLP,
    episodes: list[Episode],
    *,
    horizons: list[int],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    delta_mean: np.ndarray,
    delta_std: np.ndarray,
    device: str,
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for horizon in horizons:
        init_states = []
        action_windows = []
        targets = []
        for ep in episodes:
            if len(ep.states) <= horizon:
                continue
            for start in range(0, len(ep.states) - horizon):
                init_states.append(ep.states[start])
                action_windows.append(ep.actions[start : start + horizon])
                targets.append(ep.states[start + horizon])
        init_states_arr = np.asarray(init_states, dtype=np.float32)
        action_windows_arr = np.asarray(action_windows, dtype=np.float32)
        targets_arr = np.asarray(targets, dtype=np.float32)
        pred_states = init_states_arr.copy()
        mean_delta_states = init_states_arr + horizon * delta_mean
        for step in range(horizon):
            actions = action_windows_arr[:, step]
            x = np.concatenate(
                [
                    (pred_states - state_mean) / state_std,
                    (actions - action_mean) / action_std,
                ],
                axis=1,
            ).astype(np.float32)
            pred_delta_z = predict(model, x, device, batch_size=8192)
            pred_states = pred_states + pred_delta_z * delta_std + delta_mean

        model_rmse = z_rmse((pred_states - targets_arr) / state_std)
        persist_rmse = z_rmse((init_states_arr - targets_arr) / state_std)
        mean_delta_rmse = z_rmse((mean_delta_states - targets_arr) / state_std)
        results[horizon] = {
            "rollout_state_z_rmse": model_rmse,
            "persistence_state_z_rmse": persist_rmse,
            "mean_delta_state_z_rmse": mean_delta_rmse,
            "rollout_vs_persistence": model_rmse / persist_rmse if persist_rmse > 0 else math.nan,
            "rollout_vs_mean_delta": model_rmse / mean_delta_rmse if mean_delta_rmse > 0 else math.nan,
            "windows": int(len(init_states_arr)),
        }
    return results


def analyze_task(name: str, source: str, args: argparse.Namespace) -> dict[str, object]:
    episodes, resolved_source = load_task_episodes(source, args.max_episodes)
    train_eps, test_eps = split_episodes(episodes, args.train_frac, args.seed)
    s_train, a_train, ns_train = transitions_from_episodes(train_eps)
    s_test, a_test, ns_test = transitions_from_episodes(test_eps)
    s_train, a_train, ns_train = sample_rows(
        s_train, a_train, ns_train, max_rows=args.max_train_transitions, seed=args.seed
    )
    s_test, a_test, ns_test = sample_rows(
        s_test, a_test, ns_test, max_rows=args.max_test_transitions, seed=args.seed + 1
    )

    d_train = ns_train - s_train
    d_test = ns_test - s_test
    state_mean = s_train.mean(axis=0).astype(np.float32)
    state_std = safe_std(s_train)
    action_mean = a_train.mean(axis=0).astype(np.float32)
    action_std = safe_std(a_train)
    delta_mean = d_train.mean(axis=0).astype(np.float32)
    delta_std = safe_std(d_train)

    x_train = np.concatenate(
        [(s_train - state_mean) / state_std, (a_train - action_mean) / action_std],
        axis=1,
    ).astype(np.float32)
    y_train = ((d_train - delta_mean) / delta_std).astype(np.float32)
    x_test = np.concatenate(
        [(s_test - state_mean) / state_std, (a_test - action_mean) / action_std],
        axis=1,
    ).astype(np.float32)
    y_test = ((d_test - delta_mean) / delta_std).astype(np.float32)

    model, train_info = train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    pred_z = predict(model, x_test, args.device, args.batch_size)
    model_delta_z_rmse = z_rmse(pred_z - y_test)
    mean_delta_delta_z_rmse = z_rmse(y_test)
    zero_delta_z = (np.zeros_like(d_test) - delta_mean) / delta_std
    persistence_delta_z_rmse = z_rmse(zero_delta_z - y_test)

    rollout = eval_rollout(
        model,
        test_eps[: args.max_rollout_episodes],
        horizons=args.horizons,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        device=args.device,
    )

    return {
        "task": name,
        "replay_dir": resolved_source,
        "episodes_total": len(episodes),
        "episodes_train": len(train_eps),
        "episodes_test": len(test_eps),
        "state_dim": int(s_train.shape[1]),
        "action_dim": int(a_train.shape[1]),
        "train_transitions": int(len(s_train)),
        "test_transitions": int(len(s_test)),
        "delta_z_rmse": model_delta_z_rmse,
        "mean_delta_delta_z_rmse": mean_delta_delta_z_rmse,
        "persistence_delta_z_rmse": persistence_delta_z_rmse,
        "delta_vs_mean_delta": model_delta_z_rmse / mean_delta_delta_z_rmse,
        "delta_vs_persistence": model_delta_z_rmse / persistence_delta_z_rmse,
        **train_info,
        "rollout": rollout,
    }


def write_outputs(results: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    rows = []
    for result in results:
        base = {
            "task": result["task"],
            "state_dim": result["state_dim"],
            "action_dim": result["action_dim"],
            "episodes_total": result["episodes_total"],
            "train_transitions": result["train_transitions"],
            "test_transitions": result["test_transitions"],
            "delta_z_rmse": result["delta_z_rmse"],
            "delta_vs_mean_delta": result["delta_vs_mean_delta"],
            "delta_vs_persistence": result["delta_vs_persistence"],
            "best_val_mse": result["best_val_mse"],
            "final_train_mse": result["final_train_mse"],
            "final_val_mse": result["final_val_mse"],
        }
        for horizon, metrics in result["rollout"].items():
            row = dict(base)
            row["horizon"] = horizon
            row.update(metrics)
            rows.append(row)
    fieldnames = list(rows[0].keys())
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", default=[], help="name=path replay dir")
    parser.add_argument("--no-default-tasks", action="store_true")
    parser.add_argument("--out-dir", default="exp_local/dynamics_predictability_probe")
    parser.add_argument("--max-episodes", type=int, default=1200)
    parser.add_argument("--max-train-transitions", type=int, default=120000)
    parser.add_argument("--max-test-transitions", type=int, default=30000)
    parser.add_argument("--max-rollout-episodes", type=int, default=80)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {} if args.no_default_tasks else dict(DEFAULT_TASKS)
    for spec in args.task:
        name, path = spec.split("=", 1)
        tasks[name] = path

    results = []
    for name, source in tasks.items():
        if not str(source).startswith("hf://") and not Path(source).exists():
            print(f"SKIP {name}: missing {source}")
            continue
        print(f"RUN {name}: {source}", flush=True)
        result = analyze_task(name, source, args)
        results.append(result)
        print(
            f"DONE {name}: delta_z_rmse={result['delta_z_rmse']:.4f} "
            f"delta_vs_persistence={result['delta_vs_persistence']:.4f}",
            flush=True,
        )
    if not results:
        raise RuntimeError("No tasks completed")
    write_outputs(results, Path(args.out_dir))
    print(f"Wrote {args.out_dir}/results.csv and results.json")


if __name__ == "__main__":
    main()
