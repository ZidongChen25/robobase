"""Concurrent, immutable cache for replay-formatted demonstrations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from omegaconf import OmegaConf


CACHE_SCHEMA_VERSION = 1
CACHE_KINDS = ("all_demos", "expert_demos")


def _plain(value):
    return OmegaConf.to_container(value, resolve=True, enum_to_str=True)


def demo_cache_key(cfg, replay_signatures: dict[str, list[dict]]) -> str:
    """Hash only fields that can change replay-formatted demo contents."""

    env = dict(_plain(cfg.env))
    env.pop("eval_seed_start", None)
    replay = _plain(cfg.replay)
    replay_semantics = {
        key: replay.get(key)
        for key in (
            "action_padding",
            "action_sequence_start_offset",
            "include_tp1",
            "include_next_action",
            "auxiliary_nstep",
            "nstep_explore_truncate",
        )
        if key in replay
    }
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "env": env,
        "method_name": str(cfg.method.name),
        "demos": str(cfg.demos),
        "pixels": bool(cfg.pixels),
        "visual_observation_shape": list(cfg.visual_observation_shape),
        "frame_stack": int(cfg.frame_stack),
        # The demo env applies ObservationDelay, so the delay is baked into the
        # cached observations and caches cannot be shared across values of h.
        "obs_delay": int(cfg.get("obs_delay", 0) or 0),
        "action_repeat": int(cfg.action_repeat),
        "action_sequence": int(cfg.action_sequence),
        "execution_length": int(cfg.execution_length),
        "action_execution_start": int(cfg.get("action_execution_start", 0)),
        "use_min_max_normalization": bool(cfg.use_min_max_normalization),
        "use_standardization": bool(cfg.use_standardization),
        "min_max_margin": float(cfg.min_max_margin),
        "norm_obs": bool(cfg.norm_obs),
        "obs_norm_type": str(cfg.get("obs_norm_type", "none")),
        "replay_semantics": replay_semantics,
        "replay_signatures": replay_signatures,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


class SharedDemoReplayCache:
    """Publishes and reuses two immutable demo replay seeds.

    ``all_demos`` seeds the main replay and preserves failed demonstrations
    when the environment configuration includes them. ``expert_demos`` seeds
    the protected demo replay. Run-local files are hard links to the cache,
    so later self-imitation episodes can be appended without mutating it.
    """

    def __init__(self, root: str | Path, key: str):
        self.root = Path(root).expanduser().resolve()
        self.key = str(key)
        self.path = self.root / self.key
        self.lock_path = self.root / f"{self.key}.lock"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def is_complete(self) -> bool:
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("key") != self.key
        ):
            return False
        for kind in CACHE_KINDS:
            directory = self.path / kind
            expected = int(manifest.get("files", {}).get(kind, -1))
            if expected <= 0 or len(list(directory.glob("*.npz"))) != expected:
                return False
        return True

    @staticmethod
    def _link_tree(source: Path, destination: Path) -> int:
        destination.mkdir(parents=True, exist_ok=False)
        count = 0
        for source_file in sorted(source.glob("*.npz")):
            os.link(source_file, destination / source_file.name)
            count += 1
        if count == 0:
            raise ValueError(f"demo replay source contains no episodes: {source}")
        return count

    def publish(self, sources: dict[str, Path]) -> dict:
        if set(sources) != set(CACHE_KINDS):
            raise ValueError(f"cache sources must be exactly {CACHE_KINDS}")
        temporary = self.root / f".{self.key}.tmp.{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            counts = {
                kind: self._link_tree(Path(sources[kind]), temporary / kind)
                for kind in CACHE_KINDS
            }
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "key": self.key,
                "files": counts,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            if self.path.exists():
                shutil.rmtree(self.path)
            temporary.rename(self.path)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return manifest

    def source(self, kind: str) -> Path:
        if kind not in CACHE_KINDS:
            raise ValueError(f"unknown demo cache kind: {kind}")
        if not self.is_complete():
            raise RuntimeError(f"shared demo cache is incomplete: {self.path}")
        return self.path / kind
