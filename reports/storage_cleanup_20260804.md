# Experiment storage cleanup — 2026-08-04

## Initial state

- Filesystem: `/dev/nvme0n1p2`, 3.57 TiB total.
- Used: 3,684,770,234,368 bytes; user-available: 35,631,865,856 bytes
  (`df` rounded this to 100% used).
- `/home/zc1525`: 2.66 TiB; `robobase_jaxflat/exp_local`: 2.54 TiB.
- Unique-inode allocation under `exp_local`: snapshots 1.44 TiB, online replay
  788 GiB, demo replay 294 GiB, checkpoints 40 GiB.

The live `cqn_trunc_arms` and `cqn_official_truncated` trees, every run newer
than 36 hours, and all `cqn_no_bc` runs were excluded.  `cqn_no_bc` needs a
separate manifest because Stage 43 checkpoint selection joins validation
curves across stages; selecting from one local CSV would be incorrect.

## Dedupe gate

SHA-256 was computed for 1,614 demo-replay NPZ files (17,144,900,967 logical
bytes) across the official-truncated runs.  There were zero duplicate-content
groups, so hard-link deduplication was rejected.

## Validated cleanup

The durable manifest is `reports/storage_cleanup_20260804_cqn_flow.json`
(SHA-256 `cf6e286dda284d8498ff01fc99e809526a4c9d6ecaab3a795ed523fe149044ff`).
It admits only runs that:

1. were inactive and older than 36 hours;
2. reached the configured `num_train_frames` endpoint;
3. had at least two usable validation checkpoints outside sealed/held-out
   seeds; and
4. retained validation-best (earliest tie), raw 100k when present, and the
   final endpoint.

Applied actions:

- Removed 490 non-selected snapshots from 65 completed runs.  All retained
  checkpoint paths were verified after removal.
- Removed 116 `replay`/`demo_replay` directories from those same completed
  runs after revalidating their full inode/mtime inventory.  This preserves
  evaluation and rerun reproducibility, but those historical runs can no
  longer be resumed from their exact replay state.
- Preserved Hydra configs, train/eval CSVs, JSON summaries, logs, selected
  snapshots, 100k reporting snapshots, and endpoints.

The completion records are adjacent to the manifest as
`storage_cleanup_20260804_cqn_flow.json.snapshots.done` and
`storage_cleanup_20260804_cqn_flow.json.replay.done`.

## Result

- Final measured used bytes after cleanup and while two new runs were writing:
  3,235,792,146,432 bytes.
- Final user-available bytes: 484,609,953,792 bytes (451.33 GiB); filesystem
  usage is 87%.
- Net space recovered relative to the initial measurement: approximately
  418 GiB, including concurrent writes from the two excluded `noens` runs.

## Prevention

`replay.compression` now accepts `none` (global backward-compatible default)
or lossless `zip`.  Pixel BiGym CQN-AS and CQN-Flow launches select `zip`.
On one real 300-step, three-camera replay episode:

| Codec | Bytes | Save time | Full load time |
|---|---:|---:|---:|
| none | 19,217,203 | 0.0142 s | 0.0085 s |
| zip | 5,937,393 | 0.3457 s | 0.0485 s |

This is a 69.1% file-size reduction.  The additional save work is about
1.1 ms per environment step and occurs at episode boundaries.  Focused replay
and launch-config tests pass (6 replay tests plus the CQN-AS composition test).

Use `scripts/prune_experiment_storage.py plan` to create future manifests.
Application requires the explicit confirmation phrase and revalidates active
processes, retained checkpoints, inode/mtime metadata, and replay inventories
before performing any deletion.
