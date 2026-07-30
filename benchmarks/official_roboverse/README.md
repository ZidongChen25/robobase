# A2A RoboVerse controlled reproduction

This directory launches the unmodified official PyTorch trainer as an isolated
oracle. It does not add PyTorch to RoboBase's production JAX dependency path.

The primary oracle source is pinned to the initial public worktree:

```text
/home/zc1525/.local/share/a2a-roboverse-paper/source
596f6220f87734c39dd1e7598bda05b83690a3f7
```

There is no tagged paper release or published checkpoint. The paper v2 still
describes optimal transport, and the initial commit uses
`ExactOptimalTransportConditionalFlowMatcher`; however, official `main` changed
standard A2A to `ConditionalFlowMatcher` in commit `131d493` before paper v2.
The benchmark therefore keeps two explicit, non-interchangeable variants:

- `a2a`: initial-release OT implementation;
- `a2a_current`: current-main Conditional matcher sensitivity arm.

`a2a_current` is never marked as the pinned paper protocol. Its command records
the exact Hydra target override while still using the frozen source checkout, so
the matcher change is the only training-code difference.

`preflight.py` rejects any other commit, tracked source edits, malformed Zarr
arrays, action/state dimensions other than 9, and any demonstration count other
than the declared count. This matters because the upstream converter prints a
warning and silently uses fewer demonstrations when fewer are available.

Even when the published task label, 100-demo budget, and launcher backend all
match, train/eval manifests record only
`declared_paper_controls_match=true` and keep
`exact_paper_protocol=false`. Preflight separately records
`declared_task_data_controls_match`. The authors have not published the
checkpoints, training Zarrs, run manifests, or enough evaluation metadata to
establish artifact-level identity.

The raw expert trajectories are additionally pinned to RoboVerse data revision
`1133c84a9d5624b7670a75d4043992c57d09b5cd`. Preflight verifies each task's
recorded SHA256 and rejects non-finite state/action values before training.

## Public reproduction status

| Paper task | Public task ID used here | Public-code simulator | Status | Paper A2A / FM-UNet |
| --- | --- | --- | --- | --- |
| Close Box | `close_box` | Isaac Sim | exact mapping | 92 / 82 |
| Pick Cube | `pick_cube` | Isaac Sim | exact mapping | 92 / 70 |
| Stack Cube | `stack_cube` | Isaac Sim | exact mapping | 86 / 28 |
| Open Drawer | `libero_90.kitchen_scene1_open_bottom_drawer` | MuJoCo | blocked proxy | 92 / 34 |
| Pick-Place Bowl | `libero_90.kitchen_scene1_open_drawer_put_bowl` | MuJoCo | blocked proxy | 90 / 68 |

The last two mappings are semantic inferences, not published paper task IDs.
Their public trajectory artifacts expose only 50 unique demonstrations, while
the paper comparison requires 100. Results from them must be labelled proxy
results and cannot be included in an exact five-task reproduction claim.

The paper says the experiments use RoboVerse but does not explicitly name the
backend for each row. The simulator column is inferred from the pinned public
launchers: the generic Close/Pick/Stack recipe defaults to Isaac Sim, while the
published LIBERO launcher uses MuJoCo. The legacy manifest field
`simulator_matches_paper` therefore means "matches the pinned public recipe",
not that the backend was stated in the paper.

The frozen machine-readable declaration is in `protocol.py`.

## Fixed protocol

- 100 demonstrations, batch size 32, seed 42;
- RGB 256x256, joint position observation/action dimension 9;
- horizon 16, observation history 8, prediction and execution length 8;
- A2A 6 flow steps, FM-UNet 10 flow steps;
- at most 250 train batches per epoch;
- evaluation on trajectory initializations 0 through 49, at most 300 steps.
  The paper does not state the rollout count; the initial official launcher
  uses 50 and the table's 2-point result granularity is consistent with it.

This evaluator is not a held-out generalization benchmark. For the local proxy
datasets, the fixed evaluation IDs overlap 40/50 Pick and 50/50 Stack training
source indices. Runs currently use one training seed, so paired episode tests
do not estimate training-seed variance.

Two training arms are intentionally separate:

- `fresh30`: independent 30-epoch paper-budget run; cosine LR horizon is 30.
- `long200`: uninterrupted 200-epoch run, saved every 30 epochs plus final E200; E30 and E200
  are directly comparable along this training trajectory, but long-run E30 is
  not optimizer-schedule-equivalent to `fresh30` because its LR horizon is 200.

Do not implement 200 epochs as a stock upstream resume from E30. The upstream
resume path does not restore every scheduler/EMA state needed for that claim.

## Preflight

```bash
OFFICIAL_PY=/home/zc1525/.venvs/a2a-roboverse-paper/bin/python
DATA=/absolute/path/to/close_box_100.zarr

$OFFICIAL_PY -m benchmarks.official_roboverse.preflight \
  --task close_box \
  --dataset "$DATA"
```

For an explicitly non-paper 50-demo proxy check:

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.preflight \
  --task open_drawer \
  --dataset /absolute/path/to/open_drawer_proxy_50.zarr \
  --expected-episodes 50 \
  --allow-proxy
```

The JSON output contains `exact_paper_protocol: false` for that command.
Simulator proxies are guarded the same way. For example, using MuJoCo for one
of the three Isaac Sim paper tasks requires both `--simulator mujoco` and
`--allow-proxy`; its manifest records `simulator_matches_paper: false` and
`exact_paper_protocol: false` even with 100 demonstrations.

For a proxy dataset, run the stronger content-provenance audit once after
conversion. It reads every Zarr state, action, and RGB value in bounded chunks,
decodes every raw video, compares every episode bit-for-bit, and records a
chunk-layout-independent logical SHA-256:

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.audit_proxy_data \
  --dataset /absolute/path/to/pick_cube_proxy_100.zarr \
  --raw-success-dir /absolute/path/to/pick_cube_raw/success \
  --expected-episodes 100 \
  --expected-logical-sha256 <previously-frozen-sha256> \
  --output /absolute/path/to/pick_cube_proxy_provenance.json
```

The audit also rejects duplicate logical episodes by default. The output uses
the versioned `a2a_proxy_logical_content_v1` hash specification, so the digest
does not depend on physical Zarr compression or chunk boundaries.

## Training

Use a unique output directory for every task, method, and arm. Commands execute
by default; add `--print-command` to run preflight and print the resolved
manifest without starting training.

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.train \
  --task close_box \
  --dataset "$DATA" \
  --method a2a \
  --arm fresh30 \
  --device cuda:0 \
  --output /absolute/path/to/runs/close_box/a2a/fresh30

$OFFICIAL_PY -m benchmarks.official_roboverse.train \
  --task close_box \
  --dataset "$DATA" \
  --method fm_unet \
  --arm long200 \
  --device cuda:1 \
  --output /absolute/path/to/runs/close_box/fm_unet/long200

$OFFICIAL_PY -m benchmarks.official_roboverse.train \
  --task close_box \
  --dataset "$DATA" \
  --method a2a_current \
  --arm fresh30 \
  --device cuda:2 \
  --output /absolute/path/to/runs/close_box/a2a_current/fresh30
```

Expected checkpoint paths are `checkpoints/30.ckpt` for both arms and
`checkpoints/200.ckpt` for `long200`. Every run writes `train_manifest.json`
before launching the upstream process.

The paper-release runner also computes CPU-heavy t-SNE plots after every A2A
epoch. `--skip-latent-visualization` still traverses the diagnostic validation
loader and runs the latent/flow model calls, preserving the upstream Torch RNG
path, but replaces deterministic t-SNE/PNG generation with a no-op. It does not
skip validation, action-error sampling, updates, EMA, or checkpoints. The choice
is written as `latent_visualization_mode: rng_preserving_no_plot` and should be
held fixed across A2A arms.

## Evaluation

Evaluate both `fresh30/checkpoints/30.ckpt` and
`long200/checkpoints/{30,200}.ckpt`. The native upstream evaluator executes all
eight predicted actions before replanning.

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.eval \
  --task close_box \
  --dataset "$DATA" \
  --method a2a \
  --checkpoint /absolute/path/to/runs/close_box/a2a/long200/checkpoints/200.ckpt \
  --checkpoint-epoch 200 \
  --device cuda:0 \
  --gpu-id 0 \
  --output /absolute/path/to/eval/close_box/a2a/long200_e200
```

The launcher hashes the checkpoint into `eval_manifest.json`. Upstream success
files are written below the evaluation output's `eval/` directory.

For a source-disjoint proxy evaluation, choose a contiguous range outside the
training source IDs and pass the audited dataset provenance. For the local
100-demo proxies, Pick uses IDs 125 through 174 and Stack uses IDs 100 through
149:

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.eval \
  --task pick_cube \
  --dataset /absolute/path/to/pick_cube_proxy_100.zarr \
  --dataset-provenance /absolute/path/to/pick_cube_proxy_provenance.json \
  --eval-start-index 125 \
  --method a2a \
  --checkpoint /absolute/path/to/runs/pick_cube/a2a/long200/checkpoints/200.ckpt \
  --checkpoint-epoch 200 \
  --simulator mujoco \
  --allow-proxy \
  --device cuda:0 \
  --gpu-id 0 \
  --output /absolute/path/to/eval/pick_cube/a2a/heldout_long200_e200
```

A nonzero start is rejected without provenance. The launcher requires exactly
100 unique audited training source IDs, checks that none intersects the 50
evaluation IDs, records the range and provenance-file hash in the manifest,
and validates the same evidence again before publishing the manifest. Result
aggregation treats each range as a distinct evaluation cohort and never pairs
episodes across cohorts.

### Close Box random-initialization proxy

The public Close Box bank has only 100 states, so it cannot supply 100 training
states plus 50 held-out states.

## Strict JAX A2A

`robobase.models.official_a2a.OfficialA2A` is the plug-and-play JAX port of the
initial-release A2A policy. Its trainable parameter count is exactly
`34,656,904`, matching the upstream model after excluding normalizer buffers.
The port preserves:

- random-weight torchvision-style ResNet18 with BatchNorm replaced by GroupNorm;
- ImageNet input normalization and per-timestep proprioception concatenation;
- independent three-layer history and future Conv1D encoders;
- the 512-wide, four-layer `SimpleFlowNet` and four-layer action decoder;
- exact minibatch OT resampling, all four published losses and differentiable
  six-step left-endpoint Euler integration;
- AdamW, cosine warmup, power-law EMA, H16/O8/K8 and batch size 32;
- the upstream dataloader's otherwise implicit seed 0 and 250-step epoch cap.

Training and policy code import no Torch. RoboVerse's Isaac Sim runner itself is
Torch-based, so `jax_eval_runner.py` performs only the simulator tensor boundary
conversion. The JAX checkpoint remains the source of all policy parameters and
normalization state.

```bash
.venv/bin/python -m benchmarks.official_roboverse.jax_a2a \
  --dataset /absolute/path/to/close_box_99.zarr \
  --output /absolute/path/to/jax_a2a_e30 \
  --epochs 30 --batch-size 32 --max-train-steps 250

.venv/bin/python -m benchmarks.official_roboverse.jax_eval \
  --dataset /absolute/path/to/close_box_99.zarr \
  --checkpoint /absolute/path/to/jax_a2a_e30/checkpoints/epoch_0030.msgpack \
  --trajectory /absolute/path/to/close_box_random_init_v2.pkl.gz \
  --output /absolute/path/to/jax_a2a_e30_eval --episodes 50 --gpu 0
```

Generate a deterministic evaluation-only cohort with RoboVerse `PoseRandomCfg`
pose semantics as follows:

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.random_initialization \
  --source-trajectory /absolute/path/to/close_box/franka_v2.pkl.gz \
  --dataset-provenance /absolute/path/to/dataset_provenance.json \
  --output-dir /absolute/path/to/heldout_seed20260721
```

The manifest binds the cohort to the audited training source IDs, rejects exact
and near-duplicate poses, and hashes both source and generated trajectories.
This is a controlled held-out proxy, not a claim about the paper's unpublished
evaluation-state sampler.

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.eval \
  --task close_box \
  --dataset "$DATA" \
  --dataset-provenance /absolute/path/to/dataset_provenance.json \
  --random-initialization-manifest /absolute/path/to/heldout_seed20260721/manifest.json \
  --method fm_unet \
  --checkpoint /absolute/path/to/checkpoints/30.ckpt \
  --checkpoint-epoch 30 \
  --expected-episodes 99 \
  --flow-steps 6 \
  --fm-solver euler \
  --allow-proxy \
  --output /absolute/path/to/eval/fm_e30_euler6
```

The evaluator temporarily points the upstream `CloseBoxTask` at the generated
trajectory while retaining the upstream Isaac Sim environment and policy
runner. `--flow-steps` is an explicit non-paper override for equal-step method
comparisons; prediction and execution length remain eight.

If the upstream Isaac Sim process has written all 50 episode files and
`final_stats.txt` but remains alive during shutdown, terminate that completed
process and repeat the identical command with `--finalize-existing`. This mode
does not rerun episodes; it revalidates the result, checkpoint, dataset audit,
and random-initialization hashes before atomically publishing
`eval_manifest.json`.

## Result aggregation

After all three comparison checkpoints have been evaluated for each task and
method, validate and aggregate the native evaluator artifacts:

```bash
$OFFICIAL_PY -m benchmarks.official_roboverse.results \
  --input /absolute/path/to/eval \
  --json-output /absolute/path/to/reports/roboverse_comparison.json \
  --csv-output /absolute/path/to/reports/roboverse_comparison.csv \
  --require-full-matrix
```

`--input` accepts either an evaluation root or an individual
`eval_manifest.json` and may be repeated. Omit `--require-full-matrix` to
aggregate a subset, but every included task/method still requires the complete
`fresh30/E30`, `long200/E30`, and `long200/E200` triplet.

The aggregator rejects duplicate logical runs, manifest/checkpoint/training-arm
mismatches, checkpoint hash changes, cross-arm/cross-method control mismatches,
and evaluations that do not contain exactly 50 consistent episode records. The
JSON retains per-checkpoint provenance. The CSV has one row per
task/method with the paper target, all three success counts/rates, the E30
schedule delta, the E30-to-E200 continuation delta, and the fresh30-to-E200
delta. Proxy protocols retain their simulator and proxy flags, their paper
target is marked non-comparable, and paper-target deltas are omitted.

## Required comparison

For each task with an exact public task-ID mapping and each method, report these
three rows separately:

1. `fresh30/E30`: paper-budget reproduction against the paper table.
2. `long200/E30`: early point on the long schedule.
3. `long200/E200`: effect of continued optimization on that same run.

Never average the two proxy tasks into an "exact five-task reproduction" unless
the authors publish the missing task IDs and 100-demo artifacts.
