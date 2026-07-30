# Official Legato BiGym Boundary

This benchmark imports the official pure-JAX implementation from
[`lyfeng001/Legato-kinetix`](https://github.com/lyfeng001/Legato-kinetix) at
commit `d302701268aa3a50ec7f07189cc3af3b31014f63` (MIT license). The Kinetix
submodule is pinned at `cf7453ea103fa0b77348af1a39f689c658161613`.

The official action core and loss are not copied or edited. The files here only
provide the boundaries absent from the Kinetix repository:

- `legato_upstream.py`: fail-closed revision check and direct module loader.
- `legato_data.py`: episode-safe feature/action windows and RoboBase-compatible
  BiGym action normalization.
- `legato_features.py`: frozen visual features from a restored JAX FM encoder.
- `legato_export_features.py`: checkpoint-to-H8 dataset export through the
  normal BiGym lazy replay observation/action pipeline.
- `legato_adapter.py`: one rollout protocol for vanilla, RTC, and Legato.
- `legato_train.py`: minimal official-loss training and checkpoint output.
- `legato_eval.py`: Gym episode evaluation and a `Workspace.eval` agent bridge.

## Important Scope

The public code is a Kinetix state-observation simulation, not the real-robot
VLA stack used for the paper's main table. FlipCutlery cannot be evaluated from
the official proprio-only observation because object state is unavailable.
For BiGym, use `FrozenFMVisualFeatures` with the same restored FM encoder for
all three modes. The encoder is frozen; only the official MLP-Mixer action core
is trained.

The public `model_legato.py` uses a plus sign in its velocity target, while the
RSS/arXiv v2 algorithm uses a minus sign. This adapter intentionally retains the
public code exactly. Compare it against the repository's existing
`target_mode=paper_minus` implementation as a separate ablation.

## Protocol

The official simulation defaults are `H=8`, five Euler steps, batch size 512,
32 epochs, 1,000 linear warmup steps to a `3e-4` constant learning rate, and
randomized hard-prefix length `0..4` with exponential sampling. Incomplete
batches are dropped exactly as in the upstream trainer.
RTC is inference-only and uses the vanilla checkpoint. Legato requires its own
checkpoint because its input projection has one extra schedule channel.
The exact-public-recipe Legato run starts from random initialization. On the
43 successful FlipCutlery demonstrations, H8 produces 9,739 windows and only
19 complete batches per epoch. The 32-epoch public recipe therefore performs
608 optimizer updates and does not finish the 1,000-step learning-rate warmup.
This is a direct public-code transfer test, not a reproduction of the paper's
main comparison from a strong pretrained checkpoint. The optional
`--warm-start` path inserts the schedule channel with a zero kernel row and is
retained only as an explicitly labeled engineering ablation.

Export the aligned H8 feature dataset from the saved FM baseline. This keeps
the baseline checkpoint at its native `action_sequence=20`; only the first
eight valid actions are exported for the official core.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m \
  benchmarks.official_bigym.legato_export_features \
  --run-dir \
    exp_local/bigym_flip_cutlery_fm_fixed_baseline_repaired_1000e_b128_seed0_20260710 \
  --fm-snapshot \
    exp_local/bigym_flip_cutlery_fm_fixed_baseline_repaired_1000e_b128_seed0_20260710/snapshots/78000_snapshot.pkl \
  --pixel-dataset-root /home/zc1525/.bigym_reset_aligned \
  --state-dataset-root /home/zc1525/.bigym \
  --horizon 8 --batch-size 64 --gpu-id 1 \
  --work-dir exp_local/official_legato_feature_export \
  --output exp_local/official_legato/flip_cutlery_features_h8.npz
```

Training from that exported feature dataset:

```bash
.venv/bin/python -m benchmarks.official_bigym.legato_train \
  --dataset exp_local/official_legato/flip_cutlery_features_h8.npz \
  --output exp_local/official_legato/official_vanilla.pkl \
  --mode vanilla --epochs 32 --batch-size 512

.venv/bin/python -m benchmarks.official_bigym.legato_train \
  --dataset exp_local/official_legato/flip_cutlery_features_h8.npz \
  --output exp_local/official_legato/official_legato.pkl \
  --mode legato --epochs 32 --batch-size 512
```

Run the same closed-loop FlipCutlery evaluator for all modes. RTC consumes the
vanilla checkpoint; Legato consumes its own checkpoint. The Workspace retains
the FM checkpoint's native action output horizon and executes only the first
`--execute-horizon` actions from the official H8 core.

The policy core works in normalized action space. Evaluation explicitly clips
its returned actions to `[-1, 1]`, matching RoboBase's environment action
transform, before both execution and smoothness measurement. Result JSONs also
record `policy_action_clip_fraction`; this is necessary because raw official
outputs can exceed the normalized action interval. Directories whose names end
in `INVALID_unclipped_metrics_*` contain earlier success outcomes with invalid
raw-output smoothness diagnostics and must not be used for metric comparison.
The benchmark Workspace also consumes the ActionSequence wrapper's
`action_sequence_mask`, so an episode that terminates partway through a chunk
contributes only the action prefix that actually reached the environment.

Formal evaluation uses one environment and derives policy PRNG keys from the
episode seed, policy-call stage (bootstrap or prediction), and within-stage call
index. This keeps flow noise paired across methods even when their preceding
episodes have different lengths. The result JSON records this as
`policy_rng_mode=episode_seed_stage_call_index`. The CLI `--seed` is also the
first environment episode seed; later episodes use consecutive seeds.

```bash
RUN=exp_local/bigym_flip_cutlery_fm_fixed_baseline_repaired_1000e_b128_seed0_20260710
FM_SNAPSHOT=$RUN/snapshots/78000_snapshot.pkl

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m benchmarks.official_bigym.legato_eval \
  --run-dir "$RUN" --fm-snapshot "$FM_SNAPSHOT" \
  --policy-checkpoint exp_local/official_legato/official_vanilla.pkl \
  --mode vanilla --execute-horizon 4 --inference-delay 0 \
  --num-flow-steps 5 --num-eval-episodes 50 --num-eval-envs 1 \
  --work-dir exp_local/official_legato/eval_vanilla \
  --output exp_local/official_legato/eval_vanilla.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m benchmarks.official_bigym.legato_eval \
  --run-dir "$RUN" --fm-snapshot "$FM_SNAPSHOT" \
  --policy-checkpoint exp_local/official_legato/official_vanilla.pkl \
  --mode vanilla --execute-horizon 4 --inference-delay 2 \
  --num-flow-steps 5 --num-eval-episodes 50 --num-eval-envs 1 \
  --work-dir exp_local/official_legato/eval_naive_d2 \
  --output exp_local/official_legato/eval_naive_d2.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m benchmarks.official_bigym.legato_eval \
  --run-dir "$RUN" --fm-snapshot "$FM_SNAPSHOT" \
  --policy-checkpoint exp_local/official_legato/official_vanilla.pkl \
  --mode rtc --execute-horizon 4 --inference-delay 2 \
  --num-flow-steps 5 --num-eval-episodes 50 --num-eval-envs 1 \
  --work-dir exp_local/official_legato/eval_rtc_d2 \
  --output exp_local/official_legato/eval_rtc_d2.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m benchmarks.official_bigym.legato_eval \
  --run-dir "$RUN" --fm-snapshot "$FM_SNAPSHOT" \
  --policy-checkpoint exp_local/official_legato/official_legato.pkl \
  --mode legato --execute-horizon 4 --inference-delay 2 \
  --num-flow-steps 5 --num-eval-episodes 50 --num-eval-envs 1 \
  --work-dir exp_local/official_legato/eval_legato_d2 \
  --output exp_local/official_legato/eval_legato_d2.json
```

Naive, RTC, and Legato at delay 2 form the matched asynchronous comparison.
Vanilla at delay 0 is only the synchronous reference.

The exporter writes provenance to `<dataset>.npz.json`; each evaluation writes
its resolved policy settings, both checkpoint paths, the FM SHA-256, upstream
commit, scalar Workspace metrics, and one audited record per episode to
`--output`. Episode records contain the exact seed, success, return, executed
action count, and termination type. The CLI asserts that seeds are consecutive
and that their aggregate values reproduce the Workspace metrics.

This Legato boundary inherits the saved RoboBase FM run's 12,500-environment-
step episode limit and uses all three cameras at 20 Hz. The official A2A/FM-UNet
boundary uses one camera and a 500-step cap, so success rates across the two
benchmark families are descriptive only. Use the delay-2 naive/RTC/Legato
group for the matched Legato conclusion. A separate matched-cap sweep can use
`--episode-limit-steps 500`; omitting the flag preserves the saved FM protocol.
The result records whether the limit came from `saved_config` or `cli_override`.

Legacy saved configs with `lang_feature_source=clip` are mapped to the current
pure-JAX `tokens` source by both commands. To reproduce a particular saved
language vector instead, pass the same `--lang-feature-path` to export and eval.

## Verification

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  tests/unit/test_official_bigym_legato.py
```

The test covers the pinned upstream checkout, aligned feature windows, all
three modes, delay-state semantics, frozen visual features, one optimizer step,
checkpoint restore, and a one-episode evaluation path.
