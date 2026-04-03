# Backend Restructure Plan

## Goal

Support two execution backends in the same project:

- PyTorch: preserve the current implementation and behavior.
- JAX: add a new implementation incrementally without forcing a full rewrite up front.

The right approach for this repository is **not** "replace `torch` with `jax` in-place", and it is also **not** "keep the current torch-first method/model tree and add more JAX special cases around it".

The current codebase mixes framework details into runtime orchestration, methods, models, snapshotting, packaging, and tests. The long-term fix is:

1. make the current PyTorch implementation an explicit `torch backend`
2. stop treating `method` and `model` as implicitly "the torch implementation"
3. add JAX as a parallel backend implementation behind the same runtime contract

In other words, the architecture should become:

- shared runtime and algorithm semantics at the top
- backend-specific method/model implementations underneath

The current minimal JAX BC path is useful as a proof of concept, but it is still a prototype. It proves that a backend boundary can work; it is **not** yet the final architecture.

## Current Status

This section reflects the code as it exists now.

### Implemented so far

1. Shared backend selection exists.
   - `Workspace` still owns the high-level training loop.
   - agent creation is routed through `robobase/backends/factory.py`.
   - `backend=torch` and `backend=jax` now go through the same BC dispatch shape.
   - the default JAX backend config now targets `cuda` instead of `cpu`.

2. BC has been moved to the new backend structure at the method boundary.
   - shared BC semantics now live in `robobase/method/bc.py`.
   - torch BC implementation lives in `robobase/backends/torch/method/bc.py`.
   - JAX BC implementation lives in `robobase/backends/jax/method/bc.py`.
   - the temporary `robobase/backends/bc.py` dispatch file has already been removed and inlined into `robobase/backends/factory.py`.

3. Torch-only BC inheritance has been made explicit.
   - `robobase/method/act.py` and `robobase/method/diffusion.py` now import the torch backend BC directly instead of importing `robobase.method.bc`.
   - this means `robobase/method/bc.py` is now a real shared BC layer rather than a torch alias.

4. The minimal JAX BC vertical slice is live.
   - `backend=jax` works for offline low-dimensional BC.
   - the JAX BC path uses JAX/Optax for parameter init, forward, loss, gradients, and optimizer update.
   - the path is integrated with the shared workspace and replay flow.

5. BC now has shared model specs and backend-specific model components.
   - shared BC model semantics now live in `robobase/method/bc.py`.
   - shared BC runtime/layout helpers now live in `robobase/method/bc_runtime.py`.
   - BC config no longer points at torch model `_target_` entries.
   - `actor_model`, `encoder_model`, and `view_fusion_model` are now parsed as shared BC model specs.
   - BC config now uses backend-neutral model-spec fields such as `actor_model.hidden_dims`.
   - the old BC-only assembly files under `backends/torch/models/bc.py` and `backends/jax/models/bc.py` have been removed.
   - BC model assembly now lives in the backend method implementations again:
     - `robobase/backends/torch/method/bc.py`
     - `robobase/backends/jax/method/bc.py`
   - backend `models/` directories now keep only reusable model components.

6. JAX BC now covers the same BC surface area that the current shared BC spec exposes.
   - torch and jax now both support the shared MLP behavior-cloning path for `method=bc`.
   - recurrent sequence output now works in the JAX actor path.
   - pixel BC now works in the JAX backend for the current shared BC spec:
     - `encoder_model.type=resnet`
     - `view_fusion_model.type=multicam_feature`
   - the JAX pixel encoder is now a native JAX ResNet forward path under `backends/jax/models/encoder.py`.
   - the current implementation now uses a Flax/JAX ResNet feature extractor built on top of `jax-resnet`, instead of the older hand-written low-level XLA ResNet path.
   - pretrained ResNet weights are still imported once from timm at encoder construction time, then converted into frozen JAX/Flax variables for the feature extractor.
   - the current native JAX encoder surface is intentionally narrow:
     - frozen BasicBlock ResNets
     - currently `resnet18` and `resnet34`
   - this removes the old torch forward bridge from the JAX BC runtime path while keeping BC pixel parity for the current shared spec.

7. The native JAX image encoder path is now materially faster than the previous hand-written version.
   - this was the main reason image BC had started running slower in JAX than in torch.
   - the old native JAX encoder path was a custom low-level implementation; it worked, but it was not using a mature vision stack.
   - after switching to the `jax-resnet`/Flax feature extractor path, a local encoder-only benchmark on `resnet18` with batch `(256, 2, 3, 240, 240)` now gives:
     - torch encoder: about `0.0466 sec/iter`
     - jax encoder: about `0.0358 sec/iter`
   - this is not yet the same thing as a full end-to-end training benchmark, but it confirms that the encoder bottleneck has moved in the right direction.

8. BC can train end-to-end in both backends.
   - smoke tests were run successfully for:
     - `backend=torch` with robomimic ToolHang offline BC
     - `backend=jax` with robomimic ToolHang offline BC
   - focused unit coverage also passes for backend dispatch, JAX snapshot/speed tests, recurrent JAX BC output, native JAX ResNet encoder forward, and JAX pixel/multicam BC.
   - real transport comparison commands for `backend=torch` and `backend=jax` are recorded in `test_comparison.md`.
   - `test_comparison.md` now also records image-based ToolHang BC comparison commands for both backends.
   - those commands now explicitly fix the offline BC training length (`num_pretrain_steps=200000`), enable periodic snapshots, and turn on live robosuite evaluation so W&B can show meaningful evaluation-speed metrics.
   - the preferred comparison path is now a real robomimic training run with W&B metrics, not a separate synthetic microbenchmark script.

9. Torch model ownership has started moving under `backends/torch/models`.
   - the real implementation of `fully_connected.py` now lives under `robobase/backends/torch/models/fully_connected.py`.
   - the real implementations of `encoder.py` and `fusion.py` now also live under:
     - `robobase/backends/torch/models/encoder.py`
     - `robobase/backends/torch/models/fusion.py`
   - the old top-level torch-first model paths are now compatibility aliases:
     - `robobase/models/fully_connected.py`
     - `robobase/models/encoder.py`
     - `robobase/models/fusion.py`

10. Replay iteration is no longer tied to PyTorch `DataLoader` in `Workspace`.
   - `Workspace` now builds replay iterators through `robobase/replay_buffer/iterator.py`.
   - replay sampling now originates as NumPy batches directly from the replay buffers.
   - torch training wraps those NumPy batches into torch tensors before method update.
   - JAX training now consumes NumPy batches directly instead of going through `numpy -> torch -> numpy/JAX`.
   - `DemoMergedIterator` now works with either torch tensors or NumPy arrays.
   - for `backend=jax`, `replay.num_workers` is currently normalized to `0` to avoid forking replay workers after the JAX runtime has been initialized.

11. Backend comparison run defaults have been cleaned up for local use.
   - the default `wandb.entity` in `robobase/cfgs/robobase_config.yaml` now points at `tsztungchen25-imperial-college-london`.
   - `test_comparison.md` now uses the same entity explicitly in the torch and jax transport comparison commands.

12. W&B logging is stable again in the shared logger path.
   - `robobase/logger.py` now keeps a module-level `wandb` handle instead of relying on a local import from `Logger.__init__`.
   - this fixes runtime failures when `Logger._dump()` tries to call `wandb.log(...)` during training.

13. The BC recurrent sequence head is now more closely aligned across torch and JAX.
   - the torch BC `rnn` output head in `robobase/backends/torch/models/fully_connected.py` no longer uses `output_sequence_length` as the GRU layer count.
   - it now matches the JAX BC shape more closely: a single recurrent layer unrolled over the requested action sequence with a zero initial hidden state.
   - this removes a major implementation mismatch that could distort torch-vs-jax BC quality comparisons, especially when `action_sequence` is large.

14. Torch BC actor initialization is now aligned more closely with the JAX BC path.
   - torch BC now re-initializes its actor modules with JAX-style defaults when the actor is built in `robobase/backends/torch/method/bc.py`.
   - linear and GRU weights now use Xavier-uniform style initialization and biases are zeroed, matching the family of initialization used in the JAX BC actor implementation.
   - this change is intentionally scoped to the torch BC actor path rather than changing the global torch model initialization policy for the whole repo.

15. Offline BC launch composition is now decoupled from task-specific robomimic presets.
   - the preferred robomimic BC launch shapes are now:
     - `launch=bc_state_robomimic`
     - `launch=bc_pixel_robomimic`
   - those launch files define the BC method family plus the state/pixel observation mode for the robomimic benchmark, but they do not hardcode a specific task.
   - the task is now composed separately through `env=robomimic/<task>`.
   - this means normal commands can use:
     - `launch=bc_state_robomimic env=robomimic/transport backend=...`
     - `launch=bc_pixel_robomimic env=robomimic/tool_hang backend=...`
   - `test_comparison.md` has been updated to use this launch style for both transport and ToolHang comparisons.

16. GPU selection can now be controlled from the training command/config instead of shell-only environment variables.
   - `robobase/cfgs/robobase_config.yaml` now exposes a top-level `gpu_id`.
   - `train.py` applies that requested GPU before importing `Workspace`.
   - the entrypoint now sets:
     - `CUDA_VISIBLE_DEVICES`
     - `JAX_CUDA_VISIBLE_DEVICES`
     - `MUJOCO_EGL_DEVICE_ID`
   - when `gpu_id` is set, the process is reduced to a single visible GPU and `env.render_gpu_device_id` is forced to `0` inside that visible-device namespace.
   - this means one command-line override now keeps training and robosuite rendering on the same physical GPU.

17. Pure offline BC no longer eagerly creates live robomimic envs just to initialize the workspace.
   - `EnvFactory` now has `get_spaces(cfg)` so the workspace can build the agent and replay buffers without first spinning up a live evaluation env.
   - `RobomimicEnvFactory.get_spaces()` resolves wrapped observation/action spaces from dataset metadata and placeholder wrapping, not from a live renderer-backed env.
   - when `num_train_frames=0`, the workspace no longer creates train envs up front.
   - when `num_train_frames=0` and `env.use_live_env=true`, live robomimic eval envs are now created lazily at evaluation time and closed immediately afterward.
   - this fixes the previous behavior where image offline BC could allocate offscreen renderer memory on startup, or on a different GPU than the training process.
   - `test_comparison.md` now uses `gpu_id=...` instead of raw `CUDA_VISIBLE_DEVICES=...` so the command itself controls both training and render placement.

18. Robomimic demo preprocessing no longer computes expensive statistics unless the config actually needs them.
   - `RobomimicEnvFactory.collect_or_fetch_demos()` now computes:
     - action stats only when `use_standardization=true` or `use_min_max_normalization=true`
     - observation stats only when `norm_obs=true`
   - this removes a major source of wasted startup time for image BC, where the old path would stack all pixel observations just to compute `obs_stats` even when observation normalization was disabled.

19. Replay episode files can now be intentionally persisted and reused across runs.
   - new replay config flags:
     - `replay.persist`
     - `replay.reuse_saved`
   - when `replay.persist=true`, replay episode `.npz` files are kept on disk even without full snapshots.
   - when `replay.reuse_saved=true`, `Workspace` can skip rebuilding the demo replay from the dataset if:
     - replay files already exist in the configured replay directory
     - no demo-derived stats are required
     - no separate demo replay buffer is requested
   - `UniformReplayBuffer` now bootstraps replay metadata directly from saved episode files so the trainer can start sampling immediately on the next run.
   - this is intended specifically to avoid repeated image-demo import costs for repeated offline BC runs.

### Not done yet

1. `robobase/models/*` is still torch-first overall.
   - BC now has backend-specific model components, and the core BC model files have moved under `backends/torch/models`.
   - however, the broader algorithm stack still imports the top-level `robobase.models.*` names, which remain compatibility aliases.
   - the rest of the algorithms still point directly at torch-first model configs.

2. BC model parity is still incomplete on the JAX side.
   - JAX BC now reads the shared BC actor/encoder/view-fusion specs and executes low-dimensional plus current pixel BC.
   - the pixel encoder path is now native JAX at runtime for the current frozen ResNet BC path.
   - however, that native path is still intentionally limited:
     - frozen BasicBlock ResNets only
     - pretrained weights are still sourced from timm at initialization time and converted into the `jax-resnet` feature-model variables
   - the next concrete JAX encoder milestone should be to decide whether BC needs a trainable native JAX vision path or whether frozen-encoder parity is sufficient for the first stable JAX vision path.
   - trainable JAX-native encoder and trainable JAX-native view-fusion modules are not implemented yet.

3. JAX BC is no longer just a low-dimensional prototype, but it is still not fully architecture-complete.
   - actor-side BC functionality is much closer to the torch path now.
   - some shared runtime logic has been pulled out into `robobase/method/bc_runtime.py`.
   - JAX BC still does not inherit the torch `Method` base class, because that base class is itself torch-native.
   - pixel feature extraction now runs through a backend-native JAX encoder implementation, but that encoder is still frozen and intentionally narrower than the broader torch vision stack.

4. The rest of the stack is still incomplete from a backend-neutral perspective.
   - replay batching in `Workspace` is now backend-neutral at the iterator boundary, but the replay buffer package still retains PyTorch compatibility surfaces because `ReplayBuffer` still subclasses `torch.utils.data.IterableDataset`.
   - snapshots are still torch-shaped.
   - most methods and model trees are still torch-first outside the BC path.
   - shared config structure has not yet been fully split into algorithm config versus backend implementation config.
   - offline BC still runs through the generic replay path, even though a dedicated offline prefetch pipeline would be a better fit for steady GPU utilization.
   - lazy eval env creation is currently targeted at pure offline robomimic runs with `env.use_live_env=true`; the broader env lifecycle could still be unified further if we want the same behavior for more environment families.
   - replay cache reuse currently depends on a stable replay directory across runs, so repeated jobs should set `replay.save_dir=...` if they want to reuse the same disk cache outside a single Hydra run directory.

### Immediate next step

To make the structure correct beyond the current BC prototype, the next work should be:

1. decide whether BC needs trainable JAX-native encoder/fusion modules or whether frozen-encoder parity is sufficient for the first stable JAX vision path
2. continue shrinking reliance on top-level `robobase.models.*` imports by migrating the next torch model files and then flipping callers to backend-local imports
3. decide whether to build a backend-native multi-worker replay path for JAX or keep `replay.num_workers=0` as the supported setting there
4. design a dedicated offline-BC data path that is optimized for throughput rather than reusing the generic replay stack
   - pre-load or memory-map robomimic data into a directly sampleable offline dataset layout
   - avoid naive per-step HDF5 random reads during training
   - add a background prefetch queue so the trainer always has batches ready
   - add backend-specific host/device prefetch where useful:
     - torch: pinned host memory plus non-blocking device copies
     - jax: host prefetch plus device prefetch
   - use this path for offline BC and related IL methods where stable GPU utilization matters more than generic replay flexibility
5. align the remaining BC comparison differences that can still affect torch-vs-jax conclusions, especially LR scheduler behavior
   - now that the JAX encoder path has moved to a faster `jax-resnet` feature extractor, repeat the full image BC backend comparison and re-check whether JAX is still slower overall or whether the remaining gap is now in the input pipeline / eval path
6. repeat the same shared-spec plus backend-method assembly pattern for the next algorithms, starting with the BC-derived methods and then the actor-critic family

## What Is PyTorch-Specific Today

The main coupling points in the current tree are:

1. Runtime and orchestration
   - `train.py` and `robobase/workspace.py` still select `torch.device`, store torch RNG state, and save/load snapshots with `torch.save` and `torch.load`.
   - `robobase/utils.py` seeds torch and defines torch-native math/distribution helpers.
   - `robobase/logger.py` imports torch for TensorBoard.

2. Agent API
   - `robobase/method/core.py` makes `Method` inherit `torch.nn.Module`.
   - Public method signatures and type aliases are written in `torch.Tensor`.
   - Checkpoint helpers assume torch optimizers, LR schedulers, and AMP scaler types.

3. Models and algorithms
   - The method, model, and intrinsic reward trees are torch-native:
     - `robobase/method/*`
     - `robobase/models/*`
     - `robobase/intrinsic_reward_module/*`
   - Several components also depend on torch-only ecosystem packages:
     - `timm`
     - `torchvision`
     - `diffusers`

4. Replay and batching
   - `robobase/replay_buffer/replay_buffer.py` inherits `torch.utils.data.IterableDataset`.
   - replay iteration in `Workspace` is now backend-neutral, but the replay buffer package and some replay-oriented tests still retain torch compatibility surfaces.

5. Config and packaging
   - `setup.py` hard-depends on `torch`.
   - Hydra method configs point directly at PyTorch implementation targets in `robobase.method.*` and `robobase.models.*`.

6. Tests
   - Many unit tests assert `torch.Tensor` behavior directly, use `torch.save`/`torch.load`, or compare with `torch.allclose`.

## What The Current JAX Prototype Actually Means

Today, `backend=jax` does **not** mean the repo has already been fully backend-abstracted.

What it currently does mean:

- `Workspace` stays shared.
- envs, demo loading, replay storage, logging, and most orchestration stay shared.
- agent creation is routed through `robobase/backends/factory.py`.
- for `method=bc`, agent execution is handled by `robobase/backends/jax/method/bc.py`.
- that JAX BC path now supports both low-dimensional BC and the current shared pixel-BC surface.
- the JAX actor/update path is JAX/Optax, and the current BC pixel encoder runtime path is also native JAX.
- pretrained ResNet weights for that path are still imported from timm once at encoder initialization time.

What it does **not** mean yet:

- only a small part of `robobase/method/*` is backend-neutral today.
  - specifically, `robobase/method/bc.py` now holds shared BC semantics.
  - most of the rest of the method tree is still torch-first.
- `robobase/models/*` is not backend-neutral.
- BC config is now shared-spec based for BC itself, but the rest of the algorithm configs still point at torch-first method/model targets.
- snapshot IO and parts of the replay package are still torch-shaped in important places.

So the next architectural step is **not** "port everything straight to JAX". The BC method boundary has already started this migration, and the next step is to continue that work at the model boundary.

## Design Principles

1. Keep envs, replay storage, logging, metrics, and the high-level training loop backend-agnostic where possible.
2. Keep backend-specific tensor/module/optimizer code inside backend packages.
3. Do not build a fake common `Tensor` or `Module` abstraction that tries to hide all torch/JAX differences.
4. Keep environment and replay boundaries in NumPy; convert to torch or JAX arrays at the backend boundary.
5. Treat the current `method` and `model` trees as the **current torch implementation**, not as permanent backend-neutral abstractions.
6. Make `method=bc`, `method=drqv2`, etc. describe algorithm choice, while backend-specific code decides the concrete implementation.
7. Move the existing PyTorch stack first with minimal behavior change, then add JAX vertically.
8. Treat snapshot compatibility as an explicit migration problem, not an afterthought.

## Recommended Target Structure

```text
robobase/
  runtime/
    workspace.py
    agent.py
    snapshot.py
    seed.py
    logging.py
  backends/
    __init__.py
    base.py
    registry.py
    torch/
      __init__.py
      runtime.py
      batching.py
      checkpoint.py
      utils.py
      method/
      models/
      intrinsic_reward_module/
    jax/
      __init__.py
      runtime.py
      batching.py
      checkpoint.py
      utils.py
      method/
      models/
      intrinsic_reward_module/
  envs/
  replay_buffer/
  cfgs/
    backend/
    algorithm/
    implementation/
```

### Architectural rule

The current torch-first bodies in:

- `robobase/method/*`
- `robobase/models/*`
- `robobase/intrinsic_reward_module/*`

should not remain the primary source of truth forever. They should be migrated behind `backends/torch/...`, with temporary compatibility aliases only during the migration.

### Key boundary

The workspace should only depend on a small backend-neutral agent/runtime contract. For example:

```python
class Agent(Protocol):
    def train(self, training: bool) -> None: ...
    def act(self, observations, step: int, eval_mode: bool): ...
    def update(self, replay_iter, step: int, replay_buffer=None): ...
    def reset(self, step: int, agents_to_reset: list[int]) -> None: ...
    def checkpoint_state(self) -> dict: ...
    def load_checkpoint_state(self, state: dict) -> None: ...
```

PyTorch agents can continue to use `nn.Module` internally. JAX agents can use their own train-state structure. `Workspace` should stop assuming the agent itself is a torch module.

## Recommended Config Split

Current Hydra configs hard-code PyTorch classes. That will not scale to two backends cleanly. Split config responsibilities like this:

1. `cfgs/backend/torch.yaml`
   - backend name
   - device selection
   - compile/mixed-precision flags
   - backend runtime implementation target

2. `cfgs/backend/jax.yaml`
   - backend name
   - device/platform selection
   - JIT/precision flags
   - backend runtime implementation target

3. `cfgs/algorithm/<algo>.yaml`
   - shared hyperparameters only
   - no torch or jax `_target_` strings

4. `cfgs/implementation/torch/<algo>.yaml`
   - concrete method target
   - concrete model targets for the torch backend

5. `cfgs/implementation/jax/<algo>.yaml`
   - concrete method target
   - concrete model targets for the JAX backend

This avoids putting PyTorch class paths in shared configs and lets JAX diverge where necessary without breaking launch ergonomics.

### Important implication

The existing `cfgs/method/*.yaml` files are currently acting as "algorithm config + torch implementation config" at the same time. That coupling needs to be broken.

After the restructure:

- shared algorithm config should say "BC with these hyperparameters"
- torch implementation config should say "use the torch BC class and these torch model targets"
- jax implementation config should say "use the JAX BC class and these JAX model targets"

## Migration Plan

### Phase 0: Freeze the current PyTorch baseline

Before restructuring:

- Keep `backend=torch` as the default.
- Record smoke-test coverage for the current torch stack.
- Decide the JAX stack early.
  - Recommendation: standardize on one stack for the whole backend, for example `jax` + `optax` + one model/checkpoint library, instead of mixing styles per algorithm.

### Phase 1: Extract backend-neutral runtime interfaces

Create backend-neutral runtime modules and move framework assumptions out of `robobase/workspace.py`.

Concrete changes:

- Introduce `robobase/runtime/agent.py` with an agent protocol/interface.
- Introduce `robobase/backends/base.py` with a backend runtime interface:
  - resolve device/platform
  - seed backend RNG
  - create replay iterator
  - save/load backend checkpoint payload
  - convert batch data to backend arrays
- Move torch-specific seed, device, snapshot, and batching logic out of `robobase/workspace.py`.
- Update `Workspace` to depend on a backend object, not on torch directly.

Target outcome:

- `Workspace` remains one training loop.
- Backend code handles framework-specific setup.

### Phase 2: Make the current PyTorch implementation an explicit backend

This is the key architectural step. The repo should stop pretending that the current top-level `method` and `model` trees are already backend-neutral.

Move the existing code with minimal behavioral change:

- `robobase/method/*` -> `robobase/backends/torch/method/*`
- `robobase/models/*` -> `robobase/backends/torch/models/*`
- `robobase/intrinsic_reward_module/*` -> `robobase/backends/torch/intrinsic_reward_module/*`
- torch-specific utility helpers -> `robobase/backends/torch/utils.py`

During the migration:

- Keep thin compatibility wrappers in the old import paths so current configs and tests do not all break at once.
- Keep `robobase.method.*` and `robobase.models.*` as torch aliases temporarily.
- Move `Method` as a torch-specific base class out of `robobase/method/core.py`.
- Replace `robobase/method/core.py` with backend-neutral interfaces or runtime contracts.

Target outcome:

- The current repo still behaves the same for PyTorch.
- The code tree makes it obvious which parts are torch-only.

### Phase 3: Remove torch-first ownership from shared method/model configuration

Once the torch implementation lives under `backends/torch`, the shared configuration and construction path should no longer assume torch as the default implementation language.

Concrete changes:

- replace direct shared-config `_target_` references to `robobase.method.*`
- introduce algorithm configs that are backend-neutral
- add backend implementation configs for torch and jax
- update agent construction so both torch and jax go through the same backend-dispatch shape
- remove one-off backend special cases where possible

Target outcome:

- `backend=torch` and `backend=jax` are symmetric at the construction boundary
- shared configs describe algorithms, not torch classes
- the current minimal JAX BC path stops being a permanent exception

Current status:

- done for BC at the method boundary
- largely done for BC model specs
- not done yet for full BC pixel/model parity in JAX
- not done yet for the rest of the algorithms

### Phase 4: Decouple replay batching from PyTorch `DataLoader`

This is a critical step. JAX support will be much cleaner if replay storage stays NumPy-first and batch delivery is backend pluggable.

Concrete changes:

- Remove `ReplayBuffer`'s inheritance from `torch.utils.data.IterableDataset`.
- Define a backend-neutral replay sampling surface, for example:
  - `sample(batch_size)`
  - `iterator(...)`
- Let each backend provide its own batch iterator/prefetcher.
  - PyTorch backend can still use `DataLoader` internally if useful.
  - JAX backend can use a pure Python/NumPy iterator and optional device prefetch.

Target outcome:

- Replay buffer becomes storage + sampling logic.
- Backend becomes responsible for turning samples into backend-native batches.

### Phase 5: Introduce a new portable snapshot format

Current snapshots are torch `.pt` payloads from `Workspace.save_snapshot()`. That is fine for PyTorch-only, but not for mixed backends.

Recommended new snapshot layout:

```text
snapshots/
  latest/
    manifest.json
    runtime_state.pkl
    replay/
    agent/
```

The manifest should include:

- `snapshot_version`
- `backend`
- algorithm name
- runtime counters
- optional config metadata

Rules:

- Runtime state should contain Python and NumPy RNG state.
- Each backend stores its own agent/train-state payload inside `agent/`.
- Keep replay state backend-agnostic where possible.
- Keep a torch loader for existing v2 `.pt` snapshots so old PyTorch runs can still resume.

Target outcome:

- Snapshot v3 supports both backends cleanly.
- Old torch snapshots remain loadable through a compatibility path.

### Phase 6: Clean up packaging and optional dependencies

`setup.py` should stop forcing torch into the base install.

Recommended dependency split:

- Core install:
  - hydra
  - omegaconf
  - numpy
  - logging/video/common utilities
  - environment integrations that are backend-neutral

- `extras_require["torch"]`:
  - `torch`
  - `timm`
  - `torchvision`
  - `diffusers`

- `extras_require["jax"]`:
  - `jax`
  - `jaxlib`
  - chosen JAX optimizer/checkpoint libraries

Also clean up minor torch-only runtime imports:

- `robobase/logger.py` should use a backend-neutral TensorBoard path or make TensorBoard fully optional.

### Phase 7: Implement a minimal JAX vertical slice

Do not start by porting every algorithm. Prove the backend architecture with the smallest useful slice.

Recommended first JAX slice:

1. backend config and runtime
2. RNG/device management
3. snapshot save/load
4. one simple low-dimensional model stack
5. `bc` on low-dimensional observations only
6. smoke tests for train/eval/resume

Why start here:

- It validates the whole runtime path.
- It avoids image encoders, diffusion, and world-model complexity.
- It forces the right abstractions early.

### Phase 8: Port algorithms in leverage order

Recommended order:

1. `bc` low-dim
   - easiest end-to-end proof of architecture

2. `bc` with pixels
   - adds encoder and fusion path

3. `drqv2`
   - establishes the main actor-critic path

4. `iql_drqv2`
   - reuses most of the same backbone

5. `drm`, `alix`, `sac_lix`, `cqn`
   - grouped after the actor-critic/value-based core is stable

6. intrinsic reward modules
   - `rnd`, `icm`

7. `diffusion` and `edp`
   - depend on diffusion-specific components and EMA behavior

8. `act`
   - transformer-heavy, custom model path

9. `dreamerv3` and `mwm`
   - save for last due to world-model state, AMP/compile behavior, and more complex training state

## Testing Plan

Split tests into three categories.

1. Backend-neutral contract tests
   - workspace control flow
   - replay storage semantics
   - snapshot manifest structure
   - logging/metrics shape
   - algorithm API contract

2. PyTorch backend tests
   - adapt the current torch-heavy unit tests here
   - preserve coverage for snapshot restore, tensor shapes, replay integration, and current smoke training

3. JAX backend tests
   - mirror the contract suite
   - add backend-specific smoke tests for JIT/checkpoint/device handling

Important rule:

- Do **not** require exact numerical parity between PyTorch and JAX.
- Require API parity, shape parity, snapshot resume correctness, and comparable smoke-learning behavior on fixed toy environments.

## Guardrails

1. Do not try to make one model class switch between torch and jax internally with `if backend == ...`.
   - That will turn every model into a hybrid file and make maintenance worse.

2. Do not keep torch names in shared config fields.
   - Replace `use_torch_compile` with backend-neutral capability names such as `backend.compile`.

3. Do not keep the current top-level `method` / `model` trees as permanent torch-first canonical implementations.
   - They should become backend-neutral specs, compatibility aliases, or disappear after migration.

4. Do not let environment code see backend arrays.
   - Env and replay boundaries should stay NumPy-first.

5. Do not block JAX on pixel/world-model parity.
   - Land a thin vertical slice first.

6. Do not drop old PyTorch snapshot compatibility immediately.
   - Keep the loader during the migration window.

## Definition Of Done

Minimum acceptable end state:

1. `backend=torch` is the default and passes the current smoke/unit coverage with no behavior regression.
2. `backend=jax` exists and can:
   - instantiate through Hydra
   - train and evaluate at least one algorithm end-to-end
   - save and load snapshots
   - run its own backend test suite
3. Shared algorithm configs no longer point directly at torch method/model classes.
4. The workspace, replay storage, config structure, and snapshot layout no longer hard-code PyTorch.
5. Backend-specific code lives under explicit backend namespaces.
6. `backend=torch` and `backend=jax` are symmetric at the agent-construction boundary instead of relying on one-off JAX exceptions.

## Practical First PR Sequence

If I were implementing this in this repo, I would split the work into these PRs:

1. Add backend config group and backend runtime interfaces. No behavior change.
2. Move torch device/RNG/snapshot logic out of `robobase/workspace.py`.
3. Move current torch BC into `robobase/backends/torch/` and route both torch and jax BC through the same backend factory shape.
   - status: done
4. Move the rest of torch methods/models/intrinsic modules under `robobase/backends/torch/` with compatibility aliases.
   - status: partially done
   - BC method path is moved, and BC model components now exist under backend namespaces
   - most other algorithms and most concrete model implementations are not moved yet
5. Split shared algorithm config from backend implementation config.
   - status: partially done
   - BC algorithm config no longer points at a BC class target
   - BC model config is now shared-spec based
   - the same split still needs to be applied to the rest of the algorithms
6. Decouple replay batching from `IterableDataset` and `DataLoader`.
7. Add snapshot v3 with backward-compatible torch snapshot loading.
8. Port the actor-critic family.
9. Port advanced methods last.

That sequence keeps PyTorch stable while opening a clean path for JAX instead of mixing both frameworks inside the current files.
