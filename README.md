# RoboBase JAX: Robot Learning Baselines

This checkout is a JAX-only robot-learning stack. The installed
`robobase` package does not import or install PyTorch. Historical Hydra files
for removed methods may still be readable. The runtime factory supports BC,
ACT, Diffusion, Flow Matching, PPO, SAC, CQN, CQN-AS, CQN-Flow, and
Q-Chunking.

Top Features of RoboBase:
- Well-tuned algorithms with a focus on methods that take both low-dimensional proprioceptive robot data **and** (multiple) high-dimensional vision-sensor data.
  This is in contrast to other common frameworks (StableBaselines, CleanRL, etc) that often prioritise **only** low-dimensional **or only** high-dimensional inputs.
- "Single-file" implementation of algorithms.
- First-class support for vectorised training environments.
- Wrappers around common environments, e.g. DMC and RLBench.

## Table of Contents

1. [Install](#install)
2. [Implemented Algorithms](#implemented-algorithms)
3. [Framework Overview ](#framework-overview)
4. [Usage](#usage)

## Install

System installs:

```commandline
sudo apt-get install ffmpeg  # Usually pre-installed on most systems
```

```commandline
pip install ".[jax-cuda12]"  # NVIDIA CUDA 12
# or: pip install ".[jax]"   # CPU/non-CUDA JAX
```

### DeepMind Control

```commandline
pip install ".[dmc]"
```

### RLBench

```commandline
sudo apt-get install python3.10-dev   # if using python3.10
./extra_install_scripts/install_coppeliasim.sh  # If you dont have CoppeliaSim already installed
pip install ".[rlbench]"
```

<details>
<summary>RLBench Issues?</summary>
<br>

Note: If you got an error about not finding libGL.so.1, then you need to install the following:
```commandline
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
sudo apt-get install libgl1-mesa-dev libxrender1 libxkbcommon-x11-0
```
If you still get an error, then set the following environment variable to see if the error is more informative:
```commandline
export QT_DEBUG_PLUGINS=1
```
</details>

### BiGym

RoboBase includes a local BiGym checkout at [third_party/bigym](third_party/bigym).

```commandline
pip install ".[bigym]"
```


## Implemented Algorithms

### Historical Configurations

The following table is retained for provenance only. These implementations
were removed; selecting any of these configs raises a JAX-only unsupported
method error. The old stability marks describe the archived implementation,
not the current runtime.

| Method                                        | Paper                                                                                                                                 | 1-line Summary                                                                                                                                | Differences to paper?             | Stable             |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|--------------------|
| [drqv2](robobase/cfgs/method/drqv2.yaml)         | [Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/abs/2107.09645)               | Uses augmentation (4-pixel shifting) and layer-norm bottleneck to aid learning from pixels.                                                   | None.                             | :white_check_mark: |
| [alix](robobase/cfgs/method/alix.yaml)           | [Stabilizing Off-Policy Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2207.00986)                                    | Rather then augmentation (as in DrQV2), uses a Adaptive Local SIgnal MiXing (LIX) layer that explicitly enforces smooth featuremap gradients. | None.                             | :white_check_mark: |
| [sac_lix](robobase/cfgs/method/sac_lix.yaml)     | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) | Maximum entropy RL algorithm that has adaptive exploration.                                                                                   | Uses ALIX as the base algorithm.  | :white_check_mark: |
| [drm](robobase/cfgs/method/drm.yaml)             | [DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization](https://arxiv.org/abs/2310.19668)                   | Uses dormant ratio as a metric to measure inactivity in the RL agent's network to allow effective exploration.                                | None.                             | :warning:          |
| [dreamerv3](robobase/cfgs/method/dreamerv3.yaml) | [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)                                                            | Learns world models with CNN/MLP encoder and decoder.                                                                                 | None.                             | :white_check_mark: |
| [mwm](robobase/cfgs/method/mwm.yaml)             | [Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244)                                                            | World model (similar to DreamerV2) that uses Masked Autoencoders (MAE) for visual feature learning.                                           | None.                             | :white_check_mark: |
| [iql_drqv2](robobase/cfgs/method/iql_drqv2.yaml) | [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)                                           | Does not evaluate "unseen" actions to limit Q-value overestimation.                                                                           | Uses DrQv2 as the base algorithm. | :white_check_mark: |

### JAX Reinforcement Learning

| Method | Reference | Summary | Runtime status |
|--------|-----------|---------|----------------|
| [ppo](robobase/cfgs/method/ppo.yaml) | [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | On-policy clipped objective with GAE, timeout bootstrapping, value clipping, and KL stopping. | JAX implementation, experimental |
| [sac](robobase/cfgs/method/sac.yaml) | [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) | Squashed Gaussian actor, twin Q targets, and learned entropy temperature. | JAX implementation, experimental |
| [cqn](robobase/cfgs/method/cqn.yaml) | [Coarse-to-fine Q-Network](https://younggyo.me/cqn/) | Distributional continuous-control Q-network with multi-level action zoom-in. | JAX port of RoboBase CQN, experimental |
| [cqn_as](robobase/cfgs/method/cqn_as.yaml) | [CQN with Action Sequence](https://younggyo.me/cqn-as/) | Factorized coarse-to-fine Q heads over action chunks with per-stream GRUs and one-step replanning. | JAX implementation, experimental |
| [cqn_flow](robobase/cfgs/method/cqn_flow.yaml) | [CQN-AS](https://younggyo.me/cqn-as/) + value Flow Matching | Conditional categorical-logit or scalar value flow over every action bin, with CQN-AS chunk rollout. | JAX research implementation, experimental |
| [q_chunking](robobase/cfgs/method/q_chunking.yaml) | [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969) | Flow behavior policy proposes full chunks; twin-Q Best-of-N selection uses matched K-step returns and open-loop execution. | Pure JAX implementation, experimental |

State-only DMC launches are plug-and-play:

```commandline
python3 train.py launch=ppo_state_dmc env=dmc/cartpole_balance
python3 train.py launch=sac_state_dmc env=dmc/cartpole_balance
python3 train.py launch=cqn_state_dmc env=dmc/cartpole_balance
python3 train.py launch=q_chunking_state_dmc env=dmc/cartpole_balance
```

The three-camera BiGym adapter is launched with:

```commandline
python3 train.py launch=q_chunking_pixel_bigym env=bigym/move_plate
```

Q-Chunking's upstream mapping, replay contract, measured smoke artifacts, and
matched evaluation protocol are recorded in
[Q_CHUNKING_JAX.md](Q_CHUNKING_JAX.md). Snapshot sweeps use:

```commandline
python3 scripts/eval_q_chunking_snapshot_sweep.py --run-dir <run_dir> \
  --num-eval-episodes 50 --eval-seed-start 400
```

PPO owns an on-policy rollout buffer and never trains from replay. SAC, CQN,
CQN-AS, CQN-Flow, and Q-Chunking use the standard replay path. PPO, SAC, and
CQN require `action_sequence=1`; CQN-AS, CQN-Flow, and Q-Chunking require
`action_sequence>=2` and `execution_length=1`. Q-Chunking executes one sampled
chunk open loop and uses `replay.nstep=action_sequence`.

CQN-AS demo-driven launches are available for BiGym and RLBench. They preserve
the sequence-Q tensor, replay, rollout, and objective contracts while adapting
the network to this repository's shared JAX ResNet feature path. They are not
exact replicas of the released custom visual encoder, augmentation stack, or
per-task training lifecycle:

```commandline
python3 train.py launch=cqn_as_pixel_bigym_demo_driven env=bigym/move_plate
python3 train.py launch=cqn_as_pixel_rlbench_demo_driven
```

CQN-AS value Flow Matching 的实现定义、分阶段研究方案和实验门槛见
[cqn-flow.md](cqn-flow.md)。

### JAX Imitation Learning

| Method                                        | Paper                                                                                                   | 1-line Summary                              | Differences to paper?             | Stable    |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------|-----------------------------------|-----------|
| [bc](robobase/cfgs/method/bc.yaml)               | Behavior cloning                                                                                        | Direct action regression.                   | JAX implementation.               | :white_check_mark: |
| [diffusion](robobase/cfgs/method/diffusion.yaml) | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)   | Brings diffusion to robotics.               | None.                             | :warning: |
| [flow_matching](robobase/cfgs/method/flow_matching.yaml) | Rectified Flow                                                                                   | Learns an action-space velocity field.      | JAX implementation.               | :warning: |
| [a2a](robobase/cfgs/method/a2a.yaml) | [Action-to-Action Flow Matching](https://arxiv.org/abs/2602.07322) | Transports encoded prior-action history to a future chunk. | Pure JAX; BiGym currently feeds commanded actions, see [flow extension notes](FLOW_POLICY_EXTENSIONS.md). | :warning: |
| [legato](robobase/cfgs/method/legato.yaml) | [Learning Native Continuation for Action Chunking Flow Policies](https://arxiv.org/abs/2602.12978) | Learns delay-aware continuation throughout flow integration. | Pure JAX; paper/public-code target modes are explicit. | :warning: |
| [act](robobase/cfgs/method/act.yaml)             | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)  | Transformer and action-sequence prediction. | None.                             | :white_check_mark: |

### Algorithmic Features

| Feature (argument name)                                      | Description                                                                                                                                                                           | Methods supported |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Action sequence (action_sequence)                            | Same as action chunking in ACT, it allows a model to predict a sequence of actions per inference time                                                                                 | Imitation-learning methods, CQN-AS, and CQN-Flow |
| Frame stacking (frame_stack)                                 | Stacking current frame with previous ones to provide recent input history to the model                                                                                                | All methods       |
| Action standardization (use_standardization)                 | Based on demonstration data, perform z-score normalization on actions. Note that default option clips actions beyond $3\sigma$                                                        | All methods       |
| Action min/max normalization (use_min_max_normalization)     | Based on demonstration data, perform min/max normalization on actions.                                                        | All methods       |
| Plucker camera rays (method.encoder_model.use_plucker)       | Conditions trainable visual encoders on per-camera ray geometry                                                                                                                        | ACT, BC, Diffusion, Flow Matching, PPO, SAC, CQN, CQN-AS, CQN-Flow |

Example state-only diffusion launch:

```commandline
python3 train.py launch=dp_state_robomimic env=robomimic/tool_hang
```

## Framework Overview :memo:

### Method

All implemented methods should extend `Method`:

```python
class Method:
    @property
    def random_explore_action(self) -> np.ndarray:
        ...

    @abstractmethod
    def act(
        self, observations: dict[str, np.ndarray], step: int, eval_mode: bool
    ) -> BatchedActionSequence:
        ...

    @abstractmethod
    def update(
        self,
        replay_iter: Iterator[dict[str, np.ndarray]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> Metrics:
        ...

    @abstractmethod
    def reset(self, step: int, agents_to_reset: list[int]):
        # Called on each environment.
        ...
```

### Replay Buffer / Updates

Within the `update` method, we can access batch data from the replay buffer via:
```python
batch = next(replay_iter)
```
Batch leaves are NumPy arrays, or JAX arrays when device prefetch is enabled.
Observation data has shape `(B, T, ...)`, where `B` is batch size and `T` is
the observation history (frame stack). JAX methods own the device transfer and
keep model update/sampling loops under `jax.jit`.

### Flax Modules

Backbone, encoder, and view-fusion modules are selected through the method's
Hydra config and implemented with Flax/JAX.

If you are frame stacking on channel, i.e. `frame_stack_on_channel=true`, then:
```
(B, V, T, C, W, H)
 ⌄
(B, V, T * C, W, H)
 ⌄
|EnoderModule|
 ⌄
(B, V, Z)
 ⌄
|FusionModule|
 ⌄
(B, Z',)
 ⌄
|FullyConnectedModule|
 ⌄
(B, T', A)
```

If you are using an rnn to roll in the frame stack, i.e. `frame_stack_on_channel=false`, then:, then:
```
(B, V, T, C, W, H)
 ⌄
(B * T, V, C, W, H)
 ⌄
|EnoderModule|
 ⌄
(B * T, V, Z)
 ⌄
|FusionModule|
 ⌄
(B * T, Z')
 ⌄
(B, T, Z')
 ⌄
|FullyConnectedModule|
 ⌄
(B, T', A)
```
where `V` is the number of cameras/views, and `T'` is the action output sequence.
Note that `FullyConnectedModule` can have either a 1-dim `(Z,)` input or a 2-dim `(T, Z)` input.

To stop training, execute `ctrl-c` in the terminal. This will cleanly terminate the training process.

## Usage

The method, backbone, encoder, and architecture/preprocessing profile are
separate Hydra groups. A `launch` composes those modules with a complete data
and training lifecycle, so it is the plug-and-play entry point for a real run.

CamPose BiGym launches need reset-aligned pixel caches containing camera
intrinsics and camera-to-world poses. The cache command is pure NumPy/JAX and
does not require Torch:

```commandline
MUJOCO_GL=egl python3 scripts/cache_bigym_pixel_demos.py \
  --task move_plate --cameras head left_wrist --resolution 256x256 \
  --observation-timing pre_action --include-camera-params --force-recache
```

CamPose diffusion-policy architecture/objective path:

```commandline
python3 train.py launch=campose_dp_bigym env=bigym/move_plate
```

The launch selects 256 px input, one observation frame, a 32-action horizon,
DDPM sampling, the CamPose local-conditioned UNet, DP early Plucker fusion,
offline replay, and a fixed optimizer horizon. Components remain independently
selectable for custom launches:

```commandline
python3 train.py --cfg job method=diffusion backbone=campose_dp_unet \
  encoder=dp_early_plucker env=bigym/move_plate \
  env.camera_conditioning=plucker pixels=true action_sequence=32
```

This second command only displays the composed module config; use a complete
launch for training. The native `launch=dp_state_robomimic` is retained for old
experiments and is not labeled as CleanDiffuser parity. The matched state-only
training baseline is a separate plug-and-play launch and environment contract:

Official low-dimensional task modules are `robomimic_clean/lift`, `can`,
`square`, `tool_hang`, and `transport`; each owns its observation keys and
episode limit.

```commandline
python3 train.py launch=clean_diffuser_dp_state_robomimic \
  env=robomimic_clean/tool_hang \
  env.dataset_path=/path/to/robomimic/tool_hang/ph/low_dim_v141.hdf5
```

The corresponding ContinuousRectifiedFlow core is available as an explicitly
labeled extension, because CleanDiffuser does not publish it as a RoboMimic
task recipe:

```commandline
python3 train.py launch=clean_diffuser_rf_state_robomimic \
  env=robomimic_clean/tool_hang \
  env.dataset_path=/path/to/robomimic/tool_hang/ph/low_dim_v141.hdf5
```

CleanDiffuser's RoboMimic DiT architecture is also a plug-and-play pure-JAX
backbone. It selects fused-QKV attention, adaLN-Zero, Fourier timestep features,
and the `[256]` ReLU `MLPCondition` with sample-level condition dropout:

```commandline
python3 train.py launch=clean_diffuser_dit_ddpm_state_robomimic \
  env=robomimic_clean/lift env.dataset_path=/path/to/low_dim_abs.hdf5

python3 train.py launch=clean_diffuser_dit_fm_state_robomimic \
  env=robomimic_clean/lift env.dataset_path=/path/to/low_dim_abs.hdf5
```

The backbone matches the pinned `DiT1d` operator, but the published
CleanDiffuser RoboMimic recipe uses the EDM objective. The explicitly named
DDPM and Flow Matching launches are architecture experiments, not full recipe
parity.

The isolated state-only performance comparison remains:

```commandline
python3 benchmarks/compare_policy_backends.py --gpu 0 \
  --clean-root /tmp/CleanDiffuser-baseline-05f17fc9 \
  --objectives diffusion --ema-modes on --torch-modes eager default \
  --sample-steps 100 --pair-repeats 5
```

Only the benchmark's external worker imports Torch. The installed `robobase`
package and its dependency graph remain JAX-only.

See [IMITATION_BASELINE_AUDIT.md](IMITATION_BASELINE_AUDIT.md) for the exact
matched fields, intentional non-parity, performance evidence limits, and the
next algorithm/test plan.

ACT late fusion can be enabled independently with
`encoder=act_late_plucker`. Use `profile=campose_act` when learned image
positions, camera extrinsic tokens, and synchronized RGB/ray augmentation are
required. The official path supports one or two cameras and always reserves
two extrinsic tokens; for a two-camera BiGym launch:

```commandline
python3 train.py launch=campose_act_bigym env=bigym/move_plate
```

Flow Matching uses its own continuous-time profile and must not reuse the DDPM
profile:

```commandline
python3 train.py launch=campose_fm_bigym env=bigym/move_plate
```

The CamPose profiles match the model topology, camera-ray convention, fusion
strategy, objective, and declared preprocessing switches. Environment-specific
state layouts, gripper handling, mixed precision, and dataset adapters remain
RoboBase concerns and must be matched explicitly in an experiment config.
`campose_fm` is a JAX Rectified Flow extension; it is not an official CamPose
method.

Available JAX methods are `bc`, `act`, `diffusion`, `flow_matching`, `ppo`,
`sac`, `cqn`, `cqn_as`, `cqn_flow`, and `q_chunking`. Other historical RL
configs intentionally raise an unsupported-method error.

### Running existing algorithms/networks on custom environments.

In a new project/repo, you will need to create a minimum of 3 files:
1. A Hydra config for your environment, e.g. `myenv.yaml`
2. An environment and a corresponding `Factory` to build it, e.g. `myenv.py`
3. A launch file that hooks everything together, e.g. `train.py`

**myenv.yaml**
```yaml
# @package _global_
env:
  env_name: my_env_name
  physics_dt: 0.004  # The time passed per simulation step
  # Others ways to configure your environment
```

**myenv.py**
```python
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    AppendDemoInfo,
    ConcatDim,
)


class MyEnv(gym.Env):
  pass


class MyEnvFactory(EnvFactory):

    def _wrap_env(self, env, cfg):
        env = RescaleFromTanh(env)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        env = ConcatDim(env, 1, 0, "low_dim_state")
        env = TimeLimit(env, cfg.env.episode_length)
        env = FrameStack(env, cfg.frame_stack)
        env = AppendDemoInfo(env)
        return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        return gym.vector.AsyncVectorEnv(
            [
                lambda: self._wrap_env(MyEnv(), cfg)
                for _ in range(cfg.num_train_envs)
            ]
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        return self._wrap_env(MyEnv(), cfg)
```

**train.py**
```python
import hydra
from robobase.workspace import Workspace
from myenv import MyEnvFactory

@hydra.main(
    config_path="cfgs", config_name="my_cfg", version_base=None
)
def main(cfg):
    workspace = Workspace(cfg, env_factory=MyEnvFactory())
    workspace.train()


if __name__ == "__main__":
    main()
```


### Running novel/experimental algorithms/networks on existing environments

In a new project/repo, add:
1. A Hydra config for your method, e.g. `mymethod.yaml`
2. A method class e.g. `mymethod.py`
3. A construction branch in `robobase.factory.create_agent`

**method/mymethod.yaml**
```yaml
# @package _global_
method:
  name: mymethod
  my_special_parameter: 1
```

**mymethod.py**
```python
import numpy as np
from robobase.method.core import Method, BatchedActionSequence, Metrics
from typing import Iterator
from robobase.replay_buffer.replay_buffer import ReplayBuffer

class MyMethod(Method):
  def __init__(self, my_special_parameter: int, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.my_special_parameter = my_special_parameter

  def reset(self, step: int, agents_to_reset: list[int]):
    pass

  def update(self, replay_iter: Iterator[dict[str, np.ndarray]], step: int,
             replay_buffer: ReplayBuffer = None) -> Metrics:
    pass

  def act(self, observations: dict[str, np.ndarray], step: int,
          eval_mode: bool) -> BatchedActionSequence:
    pass
```

You can then launch that algorithm on an environment, e.g.
```commandline
python3 train.py --config-dir=. method=mymethod env=dmc/cartpole_swingup env.episode_length=1000
```
where `config-dir` adds a config directory to the Hydra config search path.

### Running novel/experimental algorithms/networks on custom environments

A combination of the two configurations described above.

## Optimisations

### Logging

In your method, only log when logging is True; this will be slight more efficient, especially if you log a lot.

```python
from robobase.method.core import OffPolicyMethod

class MyMethod(OffPolicyMethod):
  def update(self, *args):
    metrics = {}
    if self.logging:
      metrics["loss"] = 0
    return metrics

```
