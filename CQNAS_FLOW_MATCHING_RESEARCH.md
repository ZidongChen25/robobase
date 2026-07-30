# CQN-AS × Flow Matching 研究与落地方案

日期：2026-07-21

## 结论

本仓最合适的第一版组合是：

1. Flow Matching（FM）根据观测生成多个完整动作序列候选
   `a[1:M] in R^(K x D)`；
2. CQN-AS 对每个候选做分布式价值评估；
3. 选择分数最高的序列，再按 CQN-AS 的 temporal ensemble 执行第一步；
4. 第一阶段冻结 FM，并保持原 CQN-AS TD target 不变。

这个方案不需要对动作求 Q 梯度，能直接适配本仓的 coarse-to-fine
离散 critic，也是风险最低、最容易与两个独立 baseline 做公平比较的版本。

不建议第一版把 `-Q(s, a_flow)` 直接反传进 FM。当前 CQN-AS 通过
`floor -> discrete bin -> gather` 评估连续动作；它关于输入动作的梯度几乎处处为零。
因此依赖 `grad_a Q` 的 FQL actor loss、FlowQ、DFQL、QFQL/QGF 不能直接套在
现有 critic 上。

## 本仓现有基础

- [`robobase/method/cqn_as.py`](robobase/method/cqn_as.py) 已提供
  `[L, K, D, bins, atoms]` 的 C51 critic、coarse-to-fine 推理和任意 replay
  chunk 的 bin gather。
- [`robobase/method/flow_matching.py`](robobase/method/flow_matching.py)
  的 `_build_sample_from_noise_fn()` 已能从外部高斯 noise 生成完整
  `[B, K, D]` chunk；将 candidate 维并入 batch 即可并行采样。
- [`robobase/replay_buffer/uniform_replay_buffer.py`](robobase/replay_buffer/uniform_replay_buffer.py)
  已能从逐步执行动作构造连续 K 步窗口和 `tp1` 观测。
- CQN-AS 接线已经保证 replay 保存 temporal ensemble/noise 后实际执行的动作，
  且环境步数按 `execution_length` 而不是预测 horizon 计算。
- 历史 Torch EDP 在
  [`third_party/mobile-genima-bigym/robobase/method/edp.py`](third_party/mobile-genima-bigym/robobase/method/edp.py)
  已有“生成多个动作、再用 Q 选择”的本地先例，但没有支持视觉特征复用。

## 为什么候选排序比 Q-gradient 更适合

对候选序列 `a_m`，先取 C51 的期望：

```text
q[l,k,d](s,a_m) = sum_z softmax(logits[l,k,d,z]) * support[z]
```

CQN-AS 原论文没有定义一个 joint chunk scalar Q；它为 level、序列位置和动作维度
输出 factorized Q，并用相同 TD target 监督。因此候选排序必须显式定义聚合规则。
默认建议：

```text
score(s,a_m) = mean_{k,d} q[L-1,k,d](s,a_m)
```

需要同时消融：

- `mean_{l,k,d}`：与训练 loss 的聚合更一致；
- `min` 或低分位数：更保守，但容易被一个未校准 head 支配；
- 只评分序列前 `H_score <= K` 步：减少远期动作 head 噪声；
- C51 期望、下分位数和 CVaR：检查分布式 critic 是否能改善风险敏感选择。

候选选择本身是不可微的：

```text
m* = argmax_m score_target(s, a_m)
a_selected = a_m*
```

这不是缺陷。它避开 iterative FM 的 BPTT，也避开 CQN-AS 离散动作编码产生的零梯度。

## 分阶段路线

### Phase 0：独立 baseline

先分别训练并固定：

- 原生 CQN-AS greedy rollout；
- 原生 Gaussian-source FM behavior cloning。

同一任务、数据、视觉 encoder 规格和 eval seeds 下保存 checkpoint。先确认两边单独可用，
再判断组合收益，避免把 critic 或 FM baseline 的失败误判成组合失败。

### Phase 1：FM proposal + CQN-AS rerank（推荐 MVP）

训练：

- FM 保持原 flow-matching loss，可加载预训练 checkpoint 后冻结；
- CQN-AS 保持当前 C51 + demo BC loss；
- CQN-AS 的 next action/TD target 仍使用原生 coarse-to-fine greedy action。

执行或评估：

```python
features = encode_observation_once(observation)
noise = normal(key, [batch, candidates, K, D])
chunks = flow_sample_from_features(flow_params, noise, features)
scores = cqn_as_score(target_critic, features, chunks)
chosen = take_along_candidate_axis(chunks, argmax(scores))
action = temporal_ensemble(chosen)[0]
```

关键点：

- 视觉观测只编码一次，然后广播 feature；不能为 M 个候选重复跑 ResNet。
- 默认用 target critic 排序，减少 online critic 抖动；同时记录 online/target 排名一致率。
- `M_train=2..4`、`M_eval=8..16` 起步；先做吞吐和显存测量再扩大。
- temporal ensemble 在候选选择后执行，exploration noise 再加到实际单步动作上。
- 可选地加入原生 CQN-AS greedy chunk 作为额外候选，但必须单独报告；它可能因同一
  critic 的自举偏差而经常压过 FM 候选。

这一阶段改变 behavior policy，但不改变 CQN-AS Bellman operator，适合先验证“FM 是否能
提供比 coarse-to-fine greedy 更好的、数据分布内的 chunk”。

### Phase 2：proposal-constrained Double Q

如果 Phase 1 有稳定收益，再让 FM proposal 进入 TD target：

1. 对 `s_tp1` 采样 M 个 FM chunks；
2. online CQN-AS 聚合打分并选 `m*`；
3. target CQN-AS 评估同一 `a_m*`；
4. 用现有 C51 projection 构造 target。

这相当于 behavior-constrained / sampled-action Q-learning。它能减少 CQN-AS 在所有离散
bin 上取 max 产生的 OOD 过估计，但会改变原 CQN-AS 目标，也会把 FM 质量引入 bootstrap。
训练候选数应小于评估候选数，并监控 candidate max bias。

### Phase 3：stop-gradient advantage-weighted FM

若希望 critic 改善 FM 本身，可以给每个 dataset chunk 一个停止梯度的权重：

```text
A_hat(s,a_data) = score_target(s,a_data) - baseline(s)
w = clip(exp(A_hat / beta), 0, w_max)
L_FM = mean(stop_gradient(w) * ||v_theta(x_t,t,s) - (a_data-noise)||^2)
```

`baseline(s)` 可用同状态多个 FM 候选的平均/最大分数，或滑动 value baseline。
这一版仍不需要 `grad_a Q`，但属于新的 CQN-AS 适配算法，不应命名为 FlowQ。

### 暂不推荐：直接 Q-gradient guidance

若未来确实要复现 FlowQ、DFQL、QFQL 或 FQL 的 Q-loss，需要另建连续可微的 joint
chunk critic `Q_cont(s, a[1:K]) -> scalar`。它不能只把当前 CQN-AS score 包一层，
因为离散 encode/gather 仍阻断动作梯度。此路线还会重新引入 CQN-AS 论文试图规避的
高维 actor 利用 critic 误差问题，因此应作为独立算法研究，而不是 CQN-AS 的小扩展。

## 建议的软件结构

新增一个组合 Method，而不是让两个独立 agent 各自读取 replay：

```text
CQNASFlow
├── shared/trainable encoder or two explicit encoders
├── cqn critic params + target critic + critic optimizer
├── flow params + EMA params + flow optimizer
├── candidate sampler
├── chunk scorer/aggregator
└── one atomic checkpoint/state_dict
```

建议接口：

```python
FlowMatching.sample_candidates_from_features(
    params, noise, encoded_features
) -> [B, M, K, D]

CQNAS.score_action_chunks(
    critic_params, encoded_features, chunks, aggregation="final_mean"
) -> [B, M]
```

首版只支持 Gaussian flow source。A2A 和 Legato 有跨 rollout 的 action-history/reference
状态；并行候选需要“只提交被选候选的状态”接口，当前实现没有这个事务语义。

checkpoint 必须原子保存 critic、target critic、FM、EMA、两套 optimizer、RNG 和训练步数。
如果两个 checkpoint 独立加载，还要验证 observation/action normalization 元数据完全一致。

## 实验矩阵与验收门槛

至少比较：

| 组别 | FM | CQN-AS | 候选排序 | TD proposal constraint |
|---|---:|---:|---:|---:|
| CQN-AS baseline | 否 | 是 | 否 | 否 |
| FM baseline | 是 | 否 | 否 | 否 |
| Phase 1 | 冻结 | 是 | 是 | 否 |
| Phase 2 | 冻结 | 是 | 是 | 是 |
| Phase 3 | 加权更新 | 是 | 是 | 可选 |

固定任务、数据、训练环境步数、eval seeds 和 checkpoint 选择规则。除了 success/return，记录：

- candidate score 的最大值、均值、gap 和 online/target 排名一致率；
- C51 support 两端饱和率；
- FM candidate 的 pairwise L2、多样性塌缩率和动作边界命中率；
- 所选候选相对 dataset chunk 的距离；
- temporal ensemble 前后动作平滑度；
- 采样、critic 排序、encoder 各自延迟和峰值显存；
- M 增大时 return 是否上升、Q 是否上升但真实 return 下降。

Phase 1 的继续条件：至少两个任务、三个 seeds 上，真实 return/success 相对两个 baseline
有可重复收益，且候选数增加没有明显 Q exploitation。否则先校准 score 聚合、support 和
FM candidate quality，不进入联合训练。

## 与现有一手研究的关系

- [CQN-AS, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/9eecce65f1f93d7a3e7ae677c31942ab-Paper-Conference.pdf)
  使用 factorized action-sequence critic、GRU 和 temporal ensemble；它没有定义 joint
  differentiable chunk Q。本文档的 scalar aggregation 是工程上新增的明确选择。
- [Flow Q-Learning (FQL), ICML 2025](https://arxiv.org/abs/2502.02538)
  让多步 flow 只做 BC，再用 `-Q` 与 flow distillation 训练 one-step actor；Q-loss 仍要求
  可微连续 critic。其论文讨论的 rejection sampling 是本方案 candidate rerank 的最近先例，
  但不是 FQL 主算法。
- [FlowQ](https://arxiv.org/abs/2505.14139) 把 Q 作为 energy，训练 energy-guided
  probability path；它需要 energy/Q 的导数，并非无梯度 rerank。
- [ReinFlow](https://arxiv.org/abs/2505.22094) 给 flow transition 注入可学习噪声，构造
  可计算 likelihood 的 Markov process，再用 policy gradient/PPO 做在线微调。它不依赖
  `grad_a Q`，但属于 on-policy 路线，需要额外标量 return/advantage，而不是 CQN-AS TD
  critic 的直接组合。
- [Direct Flow Q-Learning (DFQL), ICML 2026](https://openreview.net/forum?id=RdkOaK4q6p)
  把 terminal Q gradient 注入每个 flow velocity step，避免跨步 BPTT；它仍明确依赖
  `grad_a Q`。
- [Q-Guided Flow Q-Learning, CoRL 2025 workshop](https://openreview.net/pdf/5464e2878ad6e3e04e9749707490861126795a26.pdf)
  在 flow 推理时加入 `beta * grad_a Q`；同样不能直接使用当前离散 CQN-AS critic。

因此，“FM proposals + CQN-AS reranking”应被准确描述为基于现有思想的新组合，而不是
声称复现 FQL、FlowQ、DFQL 或 QFQL。
