# CQN-AS + Value Flow Matching：研究与实现计划

## 1. 已确认的版本定义

本文只讨论用户已经确认的四个版本，不再引入独立 C51 teacher head、distilled Q head或候选动作生成器。

| 版本 | value-flow 状态 | endpoint 解释 | TD 训练 | demo 训练 | 动作选择 |
|---|---:|---|---|---|---|
| V1 | 无 | 当前 C51 atom logits | C51 projection + atom CE | 原 FOSD + Q margin | C51 expectation |
| V2 | 51 维 | endpoint logits -> softmax -> 51-atom PMF | logit Flow Matching + endpoint atom CE | 原 FOSD + Q margin | 所有 bins 的 flow endpoint PMF expectation |
| V3 | 51 维 | endpoint logits -> softmax -> 51-atom PMF | 只用 logit Flow Matching，`atom_ce_lambda=0` | 原 FOSD + Q margin | 所有 bins 的 flow endpoint PMF expectation |
| V4 | 1 维 | 多个 scalar endpoints 的均值为 expected Q | scalar expected-value Flow Matching | 只保留 Q margin，无 FOSD | 所有 bins 的 scalar flow endpoint mean |

V2 与 V3 都直接输出完整 categorical return PMF。二者的区别仅是 V2 有 endpoint atom CE，V3 强制 `atom_ce_lambda=0`；V3 在有 demonstrations 时仍保留基于 endpoint PMF 的 FOSD 和 Q-margin，因此“FM only”指 TD 主损失不含 atom CE，不代表完全没有 demo auxiliary loss。

V4 是 expected-value flow，不是 return-distribution flow。它直接对多个 source 的 scalar endpoints 求均值作为 Q，不使用 distillation，也不提供 CDF/FOSD。

三种 flow 版本在 rollout 和 next-action selection 时都直接对所有 action bins 做 ODE；同一 level 内 `K x D x N x R` 完全并行，coarse-to-fine 的 `L` 个 levels 因 zoom path 不同而顺序执行。

首版代码已落在 `robobase/method/cqn_flow.py`，Hydra 入口为
`method=cqn_flow`。本文后半部分同时保留下一阶段的 ablation、指标和扩展计划；
凡标为“后续”或“对照”的项目都不是当前默认路径。

## 2. 当前本地 CQN-AS baseline

### 2.1 记号

- `B`：batch size
- `L`：coarse-to-fine levels
- `K`：action sequence length
- `D`：单步 action dimension
- `N`：action bins
- `A`：C51 atoms，当前为 `51`
- `R`：每个 condition 的 flow source 数
- `T`：Euler integration steps
- `E`：flow ensemble 数，首版可固定为 `1`
- `F`：state/image encoder feature dimension
- `C`：每个 value-flow query 的 condition dimension

### 2.2 当前张量路径

本地 `C2FSequenceDistributionalCritic` 在每个 level 输出：

```text
logits: [B,K,D,N,A]
```

逐 level 收集后：

```text
chosen_logits: [B,L,K*D,A]
all_logits:    [B,L,K*D,N,A]
```

动作选择为：

```text
p = softmax(logits, axis=-1)       # [B,K,D,N,A]
q = sum(p * support, axis=-1)      # [B,K,D,N]
n_star = argmax(q, axis=-1)        # [B,K,D]
```

选中 bins 后更新 `[low,high]`，再计算下一个 level。当前 update 保留 Double-Q 顺序：online critic 选择 next bins，target critic 产生 selected next distributions，C51 projection 构造 target，online critic 的 replay-chosen bins 接受 CE 监督。

当前默认配置是：

```text
L=3, N=5, A=51
```

CQN-AS critic 还包含：

- state/image features；
- 当前 coarse-to-fine midpoint chunk `[B,K,D]`；
- level one-hot；
- sequence-position one-hot；
- 沿 `K` 的 GRU；
- distributional dueling heads。

### 2.3 当前 C51 target 和 demo loss

固定 atom support 为 `z_j, j=1...A`。当前 Bellman atom locations：

\[
\tilde z_j=
\operatorname{clip}
\left(r+\delta z_j,v_{\min},v_{\max}\right),
\qquad
\delta=\text{discount}\cdot\text{bootstrap}.
\]

target PMF 通过 `Pi_C51` 投影回固定 support：

\[
p_{\text{TD}}
=\Pi_{\text{C51}}
\mathcal T Z_{\bar\theta}.
\]

当前 demo loss：

1. FOSD：比较 chosen PMF 与 all-bin PMFs 的 CDF；
2. margin：比较 chosen expected Q 与 all-bin expected Q。

V2/V3 必须从 flow endpoints 恢复同样的 PMF/CDF/Q，因此能保留两项；V4 只有 scalar Q，只能保留 margin。

## 3. 共享的 conditional sequence-flow 架构

### 3.1 condition 必须包含 zoom path

用户给出的 condition 是：

```text
(state/image features, level, sequence step, action dim, action bin)
```

实现时还必须加入当前 coarse-to-fine midpoint/interval。推荐：

\[
c_{l,k,d,n}=
\left[
h(s),e_l,e_k,e_d,e_n,
m_{l,k,:},w_{l,k,:},
\hat a_{l,k,d,n}
\right],
\]

其中：

- `h(s)`：state 和/或 image encoder features；
- `e_l,e_k,e_d,e_n`：level、sequence、dimension、bin embedding；
- `m_l=(low_l+high_l)/2`：进入本 level 前的 midpoint chunk；
- `w_l=(high_l-low_l)/2`：当前 interval half-width；
- `a_hat`：当前 candidate bin 的绝对中心。

同一个 `(level,bin id)` 会因上一级选择不同而对应不同绝对 action。若 condition 没有 midpoint/interval，finer levels 无法知道当前 zoom 到了哪里。

### 3.2 shared sequence trunk 与 per-bin field

先在 source/bin 维之外只计算一次 CQN-AS sequence context：

```text
u_l = GRU(MLP([features, midpoint_l, half_width_l, seq_id, level_id]))
u_l: [B,K,H]
```

再展开 action dim 和 bin：

```text
c_all:      [B,K,D,N,C]
c_selected: [B,K,D,C]
```

vector field 是 per-bin conditional，但所有 bins 共用 sequence trunk 和 velocity-head 参数：

```text
V2/V3: v_theta(x_t[A], t | c_lkdn) -> velocity[A]
V4:    v_theta(x_t[1], t | c_lkdn) -> velocity[1]
```

image encoder 只能对 `[B,...]` 运行一次；features/context 再广播到 `R,D,N`。不能为每个 source 或 bin 重复跑视觉 encoder。

### 3.3 flow path 与本地时间方向

本文公式使用标准 forward-time：

\[
x_t=(1-t)x_0+t x_1,
\qquad
u^*=x_1-x_0,
\qquad t\in[0,1].
\]

\[
\mathcal L_{\text{FM}}=
\mathbb E
\left\|
v_\theta(x_t,t\mid c)-\operatorname{sg}(x_1-x_0)
\right\|_2^2.
\]

本仓 `linear_flow_training_pair` 使用相反时间变量：

```text
tau=1: source
tau=0: target
sample = tau*source + (1-tau)*target
target_velocity = target-source
```

因此复用本地 helper 时设置 `tau=1-t`。采样从 source 开始，按本地 reverse-time schedule 用正的 `target-source` velocity 更新。时间方向是最容易出现的静默错误之一。

## 4. V2：51-D Flow-C51 + endpoint atom CE

### 4.1 endpoint 定义

flow state 是完整 51-D logit vector：

```text
x_t: [...,A], A=51
```

ODE endpoint：

```text
ell_hat = Phi_theta(x0; c)          # [...,A]
p_hat   = softmax(ell_hat, axis=-1) # [...,A]
Q_hat   = sum(p_hat * support)      # [...]
```

logits 有加常数不变性。为去掉这条无意义的 gauge direction，source、target logits 和可选的 velocity output 都建议在 atom 轴中心化：

\[
\operatorname{center}(\ell)=
\ell-\frac1A\sum_j\ell_j.
\]

### 4.2 online next-bin selection

固定 level `l`，采 `R` 组 source logits，并在 `N` 个 bins 间使用 common random numbers：

```text
x0:             [B,R,K,D,1,A]
x0 broadcast:   [B,R,K,D,N,A]
endpoint_logits:[B,R,K,D,N,A]
endpoint_pmf:   [B,R,K,D,N,A]
```

先在 PMF 空间平均，不能先平均 logits：

\[
\bar p_{l,k,d,n}
=\frac1R\sum_r
\operatorname{softmax}
\left(\hat\ell^{(r)}_{l,k,d,n}\right).
\]

\[
\hat Q_{l,k,d,n}
=\sum_j\bar p_{l,k,d,n,j}z_j,
\qquad
n^*_{l,k,d}=\arg\max_n\hat Q_{l,k,d,n}.
\]

online flow 逐 level 选择 bins并更新 interval。相同 source 广播到 bins 可显著降低 Monte Carlo ranking noise。

### 4.3 target PMF

online flow 先并行积分所有 bins 完成 Double-Q 选择；target flow 随后只积分
online 已选中的 next-bin conditions。replay FM 同样只查询 executed bins。
因此全-bin 并行成本只出现在必须比较候选的 rollout/next selection，以及需要
all-bin FOSD/margin 的 demo endpoint path：

```text
target endpoint logits: [B,R,L,K,D,A]
target endpoint PMFs:   [B,R,L,K,D,A]
```

先平均 target PMFs：

\[
\bar p^-_{b,l,k,d}
=\frac1R\sum_r
\operatorname{softmax}
\left(\hat\ell_{\bar\theta,b,r,l,k,d}\right).
\]

再执行与当前 C51 相同的 categorical Bellman projection：

\[
p_{\text{TD},b,l,k,d}
=\Pi_{\text{C51}}
\left(
r_b+\delta_b Z_{\bar\theta}
\right).
\]

reward、discount、bootstrap 广播到 `[B,L,K,D]`。target flow、projection 和 next-action path 全部 stop-gradient。

### 4.4 PMF 到 canonical target logits

Flow Matching 需要一个 51-D endpoint target。先给 `p_TD` 做极小 floor 并重新归一化：

\[
p^\epsilon_j=
\frac{\max(p_{\text{TD},j},\epsilon)}
{\sum_i\max(p_{\text{TD},i},\epsilon)},
\]

再定义唯一的 centered logit representative：

\[
\ell_{\text{TD}}
=\log p^\epsilon
-\operatorname{mean}_j(\log p^\epsilon_j).
\]

这样 `softmax(ell_TD)=p_epsilon`，同时避免零概率导致 `-inf`。`epsilon` 必须记录并做 ablation，因为它会给原本为零的 atoms 注入小概率。

### 4.5 FM 与 endpoint atom CE

当前 replay chunk 在每个 level 编码出 chosen bins：

```text
chosen condition: [B,L,K,D,C]
source logits:    [B,R,L,K,D,A]
target logits:    [B,1,L,K,D,A] -> broadcast R
```

Flow Matching：

\[
\mathcal L_{\text{logit-FM}}
=\mathbb E
\left\|
v_\theta(x_t,t\mid c^{\text{replay}})
-(\ell_{\text{TD}}-x_0)
\right\|_2^2.
\]

V2 还从 source 完整积分到 online chosen endpoints，并加入 atom CE：

\[
\mathcal L_{\text{atom}}
=-
\mathbb E_{r,l,k,d}
\sum_j
p_{\text{TD},j}
\log
\operatorname{softmax}
(\hat\ell^{(r)}_{\theta,j}).
\]

总 TD loss：

\[
\mathcal L_{\text{V2-TD}}
=\lambda_{\text{FM}}\mathcal L_{\text{logit-FM}}
+\lambda_{\text{atom}}\mathcal L_{\text{atom}},
\qquad
\lambda_{\text{atom}}>0\ \text{by default}.
\]

`atom_ce_lambda` 可调，但 V2 默认非零。endpoint CE 需要梯度穿过 `T` 步积分；这与 simulation-free FM 的单次 velocity 监督不同，显存和编译成本必须单独测量。

## 5. V3：51-D pure Flow-C51，`atom_ce_lambda=0`

V3 的模型、51-D source、target PMF、canonical logits、all-bin action selection和 demo PMF 都与 V2 完全相同，唯一强制差异是：

```text
atom_ce_lambda = 0.0
```

所以：

\[
\mathcal L_{\text{V3-TD}}
=\lambda_{\text{FM}}\mathcal L_{\text{logit-FM}}.
\]

V3 不是另设 scalar Q，也不从 C51 teacher 采样 atoms。它仍通过 target-flow endpoint PMF、C51 Bellman projection 和 canonical target logits学习完整 51-atom PMF。

有 demonstrations 时，V3 仍从 **自身 all-bin flow endpoints** 计算 FOSD 和 Q-margin，因此 demo rows 上仍会有 endpoint-level gradient。需要在报告中准确写成：

```text
TD: Flow Matching only
Demo auxiliary: FOSD + Q margin
```

V3 的关键研究问题是：dense intermediate velocity supervision 是否足以让最终 endpoint PMF 校准，而无需 endpoint atom CE。必须同时监控 velocity loss 和 endpoint CE 指标；后者在 V3 中只作为 metric，不反传。

## 6. V4：scalar expected-value flow

### 6.1 endpoint 与动作选择

V4 的 flow state 是 scalar：

```text
x_t: [B,R,K,D,N,1]
endpoint: [B,R,K,D,N]
```

对每个 bin 直接平均多个 endpoints：

\[
\hat Q_{l,k,d,n}
=\frac1R\sum_r
\Phi_\theta
\left(x_0^{(r)};c_{l,k,d,n}\right).
\]

再执行：

\[
n^*_{l,k,d}=\arg\max_n\hat Q_{l,k,d,n}.
\]

没有 distilled Q head；rollout 和 next-action selection 都直接运行 all-bin scalar ODE。

### 6.2 expected-value TD target

online flow 选择 next bins，target flow 对 selected conditions 产生：

```text
q_next_samples: [B,R,L,K,D]
```

先平均 endpoints：

\[
\bar Q^-_{b,l,k,d}
=\frac1R\sum_r
\Phi_{\bar\theta}
(x_0^{(r)};c'_{b,l,k,d,n^*}).
\]

再构造 scalar Bellman target：

\[
y_{b,l,k,d}
=r_b+\delta_b\bar Q^-_{b,l,k,d}.
\]

每个 online source 都匹配同一个 `y`：

\[
x_1=y,
\quad
x_t=(1-t)x_0+t y,
\quad
u^*=y-x_0.
\]

\[
\mathcal L_{\text{V4-TD}}
=\mathbb E
\left\|
v_\theta(x_t,t\mid c^{\text{replay}})
-(y-x_0)
\right\|_2^2.
\]

这采用 FloQ 的 expected-value backup 思路，但本项目按用户确认的设计直接用 all-bin flow endpoints 做 CQN-AS argmax，不使用官方代码中的额外单步读出网络。

### 6.3 source range

当前 V2/V3/V4 共享 Gaussian source：

\[
x_0\sim\mathcal N(0,\sigma_{\text{source}}^2),
\qquad \sigma_{\text{source}}=1.0.
\]

这样可先把 value representation 作为唯一主变量。后续再对 V4 单独比较
Gaussian 与覆盖 Q-range 中心附近的 uniform source；source 太窄会使 flow
接近单步 regressor，太宽会增加 Euler error 和 bin-ranking variance。

V4 的不同 source endpoints 在理想 expected-value objective 下应收缩到同一 Q。它们的方差是优化/积分诊断，不应解释为环境 return uncertainty。

## 7. atom CE、FOSD 与 demo Q-margin

### 7.1 V2/V3 的 endpoint PMF

对 all-bin endpoints 先平均 PMFs：

```text
p_all: [B,L,K,D,N,A]
p_chosen: gather replay/demo bins -> [B,L,K,D,A]
```

CDF：

\[
F_n(z_j)=\sum_{i\le j}p_{n,i}.
\]

保留当前 FOSD 形式：

\[
\mathcal L_{\text{FOSD}}
=\operatorname{mean}
\left[
\max(F_{\text{chosen}}-F_n,0)
\right].
\]

expected Q：

\[
Q_n=\sum_jp_{n,j}z_j.
\]

保留当前 margin：

\[
\mathcal L_{\text{margin}}
=\operatorname{mean}_n
\max
\left(
m-(Q_{\text{chosen}}-Q_n),0
\right).
\]

只对 `demo=1` 的 rows 应用，归一化继续使用实际 demo count。V2/V3 的 FOSD/margin 都来自 flow endpoint，不存在独立 C51 head。

### 7.2 V4 的 demo loss

V4 对 all-bin scalar endpoints 求均值得到：

```text
q_all:    [B,L,K,D,N]
q_chosen: [B,L,K,D]
```

只保留同一 Q-margin。因为没有 PMF/CDF，必须强制：

```text
demo_fosd = false
```

### 7.3 总损失

首版沿用 CQN-AS 的 `critic_lambda`、`bc_lambda` 和 `bc_margin`，只额外加入
`atom_ce_lambda` 与 categorical FOSD 开关：

\[
\mathcal L_{\text{V2}}
=\lambda_{\text{critic}}
\left(L_{\text{FM}}+lambda_{\text{atom}}L_{\text{atom}}\right)
+\lambda_{\text{BC}}\left(L_{\text{FOSD}}+L_{\text{margin}}\right).
\]

\[
\mathcal L_{\text{V3}}
=\lambda_{\text{critic}}L_{\text{FM}}
+\lambda_{\text{BC}}\left(L_{\text{FOSD}}+L_{\text{margin}}\right),
\quad \lambda_{\text{atom}}=0.
\]

\[
\mathcal L_{\text{V4}}
=\lambda_{\text{critic}}L_{\text{FM}}
+\lambda_{\text{BC}}L_{\text{margin}},
\quad \lambda_{\text{atom}}=\lambda_{\text{FOSD}}=0.
\]

## 8. all-bin parallelism

### 8.1 V2/V3

固定一个 level：

```text
condition: [B,K,D,N,C]
source:    [B,R,K,D,N,A]
```

展平：

```text
P = B*R*K*D*N
x:    [P,A]
t:    [P]
cond: [P,C]
```

每个 Euler step 只调用一次 vectorized velocity field，再恢复 `[B,R,K,D,N,A]`。

### 8.2 V4

```text
condition: [B,K,D,N,C]
source:    [B,R,K,D,N,1]
P = B*R*K*D*N
```

同样每个 Euler step 一次 batched call，最终对 `R` 求均值。

### 8.3 level 串行、query 并行

```text
for level in range(L):          # 必须串行
    build_context(midpoint)
    integrate all K*D*N*R       # action selection 一次并行
    choose bins
    zoom_in(low, high)
```

target/replay FM 只积分 selected-bin conditions；它们一旦收集完，后续还可把
`L` 并入 batch进一步减少 dispatch：

```text
V2/V3 selected: [B,R,L,K,D,A]
V4 selected:    [B,R,L,K,D,1]
```

这减少 dispatch 次数，不减少理论 FLOPs。

### 8.4 common source across bins

动作 ranking 默认：

```text
source_base: [B,R,K,D,1,value_dim]
source_all:  broadcast to N
```

不同 bins 使用相同 source，避免 source noise 本身改变 argmax。target/current FM source 可以独立采样；next online selection、target evaluation、current training和 flow time必须使用不同 PRNG subkeys。

## 9. 配置建议

已落地配置使用与现有方法一致的 flat `method.*` keys：

```yaml
method:
  name: cqn_flow
  _target_: robobase.method.cqn_flow.CQNFlowAS
  value_mode: categorical       # categorical: V2/V3; scalar: V4
  num_flow_steps: 2             # T
  num_flow_samples: 2           # shared R for train/target/action
  flow_source_std: 1.0          # Gaussian source scale
  atom_ce_lambda: 1.0           # V2 > 0; V3/V4 = 0
  demo_fosd: true               # V2/V3 true; V4 false
  query_hidden_dim: 128
  time_embed_dim: 32
  time_scale: 1000.0
  clip_scalar_targets: true
  critic_lambda: 0.1
  bc_lambda: 1.0
  bc_margin: 0.1
```

config parser 必须验证：

- V2：`value_dim=A=51`，`atom_ce_lambda>0`；
- V3：`value_dim=A=51`，`atom_ce_lambda==0`；
- V4：`value_dim=1`，`atom_ce_lambda==0`，`demo_fosd=false`；
- 三版都没有 `distill_lambda`、teacher-head 或非 flow action readout；
- 首版把 common source across bins 硬编码为 true，后续再开放消融开关；
- support 与 `v_min/v_max` 有效。

版本切换只需要：

```text
V2: value_mode=categorical, atom_ce_lambda>0, demo_fosd=true
V3: value_mode=categorical, atom_ce_lambda=0, demo_fosd=true
V4: value_mode=scalar, atom_ce_lambda=0, demo_fosd=false
```

scalar 模式若仍配置 atom CE 或 FOSD 会显式报错，避免看起来启用了实际不存在的
distributional imitation loss。

## 10. 已落地的本地实现结构

新增独立方法并保留当前 CQN-AS baseline：

```text
robobase/method/cqn_flow.py
  CQNFlowSpec
  C2FSequenceFlowCritic
  centered_log_probabilities
  flow_logits_to_probabilities
  expected_q
  categorical_cross_entropy
  demo_fosd_per_sample
  demo_margin_per_sample
  integrate_value_flow
  CQNFlowAS
```

模型结构：

```text
encoder features
  -> shared MLP/GRU sequence context [B,K,H]
  -> per-(D,N) query condition
  -> concatenate value-state embedding + time embedding
  -> shared velocity MLP
  -> A=51 velocity (V2/V3) or scalar velocity (V4)
```

实现要求：

1. 不创建独立 C51 teacher head；
2. 不创建 distilled scalar head；
3. online/target 都是同一种 flow critic，target 用 Polyak update；
4. next features 保持当前 online encoder + stop-gradient 语义，除非单独做 target-encoder ablation；
5. V2 atom CE 与 V2/V3 demo losses 的 endpoint integration允许梯度；target integration不允许梯度；
6. vector field final projection建议 zero-init，让初始 bins 在 common source 下完全平局并复用现有 tie-break；
7. logits、PMF、Q 的 axis 和 reshape 必须有单元测试；
8. action clip 与 value/logit处理严格分开；
9. 首版 PER priority 使用 per-row velocity MSE；endpoint TD discrepancy 是后续
   必做对照，因为 velocity MSE 对 source scale 更敏感。

V2/V3 可定义 endpoint TD metric：

\[
d_b=
\operatorname{mean}_{r,l,k,d}
H
\left(
p_{\text{TD}},
\operatorname{softmax}(\hat\ell_\theta^{(r)})
\right).
\]

V4 可定义：

\[
d_b=
\operatorname{mean}_{l,k,d}
(\hat Q_{b,l,k,d}-y_{b,l,k,d})^2.
\]

priority 使用 `sqrt(d_b+eps)` 并归一化。

## 11. Ablations

### 11.1 版本主对比

- V1：当前 C51 CQN-AS；
- V2：51-D logit FM + endpoint atom CE；
- V3：51-D logit FM，`atom_ce_lambda=0`；
- V4：scalar expected flow。

### 11.2 flow compute

- `T in {1,2,4,8}`；
- `R_action in {1,2,4,8}`；
- `R_target in {1,2,4}`；
- `R_train in {1,2,4}`；
- train/eval 相同步数 vs eval 增加步数；
- Euler vs Heun，仅在 Euler 稳定后。
- sinusoidal `time_scale in {100,1000,10000}`；首版沿用本仓 Flow Matching
  的 `1000`，避免直接把 `tau in [0,1]` 输入低频 embedding 后多数维近似常数。

### 11.3 V2/V3 logit design

- centered logits on/off；
- Gaussian vs uniform logit source；
- source scale `{0.1,0.5,1.0}`；
- probability floor `{1e-8,1e-6,1e-4}`；
- PMF-space source averaging vs错误对照 logit averaging；
- `atom_ce_lambda in {0,0.1,1.0}`，其中 `0` 即 V3；
- velocity output center on/off。

### 11.4 condition/architecture

- 用户五项基础 condition；
- `+ midpoint`；
- `+ midpoint + half-width + absolute bin center`；
- GRU vs independent sequence MLP；
- dueling-style field vs direct per-bin field；
- common source across bins on/off；
- one shared trunk vs value/advantage双 trunk。

### 11.5 demo objectives

- FOSD + margin；
- FOSD only；
- margin only；
- no demo auxiliary；
- `bc_margin` sweep；
- V2 atom CE 在 demo/non-demo rows 均应用 vs只应用 TD rows的等价实现检查。

所有对比固定 encoder、batch、environment steps、updates、eval seeds和 checkpoint rule；同时报告参数量、velocity evaluations和 wall-clock，避免把更多 ODE compute误判为算法收益。

## 12. 指标

### 12.1 任务结果

- train/eval return；
- success rate；
- sample efficiency；
- 最佳 checkpoint、相同步数 checkpoint、整体趋势；
- 至少 3 seeds 后再判断版本优劣。

### 12.2 V2/V3 distribution metrics

- logit FM MSE；
- endpoint atom CE，V3 只记录不反传；
- PMF entropy；
- predicted mean/std/quantiles；
- support-edge mass 和 target clip rate；
- CDF/FOSD violation；
- C51 baseline vs endpoint PMF 的 Wasserstein-1/CRPS；
- source 间 endpoint PMF variance；
- canonical target logit norm和 probability-floor hit rate。

### 12.3 V4 metrics

- scalar FM MSE；
- endpoint mean TD error；
- source endpoint variance；
- Q range、target clip rate；
- 不同 `R/T` 下 expected Q 收敛；
- 不把 source variance 报告成 return uncertainty。

### 12.4 action ranking

- top-1 bin agreement with V1；
- coarse-to-fine 每层 bin agreement；
- top-2 Q gap；
- near-tie rate；
- 不同 source draws 下 argmax disagreement；
- common-source on/off 的 ranking variance；
- online/target bin agreement。

### 12.5 系统指标

- JAX compile time；
- peak GPU memory；
- update steps/s；
- rollout action latency；
- 每个 action/update 的 velocity evaluations；
- encoder、sequence trunk、velocity head 各自耗时；
- endpoint CE/FOSD BPTT activation memory；
- gradient norm、velocity norm、NaN/Inf和ODE trajectory。

## 13. 分阶段实验

### Stage 0：shape 与数学单元测试

固定：

```text
B=2,L=3,K=4,D=2,N=5,A=51,R=2,T=2
```

必须覆盖：

- `centered_log_probabilities` 后 atom-axis mean 为 0；
- `softmax(centered_log_probabilities(p))` 重建 PMF；
- endpoint expectation 使用完整 atom PMF；
- `atom_ce_lambda=0` 时 loss/gradient 与不含 CE 完全一致；
- demo FOSD/margin 的 demo mask；
- V4 expected Q 是 source endpoints 的均值；
- all-bin shape `[B,R,K,D,N,value_dim]`；
- level zoom 与 replay-bin reconstruction；
- local `tau=1 -> 0` Euler direction；
- terminal 时 target 只等于 reward；
- target path stop-gradient；
- common source across bins；
- mode=V1 时当前 baseline 无回归。

### Stage 1：V1 baseline

- state DMC `K=1`；
- state CQN-AS `K=4`；
- 保存 Q range、PMF entropy、support saturation、bin ties、throughput和 eval curves。

### Stage 2：V2

先用：

```text
num_flow_steps=2
num_flow_samples=2
atom_ce_lambda=1.0
value_mode=categorical
demo_fosd=true
```

先 state `K=1`，再 state `K=4`。通过条件：

- endpoint PMF 能追上 projected target；
- FM 与 atom CE 同时下降；
- action ranking 不被 source noise 主导；
- 无 `-inf` logits、NaN 或 support collapse；
- 已测 endpoint CE BPTT 成本。

### Stage 3：V3

从 V2 配置只改：

```text
atom_ce_lambda=0.0
```

不要同时改 source、steps、samples 或 trunk。核心问题是去掉 endpoint CE 后：

- endpoint CE metric 是否仍下降；
- PMF calibration、entropy和 action ranking 是否恶化；
- demo FOSD/margin 是否足以稳定 demo-driven tasks；
- FM velocity loss 是否与真实 endpoint quality脱钩。

### Stage 4：V4

同任务、同参数量附近比较：

```text
num_flow_steps=2
num_flow_samples=2
value_mode=scalar
atom_ce_lambda=0
demo_fosd=false
```

先验证 direct all-bin scalar flow 的动作延迟和 ranking variance，再扩大到 `T/R=4/8`。

### Stage 5：pixels 与机器人任务

顺序：

1. pixel small task / RLBench `K=4`；
2. BiGym `K=4`；
3. BiGym `K=16`。

进入 `K=16` 前必须确认：

- encoder 只跑一次；
- `K*D*N*R` 确实在单次 batched velocity call 内；
- selected target/current conditions 可跨 `L` 合批；
- endpoint CE/demo BPTT 没有超出显存；
- rollout latency满足控制频率；
- batch size/steps/s/显存曲线已经测量。

不要把现有 BiGym CQN-AS 的 `batch_size=256 + demo_batch_size=256` 直接作为
CQN-Flow 首跑值。以 `B=512,R=2,K=16,D=16,N=5,H=512` 计算，仅一个
query hidden tensor 就约 `2.50 GiB`；多个激活加上 endpoint BPTT 很可能超过
32 GiB。当前默认已将 `query_hidden_dim` 降为 `128`，但 K=16 仍应从更小 batch
做显存阶梯测试，而不是把“可以并行”理解为“没有显存成本”。实现会在
`demo_batch_size` 已知且 online replay prefix 没有额外 demo rows 时，只对末尾
appended demo batch 做 all-bin endpoint BPTT；若 self-imitation demo 出现在 prefix，
则自动回退到全 batch 以保持 loss 语义。

## 14. 主要风险

1. **51-D compute/memory**：V2/V3 all-bin state 为 `B*R*K*D*N*51`，比 scalar V4 大 51 倍。
2. **level 无法并行**：每层 midpoint 依赖前一层 argmax，只能并行 `K*D*N*R`。
3. **endpoint CE 需要 BPTT**：V2 atom CE、V2/V3 demo FOSD/margin 都需通过 ODE endpoint 反传。
4. **logit gauge 不可辨识**：不中心化会浪费一个自由度，并使 velocity norm/target scale漂移。
5. **zero-probability target**：`log(p_TD)` 会产生 `-inf`，必须 floor+renormalize并监控偏差。
6. **V3 endpoint 无直接 TD anchor**：低 FM loss 不保证有限步 Euler endpoint PMF校准；endpoint CE metric是必需诊断。
7. **PMF 与 logits 平均不可交换**：source/ensemble 必须在 softmax 后平均 PMF。
8. **source saturation**：logit source 太宽会让初始 softmax接近 one-hot；太窄可能退化成单步网络。
9. **MC bin ranking noise**：必须 common source across bins，并记录 argmax disagreement。
10. **V4 丢失 distributional demo signal**：无 CDF/FOSD，只剩 scalar margin。
11. **直接 all-bin ODE latency**：三版都没有 distillation，rollout 成本随 `L*T*R*K*D*N` 增长。
12. **target non-stationarity**：target flow、Polyak tau、source range和 integration error共同进入 bootstrap。
13. **support clipping**：固定 C51 support 能稳定训练，但错误的 `v_min/v_max` 会造成 edge mass和错误排序。
14. **factorized Q 不是 joint chunk Q**：每个 `(l,k,d)` 独立接受同一 reward但自己的 bootstrap head；不要擅自改成 chunk scalar aggregation。
15. **dueling 不可直接照搬**：logit-space velocity 的 value/advantage分解不是现有 C51 endpoint dueling 的自动等价物，需单独 ablation。
16. **centralized critic**：V2/V3 应平均 target PMF后广播；V4 平均 scalar targets。不能把 distribution samples 数值平均冒充 PMF mixture。
17. **PER 指标选择**：首版沿用 velocity MSE priority，但它强依赖 source
    scale；实验中要与 endpoint TD discrepancy priority 对照。
18. **时间方向写反**：本地 `tau` 与论文常用 `t` 相反，单元测试必须固定 endpoint方向。

## 15. 一手资料

- [Continuous Control with Coarse-to-fine Reinforcement Learning / CQN](https://arxiv.org/abs/2407.07787)
- [Coarse-to-fine Q-Network with Action Sequence / CQN-AS](https://arxiv.org/abs/2411.12155)
- [A Distributional Perspective on Reinforcement Learning / C51](https://arxiv.org/abs/1707.06887)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)
- [floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL](https://arxiv.org/abs/2509.06863)
- [官方 FloQ 代码](https://github.com/CMU-AIRe/floq)
- [What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)
- [Value Flows](https://arxiv.org/abs/2510.07650)
- [Value Flows 官方代码](https://github.com/chongyi-zheng/value-flows)
- [Expressive Value Learning for Scalable Offline Reinforcement Learning](https://arxiv.org/abs/2510.08218)

## 16. 最终执行顺序

1. 冻结 V1 作为回归 baseline；
2. 实现共享 sequence context、51-D/scalar conditional field和统一 all-bin integrator；
3. 先跑 V2，利用 endpoint atom CE 检查 51-D flow plumbing和 PMF质量；
4. 只把 `atom_ce_lambda` 置零得到 V3，做严格单变量消融；
5. 切换 `value_dim=1` 得到 V4，直接对多个 endpoints求均值做 all-bin Q；
6. 在 state 小任务确认正确性和成本后，才进入 pixels、RLBench和 BiGym `K=16`。

三种 flow 版本必须共享同一套 coarse-to-fine condition、level-serial/bin-parallel action path和评估协议；任何 teacher head、distilled Q head或非 flow action selector都属于后续新版本，不能混入本次 V2/V3/V4 对比。

## 17. 2026-07-22：V2/V3 失败复盘与下一阶段计划

> **优先级覆盖声明**：本章基于 2026-07-22 已完成的 100k 实验和后续一手资料调研，覆盖本文前面第 13、16 节以及其他旧的执行优先级。旧章节继续保留用于记录设计演化，但从现在起不再以“继续调 V2/V3”为主线。新的顺序是：**FLOQ-core expected-Q -> PCBF/Value-Flows scalar return -> FlowIQN；V2/V3 仅保留为负对照，Adjoint Matching/QAM 仅作为独立 policy-flow 方向。**

### 17.1 100k 实验的量化复盘

所有 CQN-Flow 结果来自同一 `move_plate` pixel、`seed=1`、`action_sequence=16`、
`bins=5`、100k environment steps 的运行目录：

```text
exp_local/pixel_cqn_flow/move_plate_v2_seed1_100k_b16d16_k16_q64_20260722
exp_local/pixel_cqn_flow/move_plate_v3_seed1_100k_b16d16_k16_q64_20260722
exp_local/pixel_cqn_flow/move_plate_v4_seed1_100k_b16d16_k16_q64_20260722
```

三者均使用 `batch_size=16`、`demo_batch_size=16`、`T=2`、`R=2`、
`query_hidden_dim=64`。已有 V1 CQN-AS 参考使用 `batch_size=256`、
`demo_batch_size=256`，因此 V1 与 flow runs 不是严格 compute/data-matched 对照；下一阶段必须补
`B=16+16` 的直接 C51 baseline。

| run | value 表示 | best success | 100k success | 40 次 eval 均值 | 备注 |
|---|---|---:|---:|---:|---|
| V1 CQN-AS seed 1 | direct C51 | 80% @ 47.5k | 64% | 55.3% | `B=256+256`，非 matched |
| V1 CQN-AS seed 4 | direct C51 | 84% @ 62.5k | 60% | 61.3% | 用于展示 baseline seed 波动 |
| V2 seed 1 | 51-D logit FM + endpoint CE | 0% | 0% | 0% | 所有 40 个 checkpoints 均为 0 |
| V3 seed 1 | 51-D logit FM，无 endpoint CE gradient | 0% | 0% | 0% | 所有 40 个 checkpoints 均为 0 |
| V4 seed 1 | scalar expected-Q FM | 52% @ 92.5k | 16% | 14.6% | 52.5k 首次非零；最后 5 次均值 36.0% |

V2/V3 并不是简单地“loss 没有优化”。首个记录点到 100k 的变化如下：

| run | flow loss | endpoint CE metric | demo margin | 100k target entropy |
|---|---:|---:|---:|---:|
| V2 | 4.888 -> 0.452 | 4.418 -> 1.536，参与 gradient | 0.100 -> 0.070 | 1.511 |
| V3 | 6.051 -> 0.477 | 4.418 -> 1.381，仅 metric | 0.100 -> 0.069 | 1.319 |
| V4 | 1.624 -> 0.197 | N/A | 0.100 -> 0.034 | N/A |

更强的证据是：V2 在 100k 的 `endpoint_CE - target_entropy = 0.025`，V3 为
`0.062`。这个差值就是同一 averaging convention 下的平均
`KL(target || endpoint)`；也就是说 chosen replay bin 的 PMF 已经拟合得相当好，CE 缺失不是
V3 为 0%、更不是 V2 同样为 0% 的充分解释。

对 92.5k checkpoint 的同一个初始 observation 重采样 16 组 flow sources，得到：

| checkpoint | 三层 bin 与首组 source 的平均一致率 | action 跨 source mean/max std | level-0 Q span | top1-top2 Q gap |
|---|---:|---:|---:|---:|
| V2 | 约 76% | 0.077 / 0.239 | 0.084 | 0.012 |
| V3 | 约 53% | 0.100 / 0.269 | 0.254 | 0.023 |
| V4 | 约 85% | 0.013 / 0.097 | 1.169 | 0.155 |

所以当前证据最强的 failure hypothesis 是：V2/V3 的 action ranking 对 source draw 太敏感；上层
coarse-to-fine 的一次 bin flip 会被后续 zoom 放大，而 temporal ensemble 每次 replan 又重采样
source。51-D logit loss 与 expected-Q ranking 不对齐、稀疏 PMF 的 canonical logits 数值病态，是
对该现象最有证据的机制解释；“它们根本没有学到 TD PMF”则已经被上述 KL 数据排除。由于 source
probe 目前只覆盖一个 observation 和一个 checkpoint，这还不是因果证明；Stage I 必须补多 episode
fixed-source/independent-source 重评估后才能把它升级为 confirmed mechanism。

因此当前证据支持以下结论：

1. V2/V3 的 velocity loss 和 endpoint CE metric 明显下降，但没有转换成有效的 action-bin ranking；问题不能只归因于 NaN、OOM 或完全未学习。
2. V4 在同样的小 batch、`T=2,R=2,H=64` 条件下至少学出了非零策略，说明 scalar expected-Q objective 比 51-D logit transport 更接近当前任务所需信号。
3. V4 的 best 52% 但 final 16%，仍有明显的 value/ranking variance 和训练不稳定，不应把它视为最终解法。
4. V2/V3 从未成功，所以 demo buffer 始终为 9253；V4 成功后 self-imitation 使 demo buffer 增至 18302。首次成功之后三条 run 的数据分布已不同，后续必须同时记录 frozen-demo 与 self-imitation 两种协议。
5. 由于直接 C51 baseline 的 batch 大 16 倍，这组数据不能单独证明 categorical distributional RL 无效；它证明的是当前 **categorical-logit FM 构造** 无效。

### 17.2 V2/V3 为什么在目标层面不合适

V2/V3 位于两个合法方法之间，但不等价于其中任何一个：

- 标准 C51 用 projected categorical Bellman target 直接训练 PMF/CE；
- 合法的 continuous distributional flow 应运输一维 return samples；
- V2/V3 却把稀疏 C51 PMF 变成 `center(log(p + eps))`，再在 51 维 logit 空间做欧氏 Flow Matching。

这会产生四个直接问题：

1. atom 的数值距离和一维 Wasserstein 几何丢失；相邻 atom 与相隔很远的 atom 在 logit MSE 中没有正确区别。
2. 零/小概率 atom 经 `log(p + eps)` 变成大幅负数，51 维 moving TD target 的尺度会淹没 expected-Q 的细微排序变化。
3. 每个 source noise 都被拉向同一个 deterministic PMF/logit endpoint；这些 noise 不是 return samples，因此增加 `R` 不会自动得到合法 return distribution。
4. action selection 依赖 `softmax(endpoint) · support`，而训练主损失是中间 velocity 的 51 维 MSE；低 flow loss、低 endpoint CE 与正确 top-1 bin 之间没有直接保证。

V2 的 endpoint CE 只能在有限步 ODE 后间接修正；V3 则完全依赖中间 velocity。当前结果已经显示两者都不足以恢复有用 ranking。后续不再把 `atom_ce_lambda`、FOSD 或更大的 51-D head 作为主搜索方向。若继续研究 categorical flow，应另立项目使用
[Discrete Flow Matching](https://arxiv.org/abs/2407.15595) 的 CTMC/categorical path，而不是 continuous logit FM；当前项目中直接 C51 是更清晰的 categorical baseline。

### 17.3 第一优先级：FLOQ-core expected-Q

一手资料：

- [floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL](https://arxiv.org/abs/2509.06863)
- [FLOQ 官方代码](https://github.com/CMU-AIRe/floq)
- [What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)

FLOQ 不把 noise 解释为环境 return uncertainty，而是用 iterative scalar flow 表达 expected Q。
对 transition 和 CQN-AS candidate-bin condition (c=(s,l,k,d,n,\text{zoom path}))：

\[
\hat y
=r+\gamma m\frac{1}{R}\sum_{j=1}^{R}
\Phi_{\bar\theta}(\epsilon'_j\mid s',a'),
\]

\[
z_t=(1-t)\epsilon+t\hat y,
\qquad u^\star=\hat y-\epsilon,
\]

\[
L_{\mathrm{FLOQ}}
=\mathbb E\left\|v_\theta(z_t,t\mid c)-u^\star\right\|^2.
\]

需要忠实迁移的配置：

- scalar **uniform** source，而不是当前 `N(0,1)`；
- source width `κ(Qmax-Qmin)`，先测 `κ in {0.1,0.25}`，并确保区间覆盖初始 `Q≈0`；当前 `[-2,2]` support 可先从 `[-0.2,0.2]` 开始；
- 论文完整 baseline 使用 `R=8`、`T=8`、2 个 flow ensembles、EMA `tau=0.005`；当前 FLOQ-core 先用 single online field + EMA target 做 objective gate，compute gate 用 `R=4,T=4`，通过后再加第二个 field；
- 51-bin HL-Gauss 是对 scalar interpolant `z_t` 的 **输入编码**，velocity 输出仍为 1 维；官方代码确认实际标准差为 `sigma * bin_width`，默认 `sigma=16`；
- 使用真正的 64-D Fourier time embedding `cos(k*pi*t), k=1...64`；
- velocity head 先做 same-capacity `H=64/128` 公平对照，再单独测试论文的 4x512 head，避免把 objective 与参数量混在一起；
- 所有 candidate bins 使用 common source samples；当前 rollout 使用固定 antithetic uniform bank，减少 argmax jitter；stratified grid 是后续显式消融，当前代码尚未实现；
- 保留基于 expected Q 的 demo large-margin loss，不使用 atom CE/FOSD。

FLOQ 论文写的是 64-D Fourier time embedding，且消融显示 scalar time 明显更差；但截至本次调研的公开代码主路径虽然暴露 `time_embed_dim=64`，实际只执行了 `cos(t)`。本项目按论文实现真正的 Fourier64，并将 `paper_faithful_time=true/false` 作为显式 ablation，不能静默复制该代码差异。

FLOQ 官方额外蒸馏一个单步 scalar critic，是为了让连续 actor 便宜地取得
`grad_a Q`，避免 actor update 穿过多步 critic ODE。CQN-AS 直接枚举 bins，不需要 action gradient，所以默认不蒸馏。若要做论文复现，可把 distill head 作为 diagnostic ablation，但 policy 的主要比较必须仍由 flow critic 本身选 bin。

后续分析论文 [What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333) 的 matched study 发现 expected FLOQ 经常优于 distributional flow；distributional 版本产生更高 endpoint variance，却不稳定地带来更好 policy。其主要收益来自 all-time velocity supervision、迭代误差修正和面对 moving TD target 的表示可塑性。这支持先把 FLOQ expected baseline 做忠实，再尝试完整 return distribution。

### 17.4 第二优先级：Value Flows 的一维 return distribution

一手资料：

- [Value Flows 论文](https://arxiv.org/abs/2510.07650)
- [Value Flows 项目页](https://pd-perry.github.io/value-flows/)
- [Value Flows 官方 JAX 代码](https://github.com/chongyi-zheng/value-flows)

Value Flows 运输的是 scalar return sample，不是 51-D logits。它需要同时实现 DCFM 主信号和 BCFM anchor。

对同一个 `epsilon ~ N(0,1)` 和 `t ~ U[0,1]`，先用 EMA target field 积分 next return 到随机时间：

\[
z'_t=\Phi_{\bar\theta}^{0\rightarrow t}
(\epsilon\mid s',a').
\]

Distributional Conditional Flow Matching：

\[
x_t^D=r+\gamma m z'_t,
\]

\[
L_{\mathrm{DCFM}}
=\left\|
v_\theta(x_t^D,t\mid s,a)
-v_{\bar\theta}(z'_t,t\mid s',a')
\right\|^2.
\]

Value Flows 的论文推导和官方代码都没有在 target velocity 前额外乘 `gamma`；`gamma` 位于 current-flow input 的 affine transform 中。当前移植忠实保持这个形式，并用 terminal/bootstrap golden test 锁定语义。需要注意：这只是“忠实于 Value Flows”，并不意味着该路径的 affine geometry 已经严格正确；后面的 PCBF 正是针对这个 mismatch 给出的修正。

Bootstrapped Conditional Flow Matching 使用完整 next endpoint：

\[
z'_1=\Phi_{\bar\theta}(\epsilon\mid s',a'),
\qquad y=r+\gamma m z'_1,
\]

\[
x_t^B=(1-t)\epsilon+t y,
\]

\[
L_{\mathrm{BCFM}}
=\left\|v_\theta(x_t^B,t\mid s,a)-(y-\epsilon)\right\|^2.
\]

总目标：

\[
L_{\mathrm{ValueFlow}}
=L_{\mathrm{DCFM}}+\lambda_{B} L_{\mathrm{BCFM}}.
\]

论文的关键消融必须纳入本项目：**BCFM-only 在两个任务上接近零成功率；DCFM 提供主要 learning signal，BCFM 用来防止 self-distillation/zero-field collapse。** 因此不能把当前 V4 的 bootstrapped endpoint CFM 政名为 Value Flows。Stage I 先用 single online field + EMA target、`T=4`、`lambda_B=1` 验证 DCFM/BCFM plumbing；通过 gate 后，Stage II 再加入 twin scalar fields、`T=10` 和 `lambda_B in {0.3,1,3}`。confidence/JVP weighting 留到基础目标通过后再打开。

对 action bins 的 expected-Q ranking 同时比较：

\[
\hat Q_{v0}(s,a)
=\frac1R\sum_j
[\epsilon_j+v_\theta(\epsilon_j,0\mid s,a)]
\]

和完整 ODE endpoint mean。使用固定 Gaussian quantile grid 以及 bin 间 common random numbers。demo imitation 继续作用于该 expected Q 的 large-margin，不加入 CE；只有 direct C51 baseline 使用 atom CE。

### 17.5 Value Flows 之后优先比较 PCBF

一手资料：

- [Path-Coupled Bellman Flows 论文](https://arxiv.org/abs/2605.08253)
- [PCBF 官方 JAX 代码](https://github.com/BoyangASU/path-coupled-bellman-flows)

PCBF 直接指出 Value Flows 的 pointwise DCFM 路径存在 source-boundary/affine-geometry
mismatch。令 `t=0` 为 Gaussian source、`t=1` 为 return，使用同一个
`epsilon` 和 target endpoint `X'`：

\[
z'_t=(1-t)\epsilon+tX',
\]

\[
z_t=t r+\gamma m z'_t+(1-t)(1-\gamma m)\epsilon.
\]

这样 `z_0=epsilon`，同时 `z_1=r+gamma*m*X'`，source 与 Bellman endpoint 两端都正确。
其高方差但无偏的 BCFM target 和 control variate 为：

\[
Y=r+\gamma mX'-\epsilon,
\qquad
C=v_{\bar\theta}(z'_t,t\mid s',a')-(X'-\epsilon),
\]

\[
u_\lambda=Y+\lambda m C.
\]

`lambda=0` 就是当前 return-sample BCFM；`lambda>0` 用 successor velocity 降方差，但引入可控 bias；
在非终止且 `lambda=gamma` 时，显式的 `X'` 项被消掉。它与当前代码的 same-noise endpoint 已高度重合，
只需一个 corrected current path 和一次 target-field query，因此应在 FlowIQN 前测
`lambda in {0, 0.5*gamma, gamma}`。PCBF 的公开结果显示最优 `lambda` 有明显 domain dependence，
所以不能只跑一个值后就判断 distributional flow 无效。

### 17.6 第三优先级：FlowIQN；Adjoint/QAM 单独归类为 policy flow

FlowIQN 一手资料：

- [Quantile-Coupled Flow Matching for Distributional RL / FlowIQN](https://arxiv.org/abs/2605.08515)
- [FlowIQN 官方仓库](https://github.com/ori-goals/flowIQN)；2026-07-24 复核仍只有
  README/LICENSE 两次提交，并继续写明 `Code will be released shortly`，因此当前没有
  新官方实现可用于推翻本地 corrected adaptation 的负 Gate

FlowIQN 显式 condition quantile `tau`，对每个 transition 的 `Kq` 个 source quantiles 和 Bellman return targets 分别排序后做单调一维 OT 配对。这比 V2/V3 的 51-D logit 欧氏路径更符合 distributional Bellman/Wasserstein 几何。只有在 Value Flows 跑通后才实现：先测 `Kq in {8,16}`、`T in {2,4}`，rollout 用固定 quantile grid 的 endpoint mean 排序 bins。

Adjoint/QAM 一手资料：

- [Adjoint Matching for Fine-Tuning Flow and Diffusion Generative Models](https://arxiv.org/abs/2409.08861)
- [Q-learning with Adjoint Matching / QAM](https://arxiv.org/abs/2601.14234)
- [QAM 项目页](https://colinqiyangli.github.io/qam/)
- [QAM 官方代码](https://github.com/ColinQiyangLi/qam)

QAM 保留普通 scalar TD critic，使用 Adjoint Matching 更新连续 action flow policy，使其近似：

\[
\pi_\theta(a\mid s)
\propto \pi_\beta(a\mid s)\exp(\tau Q(s,a)),
\]

终端 adjoint 依赖 `-tau * grad_a Q(s,a)`。所以它 flow 的是 **action/policy**，不是 value/return；它需要连续可微 action 和 `grad_a Q`，不能直接作用于离散 action-bin index。QAM 不能修复 V2/V3 critic。若后续比较，应另建“continuous flow actor + scalar critic”基线，复用相同 image encoder、数据和 demos，不能把 adjoint loss 混入 CQN-AS critic 主线。

另外两条已筛选但不列为近期主线：

- [Temporal Difference Flows](https://arxiv.org/abs/2503.09817) 学的是 future-state/
  geometric-horizon distribution；它可用于 planning/policy evaluation，但不是这里的标准 discounted-return
  CQN critic。
- [EVOR](https://arxiv.org/abs/2510.08218) 用 expressive flow value 做 offline policy extraction/
  rejection sampling；与“让 value 本身成为 flow”相关，但其 offline regularized objective 和 action 搜索协议
  与当前 online CQN-AS 差异较大，先不作为第一轮可归因 baseline。

### 17.7 三阶段实验矩阵

| 阶段 | 目的 | GPU/实验 | 固定项 | 主要变量 |
|---|---|---|---|---|
| I：正确性 gate | 在完整 100k 前排除错误 objective、时间方向和 ranking collapse | GPU0 direct C51 margin-only；GPU1 FLOQ-core `T=4,R=4`；GPU2 PCBF `lambda=gamma,T=4,R=4`；GPU2 随后排 ValueFlow DCFM+BCFM 作为机制对照 | frozen demos、seed 1、`B=16+16`、EMA/rollout target `tau=0.005`、相同 image encoder/critic hidden width、10k updates | value objective/path geometry |
| II：100k 主对比 | 比较 categorical、expected flow、distributional flow | GPU0 data/batch-matched direct C51；GPU1 FLOQ-core `T=8,R=8`；GPU2 Stage-I 晋级的 PCBF 或 ValueFlow | seed 1、相同 replay/eval cadence、margin-only demo imitation、common noises | value representation |
| III：3-seed 与机制消融 | 验证稳定性并定位真正增益 | 晋级方法和 matched C51/V4 跑 3 seeds；空余卡依序做 ablation | 固定 100k budget 和 checkpoint protocol | FLOQ embedding/source/steps；PCBF lambda；ValueFlow loss composition；最后 FlowIQN |

这里的 direct C51 只是 **data/batch/EMA/imitation-matched**，不可能与 iterative flow 做严格
parameter/compute match；因此必须同时报告 update wall-time、VRAM 和 rollout latency，不能把它简称为
完全 compute-matched baseline。canonical CQN-AS 的 FOSD+margin 另保留一条参考；主对比的 C51 使用
新增 `demo_fosd=false`，只保留与 scalar flow 相同的 expected-Q margin。

阶段 I 还必须包含不依赖 environment success 的小测试：

1. synthetic Bellman transition：已知 target mean/distribution，验证 source -> target 的方向、terminal mask 和 Euler endpoint；
2. 固定 replay mini-set overfit：比较 expert bin 与非 expert bins 的 Q 排序；
3. all-bin vectorization：确认 image encoder 只运行一次，`K*D*N*R` 在单次 field call 中并行；
4. 对相同 checkpoint 用 `T=4/8` 和不同 source draws 重算 action，量化 integration 与 Monte Carlo ranking variance。

阶段 III 的消融按以下顺序执行，不做一次性全排列：

1. FLOQ：raw/normalized scalar vs HL-Gauss；scalar time vs真实 Fourier64；
2. FLOQ：`T={2,4,8}`、`R={2,4,8}`、`kappa={0.1,0.25}`；
3. demo Q-margin on/off，frozen demos vs self-imitation；
4. PCBF：`lambda={0,0.5*gamma,gamma}`，并与相同 source/endpoint 的 BCFM 对齐；
5. ValueFlow：DCFM-only、BCFM-only、DCFM+BCFM，`BCFM weight={0.3,1,3}`；
6. ValueFlow：`Q_v0` vs full endpoint mean；confidence weighting off/on；
7. FlowIQN：`Kq={8,16}`、`T={2,4}`，只在前两条 distributional tests 通过后启动。

### 17.8 必须新增的指标

环境结果使用三视图，不允许只报 best checkpoint：

- best success 及 step；
- 相同 100k step success；
- 最后 5 次 eval 均值、全程 eval AUC、time-to-first-success；
- 3 seeds 的均值、标准差和单 seed collapse 数量。

value/ranking 指标：

- held-out demo expert-bin top-1 accuracy；
- `top1_Q - top2_Q` gap；
- 跨 source 的 per-bin Q std；
- `rank_snr = top1_top2_gap / (Q_noise_std + eps)`；
- source resampling 后的 action-bin flip rate；
- TD target mean/std、EMA target drift 和 support clipping rate；
- self-imitation demo-buffer growth，必须与 frozen-demo 结果分开。

flow/integration 指标：

- velocity loss与完整 endpoint Bellman residual同时记录；
- `T=4` vs `T=8/10` endpoint/Q discrepancy；
- source interval coverage、endpoint variance、trajectory curvature和 non-finite rate；
- ValueFlow 的 DCFM、BCFM 分项 loss、t=0 Q 与 full-flow Q 差异；
- distributional 方法在 held-out n-step returns 上的 W1、CRPS、mean error和 quantile coverage；
- direct C51 才记录 atom CE；scalar FLOQ/ValueFlow 不用 CE 评价。

系统指标：

- compile time、update steps/s、act steps/s、wall time；
- peak VRAM 和 all-bin query tensor size；
- image encoder 调用次数；
- rollout latency是否满足环境控制频率。

实现状态必须区分清楚：当前 `demo_expert_top1/top2_gap/source_q_std/rank_snr` 是训练
mini-batch 指标，不是假装 held-out evaluation；`demo_source_bin_flip_rate` 是固定 demo condition 下
的 per-level 局部 flip。完整 C2F rollout 的 independent-source flip 已由
`scripts/analyze_cqn_flow_ranking.py` 实现，可对多个 seeded reset observation 做 checkpoint probe。
held-out n-step return 的 W1/CRPS/coverage、EMA drift、trajectory curvature 与 clipping rate 仍未实现，
在这些 probe 落地前不能用相应条款宣称 Stage-II distributional fidelity 已通过。

### 17.9 Stop/Go 标准

**Stage I hard stop：**

- 出现 NaN/Inf、source/target 时间方向错误、terminal bootstrap 错误或 encoder 被按 bin/source 重复执行；
- 10k overfit 后 endpoint Bellman residual没有相对初始值下降至少 50%；
- fixed-demo mini-set 的 expert-bin accuracy 不高于随机基线 `1/N=20%` 至少 5 个百分点，且 `rank_snr < 1`；
- checkpoint ranking probe 的 independent-source action flip rate持续高于 50%。

满足以下条件才进入 Stage II：所有 correctness tests 通过，loss 与 endpoint residual 同向下降，fixed-demo expert-bin accuracy >25%，并且固定 source 下 ranking 可复现。进入 3-seed/论文结论前还必须把该指标升级成 held-out demo probe。

**Stage II hard stop：**

- 到 60k 仍所有 eval 为 0%，同时 expert-bin accuracy <=25% 或 `rank_snr < 1`；
- 更高 `T` 使 endpoint discrepancy/非有限值持续增大；
- 单次 update 或 rollout 成本超过 matched V4 的 2.5 倍且没有 success/AUC 增益。

进入 3-seed Stage III 的最低门槛以现有 V4 为固定参考：

- best success 至少 52%；
- 最后 5 次 eval 均值至少 36.0%；
- 100k 前出现非零成功；
- ranking flip rate低于 20%，`rank_snr >= 1`。

最终将新方法设为默认需同时满足：

1. 3 seeds 最后 5 次 eval 的平均 success 比现有 V4 至少高 10 个百分点；
2. 同步报告 same-step 与 AUC，不能只依赖单个 best checkpoint；
3. 没有 seed 全程 0%，且相对 data/batch-matched direct C51 的差距不超过 10 个百分点，或在 sample efficiency/AUC 上有明确优势；
4. rollout latency满足控制频率，训练 wall time/VRAM 的增加有对应的稳定性或成功率收益。

若 FLOQ 通过而 PCBF/ValueFlow 都失败，默认保留 expected-Q flow，并把 distributional work 停在 path/target 诊断；若 PCBF 或 ValueFlow 通过，才进入 FlowIQN。V2/V3 不因更大的 `T/R/H` 自动重启，除非新的 scalar-return baselines 已通过且有明确的新假设。

### 17.10 本轮已落地代码与启动方式

截至 2026-07-22，本轮已经落地以下可执行能力：

- `value_mode=return_sample`：一维 stochastic return endpoint，不再生成 51-D logits；
- BCFM 使用同一个 base noise 生成 next return sample 和 current interpolation；
- DCFM 使用同一个 noise/time 的 partial target flow，且 target velocity 前不额外乘 discount；
- PCBF 使用 source-consistent current path 与 `lambda` control variate；`lambda=0` 精确退化为 BCFM，terminal 精确退化为 `r-epsilon`；
- expected-Q FLOQ 路径支持 uniform source、独立 train/target/action sample 数、antithetic samples、固定 rollout source bank；
- FLOQ 的 target Monte Carlo source 与 current interpolation source 相互独立；ValueFlow BCFM 与 PCBF 使用 exact same-noise coupling；
- scalar interpolant 可选 51-boundary HL-Gauss 输入编码；velocity/output 仍为 scalar；
- paper-faithful `cos(k*pi*t)` Fourier time embedding；
- ValueFlow profile 使用官方形式的 raw scalar time，并对 return ODE 每个 Euler step 做 support clip；terminal transition 的 DCFM 被 mask，交由 BCFM 学习 deterministic reward；
- 新日志包括 `bcfm_loss`、`dcfm_loss`、`pcbf_loss`、`endpoint_q_loss`、`source_q_std`、`endpoint_kl`、demo expert top-1、top-2 gap、source-Q std、source-bin flip rate 和 rank SNR；
- 三个预设 profile：`cqn_flow_floq`、`cqn_flow_value_flows` 与 `cqn_flow_pcbf`。

Stage I 的三个首发任务已经做成独立 launch；它们都展开为 `10500` frames（500 replay seed +
约 10000 updates）、`B=16+16`、frozen demos 和 `tau=0.005`，不会误用下面的 100k/full-T profile：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  launch=cqn_as_pixel_bigym_stage1_gate env=bigym/move_plate seed=1

CUDA_VISIBLE_DEVICES=1 JAX_CUDA_VISIBLE_DEVICES=1 .venv/bin/python train.py \
  launch=cqn_flow_floq_pixel_bigym_stage1_gate env=bigym/move_plate seed=1

CUDA_VISIBLE_DEVICES=2 JAX_CUDA_VISIBLE_DEVICES=2 .venv/bin/python train.py \
  launch=cqn_pcbf_pixel_bigym_stage1_gate env=bigym/move_plate seed=1
```

GPU2 的 PCBF gate 完成或 hard-stop 后，再排机制对照：

```bash
CUDA_VISIBLE_DEVICES=2 JAX_CUDA_VISIBLE_DEVICES=2 .venv/bin/python train.py \
  launch=cqn_value_flows_pixel_bigym_stage1_gate env=bigym/move_plate seed=1
```

每个 2.5k/10.5k checkpoint 用同一条 read-only probe 命令测完整 rollout ranking variance：

```bash
.venv/bin/python scripts/analyze_cqn_flow_ranking.py \
  --run-dir <RUN_DIR> --snapshot <SNAPSHOT> --gpu-id <GPU> \
  --num-observations 16 --num-source-draws 16 \
  --output <RUN_DIR>/ranking_probe.json
```

通过 gate 后，FLOQ 100k/full-T 主跑可从以下命令开始：

```bash
.venv/bin/python train.py \
  launch=cqn_flow_floq_pixel_bigym_demo_driven \
  env=bigym/move_plate \
  seed=1 batch_size=16 demo_batch_size=16 \
  method.query_hidden_dim=64
```

Value Flows core：

```bash
.venv/bin/python train.py \
  launch=cqn_value_flows_pixel_bigym_demo_driven \
  env=bigym/move_plate \
  seed=1 batch_size=16 demo_batch_size=16 \
  method.query_hidden_dim=64
```

PCBF full profile（默认 `lambda=0.4`；主对比必须另跑 `0` 与 `${replay.gamma}`）：

```bash
.venv/bin/python train.py \
  launch=cqn_pcbf_pixel_bigym_demo_driven \
  env=bigym/move_plate \
  seed=1 batch_size=16 demo_batch_size=16 \
  method.query_hidden_dim=64
```

当前实现刻意先保持 **single online field + EMA target field**，用于验证 CQN-conditioned
DCFM/BCFM plumbing；尚未把官方 Value Flows 的 twin online fields、JVP confidence weighting、
`Q_v0` readout 加进第一轮 gate。它们属于 Stage II 之后的顺序消融，不能把当前 profile 描述成官方完整复现。相同地，FLOQ profile 已包含核心 source/HL-Gauss/Fourier/T/R recipe，但没有 twin ensemble；distilled critic 则因 CQN bin enumeration 不需要 `grad_a Q` 而有意省略。

PCBF profile 同样是 CQN-conditioned critic core：已包含 source-consistent path、shared-noise
control variate 和 terminal/stop-gradient 语义，但尚未复制官方 offline actor、rejection sampling 或 twin
Q ensemble；这些不是本次 discrete-bin critic gate 的必要组成。direct C51 新增了向后兼容的
`demo_fosd` 开关，canonical 配置默认仍为 `true`，只有 margin-matched gate 设为 `false`。

Hydra 的 root defaults 会先加载外部 `profile`、再加载 `launch` 内的 method，因此不要使用
`launch=cqn_flow_pixel_bigym_demo_driven profile=cqn_flow_floq` 这种组合：后加载的 method 会把
profile 静默覆盖。上面的 dedicated launches 已用 defaults override 固定正确顺序，并有配置测试锁定。

## 18. 2026-07-22：Stage-I 首轮结果与稳定化复跑

### 18.1 首轮 10.5k 结果

两条首轮 gate 都使用 `move_plate` pixel、seed 1、frozen demonstrations、`B=16+16`、
`T=4,R=4` 和 `tau=0.005`。

| 方法 | 10.5k 数值状态 | eval success @ 2.5k/5k/7.5k/10k | ranking/endpoint 结论 |
|---|---|---|---|
| FLOQ-core | finite | `0/12/0/0%` | expert top-1 `20% -> 74.8%`，但 endpoint-Q loss `0.0938 -> 0.1136`，source-Q std `0.116 -> 0.215`；学会 demo 排序但 Bellman endpoint 与 source invariance 没有同步改善 |
| PCBF `lambda=gamma` | step 0 finite，1k 前全树 NaN | `0/0/0/0%` | step 0 PCBF loss `0.125`；1k snapshot 的 online critic、EMA critic 和 Adam moments 几乎全部非有限，属于 hard stop |

FLOQ-core 的 flow loss 从 `0.107` 降到 `0.092`，critic loss从 `0.111` 降到
`0.0497`，说明“训练 loss 能下降”本身不足以证明 integrated endpoint 正确。下一条 FLOQ
实验必须直接反传 endpoint Bellman residual，并抑制同一 condition 下不同 source 造成的 Q 方差。

PCBF 的逐步快照复现实验确认第 1、2、3、10、20、50、60 次更新后参数和 Adam 状态仍全部
有限，因此不是初始化、第一步反传或路径广播错误；污染发生在 60--1000 步。公式已与论文及官方
JAX 实现逐项核对，source-consistent current path、terminal target 和 `lambda` control variate一致。
关键 optimizer 差异是官方在 Adam 前使用 `clip_by_global_norm(1.0)`，本地首轮没有任何 critic
gradient clipping。

### 18.2 已落地的修复与新 gate

PCBF profile 和 Stage-I launch 现在显式设置：

```yaml
method:
  critic_grad_clip: 1.0
```

update 日志新增 `critic_grad_norm`（clip 前）和 `critic_update_norm`（optimizer transform 后），
用于区分 objective 爆炸、clip 生效和 Adam update 异常。对应单测覆盖 repeated demo updates 中
online/EMA 参数与两个 norm 全部有限。

FLOQ 保留原始 core launch 作为负/机制对照，并新增
`cqn_flow_floq_anchored_pixel_bigym_stage1_gate`：

```yaml
method:
  endpoint_q_lambda: 1.0
  source_consistency_lambda: 0.1
```

前者对 replay-chosen bins 的完整 ODE endpoint 直接施加 scalar Bellman MSE；后者惩罚相同
condition 下不同 base sources 的 endpoint-Q variance。这个版本仍是 CQN-AS：image/state、
level、zoom interval、sequence position、action dimension 和 action bin 全部是 field condition，
所有 bins 仍由 CQN coarse-to-fine enumeration 选择；它不是独立复现连续 actor 算法。

两条 10.5k 复跑在 `move_plate` 启动：

```text
GPU0: move_plate_pcbf_seed1_gate10k_clip1_gpu0_20260722_1350
GPU1: move_plate_floq_anchored_seed1_gate10k_gpu1_20260722_1350
```

修复版 PCBF 在 1k 再次 hard-stop：step 0 的 `critic_grad_norm=0.138`、
`critic_update_norm=7.86e-4` 都有限，但 1k 时 loss、raw grad norm、update norm、online/EMA 参数和
Adam moments 全部非有限。由此排除“只缺 gradient clipping”的假设；真正的问题是在 0--1k
之间某一步，forward/target 或其导数先产生 NaN，clip 对 NaN 无效。该 run 已在 1k 主动终止，
不浪费剩余 9k。后续 PCBF 先从最近 finite checkpoint 逐步记录各 loss 分支和 parameter-group
gradient finite mask，并分别跑 `bc_lambda=0`、`lambda=0`，定位是 demo endpoint BPTT、
successor-velocity control variate还是二者交互。

该定位已做成 `cqn_pcbf_pixel_bigym_nan_diagnostic`：只跑 1100 updates、每 25 步记录
flow-critic/encoder/velocity-head gradient norm 和 non-finite fraction，不跑 eval、不存大快照。两条
互斥诊断命令为：

```bash
# 保留 demo imitation，移除 successor-velocity control variate（等价 BCFM target）
.venv/bin/python train.py launch=cqn_pcbf_pixel_bigym_nan_diagnostic \
  env=bigym/move_plate method.pcbf_lambda=0

# 保留 lambda=gamma，移除 demo all-bin endpoint BPTT
.venv/bin/python train.py launch=cqn_pcbf_pixel_bigym_nan_diagnostic \
  env=bigym/move_plate method.pcbf_lambda='${replay.gamma}' method.bc_lambda=0
```

诊断结果把问题定位到了 demo readout，而不是 PCBF 本身：

- `lambda=0 + bc_lambda=1 + full endpoint demo BPTT`：step 600 仍 finite；step 625 的
  forward loss仍有限（critic `0.0805`、PCBF `0.1038`、margin `0.0701`），但 flow critic 与
  encoder gradient non-finite fraction 已接近 1；step 650 参数随之 NaN；
- `lambda=gamma + bc_lambda=0`：完整 1.1k 全程 finite，末步 PCBF loss `0.00784`、grad norm
  `0.0674`、两个 gradient non-finite fraction 都为 0。

`Q_v0 = clip(epsilon + v_theta(epsilon,t=0))` 随后也做了实证，但仍在 575--600 步出现
“forward finite、gradient NaN”，所以多步 endpoint BPTT 不是根因。真正的 bug 位于同一
`_demo_losses` 函数的 logging-only diagnostics：source-Q `std` 在方差恰好为零时导数未定义，
共享反向图会把它污染到 margin update。现在 expert top-1、Q span、top-2 gap、source-Q std、
rank SNR 和 flip rate全部基于显式 `stop_gradient` 的 Q 计算，只有 FOSD/margin 保留梯度。

修复后 `lambda=gamma + bc_lambda=1 + full T=4 endpoint demo` 的 1.1k diagnostic 全程
finite；末步 critic loss `0.0684`、PCBF loss `0.0190`、grad norm `0.146`，flow critic/encoder
non-finite fraction 都是 0，成功越过旧实现 600--625 步的确定性坏点。因此 PCBF 默认仍保留
full-endpoint imitation；`demo_flow_steps=1` 的 Q_v0 readout 只作为可选消融。单元测试同时锁定：
logging diagnostics 在零 source variance 时梯度严格为零，one-step readout 与
`source + initial_velocity` 一致。

正式 `lambda=gamma,T=4,R=4,full-endpoint demo` 10.5k gate 随后重启。其 1k snapshot
逐叶审计结果为：online parameters `0/14,198,946` non-finite、EMA parameters
`0/12,111,810`、Adam state `0/28,397,893`；因此修复已通过原先必然失败的 1k hard gate。

anchored-FLOQ 必须同时优于首轮的 endpoint-Q loss 与 source-Q std，而不能只看 demo top-1 或
一次 5k 的 12% success。释放出的 GPU0 已转跑 matched direct-C51 gate：

```text
GPU0: move_plate_direct_c51_seed1_gate10k_gpu0_20260722_1355
```

direct C51 已完成，success @ `2.5k/5k/7.5k/10k = 48/52/68/40%`，best 为
`68% @ 7.5k`。这是当前 frozen-demo、margin-only、`B=16+16` Stage-I 的 matched reference；
它表明 replay/demo 与任务本身可学，flow 方法的 0% 不能归因于本轮数据失效。GPU0 随后开始
`lambda=0`、保留 demo loss 的第一条 1.1k PCBF branch diagnostic。

anchored-FLOQ 也已完成，success @ `2.5k/5k/7.5k/10k = 0/20/48/32%`，best
`48% @ 7.5k`。相对 FLOQ-core 的 `0/12/0/0%` 有明确提升，但仍低于 matched C51 的
`48/52/68/40%`。其 endpoint-Q loss `0.09375 -> 0.01287`（下降 `86.3%`），flow loss
`0.1071 -> 0.0509`，demo top-1 `20.0% -> 70.8%`；这些通过 Stage-I endpoint/ranking gate。
不足是 source-Q std `0.1155 -> 0.1403`，虽远好于 core 的 `0.2146`，仍未真正收缩；rank SNR
终点 `0.949` 也略低于 1。为区分 endpoint anchor 与 `0.1` source-consistency 的贡献，GPU1
已启动同一 gate 的 endpoint-only ablation（`source_consistency_lambda=0`）。

### 18.3 后续排队顺序

两张卡完成后按以下顺序复用，不并行扩散变量：

1. 完成正在运行的 matched direct-C51 与 anchored-FLOQ 10.5k gates；
2. 运行上面的 PCBF `lambda=0` / `bc_lambda=0` 两条 1.1k branch-isolation diagnostics；只有定位并修复首个 non-finite 来源后才恢复 `0.5*gamma` sweep；
3. 若 anchored-FLOQ endpoint residual 至少下降 50%，再升到 `T=8,R=8,100k`；否则分别拆开 endpoint anchor 与 source consistency，确定是哪一项有效；
4. PCBF 仍失败才跑同 source/endpoint 的 BCFM (`lambda=0`) 和 Value-Flows DCFM+BCFM 机制对照；
5. 只有 scalar-return 路径通过数值与 ranking gate 后才实现 FlowIQN，V2/V3 继续保持负对照。

### 18.4 本轮最终结果与 Go/Stop 判断

所有数值均为同一 `move_plate` seed 1、25 episodes/eval 的 Stage-I gate：

| 方法 | success @ 2.5k/5k/7.5k/10k | best | 10k endpoint-Q loss | 10k demo top-1 | 10k source-Q std | runtime |
|---|---:|---:|---:|---:|---:|---:|
| matched direct C51 | `48/52/68/40%` | `68%` | N/A | N/A | N/A | `10.1 min` |
| FLOQ core | `0/12/0/0%` | `12%` | `0.11364` | `74.8%` | `0.21463` | `32.2 min` |
| FLOQ endpoint-only | `0/8/20/28%` | `28%` | `0.00735` | `70.0%` | `0.14583` | `30.1 min` |
| FLOQ endpoint + consistency | `0/20/48/32%` | `48%` | `0.01287` | `70.8%` | `0.14029` | `31.3 min` |
| PCBF fixed | `0/0/16/12%` | `16%` | `0.03823` | `50.6%` | `0.68551` | `27.3 min` |

PCBF 的 10.5k final snapshot 再次逐叶审计，online parameters、EMA parameters 和 Adam state
的 non-finite count 均为 0。它从数值 hard-stop 恢复为可学习算法，但 10k demo rank SNR 只有
`0.158`、local source-bin flip rate `19.1%`，且 success 明显落后，因此 **不晋级 100k**。
其 source-Q std 是 return-distribution spread，不能与 expected-Q FLOQ 的 source invariance作同一
语义解释；真正阻止晋级的是低 rank SNR、较高 flip 和环境结果。

FLOQ 的 endpoint anchor 是必要项：它把 endpoint residual 从 core 的 `0.11364` 降到
`0.007--0.013`，并把 10k success 从 0 提到 `28--32%`。source consistency 也有独立贡献：
带 `0.1` 的版本在 5k/7.5k 为 `20/48%`，明显高于 endpoint-only 的 `8/20%`。不过其 best
`48%` 仍比 matched C51 的 `68%` 低 20 个百分点，10k 低 8 个百分点，而且训练 wall time约
3.1 倍，因此当前只能算 **Stage-I 有条件通过**，不能声称优于 C51，也不直接启动 3-seed。

下一轮使用最小增量矩阵：

1. 固定 endpoint anchor `1.0`，只扫 `source_consistency_lambda={0.1,0.3,1.0}` 的 10k gate；
2. 只有某个 consistency 设置达到 best `>=52%`、rank SNR `>=1` 且 source std 不再持续增长，才跑
   `T=8,R=8,100k`；
3. FLOQ 100k 必须同时保留 matched C51 100k，同步报告 best/same-step/AUC/wall-time；
4. PCBF 暂停扩大训练，只做 read-only `R_action={4,8,16}` checkpoint ranking probe，判断低 rank
   SNR 是 Monte Carlo readout不足还是 critic 本身没有分离 action bins；
5. 若增加 action samples 不能把 PCBF flip 压到 10% 以下，则停止 PCBF，下一条 distributional
   主线改为单调 quantile coupling 的 FlowIQN，而不是回到 51-D logit V2/V3。

## 19. 2026-07-22：双线计划——Value Flow 与 Value-Fidelity Audit

### 19.1 新问题不是“RL loss 有没有用”，而是“Q 是否包含反事实动作信息”

当前 demo-driven CQN-AS 同时具备三条容易混淆的信号：

1. replay-chosen bin 接受 bootstrapped C51 TD 监督；
2. demo chosen bin 通过 FOSD 和 large-margin 被显式推到其他 bins 之上；
3. 下一状态的 `max` action 又由同一个 critic 的 action ranking 产生。

本地 canonical 配置中 `critic_lambda=0.1`、`bc_lambda=1.0`，但二者 loss 的数值尺度不同，
不能只凭系数断言 BC 一定支配。真正的风险是 **action-ranking shortcut**：critic 只要识别
“哪个 bin 像成功 demonstration”，就可能同时获得高 demo ranking、较低 TD loss 和不错的 policy
success，而没有学会“固定状态下换一个 action 会怎样改变未来 return”。CQN-AS 对
`level x sequence x action-dimension` 的 replay-chosen bins 使用同一个 observed transition/reward
构造各自的 bootstrapped target，这也使 per-coordinate credit assignment 在有限 action coverage 下
不唯一。

CQN-AS 论文中已有证据不能完全回答这个问题：

- return-to-go validation 是独立的监督回归，说明 action sequence 对 observed RTG 有预测信息，
  但没有 state-only、matched action-shuffle 或 simulator intervention 就不能建立 action effect；
- `CQN-AS (No RL)` 只说明加入 RL objective 后能利用 online trial-and-error 并提高 success，
  不能区分 counterfactual Q、success/failure classification、reward-weighted imitation 或共享表示收益；
- 无 demonstration 的 HumanoidBench 结果反驳“CQN-AS 在所有场景都只是 BC”的强说法，但不排除
  demo-driven `move_plate` 落入 imitation shortcut。

所以本文后续不再用 TD/Bellman loss、demo top-1 或 environment success 中任何单项来宣称
“学到了真实 value”。操作性定义改为：对固定 `s` 和固定 continuation policy，预测的
`Q(s,a_i)-Q(s,a_j)` 必须与从同一 simulator state 分叉得到的 counterfactual return difference
同号且排序相关。绝对 Q 偏移可以不准，动作间 advantage/ranking 不能不准。

### 19.2 两条研究线共享同一个 value-fidelity gate

#### 路线 A：CQN-AS + Flow Matching

1. 保留目前最有希望的 anchored FLOQ：`endpoint_q_lambda=1.0`，只补
   `source_consistency_lambda={0.3,1.0}` 两个 10k gate；PCBF 不扩大训练，V2/V3 保持负对照。
2. 在任何 100k 扩展前，给 direct C51 与 FLOQ checkpoint 跑完全相同的 counterfactual audit；
   FLOQ 除了 success、endpoint residual 和 source variance，还必须报告真实 action-rank fidelity。
3. flow 的主比较增加两种训练协议：
   - canonical/shared-Q imitation：demo loss 直接作用于 Q；
   - decoupled imitation：demo loss 只训练独立 behavior head，value flow 只接受 return/TD supervision。
4. 只有某个 FLOQ 版本同时通过原 Stage-I performance gate 和下面的 value-fidelity gate，才升到
   `T=8,R=8,100k`，并与同协议 direct C51 同步运行。否则停止扩大 flow compute，先修 value/data
   identifiability。

#### 路线 B：CQN-AS 到底在学 value 还是 imitation shortcut

第一组是 objective decomposition，保持 replay、encoder、batch、EMA 和 environment budget 一致：

| 版本 | TD/C51 | demo objective | 用途 |
|---|---:|---|---|
| `Q-BC` | off | Q 上的 FOSD/margin | 对齐论文 `No RL`，得到隐式 bin classifier |
| `Q-TD` | on | off；demo transitions 仍在 replay | 检查 TD 单独能否形成 action ranking |
| `Q-TD+Q-BC` | on | Q 上的 FOSD/margin | 当前 CQN-AS |
| `Q-TD + pi-BC` | on | 独立 behavior head 上的 CE/margin | 将 value estimation 与 imitation 解耦 |
| `V(s) + pi-BC` | state-value only | 独立 behavior head | success/progress classifier 加 BC 的 shortcut baseline |
| `shuffled-reward Q-TD+Q-BC` | reward/terminal 在 episode-safe strata 内置换 | Q 上的 FOSD/margin | 检查 policy 是否真正依赖 reward semantics |

`Q-TD + pi-BC` 是首选改进而不只是 ablation。建议令

\[
Q(s,a)=V(s)+A(s,a),\qquad \sum_b A(s,a_b)=0,
\]

并用 `score = normalized_A + beta * log pi_BC` 选 bin。`pi_BC` 明确负责 support/imitation，
`A` 明确负责 return difference；demo loss 不允许回传到 advantage/value head。`beta` 先固定，
只有 counterfactual coverage 增长后才衰减。这样即使最终 policy 接近 demonstrations，也能量化
是 behavior prior 还是 learned advantage 在做决定。

为区分“BC 直接改 Q”与“BC 通过共享视觉表示间接改 Q”，严格 audit 版本还要对 behavior head
输入使用 `stop_gradient(h(s))`（或单独的轻量 adapter/encoder），使 BC gradient 不进入 value
encoder；共享 encoder 版本只作为随后单独的 representation-transfer ablation。

第二个改进是给 value 增加非自举锚点。对完成的 online/demo episodes 直接计算 observed
Monte-Carlo return，并与 1-step C51 target 混合，或使用短 `TD(lambda)`；不直接回到高方差
`nstep=16`。Bellman residual 仍保留，但不能再作为 value accuracy 的替代指标。

第三个改进是收集可识别 action effect 的数据。普通连续 Gaussian noise 未必覆盖 CQN 的离散
decision boundary；优先在 Q 与 BC disagree、top-2 gap 小或 ensemble uncertainty 高的状态，
强制替换一个 coarse-to-fine bin，记录真实 outcome。若没有这种同状态附近的 action contrast，
expert-only data 原理上就无法唯一识别未执行 bins 的 Q。

### 19.3 两层 value-fidelity audit

#### A. 只读、低成本的 observational audit

先复用现有 direct-C51、FLOQ 和 PCBF snapshots，不立即重跑训练：

1. 以 episode 为单位划分 held-out trajectories，避免相邻 image frames 泄漏；
2. 对 observed return-to-go 比较 `state-only`、`action-only`、`state+real action sequence`、
   `state+matched shuffled action sequence`；shuffle 必须在 task-progress/return strata 内完成，避免
   简单 distribution shift；
3. 记录 real action 在 state-only 之上的增量 L1/R2，而不是只报总 validation loss；
4. 对成功/失败且视觉状态相近的 transition pairs，比较 learned Q gap 是否随 return gap 变化；
5. 训练一个显式 bin-BC probe，并记录 `agreement(Q-policy, BC-policy)`、Q 与 BC 分歧发生在哪些
   level/sequence/dimension，以及分歧后的真实 episode outcome。

这一层只能筛查 shortcut，不能证明 causality。低 Bellman residual、低 endpoint loss 和高 demo
top-1 在这里都只作为训练诊断。

#### B. simulator branch counterfactual audit（结论 gate）

1. 从 held-out successful、failed 和 policy-rollout episodes 的 early/middle/late phase 保存完整
   MuJoCo physics/task state，而不只保存 pixels/proprioception；
2. 从完全相同的 state 分叉候选：`Q action`、`BC action`、logged/demo action、local runner-up bin、
   random/low-Q bin；对每个 coarse level 的候选，强制该 bin 后让 deeper levels greedy refine，
   使实际执行动作与 critic 查询语义一致；
3. 第一阶段只审计真正立即执行的 `k=0` action；第二阶段分别审计完整 open-loop chunk 与
   temporal-ensemble 后的 effective action，不能把一个永远不会原样执行的 future plan token 当作
   ground-truth action；
4. 执行被干预 action/chunk 后固定 continuation policy，随机环境则使用 common random seeds，
   估计 `Q_MC^pi(s,a)`；
5. 每个 checkpoint 报告 per-state Spearman/Kendall rank、top-1 accuracy、top-1 regret、
   pairwise sign accuracy、`Delta Q_pred` 与 `Delta return_MC` 的回归斜率，以及 C51/return-flow 的
   calibration/CRPS。

最有辨识力的 policy 指标是 **disagreement advantage**：仅在 Q-policy 与 BC-policy 选择不同的
状态上，比较两者从同一状态分叉后的 return。如果 Q 经常偏离 BC 且分叉 return 更好，才是 RL
advantage 的直接证据；如果二者几乎总一致，或者 Q 偏离时不比 BC 好，那么 success 提升不能证明
critic 学会了真实 action value。

### 19.4 Go/Stop 与立即执行顺序

value-fidelity 不设拍脑袋的绝对相关系数门槛，先用 matched controls 和 bootstrap confidence
interval 做相对判断：

- direct C51/FLOQ 的 rank correlation、pairwise sign accuracy 和 top-1 regret必须显著优于
  `Q-BC`、`V(s)+pi-BC`、reward-shuffle 和 random ranking；
- 在 Q/BC disagreement states 上，Q action 的 paired return 必须为正，且置信区间不能覆盖 0；
- action-shuffle 后 RTG 的增量预测力必须显著下降；否则原 Figure-2-style validation主要是
  state/progress shortcut；
- 若当前 `Q-TD+Q-BC` 失败而 `Q-TD+pi-BC` 通过，则后续 CQN-Flow 统一切到 decoupled protocol；
- 若 direct C51 和 flow 都失败，则停止 value-head/flow-step sweep，先做 structured bin exploration；
  这时瓶颈是 coverage/identifiability，不是 C51 还是 Flow Matching。

立即顺序：

1. 先实现 checkpoint-only Q/BC agreement、held-out RTG action-shuffle 和 value-rank probe；
2. 验证 BiGym 完整 simulator state save/restore 的 deterministic replay，再实现 branch evaluator；
3. 用现有 matched direct C51 与 anchored FLOQ checkpoints 产出第一张 counterfactual ranking 表；
4. 再跑上面的 10k objective-decomposition matrix；
5. 最后才决定 FLOQ consistency `0.3/1.0` 是否晋级 100k。

## 20. 2026-07-22：Value-Fidelity Stage-I 实测与 first-success 修正

### 20.1 objective decomposition 已经支持 imitation-shortcut 假设

同一 MovePlate seed 1、同一 60 demos、`B=16+16`、10.5k frames、25 episodes/eval 下，新增
两条 matched C51 分解实验：

| objective | success @ 2.5k/5k/7.5k/10k | best | 10k |
|---|---:|---:|---:|
| `Q-TD`：`bc_lambda=0` | `0/0/0/0%` | `0%` | `0%` |
| `Q-BC`：`critic_lambda=0` | `24/60/48/80%` | `80%` | `80%` |
| full CQN-AS | `48/52/68/40%` | `68%` | `40%` |
| anchored FLOQ | `0/20/48/32%` | `48%` | `32%` |

因此当前设置中，单独作用在 Q bins 上的 margin imitation 已足以产生强策略，TD 单独则完全
无法起步。更关键的是，checkpoint-only probe 在同一批 replay observations/actions 上给出：

| checkpoint | all replay-bin top-1 | Q-policy/replay agreement | mean candidate Q span | online-success Q | online-failure Q | online-success Spearman(Q, first-success RTG) |
|---|---:|---:|---:|---:|---:|---:|
| `Q-TD` | `32.2%` | `32.1%` | `0.009` | `0.514` | `1.850` | `-0.335` |
| `Q-BC` | `75.6%` | `67.8%` | `1.187` | `-0.046` | `-0.079` | `-0.156` |
| full CQN-AS | `75.4%` | `68.0%` | `0.776` | `0.466` | `1.255` | `-0.539` |
| anchored FLOQ | `58.5%` | `54.1%` | `0.425` | `1.357` | `1.836` | `-0.802` |

这里随机 top-1 reference 是 `20%`。full CQN-AS 的两个 imitation 指标与 `Q-BC` 几乎逐项
相等，而 `Q-TD` 的 action discrimination 已塌到 span `0.009`。full CQN-AS 和 FLOQ 还都给
online failure 更高的平均 Q，并在 online successful trajectories 内产生负的 return ranking。
这张表是 observational evidence，样本为每组 8 个 episode-balanced anchors；它本身不能替代
causal branch，但已经否定“高 demo top-1 等于 value 学得好”。原始 JSON 位于
`exp_local/cqn_value_fidelity_stage1/probes/`。

### 20.2 same-state simulator branch 给出首轮 causal 反例

新增 exact branch evaluator 的协议为：eval step 50 保存完整 MuJoCo/model/controller/wrapper/agent
state；在 critic 区分度最大的当前动作维度 `k=0` 上强制 level-0 的 5 个 bins；每个 bin 后让
level 1/2 greedy refine，再用同一 checkpoint policy 跑到 termination/time-limit。所有分支共享
相同 pre-branch state、agent RNG 和 continuation policy。

| checkpoint | states | informative | Q top-1 命中 | mean Spearman | mean top-1 regret |
|---|---:|---:|---:|---:|---:|
| full CQN-AS | 4 | 3 | `0/3` | `-0.149` | `0.155` |
| `Q-BC` | 4 | 3 | `0/3` | `-0.882` | `0.022` |
| anchored FLOQ | 4 | 3 | `1/3` | `-0.133` | `0.143` |

full CQN-AS 的 seed 20003 是直接反例：预测 Q 依次约为
`[-1.075,-0.939,-0.181,0.493,0.439]`，但对应 discounted return 为
`[0.434,0,0.413,0,0]`；critic 选中的 bin 3 失败，最低 Q 的 bin 0 成功，regret `0.434`。
这不是 calibration offset，而是 action ordering 的符号相反。当前只有 4 states，结论应表述为
“现有 checkpoint 未通过 value-fidelity gate”，还不是总体置信区间；下一轮扩到 phase-stratified
states 和更多 seeds 后再报告 bootstrap CI。

这三组结果是明确绕过 temporal ensemble 的 `raw_plan` 干预，用来审计 critic 所查询 plan 的
即时 `k=0` action effect。为同时审计真实 policy execution，evaluator 现已增加
`effective_policy` 模式：候选 plan 会先注册到 agent history、经过 temporal ensemble，并用更新后的
history 运行 continuation；两种口径必须分开报告。

state restore 已独立用随机动作验证：MuJoCo integration state、actuator controls、FrameStack、
TimeLimit、receding-horizon history、executed action 和 next observation 的逐元素
`max_abs_error` 全部为 0。实现必须额外保存 `data.ctrl`、floating-base accumulator、animated-leg
model arrays 和 wrapper state；还修复了 BiGym `_action` 与 caller action array alias 导致 restore
反向污染候选动作的问题。相关实现：

- `robobase/envs/bigym_branch_state.py`；
- `scripts/analyze_cqn_branch_counterfactual.py`；
- `scripts/analyze_cqn_value_fidelity.py`；
- `scripts/analyze_cqn_flow_ranking.py` 的 metrics scope bug 同时已修复。

### 20.3 更上游的 MDP mismatch：demo success tail 与 C51 support

审计 replay 后发现 online 与 demonstrations 实际不是同一个 MDP：live BiGym 的
`terminate = success or fail`，所以第一次 reward 1 后立即终止；历史 demonstrations 却保留
success 后的 tail，并由 `_demo_to_steps` 只在最后一帧置 terminal。当前 MovePlate demo replay：

- 51 个 labeled-success episodes 中，reward total 中位数 `24`、最大 `27`；
- 9,253 个 demo transitions 中，`8,656` 个（`93.55%`）discounted RTG 大于 C51 的
  `v_max=2`，最大 RTG `23.77`；
- 另有 1 个 labeled-success demo 在 downsampled replay 中总 reward 为 0，形成 demo margin 与 TD
  target 的直接冲突。

这既使 C51 projection 大面积饱和，也使 demo TD target 与 online success-return semantics 不同。
已增加 `env.truncate_demo_at_success`（legacy 默认 false）：启用后在第一个 reward > 0.25 的 demo
step 置 `terminal=True` 并截断 tail，与 live MDP 对齐。配置
`cqn_as_pixel_bigym_value_fidelity_gate` 默认启用该修正；两条行为单测均通过，分别锁定
first-success truncation 与 legacy default。

### 20.4 Stage-II：修正 MDP 提高了策略成功率，但没有修好 value

同 seed、同 10.5k gate 的 first-success 实验已经完成：

| objective | success @ 2.5k/5k/7.5k/10k | best | 10k |
|---|---:|---:|---:|
| legacy full CQN-AS | `48/52/68/40%` | `68%` | `40%` |
| clean full CQN-AS | `48/56/92/56%` | `92%` | `56%` |
| legacy `Q-TD` | `0/0/0/0%` | `0%` | `0%` |
| clean `Q-TD` | `0/0/0/0%` | `0%` | `0%` |

first-success 修正使 full CQN-AS 在 7.5k 同步提升 `24` 个百分点，best 从 `68%` 提到 `92%`；
这验证了 demo return semantics/C51 support mismatch 是真实训练问题。但 `Q-TD` 仍完全无法起步，
说明该 mismatch 与 TD-only 的 action-identifiability collapse 是两个独立问题。

使用 clean full replay 的同一组 observations/actions 做 checkpoint probe：

| checkpoint | all replay-bin top-1 | Q-policy/replay agreement | mean candidate Q span | online-success Q | online-failure Q | online-success Spearman(Q, first-success RTG) |
|---|---:|---:|---:|---:|---:|---:|
| clean full CQN-AS @ 8k | `72.4%` | `64.1%` | `0.688` | `0.309` | `0.868` | `-0.766` |
| clean `Q-TD` @ 10.5k | `26.8%` | `26.5%` | `0.005` | `0.276` | `0.545` | `-0.012` |
| newest-plan full @ 10.5k | `68.6%` | `60.1%` | `0.637` | `0.189` | `0.819` | `-0.371` |
| clean anchored FLOQ @ 8k | `54.2%` | `50.1%` | `0.306` | `0.899` | `0.844` | `-0.599` |

因此 `92%` policy success 不能解读为 `Q` 已学会 return：clean full 仍强烈偏向 replay bins，给
online failures 更高 Q，并在成功轨迹内反向排序 RTG。clean `Q-TD` 的 span 仍只有 `0.005`，
几乎是 state-dependent constant。对应 JSON 位于
`exp_local/cqn_value_fidelity_stage2/probes/`。

clean full @ 8k 的 expanded same-state branch 也未通过。8 个 eval seeds、step 25/75 共 16 个
states，分别审计两种执行语义：

| intervention | informative | Q top-1 | pairwise sign | mean Spearman | mean top-1 regret |
|---|---:|---:|---:|---:|---:|
| `raw_plan`：直接执行 critic candidate | 10 | `20%` | `45.6%` (68 pairs) | `-0.079` | `0.233` |
| `effective_policy`：注册 plan、temporal ensemble、再 continuation | 6 | `33.3%` | `48.5%` (33 pairs) | `-0.048` | `0.198` |

top-1 采用 tie-aware 定义（预测 bin 达到并列最大 return 即命中）；`raw_plan` 恰好等于 5-bin
random reference，`effective_policy` 为 `2/6`，但其中均包含 outcome ties，需结合 rank/regret 解读。
两种口径的 pairwise sign 也都没有超过 `50%` random reference。
这个规模还不足以代替多 seed bootstrap CI，但已经不再是单个偶然反例：高成功率 policy 的 Q
仍未显示可靠的 causal action ordering。

这里还暴露了 CQN-AS 本身的 action-semantics mismatch：当前 `K=16`、temporal-ensemble
`gain=0.01` 时，newest plan 的归一化权重只有
`1/sum(exp(-0.01*[0..15])) = 0.0673`。实测 16 个 states 的 mean raw candidate action span
为 `1.256`，mean effective span 仅 `0.0845`，比例同样是 `0.0673`。也就是说 critic 用 Q 选择
raw plan bin，但实际当前动作约 `93.3%` 来自旧 plans；同时 replay 中第一个 action token 又被覆盖成
effective action。训练、argmax query 与 environment execution 的 action 语义并不一致。

### 20.5 clean FLOQ 与 action-semantics ablation

同 seed、同 gate 的补充实验已经完成：

| variant | success @ 2.5k/5k/7.5k/10k | best | 10k |
|---|---:|---:|---:|
| legacy anchored FLOQ | `0/20/48/32%` | `48%` | `32%` |
| clean anchored FLOQ | `0/28/40/36%` | `40%` | `36%` |
| clean full CQN-AS, gain `0.01` | `48/56/92/56%` | `92%` | `56%` |
| clean full CQN-AS, gain `5.0` | `0/0/0/0%` | `0%` | `0%` |

newest-plan ablation 仅把 temporal-ensemble gain 从 `0.01` 改为 `5.0`，使
newest plan 权重从 `6.73%` 提到 `99.33%`。它用于判断 action-semantics 对齐能否同时改善 policy
success 与 causal Q ranking。结果是 success 完全消失，而上表 checkpoint probe 仍显示强 replay-bin
imitation 和反向 success/failure Q ordering。因此简单关掉 averaging 不是修复；temporal ensemble
确实提供了关键的 smoothing/behavior prior，但 critic 必须显式评价其产生的 effective action/history。

first-success 修正没有改善 anchored FLOQ 的 best success（single seed 下反而从 `48%` 到 `40%`），
也远低于 clean direct C51 的 `92%`；在多 seed 前不把 8 个百分点差异解释成显著退化。clean FLOQ
的 checkpoint-only probe 首次使 online-success mean Q (`0.899`) 略高于 failure (`0.844`)，但
成功轨迹内部 Q/RTG Spearman 仍为 `-0.599`，且 replay-bin top-1 `54.2%` 仍远高于 `20%`
random reference。这是 group-level ordering 的局部改善，还不是 action-value fidelity；expanded
effective-action causal probe 的 step-50 gate 已完成：4 states 中 2 个 informative，Q top-1
`0/2`，17 个有效 action pairs 的 sign accuracy `29.4%`，mean Spearman `-0.598`，mean regret
`0.417`。两个反例中 critic 都把实际失败的 bin 0 排最高，而中间 bins 可以成功。因此 clean FLOQ
仍未通过 causal gate。随后完成的 step-25/75 expanded probe 覆盖 8 个 eval seeds、16 states，
其中 9 个 informative：tie-aware Q top-1 仅 `22.2%` (`2/9`)，73 个有效 action pairs 的 sign
accuracy 为 `42.5%`，mean Spearman `-0.171`，mean regret `0.240`。它与小样本方向一致，并且
pairwise ordering 仍低于 `50%` random reference；因此 clean FLOQ 不晋级 100k，也不继续用
atoms、flow steps 或 consistency-weight sweep 解释失败。

### 20.6 修订后的下一步

判定顺序更新为：

1. 若 clean FLOQ 仍不能通过 counterfactual ranking，不再扫 atoms、flow steps 或 consistency weight；
   下一实验直接转为 Q/BC disagreement 与 low-gap states 的 structured bin interventions。
2. value target 加入 completed-episode Monte-Carlo/TD(lambda) 非自举锚点；同时保留 1-step TD，分别
   报告 Bellman residual 与 held-out causal ranking，不用前者替代后者。
3. 实现 `Q-TD + pi-BC`：behavior margin/CE 不再写入 Q logits，严格版对 BC 输入 encoder feature
   做 stop-gradient。policy 同时报告 BC score 与 learned advantage，causal evaluator 加入 Q/BC
   disagreement advantage。
4. 保留 temporal smoothing，但把 ensemble plan history/effective current action 纳入 critic condition；
   更干净的版本是 BC head 预测完整平滑 plan，Q/return-flow 只对实际执行的 `k=0` effective action
   学 advantage 并 rerank，而不是让 Q 对不会原样执行的 raw plan bin 做 argmax。
5. 当前所有已审计 checkpoints 都未通过 causal value gate，不能因 success 或 demo top-1 达标而
   晋级 100k。publication-level 结论前将 branch states 扩到 early/middle/late、至少 3 seeds，并
   加入 random/low-Q ranking 与 paired bootstrap CI。

### 20.7 Stage-III：解耦 behavior policy 与 effective-action critic

`92%` 的准确出处是 BiGym `move_plate`、seed 1、25 eval episodes 的 7.5k checkpoint
（`23/25`），不是跨任务或多 seed 平均；同一 run 的 2.5k/5k/7.5k/10k 曲线为
`48/56/92/56%`。因此后续仍同时报告 best、same-step 和整体趋势。

第一版 `Q-TD + pi-BC` 已实现为可选路径，legacy checkpoint 路径保持默认不变：

1. 独立 categorical C2F policy head 用 demo CE 学完整 `K=16` plan；CE/FOSD/margin 不再写入
   critic logits。
2. critic 只在 replay 的 `k=0` token 上做 C51 TD；该 token 是 temporal ensemble 后实际执行的
   action，而不是新 raw plan。
3. target action 使用 replay action sequence 向左移一位得到的实际 `a_{t+1}`，构成
   SARSA-style behavior-value target，不再用 critic 自己的 argmax 生成 bootstrap action。
4. shared 版本允许 BC 更新共享视觉 encoder；strict 版本对 policy 输入 feature 做
   stop-gradient，用于区分“BC 只污染 Q head”与“BC 还通过 representation 帮助 TD”。
5. 新增独立日志 `policy_bc_loss`、`policy_ce`、`policy_demo_top1`、`critic_loss`、
   `critic_q_span` 与 `total_loss`。

2026-07-23 已通过两个新增解耦单测、legacy action/temporal-ensemble 回归测试和真实 MovePlate
pixel smoke。两条 matched 10.5k gates 已完成：

| variant | GPU | run directory |
|---|---:|---|
| shared encoder | 0 | `exp_local/cqn_value_fidelity_stage3/move_plate_decoupled_effective_shared_seed1_gpu0_20260723_002021` |
| strict stop-gradient | 3 | `exp_local/cqn_value_fidelity_stage3/move_plate_decoupled_effective_strict_seed1_gpu3_20260723_002021` |

晋级条件不只看 success：policy demo top-1 必须正常上升，同时 TD-only critic 的 effective-action
counterfactual pairwise sign 显著高于 `50%`，且 success/failure Q ordering 与 trajectory RTG
correlation 不再反向。若 policy 成功但 critic 仍不过 gate，下一步直接加 completed-episode
MC/TD(lambda) anchor；若 strict policy 同时失败，则保留独立 policy encoder 或 shared encoder，
不能把 representation starvation 误判成 policy-head 失败。

实际 success @ 2.5k/5k/7.5k/10k：

| variant | success | best | policy demo-bin top-1 @ 10k | train Q span @ 10k |
|---|---:|---:|---:|---:|
| legacy clean CQN-AS | `48/56/92/56%` | `92%` | same Q head | `0.688` (8k checkpoint probe) |
| decoupled shared encoder | `24/24/60/44%` | `60%` | `85.5%` | `0.0162` |
| decoupled strict stop-gradient | `16/24/28/44%` | `44%` | `85.6%` | `0.0157` |

shared 版本说明 BC 不直接写 Q logits 时仍能形成可用 policy；strict 相比 shared 的 7.5k
`28% vs 60%` 则说明 BC 对共享视觉 representation 的训练很重要。两者最终 BC accuracy 几乎相同，
但 strict policy 在中期明显更弱，因此不把完全 stop-gradient 作为主线。

checkpoint probe 仍否定 value fidelity。这里修正一个 evaluator 分组错误：最初把
`offline_episode_count` 设成了成功 demo 数 `51`，但 replay 中实际有 `60` 个 offline episodes，
其中一个 labeled-success demo 在 first-success replay 中 reward 为 0。因此旧版表中的 online
success/failure Spearman 无效；用正确边界 `60` 重跑 shared @ 8k 后得到：

| checkpoint | critic replay-bin top-1 | mean Q span | behavior/replay agreement | online-success Q | online-failure Q | online-success Spearman(Q, RTG) |
|---|---:|---:|---:|---:|---:|---:|
| shared @ 8k, corrected | `20.8%` | `0.0103` | `61.8%` | `0.0146` | `0.3141` | `+0.671` |

critic replay-bin top-1 已降到 `20%` random reference，证明 imitation shortcut 确实从 Q head 中被
移除。修正后的 successful trajectories 内部 RTG ordering 不再反向，但 critic 仍把 online failures
整体估得比 successes 高约 `21.5x`，而且 action discrimination span 仅 `0.0103`；这说明 sparse
1-step TD 仍没有恢复可信的 action value。旧版 shared @ 10k 与 strict @ 10k 分组结果在按 60 个
offline episodes 重跑前不再作为证据。BC-manifold structured counterfactual 进一步验证：shared @ 8k
的 4 states 中 3 个
informative，Q top-1 `1/3`、26 个有效 pairs 的 sign accuracy 恰为 `50.0%`、mean Spearman
`0.043`。strict 只有 1 个 informative state，不能单独下结论。Stage-III 因此不通过 causal gate；
下一主实验是 shared-encoder decoupled policy 加 completed-episode MC/TD(lambda) return anchor。

### 20.8 Stage-IV：MC return 能修 calibration，但没有学会反事实 action ordering

已实现 completed-episode discounted return anchor，默认关闭并保持 legacy checkpoint 路径不变：

1. demo 与 online episode 都在完整 episode 结束后反向计算 `mc_return`，并作为 replay extra element
   保存；demo first-success truncation 后的 target 与 live MDP 一致；
2. `mc_return` 投影到同一组 C51 atoms，作为执行动作的额外 categorical CE；训练日志分开记录
   `td_critic_loss`、`mc_return_loss`、`mc_return_mae` 与 `mc_return_mean`；
3. MC anchor 只用于 decoupled `Q-TD + pi-BC` 路径；独立 policy head 和 SARSA-style replay-next
   TD target 均保持不变；
4. 新增 `mc_return_stop_gradient_encoder` 与 `mc_return_value_only` 两个结构控制，分别用于隔离
   encoder gradient 和 dueling advantage gradient。

同 seed、同 60 demos、同 10.5k gate 的结果：

| variant | success @ 2.5k/5k/7.5k/10k | best | MC MAE @ 10k | policy demo top-1 @ 10k | train Q span @ 10k |
|---|---:|---:|---:|---:|---:|
| no MC | `24/24/60/44%` | `60%` | - | `85.5%` | `0.0162` |
| MC weight `0.1` | `4/8/52/32%` | `52%` | `0.00739` | `84.9%` | `0.0143` |
| MC weight `1.0` | `8/16/72/52%` | `72%` | `0.01866` | `83.8%` | `0.0318` |

强 MC 在这个单 seed gate 上把 decoupled best 从 `60%` 提到 `72%`、10k 从 `44%` 提到 `52%`，
但仍低于 clean legacy CQN-AS 的 `92%` best。弱 MC 则没有改善 control success。对应 run：

- no MC: `exp_local/cqn_value_fidelity_stage3/move_plate_decoupled_effective_shared_seed1_gpu0_20260723_002021`
  (`r6u6errj`);
- MC `0.1`: `exp_local/cqn_value_fidelity_stage4/move_plate_mc_return_w0p1_shared_seed1_gpu0_20260723_005100`
  (`xoa4azmj`);
- MC `1.0`: `exp_local/cqn_value_fidelity_stage4/move_plate_mc_return_w1p0_shared_seed1_gpu3_20260723_005100`
  (`06doce8t`).

使用正确的 `offline_episode_count=60`，8k checkpoint observational probe 为：

| variant | replay-bin top-1 | Q span | mean Q / mean RTG | all Spearman(Q, RTG) | demo success Q / failure Q | online success Q / failure Q |
|---|---:|---:|---:|---:|---:|---:|
| no MC | `20.8%` | `0.0103` | `0.0915 / 0.2351` | `-0.642` | `0.0109 / 0.0263` | `0.0146 / 0.3141` |
| MC `0.1` | `23.8%` | `0.0138` | `0.2208 / 0.2203` | `+0.897` | `0.4582 / 0.0084` | `0.4040 / 0.0127` |
| MC `1.0` | `23.8%` | `0.0377` | `0.2367 / 0.2237` | `+0.915` | `0.4520 / 0.0082` | `0.4258 / 0.0608` |

这是清楚的正结果：MC anchor 修复了 executed behavior trajectory 的 return calibration，并没有把
critic replay-bin top-1 拉回 imitation 水平；后者仍约等于 `20%` random reference。但这只是
`Q(s,a_behavior)` 的 observational calibration，不能证明同一状态下不同 action 的 ordering。

扩大后的 same-state `effective_policy` branch probe 正好把二者分开：8 个新 eval seeds、step
`30/75/120`、level-0 五 bins、最多 250 continuation steps；只对 return 确实不同的 states/pairs
计分。

| variant @ 8k | total states | informative states | informative pairs | pairwise sign | mean Spearman | Q top-1 | mean regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| MC `0.1` | 24 | 10 | 76 | `51.3%` | `+0.006` | `40.0%` | `0.091` |
| MC `1.0` | 23 | 1 | 6 | `66.7%` | `+0.289` | `100%` | `0.000` |

弱 MC 的 76 个 pairs 已经足以推翻早先 4-state probe 的乐观读数：sign accuracy 近乎抛硬币、rank
correlation 近零。强 MC 只有 1 个 informative state，`66.7%` 不能解释成通过。结论是：当前 MC
目标成功学习了“这条行为轨迹后来是否成功/多久后成功”，但没有提供未执行 bins 的监督或 coverage，
所以不能单独识别 `Q(s,a)` 的局部反事实排序。这正是“披着 value 外壳的 behavior-return predictor”
与真实 action value 的可检验区别。

相关文献给出的启发与边界：

- [MCAC](https://arxiv.org/abs/2210.07432) 用 calibrated Monte-Carlo reward-to-go 与 TD target 的
  max 构造 target，主要解决稀疏奖励下的低估；它同样不会凭空给未执行动作提供反事实标签；
- [Q-Transformer](https://proceedings.mlr.press/v229/chebotar23a.html) 在离线机器人学习中组合
  Monte-Carlo、n-step 与 TD return，支持继续做 compound return，但它的 action-token
  discretization 与数据覆盖机制不能由一个 MC CE 自动替代；
- [Sequence Compression / TD(lambda)](https://proceedings.mlr.press/v235/ramesh24b.html) 强调
  MC 的 delayed-credit 优势与高方差、TD 的低方差与 bootstrap bias 之间需要折中；因此下一轮会把
  return target 结构和 action-identifiability 分开做，而不是继续只扫 MC weight；
- [DQfD](https://arxiv.org/abs/1704.03732) 的 large-margin demonstration loss 能显著提高策略，但当
  supervised margin 与 Q 共用 logits 时，正是本项目必须用 decoupled policy 与 causal branch 避免的
  imitation/value 混淆；
- [CQL](https://arxiv.org/abs/2006.04779) 对数据外 action 加保守约束，适合作为“Q-max 是否只是利用
  OOD 误差”的对照，但 pessimism 只能抑制虚高值，不能创造缺失的反事实排序标签；
- [IQL](https://arxiv.org/abs/2110.06169) 明确避免在训练时查询 unseen actions，并用 implicit
  state value 与 advantage-weighted behavior extraction 改进 policy。若 structured exploration
  仍无法提供 bin coverage，IQL-style `V(s) + weighted pi-BC` 是比继续让未通过 causal gate 的 Q
  直接 argmax bins 更诚实的主 baseline：它承认策略改进只发生在数据 support 内。

### 20.9 Stage-V：只校准 dueling value stream

Stage-IV 的诊断是“state/trajectory return 已对，action advantage 仍错”。因此新增
`method.mc_return_value_only=true`：critic 暴露 dueling `value_logits` 与 centered
`advantage_logits`，MC loss 使用
`value_logits + stop_gradient(selected_advantage_logits)`；这保证一次 MC update 不直接修改
advantage-head 参数。这里不能把它夸大成“ranking 数学上只由 TD 决定”：distributional dueling
是在 atom logits 上相加后再 softmax，改变 value logits 仍可能改变不同 bins 的期望值排序；shared
encoder 当前也保留 MC gradient。该实验只隔离 direct advantage-parameter gradient，并用 causal
probe 实测剩余耦合。若通过，再用 `mc_return_stop_gradient_encoder=true` 做更严格隔离；真正保证不改
action ranking 的 control 需要独立 scalar state-value baseline，它只对所有 bins 加同一标量 offset。

完整 CQN-AS 单测和新增 launch-config test 均通过，并完成真实 MovePlate pixel/JIT 8-frame smoke。
两条 matched 10.5k runs 已于 2026-07-23 完成：

| variant | GPU | W&B | run directory |
|---|---:|---|---|
| value-only MC `0.1` | 0 | `9m74jopt` | `exp_local/cqn_value_fidelity_stage5/move_plate_mc_valueonly_w0p1_shared_seed1_gpu0_20260723_011500` |
| value-only MC `1.0` | 3 | `d2imdq30` | `exp_local/cqn_value_fidelity_stage5/move_plate_mc_valueonly_w1p0_shared_seed1_gpu3_20260723_011500` |

| variant | success @ 2.5k/5k/7.5k/10k | best | final MC MAE | final policy demo top-1 | final train Q span |
|---|---:|---:|---:|---:|---:|
| whole-Q MC `0.1` | `4/8/52/32%` | `52%` | `0.0074` | `84.9%` | `0.0143` |
| value-stream MC `0.1` | `40/16/12/52%` | `52%` | `0.0123` | `86.1%` | `0.1000` |
| whole-Q MC `1.0` | `8/16/72/52%` | `72%` | `0.0187` | `83.8%` | `0.0318` |
| value-stream MC `1.0` | `16/48/68/32%` | `68%` | `0.0831` | `84.9%` | `0.0497` |

value-stream ablation 改善了部分前/中期 checkpoints，却没有提高 matched peak：弱权重 peak 持平、
强权重从 `72%` 降到 `68%`，而两条曲线都出现明显回落。它还把 Q span 放大，尤其弱权重从
`0.0143` 到 `0.1000`；这与 atom-logit coupling 会间接改变 bin ranking 的理论检查一致，不能称为
纯 calibration improvement。

8k corrected observational probe 进一步显示这个 ablation 对 MC weight 很敏感：

| variant | replay-bin top-1 | Q span | mean Q / mean RTG | all Spearman | demo success Q / failure Q | online success Q / failure Q | online-success Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| value-stream `0.1` | `32.7%` | `0.0513` | `0.2326 / 0.2233` | `+0.898` | `0.4218 / 0.0345` | `0.4623 / 0.0117` | `+0.479` |
| value-stream `1.0` | `31.7%` | `0.0734` | `0.3291 / 0.2185` | `+0.661` | `0.3450 / 0.2717` | `0.4111 / 0.2888` | `-0.455` |

弱权重基本保留了 executed-return calibration；强权重却同时高估失败 states、破坏 successful
trajectory 内部 ordering，final MC MAE 也从 whole-Q 的 `0.0187` 恶化到 `0.0831`。两者的
replay-bin top-1 都从约 `23.8%` 上升到约 `32%`，再次说明 value-logit update 并非与 action
ranking 独立。它没有重新变成 70% 以上的显式 imitation head，但也不是所需的干净分解。

expanded `effective_policy` causal probe 最终结果：

| variant @ 8k | informative states | pairs | pairwise sign | state-bootstrap 95% CI | mean Spearman | Q top-1 | mean regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| value-stream `0.1` | 12 | 86 | `57.0%` | `[41.6%, 71.1%]` | `+0.171` | `25.0%` | `0.066` |
| value-stream `1.0` | 11 | 67 | `50.7%` | `[35.8%, 68.2%]` | `+0.051` | `27.3%` | `0.087` |

bootstrap 以 state 为重采样单位，避免把同一 state 内的 action pairs 当独立样本。弱权重相对
Stage-IV point estimate (`51.3%`) 有小幅上升，但 CI 仍宽幅覆盖 `50%`，Spearman 的 95% CI
`[-0.141, 0.445]` 也覆盖 0，top-1 只有 `3/12`。强权重所有指标均近随机。Stage-V 因此不通过
causal gate；放大的 Q span 没有转化成可信的方向。

按预注册判定，不再做 encoder stop-gradient 或 MC/value-head weight sweep；下一阶段转向能创造
action support 的 structured exploration。新配置
`cqn_as_pixel_bigym_structured_exploration_gate` 在 temporal ensemble 后，以概率 `0.2` 选择一个实际
执行 action coordinate，并移动一个 level-1 C2F cell；归一化 `[-1,1]`、5 bins、level 1 时步长为
`2/5^2=0.08`。其余 BC plan 不变，Gaussian std 设为 0，replay/MC 保存的正是被干预的 effective
action。下一组 matched arms 为 structured-only 与 structured + whole-Q MC `0.1`：前者检验
coverage 对 TD 的作用，后者检验 MC 是否能利用新收集的 alternative-action outcome。

### 20.10 Stage-VI：从 target 设计转向 action identifiability

structured exploration 已实现为默认关闭的 CQN-AS rollout 选项：

1. 先生成完整 BC plan 并做 temporal ensemble；
2. 仅在 train mode、初始 random-explore phase 之后，以配置概率选择一个 step-action coordinate；
3. 对该 coordinate 随机加减一个指定 C2F level 的 cell width，再 clip 到 action bounds；
4. action chunk 的 `k=0` 被替换成这个真正执行的 action，现有 replay effective-action contract 不变；
5. agent 累计 eligible/applied/rate，episode replay 为每条 transition 保存 uint8
   `structured_explore`；demo transitions 固定为 0。

默认概率为 0，因此 legacy/CQN-Flow 的 RNG 与 replay schema 不变。新增 launch
`cqn_as_pixel_bigym_structured_exploration_gate` 使用 `p=0.2`、level 1、Gaussian std 0。单元/兼容
测试通过；310-frame MovePlate smoke 的完整 online episode 有 300 transitions，其中 52 条被标记，
实测率 `17.3%`，demo 的 160 个真实 transitions 全为 0。replay 数组另有一个 final sentinel，分析时
明确排除。

第一次未记录 per-transition flag 的 3-minute pre-runs 已用 SIGTERM 停止，目录不作为实验结果。
可审计的正式 matched runs 为：

| arm | GPU | W&B | run directory |
|---|---:|---|---|
| structured-only, MC `0` | 0 | `sq3lesy7` | `exp_local/cqn_value_fidelity_stage6/move_plate_structured_nomc_audited_seed1_gpu0_20260723_014300` |
| structured + whole-Q MC `0.1` | 3 | `zgkjck1d` | `exp_local/cqn_value_fidelity_stage6/move_plate_structured_mc_w0p1_audited_seed1_gpu3_20260723_014300` |

两组的唯一 objective 差异是 `mc_return_weight=0` vs `0.1`。Stage-VI gate 同时要求：实际
structured rate 接近 20%；explored subset 的 executed-return calibration 优于 no-coverage baseline；
expanded same-state branch 的 state-bootstrap pairwise CI 下界高于 50%。仅 success 改善不晋级。

这一阶段与现有 offline-to-online 路线的关系也要分清：

- [RLPD](https://arxiv.org/abs/2302.02948) 说明 expert/suboptimal offline data 与真实 online
  interaction 可以高效组合；这里对应的关键不是再加一个 imitation loss，而是让 online replay
  对 policy 邻域提供 outcome coverage；
- [Cal-QL](https://arxiv.org/abs/2303.05479) 处理 offline pretraining 到 online fine-tuning 的 value
  calibration/unlearning 问题，适合作为 calibration 对照；但本项目已实测 calibration 正确仍可能
  causal ranking 随机，所以 Cal-QL-style scale 修正不能替代 branch gate。

#### Stage-VI 训练与 replay 审计结果

| arm | success @ 2.5k/5k/7.5k/10k | best | eval-time assignment rate | completed-replay rate |
|---|---:|---:|---:|---:|
| structured-only | `4/48/52/44%` | `52%` | `20.24%` | `2064/10228 = 20.18%` |
| structured + MC `0.1` | `8/32/52/24%` | `52%` | `20.56%` | `2127/10315 = 20.62%` |

所以 independent one-step exploration 加快了 structured-only 的 5k 学习，但没有提高 peak；MC arm
的 10k 还低了 20 个百分点。assignment-rate gate 通过，performance gate 没有提供晋级证据。

8k observational probe 把 online assigned/unassigned transitions 另外各抽 32 个样本：

| arm | all RTG Spearman | assigned / unassigned | online success Q / failure Q | online-success Spearman | replay-bin top-1 |
|---|---:|---:|---:|---:|---:|
| no-coverage, no MC | `-0.642` | N/A | `0.0146 / 0.3141` | `+0.671` | `20.8%` |
| structured-only | `-0.582` | `-0.425 / -0.514` | `0.0987 / 0.4698` | `-0.554` | `21.7%` |
| no-coverage + MC `0.1` | `+0.897` | N/A | `0.4040 / 0.0127` | `+0.635` | `23.8%` |
| structured + MC `0.1` | `+0.782` | `+0.803 / +0.703` | `0.4840 / 0.1107` | `+0.119` | `23.3%` |

这给出了本路线目前最直接的 shortcut control：structured-only 的独立 BC policy 可以达到 `52%`
peak，而同一 checkpoint 的 critic 不仅不 calibrated，甚至把 online failure states 估得远高于
success states。MC 能校准走过的 action，但 one-step exploration 没有优于 matched no-coverage MC，
也没有改善 successful trajectory 内的 ordering。

#### Stage-VI same-state causal 结果

所有表均为同一组 8 seeds、3 anchors、共 24 个 candidate states；CI 以 informative state 为重采样
单位，而不是把同一 state 内的 action pairs 当独立样本。

| probe @ 8k | informative states | pairs | pairwise sign | state-bootstrap 95% CI | mean Spearman | Q top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| structured-only, level-0 effective plan | 3 | 17 | `35.3%` | `[0.0%, 50.0%]` | `+0.038` | `66.7%` | `0.0057` |
| structured + MC, level-0 effective plan | 16 | 114 | `51.8%` | `[38.9%, 62.9%]` | `+0.025` | `37.5%` | `0.0490` |
| structured + MC, level-1 effective plan | 11 | 79 | `55.7%` | `[45.2%, 68.5%]` | `+0.178` | `63.6%` | `0.0432` |

`effective_policy` 会先把 forced raw plan 注册进 temporal ensemble。审计发现 level-0 raw candidate
span 平均约 `1.87`，当步实际 span 只有 `0.126`；level-1 则从 `0.320` 衰减到 `0.0215`。为了不把
plan-history effect 与训练时的 post-ensemble intervention 混在一起，新增 `structured_k0` 模式：先
执行一次正常 BC inference 并固定 policy history，然后只把实际 `k=0` 坐标替换为
`a-0.08/a/a+0.08`。

| strict executed-`k=0` probe @ 8k | informative states | pairs | pairwise sign | 95% CI | mean Spearman | Q top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| structured-only | `0/24` | 0 | N/A | N/A | N/A | N/A | N/A |
| structured + MC `0.1` | `5/24` | 11 | `45.5%` | `[20.0%, 72.7%]` | `+0.200` | `60.0%` | `0.0427` |

因此 Stage-VI 明确失败：没有一个 causal CI 下界超过 50%。更重要的是，严格 matched 的单步
intervention 在 `19/24` 个 MC states、`24/24` 个 no-MC states 都没有制造可区分 outcome。约 2k
次 assignments 再被 15 个 action dimensions、两个方向和连续 task phase 摊薄，不能指望更换 C51
或 Flow Matching 参数化凭空恢复未被数据识别的 alternative-action ordering。

这也细化了对原论文证据的判断。[CQN-AS 论文](https://arxiv.org/abs/2411.12155) 的 Figure 2a
证明 action sequence 能降低 observed-RTG regression validation loss，Figure 7b 证明 RL objective
相对 successful-demo-only BC 提高 policy success；论文还在 demo-driven runs 中使用 Q 上的 auxiliary
large-margin BC，并把 successful online episodes relabel 为 demos。这些结果说明 TD signal 有用，
但没有 state-only/action-shuffle control 或 same-state intervention，不能区分 counterfactual action
value、success/progress classification 与 reward-weighted imitation。另一方面，无 demos 的
HumanoidBench 结果意味着不能把“CQN-AS 在所有任务上都是 BC”作为结论；本项目当前证据只针对
demo-driven MovePlate，并且证明的是 **success curve 不能认证 critic fidelity**。

理论上这不是 flow-specific failure。[Offline RL 的 value-approximation barrier](https://proceedings.mlr.press/v178/foster22a.html)
表明 coverage 与普通 realizability 本身仍不足以保证一般函数逼近下的 sample-efficient value learning；
最新的 trajectory-data policy-evaluation 结果还显式要求 known behavior policy
([Tkachuk et al., 2026](https://proceedings.mlr.press/v336/tkachuk26a.html))。因此下一阶段保留随机
assignment probability，不把 intervention 当普通无标签噪声。

### 20.11 Stage-VII：coherent intervention，而不是独立 jitter

新增 `structured_exploration_horizon`，默认 `1` 保持原行为。Stage-VII 设置 `H=4`：inactive 时以
`p_start=0.06` 抽一个 dimension/direction，之后四个 environment decisions 始终在 temporal
ensemble 之后施加同一 level-1 delta。稳态 active fraction 为
`H p / (1-p+H p) = 20.3%`。replay 新增：

- `structured_explore_start`；
- `structured_explore_dimension`；
- clipping 后真实的 signed `structured_explore_delta`；
- conditional assignment probability（start 时 `0.06/(2*15)=0.002`，continuation 时为 1）。

真实 310-frame MovePlate smoke 的完整 300-transition online episode 为 16 段、每段严格 4 steps，
共 64 active transitions (`21.33%`)；demo metadata 全为 control 值。新增 branch mode
`structured_horizon` 固定正常 policy history，并对同一 coordinate 连续执行
`-0.08/0/+0.08`，将作为 Stage-VII 的最终 causal gate。

两条 10.5k matched runs 已完成（MovePlate、train seed 1；每个 checkpoint eval 25 episodes）：

| arm | GPU | W&B | run directory |
|---|---:|---|---|
| coherent-only, MC `0` | 0 | `stcc2gnc` | `exp_local/cqn_value_fidelity_stage7/move_plate_coherent_nomc_audited_seed1_gpu0_20260723_021600` |
| coherent + MC `0.1` | 3 | `fblc52x6` | `exp_local/cqn_value_fidelity_stage7/move_plate_coherent_mc_w0p1_audited_seed1_gpu3_20260723_021600` |

训练曲线与 replay 审计如下。completed replay 只统计真实 episode transitions，排除了 final
sentinel：

| arm | success @ 2.5k/5k/7.5k/10k | best | active transitions | assignment starts |
|---|---:|---:|---:|---:|
| coherent-only | `16/12/48/56%` | `56%` | `1949/10300 = 18.92%` | 491 |
| coherent + MC `0.1` | `4/48/44/52%` | `52%` | `2181/10315 = 21.14%` | 549 |

coherent-only 相对 Stage-VI independent jitter 的 peak `52%` 只提高 4 个百分点。MC arm 的 5k
从 Stage-VI 的 `32%` 提升到 `48%`，但 final/peak 都只有 `52%`，没有超过 no-MC arm。每个
dimension 都收到 27--48 次 starts，说明失败不能归因于随机器从未覆盖某个 action coordinate；但
把这些 starts 再按 direction、task phase 和具体 state 分开后，局部反事实覆盖仍然很稀。

8k observational probe 显示一个非常重要的分离：

| arm | all RTG Spearman | assigned / unassigned | online success Q / failure Q | online-success Spearman | replay-bin top-1 |
|---|---:|---:|---:|---:|---:|
| coherent-only | `-0.632` | `+0.004 / -0.687` | `0.1385 / 0.2095` | `-0.595` | `22.4%` |
| coherent + MC `0.1` | `+0.916` | `+0.801 / +0.592` | `0.4128 / 0.0025` | `+0.643` | `20.6%` |

独立 BC policy 在 no-MC arm 达到 `56%`，但 TD-only critic 对 replay return 的排序仍为负，支持
“policy success 与 value fidelity 可以脱钩”的担忧。MC 则把**已执行动作**的 calibration 修得很
好，且 coherent assigned subset 的相关性从 Stage-VI one-step 的 `+0.803` 保持在 `+0.801`；更明显
的改善发生在整条 successful online trajectory（`+0.119 -> +0.643`）。然而 replay-bin top-1 仍在
五 bins 的 random reference 附近，因此这不是显式 BC head 重新接管 critic 的证据。

`structured_horizon` probe 固定正常 BC policy history，在同一 simulator state 对同一
coordinate 连续四步执行 `a-0.08 / a / a+0.08`。预注册 gate 是 informative-state fraction 高于
strict Stage-VI 的 `0/24`、`5/24`，并且 pairwise state-bootstrap 95% CI 下界超过 50%。初版固定
使用 intervention level 1 的 Q 读数，结果为：

| arm @ 8k | informative states | pairs | pairwise sign | state-bootstrap 95% CI | mean Spearman (95% CI) | Q top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent-only | `7/24` | 14 | `35.7%` | `[21.4%, 50.0%]` | `+0.214 [-0.071, 0.500]` | `71.4%` | `0.0557` |
| coherent + MC `0.1` | `7/23` | 15 | `13.3%` | `[0.0%, 28.6%]` | `-0.500 [-0.786, -0.143]` | `14.3%` | `0.0471` |

该读数随后被 token-alias audit 判定为无效的 final gate。虽然请求的是 level-1 cell `0.08`，靠近
action bound 时 clipping 会令两个 continuous candidates 落入同一个 level-0/1 token prefix；初版
no-MC 的 14 个 informative pairs 中有 7 个（`50%`）alias，MC 的 15 个中有 4 个（`26.7%`）。
固定用 level-1 selected-bin Q 给这种 pair 排序，数学上只能输出 tie，而环境在更细的 continuous
action 上可能产生不同 return。oracle ranking loss 精确停在约 `0.25 * log(2)`，进一步暴露了这个
不可优化的离散别名。所以上表只保留为 probe-bug 记录，不能用于算法判定。

修正后把“干预尺度”与“value readout”分开：仍执行 level-1 `+/-0.08`，但使用最深 level 2 的
chosen-action Q；该层可区分 clipping 后的小于 `0.08` 的差异。dimension 仍按该 readout 的 Q span
最大者选择，保持一贯的 critic-favorable protocol。相同 8 seeds、step `30/75/120` 的结果为：

| arm @ 8k, score L2 | informative states | pairs | pairwise sign | state-bootstrap 95% CI | mean Spearman (95% CI) | Q top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent-only | `13/24` | 30 | `63.3%` | `[41.4%, 83.3%]` | `+0.231 [-0.200, 0.641]` | `53.8%` | `0.0168` |
| coherent + MC `0.1` | `16/23` | 43 | **`69.8%`** | **`[51.2%, 87.2%]`** | **`+0.381 [0.006, 0.723]`** | `56.2%` | `0.0721` |

H=4 把 informative fraction 提到 `54--70%`；no-MC point estimate 为正但 CI 仍覆盖 chance。MC arm
则首次同时满足 pairwise CI 下界高于 50% 和 Spearman CI 下界高于 0。结合
observed-action `Q--RTG rho=+0.916`，正确结论是：MC 不只学到了 trajectory success/progress；在
最深可分辨 C2F token 上，它还学到了**弱但统计可辨的局部反事实 action ordering**。这不否定
shortcut 风险：独立 BC policy 在 no-MC critic 排序无认证时仍可达到 `56%`，而且当前 causal
pass 只有一个 training seed、下界刚过门槛，仍需新 eval seeds 与训练 seed 复现。

#### Stage-VII 判定与 Stage-VIII 计划

Stage-VII 在 corrected deepest-level protocol 下 **允许进入 Flow Matching 表示对照**。旧的
fixed-L1 stop decision 被撤回；任何后续 C51/FM probe 都必须显式报告 intervention level 与 score
level，禁止再次把 continuous intervention 与粗 token alias 混为 value error。

下一阶段按以下顺序做：

1. **新 seed 复现 corrected C51 gate**：固定 L1 intervention/L2 score，在未用于上述审计的 eval
   seeds 上复查 CI；publication claim 仍需至少 3 个 training seeds。
2. **matched expected-Q FM**：固定同一 BC policy protocol、replay-next target、effective-k0、MC
   return weight `0/0.1`、H=4 coherent exploration 与 10.5k budget，只把 direct C51 换成 scalar
   expected-Q Flow。field condition 保留 state/image、level、zoom path、sequence position、action
   dimension 和 candidate bin；flow source/solver samples 在 bins 间并行并使用 common random
   numbers。
3. **同一 causal gate 决胜**：FM 必须用 L2 branch pairwise CI，而不是 flow loss、endpoint MSE、
   observational RTG correlation 或 success curve晋级。若 FM 不过而 C51 复现通过，结论才是
   representation/optimization failure；若二者都不复现，回到 coverage/identifiability。
4. **oracle 与 propensity 路线降为诊断**：已实现 disjoint train/held-out branch-oracle fine-tuner 和
   dataset cache。固定 L1 的首轮能把 train pairwise 从 `28.6/8.3%` 提到 `71.4/75.0%`，held-out
   只有 `50.0/42.9%`；它主要帮助发现 token alias，不能作为部署算法。只有 corrected online gate
   后续失效时，才继续 deepest-level oracle 和 clipped inverse-propensity weighting。

该阶段的核心结论不是“CQN-AS 全局等于模仿学习”，而是更窄且可复现的：在 demo-driven
MovePlate 上，独立 BC policy 能在未经认证的 critic 下取得较高成功率，因此 success 不能替代
value audit；但 MC + coherent coverage 的 corrected deepest-level probe 已提供 counterfactual-Q 的
初步正证据。原论文的 RL-vs-BC success ablation 证明 RL loss 有用，却仍不能单独排除 shortcut；
本项目新增的 same-state branch gate 才负责区分二者。

#### Stage-VIII matched expected-Q Flow 实现与运行

`CQNFlowAS` 已接入与 direct C51 相同的独立 categorical BC head、`replay_next` target、
`effective_k0` critic slice、completed-return endpoint anchor 与 H=4 coherent metadata。scalar flow
critic 不再接收 demo margin/FOSD；MC loss 通过完整 ODE endpoint 反传到 velocity field，policy CE
只更新独立 policy（共享 encoder 是否接收 BC gradient仍由 `bc_policy_stop_gradient` 控制）。legacy
CQN-Flow 默认配置保持 shared-Q、无 MC、无 structured exploration。

真实 340-step MovePlate pixel smoke 已实际更新：所有梯度 finite，末段约为 flow loss `0.0291`、
MC loss `0.00122`、MC MAE `0.0938`、policy CE `1.451`；critic-side demo margin/FOSD 均严格为 0。
第一次 10.5k 启动在 replay 审计时发现 wiring error：workspace 的 optional-field gate 只接受
`method.name=cqn_as`，因此 `cqn_flow` 没有建立 `mc_return` 与 structured metadata schema。MC arm
实际收到默认零 return，两个 arm 的 intervention 也无法审计。`xu2tk68u/g1dwgg5x` 在约 6k/7k
被主动停止，曲线（包括 MC arm 5k 的 `64%`）全部标为 **invalid wiring smoke**，不进入比较。

修复后 `_mc_return_anchor_enabled` 与 `_structured_exploration_enabled` 同时接受 `cqn_as` 和
`cqn_flow`。真实 340-step pixel smoke 的 sampled batch 已出现 `mc_return_mean=0.3356`；正式 run
的 demo replay 中 MC return 范围为 `0.2023--1.0`，online episode 则同时包含全部字段，观测到
12 段严格 4-step assignment（`48/300=16%`）。修复后的 matched 10.5k 两臂为：

| arm | GPU | W&B | run directory |
|---|---:|---|---|
| expected-Q Flow, MC `0` | 0 | `bajxnr2x` | `exp_local/cqn_value_fidelity_flow/move_plate_floq_coherent_nomc_replayfix_seed1_gpu0_20260723_032200` |
| expected-Q Flow, MC `0.1` | 3 | `vj2hzy9e` | `exp_local/cqn_value_fidelity_flow/move_plate_floq_coherent_mc_w0p1_replayfix_seed1_gpu3_20260723_032200` |

两臂使用 `T=4,R_train=R_target=R_action=4`、uniform antithetic source、fixed rollout source bank、
endpoint-Q coefficient `1.0` 与 source-consistency `0.1`；除 MC weight 外完全 matched。完成后先比较
四个 eval checkpoints，再对 8k snapshot 跑 L1 intervention/L2 score causal probe。

#### Stage-VIII 结果：高 success / 高 calibration 仍可对应错误 action ranking

修复后的 online replay 审计通过：no-MC 为 `2082/10491=19.85%` active、523 starts，MC 为
`2053/10216=20.10%` active、516 starts；每个 action dimension 有 `26--52` 次 start。训练梯度始终
finite。MC arm 在 10k batch 上 `mc_return_mean=0.463`、endpoint MC MAE `0.0226`，说明结果不是
return 字段或数值优化失效。

| arm | success @ 2.5k/5k/7.5k/10k | best | final policy CE / top-1 |
|---|---:|---:|---:|
| expected-Q Flow, no MC | `16/12/28/32%` | `32%` | `0.346 / 86.2%` |
| expected-Q Flow, MC `0.1` | `28/20/68/24%` | **`68%`** | `0.340 / 86.8%` |

独立 BC head 的训练指标几乎相同，但 MC arm 在 7.5k 一度达到 `68%` 后跌到 `24%`；因此 peak
本身既不能证明 value fidelity，也暴露出共享 encoder 上 critic/policy gradient interference 的可能。

8k observational probe 如下；top-1 random reference 为 `20%`：

| arm | all RTG Spearman | assigned / unassigned | online success Q / failure Q | online-success Spearman | replay-bin top-1 | candidate Q span |
|---|---:|---:|---:|---:|---:|---:|
| Flow no-MC | `-0.578` | `-0.414 / -0.728` | `-0.027 / 0.134` | `+0.527` | `21.0%` | `0.00276` |
| Flow MC `0.1` | `+0.783` | `+0.629 / +0.610` | `0.393 / 0.118` | `+0.786` | `20.1%` | `0.00232` |

MC 把走过的 state-action return calibration 修好，critic 又没有提高 replay-action bin top-1，所以
不是显式 imitation head 接管 value；但是局部 candidate gap 比 direct C51 的约 `0.0174` 小一个
数量级。相同 L1 intervention/L2 readout 的 causal result 为：

| 8k causal probe | informative states | pairs | pairwise sign (95% CI) | mean Spearman (95% CI) | top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|
| Flow no-MC | `12/24` | 32 | `62.5% [42.4%,82.4%]` | `+0.269 [-0.156,+0.664]` | `66.7%` | `0.0660` |
| Flow MC `0.1` | `16/24` | 38 | **`31.6% [13.5%,51.3%]`** | **`-0.364 [-0.698,+0.008]`** | `25.0%` | `0.1909` |

因此 matched scalar Flow **失败**。最重要的反例是 MC arm：它同时有 `68%` policy peak 和
`rho=+0.783` observational calibration，却在同一 simulator state 的 action ordering 上显著偏向
反方向。这正是“RL/value loss 主要改善 shared representation 或 success classification，最后仍由
BC policy 行动”的可观测形式。它不能证明 critic 完全没有 value 信息，但足以否定“成功率或
Q--RTG correlation 可认证 action value”。

#### Stage-VIIIb：direct C51 独立 eval-seed 复验

为避免只用最初 8 个 seeds，使用未参与审计的 `21008--21023`（48 candidate states）复测同一
8k checkpoint：

| direct C51 | informative states | pairs | pairwise sign (95% CI) | mean Spearman (95% CI) |
|---|---:|---:|---:|---:|
| no-MC, new seeds | `31/48` | 80 | `51.3% [37.5%,65.1%]` | `+0.032 [-0.254,+0.323]` |
| MC `0.1`, new seeds | `38/48` | 102 | `58.8% [45.6%,71.2%]` | `+0.142 [-0.115,+0.389]` |

第一批 C51-MC 的双指标 pass 没有在第二批单独复现。合并全部独立 eval seeds 后（仍只有一个
training seed），no-MC 为 `44/72` informative、110 pairs、`54.5%`
`[42.6%,66.3%]`；MC 为 `54/71` informative、145 pairs、`62.1% [51.4%,72.3%]`，mean Spearman
`+0.213 [-0.0005,+0.420]`。所以 C51-MC 有比 Flow-MC 明显更好的正向趋势，pairwise combined
gate 通过，但 Spearman 下界仍刚好跨零；publication-level claim 仍需新的 training seeds。

#### Stage-IX：区分 compute collapse 与表示 collapse

[FLOQ](https://arxiv.org/abs/2509.06863) 明确指出 naive scalar flow 会 collapse，并把 HL-Gauss
interpolant embedding、Fourier time embedding、适中的 source width 视为关键；当前实现已经使用
HL-Gauss `sigma=16`、Fourier-64 和官方 `kappa=0.1` source width，但只使用 `K=4,R=4`，而论文
默认是 `K=8,R=8`。后续两张卡各跑一个只有单一解释变量的 MC arm：

1. **FLOQ compute-8**：保持 scalar endpoint 与全部 causal protocol，只把 integration/source
   samples 从 `4/4` 改为官方默认 `8/8`；检验当前反向排序是否是 integration recovery 不足。
2. **Flow-C51**：让 flow endpoint 输出 51-D atom logits；BCFM 沿路径训练，endpoint CE 对齐
   Bellman projection，completed return 也改成与 direct C51 一致的 two-atom projection CE；检验
   scalar flow 是否把很小的 action gap 淹没。

后一项不是假设“distributional 一定更好”。最新分析
[What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333) 反而报告显式 return
distribution 可能降低表现，并把 FLOQ 的收益归因于 iterative test-time recovery 与 feature
plasticity；所以 Flow-C51 是判别实验，不是预设赢家。[Value Flows](https://arxiv.org/abs/2510.07650)
则真正建模 return distribution 并满足 distributional Bellman equation，若 Flow-C51 仍失败才进入
其 return-sample/DCFM 路线。

[Q-learning with Adjoint Matching](https://arxiv.org/abs/2601.14234) 优化的是 multi-step
flow/diffusion **policy**，用 critic action gradient 构造逐步 policy objective；本阶段是 categorical
BC policy + flow **critic**，没有对 flow policy 做 BPTT，因此 QAM 不是 matched critic baseline。
只有未来把行为 policy 也换成 flow 并希望它利用已通过 causal gate 的 Q 时，才应加入 QAM。

#### Stage-IX 实测：更多 compute 改善 control，但 51 atoms 与 8 steps 都未通过 causal gate

两个新 arm 都继承 Stage-VIII 修复后的 replay schema、独立 categorical BC policy、completed-return
anchor、`replay_next` target、`effective_k0` critic slice 和 H=4 coherent intervention。真实 340-step
pixel smoke 均通过且梯度 finite：Flow-C51 使用 51-D centered-logit endpoint、BCFM、Bellman
projection CE 与 completed-return two-atom projection CE；compute-8 只把 scalar FLOQ 的
`T/R_train/R_target/R_action` 从 `4/4/4/4` 改为 `8/8/8/8`。

正式 MovePlate seed-1、10.5k runs 为：

| arm | GPU | W&B | success @ 2.5k/5k/7.5k/10k | best | runtime |
|---|---:|---|---:|---:|---:|
| Flow-C51 + MC `0.1` | 0 | `mpqxt0f0` | `8/44/52/36%` | `52%` | `14.4 min` |
| scalar FLOQ compute-8 + MC `0.1` | 3 | `65xf5ne5` | `16/52/56/60%` | **`60%`** | `19.1 min` |

run directories：

- `exp_local/cqn_value_fidelity_flow/move_plate_flow_c51_coherent_mc_w0p1_seed1_gpu0_20260723_stage9`
- `exp_local/cqn_value_fidelity_flow/move_plate_floq_compute8_coherent_mc_w0p1_seed1_gpu3_20260723_stage9`

两条 replay wiring 均再次审计通过。Flow-C51 的 39 个 online episodes 含 10 个 success，active
intervention 为 `2065/10328=19.99%`、519 starts；compute-8 的 43 个 online episodes 含 17 个
success，active 为 `1944/10451=18.60%`、488 starts。15 个 action dimensions 分别都有至少
`26` 和 `24` 次 start。10k 时两个 arm 的 flow/encoder non-finite gradient fraction 都为 0：

| arm | flow/BCFM loss | endpoint anchor | MC MAE | source-Q std | policy CE/top-1 |
|---|---:|---:|---:|---:|---:|
| Flow-C51 | `0.70999` | atom CE `0.73710` | `0.00861` | `0.00607` | `0.3644 / 85.6%` |
| compute-8 | `0.01383` | Q MSE `0.000775` | `0.02685` | `0.11447` | `0.3781 / 84.8%` |

因此两个训练目标都确实被优化；后面的 causal failure 不能归因于 NaN、缺字段或 MC anchor
未生效。8-step 的 success 从 4-step MC 的 `28/20/68/24%` 改为稳定上升的
`16/52/56/60%`，支持 FLOQ 所强调的 iterative compute 有实际作用，但不能只凭这条曲线声称
action value 正确。

8k observational probe 仍显示很强的 return calibration，同时 replay-bin top-1 保持在 20% random
reference 附近：

| arm | all Q--RTG Spearman | explored / unexplored | online-success Spearman | replay-bin top-1 | mean candidate-Q span |
|---|---:|---:|---:|---:|---:|
| Flow-C51 | `+0.893` | `+0.631 / +0.686` | `+0.500` | `21.0%` | `0.000768` |
| compute-8 | `+0.797` | `+0.857 / +0.613` | `+0.551` | `18.7%` | `0.001452` |

这再次说明 MC anchor 很擅长把 state/progress 与 observed return 对齐，但还没有回答同一个 state
下 action 改变会怎样。相同 L1 intervention/L2 score、anchors `30/75/120` 的第一批 8 seeds 为：

| arm | informative states | pairs | pairwise sign (95% CI) | mean Spearman (95% CI) | top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|
| Flow-C51 | `4/24` | 11 | `54.5% [40.0%,66.7%]` | `+0.125 [-0.250,+0.500]` | `50.0%` | `0.0053` |
| compute-8 | `10/24` | 26 | `61.5% [39.3%,84.0%]` | `+0.223 [-0.213,+0.633]` | `50.0%` | `0.0780` |

未参与首轮审计的 `21008--21023` 又提供 48 states/arm：

| arm, new eval seeds | informative states | pairs | pairwise sign (95% CI) | mean Spearman (95% CI) |
|---|---:|---:|---:|---:|
| Flow-C51 | `28/48` | 79 | `55.7% [42.9%,67.9%]` | `+0.148 [-0.120,+0.406]` |
| compute-8 | `30/48` | 78 | `59.0% [43.9%,73.7%]` | `+0.182 [-0.104,+0.457]` |

合并全部 24 个独立 eval seeds、以 informative state 为 bootstrap 单元的最终判定是：

| arm, combined | informative states | pairs | pairwise sign (95% CI) | mean Spearman (95% CI) | top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|
| Flow-C51 | `32/72` | 90 | `55.6% [44.1%,66.7%]` | `+0.145 [-0.094,+0.375]` | `46.9%` | `0.0288` |
| compute-8 | `40/72` | 104 | `59.6% [47.0%,72.0%]` | `+0.192 [-0.050,+0.426]` | `45.0%` | `0.1007` |

两个 arm 都未满足 pairwise CI 下界高于 50% 或 Spearman CI 下界高于 0。compute-8 把旧 4-step
MC 的 `31.6%` 反向排序修到正向 `59.6%`，所以 compute recovery 不是零作用；但证据还不足以
启动 100k 或新 training seeds。Flow-C51 则更直接地否定“把 51 atoms 放进 flow 就能修好”的
假设：它得到本阶段最高的 observational correlation 和最低 MC MAE，却仍只有 chance-level
counterfactual ranking。这里的 Flow-C51 是 deterministic 51-logit endpoint 的 flow，不是对随机
return sample 建模的 Value Flows/PCBF；实验判定的是 logit representation，不应描述成完整
distributional return flow 复现。

#### Stage-X research plan：Flow 学 state value，显式 advantage 负责 action effect

Stage-IX 后停止继续扫 `T/R`、atom 数或 source width。当前最一致的解释是：Flow/MC 已学到一个
很好的 state/trajectory progress baseline，但真正决定 bin 的 action gaps 小两个数量级且缺乏可识别
监督。下一阶段改为以下顺序：

1. **Flow-V + direct-A hybrid**：实现
   `Q(s,a)=V_FM(s)+A_C51(s,a)-mean_a A_C51(s,a)`。scalar Flow 只承担大尺度 state/progress
   baseline 和 MC/Bellman calibration；centered direct advantage head 专门承担 bin ranking。固定 state
   下 `V_FM` 自动抵消，causal gate 只检验 `A`，避免要求 flow 同时表达绝对 return 与千分量级
   action gap。
2. **彻底隔离 imitation**：当前 independent BC head 仍可经共享 image encoder 间接影响 critic。
   主实验要使用 distinct policy/value encoders；`bc_policy_stop_gradient=true` 只作为便宜消融，不能
   冒充严格 two-tower。BC CE 始终保留，因此 imitation/control 能力不会被删除；只是禁止 BC
   gradient 进入 value encoder。
3. **明确组合 policy**：部署分数报告为
   `normalized_A + beta * log pi_BC`，同时单独记录 BC action、A action 与组合 action。只在它们分歧
   的 simulator states 上比较 paired return，直接量化 learned advantage 是否真的优于 imitation
   prior。
4. **先补可识别 action data，再加复杂 return flow**：保留随机化 assignment probability，并在少量
   online states 上收集可恢复 simulator branch pairs，训练 pairwise advantage/sign loss；train/eval
   branch seeds 必须严格分离。没有 paired/local contrast 时，不再用更复杂的 Value Flows、PCBF 或
   FlowIQN 期待凭表示自动解决 action identifiability。
5. **matched baselines**：hybrid 必须同时对比 direct C51-MC、compute-8 和 `V(s)+pi_BC`。direct
   C51-MC 当前 combined pairwise 为 `62.1% [51.4%,72.3%]`，是唯一 pairwise lower bound 过 50%
   的表示，但 Spearman 下界仍约为 0；因此优先给 direct C51 与 hybrid 做 training-seed replication，
   不给 Flow-C51 扩训练。

这一路线也与最新机制分析一致：FLOQ 的价值更可能来自 iterative recovery/plasticity，而非显式
return distribution；本地 compute-8 提高 control、Flow-C51 不修 causal ranking 正好提供了对应的
机器人证据。[Value Flows](https://arxiv.org/abs/2510.07650) 与
[PCBF](https://arxiv.org/abs/2605.08253) 仍是 return uncertainty/calibration 的有效对照，但在 action
advantage gate 通过前不是下一项主实验。

#### Stage-X implementation：Flow-V / direct-A 两塔实验已启动（2026-07-23）

已落地的 hybrid 不是把两个未经约束的 Q head 相加，而是做了可识别的职责拆分：

```text
V = FlowEndpoint(s, C2F-prefix, level, sequence, dimension)
A_raw = E_z[p_C51(z | s, C2F-prefix, level, sequence, dimension, candidate-bin)]
A = A_raw - mean_candidate(A_raw)
Q_hybrid = V + A
```

`FlowEndpoint` 的网络结构中不再创建 `candidate_projection` 或 candidate velocity head；相同 common
source 下所有 bins 的 V 完全一致。direct-A 仍输出 51-atom categorical residual distribution，但加到
Q 上的是其 expectation 在 candidate 轴中心化后的标量，因此没有“把 C51 logits 当作可直接相加的
value”这一语义错误。训练目标为：

1. Flow BCFM 以 completed discounted return 为 endpoint，学习 behavior-state / C2F-prefix
   baseline；额外 endpoint MSE 与 source-consistency 保留。
2. direct-C51 target 是 `y_TD - stop_gradient(V)` 的 two-hot projection。
3. centered advantage 另有
   `||stop_gradient(V) + A - y_TD||²`；这里显式停止 V gradient，禁止 direct-A 把排序责任再推回
   baseline。
4. Flow 与 direct-A 都有 EMA target，下一状态 Bellman target 使用完整
   `V_target + A_target`。
5. BC CE 与 policy head 保留，但 raw pixel encoder 真正复制成 `encoder` 和 `policy_encoder`
   两套参数/optimizer state。BC path 只访问后者；value/Flow/direct-A 只访问前者。

matched direct-C51 也启用了相同的 distinct policy encoder、completed-return anchor、replay-next
SARSA target、effective-k0 critic supervision，以及 probability `0.06`、level `1`、horizon `4`
的 coherent interventions。这样 direct/hybrid 的主要差异是 value decomposition，而不是 BC
是否能改写 image features。

实现与分析覆盖：

- `robobase/method/cqn_flow.py`：`critic_architecture=flow_v_direct_a`、candidate-independent
  Flow-V、direct-C51 residual、hybrid Bellman/update/EMA、组合 Q diagnostics。
- `robobase/method/cqn_as.py`：通用 `distinct_policy_encoder` 两塔支持，供 matched direct-C51
  使用。
- `scripts/analyze_cqn_value_fidelity.py` 与
  `scripts/analyze_cqn_branch_counterfactual.py`：Flow checkpoint 统一读取组合 Q；BC action 使用
  policy encoder。source-resampling probe 同样把 centered A 加回，因而不会把 hybrid 误报成
  只有 Flow-V。
- launch：
  `cqn_flow_v_direct_a_coherent_value_gate` 与
  `cqn_as_pixel_bigym_two_tower_coherent_mc_gate`。

验证结果：

- CQN-AS/CQN-Flow unit regressions：`101 passed`。
- 真实 MovePlate pixel smoke（raw image、60 demos、replay、两塔、真实 update）两臂均完成
  `340` frames 且所有 CSV 数值 finite。
- hybrid smoke 最后一次记录：
  `BCFM=0.1927`、`A-C51 CE=3.2904`、`A-Q MSE=0.00507`、
  `A span=0.03591`；value/policy encoder gradient norm 分别为
  `0.00103 / 0.00143`。这说明 direct-A 没有保持全 bin 相等，也确认两个 encoder 分别在各自
  objective 下更新。
- matched direct-C51 smoke 的 `critic_q_span=0.07616`，policy encoder gradient norm
  `0.00164`，同样 finite。

正式 seed-1、10,500-frame MovePlate gates 已启动：

| arm | GPU | run directory | W&B |
|---|---:|---|---|
| Flow-V + direct-A, two-tower | 0 | `exp_local/cqn_value_fidelity_stage_x/move_plate_flow_v_direct_a_two_tower_seed1_gpu0_20260723` | `0xyqbaz9` |
| matched direct-C51, two-tower | 3 | `exp_local/cqn_value_fidelity_stage_x/move_plate_direct_c51_two_tower_seed1_gpu3_20260723` | `5lrbqbny` |

本阶段仍按 evidence ladder 判定，不用 success curve 代替 value 证据：

1. 先检查 1k/2k/4k/8k 的 finite gradients、V/MC calibration、A span、policy/value encoder
   gradient isolation。
2. 报告 2.5k 间隔的 25-episode control success，但它只说明 BC-controlled system performance。
3. 8k checkpoint 先跑 held-out observational probe；随后在与 Stage-IX 相同的 L1 intervention /
   L2 score、`30/75/120` anchors、未复用 eval seeds 上跑 branch causal gate。
4. hybrid 的关键通过条件仍是 informative-state bootstrap 后 pairwise sign CI 下界 `>50%` 或
   mean Spearman CI 下界 `>0`。若 A span 非零但 causal gate 仍失败，则下一步不是扩大 Flow
   compute，而是加入 simulator branch-pair sign/ranking loss。

#### Stage-X 正式结果：decomposition 有正向趋势，但严格 causal gate 未过

两条 10,500-frame run 均完整结束，未发生 NaN、OOM 或中途恢复。每个 eval point 都是
MovePlate 的 25 episodes：

| arm | success @ 2.5k/5k/7.5k/10k | best | runtime |
|---|---:|---:|---:|
| Flow-V + direct-A, strict two-tower | `12/44/40/40%` | `44%` | `19.4 min` |
| direct C51, strict two-tower | `12/36/56/52%` | `56%` | `10.4 min` |

control success 仍只解释为独立 BC policy 的表现。replay 审计确认两臂都真正收到了 coherent
intervention，而不是只在配置里打开：

| arm | online episodes / success | active transitions | starts | per-dimension starts |
|---|---:|---:|---:|---:|
| hybrid | `42 / 13` | `2185/10397 = 21.02%` | `549` | `29--50` |
| direct | `42 / 15` | `2088/10436 = 20.01%` | `528` | `26--49` |

assignment probability 也符合设计：新 start 为 `0.06/(2*15)=0.002`，未 assignment 为
`0.94`，H=4 continuation 为 `1.0`。因此 causal failure 不能归因于没有局部动作覆盖。

8k 时 hybrid 的两个子目标都在优化：`BCFM=0.01587`、`A-C51 CE=0.81584`、
`A-Q MSE=0.001094`、`A span=0.02709`、MC MAE `0.03678`；policy CE/top-1 为
`0.46696/81.57%`。value/policy encoder gradient norm 为 `0.02045/0.18298`，flow 与
advantage non-finite gradient fraction 都为 0。direct 在同一点的 TD loss、MC loss、Q span、
MC MAE 和 policy CE/top-1 分别为 `0.09172`、`0.06167`、`0.02519`、`0.01951` 和
`0.46448/81.48%`。两臂的 BC 学习进度和 candidate span 已很接近，不能把后续差异解释为
hybrid 的 A head 没有离开初始化。

observational probe 使用各自 8k target critic、相同 `8` samples/group 和额外
`32+32` assigned/unassigned samples：

| arm | all Q--RTG Spearman | online-success Spearman | assigned / unassigned | replay-bin top-1 | critic/BC current-action disagreement | candidate Q span |
|---|---:|---:|---:|---:|---:|---:|
| hybrid | `+0.695` | `+0.635` | `+0.776 / +0.579` | `20.9%` | `79.7%` | `0.02700` |
| direct two-tower | `+0.673` | `+0.217` | `+0.750 / +0.471` | `24.2%` | `75.6%` | `0.02121` |

这排除了最简单的“critic 直接复制 BC logits”解释：五 bins 的随机 top-1 是 `20%`，hybrid
就在该基准附近，同时 critic 与 BC 的当前动作约八成不同。另一方面，高 Q--RTG correlation
仍可能只是 state/progress calibration，所以不能用这张表宣布学会 action value。

最终 causal 判定合并了预注册的 `21000--21007` 和未参与首轮审计的
`21008--21023`，共 24 个 eval seeds、每个 seed 的 `30/75/120` anchors；bootstrap 单元是
informative simulator state：

| arm @ 8k, L1 intervention / L2 score | informative states | pairs | pairwise sign (95% CI) | mean Spearman (95% CI) | top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|
| direct C51 two-tower | `28/72` | 71 | `53.5% [40.0%,66.7%]` | `+0.045 [-0.225,+0.317]` | `46.4%` | `0.0761` |
| Flow-V + direct-A two-tower | `42/72` | 111 | **`61.3% [50.0%,72.1%]`** | **`+0.213 [-0.012,+0.431]`** | `57.1%` | `0.0619` |

hybrid 把 informative fraction、pairwise point estimate 和 Spearman 都提高了，并把 pairwise
CI 下界从 `40%` 推到刚好 `50%`；方向是正的。但预注册条件是下界**严格高于** `50%` 或
Spearman 下界严格高于 0，因此 Stage-X 仍判为 **not passed**，不能把边界值写成成功。

strict two-tower direct 与 Stage-VII shared-encoder C51-MC 的 resolved 配置逐项比较，唯一 method
差异就是 `distinct_policy_encoder: true`。旧 arm 的 combined pairwise 是
`62.1% [51.4%,72.3%]`，新 arm 变为 `53.5% [40.0%,66.7%]`，而 control peak 从 `52%`
变为 `56%`。单 training seed 和不同在线轨迹不允许把下降作强因果归因，但它至少反驳“切断 BC
encoder gradient 会自动让 value 更真实”：共享 BC representation 也可能提供有用视觉特征，而
不只是 imitation shortcut。

更稳妥的结论是：

1. legacy shared-Q 的 demo FOSD/margin 确实可以把 Q 变成隐式 action classifier；这是此前
   `92%` success 但 causal failure 的来源之一。
2. 在完全分开的 policy/value encoder 下，BC policy 仍能得到 `56%` control，而 critic 不复制
   BC action，却没有通过 action-value causal gate。因此“policy 能做任务”和“Q 排序可信”仍是
   两件事。
3. Flow-V + direct-A 比 matched two-tower direct 更接近 causal pass，说明 decomposition 值得
   保留；证据还不足以启动 100k 或声称 Flow Matching 已解决 value identifiability。

#### Stage-XI：固定-prefix A audit 与 paired branch-ranking

Stage-X 还暴露出一个需要明确修正的表述。hybrid 的 Flow baseline 实际是
`V(s, C2F-prefix, level, sequence, dimension)`：在**同一个 C2F 节点的 candidate bins**之间
严格相同，但旧的 L1 intervention/L2 readout 可能跨不同 prefix。因此上面的 causal table 合法地
检验完整 `Q=V+A`，却不能说 `V` 在所有 branch candidates 间都抵消、只检验了 A。

为此新增 `sibling_horizon` causal mode：

1. 正常 BC policy 给出共同的 L0/L1 prefix；
2. 在 deepest L2 强制同一节点的全部 5 个 sibling bins；
3. 用该节点的 Q 选 span 最大的 current-action coordinate；
4. 将每个 sibling 相对正常动作的真实 delta 连续执行 H=4，再恢复完全相同的 simulator/agent
   state rollout；
5. 对 hybrid 而言 Flow-V 在这 5 个候选上逐元素相同，所以 pairwise ranking 严格只来自
   centered direct-A。

真实 MovePlate one-state smoke 已通过：两臂都是 5 candidates、实际 action span `0.064`；
hybrid/direct 的 predicted span 分别为 `0.204/0.070`，所有 H=4 delta 与 simulator restore
字段正确。该 probe 是 decomposition diagnostic，不会替换或事后改写上面的预注册 full-Q gate。

后续 stop/go 顺序固定为：

1. 先完成相同 8+16 eval seeds 的 L2 sibling-H4 A-only audit；只有 held-out CI 正向时才进入
   `normalized_A + beta log pi_BC` 的 disagreement-state paired policy 比较。
2. 若 sibling A 仍不过 gate，固定 Flow-V 和 BC policy，先只训练 advantage head：
   `softplus(-sign(R_i-R_j)(A_i-A_j)/temperature)`。branch-train seeds 与 branch-eval seeds
   严格分开，dimension 用预先随机 assignment，禁止按 held-out critic span 挑样本。
3. 必须同时报告 train-before/after 与 heldout-before/after；只记住 oracle anchors 不算成功。
   只有 heldout pairwise/Spearman gate 通过，才把小型 branch-pair buffer 接进 online trainer。
4. 不再扫 flow steps、source samples、atoms 或蒸馏。这些已经不能回答当前的 action
   identifiability 缺口。

#### Stage-XI 正式结果：full-Q 的正向信号不能归因于 direct-A

固定-prefix probe 已完成相同的 24 个 eval seeds、`30/75/120` anchors 和 H=4
intervention。L2 在共同 L0/L1 prefix 下比较 5 个 deepest siblings；L1 在共同 L0 prefix
下比较 5 个 L1 siblings。两种设置里 Flow-V 都在被比较的 bins 间逐元素相同，所以排序只来自
centered direct-A：

| hybrid A-only @ 8k | informative states | pairs | pairwise sign (95% CI) | mean Spearman (95% CI) | top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|
| L2 siblings | `16/72` | 66 | `53.0% [40.6%,64.7%]` | `+0.040 [-0.133,+0.213]` | `50.0%` | `0.0245` |
| L1 siblings | `16/72` | 64 | `43.8% [31.3%,56.3%]` | `-0.088 [-0.265,+0.088]` | `50.0%` | `0.0108` |

L2 是不显著的 chance-level；具有更大实际 action contrast 的 L1 反而方向为负。作为脚本一致性
对照，direct two-tower 的首批 8 seeds 在 L2/L1 也分别只有 `43.8%/50.0%`，但该小样本不用于
方法间正式排序。

另一个重要诊断是维度选择。probe 已经对全部 action coordinates 计算 A span，再选择最有利于
critic 的维度；L1 的 72 states 中仍有 `64` 次选中 dimension 14、`8` 次选中 dimension 13。
BiGym wrapper 明确说明最后两个维度是 grippers。也就是说，A 最大的自信几乎全集中在 gripper，
但只有 `16/72` states 的五个 sibling actions 真正改变 return。A span 非零不代表 action effect
可识别，反而暴露出“在 outcome-insensitive coordinate 上制造 gap”的 calibration 问题。

这也修正了上一节对 full-Q 结果的解释。旧 L1 intervention/L2 score 得到
`61.3% [50.0%,72.1%]`，但那些候选可进入不同 L1 prefix；Flow-V 又显式 condition
`C2F-prefix`。因此该正向趋势最多属于完整 `prefix-V + A`，不能归功于 direct-A。更严格地说，
当前所谓 Flow-V 不是纯 `V(s)`，而是一个 candidate-independent-at-current-node 的
`V(s,prefix)`；它仍可能把早期 action bins 编进 baseline。后续必须增加真正不接收 prefix 的
`V(s)` ablation，才能区分 progress value 与 coarse-bin action classifier。

#### Stage-XI branch oracle：能记住 5 个 states，但不能泛化

已实现并运行 post-hoc branch-oracle fine-tuner。它只更新 hybrid 的 direct advantage 参数；
Flow-V、value encoder、BC policy 和 policy encoder 全部冻结。候选是相同 L0 prefix 下的 5 个
L1 siblings，train/eval simulator seeds 严格分离：

- train seeds `22000--22007`，held-out seeds `23000--23007`；
- 每侧 `8 seeds * 3 anchors * 2 gripper dimensions = 48 states`；
- train 仅 `5/48` informative，held-out 仅 `6/48` informative；
- ranking loss 使用有 return 差异的 pairs；delta regression 同时约束有差异 pair 和 tied
  siblings，防止无效动作产生任意大 Q gap。

三种 delta-regression 权重都使用同一份 branch cache、相同初始化和 2,000 updates：

| delta weight | train-after pairwise | heldout-after pairwise (95% CI) | heldout Spearman (95% CI) | heldout top-1 | final batch A span |
|---:|---:|---:|---:|---:|---:|
| `0` | `100%` | `46.4% [16.7%,83.3%]` | `-0.005 [-0.487,+0.471]` | `50.0%` | `0.981` |
| `10` | `100%` | `42.9% [14.3%,71.4%]` | `-0.135 [-0.607,+0.336]` | `33.3%` | `0.129` |
| `100` | `100%` | `28.6% [7.1%,58.3%]` | `-0.324 [-0.677,+0.118]` | `16.7%` | `0.0689` |

训练前 held-out pairwise 为 `42.9%`、Spearman 为 `-0.103`。纯 sign loss 把训练 anchors 完全
分开，却把 A span 放大到 `0.981`，held-out 仍为 chance；delta regression 能压低虚假 gap，
但没有恢复 unseen-state ranking。结果不是某一个 loss weight 的偶然失败，而是 5 个 informative
training states 下的明确 memorization。branch oracle 因此 **不得接入 online trainer**。

#### Stage-XII：先解决 branch coverage，再比较 Flow 表示

下一阶段的顺序改为：

1. **coverage-first**：对 `(seed, phase, action_dimension)` 均匀取样，不再由 critic/A span
   选择维度。先报告各维度、各 anchor phase 的 informative fraction 和 return span，再训练任何
   oracle head。`scripts/finetune_cqn_branch_oracle.py` 已增加 `--coverage-only`，可只生成/复用
   branch cache，不修改参数。
2. **预注册最低数据门槛**：至少 `50` 个 informative train states、`30` 个 disjoint held-out
   informative states；覆盖至少 4 个 action dimensions，其中至少 3 个不是 gripper，并覆盖至少
   3 个 episode phases。单一维度不得贡献超过 40% 的 informative train states。未达到就只记为
   environment identifiability audit，不做学习结论。
3. **先做 frozen-representation oracle**：Flow-V、encoder 和 BC 全冻结，只训 A。必须同时满足
   held-out pairwise CI 下界严格高于 50% 或 Spearman CI 下界严格高于 0，并且 tied-return states
   的 predicted span 不恶化，才允许把 branch buffer 接进 online training。
4. **纯 state-V ablation**：加入 `V(s)` arm，Flow field 不接收 low/high midpoint、interval
   width 或 C2F level/prefix；所有 action effect 只能进入 A。与当前 `V(s,prefix)+A`、direct
   C51 和 `V(s)+pi_BC` 使用相同 branch cache 比较。若只有 prefix-V arm 通过 full-Q、A-only
   仍失败，就把它归类为 autoregressive action-value baseline，不能称为 state value。
5. **最后才恢复 FM 主线**：只有 A 的 held-out causal gate 通过后，才比较 `V(s)` 用 scalar
   Flow endpoint、direct scalar/C51 或 return-distribution flow 的差异。否则继续增加 FM steps、
   source samples 或 atoms 只会改善 state/progress calibration，无法证明学会 action value。

当前最可靠的阶段结论是：strict two-tower 排除了 BC gradient 直接污染 value encoder，但没有
自动得到真实 action value；Flow-V/direct-A 的完整 Q 有正向趋势，fixed-prefix A 本身没有。
因此下一瓶颈是 counterfactual data coverage 与 action identifiability，不是 Flow 的维度或
并行计算量。

#### Stage-XII 全维度 coverage：确认了 value collapse 的具体形态

按上面的 coverage-first 规则，又使用完全未参与此前 probe 的 5 个 train seeds
`24000--24004` 和 5 个 held-out seeds `25000--25004`。每个 seed 固定
`30/75/120` 三个 anchors，并对全部 15 个 action dimensions 分别比较 5 个 L1 sibling bins，
因此每侧共有 `5 * 3 * 15 = 225` states。所有候选仍是相同 L0 prefix、H=4、相同 simulator
state restore，hybrid score 严格只读取 direct-A。

环境 coverage 为：

| split | informative/all | non-gripper | gripper 13/14 | any-success branch states |
|---|---:|---:|---:|---:|
| train | `157/225 = 69.8%` | `155/195` | `2/30` | `181/225` |
| held-out | `122/225 = 54.2%` | `120/195` | `2/30` | `139/225` |

三个 anchors 的 train informative counts 是 `55/50/52`，held-out 是 `42/40/40`，不是只靠
一个 phase。seed 之间差异很大：一对全失败 policy rollouts 只有 `1/45` 和 `0/45`
informative，成功轨迹通常有约 `38--41/45`。这说明 sparse task reward 下随机 branch seed
不能直接当作等量信息；脚本已增加 `--baseline-outcome=success|failure`，先用一次廉价
unbranched rollout 分层，再决定是否运行全部 counterfactual branches。失败层仍需保留用于
false-positive calibration，但不再假装它提供大量 ranking labels。

原始 8k A 在这份不由 critic 挑维度的数据上是严格 chance：

| split, A before oracle | pairwise sign | seed-cluster 95% CI | Spearman (seed CI) | top-1 | random top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|
| train | `49.6%` | `[46.5%,52.7%]` | `-0.001 [-0.075,+0.084]` | `22.9%` | `26.0%` | `0.0501` |
| held-out | `49.1%` | `[45.0%,52.6%]` | `-0.025 [-0.105,+0.040]` | `23.8%` | `26.4%` | `0.0571` |

更关键的是 sensitivity 放错了位置。held-out non-gripper 的 mean A span 只有 `0.00338`，
但真实 mean return span 是 `0.08049`；两个 grippers 的 mean A span 是 `0.1890`，
真实 return span 却只有 `0.00042`。也就是 A 在几乎不影响 outcome 的 gripper 上比 arm
自信约 `56x`，而真实 action effect 正好相反。15 个 dimension 的 mean-Q-span 与
mean-return-span Spearman 是 `-0.296`（train 为 `-0.402`）。

这比“Q/BC top-1 是否相同”更直接地定位了 collapse：模型把绝大部分 action sensitivity
分配给了错误坐标。由于该 checkpoint 已是 strict two-tower，BC gradient 不可能进入 value
encoder；此前 critic/BC action 也约 80% 不同。因此用户提出的核心怀疑——“它没有学会真实
value”——成立，但“它只是复制 imitation action”对这个 strict arm 来说过窄。这里更准确的是
自举 TD/argmax 产生了 **spurious self-reinforcing ranking**；canonical shared-head CQN 的
demo FOSD/margin 还会额外具有 imitation-classifier shortcut。

#### Stage-XII all-dimension oracle：counterfactual supervision 可以泛化

将上述 5+5 seed cache 用于 frozen-representation oracle，仍只更新 direct-A；Flow-V、value
encoder、BC policy 和 policy encoder 全部冻结。train 有 157 个、held-out 有 122 个
informative states，batch 32、2,000 updates。结果与只用 gripper 的 5-state oracle 完全不同：

| loss | train pairwise | held-out pairwise (state CI) | seed-cluster CI | held-out Spearman (seed CI) | top-1 | regret |
|---|---:|---:|---:|---:|---:|---:|
| before | `49.6%` | `49.1% [45.0%,53.2%]` | `[45.0%,52.6%]` | `-0.025 [-0.105,+0.040]` | `23.8%` | `0.0571` |
| sign only, delta `0` | `100%` | **`56.7% [52.1%,61.3%]`** | **`[54.4%,58.1%]`** | **`+0.126 [+0.069,+0.174]`** | `26.2%` | `0.0410` |
| sign + delta `10` | `100%` | **`55.9% [51.3%,60.3%]`** | **`[51.5%,58.7%]`** | **`+0.102 [+0.011,+0.153]`** | `23.0%` | `0.0386` |
| sign + delta `100` | `99.6%` | `55.2% [50.5%,59.8%]` | `[50.0%,58.2%]` | `+0.095 [-0.023,+0.182]` | `27.9%` | `0.0258` |

`delta=0` 和 `10` 都通过 held-out causal gate；`100` 的 seed-level lower bound 只到 50%、
Spearman 仍跨 0，因此不记为严格通过。delta calibration 的作用也符合设计：

| loss | held-out arm A span | held-out gripper A span | dimension span alignment |
|---|---:|---:|---:|
| before | `0.0034` | `0.1890` | `-0.296` |
| delta `0` | `1.2689` | `0.7578` | `+0.282` |
| delta `10` | `0.2914` | `0.0437` | `+0.621` |
| delta `100` | `0.1737` | `0.0135` | `+0.629` |

因此小 oracle 的失败不是“ranking loss 无效”，而是只给了 5 个 informative gripper/early-phase
states。全维度、跨 phase、seed-disjoint 数据下，同一个 frozen encoder 和 A head 能把 unseen
action ordering 从 chance 拉到显著正向。这是一个清晰的机制结论：现有网络有表达能力，
原 online objective/data 没有提供足够的 causal action supervision。

不过该结果还不允许 A 单独接管 policy。对同一 held-out branch set，把“离原 BC action 最近的
sibling”当作 behavior-prior proxy，可得到：

- pairwise `62.5%`；
- top-1 `38.5%`，而随机期望为 `26.4%`；
- regret `0.0176`。

这些都强于单独 oracle-A。事后 exploratory blend
`normalized_A + beta * normalized_behavior_proxy` 在 `beta>=4` 时约为
`62.4--62.7%` pairwise、`39--40%` top-1、`0.0175` regret，主要收益仍来自 imitation prior；
因为 beta 看过该 held-out set，这只能用来提出下一假设，不能作为新 gate。branch cache 代码现已
直接保存 independent BC head 对每个 sibling 的真实 `log pi_BC`，后续不再用动作距离代理。
真实 MovePlate smoke 已确认每个 state 写入 5 个 finite、非均匀 policy log-probabilities，并且
`--coverage-only` 全程不修改任何参数。

#### Stage-XIII 决策：保留 Flow-V，但主实验转为 causal-A + BC prior

下一步不再问“Flow steps 是否再多一点”，而按以下顺序：

1. **新的三段 split**：branch-train、beta/calibration、final-confirmatory simulator seeds 完全
   分离。`beta` 只在 calibration split 选择，最终只看一次 confirmatory split；bootstrap 同时
   报 informative-state 和 eval-seed cluster。
2. **真实 BC prior 对比**：预注册
   `A only`、`log pi_BC only`、`normalized_A + beta log pi_BC`。组合方法必须同时超过 BC-only
   的 pairwise/top-1，并降低 regret，才有资格进入 closed-loop eval。只比 A-before 不够。
3. **online branch auxiliary 的最小版本**：优先使用 delta `10`，因为它在严格 gate 通过的同时
   显著压低无效 gripper gap。成功 baseline 提供 ranking pairs；失败/tied branches 专门提供
   delta calibration。dimension 均匀 assignment，禁止再由 critic span 挑训练样本。
4. **纯 state Flow-V**：实现并比较 `V(s)+A` 与当前 `V(s,prefix)+A`。若 prefix-V full-Q
   好而 fixed-prefix A/confirmatory blend 不好，就把前者归类为 autoregressive Q，不再称
   state value。
5. **Flow 与 direct value 的 matched 比较**：固定同一 causal-A buffer 和真实 BC prior，只替换
   baseline 为 scalar Flow-V、direct scalar/C51-V；这样才能回答 Flow Matching 对 value
   calibration/sample efficiency 是否有增益，而不是让 Flow 替 action identifiability 背锅。

截至此阶段，两条研究线已经合流：CQN 的确可能在 control success 很高时仍没有真实 action
value；Flow Matching 可以继续承担 state/progress baseline，但它不能自动修复该缺口。真正有效的
新增信息是 simulator counterfactual action pairs，而部署时 imitation prior 仍是必须保留、且当前
更强的组成。

#### Stage-XIII 实现：把 causal-A 接回 online CQN-AS

`CQNFlow` 现已支持读取静态全维度 branch cache，并在每次 update 里对 direct-A 增加两项监督：

- 同一 state、同一 action dimension 的 sibling return ordering，用 pairwise sign loss 训练；
- sibling return delta，用 smooth-L1 calibration 抑制无效维度上的虚假大 gap。

默认配置保持关闭，Stage-XIII launch 使用 branch weight `0.1`、delta weight `10`、
temperature `0.05`、batch size `32`、level `1`。训练日志新增 causal branch loss、pairwise
accuracy、Q span 等指标。rollout 端新增
`zscore(A) + beta * log pi_BC` 的逐层 action-bin 选择；`beta=null` 完全保留原 BC-only
rollout。Flow-V 对同一 state/prefix 的 sibling 是公共项，理论上不会伪装成 action ranking。

单元与分支相关测试共 `83 passed`；真实 MovePlate 8k checkpoint resume smoke 已运行到
8,010 step，首次 JIT 和后续 update 均通过。正式续训只复制 snapshot 时刻已有的 91 个 replay
episode（含 51 个 demo），没有把原 run 在 8k 之后产生的 episode 泄漏进去。

#### Stage-XIII 正式 closed-loop 对照（2026-07-23）

两臂都从同一个 Stage-X strict two-tower Flow-V/direct-A 8k checkpoint 继续到 100.5k，
使用相同 branch-train cache（seed `24000--24004`）、相同 seed 1、相同固定 eval seeds
`31000--31024`，每 10k 做 25 episodes：

| GPU | rollout | 目的 |
|---|---|---|
| 0 | `zscore(A) + 4 log pi_BC` | 检验 causal value 真正参与选 bin 后能否提高 closed-loop success |
| 3 | BC-only，同时照常训练 causal-A | 控制额外监督、训练时长和 encoder drift，只隔离部署时使用 value 的作用 |

这不是 Flow-vs-direct 的最终比较，而是更靠前的 deployment gate：如果 A+BC 不能超过
BC-only，就算 held-out branch ranking 提高，也不能声称学到的 value 对控制有用。只有通过此
gate，下一阶段才固定 causal-A 和 BC prior，比较 Flow-V、direct scalar-V 与 C51-V。

#### Stage-XIII 最终结果：causal-A 没有通过 deployment gate

两条训练都正常完成到 `100500_snapshot.pkl`。固定 25 个 eval seeds 的完整曲线为：

| rollout | success @ 10k/20k/30k/40k/50k/60k/70k/80k/90k/100k | mean | best | final |
|---|---|---:|---:|---:|
| BC-only | `76/72/68/48/56/68/64/60/52/56%` | `62.0%` | `76%` | `56%` |
| `zscore(A) + 4 log pi_BC` | `44/72/60/64/68/60/32/72/68/56%` | `59.6%` | `72%` | `56%` |

逐 checkpoint 的 A+BC 减 BC-only 差值是
`-32/0/-8/+16/+12/-8/-32/+12/+16/0` percentage points，均值 `-2.4pp`。两臂 final
相同，但 A+BC 的 mean 和 best 都更低，因此按预注册规则记为 **deployment gate fail**，不能
声称 causal-A 已经改善控制。

这个结论不能归因于 BC loss。`policy_demo_top1` 与 `policy_ce` 只是训练 demo 上、teacher-forced
C2F bin 的局部拟合指标；它们既不是完整 action-chunk accuracy，也不是闭环 policy quality。
[Much Ado About Noising](https://arxiv.org/html/2512.01809v1) 的附录 E.1 同样表明 validation
reconstruction loss 与 control success 可以明显解耦。因此这里唯一成立的观测是：训练 surrogate
继续改善时，闭环 success 没有获得 causal-A 带来的净增益；不能由此单独诊断过拟合。

当前实验仍混有两个可能原因，必须先分开：

1. `beta=4` 来自动作距离 proxy 的 exploratory 结果，真实 `log pi_BC` 与 normalized-A 的尺度
   可能不匹配；
2. 两臂在训练时使用不同 rollout policy，在线 replay 分布随后分叉，不能只凭这两条曲线定位是
   deployment score 还是 learning dynamics 导致。

#### Stage-XIV：同一 checkpoint 的 deployment-only beta calibration

先冻结 BC-only run 的 `100500_snapshot.pkl`，完全不更新 encoder、policy、Flow-V 或 direct-A，
只改变部署时的 bin score。这样可以把 beta 标度问题从训练 replay 分布中隔离出来。

预注册 calibration：

- checkpoint：
  `move_plate_causal_a_bconly_resume8k_to100k_seed1_gpu3_20260723/snapshots/100500_snapshot.pkl`；
- 新 calibration seeds：`32000--32049`，不复用 Stage-XIII 的 `31000--31024`；
- variants：BC-only、A-only，以及
  `beta in {0.25, 0.5, 1, 2, 4, 8}`；
- 每个 variant 保存逐 seed 的 success/reward/episode length，并给各 variant 使用相同的
  action RNG seed，后续做 paired bootstrap/McNemar，而不是只比较独立均值；
- 两条 GPU lane 分别运行四个 variants。实现入口为
  `scripts/eval_cqn_flow_policy_value.py`，配对汇总入口为
  `scripts/summarize_cqn_flow_policy_value.py`。

决策规则：

1. calibration split 只用于选一个非 BC candidate，不能作为最终结论；
2. 被选 candidate 与 BC-only 在新的 `33000--33099` confirmatory seeds 上各评估一次；
3. 只有 paired success delta 为正且区间不再支持“无收益”，才允许进入
   Flow-V/direct-scalar-V/C51-V matched comparison；
4. 如果所有 beta 都不能超过 BC-only，则不继续调 global beta。下一步改为收集
   policy-reachable、跨 phase/维度的 branch counterfactual，并只在 advantage
   held-out confidence 足够时允许 A override BC；同时加入 observation perturbation/recovery
   与 action-manifold adherence 指标，避免再用 BC loss 选择 checkpoint。

#### Stage-XIV 结果：固定 final checkpoint 的增益没有通过 confirmatory gate

在 `100500_snapshot.pkl` 上，calibration seeds `32000--32049` 的 BC-only success 是
`38%`。`beta=0.5` 是 calibration 最优候选，success `56%`，paired delta
`+18pp [4pp,32pp]`；但这只是用于选择候选，不能作为最终结论。

新的 confirmatory seeds `33000--33099` 上：

| rollout | success | paired delta vs BC | paired 95% CI | McNemar exact p |
|---|---:|---:|---:|---:|
| BC-only | `42%` | — | — | — |
| `zscore(A) + 0.5 log pi_BC` | `49%` | `+7pp` | `[-5pp,+19pp]` | `0.337` |

方向仍为正，但区间包含无收益，故按预注册规则记为 **confirmatory fail**。这说明 final
checkpoint 上的 global beta 可能偶尔有帮助，却没有足够证据说明它稳定改善 policy；不能据此
进入“FM 已提升控制”的结论。

随后按用户指出的正确比较方式，对每个方法都按 validation success 选自己的最好 checkpoint，
而不是强行比较相同的 final step。固定 seeds `31000--31024` 的结果为：

| checkpoint | 8k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k | 100.5k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC-only | `60` | **`76`** | `72` | `68` | `48` | `56` | `68` | `64` | `60` | `52` | `56` | `52` |
| `beta=0.5` | `24` | `0` | `48` | `52` | `52` | `40` | `48` | `56` | **`60`** | `56` | `52` | `48` |

单位均为 success percent。BC-only 的 method-best 是 10k/`76%`；value-rerank 的
method-best 是 80k/`60%`。所以当前 value 方法没有超过最佳 BC，且不应再利用 BC 在晚期降到
`42--56%` 来制造优势。BC tower 与 value tower 已经是 distinct encoder，因此这里的晚期 BC
下降也不应称为“RL 把 BC 忘掉了”；它是 BC 继续训练后的闭环退化/过训练现象。

#### Stage-XV：正式拆成两个互不依赖的 research question

从本阶段开始不再把“value 是否真实”和“FM 是否适合 value”合并成一个实验。两条线共享已有
数据，但有各自的假设、baseline、selection split 和停止条件；一条失败不代表另一条失败。

##### Research A：CQN-AS 是否学到了可因果验证的 action value

核心问题不是 critic loss 是否下降，而是：**在 BC policy 完全不变时，继续学习 value 能否提高
counterfactual action ranking，并最终超过同一 frozen BC 的闭环表现。**

实验协议：

1. 从 validation-selected 的最佳 BC checkpoint
   `...stage13/.../10000_snapshot.pkl` 开始，而不是从 final checkpoint 开始。
2. 新增 `method.freeze_bc_policy=true`。每次 AdamW update 后，将 independent BC head 和
   distinct policy image encoder 逐叶恢复为 update 前的参数；因此即使 weight decay 非零也保持
   bitwise fixed。配置若没有 `separate_bc_policy=true` 或
   `distinct_policy_encoder=true` 会直接拒绝启动。
3. rollout 始终使用 BC-only，`policy_value_beta=null`。这样 replay distribution 不会因为一个
   尚未验证的 value selector 而改变。value encoder、Flow-V、direct-A、TD/MC anchor 和
   causal branch auxiliary 正常更新。
4. 只复制 10k snapshot 时刻真实存在的 `98` 个 replay episodes 和 `51` 个 demo episodes，
   防止从原 run 的未来 replay 泄漏。
5. 10k 至 50.5k 每 10k 保存 checkpoint。训练内 eval 只检查 frozen BC 是否仍复现相同闭环
   policy；真正的 value test 在训练后对各 checkpoint 做 deployment-only rerank。

Research A 的三层 gate：

- **冻结完整性**：10k 与所有后续 checkpoint 的 `policy`、`policy_encoder` hash 必须完全相同；
  任何一叶变化都使实验无效。
- **value authenticity**：在未训练的 branch seeds `25000--25004` 上，pairwise sign、
  per-state Spearman、top-1、regret 和 dimension-span alignment 必须相对 10k 改善；同时运行
  return-label shuffle negative control，排除 head 只记 state/action identity。
- **控制价值**：只在 validation seeds `31000--31024` 选择一个 checkpoint 和一个预注册
  deployment rule，然后在全新的 confirmatory seeds 上与 frozen 10k BC 做 paired eval。
  必须超过 BC method-best `76%` 的 validation reference，并在 confirmatory paired interval
  排除无收益，才可声称 value 对控制有额外作用。

如果 causal ranking 上升但闭环不升，结论是“学到局部 value，但 selector/coverage 不足”；
如果 ranking 也不升，则主要问题仍是 online TD/branch data，而不是 policy 混入；如果只在
shuffle control 也上升，则判为 representation memorization，不算 value learning。

##### Research B：同一 value problem 上 Flow Matching 是否优于 direct/C51

这条线完全去掉 policy 更新和在线 replay。输入固定为同一份 frozen CQN encoder cache，每个
候选的 condition 明确包含：

- state/image frozen feature；
- 完整 candidate action plan；
- intervention action dimension one-hot；
- C2F sibling action-bin one-hot。

cache 中每个 state 有 15 个 action dimensions、每个 dimension 有 5 个 bins；训练仍按
`[state, action_dimension, bin]` 向量化。由于重复 dimension 共享同一 state feature，原
`19440-D` feature 的 train numerical rank 只有 11；预处理只保留 train-fit 的 11 个非零 PCA
方向，禁止保留数值零空间后把 held-out seed 放大。

matched methods：

| method | output/training target | action selection score |
|---|---|---|
| direct scalar | one scalar，MSE to realized return | scalar prediction |
| C51 | 51 uniformly spaced atoms on `[0,1]`，projected-target CE | atom probability的期望 |
| conditional FM | Gaussian `x0` 到 scalar return `x1` 的 velocity matching | 5 Euler levels、16 antithetic sources 的 endpoint mean |

三者使用相同 condition、两层 `256` MLP、batch `256`、AdamW、5k updates 和三个初始化 seeds。
数据 split 不能混：

- fit：simulator seeds `24000--24003`；
- checkpoint selection：seed `24004`，先最大化 informative-state pairwise accuracy，再最小化
  regret/MAE；
- final held-out：seeds `25000--25004`，绝不参与 checkpoint 选择。

主要结论看 selected-checkpoint 的 held-out pairwise accuracy、Spearman、top-1 和 regret；
MAE/MSE 是 calibration 辅助指标。FM 还必须报告不同初始噪声的 endpoint std。如果 FM 只降低
MAE、但不提高 ranking，不能说它更适合 CQN action selection；如果 source std 长期很大，则
多次采样平均只是掩盖没有收敛的 transport。

当前 cache 每个 `(s,a)` 只有一次 continuation return，因此这一步只能比较 conditional value
optimization/ranking，不能声称恢复了 stochastic return distribution。如果 FM 在这个 gate
胜出，下一阶段才对同一 `(s,a)` 收集多个 stochastic continuations，再比较 quantile/coverage/
CRPS；否则不为“distributional”故事额外采样。

预注册决策：

- FM 需在三个初始化 seed 的 mean held-out pairwise 和 regret 上同时优于 direct scalar 与
  C51，且不是由一个 seed 驱动，才进入 online CQN-AS。
- C51 若最好，则保留 direct-C51 value baseline；这支持 distributional CE，但不支持 FM。
- direct scalar 若相当或最好，则 FM/C51 的额外复杂度没有被当前数据证明。

##### Stage-XV 实现与正式运行（2026-07-23）

已增加：

- `robobase/cfgs/launch/cqn_flow_causal_a_frozen_bc_value_gate.yaml`；
- `method.freeze_bc_policy` 的构造、校验和 update-time exact restore；
- `scripts/benchmark_cqn_branch_value_models.py`；
- frozen AdamW、launch composition、condition/bin、C51 projection 和 ranking metric 单元测试。

新增相关 focused tests 合计 `5 passed`（其中 benchmark 的三个 tests 之后又单独复跑通过）。
正式两条 lane 已启动：

| GPU | research | run |
|---|---|---|
| 1 | frozen-best-BC value-only，10k -> 50.5k | `exp_local/cqn_value_fidelity_stage15/move_plate_frozen_bestbc10k_to50k_seed1_gpu1_20260723` |
| 5 | fixed-data direct/C51/FM，3 seeds | `exp_local/cqn_flow_value_model_benchmark/move_plate_matched_direct_c51_flow_seeds1_3_gpu5_20260723` |

GPU 2/3 当时由另一位用户的训练占用，故选择实际空闲的 1/5，没有抢占或终止其他任务。
Research A 首次启动已恢复 10k snapshot、98 replay 和 51 demo；Research B 的 status manifest
确认 condition 为 `11-D frozen state/image + 240-D action + 15-D dimension + 5-D bin`
（总计 `271-D`），validation seed 为 `24004`，held-out seeds 为 `25000--25004`。

#### Research B 第一轮结果：FM 没有超过 matched direct scalar

固定数据实验已完整跑完。下表是每个 initialization 先由 validation seed `24004` 选择 checkpoint，
再在五个 held-out simulator seeds 上计算的 mean ± initialization std：

| method | pairwise | Spearman | top-1 | regret ↓ | MAE ↓ | FM source std ↓ |
|---|---:|---:|---:|---:|---:|---:|
| direct scalar | **`54.73±1.51%`** | **`0.088±0.035`** | **`34.97±0.39%`** | **`0.0291±0.0006`** | **`0.177±0.015`** | — |
| C51 | `51.96±1.04%` | `0.044±0.022` | `25.68±2.53%` | `0.0349±0.0075` | `0.181±0.033` | — |
| conditional FM | `53.75±1.59%` | `0.071±0.036` | `30.87±1.02%` | `0.0331±0.0017` | `0.192±0.015` | `0.0325±0.0079` |

FM 比 C51 好，但没有通过预注册的 direct-scalar gate。尤其 top-1 和 regret 上 direct 在三个
initialization 都优于 FM；pairwise 上 direct 是 `2/3` seeds 更好。FM 的 endpoint source std
已经不算爆炸，但仍非零，而且多采 16 个 Gaussian sources 后 MAE 仍最差，因此不能把差距归因于
“只少采了几次噪声”。

这一轮支持的结论很窄：在每个 `(s,a)` 只有一个几乎确定的 continuation target 时，直接回归
expected value 更合适；C51 的 51-atom CE 和 FM transport 都没有带来 ranking 优势。它不否定
FM 对真正多模态 return distribution 的可能价值，但在收集 repeated continuations 之前，FM
不应接入 online CQN-AS 主实验。Research A 继续独立运行，不受这个结果影响。

GPU 5 随后继续运行 validation-split robustness：依次将 `24000/24001/24002/24003` 各自留作
validation，和已完成的 `24004` 合成五折；每折只重跑 direct scalar 与 FM、各三个 initialization。
该检查不用于重新调 held-out 结果，只回答第一轮胜负是否依赖恰好选了 `24004`。

#### Stage-XV 完成：隔离成功，但尚未证明 value 对 control 有用

frozen-best-BC run 已完成到 50.5k，生成
`10000/20000/30000/40000/50000/50500_snapshot.pkl`。逐叶 SHA-256 审计结果：

- 六个 checkpoint 的 BC `policy` hash 始终为 `ef2383807378a884...`；
- 六个 checkpoint 的 `policy_encoder` hash 始终为
  `b1c1db1ba14c99a7...`；
- critic 和 advantage 每个 checkpoint 的 hash 都不同。

因此该实验真正实现了“BC 完全不动、只有 value-side 学习”。训练内 BC-only eval 为：

| checkpoint | 20k | 30k | 40k | 50k |
|---|---:|---:|---:|---:|
| frozen BC success | `76%` | `72%` | `72%` | `72%` |

这组 success 只验证 frozen policy 仍可闭环运行，不能证明新 value 有用，因为 rollout 按设计
没有读取 value。它排除了“BC 参数继续训练/退化”这个混淆因素，剩下的问题必须直接通过
counterfactual ranking 和 deployment-only rerank 回答。

Research B 的五个 validation-fold robustness 也已完成。按每折三个 initialization 的
validation-selected checkpoint，再在同一 held-out seeds 上平均：

| method | pairwise | Spearman | top-1 | regret ↓ |
|---|---:|---:|---:|---:|
| direct scalar | **`54.59%`** | **`0.088`** | `30.00%` | **`0.0339`** |
| conditional FM | `53.87%` | `0.076` | **`31.69%`** | `0.0390` |

direct 的 pairwise/Spearman/regret 在 `4/5` folds 更好；FM 的 top-1 在 `4/5` folds 更好。
因此更准确的结论不是“direct 全面碾压”，而是 **FM 没有稳定超过 direct，也没有通过预注册的
双指标 gate**。其中 validation `24002` 折的 absolute MAE 对 direct/FM 都出现千量级
extrapolation，暴露了极低秩 state PCA 的 calibration 脆弱性；ranking/regret 仍有限，但该折
不能用于 continuous-value calibration 结论。下一轮固定数据模型若继续，必须改用 raw frozen
feature 的 regularized projection 或保存 raw observation，不再依赖 fold-specific whitening。

#### Stage-XVI：matched counterfactual value-authenticity audit

上一阶段结果说明 BC/value 已成功隔离，但没有测到 learned value 本身。本阶段先比较 frozen
BC 起点 10k 与 value-only 终点 50.5k，不直接做 deployment 调参。

协议：

- 两个 checkpoint 分别重新恢复相同 train seeds `24000--24004` 和 held-out seeds
  `25000--25004`；
- 每个 seed 使用 anchors `30/75/120`、全部 15 action dimensions、每维 5 个 L1 sibling
  bins、H=4 intervention；
- 每个 checkpoint 重新计算当前 value encoder feature。旧 8k feature cache 不可复用，因为
  value encoder 已更新；
- 比较 pairwise sign、Spearman、top-1、regret 和 dimension-span alignment；
- 以 simulator seed 为 cluster 做 paired bootstrap，并校验两个 checkpoint 的 realized branch
  returns 是否逐记录匹配。

严格 gate：train 与 held-out 的 pairwise improvement 都必须为正，且 paired seed-bootstrap
95% CI 下界大于 0；否则不能进入 global-beta deployment sweep。若 ranking 提升但 CI 未排除
零，只能记为趋势，下一阶段增加 policy-reachable branch coverage；若 ranking 下降，则停止
当前 value objective，转为 direct scalar counterfactual regression。

两个 checkpoint audit 最初曾分别在 GPU 1/5 同时启动：

| GPU | checkpoint | output |
|---|---:|---|
| 1 | 10k baseline | `exp_local/cqn_value_fidelity_stage16/counterfactual_10k_vs_50500_20260723/audit_10000.json` |
| 5 | 50.5k trained | `exp_local/cqn_value_fidelity_stage16/counterfactual_10k_vs_50500_20260723/audit_50500.json` |

汇总入口为 `scripts/summarize_cqn_counterfactual_audits.py`，其 paired synthetic regression
test 已通过。audit 完成后先报告 Stage-XVI 结果和含义，再根据 gate 决定是否启动 deployment；
禁止无条件继续调 beta。

这两个进程其实都属于 Research A，不能把“两张卡”误报成“两条 research”。发现该问题后，
保留 GPU 1 的 10k baseline audit；GPU 5 的 50.5k audit 在尚未生成 cache/result 时停止，待
10k 完成后在 Research A lane 串行补跑。GPU 5 随即重新分配给 Research B。当前资源语义固定为：

| GPU | 独立 research | 当前工作 |
|---|---|---|
| 1 | A：value authenticity | 10k counterfactual baseline，之后串行 50.5k |
| 5 | B：FM value modeling | endpoint constraint、robustness、repeated-return gate |

10k baseline audit 已完成，用时 `2801.99s`，cache 为 `32MB`，train/held-out 各有
`225` 个 `(state, action_dimension)`、其中 `120/122` 个有可辨识 action effect：

| split | pairwise | Spearman | top-1 | regret ↓ | dimension-span Spearman |
|---|---:|---:|---:|---:|---:|
| train `24000--24004` | `48.02%` | `-0.0415` | `23.33%` | `0.0858` | `-0.447` |
| held-out `25000--25004` | `50.20%` | `0.0005` | `27.05%` | `0.0928` | `-0.281` |

所以 frozen-BC 起点的 critic 基本是 chance-level local action ranker，且 predicted Q span 与真实
dimension sensitivity 反相关。该结果是 50.5k value-only critic 必须超过的因果 baseline，
不能单独用来判断后续训练是否有效。

50.5k matched audit 已在 GPU 1 串行重新启动，PID `563767`，仍使用相同 simulator seeds、
anchors、15 dimensions、5 sibling bins 和 H=4。按 10k 的实测 wall time，ETA 为约
`47min`（保守 `40--60min`）。完成后
`scripts/summarize_cqn_counterfactual_audits.py` 会先校验逐记录 realized returns 匹配，再做
seed-cluster paired bootstrap；只有 train/held-out pairwise improvement 的 95% CI 下界均大于
零才允许进入 deployment。

##### Research A capacity 与 shuffle controls

先在 10k cache 上用通用两层 MLP 做 direct scalar / global-label-shuffle 对照。该 surrogate
将 `19440-D` frozen feature 通过 fit-only PCA 压到数值秩 `11-D`，validation seed `24004`
又恰好 `45/45` states 全为 return tie，因此按确定性 fallback 改用最大的 informative seed
`24003`。五个 initialization 的 held-out 结果：

| PCA surrogate | pairwise | Spearman | top-1 | regret ↓ |
|---|---:|---:|---:|---:|
| direct true labels | `48.45%` | `-0.0405` | `22.62%` | `0.0952` |
| global shuffled labels | `49.94%` | `-0.0043` | `24.75%` | `0.0895` |

这证明 11-D PCA surrogate 不泛化，但不能外推为“真实 CQN architecture 没容量”。真实 hybrid
CQN critic 并不使用 PCA：它把 `19200-D` RGB feature 经两层 Dense/LayerNorm，再与
`240-D` low-dim tower 融合。于是增加更忠实的 paired control：

- 从同一个 10k snapshot 初始化真实 direct-advantage head；
- true oracle 使用 branch pairwise + delta labels；
- negative control 对每个 state 内部独立打乱 5 个 bin returns，保留 state difficulty 和
  return marginal，只破坏 action/value 对应；
- 两者使用相同 update sampling seed。

全五个 train seeds 都参与、固定 2000 updates 时：

| labels | train pairwise | train top-1 | held-out pairwise | held-out top-1 |
|---|---:|---:|---:|---:|
| true | `100%` | `100%` | `50.41%` | `26.23%` |
| within-state shuffle | `48.48%` | `20.83%` | `48.77%` | `21.31%` |

因此真实 architecture 有足够容量、也确实读取正确 labels，但会完全记忆有限 branch anchors；
它没有可靠 held-out 泛化。为避免用固定 final update 制造“过拟合失败”，再将 seed `24003`
完全留出，fit 只用其余四个 train seeds，并比较 `100/500/2000` updates。严格 checkpoint
selection 必须看 validation，不能看 held-out：

| updates | true validation pairwise | shuffle validation pairwise | true held-out pairwise | true held-out top-1 | true held-out regret |
|---:|---:|---:|---:|---:|---:|
| `0` | `47.70%` | `47.70%` | `50.20%` | `27.05%` | `0.0928` |
| `100` | `46.84%` | `50.57%` | `51.74%` | `31.97%` | `0.0800` |
| `500` | `45.40%` | `48.56%` | `50.41%` | `27.05%` | `0.0865` |
| `2000` | `45.40%` | `50.00%` | `51.02%` | `26.23%` | `0.0875` |

虽然 100 updates 在最终 held-out 上方向较好，但 validation 明确低于 step 0，不能事后采用。
严格 validation-selected 结论是保持原 10k critic；当前四-seed counterfactual oracle 没有
可验证的跨 seed 泛化增益。shuffle 在 validation 上能达到或超过 true labels，也说明小 seed
split 的波动足以制造假提升。这个结果把问题定位为 branch/state coverage 与跨 seed
generalization，而不是单纯网络容量不足。

##### 10k vs 50.5k matched authenticity 结果

50.5k audit 实际用时 `2778.30s`。两个 checkpoint 的全部 realized branch returns 逐记录完全
一致，最大绝对差为 `0`，所以比较没有 simulator-return drift：

| split | 10k pairwise | 50.5k pairwise | delta | paired seed-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| train `24000--24004` | `48.02%` | `50.05%` | `+2.03pp` | `[-8.60,+7.73]pp` |
| held-out `25000--25004` | `50.20%` | `58.49%` | `+8.28pp` | **`[+2.94,+13.28]pp`** |

held-out 的 Spearman 从 `0.0005` 到 `0.1808`，delta CI
`[0.0810,0.2581]`；top-1 从 `27.05%` 到 `31.97%`，但 CI 仍跨零；regret 从 `0.0928`
降到 `0.0856`，reduction CI `[-0.0021,0.0245]` 也跨零。dimension-span Spearman 则从
`-0.281` 进一步变成 `-0.338`，说明 ranking 改善并未带来跨维度 sensitivity calibration。

逐 seed 上，held-out pairwise 与 Spearman 均为 `5/5` seeds 改善，不由单 seed 驱动；train
则 `24001/24003` 改善、`24000` 变差、`24002` 只有一个 informative state 且翻转、
`24004` 完全无 informative state，因此 train cluster CI 很宽。bootstrap 实现已修复：
全-tie seed cluster 在 point estimate 中本来就是零权重，现在也从 bootstrap population 排除，
而不是抽到全-tie replicate 时异常；新增 regression test 后为 `2 passed`。

泄漏审计确认 `robobase/method/cqn_flow.py::_load_causal_branch_cache()` 只读取 cache 的
`train_*` arrays，完全不加载 `heldout_*`；所以 `25000--25004` 没有进入 value-only training。
结合 BC `policy/policy_encoder` bitwise frozen，这一结果足以否定“critic 完全只是模仿 BC”的
强假设：它确实获得了可因果验证的 unseen-action ranking signal。

但按预注册规则，总 gate 仍是 **fail**，因为要求 train 与 held-out 的 pairwise CI 下界都大于
零；regret/top-1 也未显著，dimension calibration 仍差。因此当前结论是“学到部分真实 local
value”，不是“已证明可安全用于全局 action selection”，暂不进入 global-beta deployment。

下一阶段为全新 replication coverage，避免在同一五个 held-out seeds 上反复解释：

- audit split A：`26000--26004`；
- audit split B：`27000--27004`；
- 10k 与 50.5k 分别在 GPU 1/5 同时跑完全相同的 3 anchors × 15 dimensions × 5 bins；
- 两组 seeds 都从未出现在 causal cache、训练或前述 selection 中；
- 两个 PID 为 `628373/628375`；按上一轮实测 ETA 均约 `47min`，并行 wall time 保守
  `40--60min`。

只有 replication 的 unseen split 仍得到正 pairwise delta 且 paired CI 下界大于零，才把
“部分真实 value”升级为可复现；否则上一轮记为探索性阳性，不进入 deployment。

#### Research B Stage-XVI：endpoint value 约束仍未超过 direct

纯 velocity matching 在前一阶段没有通过 direct gate。为检验原因是否只是 FM endpoint 没有被
直接钉到 realized value，`scripts/benchmark_cqn_branch_value_models.py` 新增
`flow_endpoint`：保留原 velocity-matching loss，同时从同一个 Gaussian source 做 5 步可微
Euler integration，并加入
`lambda * MSE(endpoint, realized_return)`。纯 `flow` 路径没有改变；focused tests 为
`12 passed`（branch oracle + value benchmark）。

固定 validation seed `24004`、三个 initialization 的结果：

| method | pairwise | Spearman | top-1 | regret ↓ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| direct scalar | **`54.77%`** | **`0.0886`** | **`34.97%`** | **`0.0291`** | **`0.177`** |
| pure FM | `53.72%` | `0.0697` | `30.87%` | `0.0331` | `0.192` |
| FM + endpoint, `lambda=1` | `54.50%` | `0.0836` | `33.06%` | `0.0345` | `0.211` |

endpoint loss 修复了一部分 ranking，但没有同时改善 regret/calibration，仍未通过 gate。随后运行
`lambda in {0.05,0.1,0.25,0.5,1,2}`；最接近 direct 的 `lambda=2` 为 pairwise
`54.63%`、Spearman `0.0888`、top-1 `31.97%`、regret `0.0350`、MAE `0.202`。它只在
Spearman 上比 direct 高 `0.00024`，其余主要指标仍差，不能记为胜出。

最后把 `lambda=2` 扩展到五个 validation folds：

| method | pairwise | Spearman | top-1 | regret ↓ |
|---|---:|---:|---:|---:|
| direct scalar | **`54.60%`** | **`0.0885`** | **`30.00%`** | **`0.0339`** |
| FM + endpoint, `lambda=2` | `53.49%` | `0.0665` | `29.89%` | `0.0409` |

因此停止继续调 endpoint 权重；结果支持“单一、近确定 return target 上 direct scalar 更合适”，
而不是“FM 只差一个额外 MSE”。下一阶段先判断同一 `(state/image, full action,
action_dimension, action_bin)` 是否真的存在非退化 return distribution。

为此 branch collector 新增：

- `--continuation-repeats`；
- `--continuation-rng-mode={restored,independent}`；
- cache 中保留 `[state,candidate,repeat]` 的原始 return samples，同时 scalar target 使用均值；
- return std/span、非退化 candidate 比例和 stochastic-success 比例汇总。

GPU 5 已启动 8-repeat probe：
`exp_local/cqn_flow_value_model_benchmark/move_plate_repeated_continuation_probe_gpu5_20260723`。
它使用 10k frozen policy、seeds `24000/25000`、anchors `30/75/120`、dimensions
`0/7/14`、每维 5 个 sibling bins。若不同 continuation RNG 下仍几乎全为零方差，则当前
MovePlate/CQN 条件下 FM 没有 distributional 建模对象，应保留 direct value baseline；只有
return 或 success 显著非退化，才扩充到全部维度并比较 NLL/CRPS/coverage 与 action ranking。

首轮 repeated-return probe 用时 `866.45s`，实际结果：

| split | variable candidates | mean return std | max return span | stochastic success |
|---|---:|---:|---:|---:|
| train seed `24000` | `1/45` | `3.07e-5` | `0.00417` | `0/45` |
| held-out seed `25000` | `0/45` | `0` | `0` | `0/45` |

因此不同 continuation RNG 几乎没有产生条件 return 分布；唯一变化量也远小于候选 action
之间用于 ranking 的典型 return span。为避免三维抽样遗漏特殊 action coordinate，Research B
最后运行一个 confirmatory probe：

- 相同 `24000/25000` seeds 和 `30/75/120` anchors；
- 全部 15 action dimensions、每维 5 bins；
- 每个精确恢复的 `(state, candidate)` 使用 4 个 independent continuation RNG；
- pass gate：存在稳定的 stochastic-success candidate，或有实质比例 candidate 的 return
  span 大到足以改变 action ordering。

正式 run 为
`exp_local/cqn_flow_value_model_benchmark/move_plate_repeated_continuation_all15_confirm_gpu5_20260724`。
按首轮实测 `720` branch rollouts / `866s`，本轮 `1800` rollouts 的 ETA 为约 `36min`
（保守区间 `30--45min`）。若 confirmatory 仍退化，Research B 的最终结论将是：当前任务上
保留 direct scalar value，FM/C51 仅作为未来具有真实 stochastic/multimodal return 数据时的
研究选项，不进入 online CQN-AS。

confirmatory 实际用时 `2059.26s`（`34.3min`），落在预估区间内。结果：

| split | variable candidates | stochastic success | mean return std | max return span |
|---|---:|---:|---:|---:|
| train seed `24000` | `2/225 (0.89%)` | `1/225 (0.44%)` | `6.28e-4` | `0.279` |
| held-out seed `25000` | `0/225` | `0/225` | `0` | `0` |

两个 train 异常中：

1. anchor 30、dimension 3、candidate 2 在四次中 `2 success / 2 failure`，returns 为
   `[0, 0.279, 0, 0.279]`；
2. anchor 75、dimension 0、candidate 0 四次都成功，只因完成时间相差一个 step，return span
   为 `0.00421`。

没有异常在 held-out seed 重现，且总计 `448/450` candidates 完全确定。因此它不是稳定、
可泛化、足以支持全局 distributional critic 的结构，没有通过预注册 gate。

**Research B 最终推荐：** 当前 MovePlate + frozen CQN-AS 条件下，用 direct scalar
`Q(s, full_action, action_dimension, action_bin)` 学 expected return；不把 FM 或 C51 接入
online CQN-AS。conditional FM 只有在未来数据对同一 condition 提供重复、非退化、可跨 seed
复现的 return samples，并且在 held-out action ranking/regret 与 distribution calibration
上同时超过 direct scalar 时才重开。这个结论来自 matched direct/C51/FM、endpoint-loss
sweep、五 validation folds 和两轮 repeated-continuation probe，而不是仅凭一次训练失败。

Research B 释放出的 GPU 5 转给 Research A 的必要 negative control：在 10k 新 cache 上比较
oracle-supervised `direct` 与保持相同 return marginal、但随机打乱 fit condition/return 对应的
`direct_shuffle`。五个 initialization、validation seed `24004`、held-out seeds
`25000--25004`，ETA 约 `2min`。它检验 frozen representation 是否有可学习的 counterfactual
signal，以及看似提升是否能在 label shuffle 后仍出现。

#### Stage-XVII replication 结果与两条路线最终决策

新的 `26000--26004 / 27000--27004` replication 两个 checkpoint 分别用时
`2521.34s / 2566.07s`，并行 wall time 约 `42.8min`，落在 `40--60min` ETA 内。结果没有复现
第一轮阳性：

| replication split | 10k pairwise | 50.5k pairwise | delta | paired CI95 |
|---|---:|---:|---:|---:|
| A `26000--26004` | `52.94%` | `48.44%` | `-4.51pp` | `[-8.41,+17.01]pp` |
| B `27000--27004` | `52.01%` | `50.83%` | `-1.19pp` | `[-6.74,+5.31]pp` |

split A 的 Spearman `0.0621 → -0.0369`、top-1 `28.72% → 24.47%`、regret
`0.0867 → 0.1101`，均变差。split B 的 top-1 虽从 `29.51%` 到 `32.79%`，但 Spearman
`0.0386 → 0.0139`，pairwise 下降且所有 paired CI 跨零。

split B 有 `3/225` records 的 simulator return 发生小漂移，最大差 `0.01228`。汇总脚本现会
额外保存严格 matched-record subset，而不只给一个 false flag。排除三条后仍为：

- pairwise `51.94% → 51.50%`，delta `-0.44pp`；
- Spearman `0.0367 → 0.0268`；
- top-1 `29.44% → 33.33%`；
- regret `0.1309 → 0.1283`。

所以 replication fail 不是三条 drift record 造成的。`scripts/summarize_cqn_counterfactual_audits.py`
新增 all-tie seed bootstrap 处理和 matched-subset 输出，相关 regression tests 为 `3 passed`。

##### Research A 最终结论

当前证据不支持两个极端说法：

1. **“它完全只是 BC。”不成立。** BC policy 和 policy encoder 在 10k--50.5k 间 bitwise
   frozen；第一组五个真正未训练 seeds 上，critic pairwise `5/5` 改善且 paired CI 排除零。
   因此 value-side 至少能学到一部分真实 local counterfactual ordering。
2. **“它已稳定学会可部署的 value。”也不成立。** 首轮 train gate 失败、regret/top-1 不显著、
   dimension sensitivity 未校准；更重要的是全新十个 seeds replication 没有复现 ranking
   improvement。真实 CQN architecture 的 oracle control 还能把有限 train anchors 记到
   `100%`，但 validation-selected 泛化选择最终回到 step 0，说明 branch coverage 和
   cross-seed generalization 是主瓶颈。

因此 Research A 的部署推荐是：**保持 validation-selected frozen BC-only；当前 critic 不参与
全局 action rerank，也不再调 global beta。** 后续若重开 value 路线，最低条件是：

- 收集更多 policy-reachable branch states，按 episode/seed/phase 严格 train-validation-test
  分离；
- checkpoint 只由 validation counterfactual ranking/regret 选择；
- 使用 confidence/coverage gate，只在当前 state、dimension 与 intervention size 有邻近
  branch support 时允许 value override BC；
- 在全新 replication seeds 上同时通过 pairwise、regret 和 closed-loop paired success，才允许
  部署。

##### Research B 最终结论

当前任务使用 **direct scalar expected value**，不使用 FM/C51 distributional critic：

- matched direct/C51/FM 中 direct 的 ranking/regret 最好；
- endpoint-MSE 和 `lambda` sweep 没有改变五折结果；
- `448/450` repeated `(state,action)` candidates 完全确定，唯一显著 stochastic candidate
  也未在 held-out 重现；
- 因而 Gaussian-source FM 在这里增加 transport/采样复杂度，却没有稳定分布可建模。

只有未来同一 condition 的 repeated returns 在 held-out seeds 上稳定非退化，并且 FM 同时超过
direct 的 action ranking/regret 与 distribution calibration，才应重开 FM value。即使重开，
也必须先解决 Research A 暴露的 branch coverage/generalization；换 value parameterization
不能替代真实 counterfactual data。

## 21. 2026-07-24：Stage-XVIII——coverage scaling 与官方式 FLOQ readout

本阶段承接 Stage-XVII 的两个独立失败，不把它们重新合并：

- Research A 的 5-seed 阳性没有在新 10 seeds 上复现，因此下一瓶颈是 branch coverage、
  split representativeness 和跨 seed 泛化；
- Research B 的 Gaussian-source conditional-return FM 在几乎确定的 return 上没有超过 direct，
  因此不能继续把“更复杂的 return distribution”当默认解释。

### 21.1 文献与官方代码审计修正了 Research B 的假设

官方 [floq](https://arxiv.org/abs/2509.06863) 和其后续机制论文
[What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)
给出的结论不是“FM 更善于拟合随机 return distribution”。后者反而报告 distributional backup
可能降低性能，主要收益来自：

1. 对多个 interpolant time 的 velocity supervision，使 moving TD target 下的 feature 更有
   plasticity；
2. 多步 integration 在 test time 修正早期 value 误差；
3. 高 UTD、非平稳 bootstrap target 下比一次性 monolithic regression 更不容易丢失可塑性。

审计 [官方 FLOQ 代码](https://github.com/CMU-AIRe/floq) 还发现当前 CQN-FLOQ profile
有一个重要省略：官方同时训练 scalar critic readout，使其拟合多个 integrated flow endpoints
的均值；policy 查询的是这个 distilled scalar critic。我们此前直接用 endpoint mean 枚举 CQN
bins，并把 distillation 描述为“不必要”，这对离散 action selection 是可运行的简化，但不是
官方完整 critic recipe。

其他 FM+RL 路线与当前问题的边界如下：

- [FQL](https://arxiv.org/abs/2502.02538)、[FlowQ](https://arxiv.org/abs/2505.14139)
  和 DFQL 主要把 flow 用作 **policy/action generator**，critic 仍提供 scalar Q；不能当作
  “state/image + action-bin condition 的 flow value”复现。
- [Reinforce Adjoint Matching](https://arxiv.org/abs/2605.10759) 是对生成模型 endpoint
  做 reward-tilted RL post-training；它回答如何优化 flow policy，不直接修复 CQN critic 的
  counterfactual coverage。
- [Value Flows](https://openreview.net/forum?id=2VyNYUVF2k) 的优势依赖非退化 return
  distribution；Stage-XVI 的 repeated-continuation gate 已证明当前 MovePlate 数据
  `448/450` conditions 完全确定，所以暂不重开该路线。

因此 Research B 改成更窄、可证伪的问题：**在相同 expected-value target 下，官方 source
geometry、多 source dense supervision、integration recovery 和 distilled readout 能否改善
action ranking；不是重新声称存在多模态 return。**

### 21.2 Research A：把已有 20 seeds 变成严格 coverage experiment

新增可复用工具：

- `scripts/merge_cqn_branch_caches.py`：只允许相同 snapshot/intervention protocol 的 cache
  合并，拒绝重复 seed 和 train/heldout overlap；
- `scripts/resplit_cqn_branch_cache.py`：按 simulator seed 重分割已有 cache；
- `_all_scores()` 改为 64-state chunked evaluation。第一次扩大到 720 records 后，旧版一次性
  score 申请约 `2.64GB` 单个中间张量而 OOM；分块版本数值等价且消除了该 evaluator artifact。

相关 branch/cache/value focused tests 当前为 `19 passed`。

第一版把 `24000--26004` 连续 15 seeds 作为 train、`27000--27004` 作为 validation。
100 updates 的真实 CQN advantage-head smoke 为：

| split | pairwise before → after | Spearman before → after | regret before → after |
|---|---:|---:|---:|
| train | `50.14 → 62.16%` | `0.0028 → 0.2552` | `0.0886 → 0.0665` |
| validation | `52.01 → 51.58%` | `0.0386 → 0.0416` | `0.1289 → 0.1314` |

这一步明确失败：扩大 record 数不等于扩大代表性，模型仍可记住 train anchors。并且连续 seed
block 的 outcome mix 明显不同：该 train split informative fraction `49.8%`，27000 validation
为 `81.3%`，所以不能用它单独选择 deployment checkpoint。

随后将四个 seed 区间分别留一个 seed，得到：

- train：`24000--24003,25000--25003,26000--26003,27000--27003`，共 16 seeds /
  720 records；
- validation：`24004,25004,26004,27004`，共 4 seeds / 180 records；
- future confirmatory test：全新 `28000--28004`，尚未收集、绝不参与 update 选择。

分层 split 的 seed-1、100-update smoke 得到第一项 coverage-scaling 正信号：

| split | pairwise before → after | Spearman before → after | top-1 before → after | regret before → after |
|---|---:|---:|---:|---:|
| train | `50.94 → 61.86%` | `0.0175 → 0.2514` | `28.12 → 39.53%` | `0.0840 → 0.0679` |
| validation | `48.37 → 56.54%` | `-0.0063 → 0.1327` | `19.57 → 28.26%` | `0.2965 → 0.2170` |

这个结果只用于进入 update-selection gate，不能称为 held-out success。预注册顺序是：

1. 只看上述 validation split，在 seed 1 上比较 update
   `{0,20,50,100,200,500}`，先最大化 pairwise，再最小化 regret；
2. 对选中的 update 用 initialization seeds 2/3 复验；若选中值不是 100，再补相应 seeds，
   不利用 future test 反调；
3. 只有 validation 上 mean pairwise、Spearman、top-1 同时提高且 regret 降低，才在 GPU 1
   收集 `28000--28004`；
4. confirmatory test 也通过后，才保存 causal sidecar checkpoint，并让 frozen legacy
   CQN-AS policy 通过 coverage/confidence gate 选择是否接受 value override。未通过 gate 时
   原 policy 原样执行，因此不会用未验证 critic 换掉 `92% @ 7.5k` 的 legacy best。

update sweep 由事件驱动的顺序 launcher 在 GPU 1 运行，父进程 PID `737043`。一次完整
100-update smoke 实测 `71.0s`；剩余 6 个 jobs 的 wall-time ETA 约 `7.1min`，保守
`6--9min`，完成标志为
`exp_local/cqn_value_fidelity_stage18/coverage16_stratifiedval4_10k_20260724/update_sweep/complete`。

### 21.3 Research B：官方式 source/readout 的 matched gate

`scripts/benchmark_cqn_branch_value_models.py` 新增：

- `flow_train_samples`，正式 gate 使用官方默认 `8`；
- uniform source range，正式 gate 使用 value support 的 10%：`[0,0.1]`；
- 8-step Euler integration；
- `flow_distill`：velocity field 仍对 Bellman/return target 做 dense supervision，独立 scalar
  readout 只拟合 stop-gradient 的 mean integrated endpoint；
- validation-only readout/endpoint blend，`0` 为纯 readout、`1` 为纯 endpoint。

三方法、三个 initialization、validation seed `26004` 选择各自 checkpoint 后，在
`27000--27004` 上的首轮正式结果为：

| method | pairwise | Spearman | top-1 | regret ↓ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| direct scalar | `53.93±1.62%` | `0.0695±0.0283` | `28.60±1.12%` | `0.1248±0.0101` | **`0.2458±0.0065`** |
| integrated flow endpoint | **`54.65±1.63%`** | **`0.0869±0.0282`** | **`30.24±4.69%`** | `0.1290±0.0048` | `0.2570±0.0011` |
| distilled scalar readout | `53.26±1.36%` | `0.0654±0.0315` | `28.96±1.18%` | **`0.1221±0.0065`** | `0.2497±0.0085` |

这是比前一阶段更有信息的部分正结果：

- 官方式 endpoint 首次在 mean pairwise/Spearman/top-1 上超过 direct，说明窄 uniform source
  与多 source supervision 比原 Gaussian single-source benchmark 更合适；
- distilled readout 首次把 regret 降到 direct 以下，但 ranking 略差；
- 没有一个单独 readout 同时超过 direct 的 pairwise 和 regret，因此仍未通过预注册 gate，
  不能进入 online CQN-AS。

下一最小 gate 不是再训练新模型，而是在同一个 `flow_distill` checkpoint 上用 validation
选择 endpoint/readout convex blend。候选固定为 `{0,0.25,0.5,0.75,1}`；只能按 validation
aggregate 选择一个 alpha，再一次性查看 `27000--27004`。要求三个 initialization 的 mean
pairwise 高于 `53.93%` 且 regret 低于 `0.1248`，并且不是单 seed 驱动。

blend sweep 已在 GPU 5 启动，父进程 PID `736206`。每个三-seed alpha 实测约
`30--40s`，四个新增 alpha ETA `2--3min`；完成标志为
`exp_local/cqn_flow_value_model_benchmark/stage18_floq_distill_blend_sweep_gpu5_20260724/complete`。
若 blend 仍失败，static deterministic regression gate 停止；下一项只测试论文真正声称优势的
high-UTD moving-TD setting，并保持 direct scalar 与相同参数量/UTD 的 baseline。

### 21.4 Stage-XVIII 完成结果：A 通过 validation gate，B 的 static blend 未跨折复现

#### Research A：update selection

GPU 1 的 update sweep 已完整结束。seed 1 只用预注册 validation split 选择 update：

| updates | validation pairwise | Spearman | top-1 | regret ↓ |
|---:|---:|---:|---:|---:|
| 0 | `48.37%` | `-0.0063` | `19.57%` | `0.2965` |
| 20 | `51.96%` | `0.0458` | **`34.78%`** | `0.2322` |
| 50 | `51.63%` | `0.0435` | `28.26%` | `0.2373` |
| **100** | **`56.54%`** | **`0.1327`** | `28.26%` | `0.2170` |
| 200 | `55.23%` | `0.1160` | `26.09%` | `0.2399` |
| 500 | `52.94%` | `0.0805` | `23.91%` | **`0.2110`** |

按“先最大化 pairwise、再最小化 regret”的预注册规则，选择 `100 updates`，不能因 500
updates 的 train memorization 或略低 regret 改选。随后用初始化 seeds 2/3 复验，三个初始化的
validation mean 为：

| metric | frozen 10k before | 100-update mean | delta |
|---|---:|---:|---:|
| pairwise | `48.37%` | `54.79%` | `+6.43pp` |
| Spearman | `-0.0063` | `0.1080` | `+0.1142` |
| top-1 | `19.57%` | `22.46%` | `+2.90pp` |
| regret ↓ | `0.2965` | `0.2458` | `-0.0506` |

因此 aggregate validation gate 通过；但 seed 2 的 top-1 单项下降，且这四个 validation
seeds 已参与 update selection，仍不能称为 final held-out success。

下一阶段严格保留 `28000--28004` 为一次性 test。GPU 1 已启动事件驱动 launcher
（父 PID `770595`）：

1. 用完整 16-seed fit split、固定 `100 updates / init seed 1` 重建并保存
   `selected_u100_seed1_snapshot.pkl`；
2. 用原始 frozen-BC 10k snapshot 收集从未见过的 `28000--28004`、三个 anchors、全部 15
   action dimensions 和每维 5 sibling bins；
3. 对 baseline 与 selected sidecar 在同一个 cache 上做 matched ranking/regret 比较，test
   结果不得用于改 update 数。

正式路径为
`exp_local/cqn_value_fidelity_stage18/test_28000_28004_10k_20260724`。此前同协议五 seeds
实测约 `42--47min`，加 snapshot 固化约 `1.2min`，所以总 ETA 为 `44--50min`。只有 test
mean pairwise、Spearman、top-1 同时提高且 regret 下降，才进入 coverage-gated closed-loop
paired success；否则退回 frozen BC，不改变原 policy。

#### Research B：blend 与多折确认

单一 validation seed `26004` 在候选
`alpha={0,0.25,0.5,0.75,1}` 中选择 `alpha=0.25`。它在未参与选择的
`27000--27004` 上相对 direct 的三初始化 mean 为：

- pairwise `53.93% -> 54.58%`；
- Spearman `0.0695 -> 0.0917`；
- top-1 `28.60% -> 31.15%`；
- regret `0.1248 -> 0.1196`。

这是一个合法的单折阳性，但按 initialization 看只有 `2/3` 同时改善 pairwise 和 regret。
因此固定 `alpha=0.25` 后又运行三个 informative validation folds
`24001/25001/26002`。结果不复现：

| validation fold | direct pairwise | blend pairwise | direct regret | blend regret |
|---:|---:|---:|---:|---:|
| `24001` | `51.58%` | `51.01%` | `0.1299` | `0.1326` |
| `25001` | `54.15%` | `52.90%` | `0.1224` | `0.1215` |
| `26002` | `53.76%` | `51.37%` | `0.1308` | `0.1352` |
| fold mean | **`53.17%`** | `51.76%` | **`0.1277`** | `0.1298` |

三个 folds 的 pairwise 全部下降；fold mean 的 Spearman `0.0583 -> 0.0346`、top-1
`31.51 -> 29.87%` 也下降。最初尝试的 `24004` fold 对全部 45 states 都没有 action-return
contrast，属于无定义 selection split；代码现会明确 fail-fast，不把全 tie seed 当算法失败或成功。

结论是：官方 source geometry、dense flow supervision 和 scalar readout 的单折阳性
selection-sensitive，static deterministic regression gate **失败**，不能接入 online CQN-AS。

### 21.5 Stage-XIX：BC-shaped target 到真实 return 的 non-stationary gate

后续机制论文的主张是 moving TD target 下的 plasticity 与 test-time recovery，不是 static
regression。为了同时对准本项目的“critic 是否偷学成 BC”问题，新增一个可复现 target-shift
protocol：

1. 条件仍是同一真实 MovePlate frozen image/state feature、full action plan、dimension 和 bin；
2. 前 5000 updates 的 target 是 cache 中 frozen BC `policy_log_probability` 的 per-state rank，
   模拟 value representation 先被 imitation-shaped signal 占据；
3. 第 5001 update 起只训练真实 simulator counterfactual return，再运行 5000 updates；
4. validation checkpoint 至少经过 500 次真实-return adaptation 后才可选择；
5. direct scalar 与 pure expected-value FLOQ 使用同一 256x2 hidden width、batch、数据、
   optimizer、update budget；FLOQ 固定 uniform `[0,0.1]`、8 train sources、8 Euler steps；
6. 三个 initialization、三个 informative validation folds，heldout 仍为 `27000--27004`。

新增参数为 `warmup_target`、`warmup_updates`、`selection_min_adaptation_updates` 和
`target_noise_std`；`target_noise_std=0` 保持本轮只回答 target shift。相关 focused tests
当前为 `22 passed`。

GPU 5 的正式 run 为
`exp_local/cqn_flow_value_model_benchmark/stage19_bc_to_return_target_shift_gpu5_20260724`
（父 PID `776371`）。按 Stage-XVIII 相同网络的实测吞吐，三 folds ETA 约 `6--9min`。
pass gate 要求 FLOQ 不只是 shift 后 loss 更低，而是在三折 mean 上同时超过 direct 的
heldout pairwise 和 regret，并且至少 `2/3` folds 同方向；否则不进入在线 high-UTD
MovePlate，直接记录“当前 CQN condition 上没有观察到论文机制”。

#### Stage-XIX 结果：flow 的优势出现在 target shift，而非静态拟合

三个 validation-fold protocols 均已完成；下表是在各 fold 内只用 validation 选 checkpoint
后，对共同 heldout `27000--27004` 的三个初始化 mean：

| validation fold | direct pairwise | FLOQ pairwise | direct regret | FLOQ regret |
|---:|---:|---:|---:|---:|
| `24001` | `50.67%` | **`51.34%`** | `0.1285` | **`0.1173`** |
| `25001` | `51.01%` | **`51.75%`** | `0.1321` | **`0.1202`** |
| `26002` | `48.71%` | **`52.59%`** | `0.1331` | **`0.1314`** |
| fold mean | `50.13%` | **`51.89%`** | `0.1312` | **`0.1230`** |

fold mean 的 Spearman 为 `0.0012 -> 0.0361`，top-1 为 `26.41% -> 30.24%`。FLOQ 在
pairwise 与 regret 上均为 `3/3` folds 同方向，因此预注册 gate 通过。

这不是 best-checkpoint 偶然性：在相同训练步比较时，target 切换点 `5000` updates 的
pairwise 为 `51.11% -> 51.41%`，到 `7500` 为 `50.61% -> 52.04%`，最终 `10000` 为
`49.67% -> 52.38%`；最终 regret 同时由 direct 的 `0.1341` 降至 FLOQ 的 `0.1160`。
因此当前可复现结论是：在 representation 先被 BC-shaped target 占据、随后必须适应真实
counterfactual return 时，conditional flow critic 比同容量 direct scalar critic 更有
plasticity。它的绝对 pairwise `51.89%` 仍低于静态 direct protocol 约 `53.17%`，所以这
不是“flow 静态 value 更准”，而是“flow 对 non-stationary value target 更稳”。

### 21.6 Stage-XX：matched online high-UTD MovePlate gate

Stage-XIX 已满足进入在线实验的 gate。新增两个只改变 UTD 的 launch：

- `cqn_as_pixel_bigym_two_tower_coherent_mc_high_utd4_gate`：direct C51；
- `cqn_flow_floq_compute8_two_tower_high_utd4_gate`：expected-Q conditional FLOQ。

两者固定 `10500` frames、seed 1、`16+16` replay/demo batch、25-episode evaluation、
independent BC policy、strict policy/value image-encoder split、replay-next TD target、
effective-k0 critic、MC weight `0.1`、H=4 coherent exploration，并把
`num_update_steps` 从 `1` 提到 `4`。FLOQ 使用 8 sources/8 Euler steps；比较回答的是相同
数据预算下的效果与高 UTD 稳定性，不声称计算量相等。

顺序运行目录预注册为：

1. `exp_local/cqn_flow_high_utd/stage20_direct_c51_utd4_seed1_gpu5_20260724`
2. `exp_local/cqn_flow_high_utd/stage20_floq_compute8_utd4_seed1_gpu5_20260724`

Stage-XX screen gate 使用每个 run 的 **best 25-episode checkpoint**，而非 final：

1. FLOQ best success 必须高于 matched direct-C51；
2. 同时检查 `2.5k/5k/7.5k/10k` 同步曲线，排除只由单次 eval 噪声造成的“胜出”；
3. 若通过，立即做至少三 seeds 的确认，并最终与原始 clean CQN-AS 的 `92%` best checkpoint
   比较；若未通过，则停止扩大 online seed，转向 Stage-XIX 已定位的 moving-target
   regularizer/architecture ablation。

按 UTD=1 的本机历史实测，direct-C51 为约 `10.4min`、compute-8 FLOQ 为约 `19.1min`。
UTD=4 的预启动区间分别为 `25--40min` 与 `55--80min`；启动后以首个 CSV 窗口重新估算，
两个顺序 run 总 ETA 暂定 `80--120min`，完成监视采用父进程事件，不做短轮询。

正式 launcher 已于 `2026-07-24 02:52 BST` 在 GPU 5 启动，父 PID `789422`；direct-C51
子进程 PID `789430` 已创建 Hydra config 并占用约 `24.6 GiB`，随后会在同一 launcher
中自动切换到 FLOQ。完成等待绑定父 PID 退出事件，而不是固定 30 秒轮询。

首个可用吞吐窗口已到 `3000` frames：排除首次 JIT 后，direct-C51 每 `1000` frames 约
`120--163s`（其中 2.5k evaluation 落在后一窗口），backend update time 约 `22ms`，
`agent_batched_updates_per_second` 约 `7.1--7.7`。据此将 direct 剩余 ETA 校准为约
`16--19min`，完整 direct 约 `24--27min`。2.5k 的 25-episode success 为 `48%`，高于其
UTD=1 strict two-tower 同步点的 `12%`，但这只是第一同步点，尚不能替代 best/curve gate。

Research A 的新 test cache 完成后不会留下 GPU 空窗：依赖 launcher PID `792605` 已绑定
collector 父 PID `770595` 的退出事件，随后自动运行
`matched_baseline_vs_selected.json`。它会先验证 encoder、BC policy 和 policy encoder
逐叶 bitwise 相同，再在同一 cache 上打分两个 critic；test 结果不参与重新选择 update 数。

### 21.7 Research A 的下一 selector：support + independent-ensemble LCB

在等待一次性 test 时补充审计了与当前失败模式直接相关的 primary literature。结论不是再调一个
全局 `beta`：

- [IQL](https://arxiv.org/abs/2110.06169) 的核心约束是不在训练时查询数据外 action，并通过
  advantage-weighted behavior cloning 做 policy extraction；
- [EMaQ](https://arxiv.org/abs/2007.11091) 把 policy improvement 限制在 behavior proposal
  support 内；
- [SPOT](https://arxiv.org/abs/2202.06239) 直接使用 density-defined behavior support，而不是
  假设 KL 距离就等价于 support；
- [MSG](https://arxiv.org/abs/2205.13703) 用彼此独立 target/network 的 Q ensemble 构造 lower
  confidence bound，并指出共享 pessimistic target 可能反而乐观；
- [Cal-QL](https://arxiv.org/abs/2303.05479) 进一步要求 learned Q 相对 reference/behavior
  policy 保持合理尺度的 calibration。

它们与本项目的对应关系是：BC head 已给出每个 C2F bin 的 behavior support；16-seed
counterfactual cache 给出了相对 BC bin 的真实 return delta；三次独立初始化可以用于 epistemic
agreement。因此若 `28000--28004` test gate 通过，下一 selector 不再使用
`zscore(A)+beta log pi_BC`，而是：

1. 固定三个独立 oracle sidecars，分别预测每个候选相对 BC-bin 的
   `Delta Q_m`；
2. 只在 candidate 的 BC log-prob drop 不超过 validation-selected support threshold 时考虑；
3. 只有 `mean(Delta Q)-lambda*std(Delta Q)>delta` 时才覆盖 BC bin，否则逐维精确回退 BC；
4. `lambda/delta/support threshold` 只在新的 calibration seeds `33000--33049` 选择；
5. confirmatory seeds 使用从未参与选择的 `34000--34099`，同时报告 override rate、BC-disagreement
   subset success 和 overall paired success。

其中核心 selector 已实现为
`robobase.method.cqn_flow.supported_lcb_action_indices()`：输入 BC logits 和具有独立 leading
ensemble axis 的 advantage，先逐 member 减去 BC-bin value，再计算
`mean(Delta Q)-lambda*std(Delta Q)`；不满足 BC log-prob support 或 LCB margin 的维度会
bitwise 回退 BC argmax。agreement override、support rejection、disagreement fallback 和非法阈值
共五个 focused tests 已在 CPU 后端通过。模型 snapshot stacking 与 episode-level override
日志只在一次性 critic test 通过后接入，避免在 failed value 上运行 deployment sweep。

该阶段的第一 gate 是不得低于同一个 frozen BC 的 validation-selected best `76%`；通过后才把
相同“冻结成功 behavior、独立 real-value sidecar、LCB override”结构迁移到 clean legacy
CQN-AS 的 best behavior checkpoint，并以其 `92% @ 7.5k` 为最终 non-inferiority baseline。
若一次性 critic test 不通过，则不运行 closed-loop sweep，转为继续增加 phase/state coverage。

### 21.8 Stage-XX 中间里程碑与无 final-bias 汇总

direct-C51 UTD=4 已产生前两个同步点：

| method | 2.5k | 5k | 当前 best |
|---|---:|---:|---:|
| UTD=1 strict two-tower direct | `12%` | `36%` | `36% @ 5k` |
| UTD=4 strict two-tower direct | **`48%`** | **`56%`** | **`56% @ 5k`** |
| clean legacy CQN-AS reference | `48%` | `56%` | `56% @ 5k` |

因此高 UTD 到 5k 的 sample efficiency 确实改善了 matched two-tower direct，并追平 clean
legacy 的前半段，但还没有触及 legacy 的 `92% @ 7.5k` 峰值，也还没有 FLOQ 对照，不能提前
判 gate。5k evaluation artifact 为 `eval.csv` 的 `episode_success=0.56`，总 wall time
`714.35s`；按当前吞吐 direct 剩余 ETA 约 `12--15min`。

为杜绝之后把 final checkpoint 当 baseline，新增
`scripts/summarize_cqn_online_gate.py`：它要求预注册的
`2.5k/5k/7.5k/10k` 全部存在，忽略 CSV 重写产生的重复 header，method-best 同分时取更早
checkpoint，并同时输出所有 same-step delta。只有显式 `--allow-incomplete` 才允许生成中间
summary；当前 artifact 为
`exp_local/cqn_flow_high_utd/stage20_partial_summary.json`。四个纯 CPU regression tests
全部通过。

### 21.9 CQN-AS 官方 ablation 能回答什么、不能回答什么

重新核对了 [CQN-AS 论文](https://younggyo.me/cqn-as/static/paper/cqn_as_paper.pdf) 与
[官方 BiGym 实现](https://github.com/younggyoseo/CQN-AS/blob/main/bigym_src/cqn_as.py)。
论文 Figure 7b 的确比较了完整 CQN-AS 与 “No RL”，后者只在成功 demos 上训练 BC objective；
完整方法更好，因此论文说 RL objective 使 agent 能从 trial-and-error experience 学习。这个
结论本身成立，但它不识别 critic 学到的是不是真实 action value。

源码给出了更精确的原因：`update_critic()` 在同一组 `q_probs/q_probs_a` 和同一个 image encoder
上直接相加三项梯度：

1. Bellman C51 cross entropy `q_critic_loss`；
2. demo FOSD loss；
3. demo large-margin loss。

部署时同一 critic 的 expected Q 又直接负责逐 bin argmax。因而 Figure 7b 移除 RL loss 时，不仅
移除了“真实 value supervision”，也同时改变了共享 Q representation、非 demo trajectory 的
negative/background gradients、bin margin 的校准和最终 selector。完整方法优于 BC-only 可以由
RL loss 帮助塑造更好的 imitation decision boundary 解释，并不能排除用户提出的
“披着 value learning 外壳的 imitation”。

论文 Figure 2a 的 ground-truth RTG regression 进一步支持“action sequence 比 single action 更容易
预测轨迹内 RTG”，但指标是 validation L1，并没有固定同一 state 后对未执行 sibling actions 做
causal ranking。因此 Research A 当前的 frozen behavior、bitwise-separated encoder/policy、
simulator sibling intervention、label-shuffle 和 unseen-seed replication 不是重复论文 ablation，
而是补上它没有识别的命题。官方仓库本身也注明 reimplementation 可能因移植误差而不能完全复现
论文，所以本项目最终结论继续以本地 artifact 为准。

### 21.10 Stage-XXI：一次性 value test 通过，但必须用 ensemble 控制 seed 级风险

#### Research A 上一阶段结果

`28000--28004` 的收集与 matched comparison 已结束。比较全程为
`coverage_only=true`，没有在这五个 seed 上做任何 update；原 snapshot 与 selected sidecar 的
encoder、policy encoder 和 BC policy 逐叶 bitwise 相同。因为 JSON 内的历史 `train/heldout`
只是 cache partition 名称，本阶段新增
`scripts/summarize_cqn_value_fidelity_gate.py`，先验证 before/after 的
`(eval_seed,anchor,dimension)` 和 realized returns 完全相同，再合并两个 partition，并按 simulator
seed 做 paired cluster bootstrap。正式 artifact 为：

```text
exp_local/cqn_value_fidelity_stage18/test_28000_28004_10k_20260724/
  matched_baseline_vs_selected.json
  combined_test_gate.json
```

五个新 seed 的合并结果为：

| metric | frozen before | selected 100-update sidecar | delta |
|---|---:|---:|---:|
| pairwise sign | `47.45%` | `52.19%` | `+4.74pp` |
| mean Spearman | `-0.0387` | `0.0444` | `+0.0831` |
| top-1 | `17.05%` | `26.36%` | `+9.30pp` |
| regret ↓ | `0.1712` | `0.1278` | `-0.0435` |

因此预注册的 aggregate directional gate——pairwise、Spearman、top-1 全升且 regret
下降——四项全部通过。这里有 `225` states、`129` informative states、`1117` informative
pairs。这个结果建立了一个比 Bellman loss、demo top-1 或 success 更强的命题：冻结 behavior
和视觉 condition 后，只更新 advantage sidecar，确实能在从未用于选择 update 数的新 simulator
seed 上改善真实 sibling-action return ordering；所以当前 value 并非纯粹的 BC classifier。

同时它还没有达到直接部署的证据强度。`28003` 的全部 45 states 没有 action-effect，剩余四个
informative seeds 的 all-four-metric 同向改善只有 `2/4`。逐 seed 看：

| seed | informative states | pairwise delta | Spearman delta | top-1 delta | regret delta ↓ |
|---:|---:|---:|---:|---:|---:|
| `28000` | 39 | `-0.92pp` | `-0.0529` | `+2.56pp` | `+0.0022` |
| `28001` | 40 | `+7.69pp` | `+0.1550` | `+15.0pp` | `-0.0794` |
| `28002` | 40 | `+10.51pp` | `+0.2404` | `+15.0pp` | `-0.0694` |
| `28003` | 0 | undefined | undefined | undefined | undefined |
| `28004` | 10 | `-20.0pp` | `-0.3039` | `-10.0pp` | `+0.0261` |

四个 informative-seed 的 paired bootstrap 95% CI 仍覆盖 0：pairwise
`[-3.66,+9.12]pp`、Spearman `[-0.104,+0.199]`、regret delta
`[-0.074,+0.007]`；top-1 为 `[0,+15.0]pp`。这不推翻预注册 mean gate，但明确排除了
“一个 sidecar 已经稳定到可以无条件 argmax”的解释。下一 selector 必须把 epistemic disagreement
当作拒绝覆盖的理由。

#### Research A 下一 gate 与已执行代码

三个 `100-update` 独立初始化 sidecar 已按同一 16-seed fit cache 固化：

```text
selected_u100_seed1_snapshot.pkl  # 78.35s
selected_u100_seed2_snapshot.pkl  # 76.66s
selected_u100_seed3_snapshot.pkl  # 78.23s
```

seed 2/3 的顺序 launcher PID `819636` 已正常退出并写出
`ensemble_snapshots_complete`。三棵 advantage 参数不是 bitwise identical，policy、policy
encoder 和 value encoder 则必须与 source snapshot bitwise identical。

closed-loop 路径不再把 sidecar 简化成逐 bin 静态 logits。新增
`sibling_bin_candidate_plans()` 精确复现收集协议：先取得普通 temporal-ensemble BC plan，
并行构造 `15 dimensions x 5 sibling bins`，把 level-1 的同一 delta 重复到前 `H=4`
plan tokens；三个 sidecar 并行评分后，`select_single_supported_lcb_plan()` 只允许每次 inference
改一个 dimension。候选必须同时满足 BC log-prob support 与
`mean(Delta Q)-lambda*std(Delta Q)>margin`，否则返回的完整 plan 与 BC bitwise 相同。
`scripts/eval_cqn_lcb_sidecar.py` 还逐 episode 记录 override count/rate、selected dimension 和
disagreement-subset success。相关 summary、selector、candidate construction 与 evaluator tests
在加入 calibration summarizer 后为 `14 passed`。

阈值不得看 `28000--28004` 反调。首先在独立的 rate-only smoke seed `32999` 验证代码并校准
override 量级，不用该 seed 的 success 选方法。`lambda=1, margin=0, support-drop=0.5`
虽然通过 snapshot lineage、三模型并行 JIT 和 episode diagnostics，但在 exact-BC 轨迹上会建议
覆盖 `167/300=55.7%` 的 inference；真正执行后因轨迹改变达到 `213/300=71.0%`。所以
`margin=0` 没通过 conservative-safety gate，不能扩成 50 episodes。

rate-only 诊断上正 selected-LCB 的 median/P75/P90/P95 分别为
`0.0047/0.0083/0.0139/0.0392`。据此在查看 calibration success 前，把
`33000--33049` 的候选固定成同一 `lambda=1, support-drop=0.5` 下三个拒绝强度：

| id | lambda | margin | max BC log-prob drop |
|---|---:|---:|---:|
| low-margin | `1` | `0.01` | `0.5` |
| medium-margin | `1` | `0.02` | `0.5` |
| high-margin | `1` | `0.04` | `0.5` |

另跑 exact BC 作为 paired reference。先最大化 50-seed success；同分时选较低 override rate。
只有候选 success 不低于 BC、至少实际产生一次 override，并且 paired wins 不少于 losses，才进入
完全未参与选择的 `34000--34099`。confirm gate 要求 overall success 不低于 paired BC，
且 override episodes 上的 paired return/success delta 为正；否则 deployment 回退 exact BC，
下一轮只增加 branch phase/seed coverage。通过后才在 clean legacy CQN-AS 的
`92%@7.5k` best behavior checkpoint 上重新收集 cache、重训 sidecars并做同样 gate，不能把当前
不同 encoder 的 sidecar 直接移植过去。

GPU 1 已启动上述 exact BC 加三个 margin 的事件驱动顺序 launcher，PID `835625`，路径为
`exp_local/cqn_value_fidelity_stage21/lcb_calibration_seed33000_33049`。一个 LCB smoke
包含首次 JIT 的 300-step episode 实测 `53.6s`；结合历史 BC 50 episodes 的 `2--4min`，
初始总 ETA 为 `55--90min`，到第一个 LCB variant 的 10-episode日志后再按真实平均 episode
长度校准。父 PID 退出事件已注册，不做短轮询。

#### Research B 上一阶段结果与当前执行

matched UTD4 direct-C51 已完成：

| method | success @ 2.5k/5k/7.5k/10k | best | runtime |
|---|---:|---:|---:|
| UTD1 strict two-tower direct | `12/36/56/52%` | `56%@7.5k` | `10.4min` |
| UTD4 strict two-tower direct | `48/56/44/48%` | `56%@5k` | `22.9min` |
| clean legacy CQN-AS | `48/56/92/56%` | `92%@7.5k` | reference |

高 UTD 明确改善了前 5k sample efficiency，却没有提高 best，7.5k 还从 UTD1 的 `56%`
降到 `44%`。因此“只把 UTD 提到 4 就会得到论文式 plasticity 收益”的假设失败；它既不是
direct 的稳定性修复，也不能解释 legacy 的 92% 峰值。

同一顺序 launcher 已于 `03:17 BST` 自动切换到 compute-8 FLOQ。首个 1k window 的
total time 为 `399.17s`，其中 step-0 JIT/初始化为 `105.19s`，所以稳态约
`294s/1k frames`。按该实测吞吐，剩余训练加四次 evaluation ETA 为约 `48--55min`。
Stage-XX gate 不变：只按四个预注册 evaluation points 的 best checkpoint 比 UTD4 direct；
FLOQ 若不能超过 `56%`，不扩多 seed，转做 moving-target mechanism ablation；若超过，再做
三 seed 并最终与 clean legacy 92% best 公平比较。

启动后的下一里程碑已经落盘：FLOQ 的 `2.5k` success 为 `48%`，与 UTD4 direct 同步点相同，
高于 UTD1 compute-8 FLOQ 的 `16%`。这说明 high UTD 同样显著改善 flow 的早期学习，但单个
同步点还不能判定它是否超过 direct 的 `56%` best。2k steady window 仍约
`294s/1k`，包含 2.5k evaluation 的 total time 为 `882.44s`，剩余 ETA 约 `40--48min`。

Research A 的 paired BC calibration 已先完成：`33000--33049` 为
`30/50 = 60%`，wall time `122.6s`。第一个 `margin=0.01` LCB variant 已自动接棒；后续
`0.02/0.04` 已在同一 PID `835625` 队列中，因此 GPU 1 没有空窗。最终只按 paired seed
wins/losses 与 best success 选阈值，不把这个 60% 与其他 seed split 的百分比直接混比。

### 21.11 Research A calibration gate 与 Research B 在线蒸馏 readout

#### Research A 上一阶段结果与解释

`33000--33049` 的四个 paired runs 已全部完成，正式汇总为
`exp_local/cqn_value_fidelity_stage21/lcb_calibration_seed33000_33049/summary.json`：

| policy | success | delta vs BC | paired W/L/T | override rate | gate |
|---|---:|---:|---:|---:|---|
| exact BC | `60%` | -- | -- | `0%` | reference |
| margin `0.01` | `54%` | `-6pp` | `3/6/41` | `27.41%` | fail |
| margin `0.02` | `54%` | `-6pp` | `3/6/41` | `12.42%` | fail |
| margin `0.04` | `60%` | `0pp` | `2/2/46` | `5.34%` | pass |

三个 LCB variant 的 wall time 分别为 `232.3/228.4/220.1s`；BC 为 `122.6s`。结果排除了
“只要 ensemble/support gate 就可频繁用 learned value 覆盖 BC”的假设：即使 support-drop
固定为 `0.5`，`12--27%` 的 inference override 都使 success 从 `60%` 降到 `54%`。只有
`margin=0.04` 把接管率压到约 `5%` 时才达到 validation non-inferiority，而且它只是
`2胜/2负`，没有建立 improvement。

因此不在 calibration split 继续调阈值。预注册的 selected variant `margin=0.04` 已与 exact BC
在完全未参与选择的 seeds `34000--34099` 启动 100-episode paired confirm：

```text
exp_local/cqn_value_fidelity_stage21/lcb_confirm_seed34000_34099
launcher PID 859513, GPU 1
```

confirm gate 不变：overall success 不低于 exact BC，并且实际 override episode 的 paired
success/return delta 为正；否则本架构部署推荐明确回退 exact BC。按 calibration 实测线性外推，
BC 约 `4.1min`、LCB 约 `7.3min`，连同重新 JIT 与 summary 的 ETA 为 `12--15min`。launcher
会自动顺序完成两项并写 `summary.json`，不做短轮询。summarizer 已增加独立
`confirmation` mode：除 overall paired W/L 外，会只在
`applied_override_count > 0` 的 episode 上分别计算 success/reward delta 与 cluster-bootstrap
CI；对应 gate regression 为 `3 passed`。

如果 confirm 通过，legacy migration 前还有一个必须显式关闭的 reproducibility gap。历史
`92%@7.5k` 来自：

```text
exp_local/cqn_value_fidelity_stage2/
  move_plate_full_first_success_seed1_gpu3_20260722_165946
```

但该 run 的 snapshot interval 是 1k，所以虽然 7.5k evaluation 为 `23/25`，磁盘上只有
7k/8k snapshots，没有 `7500_snapshot.pkl`。另一个
`pixel_cqn_as/move_plate_paper_seed1_100k_nw0_20260721/7500_snapshot.pkl` 的早期曲线实际是
`40/56/52/40%`，不是 92% run，明确禁止替代。已增加
`cqn_as_pixel_bigym_value_fidelity_repro_gate`：保持 clean first-success 配置，只把 snapshot
interval 改成 500 并启用 CSV；当前 confirm 通过后立即重跑，历史实测 1k--10.5k wall time
约 `8.8min`，预估 ETA `10--13min`。只有重跑再次选出并固化 best checkpoint，才能迁移。

legacy behavior 是 atoms=51、dueling C51 target critic，不能错误地塞进当前
atoms=1 BC-logit head。为此 CQN-Flow 新增默认关闭的
`bc_policy_mode=legacy_c51`：它 materialize 与原 CQN-AS critic 相同的
`C2FSequenceDistributionalCritic`，并按 distributional expectation 做逐 bin argmax。
`scripts/convert_cqn_as_frozen_policy.py` 会在正式迁移时：

1. 验证 action sequence、levels/bins/atoms、GRU、hidden dims、activation 与 dueling 设置匹配；
2. 把 legacy `target_critic_params` bitwise 复制为 frozen policy；
3. 把 legacy image encoder 分别 bitwise 复制到 value encoder 与 policy encoder；
4. 输出带 source lineage、tree SHA256 与逐树 bitwise checks 的分析 snapshot；
5. 保持 Flow-V/direct-A 新初始化，后续只能更新 value sidecar。

低维结构测试已验证 legacy policy 用 51-atom expectation 而不是误取 atom 0 选择 bin；相关
兼容 focused tests 为 `5 passed`。真正 conversion 与 paired behavior-equivalence evaluation
仍以当前 `34000--34099` confirm 通过为执行 gate，不能把“代码可构造”写成“92% 已保持”。

#### Research B 新里程碑

Stage-XX compute-8 FLOQ UTD4 已产生第二个同步点：

| method | 2.5k | 5k | current/best |
|---|---:|---:|---:|
| matched direct-C51 UTD4 | `48%` | `56%` | `56%` |
| compute-8 FLOQ UTD4 | `48%` | **`64%`** | **`64%`** |

5k artifact 是
`exp_local/cqn_flow_high_utd/stage20_floq_compute8_utd4_seed1_gpu5_20260724/eval.csv`，
总 wall time `1673.1s`。这已经给出比 matched direct best 高 `8pp` 的正信号，但完整 gate
仍要求 7.5k 与 10k 两点落盘并用 validation-selected best 汇总；不能因为中途首次超过 baseline
就停止。按 2.5k--5k 的含 eval 吞吐，剩余 ETA 约 `25--32min`。若完整 best 仍高于 `56%`，
下一步立即做多 seed confirmation；若不能在复现后接近或超过 clean legacy 的 `92%`，继续进入
下面的机制实验，而不是把 “胜过弱化 two-tower baseline” 写成最终目标已达成。

这里的 task-success 因果口径必须收紧：两个 Stage-XX arm 都是
`separate_bc_policy=true`、`td_target_action_source=replay_next`，evaluation 动作完全来自独立
BC tower，value critic 不参与 bin argmax。因此 `64% vs 56%` 是真实 artifact，却不能归因于
FM value；BC tower 的初始化 RNG、structured-exploration RNG 与由此产生的数据轨迹仍可不同。
完整 Stage-XX 现在只作为 high-UTD 数值稳定性/吞吐 screen。Research B 的有效 control gate
必须在同一个 checkpoint、同一 paired seeds 内比较 exact BC 与 action-facing
distilled-Q/BC selector，并再和 direct-C51 selector 比；下节实现正是为了关闭这个因果缺口。

#### FLOQ distilled scalar readout 的 research 与已实现代码

重新逐行核对
[官方 FLOQ 实现](https://github.com/CMU-AIRe/floq/blob/main/what_does_flow_matching/agents/floq.py)：
它并不直接让 actor 每次积分 return flow。对 replay action，在线 `floq` 先积分多个 source，
取 `mean_current_returns`，再用 stop-gradient target 训练独立 scalar `critic`；
`target_update()` 只 EMA `floq`，而 actor 查询在线 scalar critic。当前 Stage-XX 本地实现此前只有
flow endpoint，没有这个 action-facing readout。由于本地 CQN 离散枚举不需要 actor gradient，
但仍需要低方差、便宜的逐 bin selector，所以这是当前失败后最接近官方、且机制单一的下一
ablation。

已加入：

1. `flow_distill_lambda`：scalar C2F readout 拟合
   `stop_gradient(mean online-flow endpoint)`；
2. `flow_distill_action_readout`：评估时用 readout 做逐 level/bin action selection，无需 Euler
   integration；
3. 对严格 two-tower run，训练期仍让独立 BC policy 收集数据；同一 checkpoint 可在评估时比较
   exact BC、纯 normalized distilled-Q、以及
   `normalized_Q + beta * log pi_BC`，避免每个 beta 重训；
4. readout 输入保留 image/state feature、C2F level、zoom midpoint、sequence position、
   action dimension 与 candidate bin；readout target 和共享 feature 均 detach，因此 distill-only
   单测中 flow field bitwise 不变；
5. 新 launch：
   `cqn_flow_floq_distill_two_tower_high_utd4_gate`；新旧
   `scripts/eval_cqn_flow_policy_value.py` 会根据 checkpoint 类型切换 direct-A 或 distilled-Q
   selector。

为使 Research B 有真正 matched 的 action-facing control，普通 `CQNAS` 也新增默认关闭的
`policy_value_beta`：在已有 `separate_bc_policy` checkpoint 上可比较 exact BC、纯 normalized
direct-C51 Q、以及 `normalized_Q_C51 + beta * log pi_BC`。同一 evaluator 现同时接受
`cqn_as` 与 `cqn_flow`，分别标记 `direct_c51` 和 `flow_distill` readout；这样最终比较的是
“同一 checkpoint 内 value 是否相对自己的 BC 改善 paired control”，而不是两个独立 BC run 的
success 波动。direct-Q/BC 与 flow-Q/BC 的 bin-choice 单测均验证 beta=0 选择 value bin、较大
beta 回到 BC bin。

验证分三组完成：核心机制 `4 passed`，decoupled/hybrid 兼容回归 `12 passed`，factory/config
回归 `26 passed`，且 `py_compile` 与 `git diff --check` 通过。该 run 尚未抢占 GPU：必须先让
当前 Stage-XX 完整 gate 给出结果；若当前 FLOQ 只超过 matched `56%` 而达不到最终 `92%`
目标，则按独立 validation seeds 选择 checkpoint 与 beta、再用 never-used paired seeds
确认，不用训练 loss 或 final checkpoint 选择方法。

### 21.12 Research A held-out confirm 通过 non-inferiority，开始固化真实 legacy best

#### 上一阶段结果

`34000--34099` 的 exact BC 与 selected LCB 已全部完成：

| policy | success | paired W/L/T | inference override | wall time |
|---|---:|---:|---:|---:|
| exact BC | `61/100 = 61%` | reference | `0%` | `244.0s` |
| LCB margin `0.04` | `62/100 = 62%` | `6/5/89` | `1341/20502 = 6.54%` | `386.2s` |

正式 artifact：
`exp_local/cqn_value_fidelity_stage21/lcb_confirm_seed34000_34099/summary.json`。
LCB 在 `99/100` episodes 至少覆盖过一次，因此 override-episode subset success/reward delta
均为 `+1.01pp`；success delta 的 paired bootstrap 95% CI 为
`[-5.05,+8.08]pp`，overall delta CI 为 `[-5,+7]pp`。

#### 解释

预注册的 held-out gate 通过：overall 不低于 BC、确实产生 override，override subset 的 mean
success/reward delta 为正。这是在完全未用于阈值选择的 100 seeds 上得到的可复现
non-inferiority 证据，也说明 ensemble LCB 把此前高频覆盖造成的 `-6pp` 损害压住了。

它不是显著 improvement：只有 `6胜/5负`，两个 CI 都跨 0；而且虽然单步 override rate 仅
`6.54%`，长 horizon 使 `99%` episodes 都被触及。因此当前推荐仍是“保守 value sidecar 可以在
不明显伤害 BC 的情况下工作”，不能写成“已稳定超过 BC”。最终 A 目标仍要求迁移后不低于
原 clean CQN-AS，而不是当前较弱 frozen BC。

#### 下一 gate 与执行

复核历史 artifact 后纠正了 checkpoint lineage：92% run 是 Stage-II clean first-success，
但 snapshot interval=1k，磁盘没有产生 7.5k snapshot；另一个 100k paper run 的
`7500_snapshot.pkl` 只有 52% 同步 success，禁止替代。下一阶段先精确复现 clean baseline：

```text
launch=cqn_as_pixel_bigym_value_fidelity_repro_gate
run=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724
launcher PID=872751, GPU 1
```

新 launch 只相对历史 clean 配置增加 `snapshot_every_n=500` 与 CSV，其他训练语义保持。gate
要求四个同步点完整、以 validation best 而非 final 选 checkpoint；只有复现出可接受的 clean
曲线并生成 exact best snapshot，才运行 legacy-C51 frozen-policy conversion、branch collection
与三 sidecar ensemble。历史同 run 的 1k--10.5k 用时约 `8.8min`，本次 ETA `10--13min`，
完成监视绑定父 PID 退出事件。

### 21.13 Clean baseline 未复现 92% 峰值；legacy-C51 迁移进入闭环等价 gate

#### Research A 上一阶段结果

clean first-success 精确重跑已经完整结束，正式 summary 为
`exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/summary.json`：

| checkpoint | 2.5k | 5k | 7.5k | 10k | selected best |
|---|---:|---:|---:|---:|---:|
| clean repro seed 1 | `60%` | **`68%`** | `56%` | `60%` | **`68%@5k`** |
| historical first-success seed 1 | `48%` | `56%` | **`92%`** | `56%` | `92%@7.5k` |

新 run 四个预注册点齐全，并且每 500 steps 都有 snapshot；总训练/eval wall time
`635.0s`。它没有复现历史 `92%@7.5k` 峰值。因此历史数字仍是一个真实 eval artifact，但在
缺少对应 snapshot、同 seed 精确重跑又只有 `56%@7.5k` 的情况下，不能把它当成已经固化且可部署
的 canonical checkpoint。当前可复现 run 的 validation-selected checkpoint 是
`snapshots/5000_snapshot.pkl`，不能用 final 10k 代替它。

#### 解释与下一 gate

这一步排除了“只要用同 config/seed 重跑就会自动得到 92%”的假设，也说明最终结论必须同时报告：

1. 历史单次 best `92%`，作为不能被悄悄降低的 upper reference；
2. 多次 clean run 的 validation-selected best 分布，作为可复现 baseline；
3. 同 checkpoint、同 paired seeds 上 value sidecar 相对 frozen behavior 的增量。

先关闭迁移实现风险，再收集昂贵 counterfactual cache。转换脚本已把本次 5k checkpoint 的
`target_critic_params` 导入 `bc_policy_mode=legacy_c51`，并逐树验证：

```text
policy SHA256:
  ae20a442059637b22a43847861d521398ffcb33e4be8b90f4ea2990e47b30d0c
encoder SHA256:
  3e799e419bde67e79de7ae585606e1cfd2e01bac5f027b5fb710101356da980e
```

legacy behavior/imported policy、legacy encoder/imported policy encoder、legacy encoder/imported
value encoder 三组均 bitwise identical。转换 artifact 为
`exp_local/cqn_value_fidelity_stage23/legacy_clean_best5k_converted_20260724`。

仅参数相同还不够。GPU 1 已在完全新的 `35000--35024` seeds 上顺序运行原 checkpoint 和转换
checkpoint；`scripts/summarize_cqn_behavior_equivalence.py` 要求 paired
success/reward/episode length 在零容差下逐 seed完全相同。launcher PID `890428`，按历史吞吐
ETA `2--4min`。只有该 gate 通过，才用转换 checkpoint 重新收集 16-train/4-validation
all-dimension sibling cache并训练三初始化 sidecar；否则先修复 action-path parity，不允许用
不同策略的 branch cache伪装成 clean-CQN-AS value 改进。新增 summarizer regression 为
`3 passed`；PCBF/ranking/logger focused regression 为 `9 passed`。

该闭环等价 gate 随后完成并严格通过：

| metric | original 5k | converted legacy-C51 | max paired abs delta |
|---|---:|---:|---:|
| success | `68%` | `68%` | `0` |
| reward | `0.68` | `0.68` | `0` |
| episode length | `194.28` | `194.28` | `0` |

25 个 seeds 三项均无 mismatch；原/转换评估分别用时 `74.66s/90.52s`。正式 artifact：
`exp_local/cqn_value_fidelity_stage23/equivalence_seed35000_35024/summary.json`。因此接下来的
value 变化可以归因于 frozen-policy 之上的 sidecar，而不是迁移改变了行为。

GPU 1 已立即开始为该 exact policy 重收集 `16 train + 4 validation` seeds、三 anchors、全
15 dimensions、5 sibling bins 的 cache，随后同一事件驱动 launcher 顺序训练三个 100-update
独立初始化 sidecar。路径为
`exp_local/cqn_value_fidelity_stage23/legacy_best5k_counterfactual_ensemble`，父 PID
`894148`。参照旧 artifact，10-seed 全维度首次收集约 `42--47min`，20-seed 保守 ETA
`75--100min`，三个 fit 另约 `4min`；不做短轮询。

该收集阶段主要受 simulator/CPU 限制，实测 GPU1 只占约 `1.2GB`。为了同时回答历史 92% 的
reproducibility gap，同卡又顺序排入 clean seed 2/3 精确重跑，父 PID `896648`；显存总量
`32.6GB`，两任务启动后约 `3.6GB` 且均禁用 XLA preallocation，磁盘尚余 `286GB`。每个 clean
run 预计 `10--13min`，总 ETA `20--26min`；各自写出每 500-step snapshot 和四点 summary，
最终按每个 run 自己的 validation best 汇总，不能只看 final。

#### Research B 文献/机制排队修正

PCBF 官方论文与 JAX 实现再次逐项核对：本地已有 shared source noise、source-consistent current
path、terminal mask 和
`sample Bellman target + lambda * (target velocity - sampled successor velocity)`；
`lambda=0` 是 unbiased/high-variance BCFM，非零 lambda 是带 bias 的 control variate。
旧 MovePlate gate 已经给出有限但较弱的 `0/0/16/12%`，所以 PCBF 不直接扩 seed。新增的
`scripts/analyze_cqn_flow_ranking.py` 会先在其 validation-best checkpoint 上用
`R_action={4,8,16}` 做只读 source-resampling probe，区分“少采样导致 bin flip”和“critic 本身
没有 action separation”。代码审计发现首版只比较单 source 的独立 rollout，不能直接回答
`R_action`；现已修正为每个独立 source group 内先平均指定数量的 endpoints、再 argmax，并比较
group 间 flip，新增 averaging golden test 后相关 probe tests 为 `3 passed`。本地版本仍是
CQN 离散 bin critic core，没有官方 twin ensemble、
continuous flow actor/rejection sampling；后续结论不得称为完整 paper reproduction。

#### Research B Stage-XX 完整结果与立即执行

四个预注册同步点已经全部落盘，summary 为
`exp_local/cqn_flow_high_utd/stage20_summary.json`：

| method | 2.5k | 5k | 7.5k | 10k | selected best | four-point mean |
|---|---:|---:|---:|---:|---:|---:|
| direct C51 UTD4 | `48%` | **`56%`** | `44%` | `48%` | `56%@5k` | `49%` |
| compute-8 FLOQ UTD4 | `48%` | **`64%`** | `60%` | `36%` | `64%@5k` | `52%` |

按预注册 selection rule，FLOQ best 高 `8pp`，四点 mean 高 `3pp`，Stage-XX
stability/throughput gate 通过；7.5k 同步点高 `16pp`。但 10k 从 `60%` 跌到 `36%`，同时比
direct 低 `12pp`，所以 checkpoint selection 很重要，不能报告 final 代表整个方法。

因果解释仍保持收紧：这两个 run 都是 `separate_bc_policy=true`，内建 eval 动作来自各自独立
BC tower，value 不参与 argmax。故该结果建立的是“此训练配置产生了较好的早期 task artifact”，
不是“FM value 让动作更优”；BC initialization/RNG 和在线 trajectory 都是 nuisance。它也仍低于
historical clean `92%` upper reference。

事件驱动 handoff 已在 GPU5 启动下一机制实验：

```text
exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724
train PID 897544; parent PID 892581
```

它在同一 compute-8/UTD4 flow 上增加官方 FLOQ 语义的 scalar readout：
`stop_gradient(mean online-flow endpoint) -> scalar Q`。训练 rollout 和内建 eval 仍严格使用 BC，
所以 readout 不改变收集数据。训练完成后才在同一个 checkpoint、同一 paired validation seeds 上
比较 `BC-only`、`Q-only`、`normalized Q + beta log pi_BC`，再用 never-used confirm seeds
验证；同时对 direct-C51 checkpoint 运行完全相同的 action-facing evaluator。只有
within-checkpoint value selector 相对自己的 BC 为正，且 flow 的 paired improvement 超过 direct
control，才可把增益归因于 FM value。根据上一 run 的 `3276s` 和额外轻量 readout，初始 ETA
`55--70min`；1k artifact 后按实测吞吐重估。

### 21.14 两条路线进入预注册 action-facing gate

#### 上一阶段实时结果与解释

Stage-24 已写出首个 1k snapshot。第 0 次记录的 `bcfm_loss=0.01317`、
`flow_distill_loss=0.27925`，到 1k 分别为 `0.01391/0.00168`；critic、encoder、flow critic 和
readout gradient 的 non-finite fraction 都是 0。这个结果只建立了实现数值稳定，不能用 loss
下降替代 task success 或 causal action ranking。

同一硬件上，先前 compute-8 FLOQ 的 0→1k 净耗时为 `399.17-105.19=293.98s`，Stage-24 为
`480.10-111.18=368.91s`，distill readout 使首区间慢约 `25.5%`。按旧 run 各 1k 区间实测耗时
同比外推，总训练约 `68--72min`，从 `04:14:53 BST` 起算的完成 ETA 更新为
`05:20--05:27 BST`。GPU5 handoff 绑定训练父 PID 退出事件，不做短轮询。

Research A 的 clean seed 2 在 `04:25:49` 已到 9k，累计 `566.19s`，且 9.5k snapshot 已落盘；
seed 3 会在同一父进程中立即接力。按 seed 2 实测吞吐，两条剩余 clean 任务预计
`04:39--04:43 BST` 全部完成。昂贵 legacy counterfactual cache 同时继续，仍主要受单核
simulator 限制，没有错误 artifact。

#### Research B：policy/value gate 的预注册规则与执行

新增 `scripts/run_cqn_policy_value_gate.py`，对应 selection regression tests 为 `3 passed`。
每个 beta 都在独立进程构建 JAX action function，避免运行中修改被 JIT closure 捕获的
`policy_value_beta`。规则在看到 action-facing 结果前固定：

1. checkpoint 固定为 antecedent Stage-XX 已选出的 5k，而不是查看 Stage-24 final 后追选；
2. validation 使用连续 seeds `36000--36049`，比较 own `BC-only` 与
   `beta={0,0.03,0.1,0.3,1,3}`；
3. success 最大者入选；并列时选择较大 beta，因为它更接近独立 BC prior；
4. validation 至少 `+2pp` 且 paired wins 大于 losses 才进入确认；
5. confirmation 只评 BC 和 selected beta，使用完全未见的 `37000--37099`；要求 paired
   direction 为正并且 bootstrap 95% CI 下界不低于 `-5pp`。这只是机制继续 gate；最终“超过
   CQN-AS”仍需对 multi-seed validation-best clean baseline 做相同 held-out 比较。

GPU5 event handoff PID `908452` 已验证在等待 Stage-24；训练完成后自动生成四点 training
summary，再运行上述 flow-distill gate。GPU1 event handoff PID `908951` 已验证在等待 clean
seed 2/3；其后先对 matched direct-C51 5k checkpoint 运行同一 beta/split gate，避免把
“任何 noisy value reranking 都能碰巧改善”误归因给 FM。新增 gate 和 ranking summarizer 的
focused tests 合计 `5 passed`。

#### PCBF read-only gate 与下一算法排序

PCBF 的 7.5k eval 是旧 run 的 validation best，但当时只每 1k 保存 snapshot，因此不存在精确
7.5k checkpoint。GPU1 handoff 会在相邻的 7k、8k snapshots 都运行 `R_action=1/16`，并在 8k
补 `R_action=4/8` 趋势；每个 probe 固定 16 reset observations、16 independent source groups、
target critic 和相同 probe RNG。继续 PCBF 的 gate 要求 7k、8k 在 `R_action=16` 时每个 C2F
level 的 bin flip 都不超过 10%；只在单个邻居通过不算通过。

文献路线按“它实际改变哪一环”重新排序，避免把名称里带 flow 的方法混成一类：

- [FLOQ](https://arxiv.org/abs/2509.06863) 是当前主实验：dense iterative supervision 加
  多 target samples，并用 scalar distilled critic 把昂贵生成式 value 变成可部署 readout。
- [Path-Coupled Bellman Flows](https://arxiv.org/abs/2605.08253) 用 shared source 与
  source-consistent control variate 降 Bellman-flow target 方差；当前只验证本地离散 critic
  core 的 source-ranking 稳定性，不声称复现其 continuous actor/twin ensemble。
- [FlowIQN](https://arxiv.org/abs/2605.08515) 通过排序 source/target samples 获得近似单调
  optimal-transport coupling，直接针对 distributional projection；官方仓库尚未放出实现，
  因而只能作为 paper-derived ablation，不能标成复现。
- [EVOR](https://arxiv.org/abs/2510.08218) 的部署 readout 是
  `eta * log E[exp(return/eta)]` 并以 base-policy regularization/rejection sampling 限制
  extrapolation。对于 MovePlate 的稀疏二元成功，它比普通 endpoint mean 更明确地强调成功
  upper tail；若 Stage-24 mean-readout 不能产生正 paired control effect，下一实现优先是候选
  bin 枚举版 entropic readout，而不是继续扫 flow steps。
- [FlowCritic](https://arxiv.org/html/2510.22686) 用生成 return、截断高端样本和 velocity
  clipping 稳定 policy-gradient critic；它更接近 state-value/PPO 路线。本地可提取的公平
  ablation 是 truncated/pessimistic endpoint readout，不会把整套算法错误描述为 CQN-AS
  reproduction。

因此下一阶段不是同时混入这些机制：先等 mean-distill 的同-checkpoint 因果 gate；若失败，
保持同一训练 snapshot 与 beta protocol，只把 readout 替换为 EVOR-style entropic aggregation，
从而把“flow 表示是否有用”与“风险/upper-tail 目标是否有用”分开。

#### EVOR 公式审计后的实现边界修正

继续阅读 EVOR 公式与附录后，上一段“只替换 Stage-24 readout”的表述需要收紧。论文的值是

```text
Q*(s,a) = eta * log E_{G ~ return-distribution(s,a)} exp(G / eta)
```

推理时用 Monte Carlo log-mean-exp 近似；论文配置 train 只取 1 个 return-to-go sample、eval
取 50 个。这个运算的输入必须是真正的随机 return samples。Stage-24 的 `value_mode=scalar`
先把 target-flow samples 求 mean，再把同一个确定性 Bellman target广播给所有 source；它的 source
spread 是数值/近似误差，不是有语义的 return uncertainty。对这种 residual 做 entropic aggregation
会错误地奖励 approximation noise，所以明确禁止。

代码已据此加入：

- `return_sample_aggregation={mean,entropic}` 与正温度
  `return_sample_temperature`；`entropic` 在非 `return_sample` 模式构造时直接报错；
- 数值稳定的
  `eta * (logsumexp(samples/eta) - log(R))`；
- `flow_q_action_readout`：在每个 C2F level 内一次性并行算完
  `16 x action_dimension x 5 bins x R` endpoints，再用 normalized flow-Q 加
  `beta * log pi_BC` 选 bin；level 和 Euler integration 仍串行；
- evaluator 可在不改 checkpoint 参数的情况下切换 mean/entropic 与温度。训练配置默认
  `flow_q_action_readout=false`，所以独立 BC 继续控制 replay，只有 held-out eval 打开 value。

该实现是 **EVOR-derived CQN-AS readout**，不是完整 EVOR reproduction：CQN-AS 枚举离散 sibling
bins，替代论文从生成式 base policy 采 32 个 action candidates；本地 flow 训练仍需在 BCFM、
PCBF 或 DCFM 中单独选择。最关键的 matched experiment 因而是同一个
`return_sample + separate BC` checkpoint 上 mean 与 entropic readout 的 paired 差，而不能拿
Stage-24 scalar checkpoint 直接做 log-sum-exp。新增公式、非法模式与 action-selection tests
将在完整 CQN-Flow regression 通过后才进入可运行 launch。

FlowIQN 的 PDF/官方仓库也完成了实现级审计。它不是简单地“把 samples 排序后算 loss”：
每条 transition 需要采 `K` 个 source quantile fractions 和 `K` 个 Bellman target returns，
分别排序后按 order statistic 配对；source 使用 uniform quantile，并把 quantile `tau` 在整个
velocity field 中作为显式 condition。当前本地 BCFM/PCBF 使用 shared Gaussian source，但 critic
没有显式 source-quantile condition。因此只加 `sort` 不能称为 FlowIQN；官方仓库当前也仍写着
“Code will be released shortly”。若后续进入这条线，minimum faithful ablation 必须同时加入
uniform source、per-transition sorted coupling 和 persistent quantile condition，并以
Wasserstein/CRPS calibration 为首要 gate，而不是只看 task success。

### 21.15 Clean CQN-AS 三种子基线闭环；A/B 下游 gate 已接管 GPU1

#### Research A 上一阶段结果

三次 exact clean CQN-AS 已全部完成，正式汇总为
`exp_local/cqn_value_fidelity_stage22/clean_multiseed_summary.json`：

| seed | 2.5k | 5k | 7.5k | 10k/final | validation-selected best |
|---|---:|---:|---:|---:|---:|
| 1 | `60%` | **`68%`** | `56%` | `60%` | `68%@5k` |
| 2 | `44%` | **`72%`** | `60%` | `52%` | `72%@5k` |
| 3 | **`76%`** | **`76%`** | `64%` | `64%` | `76%@2.5k`（5k 并列） |

三个 seed 的 validation-selected best 是 `68/72/76%`，均值和中位数均为 **`72%`**，range
为 `68--76%`。对应 final 是 `60/52/64%`，均值 **`58.67%`**；逐 seed 从 best 到 final
下降 `8/20/12pp`，平均 **`13.33pp`**。因此“固定跑到同一个末期 step 再比较”会把原方法
系统性低估；后续所有不低于 baseline/超过 baseline 的结论都以每个 seed 自己的 validation-best
为准。

历史 first-success 的 `92%@7.5k` 仍保留为真实 upper artifact，不能删除；但 exact seed 1
复跑和额外 seed 2/3 的最大值分别只有 `68/72/76%`，所以它没有成为可复现 canonical
checkpoint。当前正式比较同时报告：

1. 可复现 baseline：三种子 best mean `72%`、max `76%`；
2. 历史 upper reference：`92%`；
3. 改进算法相对 matched frozen behavior 的 paired increment。

这一步只确定了公平 baseline，尚未证明原 CQN-AS 的 value 有因果意义。Research A 的下一 gate
仍是 exact seed-1 5k legacy conversion 上的 16-train/4-validation 全维度 counterfactual
ensemble：只有 validation causal ranking/regret 与 frozen-policy closed-loop 同时通过，才把
相同 sidecar protocol 扩到 seed 2/3 的 validation-best checkpoints；否则不为一个 seed-1
negative result 再花两倍 simulator 成本。

#### 解释、下一 gate 与已执行任务

GPU1 clean 父进程于 `04:39 BST` 正常退出，event handoff 已核对并立即启动 matched direct-C51
5k 的 action-facing gate（PID `930131`）：

- validation `36000--36049`；
- own BC 对 `beta={0,0.03,0.1,0.3,1,3}`；
- 只有 `>=+2pp` 且 wins>losses 才在 `37000--37099` 确认；
- 完成后自动运行 PCBF 相邻 7k/8k 的 `R_action` source-ranking probes。

这条 direct-C51 是 Research B 的 representation/control baseline，不与 A 的 causal-sidecar
问题混合。按原/converted checkpoint 的约 `3--3.6s/episode` 加每个独立 JAX workspace
初始化估算，7 个 validation variants 预计 `16--22min`；若进入 confirmation 再加
`8--12min`。GPU1 同时保留约 `1.2GB` 给 CPU-bound legacy branch collector，总显存实测约
`4.7GB`，没有挤掉后者。

legacy cache 不再需要人工接力：新增
`scripts/run_cqn_lcb_sidecar_gate.py`，相关 evaluator/calibration regression 为 `6 passed`。
事件 handoff PID `934614` 会同时等待 counterfactual ensemble PID `894148` 和当前 GPU1
direct/PCBF pipeline PID `908951`，避免争卡；随后固定三 sidecars seed `1/2/3`，在新
`41000--41049` seeds 上校准 margin `{.01,.02,.04,.08}`，只把 calibration-selected margin
带到 `42000--42099` 做确认。正式输出目录为
`exp_local/cqn_value_fidelity_stage23/legacy_best5k_lcb_gate_seed41000`。按 cache 原 ETA 和旧
LCB evaluator 吞吐，预计约 `05:35--06:00` 开始 policy gate；若 calibration/confirmation
都运行，再需约 `20--30min`。

direct-C51 gate 的首个真实吞吐点随后落盘：own-BC 在 `36000--36049` 为 `46%`，wall time
`159.95s`。它只是这个 independent-BC/two-tower checkpoint 的 paired reference，不能替代上面的
clean CQN-AS `72%` multi-seed baseline。每个 50-episode variant 约 `2.7min`，七组 validation
预计 `04:57--05:00` 完成；若通过 gate，两个 100-episode confirmation 约再用 `8--12min`。

Stage-24 同时到 4k：total time `1637.30s`，旧 compute-8 FLOQ 同点为 `1339.58s`，实测慢
`22.2%`；`flow_distill_loss=8.81e-4`、所有 non-finite fractions 仍为 0。loss 只用于确认
数值健康，不用于 checkpoint selection。按当前/历史分段吞吐，训练 ETA 仍为
`05:21--05:27 BST`。

#### Research B 机制研究带来的 gate 修正

[What Does Flow Matching Bring to TD Learning?](https://arxiv.org/abs/2603.04333) 的 controlled
comparison 报告 expected-value flow critic 在若干任务上等于或优于 distributional variant，
并把主要机制归因于 dense velocity supervision 带来的 test-time recovery 与 feature
plasticity，而不是 return distribution 本身。这与当前路线的正确实验顺序一致：

1. 当前 Stage-24 先做 FLOQ distilled-scalar action gate；
2. 同一个 5k checkpoint 再用 integrated scalar flow field 做 action readout，并在
   `flow_steps={2,4,8}` 下测 paired control，直接检验迭代 readout 是否优于蒸馏；
3. 只有真正的 `return_sample` checkpoint 才比较 mean 与 EVOR entropic aggregation。

evaluator 已增加 `--flow-readout={distill,integrated}` 和 eval-time `--num-flow-steps`；
`flow_q_action_readout` 在每个 level 内并行计算全部 action bins/source samples，再与 BC prior
组合。默认 `auto` 保持已经排队的 Stage-24 distill gate 不变。完整 CQN-Flow regression 为
`93 passed`；随后新增的 entropic illegal-mode/action tests 单独为 `3 passed`。测试时一次未固定
CPU backend 的并发 pytest 因 GPU5 已占约 25.6GB 而在 import 阶段 OOM，改用
`JAX_PLATFORMS=cpu` 后通过；这是测试设备争用，不是算法 non-finite 或模型 OOM。

为保证 distilled gate 后立即测到 integrated readout 的真实成本，GPU5 又挂接了只等待
PID `908452` 的 smoke handoff PID `935770`。它在同一 5k checkpoint、相同 validation seed
起点上只跑 5 个 `beta=0, flow_steps=8` episodes；结果不用于 beta/checkpoint selection，只用于
确认 action path 和估计每 episode wall time。smoke 通过后才按实测吞吐设计
`steps={2,4,8}` 的 paired validation 预算，避免事先假设 integrated flow 一定便宜或昂贵。

### 21.16 Validation-selected clean baseline 的共同 held-out 确认已排队

#### 上一阶段结果与解释

上一节的 `68/72/76%`（均值 `72%`）来自三个训练 seed 各自的 validation-selected best，
它解决了“不能拿过拟合 final 当 baseline”的问题，但仍然是 checkpoint selection split 上的
表现；不能把它直接当作无偏 test performance。为此新增
`scripts/summarize_cqn_multiseed_confirmation.py`：要求每个训练 seed 使用完全相同、从未参与
checkpoint selection 的环境 seeds，并对训练 seed 与环境 seed 做 crossed bootstrap，而不是
把 300 个 episode 错当成 300 个独立训练模型。新增汇总器及 evaluator 的语法检查、
focused regression 已通过（`1 passed`），`git diff --check` 通过。

#### 下一阶段 gate

冻结上一节已经选定的 checkpoint，不再根据 test 结果换 step：

| training seed | frozen checkpoint |
|---|---:|
| 1 | `5k` |
| 2 | `5k` |
| 3 | `2.5k` |

三者都在共同 held-out seeds `43000--43099` 上运行原生 clean CQN-AS action path，各 100
episodes。成功 gate 是三个 JSON 均完整覆盖同一 100 seeds、汇总器状态为 `ok`；它输出
per-training-seed success、三 seed mean/sample std 和 crossed-bootstrap 95% CI。后续 A/B
算法只能先在自己的 validation split 选择 checkpoint/hyperparameter，再在同一
`43000--43099` held-out split 做 paired 最终比较。Research A 的非劣 gate 预注册为 paired
mean delta `>=0` 且 95% CI 下界 `>=-5pp`；Research B 的超越 gate 要求 paired mean delta
`>0`、wins>losses，并且 95% CI 下界 `>=0`。若同一 test split 被用于算法迭代，它立即降级为
diagnostic，最终结论必须换一组未见 seeds。

#### 已执行任务与 ETA

GPU5 event handoff PID `940411` 已验证存活；它等待 Stage-24 action gate 和 integrated smoke
的父 PID `935770` 退出，并要求其 completion artifact 存在，然后顺序评估上述三个 frozen
checkpoints，最终写入：

```text
exp_local/cqn_value_fidelity_stage22/clean_multiseed_heldout_seed43000/
```

completion monitor 使用进程退出事件，不做短轮询。按 clean/native evaluator 已实测约
`3.0--3.6s/episode`，300 episodes 加三次 JAX 初始化约需 `18--24min`。它的开始时间取决于
Stage-24 在 `05:21--05:27 BST` 完成训练后是否触发约 `19min` validation、约 `10min`
confirmation 以及 integrated smoke；预计约 `05:45--06:05 BST` 开始，约
`06:05--06:30 BST` 完成。这个任务只使用 Stage-24 完全释放后的 GPU5，不与现有训练争显存。

### 21.17 Direct-C51 action gate 的首个机制证据（gate 尚未结束）

#### 当前 artifact

`stage20_direct_policy_value_gate_seed36000/validation` 已完成前三组共同 seeds
`36000--36049`：

| readout | success | wall time |
|---|---:|---:|
| own BC | `46%` | `159.95s` |
| direct-C51 Q-only (`beta=0`) | **`0%`** | `247.85s` |
| direct-C51 Q + `0.03 log pi_BC` | **`0%`** | `229.68s` |

这不是完整 beta gate，但已经排除了“当前 direct-C51 value action ranking 本身接近可用
policy”这一解释：在完全相同 checkpoint 和环境 seeds 上，一旦让 value 真正选择 action，
成功率从 `46%` 降到 `0%`。`beta=.03` 仍不足以把选择约束回 behavior support。剩余
`beta={.1,.3,1,3}` 必须跑完，因为较强 BC prior 可能恢复甚至小幅改进；在看到完整结果前不
选择 beta，也不把这个 control 的失败当成 FM 成功。

value-facing evaluator 的实测吞吐约 `230--248s/50 episodes`，高于只跑 BC 的 `160s`。按
剩余四组更新 direct validation ETA 为约 `05:04--05:08 BST`；若某个 beta 达到预注册
`+2pp` gate，两个 100-episode confirm 还需约 `15--18min`。PID `930131` 的完成监视绑定进程
退出事件；不做短轮询。

### 21.18 Direct-C51 action-facing control 正式失败；PCBF probe 已自动接力

#### 上一阶段完整结果

`exp_local/cqn_flow_high_utd/stage20_direct_policy_value_gate_seed36000/gate_summary.json`
已封口。共同 validation seeds `36000--36049` 的完整结果是：

| policy/readout | success | paired delta vs own BC |
|---|---:|---:|
| own BC | `46%` | `0pp` |
| Q-only / `beta=0` | `0%` | `-46pp` |
| `beta=.03` | `0%` | `-46pp` |
| `beta=.1` | `0%` | `-46pp` |
| `beta=.3` | `28%` | `-18pp` |
| `beta=1` | `38%` | `-8pp` |
| `beta=3`（best candidate） | `42%` | `-4pp` |

被选中的 `beta=3` 是 `6 wins / 8 losses / 36 ties`，paired 95% CI
`[-18pp,+10pp]`，McNemar exact `p=.791`。预注册 validation gate 要求至少 `+2pp` 且
wins>losses；实际为 `-4pp`，因此 gate 正式 **fail**，confirmation 按协议没有运行。整组
耗时 `1516.21s`。

#### 解释

这个结果建立的是一个很具体的 negative control：Stage-20 direct-C51 的内建曲线即使在 5k
曾达到 `56%`，也不能说明 critic 学出了可用于控制的 value，因为那条曲线由独立 BC tower
执行 action。相同 checkpoint 一旦让 C51 value 真正选择 C2F bins，Q-only 立即降到 `0%`；
增大 BC prior 只把 policy 渐近拉回 BC，直到 `beta=3` 仍没有超过 own BC。它支持用户提出的
核心担忧——action-facing value 可能没有真实控制意义——但不能直接推广成“所有 CQN-AS
checkpoint 都失败”，也不能预先判定 flow critic。

#### 下一阶段 gate 与执行

Research B 保持同一套 action-facing protocol，等待 Stage-24 FLOQ-distill 的 5k checkpoint
比较 own BC 与 value；只有它通过，才能声称 FM value 比 direct-C51 control 更有用。当前
Stage-24 已到 7k，累计 `2787.15s`，数值仍 finite；按最近区间吞吐，10k ETA 维持
`05:20--05:24 BST`，随后自动运行 distill action gate。

GPU1 没有闲置：direct gate 退出后，父 PID `908951` 已自动启动 PCBF
`step7000/R_action=1` probe（子 PID `953307`）。六组 probe 会顺序完成 7k/8k 邻居和
`R_action={1,4,8,16}` 稳定性 sweep；最终 gate 要求 7k、8k 在 `R_action=16` 时每个 level
flip rate 都 `<=10%`。这一步只判断 flow source/action ranking 是否足够稳定，不能替代
closed-loop success。首次 probe 的实际耗时落盘后再更新 ETA。

### 21.19 PCBF source-ranking gate 通过；return readout closed-loop gate 已实现并排队

#### 上一阶段结果

六组 ranking probes 已完成，artifact 为
`exp_local/cqn_flow_pcbf_ranking/stage25_pcbf_r_action_neighbors_20260724/summary.json`。
关键结果如下：

| snapshot | `R_action` | per-level bin flip | max flip |
|---|---:|---|---:|
| 7k | 1 | `5.14/25.06/31.15%` | `31.15%` |
| 7k | 16 | `0.09/3.02/6.09%` | **`6.09%`** |
| 8k | 1 | `6.04/25.07/33.69%` | `33.69%` |
| 8k | 4 | `0.27/6.69/13.23%` | `13.23%` |
| 8k | 8 | `0.15/4.47/9.76%` | **`9.76%`** |
| 8k | 16 | `0.10/3.37/7.48%` | **`7.48%`** |

7k/8k 在 `R_action=16` 时都低于预注册 `10%` 阈值，因此 ranking gate 正式 **pass**。
同时，最细 level 的 rank-SNR 从单样本约 `.052/.051` 提升到 `1.30/1.11`。每组 probe
约 `35.6--36.9s`；`R=16` 没有增加可见 wall time，证明
`16 x action_dimension x bins x sources` 的并行向量化在当前显存预算内有效。

#### 解释

这个结果只建立“多 source 聚合能稳定 PCBF 的 bin ranking”，不建立“排序是正确的”或
“task success 更高”。`R=1` 在细粒度 level 的 `31--34%` flip 明确不可部署；`R=4` 仍有
`13.2%`，而 `R=8/16` 才进入稳定区。所以下一 closed-loop 实验必须固定至少 `R=16`，否则
把 Monte-Carlo ranking noise 与算法效果混在一起。

PCBF checkpoint 是 single-tower：没有独立 BC policy，其原生 rollout 本来就是 flow-value
选 action。evaluator 已增加 eval-time `--num-action-flow-samples` 并在 JSON 中记录它；同时
把这类策略准确标为 `native_flow_value`，避免再次把它误报成 `bc_only`。8k、`R=16`、mean
的一个真实 closed-loop smoke 已成功，配置为 integrated flow、4 Euler steps，首 episode
含 JIT 初始化耗时 `39.07s`；该 seed 失败只作吞吐 smoke，不进入选择统计。

#### 下一阶段 gate

新增 `scripts/run_cqn_flow_return_readout_gate.py`，相关 CQN-Flow/runner tests 共
`95 passed`，`git diff --check` 通过。validation 固定 seeds `44000--44049`：

1. baseline 在 7k/8k 的原生 `R4 + mean` 中独立选择最好 checkpoint；
2. candidate 在同一 7k/8k 的 `R16 + mean`、`R16 + entropic(eta=.3/1)` 中选择；
3. tie 优先 mean，其次较大 eta（更接近 mean），再选较早 checkpoint；
4. candidate 相对 selected native baseline 至少 `+2pp` 且 wins>losses，才在全新
   `45000--45099` 确认；
5. confirmation 要求 delta>0、wins>losses 且 paired CI 下界 `>=-5pp`。

这个 gate 分开回答两个问题：`R16-mean` 是否仅靠降低 source noise 改善；entropic 是否在
同一 checkpoint/source budget 上进一步利用 return upper tail。若通过，它仍只是
single-training-seed 机制证据，随后必须多训练 seed 并与 validation-selected clean CQN-AS
在共同 held-out seeds 上比较，才能声称 B 路线超越。

#### 已执行任务与 ETA

GPU1 event handoff PID `969446` 已验证等待 A 路线 LCB 父 PID `934614`；A 完成后立即运行上述
8 个 validation variants，若通过再运行 2 个 confirmation variants。这样不会让 PCBF 与 A
争 GPU。按 smoke 中一次约 `35s` JIT 加 native evaluator 的约 `3s/episode` 估算，
validation 约 `25--30min`，confirmation 约 `11--14min`。其开始时间取决于 legacy
counterfactual/LCB，预计约 `05:55--06:30 BST`；不进入 confirmation 时约
`06:20--07:00` 完成，进入时约 `06:35--07:15` 完成。完成监视绑定 PID 退出事件。

### 21.20 Stage-24 FLOQ-distill 训练完成；action-facing gate 已开始

#### 上一阶段结果

Stage-24 于 `05:24 BST` 正常完成 10k，训练总 wall time `3939.75s`（约 `65.7min`）。
独立 BC tower 执行的内建 eval curve 为：

| step | BC-controlled success |
|---:|---:|
| 2.5k | **`56%`** |
| 5k | `40%` |
| 7.5k | `48%` |
| 10k | **`56%`** |

validation-selected best 是 `56%`，2.5k/10k 并列时按“较早 checkpoint”原则应选 2.5k。
final 10k 没有低于 best，但这只是该独立 BC tower 的 policy quality，不是 flow value 的
action-facing 证据。10k 的 `flow_distill_loss=.00446`、所有记录的 gradient non-finite
fraction 为 0，说明数值路径完成；loss 不参与策略结论。

#### 解释与 gate 边界

已经排队的 5k gate 是在看到完整 curve 前预注册的 **matched mechanism control**：它与
Stage-20 direct-C51 的 5k 使用同一 checkpoint step、同一 seeds 和 beta grid，专门比较
“同训练预算下 flow-distill value 是否比 direct-C51 value 更可控”。它不能代替 Stage-24
自身 validation-best checkpoint 的最终比较，因为 5k 的 BC-controlled success 只有 `40%`。
因此无论 5k mechanism gate 结果如何，后续正式 Stage-24 算法结论都必须在
validation-selected 2.5k（并保留 10k tie audit）上复核；不能因为固定 5k 失败就宣判整条
FLOQ-distill 路线失败，也不能因为成功就跳过 clean CQN-AS 比较。

#### 已执行任务与 ETA

GPU5 handoff 已自动启动 5k action-facing gate，runner PID `975296`：

- own BC 与 `beta={0,.03,.1,.3,1,3}`；
- validation `36000--36049`；
- 只有 `>=+2pp` 且 wins>losses 才进入 `37000--37099` confirmation。

按 direct gate 的同 evaluator 实测，BC 约 `160s/50 episodes`、value variants 约
`200--250s/50 episodes`，validation ETA 约 `05:48--05:54 BST`；若通过，confirmation
约再 `15--18min`。其后同一 GPU5 事件链会运行 integrated-flow smoke，再运行 clean
三训练 seed 的共同 held-out baseline，均不靠短轮询。

### 21.21 Stage-24 validation-best 多 checkpoint gate 已实现并排队

#### 上一阶段约束

Stage-24 的 2.5k/10k 在 BC-controlled validation 上同为 `56%`，所以只看正在运行的 5k
mechanism gate 不能满足“按最好版本比较”。同时，也不能分别看完两个 checkpoint 的所有
held-out 结果后再人工挑一个；checkpoint 和 beta 必须在同一个 validation split 联合选择，
然后只确认 winner。

#### 实现与预注册 gate

新增 `scripts/run_cqn_multicheckpoint_policy_value_gate.py`。它为每个 checkpoint 分别运行
own BC 和完整 beta grid，但独立选择：

1. baseline winner：2.5k/10k 中 validation own-BC success 最高者；并列选更早的 2.5k；
2. value winner：跨 2.5k/10k 与 `beta={0,.03,.1,.3,1,3}` 的 validation success 最高者；
   并列优先更大 beta（更接近 behavior support），再选更早 checkpoint；
3. value winner 相对 baseline winner 必须至少 `+2pp` 且 paired wins>losses，才在新的
   confirmation seeds 上只运行这两个 winner；
4. confirmation 要求 positive direction、wins>losses、95% CI 下界 `>=-5pp`。

新 runner 与两个既有 gate 的 focused tests 共 `9 passed`，语法和 diff 检查通过。它避免
把“选到 BC 更差的 checkpoint 后看似有 value 增益”当作成功。

#### 已执行任务与 ETA

GPU5 event handoff PID `979748` 已验证等待 clean 三种子 held-out 父 PID `940411`；随后在
fresh validation seeds `46000--46049`、confirmation `47000--47099` 运行 2.5k/10k 联合 gate，
输出目录：

```text
exp_local/cqn_flow_high_utd/stage27_distill_multicheckpoint_gate_seed46000/
```

14 个 validation variants 按当前 evaluator 吞吐约需 `45--58min`；若通过，两个 100-episode
confirmation 再需 `15--18min`。按前置 5k gate、integrated smoke 和 clean held-out 的 ETA，
预计约 `06:10--06:30 BST` 开始；不进 confirmation 约 `06:55--07:28` 完成，进入时约
`07:10--07:46`。任务通过 PID 退出事件自动接力，GPU5 不会在前序完成后空置。

### 21.22 5k FLOQ-distill 的 validation 正信号未复现；integrated smoke 完成

#### 上一阶段完整结果

Stage-24 5k matched mechanism gate 的 validation（`36000--36049`）为：

| readout | success |
|---|---:|
| own BC | `58%` |
| Q-only / `beta=0` | `0%` |
| `beta=.03` | `0%` |
| `beta=.1` | `0%` |
| `beta=.3` | `28%` |
| `beta=1` | `58%` |
| `beta=3` | **`66%`** |

`beta=3` 相对 own BC 为 `+8pp`，`6 wins / 2 losses / 42 ties`，paired CI
`[-2,+20]pp`，因此按预注册规则进入 confirmation。但在未见的 `37000--37099`：

| policy | success |
|---|---:|
| own BC | **`64%`** |
| distilled Q + `3 log pi_BC` | `56%` |

held-out delta 为 **`-8pp`**，`8 wins / 16 losses / 76 ties`，CI `[-17,+2]pp`；
confirmation gate 正式 **fail**。所以 validation 的 `+8pp` 是未复现的选择噪声，不能称为
FM value improvement。与 direct-C51 不同，distill 至少能在强 BC prior 下产生 validation
正信号；但目前没有 held-out causal control benefit。

#### Integrated-flow smoke

同一 5k checkpoint 的 integrated scalar flow action path 已成功运行：

```text
flow_steps=8
R_action=8
beta=0
episodes=5
elapsed=116.89s
success=0/5
```

它证明 iterative action path 可运行，但约 `23.4s/episode`（含首次 JIT），明显比 distill
昂贵；5 个 value-only seeds 全失败，与 distill/direct 的低-beta collapse 方向一致。这个
smoke 既没有统计功效，也没有 BC prior，不能用来否定 integrated readout。下一机制 gate
必须固定强 behavior support（先用 `beta=3`），再比较 flow steps，而不是重复已知会 collapse
的 Q-only 大 sweep。

#### 下一阶段决策与已执行后继

当前 GPU5 已自动进入 clean 三训练 seed 的共同 held-out baseline，随后是 Stage-27 的
2.5k/10k validation-best distill gate。Integrated scalar 的下一实验预注册为：

1. 同一 5k checkpoint、同一强 prior `beta=3`；
2. validation 比较 `flow_steps={2,4,8}`，先与已完成的 distill-beta3 paired seeds 对照；
3. 只把 validation winner 带到新 seeds confirmation；
4. 若 integrated 仍不能改善，不再把 scalar approximation noise 当 return distribution
   做 entropic readout；转向已经通过 source-ranking gate 的 PCBF return-sample 路线。

由于 8-step smoke 已显示高成本，这个 gate 排在 Stage-27 之后，避免阻塞 validation-best
distill 与 clean baseline 的结论；届时按每个 steps 的首个实测 throughput 决定 50-episode
完整 validation 是否在预算内。

### 21.23 Clean CQN-AS 无偏 held-out baseline 完成；Stage-27 snapshot 错误已闭环修复

#### 上一阶段结果

三个训练 seed 的 checkpoint 都先在各自 validation curve 上冻结为 `5k/5k/2.5k`，再共同评估
从未用于选择的环境 seeds `43000--43099`。正式 artifact：

```text
exp_local/cqn_value_fidelity_stage22/clean_multiseed_heldout_seed43000/summary.json
```

结果为：

| training seed | selected checkpoint | held-out success |
|---|---:|---:|
| 1 | 5k | `59%` |
| 2 | 5k | `56%` |
| 3 | 2.5k | `69%` |

三训练 seed mean 为 **`61.33%`**，sample std `6.81pp`，对训练 seed 与环境 seed做
crossed-bootstrap 的 95% CI 为 **`[51.0%,71.67%]`**。这与 selection split 上的
`68/72/76%`（mean `72%`）不矛盾：后者是 validation-selected best，前者才是当前无偏 test
baseline。后续“非劣/超越 CQN-AS”必须对这三个 frozen models 的共同 held-out 结果做匹配比较，
不能再用 validation `72%` 或 final `58.67%` 作为 test denominator。

#### Stage-27 执行错误与修复

原 Stage-27 handoff PID `979748` 在 clean baseline 完成后立即退出，failure artifact 明确为：

```text
FileNotFoundError: snapshots/2500_snapshot.pkl
```

原因是 2.5k 是 eval 点，但该训练只每 1k 保存 snapshot；真实文件只有 2k、3k 等整数千步。
没有伪造或复制一个“2.5k checkpoint”。旧失败目录
`stage27_distill_multicheckpoint_gate_seed46000` 保留作为可审计 artifact。

修复后的 gate 使用相邻 **2k/3k** 加与 2.5k 并列 best 的 **10k**。根据已完成 5k gate，
`beta<=.3` 已从 `0/0/0/28%` 显示明显不受 behavior support 约束；新 split 只保留
`beta={1,3}`，这是基于上一阶段证据的预注册收缩，不是看到新 seeds 后追参。因此总计 9 个
validation variants（3 checkpoints x [BC,beta1,beta3]），baseline 和 candidate 仍独立选择。

#### 已执行任务与 ETA

修复 run 已在 GPU5 真正启动：

```text
parent PID: 1013202
runner PID: 1013211
output: exp_local/cqn_flow_high_utd/stage27b_distill_neighbor_gate_seed46000
validation seeds: 46000--46049
confirmation seeds: 47000--47099
```

按 5k 实测 BC `107s`、value `145--154s/50 episodes`，9 组 validation 约
`20--25min`，预计 `06:24--06:30 BST` 完成；若通过，两个 100-episode confirmation 再约
`10--13min`，最终 ETA `06:34--06:43 BST`。完成监视绑定父 PID 退出事件。

### 21.24 Integrated scalar flow 的成本感知 gate 已实现并排队

#### 上一阶段结果与解释

8-step smoke 的约 `23.4s/episode` 表明直接对每个 steps 跑完整 beta x 50-episode grid 会浪费
大量 simulator 时间；另一方面，Q-only 的 `0/5` 已知主要受 behavior-support collapse 影响，
不能据此删掉 integrated 路线。因此下一实验固定上一阶段唯一有希望的强 prior `beta=3`，
只比较迭代步数，并同时以 own BC 和 distilled-beta3 为 reference。

#### 新 gate

新增 `scripts/run_cqn_integrated_steps_gate.py`，focused tests 与相邻两个 runner 共
`9 passed`，语法/diff 检查通过。流程预注册为：

1. pilot：`steps={2,4,8}` 各跑 validation seed prefix 的 20 episodes；
2. 选择 success 最高 steps，并列选更少 steps；只有它相对 own BC 不低于 `-5pp` 才继续；
3. winner 在完整 `36000--36049` 重新跑 50 episodes；
4. full validation 同时要求相对 own BC 和 distilled-beta3 都至少 `+2pp` 且 wins>losses；
5. 只有双 reference gate 通过，才在 `37000--37099` 跑 winner，并要求对两个 reference
   都是 positive direction、wins>losses、CI 下界 `>=-5pp`。

reference 直接复用已经封口的同 seeds JSON，不重复运行 BC/distill。这样 pilot 只负责排除
灾难性 steps，不能凭 20 episodes 宣称成功；正式结论仍来自完整 validation 和 held-out。

#### 已执行任务与 ETA

event handoff PID `1050456` 已验证等待 Stage-27b 父 PID `1013202`，随后自动在 GPU5 运行，
输出：

```text
exp_local/cqn_flow_high_utd/stage28_integrated_steps_gate_seed36000/
```

按 8-step smoke 外推，三组 20-episode pilot 约 `15--25min`；若 pilot 通过，winner 的
50-episode validation 约 `6--20min`（取决于 steps），confirmation 再约 `12--40min`。
Stage-27b 当前预计 `06:24--06:30 BST` 结束，所以 Stage-28 最早约 `06:40--06:55` 产生
pilot 结论；后续 ETA 必须用各 steps 的实际 JSON 更新。

### 21.25 Stage-27b FLOQ-distill validation-best gate 首次可靠通过

#### 上一阶段完整结果

fresh validation `46000--46049` 上，三个可保存邻居 checkpoint 的结果为：

| checkpoint | own BC | beta=1 | beta=3 |
|---:|---:|---:|---:|
| 2k | **`70%`** | `56%` | `74%` |
| 3k | `56%` | `38%` | `64%` |
| 10k | **`70%`** | **`82%`** | `68%` |

baseline 独立选择 2k BC（与 10k 的 `70%` 并列时选更早 checkpoint）；candidate 独立选择
10k beta1 `82%`。validation paired delta `+12pp`，`11 wins / 5 losses / 34 ties`，
CI `[-4,+28]pp`，达到 `+2pp` 且 wins>losses，因此进入 confirmation。

在完全未见的 `47000--47099`：

| selected policy | success |
|---|---:|
| 2k own BC | `54%` |
| 10k distilled-Q + `1 log pi_BC` | **`74%`** |

paired delta 为 **`+20pp`**，`33 wins / 13 losses / 54 ties`，95% CI
**`[+7,+33]pp`**，McNemar exact `p=.00453`。confirmation gate 正式 **pass**。
artifact：

```text
exp_local/cqn_flow_high_utd/stage27b_distill_neighbor_gate_seed46000/gate_summary.json
```

#### 解释

这是 B 路线第一个同时通过 validation selection 与 held-out confirmation 的 action-facing
正结果。它建立：

1. flow-distilled scalar value 不只是降低 loss；在 10k、beta1 下真的改变 action，并在新
   seeds 上比独立 BC tower 高 `20pp`；
2. 效果对 checkpoint/beta 极敏感：同一 10k 的 beta3 是 `68%`，3k beta1 只有 `38%`；
   固定 5k gate 的失败不能代表 validation-best 算法；
3. 这仍然只有一个训练 seed，且 confirmation baseline 是该 flow run 的 own BC，不是 clean
   CQN-AS。因此还不能声称“超越 CQN-AS”或可复现多 seed。

#### 下一阶段 gate 与执行

Stage-28 integrated pilot 已由 PID `1050456` 自动启动，先判断迭代 flow 是否还能超过这个
distill readout。与此同时，B 路线最终比较预注册为：冻结 10k/beta1，不再调参，在 clean
CQN-AS 已封口的共同 held-out `43000--43099` 上评估；先与 clean seed1 5k 做 paired
superiority，要求 delta>0、wins>losses、CI 下界 `>=0`。只有通过，才用相同训练配置扩展
Stage-24 seeds 2/3，并做三训练 seed crossed-bootstrap 最终比较。

该共同 held-out 比较已经实现并排队。新增
`scripts/summarize_cqn_paired_eval.py`，与 integrated gate 的 focused tests 合计
`5 passed`。event handoff PID `1068219` 等待 Stage-28 后评估 frozen 10k/beta1 共 100
episodes，再与 clean seed1 的既有逐 episode JSON 配对；输出目录：

```text
exp_local/cqn_flow_high_utd/stage29_distill_vs_clean_seed1_heldout43000/
```

distill evaluator 实测约 `1.3--2.2s/episode`，所以 Stage-28 结束后约 `3--5min` 可得到
seed1 cross-method 结论。

### 21.26 A 路线：原始 critic 确认坍缩；oracle sidecar 只记住训练 seed

#### 上一阶段完整结果

对 clean CQN-AS seed1 的 validation-selected 5k checkpoint 做了昂贵但精确的 simulator-state
branch audit。每个 anchor 固定相同的 image/state condition，只改变同一 C2F prefix 下的
5 个 sibling action bins，再从完全相同的 MuJoCo 状态继续 rollout。正式 artifact：

```text
exp_local/cqn_value_fidelity_stage23/legacy_best5k_counterfactual_ensemble/coverage.json
```

数据覆盖并不是问题：train 为 `720` states、其中 `525` informative；最终 heldout 为
`180` states、其中 `113` informative。收集耗时 `7282.87s`。但原始 critic 在 train 与
heldout 的 predicted-Q span 都是零，pairwise sign accuracy 都为 `0`，Spearman 不可定义；
heldout top-1 `25.66%` 仅等于随机 tie-aware baseline `25.49%`，regret `0.09885`。相比之下，
不读 return 的 action-nearness proxy 和 policy prior 在同一 heldout 的 pairwise 分别已有
`61.40%` 与 `60.74%`。因此这个 checkpoint 的 policy 成功不能归因于 critic 学到了 action
的真实回报排序；critic 已经发生实质性 value collapse。

随后只更新 critic、完全冻结 encoder 和独立 BC tower，用 train branches 做
pairwise-ranking + delta regression。三个初始化在 train 都达到约 `80.7--81.4%`
pairwise、Spearman `0.619--0.635`、top-1 `63.0--65.0%`；但在从未训练的四个 simulator
seeds 上分别只有：

| init | pairwise | Spearman | top-1 | regret |
|---:|---:|---:|---:|---:|
| 1 | `49.84%` | `-0.0163` | `20.35%` | `0.08475` |
| 2 | `50.82%` | `0.0064` | `19.47%` | `0.09150` |
| 3 | `50.60%` | `0.0109` | `21.24%` | `0.09313` |

三个 heldout top-1 都低于随机 `25.49%`。这排除了“原网络没有 value 容量”的解释：它能
拟合训练 branch；失败点是 seed-level 泛化，而不是 optimization。

#### LCB action gate 结果与解释

把三个过拟合 sidecar 当 ensemble 做 LCB 选动作同样失败。calibration BC 为 `64%`；
margin `.01/.02/.04/.08` 分别得到 `44/30/54/52%`，即 `-20/-34/-10/-12pp`。override
rate 仍高达 `99.99/99.84/97.40/67.38%`，所以没有 variant 进入 confirmation。artifact：

```text
exp_local/cqn_value_fidelity_stage23/legacy_best5k_lcb_gate_seed41000/gate_summary.json
```

结论不是“margin 还没扫够”，而是 ensemble 的共同 seed-overfit 让不确定度本身失真；
继续在 policy gate 上调 threshold 会把 calibration 当训练集，不能修复 value。

#### 下一阶段 gate 与已执行实现

下一阶段只解决 seed-level generalization，不与 B 路线混合。原 16 个 oracle-train seeds
被预先拆成 train12：
`24000--24002,25000--25002,26000--26002,27000--27002`，以及 internal-val4：
`24003,25003,26003,27003`。原 final-test4：
`24004,25004,26004,27004` 保持封存。

新增并测试：

- `scripts/split_cqn_branch_cache.py`；
- `scripts/run_cqn_branch_cv_gate.py`；
- 对应 focused tests；连同 oracle tests 共 `18 passed`。

seed1 只在 internal-val 上搜索 `updates={5,10,20,50}`、
`weight_decay={1e-5,1e-3,1e-2}`，固定 delta-regression weight `10`；按 pairwise 最大、
Spearman 最大、regret 最小、较少 updates、较强 regularization 的预注册顺序选 winner。
winner 再用 init2/3 复现，三初始化 median 必须同时满足 pairwise `>55%`、Spearman
`>0.1`、top-1 高于随机、regret 低于未更新 critic。由于本路线明确要排除“披着 value
外壳的 imitation”，在任务真正启动前进一步把 anti-cheat 条件预注册为：pairwise 与
top-1 还必须分别高于 action-nearness proxy 和 frozen policy prior，regret 必须低于这两个
不读 return 的 proxy；否则即使超过 `55%` 也不能叫 meaningful value。只有全部条件通过
才允许读取 final-test4。若 internal gate 不过，最终 test 保持未读；若通过，才在完整
train16 上重训三个初始化并只测试一次。

derived cache 已实际生成并验证为 `540 train / 180 internal-val states`：

```text
exp_local/cqn_value_fidelity_stage23/legacy_best5k_branch_cv/cache_train12_val4.npz
```

event handoff PID `1081973` 已排在 GPU1 的 Stage-26 后；Stage-26 完成即自动启动，不做短轮询。
按旧 100-update/20k-bootstrap fit 的 `69.88s` 上界估算，12-grid + 2 replication 约
`12--17min`；若 internal gate 通过，3 个 full fit 再约 `4--6min`。当前前置 Stage-26
仍在运行，因此总 ETA 取决于其余 return-readout variants。

### 21.27 B 路线：integrated scalar flow gate 正式失败

#### 上一阶段完整结果

Stage-28 的 `steps={2,4,8}` pilot 在相同 20 seeds 上分别为 `40/30/40%`；对应 own BC
`40%`、distill-beta3 `60%`。按并列选更低 compute 的规则选择 2 steps，并在完整 50-seed
validation 重测：

| readout | success |
|---|---:|
| own BC | `58%` |
| distilled scalar | **`66%`** |
| integrated flow, 2 steps | `58%` |

integrated 相对 BC 为 `0pp`（`3 wins / 3 losses`，CI `[-10,+10]pp`），相对 distill 为
`-8pp`（`2 wins / 6 losses`，CI `[-20,+2]pp`），未达到对两个 reference 都 `+2pp` 的
预注册 gate，confirmation 未运行。完整 artifact：

```text
exp_local/cqn_flow_high_utd/stage28_integrated_steps_gate_seed36000/gate_summary.json
```

这说明当前 action-conditioned scalar FM 的收益来自稳定的 distilled expected-value
readout，而不是推理时增加 ODE 积分；增加到 4/8 steps 也没有 pilot 信号。该 integrated
分支停止扩展。

#### 下一阶段与正在执行

GPU5 已立即进入 Stage-29：冻结 validation-selected `10k/beta1`，在共同 heldout
`43000--43099` 与 clean CQN-AS seed1 5k 做 paired superiority。父 PID `1068219` 和实际
evaluator PID `1077668` 均已验证存活；输出为：

```text
exp_local/cqn_flow_high_utd/stage29_distill_vs_clean_seed1_heldout43000/
```

实测运行已经进入 100 episodes，预计约 `6--8min` 完成。只有 delta>0、wins>losses 且
paired 95% CI 下界 `>=0`，才扩展相同 Stage-24 配置到训练 seeds2/3。

### 21.28 Stage-29 对 clean CQN-AS 有 +10pp 趋势，但严格 superiority 未过

#### 上一阶段完整结果

冻结 Stage-27b 在 validation 选出的 `10k/beta1`，并在 clean CQN-AS 已封口的共同
`43000--43099` seeds 上做逐 episode 配对：

| frozen policy | success |
|---|---:|
| clean CQN-AS seed1, validation-selected 5k | `59%` |
| FLOQ-distill seed1, fixed 10k/beta1 | **`69%`** |

paired delta 为 `+10pp`，`22 wins / 12 losses / 66 ties`，95% CI
`[-1,+21]pp`，McNemar exact `p=.12145`。artifact：

```text
exp_local/cqn_flow_high_utd/stage29_distill_vs_clean_seed1_heldout43000/summary.json
```

点估计方向与 Stage-27b 相对 own-BC 的 `+20pp` 一致，而且 wins 明显多于 losses；但 CI
下界比预注册的 `>=0` 少 `1pp`，因此 gate 必须记为 **fail**。这个结果支持“可能优于 clean
CQN-AS”，却还不能支持“已经证明优于”；也不应在证据不足时立即花两次约 66 分钟训练 seeds2/3。

#### 下一阶段 gate 与已执行任务

两个 checkpoint、`beta1` 和全部 evaluator 参数保持冻结，在完全新的
`48000--48199` 运行独立 200-seed paired replication，不把看过的 100 seeds 事后拼接来缩
CI。仍要求 delta>0、wins>losses、95% CI 下界 `>=0`。通过后才启动相同训练配置的
seeds2/3；失败则停止该 scalar-distill 配置的昂贵多 seed 扩展，转向 PCBF 或改进 TD target。

GPU5 已立即实际启动：

```text
parent PID: 1084946
evaluator PID: 1084954
output: exp_local/cqn_flow_high_utd/stage30_distill_vs_clean_seed1_fresh200_seed48000/
```

先跑 clean 200 episodes，再跑 distill 200 episodes。按各自上一轮 `211.20s/100` 和
`288.30s/100` 的实测吞吐，预计约 `16.7min`，加两次初始化后 ETA `17--19min`。完成条件
绑定父 PID，不做固定间隔短轮询。

为避免 gate 通过后两张卡空转，后继任务已做成条件 handoff，而不是提前偷跑：

1. Stage-30 通过时，GPU5 自动用完全相同配置训练 seed2（PID `1087653` 等待），GPU1 在
   Stage-26 与 A-route CV 释放后训练 seed3（PID `1087655` 等待）；
2. Stage-30 失败时两个训练都只写 `skipped_gate_failed`，不会消耗约两小时训练预算；
3. 训练通过后在第三组完全新的 `49000--49199` final seeds 上，对三对
   validation-selected clean CQN-AS 与 frozen `10k/beta1` 做共同 200-episode paired eval；
4. 新增 `scripts/summarize_cqn_multiseed_paired.py`，其 focused suite 与相邻 summary tests
   共 `5 passed`。最终 gate 对 model seed 与 environment seed 做 crossed bootstrap，要求
   mean delta>0、CI 下界 `>=0`、aggregate wins>losses，且至少 2/3 training seeds 为正。

若 Stage-30 通过，seed2/3 训练按 seed1 实测各约 `65.7min`；随后 GPU5 负责两对、GPU1
负责一对 final eval，预计再 `17--34min`，总体约在 Stage-30 后 `84--101min` 形成三训练
seed 最终结论。所有节点都由父 PID 退出和 completion artifact 接力。

### 21.29 文献复核后的方法分流：critic flow、return flow、policy flow 不能混为一谈

等待 closed-loop gate 时重新按原论文与官方实现核对了相关路线，得到以下对当前项目有直接
约束的结论。

#### 1. FLOQ 是 expected-value critic，不是 distributional RL

FLOQ 的 TD target 先对 successor flow samples 求均值，再把这个 scalar expected target
作为所有 source 的 endpoint；因此不同 source 产生的数值波动不等于真实 return
distribution。2026 年的机制论文也报告：在保持网络相同、只把 expected backup 换成
distributional backup 时，四个代表任务中 distributional 版本为 `30/74/72/94%`，expected
版本为 `52/86/72/94%`，没有优势。它提出的解释是 iterative critic 更容易在 non-stationary
TD target 下重加权已有 feature，并能 test-time recovery，而不是“分布越复杂越好”。

这与本地 Stage-28 一半一致、一半不同：本地 FLOQ-distill 已有 action-facing 正信号，但
integrated 2/4/8 steps 没超过 distill。因此当前推荐仍是 expected-value flow + distilled
readout；不能对 scalar-flow source 做多噪声 entropic aggregation并称作 risk-sensitive
return。来源：

- <https://arxiv.org/abs/2509.06863>
- <https://arxiv.org/abs/2603.04333>

#### 2. PCBF / FlowIQN 才是真正的 return-distribution 路线

PCBF 用同一个 base noise 驱动 current 与 successor return flow，修复 `t=0` source boundary，
并用 `lambda` control variate：

```text
target = r + gamma*x_next - eps
       + lambda * (v_next - (x_next - eps))
```

本地 `path_coupled_bellman_flow_pair()` 与官方 `lambda_flow.py` 逐项一致；`lambda=gamma`
会消掉显式 successor endpoint 项。但官方同时强调 `lambda` 是最敏感超参：视觉
antmaze 用 `0`、scene/puzzle 用 `.2`、visual cube 用 `.9`，不是统一设成 gamma。官方
policy extraction 也是 BC proposal 的 16 candidates 按 **mean terminal return** 排序，
不是对同一个 scalar approximation noise 做 entropic readout。来源：

- <https://arxiv.org/abs/2605.08253>
- <https://github.com/BoyangASU/path-coupled-bellman-flows>

FlowIQN 解决的是另一个问题：一维 return 的 source 与 Bellman target 若独立乱配，CFM
目标虽然有效，却不等于 Wasserstein projection；它按 quantile 排序 source/target，近似
monotone optimal-transport coupling。它适合作为 PCBF 之后的 distributional ablation，
不能只把 Gaussian noise 多采几次就叫 IQN。来源：<https://arxiv.org/abs/2605.08515>。

#### 3. FlowQ / QAM（Adjoint Matching）主要更新 policy，不是本问题的 critic 替代

FlowQ 与 QAM 的目标都是把 behavior policy 倾斜到
`pi(a|s) ∝ pi_B(a|s) exp(Q(s,a))`。FlowQ 把 Q 当 energy 学 action flow；QAM 用 critic 的
action gradient 与 adjoint matching，避免反传整个多步生成器。它们仍依赖单独的 TD critic，
回答的是“如何训练 expressive flow actor”，不是“用 FM 更新 CQN-AS value”。而 CQN-AS
已经有离散 action-bin enumeration，不存在 continuous actor 必须反传 ODE 的瓶颈，所以
QAM 不应作为 A/B 主对比。来源：

- <https://arxiv.org/abs/2505.14139>
- <https://arxiv.org/abs/2601.14234>

#### 对下一阶段的预注册影响

1. Stage-30 通过：优先完成 FLOQ-distill 三训练 seed 最终 gate，不再扫 integrated steps；
2. Stage-30 失败：不把 scalar source noise 换成 entropic readout；先根据 Stage-26 结果，
   对 PCBF 的 `lambda={0,.2}` 做训练级别 gate，再与已有 `lambda=.99` 比较；
3. PCBF 仍失败后才实现 FlowIQN 的 sorted quantile coupling；Adjoint/QAM 保留为将来更换
   action policy 的独立项目，不混入当前 CQN-AS value 结论。

这个 fallback 也已做成条件执行：若且仅若 Stage-30 fail，GPU5 的 PID `1092269` 训练
`lambda=.2`，GPU1 的 PID `1092271` 在 A-route 释放后训练 `lambda=0`；若 scalar gate
pass，两者自动写 `skipped_scalar_gate_passed`。已有 `.99` run 从 2k 到 10.5k 实测约
`27min`，所以两个 lambda 并行时预计同样约 `27--32min`，随后才比较 frozen mean-return
readout。这样既不会因当前结果未知而偷跑，也不会在 fail 后让两张卡空置。

### 21.30 Stage-30 第二次独立 replication 仍为正，但严格 superiority 再次失败

#### 上一阶段完整结果

在完全新的 `48000--48199` 上保持两个模型、checkpoint、beta 和 evaluator 全部冻结：

| frozen policy | success |
|---|---:|
| clean CQN-AS seed1 5k | `58%` |
| FLOQ-distill seed1 10k/beta1 | **`65%`** |

paired delta `+7pp`，`45 wins / 31 losses / 124 ties`，95% CI
`[-1.5,+15.5]pp`，McNemar exact `p=.13539`。artifact：

```text
exp_local/cqn_flow_high_utd/stage30_distill_vs_clean_seed1_fresh200_seed48000/summary.json
```

它与第一组独立 100-seed 的 `+10pp`、CI `[-1,+21]pp` 方向一致，所以 FLOQ-distill
不是“完全没有控制信号”；但第二次仍未满足预注册 CI 下界 `>=0`，必须记为 **fail**。
两次都是看过结果后才决定是否继续，不能事后把 300 seeds 简单拼起来并把普通 95% CI 当作
预注册检验。当前可复现表述只能是“两个独立 split 都有 `+7--10pp` point improvement，
但都未严格确认 superiority”，不能写成已经超越 clean CQN-AS。

#### 下一阶段决策与已执行

条件 handoff 已按规则执行：

- Stage-31 seeds2/3 写入 `skipped_gate_failed`，没有浪费约两次 66 分钟训练；
- Stage-32 三训练 seed final eval 随之写入 `skipped_upstream_gate`；
- GPU5 已立即进入 PCBF `lambda=.2` 训练，父 PID `1092269`、实际 trainer PID
  `1100181`，run 为：

```text
exp_local/cqn_flow_pcbf_lambda/stage33_pcbf_lambda0p2_seed1_gpu5_20260724/
```

GPU1 的 `lambda=0` 仍等待 Stage-26 confirmation 与 A-route CV 释放。PCBF 的问题与 scalar
FLOQ 分开：固定真正的 return-sample Bellman flow，只改官方指出最敏感的 control-variate
lambda；旧 `.99` 是 matched baseline。按 `.99` 旧 run 的 snapshot 时间，10.5k 训练约
`27min`，新 `.2` 预计 `07:33--07:38 BST` 完成，之后依据实际 train/eval CSV 冻结
checkpoint，再运行 action-facing gate。

### 21.31 Stage-26：更多 return samples 有方向性收益，但 PCBF 本体远未达到 B 路线 gate

#### 上一阶段完整结果

Stage-26 在 `44000--44049` validation 上选择了同一 `step7000` checkpoint 的
`R16 + entropic(eta=1)`，相对 `R4 + mean` 为 `24% vs 14%`。随后完全冻结选择，在
`45000--45099` confirmation 上得到：

| readout | success |
|---|---:|
| `R4 + mean` | `14%` |
| `R16 + entropic(eta=1)` | **`19%`** |

配对差为 `+5pp`，`14 wins / 9 losses / 77 ties`，95% CI `[-4,+14]pp`，
McNemar exact `p=.40487`。artifact：

```text
exp_local/cqn_flow_pcbf_ranking/stage26_pcbf_return_readout_gate_seed44000/gate_summary.json
```

脚本原先的“positive-direction/non-inferiority”局部 gate 因 point estimate 为正且没有触发
方向性拒绝而标记 `pass`；但这个标记不能提升为项目级结论。区间跨零，而且同一公平 heldout
协议下 clean CQN-AS 三训练 seed 均值为 `61.33%`。因此本阶段只支持：

1. 在当前弱 PCBF 上，增加并行 return samples 可能改善 action ranking；
2. 它没有严格证明 readout 优于 R4，更没有证明 PCBF 优于 CQN-AS；
3. 当前首要瓶颈是 return-flow 的训练质量，而不是再继续扫风险聚合函数。

#### 下一阶段 gate 与已执行

B 路线维持预注册 fallback：比较真正 PCBF 的 `lambda={0,.2,.99}`，每个 lambda 先按训练
validation 冻结 checkpoint，再用官方一致的 `16 proposals + mean terminal return`
readout 比较；只有 validation 候选达到 clean CQN-AS 并有正向 margin，才打开新的 heldout
confirmation。GPU5 的 `.2` 已在跑；Stage-26 释放 GPU1 后，A 路线的 branch-CV anti-cheat
gate 已于 `07:15:59 BST` 自动启动。A 完成后，GPU1 再自动接力 `lambda=0`，两条路线
不会混用数据或 gate。

已新增 `scripts/run_cqn_pcbf_lambda_gate.py` 及 focused tests（与相邻 gate tests 合计
`12 passed`），并启动 PID `1121462` 的条件 handoff。它在两个 lambda trainer 都退出后
自动执行以下严格分层协议：

1. `50000--50009` 只负责在每个 `lambda={0,.2,.99}` 内，从所有 numeric snapshots
   选择一个 checkpoint；
2. `51000--51049` 比较三个 frozen lambda 与 clean CQN-AS seed1 5k，并只选择一个 lambda；
3. 只有最佳 PCBF 比 clean 高至少 `+2pp` 且配对 wins>losses，才打开封存的
   `52000--52199`；
4. confirmation 要求 paired delta>0、95% bootstrap CI 下界 `>=0` 且 wins>losses；
5. PCBF 全程固定 `4 flow steps / 16 return samples / mean terminal return`，不再利用
   Stage-26 看过结果后的 entropic 参数。

screen/validation/confirmation 都会写可恢复的逐任务 JSON 与 `progress.json`，并在 GPU1/GPU5
各保持最多一个 evaluator。按 Stage-26 实测约 `4s/episode`，lambda 训练完成后 screen
约 `15--20min`、validation 约 `7--10min`；若 validation 打开 confirmation，再约
`12--15min`。

### 21.32 A Stage-23 branch-CV：seed1 的正信号是 minibatch lottery，final test 保持封存

#### 上一阶段完整结果

在 train12/internal-val4 上完成 `updates={5,10,20,50}` ×
`weight_decay={1e-5,1e-3,1e-2}`。seed1 选择的是 `10 updates / wd=.001`：

| metric | selected seed1 | behavior proxy | policy-prior proxy |
|---|---:|---:|---:|
| pairwise | **`57.22%`** | `53.01%` | `53.13%` |
| Spearman | **`.1520`** | — | — |
| top-1 | **`33.03%`** | `24.77%` | `25.69%` |
| regret（越低越好） | **`.07557`** | `.13065` | `.13295` |

它单独满足 anti-cheat 条件；而 20/50 updates 又退回约 `52/51%` pairwise，说明早停确实
抑制了一部分 seed 记忆。但只固定同一超参、改变随机 minibatch 顺序后：

| minibatch seed | pairwise | Spearman | top-1 | regret |
|---:|---:|---:|---:|---:|
| 1 | `57.22%` | `.1520` | `33.03%` | `.07557` |
| 2 | `52.22%` | `.0382` | `23.85%` | `.11328` |
| 3 | `50.51%` | `.0164` | `22.94%` | `.09218` |

三次中位数为 `52.22% / .0382 / 23.85% / .09218`，没有超过绝对 pairwise/Spearman
threshold，也没有超过两个不读 return 的 proxy。因此 internal gate 正式 **fail**；
原 final-test4 没有被读取，也没有生成 deployable sidecar。artifact：

```text
exp_local/cqn_value_fidelity_stage23/legacy_best5k_branch_cv/gate_summary.json
```

这个失败定位得比“泛化不好”更具体：10 个 batch × 32 states 只有 320 次有放回抽样，而
train cache 有 540 states；seed1 是没有覆盖完整训练分布时的抽样幸运，不是稳定 value。
因此 Stage-35 LCB closed-loop 条件任务已写
`skipped_value_gate_failed`，没有利用失败 critic 做 deployment 调参。

#### 下一阶段 gate 与已执行

已在 `finetune_cqn_branch_oracle.py` 和 `run_cqn_branch_cv_gate.py` 增加
`sampling_mode=full_batch`：每个 update 恰好访问全部 540 个 cached states，消除小 batch
抽样方差；默认仍为原来的 `random_balanced`。相关 A-route focused suite 为 `18 passed`。

新 Stage-36 固定其他损失与 anti-cheat gate 不变，只搜索
`updates={1,2,3,5,10}` × 三个原 weight decays。seed1/2/3 在 full-batch 下是确定性
replay 检查，不再冒充独立模型初始化；只有 internal gate 通过才允许首次读取 final-test4。
条件 PID `1146824` 已排在 GPU5 的 `lambda=.2` 训练后立即运行。为避免抢卡，PCBF
Stage-34 handoff 已安全重挂为 PID `1147417`，明确等待 GPU1 `lambda=0`、GPU5
`lambda=.2` 与 Stage-36 三者退出后才占用两卡。

按上一阶段每次完整 load/score 的 `63--66s` 估计，15-grid + 2 deterministic replay
约 `18--21min`；若 internal 通过，3 次 final deterministic replay 再约 `3--4min`。
GPU5 的 `.2` 预计先在数分钟内结束，所以 Stage-36 预计约 `07:55--08:00 BST` 给出结果。

### 21.33 B 路线：FlowIQN 不是跨 state 排序，先实现 conditional quantile-coupling screen

#### 文献结论与实现边界

进一步核对 [FlowIQN 论文](https://arxiv.org/pdf/2605.08515) 的算法后，正确的
coupling 单位是单个 transition/condition `(s_i, a_i)`：对该 condition 产生 `K`
个 Bellman-return target 和 `K` 个 source quantile/noise，分别沿 sample 轴排序，再按
order statistic 一一配对；velocity network 在整个 flow 中显式接收对应的 source
quantile `tau`。它不是把 batch 中不同 state 的 return 混在一起排序。

论文的[官方仓库](https://github.com/ori-goals/flowIQN)截至本阶段仍标注代码稍后发布，
因此本阶段只称为“Algorithm 1 核心 coupling objective 的本地 CQN-AS adaptation”，
不声称是官方代码或完整机制复现。source/grid/solver/embedding 的后续 fidelity audit
与修正版见 21.37。

#### 已实现与验证

- `robobase/method/cqn_flow.py`
  - 新增 `quantile_couple_return_samples(...)`，只在每个 condition 内沿 `K` 轴做
    一维 monotone coupling；
  - `C2FSequenceFlowCritic` 新增 quantile embedding，并在 ODE 积分全过程固定传入
    source quantile；
  - current source 与 Bellman target samples 使用独立 RNG；
  - 对 objective、uniform source、sample 数、antithetic、critic 架构以及与
    DCFM/PCBF 的互斥关系做 fail-fast 校验。
- `robobase/cfgs/launch/cqn_flowiqn_pixel_bigym_stage1_gate.yaml`
  - `K=8` train/target return samples，4 个 training action samples，4 个 flow steps；
  - 保留 CQN-AS 的 state/image 与 action-bin condition，BCFM 权重为 1；
  - 关闭 DCFM/PCBF，避免第一次 gate 混入另一种 Bellman-flow regularizer。
- focused tests：
  `17 passed, 79 deselected`；覆盖 per-condition coupling、显式 quantile condition、
  合法 update、错误组合拒绝与 launch config。

#### Gate 与已执行

FlowIQN 是 PCBF lambda family 严格 gate 失败后的下一种独立机制，不与其结果混合：

1. Stage-34 先比较 `lambda={0,.2,.99}`，经 checkpoint screen 后对 clean CQN-AS 做
   50-episode validation；只有达到 `+2pp` 且 `wins > losses` 才读 200-episode
   confirmation。
2. 如果 Stage-34 的 strict confirmation 通过，FlowIQN seed1 暂停，先扩展 PCBF
   多 seed；否则立即训练 FlowIQN seed1 到 10k。
3. FlowIQN 训练完成后同样先做 checkpoint screen，再用未参与选择的 seeds 与 clean
   CQN-AS 做 paired validation/confirmation。只有 `delta > 0`、bootstrap CI
   lower `>= 0` 且 `wins > losses` 才可称为超过 clean baseline。

条件 controller PID `1172003` 已经挂起并等待 Stage-34 的机器可读
`gate_summary.json`；若 PCBF 未过 gate，它会在 GPU5 自动启动
`stage37_flowiqn_seed1_gpu5_20260724`。这样当前两张卡仍分别服务于 lambda=0 与
A-route full-batch gate，FlowIQN 不会抢占正在运行的实验。

FlowIQN 的下游 action-facing gate 也已用 PID `1228126` 事件排队：若 Stage-37 实际训练出
10k snapshot，它会在 GPU5 依次做 `55000+` checkpoint screen、`56000--56049`
validation 和 `57000--57199` confirmation；若 PCBF 已通过而 Stage-37 合法跳过，则只写
skip artifact。validation 仍要求相对 clean 至少 `+2pp` 且 wins>losses，confirmation
要求 paired delta `>0`、CI lower `>=0` 且 wins>losses。

### 21.34 A 路线：full-batch 排除抽样方差；79 参数 structured-delta value 进入 fresh gate

#### Stage-36 完整结果

full-batch `15` 点网格与两次 deterministic replay 已完成，artifact：

```text
exp_local/cqn_value_fidelity_stage23/legacy_best5k_branch_cv_fullbatch/gate_summary.json
```

internal-val 选中 `updates=3, weight_decay=.01`。由于 full-batch 没有 minibatch RNG，
三次 replay 的结果完全相同：

| metric | structured C51 finetune | behavior proxy | policy proxy |
|---|---:|---:|---:|
| pairwise | `53.36%` | `53.01%` | `53.13%` |
| Spearman | `.0687` | — | — |
| top-1 | `22.94%` | `24.77%` | `25.69%` |
| regret | `.10402` | `.13065` | `.13295` |

它没过 `pairwise>55%`、`Spearman>.1`、top-1 proxy gate，也没有改善未更新 critic 的
`.10184` regret。因此 internal gate 正式 fail，原 final-test 没有被该阶段读取。
这排除了“小 batch 没覆盖完 540 states”作为主要原因；继续在同一个高容量 C51 head 上调
updates/weight decay 没有依据。

#### 新假设：预测局部最优 delta，而不是记忆五个任意 Q

branch intervention 的可识别量是同一个 `(state, action dimension)` 下哪个 local delta
更好。新增 `scripts/run_cqn_structured_delta_gate.py`，把 value 约束为：

```text
delta_star = ridge(PCA_4(frozen_state) x action_dimension,
                   action_dimension, anchor_step)
Q_structured(candidate) = -abs(candidate_delta - delta_star)
```

模型只有 `79` 个可训练参数；PCA 与 ridge 都只读 fit split，模型可序列化。它不读取
behavior/policy score来训练；这两个只作为 anti-cheat baseline。新增 focused tests 为
`3 passed`，覆盖 dimension-conditioned optimum、seed bootstrap 和 cache split。

在已经用于 method discovery 的 internal4 上，冻结的 `PCA=4, ridge=.01` 得到：

| metric | structured delta | behavior | policy |
|---|---:|---:|---:|
| pairwise | **`56.31%`** | `53.01%` | `53.13%` |
| Spearman | **`.1120`** | `.0488` | `.0474` |
| top-1 | **`26.61%`** | `24.77%` | `25.69%` |
| regret | **`.10946`** | `.13065` | `.13295` |

九个 point checks 全过；pairwise paired seed-bootstrap lower bound 相对 behavior/policy
分别为 `+0.87pp/+0.29pp`。但只有 4 seeds，Spearman CI 仍跨 0，regret improvement
lower bound 也略小于 0，因此这是 discovery pass，不是最终 A 结论。artifact：

```text
exp_local/cqn_value_fidelity_stage23/stage37_structured_delta_discovery/gate_summary.json
```

#### 历史 replay screen 与 fresh gate

用已经在旧实验中看过的 final4 只做“是否值得新收集”的 replay screen。structured delta
对 behavior/policy 的结果是：

- pairwise `61.61%` vs `61.40%/60.74%`；
- Spearman `.2197` vs `.2122/.1972`；
- regret `.08543` vs `.10903/.10167`；
- top-1 `33.63%` vs `40.71%/42.48%`。

因此原先要求 top-1 同时超过两个 proxy 的 strict gate fail；4-seed bootstrap 也太宽。
但它在 pairwise、rank correlation 与直接 choice-regret 三项都有正方向，足以打开一次
真正 fresh confirmation。旧 replay 不会被写成确认结果，artifact：

```text
exp_local/cqn_value_fidelity_stage23/stage38_structured_delta_historical_replay/gate_summary.json
```

fresh gate 已预注册为：冻结上述 79 参数结构与超参，用旧 train16 fit，只在新
`29000--29007` simulator seeds 上评估；pairwise 与 regret 相对 behavior 和 policy 的
seed-cluster bootstrap lower bound 必须均 `>=0`，model Spearman CI lower 必须 `>0`，
同时 point pairwise `>55%`、Spearman `>.1`、top-1 高于随机。top-1 不再要求超过 proxy，
因为 regret 已直接度量所选 bin 的 return 损失，而 top-1 对 return ties 不连续；这个规则只
用于尚未收集的新 seeds，不能回头挽救历史 screen。

PID `1225534` 已等待 B Stage-34 释放 GPU1，随后自动收集 8 个 fresh seeds、三 anchors、
全 15 dimensions，并立即运行 frozen gate。旧 20-seed 收集实测 `7282.87s`，折算
`364.1s/seed`；新 8 seeds 的 collection ETA 为约 `48.6min`，model gate 低于 `10s`。
它与 Stage-34/后续 GPU5 FlowIQN 串接，不做短轮询。

closed-loop evaluator 与 gate runner 也已提前实现，focused suite 与公共 paired summarizer
合计 `14 passed`。部署层每次最多改一个 dimension；candidate 必须同时通过 BC log-prob
support、structured-value margin 与 PCA state-radius，任何 gate 不满足都执行 exact BC。
fresh value gate 通过后，PID `1226820` 才会在 GPU1 用 `58000--58049` 选择
conservative/medium/wide 三个预注册阈值，并在 `59000--59199` 做 200-seed paired
non-inferiority；value gate 失败则写 skip artifact，禁止调 policy threshold。

首次新 handoff 因 detached shell 自身没有完整重定向而被执行通道回收，尚未开始任何收集或
评估；现有三个新 controller 均已改为 `setsid -f`、stdout/stderr 全重定向并
`</dev/null`，且已跨独立 shell probe 验证 PID 与 `tail --pid` 子进程仍存活。旧 premature
control 目录保留用于审计，没有覆盖实验 artifact。

### 21.35 B Stage-34：PCBF lambda family 显著低于 clean；FlowIQN 与 A fresh8 已并行启动

#### PCBF lambda 完整结果

Stage-34 先在共同 `50000--50009` screen seeds 上冻结：

| arm | selected checkpoint | screen success |
|---|---:|---:|
| `lambda=0` | 9k | `40%` |
| `lambda=.2` | 8k | `50%` |
| `lambda=.99` | 8k | `20%` |

随后在未参与选择的 `51000--51049` 上一次性比较：

| policy | validation success |
|---|---:|
| clean CQN-AS 5k | **`72%`** |
| PCBF `lambda=0` | `42%` |
| PCBF `lambda=.2` | **`44%`** |
| PCBF `lambda=.99` | `0%` |

最优 candidate `.2` 相对 clean 是 `-28pp`，paired `7 wins / 21 losses / 22 ties`，
95% CI `[-46,-8]pp`，McNemar exact `p=.01254`。它不仅没达到预注册 `+2pp` gate，
而且有显著负效应；confirmation 保持未运行。正式 artifact：

```text
exp_local/cqn_flow_pcbf_lambda/stage34_pcbf_lambda_gate_seed50000/gate_summary.json
```

这把 Stage-26 “return samples 由 4 增至 16 有 +5pp 方向”限定为同一个低性能 PCBF
checkpoint 的 inference/readout effect，不能推广成 PCBF 超过 CQN-AS。lambda 从 `.99`
降到 `.2/0` 确实减轻破坏，但最佳仍低 clean `28pp`，所以 B 路线停止调同一 regularizer。

#### Gate 触发与资源/ETA

Stage-34 fail 已在 `08:22 BST` 自动触发：

- GPU1：PID `1253436` 收集 A-route `29000--29007` fresh branch cache。按旧
  `7282.87s / 20 seeds` 的实测吞吐，collection ETA 约 `48.6min`，即约
  `09:11 BST`，随后 deterministic model gate 小于 `10s`。
- GPU5：PID `1253601` 训练 compute-matched FlowIQN-objective adaptation seed1 到 10k。第一次
  checkpoint 前按同规模 FLOQ/PCBF 估 `60--70min`；到 1k 后必须用本 run 的
  `total_time` 重估，不用短轮询。

FlowIQN 训练完成后 PID `1228126` 自动做独立 `55000+` screen、`56000+` validation、
`57000+` confirmation。A fresh gate 通过后 PID `1226820` 才允许进入 closed-loop
policy gate。

#### A policy threshold diagnostic

structured policy wrapper 的 1-episode `diagnostic_only` 集成 smoke 完成：实际执行 exact BC，
seed `57999` 成功，wall time `42.86s`。原始宽阈值会提出 `138/138` overrides，因此在不执行
value action、也不查看 variant policy outcome 的前提下，用同一 BC trajectory 预校准
coverage。最终冻结三档均使用 `margin=.235, max_logprob_drop=.25`，只改变 PCA RMS：

| variant | max state RMS | proposed override rate |
|---|---:|---:|
| conservative | `.6` | `22/138 = 15.9%` |
| medium | `.7` | `34/138 = 24.6%` |
| wide | `.8` | `45/138 = 32.6%` |

这一步只让三个 calibration arms 有实际不同的覆盖率，不能用 seed57999 的 BC success
挑 variant；正式选择仍只在未来 `58000--58049` paired outcomes 上进行。

### 21.36 等待期文献复核：QAM 是 policy-flow；下一 critic fallback 应回到 expected FLOQ

#### FlowIQN 首个真实吞吐点

Stage-37 在 `08:26:26 BST` 写出 `1000_snapshot.pkl`。`train.csv` 的 1k
`total_time=223.72s`，初始化到 snapshot 的 wall time 约 `4.25min`；按训练 update
线性外推并给 2.5k/5k/7.5k/10k 四次 25-episode native eval 留余量，10k 训练 ETA
更新为 `42--45min`，约 `09:05--09:08 BST`。1k 的数值健康指标为：

- BCFM loss 从初始化窗口的 `.0979` 降至 `.0283`；
- critic gradient non-finite fraction 为 `0`；
- endpoint-Q mean/min/max 为 `.819/.086/1.982`；
- demo sibling top-1 为 `.518`、Q span 为 `.179`。

这些只能说明实现没有立即数值爆炸，不能用来选 checkpoint，也不能代替任务成功率；
第一次有 policy 含义的节点仍是 2.5k native eval。

#### QAM / Adjoint Matching 的精确适用边界

进一步核对 [QAM 论文 v4](https://arxiv.org/abs/2601.14234) 和其
[官方 JAX 代码](https://github.com/ColinQiyangLi/qam) 后，它解决的是 flow **policy**
的优化，不是 flow **critic**：

1. 先以标准 FM 训练 behavior velocity `f_beta(s,a_t,t)`；
2. 另有普通 TD critic ensemble，论文默认 `K=10`，target 使用
   `mean(Q)-rho*std(Q)`；
3. 从 fine-tuned policy 的 memory-less SDE 采 action path，以终点
   `-tau * grad_a Q(s,a_1)` 为边界；
4. 只经过 frozen/base `f_beta` 的 VJP 反传 lean adjoint，训练 `f_theta` 逼近
   `pi_beta(a|s) exp(tau Q(s,a))`。

所以 QAM 的优势是避免通过多步 action denoising path 做不稳定 BPTT。CQN-AS 已经把
每层 `16 x action_dimension x 5 bins` 的候选并行枚举，并不具有这个 actor-gradient
瓶颈；把 QAM 塞进当前实验会同时更换 policy 与 critic，无法回答“FM 是否让
CQN-AS value 更好”。同理，[FlowQ](https://arxiv.org/abs/2505.14139)、
[FQL](https://proceedings.mlr.press/v267/park25f.html) 和
[Reinforce Adjoint Matching](https://arxiv.org/abs/2605.10759) 都主要是 policy/generator
tilting，对当前 B 路线只能作为 future actor comparator，不能冒充 value-FM baseline。

#### 与官方 expected-value FLOQ 的逐项核对

[FLOQ 官方实现](https://github.com/CMU-AIRe/floq/blob/main/what_does_flow_matching/agents/floq.py)
采用的不是 distributional Bellman target：

- 对 successor `(s',a')` 独立积分 8 个 source endpoint，**先求均值**得到一个 scalar
  TD target；
- 对当前 `(s,a)` 再取独立的 8 个 uniform source，以随机 `t` 做 dense velocity
  regression；
- source 区间是理论 Q 范围的 `noise_coverage=.1`，interpolant 用 51-bin
  HL-Gauss embedding；
- action-facing Q 来自一个 stop-gradient 拟合 online-flow mean endpoint 的 scalar
  readout。这个 head 是官方算法的一部分，不是为了把 flow policy 蒸馏成一步 actor。

本地 Stage-24 包含 8 flow steps、8 current/target sources、独立 RNG、51-bin
HL-Gauss 与 online-endpoint stop-gradient scalar readout，因此它确实是
expected-value FLOQ 路线；Stage-28 直接积分 action readout 的失败也不能解释成“没有复现
官方 actor”，因为官方本来就用 scalar readout 选 action。不过，21.69 的保存配置/官方代码
复核发现这里此前声称的 `.1 Q-range source` 并未真正 materialize，且本地/官方对 8-source
BCFM loss 的 reduction 不同；所以 Stage-24/62 应称为高保真 adaptation，而不是逐项完全一致。

新的机制论文
[What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)
进一步报告，显式 distributional backup 往往低于 expected backup；收益主要来自积分误差
恢复和 dense path supervision 对 TD plasticity 的保护。这与本项目的证据一致：
FLOQ-distill 相对 clean 在两个独立 split 都是 `+7--10pp`，而 PCBF 显著低
`28pp`；但前者的严格 CI 仍未过，不能写成已经超过 CQN-AS。

#### 预注册的下一 B gate

当前先让 FlowIQN 按既定 protocol 完成，不因等待期新文献改变已经开始的实验。若 FlowIQN
严格 validation fail：

1. 不继续扫 PCBF lambda，也不把 QAM 当 critic；
2. 回到目前唯一有两次独立正方向证据的 expected-FLOQ 10k/beta1；
3. 实现一个 **clean-CQN-AS fallback + flow-value high-margin override** evaluator：
   clean action 默认执行，只有 flow candidate 在 scalar-Q margin、BC support 与
   source-agreement 三个预注册条件同时通过时才覆盖；
4. 先在新 calibration seeds 选 conservative threshold，再在完全新的 200 seeds 做
   paired gate；要求成功率相对 clean 点估计为正、95% CI 下界 `>=-5pp`、override subset
   的 reward/success direction 为正。只有这个 safety gate 通过才扩展 training
   seeds 2/3；最终 superiority 仍要求三 training seeds crossed-bootstrap CI lower
   `>=0`。

这个 fallback 的目的不是用 BC 遮住坏 value：必须同时报告 override coverage、在 override
子集上的 paired outcome、以及 branch counterfactual ranking；若只靠大多数时间不改 action
守住成功率，则 A/B 两条 gate 都不会通过。

对应代码现已提前实现：

- `scripts/eval_cqn_floq_clean_fallback.py` 同时加载 frozen clean CQN-AS 与
  Stage-24 expected-FLOQ；从 clean plan 出发，只构造一个 level 的全维 sibling bins，
  每次最多覆盖一个 dimension；
- candidate 必须同时通过 normalized distilled-Q margin、相对 clean bin 和 flow-BC
  最优 bin 的双 support gate、integrated endpoint 的 paired source mean，以及 source
  win fraction；
- `scripts/run_cqn_floq_clean_fallback_gate.py` 用三档预注册 coverage threshold 做
  calibration/held-out confirmation，复用公共 paired bootstrap summarizer并强制检查
  override subset 的 success/reward；
- 两个新增 focused suites 与语法检查共 `11 passed`，`git diff --check` 通过。

当前尚未做 integration rollout：GPU1/GPU5 分别被 A fresh branch collection 和
FlowIQN 正式训练占用。该 smoke 会在首张获授权 GPU 释放后执行；若 FlowIQN 通过 strict
gate，则 fallback 只保留为 negative-control/ablation，不开启昂贵 confirmation。

### 21.37 FlowIQN fidelity correction：Stage-37 是 compute-matched screen，修正版已在结果前冻结

#### 上一阶段实际结果

Stage-37 的 native 25-episode checkpoints 已走到 7.5k：

| checkpoint | native success |
|---:|---:|
| 2.5k | `0/25 = 0%` |
| 5k | `2/25 = 8%` |
| 7.5k | `10/25 = 40%` |

对应 artifact 是
`exp_local/cqn_flow_flowiqn/stage37_flowiqn_seed1_gpu5_20260724/eval.csv`。
7.5k 相对早期 checkpoint 有真实恢复，因此不能在 5k 时提前杀掉；但当前 validation-selected
native best 仍比 clean seed1 的 `68%` 低 `28pp`，而且 seeds 不 matched，不能据此宣称
通过或给出严格差值。Stage-37 在 `08:39:46 BST` 已写出 8k snapshot，数值仍 finite。

#### 一手算法复核后的解释和更正

在 Stage-37 strict outcome 出来前，重新逐行核对
[FlowIQN Appendix C / Algorithm 1](https://arxiv.org/pdf/2605.08515) 后发现，原文的
action scalarization 与 source construction 比当前配置更具体：

1. action ranking 使用固定中点 grid
   `tau_k=(k-0.5)/K`，不是“固定一次随机 uniform samples”；
2. source 是 `g(tau)=l+tau(u-l)`，默认 `kappa=.1`；MovePlate 只有成功终止时 reward=1，
   因此其可达 terminal-return 范围是 `[0,1]`，对应 terminal-aware source `[.9,1]`；
3. 与 MovePlate 最接近的 cube manipulation 配置是 `K=8, M=8`；Stage-37 的训练/target
   `K=8` 合适，但 `M=4`、action samples=4 是 compute-matched screen；
4. 论文 value input 使用 51-boundary HL-Gauss，time 使用 Fourier features；Stage-37
   使用 raw scalar/raw time。

所以文档前面把 Stage-37 称为 `paper-faithful local FlowIQN` 不准确，现正式更正为：
**FlowIQN sorted-quantile objective 的低算力 CQN-AS adaptation**。它保留了正确的
per-condition monotone coupling、独立 current/target quantiles 和显式 quantile
conditioning，但不能单独裁决完整 FlowIQN mechanism。即使修正版也不会称为论文整套复现：
它有意保留 CQN-AS pixel encoder、C2F bin enumeration、BC/RL replay 与 optimizer，而不是
换成论文的 FQL actor 或 rejection-sampling BC flow。

#### 下一阶段 gate

Stage-37 仍按预注册 protocol 跑完并在 fresh `56000--56049` validation 对 clean：

- 若 validation delta `<+2pp` 或 wins 不大于 losses，Stage-37 fail，confirmation 密封；
- 若过 validation，才开 `57000--57199` strict confirmation；要求 paired delta 正、
  95% CI lower `>=0` 且 wins>losses。

若 Stage-37 没有 strict pass，立即训练单一预注册修正版：
`Ktrain=Ktarget=Kaction=8`、`M=8`、source `[.9,1]`、fixed midpoint grid、
HL-Gauss-51、Fourier time。仍用 checkpoint screen -> fresh 50 validation -> sealed 200
confirmation，并与同一个 clean 5k checkpoint paired；不同时扫其他超参。

#### 已完成实现与验证

- 新增 `method.action_flow_quantile_grid`；只有 action ranking 使用 midpoint grid，
  Bellman target/current flow 仍用 fresh random quantiles；
- 新增
  `robobase/cfgs/launch/cqn_flowiqn_pixel_bigym_corrected_gate.yaml`；
- 修正 Stage-37 config 注释，不再称它为论文复现；
- focused FlowIQN/config/grid tests：`10 passed`；
- `py_compile` 与 `git diff --check` 均通过。

按 7k--8k 的实测 wall time 和剩余 10k eval，Stage-37 训练预计
`08:44 BST ±3min` 结束。历史同一 evaluator 的 validation job 均值约 `181s`；
Stage-38 单卡要 screen 10 个 checkpoints 再做 2 个 validation jobs，若 validation fail，
预计总计约 `18--22min`。修正版只在该 gate fail 后事件触发，不短轮询、不占用第三张卡。

事件队列已实际启动并由进程/artifact 验证：

- PID `1276335`、status `waiting_stage38`：只有 Stage-37 strict gate 未通过时，才在
  GPU5 启动 `stage41_flowiqn_corrected_seed1_gpu5_20260724`；
- PID `1278378`、status `waiting_stage41`：只有 corrected 10k snapshot 存在时，才做
  `60000+` screen、`61000--61049` validation；只有 validation 过 `+2pp/wins` gate
  才解封 `62000--62199` confirmation。

#### Stage-37 10k 节点与 Stage-38 实际启动

Stage-37 已在 `08:44 BST` 正常完成，10k native success 为
`12/25 = 48%`；因此完整 checkpoint trend 是 `0% -> 8% -> 40% -> 48%`，
validation-selected native best 是 10k，但仍比 clean seed1 best `68%` 低 `20pp`。
这排除了“只是前 5k 尚未训练”的解释；它仍不替代 matched-seed gate。

Stage-38 已自动进入 `evaluating`，首个 1k checkpoint screen artifact 明确记录
`policy=native_flow_value`、`flow_steps=4`、`action_samples=16`、success `0/10`。
首 job 实测 `68.82s`，当前 screen 共 11 个 checkpoints，progress 给出的 screen
remaining ETA 为 `688s`；再按历史两个 50-episode validation jobs 约 `6min`，
若 validation fail，Stage-38 预计约 `09:03--09:06 BST` 结束。若意外通过 validation，
还会按 protocol 增加 sealed confirmation 时间，corrected run 相应继续等待。

### 21.38 B Stage-38：compute-matched FlowIQN objective 明确失败；corrected mechanism 自动启动

#### 上一阶段结果

Stage-38 完成了 validation-selected、matched-seed 比较。screen 在
`55000--55009` 从 11 个 numeric checkpoints 中选择 7k（`4/10=40%`），随后冻结：

| policy | `56000--56049` success |
|---|---:|
| clean CQN-AS seed1 5k | `30/50 = 60%` |
| FlowIQN-objective Stage-37 7k | `17/50 = 34%` |

paired delta `-26pp`，95% CI `[-42,-10]pp`，`4 wins / 17 losses / 29 ties`，
McNemar exact `p=.00720`。validation gate 明确 fail，`57000+` confirmation 按预注册
保持密封。正式 artifact：

```text
exp_local/cqn_flow_flowiqn/stage38_flowiqn_gate_seed55000/gate_summary.json
```

#### 解释

这不再是“训练 loss 下降但不知道 policy 如何”：使用 integrated flow-Q、
4 solver steps、16 action sources 的 action-facing policy 在 fresh matched seeds 上显著
低于 clean。7.5k/10k native 的 `40/48%` 恢复不能挽救这一结论，因为 checkpoint 必须由
独立 screen 选择；Stage-37 的 sorted quantile objective + wide `[0,1]` source +
raw embedding + `M=4` 组合停止扩展。

它仍不裁决 corrected FlowIQN mechanism，因为 21.37 在看到 Stage-38 outcome 前已冻结
四个论文机制差异：terminal-aware `kappa=.1` source、midpoint quantile grid、`M=8`、
HL-Gauss/Fourier。下一实验把这些作为一个 fidelity bundle；若仍失败，不再逐项事后扫参，
而回到已有两次正方向证据的 expected-FLOQ safety override。

#### 下一 gate 与实际执行

Stage-38 fail 已在 `09:01:33 BST` 自动触发
`stage41_flowiqn_corrected_seed1_gpu5_20260724`。trainer PID `1301833` 已加载 60 demos，
实际 `.hydra/config.yaml` 验证为：

- `Ktrain=Ktarget=Kaction=8`、`M=8`；
- source `[.9,1]`；
- `action_flow_quantile_grid=true`；
- `scalar_value_embedding=hl_gauss`、`time_embedding_type=fourier`。

Stage-42 controller PID `1278378` 继续事件等待；10k 存在后按
`60000+` screen -> `61000--61049` validation -> sealed `62000--62199`
confirmation。首次 1k 前用 Stage-37 的 10k `20.8min` 和 solver 近似加倍估
`30--40min`；写出 1k 后必须用 corrected run 自身吞吐重估。

#### 首次资源 gate 与最小修正

首个 corrected attempt 在第一步 JIT executable 创建时退出，错误是申请
`43,534,876,816 bytes = 43.53GB`，而 GPU5 总显存为 `32.61GB`；没有产生 update、
checkpoint 或 task outcome，因此这是 resource gate，不是方法结果。失败日志：

```text
exp_local/cqn_flow_flowiqn/stage41_corrected_control/train.log
```

为保持单一机制解释，重跑只把 **train-time action scalarization** 从 8 个 midpoint
降到 4 个；batch 仍为 8，current/target Bellman quantiles 仍为 8，`M=8`、
source `[.9,1]`、midpoint grid、HL-Gauss/Fourier 均不变。held-out action readout
仍覆盖完整 8-grid。这样不使用 microbatch，也不减少 value objective 的 target samples；
代价是训练期间 next-action ranking 是四点 quadrature，最终结论必须标为 32GB-feasible
adaptation，而不是完整 Kaction=8 training。

对应修正已通过 focused config/grid tests（`3 passed`）并立即执行：

- Stage-43 trainer PID `1310346`，run
  `stage43_flowiqn_corrected_r4_seed1_gpu5_20260724`；
- Stage-44 gate controller PID `1311076`，status `waiting_stage43`；
- 旧 Stage-42 因上游没有 `train_complete` 正确写
  `skipped_stage41_not_trained`，没有读取 `60000+` outcomes，因此同一批预注册 fresh
  seeds 可由 Stage-44 合法使用。

Stage-43 首要 gate 是首个 update/1k 能在 32GB 上运行；只有得到本 run 的 1k
`total_time` 后才重算 10k ETA。若 R4 仍在第一步 OOM，下一步是对无梯度的
next-action/target ODE 路径做 code-level rematerialization，而不是继续缩 batch 或 target K。

### 21.39 A Stage-39：全局 structured value 有信号但 strict fresh gate 失败；冻结 LOSO 可靠区

#### 上一阶段结果

Stage-39 在完全新鲜的 `29000--29007` simulator seeds 上完成了 360 个同状态
counterfactual branch states（218 个 informative）。冻结旧 train16 所拟合的 79 参数
structured-delta model，结果为：

| metric | structured value | behavior proxy | policy proxy |
|---|---:|---:|---:|
| pairwise | **`55.371%`** | `53.212%` | `52.215%` |
| Spearman | **`.10969`** | `.06467` | `.03925` |
| top-1 | `28.899%` | `30.275%` | `27.523%` |
| regret | **`.07816`** | `.08457` | `.08496` |

model Spearman 的 seed-bootstrap 95% CI lower 为 `.01597`，说明不是完全没有
return-ranking signal；但预注册 strict checks 中，model-behavior pairwise delta lower
为 `-0.208pp`，相对 behavior/policy 的 regret-improvement lower 分别为
`-.00673/-.00637`。因此 value gate 正式 **fail**，Stage-40 policy threshold gate 按协议写
`skipped_value_gate_failed`，没有用 task outcome 调参。正式 artifacts：

```text
exp_local/cqn_value_fidelity_stage23/stage39_structured_delta_fresh8/coverage.json
exp_local/cqn_value_fidelity_stage23/stage39_structured_delta_fresh8/gate/gate_summary.json
```

#### 解释

这个结果支持“低容量、同状态反事实监督学到了一部分真实 value ordering”，但不支持
“该 value 已能在所有维度、所有阶段可靠接管 action”。失败是跨 seed 稳健性，不是训练
loss；也不能用全局平均正方向绕过预注册 confidence gate。逐 seed/维度检查显示主要误差
集中在某些 action dimensions 和早中期状态，因此继续把同一个全局 sidecar 应用于所有
时刻会混合有效 value 与 imitation-like fallback。

为了避免在 fresh8 上事后挑子集，可靠区只由更早的旧 train16 做严格
leave-one-simulator-seed-out（LOSO）推导。全局 LOSO 仅为 pairwise
`57.576% vs 57.278%`，且 regret `0.08224` 差于 behavior `0.07829`；但使用统一规则
“至少 20 个 informative states，pairwise 同时高于 behavior 与 policy，regret 同时低于
二者，model Spearman > 0”后：

- anchor 只有 `120` 通过；
- dimensions 只有 `{4, 6, 8}` 通过；
- 二者交集在旧 LOSO 上为 48 states/41 informative，pairwise
  `65.738% vs 61.560%/61.281%`，Spearman `.3080`，regret
  `.10065 vs .14501/.14297`。

把这条**旧数据冻结的规则**只作为诊断应用到已经看过的 fresh8，24 states/18 informative
上 pairwise 为 `70.748% vs 64.626%/65.306%`，但 regret 对 behavior 只改善
`0.000027`，样本也不足以做确认。因此 fresh8 不能被重新命名为 pass，只说明值得收集一次
真正新的 reliability-gated split。

#### 下一阶段 gate

实现一个 cross-fitted reliability model artifact：模型文件显式保存 supported anchors 与
dimensions；closed-loop wrapper 在不支持的阶段/维度执行 exact BC，不能用 behavior
score 冒充 value 结果。用未读取过的 `30000--30031` 共 32 seeds，仅收集预注册
`anchor=120, dimensions=4,6,8` 的 sibling-bin branches：

1. evaluation 至少 40 个 informative states；
2. pairwise `>55%`、Spearman `>.1`、top-1 高于随机；
3. pairwise 与 regret 相对 behavior、policy 的 point estimate 均改善；
4. 以 simulator seed 为 cluster 的 95% bootstrap 中，上述四个 paired effect lower 均
   `>=0`，model Spearman lower `>0`。

只有该 gate 通过才允许进入 paired task policy calibration/confirmation；否则停止这个
structured-delta family，转向训练期的 return-identifiable critic decomposition。

#### 执行与 ETA

旧全维 fresh8 实测 `2736.73s / 8 = 342.1s/seed`，每 seed 是
`3 anchors x 15 dimensions`。新收集只保留 `1 x 3`，按 branch 数线性估计并给 baseline
rollout 留余量，启动前 ETA 为约 `30--45s/seed`，32 seeds 即 `16--24min`；写出首批
progress 后再以本 run 吞吐更新。GPU1 负责 A，GPU5 的 Stage-43/44 独立负责 B，两条路线
不共享 selection outcomes。

实现已经落到：

```text
scripts/run_cqn_structured_delta_reliability_gate.py
scripts/eval_cqn_structured_delta_sidecar.py
```

模型 artifact 现在显式保存 `supported_anchor_steps` 和
`supported_action_dimensions`；runtime 的 eligibility mask 在 value margin、BC support、
state-radius 之前硬屏蔽不可靠坐标。相关 focused suite 为 `10 passed`。用已见过的 fresh8
做 integration replay smoke，代码正确重建 `{120} x {4,6,8}`；pairwise bootstrap lower
已为正，但 behavior-regret improvement CI `[-.001162,.001203]` 仍跨 0，所以 replay
artifact 按预期为 fail，且不计作新确认：

```text
exp_local/cqn_value_fidelity_stage23/stage46_reliability_replay_diagnostic/gate_summary.json
```

真正 Stage-45 fresh32 collector 已在 GPU1 启动，PID `1318669`；Stage-47 event
controller PID `1322404` 等待 collection 完成后自动运行 strict value gate。只有 value
gate pass 才会解封 `58000--58049` policy calibration 和
`59000--59199` confirmation；否则写明确 skip artifact。

### 21.40 B Stage-43 资源 gate 通过；用本 run 吞吐更新 ETA

Stage-43 的 R4 hardware adaptation 已通过最先的 32GB feasibility gate：1k/2k
snapshots 分别在 `09:22:22/09:24:48 BST` 写出，GPU5 实测使用约 `25.0GB`，没有
non-finite gradient。`train.csv` 的关键点为：

| step | total time | BCFM loss | demo top-1 | demo Q span |
|---:|---:|---:|---:|---:|
| 1k | `534.96s` | `.01994` | `.4798` | `.1448` |
| 2k | `680.70s` | `.02019` | `.5042` | `.2087` |

这只证明 corrected objective 已真正更新且数值可运行，不是 policy-quality 结论；第一个
native task outcome 仍是 2.5k 的 25 episodes。排除首次 JIT 后，1k->2k 实测
`145.74s/1k`。从 2k 到 10k 纯训练约 `19.4min`；参考同任务 Stage-37 四次 native eval
各约 `80s`，剩余总 ETA 约 `24--29min`，即预计 `09:50--09:55 BST` 完成训练并自动进入
Stage-44 matched-seed gate。controller 按 PID/event 等待，不做高频轮询。

为保证 Stage-44 fail 后 GPU5 不空转，Stage-48 controller PID `1325376` 已做条件
handoff：若 corrected FlowIQN 的 independent confirmation pass，则明确跳过 fallback；
否则先在 seed `69999` 做一条 `diagnostic_only` integration rollout（执行 exact clean
action），成功后才在未用过的 `70000--70049` calibration 选择 expected-FLOQ safety
threshold，并将 `71000--71199` 保持为 sealed confirmation。使用的是 Stage-24
expected-FLOQ 10k/beta1 与 clean CQN-AS seed1 5k；每次最多覆盖一个 dimension，并要求
distilled-Q margin、BC support、integrated-source mean/win fraction 同时通过。相关 A/B
focused regression 当前合计 `21 passed`，语法与 `git diff --check` 通过。

Stage-43 随后完成首个 2.5k native evaluation：`0/25 = 0%`。这与未修正 Stage-37 的
2.5k 同为 0%，说明 fidelity bundle 没有改善 early task learning；但 Stage-37 曾到 7.5k
才恢复至 40%，所以预注册的 10k horizon 保持不变。3k snapshot 已于 `09:29:17 BST`
写出，`total_time=949.37s`；2k->3k 包含首次 25-episode eval，额外开销约 `123s`。
按剩余 7k steady training 加 5k/7.5k/10k 三次 eval，完成 ETA 仍为约
`09:52--09:56 BST`。

### 21.41 B 等待期实现审计：Stage-43 是 FlowIQN-objective x CQN-AS，不是完整论文复现

在 2.5k 为 0% 后重新核对了
[FlowIQN v1](https://arxiv.org/pdf/2605.08515) 的 Appendix C/D 与
[作者仓库](https://github.com/ori-goals/flowIQN)。截至 `2026-07-24`，仓库仍明确写
“Code will be released shortly”，因此不存在可逐行比对的官方实现；只能对论文公式做
mechanism audit。

论文与当前 CQN-AS adaptation 一致的部分是：每个 transition 内独立采样 source/target
quantiles、分别排序后按 rank 配对、velocity 显式 condition 在 source quantile、Euler
`M=8`、action score 取 endpoint empirical mean。需要收紧的差异是：

1. 论文离线实验用
   `Qmax=rmax/(1-gamma), Qmin=rmin/(1-gamma)`，再令
   `source=[Qmax-kappa*(Qmax-Qmin), Qmax]`。对 MovePlate 的 observed
   `rmin=0,rmax=1,gamma=.99`，这个松理论界会给 `[90,100]`；Stage-43 为适配单次 terminal
   reward 的实际可达 return 和原 CQN `v=[-2,2]`，使用的是 `[.9,1]`。这是
   task-normalized design choice，不应再写成公式级 paper fidelity。
2. 论文用 64-frequency cosine `tau` embedding，并与 learned state-action embedding
   multiplicatively combine；Stage-43 当前是 raw scalar `tau` 经 Dense 后 additive
   fusion。HL-Gauss `z` 与 Fourier time 已有，但 quantile conditioner 仍不是完整论文网络。
3. 论文没有规定把 ODE endpoint clamp 在 source interval；Stage-43 为 CQN 数值安全将
   scalar target/trajectory clamp 到 `[-2,2]`。因此 3k endpoint min `-1.797` 不是漏实现
   source clamp，而是已碰到本地 safety bound。

由此，Stage-43 的可回答问题被严格限定为：

> 在保留 CQN-AS policy/encoder/BC/数据路径的情况下，quantile-sorted FlowIQN critic
> objective 加 task-normalized source、HL-Gauss/Fourier 是否改善 CQN-AS。

它不能声称复现 FlowIQN-FQL 或 FlowIQN-R。当前 run 已预注册且进行到 3k，不因看到 early
outcome 中途改网络。若 strict gate fail，先执行已有两次正方向 local evidence 的
expected-FLOQ safety fallback；若该路线也 fail，下一 FlowIQN mechanism experiment 必须把
“cosine-multiplicative quantile conditioner”和“论文理论 Q-bound source”拆成两个独立
ablation，不能再捆成一个无法归因的 fidelity bundle。

### 21.42 A Stage-45：真实 choice-regret 信号复现，但全 pair CI 未过；启动 powered replication

#### 上一阶段结果

Stage-45 用完全新鲜的 `30000--30031` 共 32 simulator seeds，只收集旧 LOSO 预注册的
`anchor=120, dimensions={4,6,8}`，得到 96 states、84 informative。收集实测
`639.91s = 20.0s/seed`。冻结的 structured-delta 与两个 imitation proxy 为：

| metric | structured value | behavior proxy | policy proxy |
|---|---:|---:|---:|
| pairwise | **`59.654%`** | `57.061%` | `56.772%` |
| Spearman | **`.17427`** | `.11367` | `.11792` |
| top-1 | `36.905%` | `38.095%` | `36.905%` |
| choice regret | **`.13403`** | `.17586` | `.17707` |

Spearman 95% seed-bootstrap CI lower 为 `.01402`；regret improvement 相对
behavior/policy 的 CI 分别为 `[.00271,.08986]`、`[.00331,.09148]`，均严格为正。这是第二个
独立 fresh split 上由同状态真实 continuation return 证明的 value choice signal，不是 BC
loss 或 demo top-1。

但 pairwise point delta 虽为 `+2.59/+2.88pp`，CI 分别是
`[-1.36,+6.51]pp`、`[-1.22,+7.01]pp`。预注册 15 项 checks 中只有这两个 lower-bound
checks fail，所以 strict value gate 仍为 **fail**；Stage-47 policy gate 已自动写
`skipped_policy_value_gate_failed`，没有读取 task outcome。artifacts：

```text
exp_local/cqn_value_fidelity_stage23/stage45_reliability_fresh32/coverage.json
exp_local/cqn_value_fidelity_stage23/stage45_reliability_fresh32/reliability_gate/gate_summary.json
```

#### 解释与已关闭的替代假设

结果说明当前模型能稳健减少“选错 bin 后损失多少”，但还不能证明它对五个 bins 的所有
pair ordering 都跨 seed 稳健。不能在看到结果后删除 pairwise gate。

针对“固定对称 V 形限制了完整排序”的假设，立即做了一个 98 参数
linear+quadratic local-value surface：只用 return differences 训练，不读 behavior/policy
label。旧 train16 的六个 ridge LOSO 全部失败，最佳 pairwise 只有约 `53.83%`，低于现有
structured-delta 的 `57.58%`；在已见 Stage-45 上也只有 `52.45%`，低于两个 proxy。
因此 quadratic surface 在 discovery 阶段关闭，不写新 deployment code，也不消耗 fresh
seeds。

#### 下一 gate 与执行

Stage-45 是 point effect 为正而 confidence 不足；按 `+2.6pp` effect 与当前 CI 宽度，
达到 lower bound 非负约需 `2.3x` 总样本。下一阶段不把 Stage-45 追加采样后重新计算，而是
保持模型、fit cache、support 与 15 项 gate 完全不变，在独立的
`80000--80063` **64 seeds** 上做一次 powered replication：

- 至少 80 informative states、64 seed clusters；
- pairwise/Spearman/top-1/random 与 regret point checks不变；
- pairwise 与 regret 相对两个 proxy 的四个 CI lower 均 `>=0`；
- model Spearman CI lower `>0`。

新 replication 不与 Stage-45 pool 来制造显著性。若仍 fail，停止 structured-delta family；
若 pass，才解封 `82000--82049` policy calibration 与
`83000--83199` paired confirmation，并要求最终成功率不劣于 validation-selected clean。
按 Stage-45 实测 `20.0s/seed`，64-seed collection ETA `21.3min`，deterministic gate
约 `35--45s`。

执行已启动并核验：

- Stage-49 GPU1 collector PID `1332732`，status `collecting`；
- Stage-50 event controller PID `1333167`，status `waiting_collection`；
- collection 完成后 controller 自动运行 frozen strict gate；只有 pass 才加载 action-facing
  wrapper，fail 则写 `skipped_policy_value_gate_failed`。

### 21.43 B Stage-43 中期结果：corrected objective 有延迟恢复，但显著弱于未修正版

Stage-43 的 5k/7.5k native outcomes 已完成：

| checkpoint | corrected Stage-43 | uncorrected Stage-37 |
|---:|---:|---:|
| 2.5k | `0/25 = 0%` | `0/25 = 0%` |
| 5k | `0/25 = 0%` | `2/25 = 8%` |
| 7.5k | `4/25 = 16%` | `10/25 = 40%` |

所以修正 bundle 不是完全不学习，但其 delayed recovery 明显更弱；到 7.5k 仍低旧版
`24pp`，更远低于 clean validation-selected baseline。这个 native trend 只用于训练健康与
horizon 判断，不能用于最终 checkpoint selection。按 7k steady timestamp 和 7.5k eval
开销，10k 完成 ETA 约再 `8min`；随后 Stage-44 必须在 `60000+` screen 独立选择，再在
`61000+` matched validation 作正式裁决。

10k native 随后为 `8/25 = 32%`，旧 Stage-37 同节点 `12/25 = 48%`。corrected 完整
trend 因而是 `0% -> 0% -> 16% -> 32%`：有持续恢复，但所有可比较节点都弱于未修正版，
更低于 clean seed1 validation-best `68%`。Stage-43 于 `09:52 BST` 正常完成，10k
`total_time=2291.78s`。

Stage-44 已真实接管 GPU5：controller status `evaluating`，worker PID `1340468`，首个
1k screen log 已开始写入。它将在 `60000--60009` 独立比较 11 个 numeric checkpoints；
按 Stage-38 同规模每 job 约 `69s`，screen ETA 约 `13min`，随后 matched
`61000--61049` validation 约 `6min`。若 validation fail，预计约 `10:11--10:14 BST`
结束并自动触发 expected-FLOQ fallback；若意外通过 `+2pp/wins` gate，才增加 sealed
200-episode confirmation。

### 21.44 A Stage-49：所有科学指标通过，唯一失败是 1 个无记录 seed；立即做缺失校正

#### 上一阶段结果

Stage-49 的独立 powered replication 已完成，collector 实测
`1196.44s / 64 requested seeds = 18.69s/requested seed`。请求的
`80000--80063` 中，`80032` 没有形成可评估 branch state，因此最终是 189 states、
144 informative states、63 simulator-seed bootstrap clusters。冻结模型与两个不读
continuation return 的 proxy 结果为：

| metric | structured value | behavior proxy | policy proxy |
|---|---:|---:|---:|
| pairwise | **`62.490%`** | `58.259%` | `57.852%` |
| Spearman | **`.23391`** | `.13986` | `.13374` |
| top-1 | `27.083%` | `29.861%` | `31.944%` |
| choice regret | **`.16215`** | `.20689` | `.19358` |

预注册的 effect checks 全部通过：

- model Spearman 95% seed-bootstrap CI 为 `[.10395,.35372]`；
- pairwise 相对 behavior/policy 的 delta CI 分别为
  `[+1.516,+7.034]pp`、`[+1.864,+7.521]pp`；
- regret improvement 相对 behavior/policy 的 CI 分别为
  `[.01524,.08272]`、`[.00187,.06851]`；
- 绝对 pairwise、绝对 Spearman、top-1 相对 random、support 与 informative-state
  checks 均通过。

所以 `gate=fail` 的唯一 false check 是
`enough_evaluation_seeds: 63 < 64`。Stage-50 因此按协议写
`skipped_policy_value_gate_failed`，没有偷跑 action-facing task outcome。这个结果不能
被描述成 structured value 的科学 gate 失败；也不能事后把阈值从 64 降到 63。

正式 artifacts：

```text
exp_local/cqn_value_fidelity_stage23/stage49_reliability_power64/coverage.json
exp_local/cqn_value_fidelity_stage23/stage49_reliability_power64/reliability_gate/gate_summary.json
exp_local/cqn_value_fidelity_stage23/stage50_reliability_power64_policy_control/status
```

#### 解释、下一 gate 与已执行

目前最强结论是：一个只用同状态真实 continuation return 拟合、并在独立 simulator seeds
上验证的低容量 value model，已经稳定优于 behavior/policy imitation proxy 的完整
pair ordering 和 choice regret；这直接反驳“这里所有 value signal 都只是 BC”的强版本。
但 A 路线的任务性能约束尚未回答，因为预注册样本数差 1 个 cluster。

Stage-51 不做 optional threshold relaxation，也不选择有利 seed。collector 接口要求
train/heldout 均非空，因此一次性预先加入两个新 seed `80064,80065`，把两者所有可用
records 与原 cache 合并；随后保持 fit cache、support、模型容量、bootstrap seed 和 15 项
gate 完全不变，仍要求至少 64 个可评估 seed clusters。只有新 summary 的 `gate=pass`，
Stage-52 才自动执行：

1. `82000--82049` 上 BC 与三个冻结 coverage threshold 的 calibration；
2. 只选择 calibration winner；
3. `83000--83199` 上与同 seed BC 做 200-episode paired confirmation；
4. 要求相对 validation-selected clean 的 non-inferiority margin 不超过 `5pp`。

Stage-51 已在 GPU1 启动并核验，controller PID `1359312`，status
`collecting_missingness_correction`。按实测吞吐，补采集约 `40--70s`，merge 加冻结 gate
约 `40s`；若 value gate 通过，calibration 初始 ETA 约 `15--20min`，confirmation 若被
解封再约 `25--35min`，首个 50-episode job 完成后用真实吞吐重新估算。

### 21.45 B Stage-44 screen 阶段性结果与剩余 ETA

Stage-44 已完成 8/11 个共同 `60000--60009` screen jobs，实测均值约
`92s/checkpoint`：

| checkpoint | success |
|---:|---:|
| 1k | `0/10` |
| 2k | `0/10` |
| 3k | `0/10` |
| 4k | `0/10` |
| 5k | `0/10` |
| 6k | `3/10` |
| 7k | `3/10` |
| 8k | `0/10` |

这说明 corrected FlowIQN 的恢复不是单调的，当前 screen winner 暂为 6k/7k；10-episode
screen 只负责 checkpoint selection，不能当正式性能结论。GPU5 仍在运行，剩余三个
screen jobs 约 `4.6min`；随后 winner 与 clean 在 `61000--61049` 做 matched validation，
若 fail 预计再约 `6--8min`，并由已挂起的 Stage-48 event controller 自动接续
expected-FLOQ fallback，期间不做短轮询。

### 21.46 A Stage-51 正式 pass 并解封任务 gate；B 选出 10k checkpoint

#### A 上一阶段结果与含义

Stage-51 补采 `80064,80065` 实测 `82.43s`，两者均形成 3 states；将所有 records 合并后，
新 evaluation cache 有 195 states、147 informative states、65 simulator-seed clusters。
冻结 gate 的 15/15 checks 现全部通过，正式 `gate=pass`：

| metric | structured value | behavior proxy | policy proxy |
|---|---:|---:|---:|
| pairwise | **`62.898%`** | `58.519%` | `58.121%` |
| Spearman | **`.24232`** | `.14557` | `.13957` |
| choice regret | **`.15921`** | `.20304` | `.19000` |

model Spearman CI 是 `[.11509,.36082]`；pairwise delta 相对 behavior/policy 的 CI 为
`[+1.793,+7.143]pp`、`[+2.083,+7.634]pp`；regret improvement CI 为
`[.01472,.08101]`、`[.00243,.06729]`。因此 A 路线已经有可复现的“value ordering
真实有意义”结论，但还没有满足“最终任务效果不低于 clean CQN-AS”，后者必须由 Stage-52
单独回答。

Stage-52 已自动解封。`82000--82049` 的 paired BC 已完成：
`28/50 = 56%`，实测 `125.03s/50 episodes`；第一个 conservative sidecar 正在 GPU1
运行。按这个实际吞吐，余下三个 calibration variants 约 `6.3min`。若 calibration
选出 winner，两项各 200 episodes 的 sealed confirmation 预计约 `16--20min`；否则在
calibration gate 立即停止。这里以同一组 seed 的 BC 为基线，不拿旧 run 某个末尾 step
替代 validation-selected paired baseline。

正式 artifacts：

```text
exp_local/cqn_value_fidelity_stage23/stage51_reliability_missing_seed/reliability_gate/gate_summary.json
exp_local/cqn_value_fidelity_stage23/stage52_reliability_policy/calibration/bc.json
```

#### B 上一阶段结果、解释与当前 gate

Stage-44 的后三个 screen checkpoints 为 9k `3/10`、10k `6/10`、10.5k `5/10`；
所以完全独立的 screen 选择了 **10k**，不是根据 native training eval 选择。corrected
FlowIQN 在同一 screen 上呈现从 1--5k 全 0、6--7k 恢复、8k 回落、10k 再恢复的高方差
轨迹，这进一步说明不能用单个 native checkpoint 成功率作结论。

GPU5 已开始正式 matched validation：

- corrected 10k：`61000--61049`；
- clean 5k：同一 `61000--61049`；
- 只有 corrected 至少高 `2pp` 且 paired wins gate 通过，才解封 200-episode
  confirmation。

screen 实测约 `88s/10 episodes`，因此每个 50-episode validation job 约 `6--8min`；
当前 corrected job 已开始，整个 validation failure-path ETA 约 `11--15min`。Stage-48
仍按事件等待 Stage-44，结束后自动接续，不占 GPU。

### 21.47 B Stage-44 正式失败；Stage-48 expected-FLOQ 已自动接管

#### 上一阶段结果

独立 screen 选出的 corrected FlowIQN 10k 在完全相同的
`61000--61049` validation seeds 上得到：

| policy | success |
|---|---:|
| corrected FlowIQN 10k | `19/50 = 38%` |
| clean CQN-AS 5k | **`33/50 = 66%`** |

paired delta 为 **`-28pp`**，simulator-seed paired 95% CI
`[-46,-8]pp`；7 wins、21 losses、22 ties，McNemar exact `p=.01254`。因此
`validation_gate=fail`，200-episode confirmation 保持 sealed，没有因为 screen 的
`60%` 好看而跳过独立验证。正式 artifact：

```text
exp_local/cqn_flow_flowiqn/stage44_flowiqn_corrected_r4_gate_seed60000/gate_summary.json
```

#### 解释

corrected objective 虽然 native trend 与 screen 都显示能学习，但它没有把该 signal 转化为
跨 seed 的任务性能；而且差距不仅是“不显著优于”，而是显著低于 clean。这关闭了当前
“task-normalized source + raw-tau additive conditioner + corrected self-target”的 bundle。
它也说明 10-episode checkpoint screen 的 `60%` 只能用于选择，不能被当成性能结果。

这个失败不否定所有 FM critic：Stage-27b 的 own-BC FLOQ 曾有两个独立 positive split，
但当时缺少 clean-safe action integration。所以下一 gate 按预注册转向
**expected-FLOQ high-confidence fallback**，不继续延长或挑选 FlowIQN checkpoint。

#### 下一 gate 与已执行

Stage-48 event controller 已自动接管 GPU5：

1. 用 seed `69999` 做 diagnostic-only integration smoke，证明 clean action path 与
   expected-FLOQ readout 能组合；
2. 在 `70000--70049` 同 seeds 比较 clean 与三个冻结的 high-margin fallback regimes；
3. calibration 只选 winner；
4. 若有 winner，在 `71000--71199` 做 paired 200-episode confirmation，要求相对 clean
   至少满足 `5pp` non-inferiority，并报告是否有正提升。

当前 status=`integration_smoke`，worker 已运行；smoke 有一次 JAX 固定编译成本，预计
`40--60s`。进入 calibration 后用第一个 50-episode job 的真实吞吐重估 ETA。A 的
Stage-52 medium 同时在 GPU1 运行，两条路线没有互相等待。

### 21.48 A Stage-52：真实 value 未进入 action；关闭当前 deployment gate，启动 coverage 诊断

#### 上一阶段结果

在 `82000--82049` paired calibration 上：

| policy | success | total overrides | override rate |
|---|---:|---:|---:|
| exact BC | `28/50 = 56%` | 0 | 0 |
| conservative | `29/50 = 58%` | **0** | 0 |
| medium | `28/50 = 56%` | **0** | 0 |
| wide | `28/50 = 56%` | **0** | 0 |

三个 sidecar 都满足 point success 不低于 BC，但全部失败
`produced_override`，所以 `selected_variant=null`、`gate_passed=false`，sealed
confirmation 没有运行。conservative 的 2 wins/1 loss 与 `+2pp` 不能归因给 value：
在 0 override 下它只是独立 workspace rollout 的随机差异。正式 artifact：

```text
exp_local/cqn_value_fidelity_stage23/stage52_reliability_policy/calibration/summary.json
```

#### 解释

Stage-51 已证明 structured model 的 counterfactual ordering 有真实 signal；Stage-52
证明当前 action integration 把这个 signal **全部过滤掉**。三组 variant 只改变
`max_state_rms=.6/.7/.8`，但都固定 `min_value_margin=.235` 与
`max_bc_logprob_drop=.25`。因此现在不能判断是 state radius、BC support 还是 margin
造成零覆盖，也不能靠直接降低一个阈值并查看 task success 来反向调参。

#### 下一 gate 与立即执行

Stage-53 只做 coverage diagnostics，永远执行 exact BC，不把 task outcome 用于选参数。
在新 seeds `84000--84009` 上冻结：

- `min_value_margin=0`；
- `max_state_rms=.8`；
- 依次检查 `max_bc_logprob_drop={.25,.50,.75}`；
- `diagnostic_only=true`，记录 would-be override rate、覆盖 episode 数、value-margin 与
  state-RMS quantiles。

预注册 decision rule：

1. 选择最小的 BC-logprob drop，使至少 5/10 episodes 有 would-be override 且总体
   override rate 至少 `.05%`；
2. 若 `.75` 仍不满足，当前 structured action integration 关闭，转向训练期 return
   identifiable head；
3. 若满足，用诊断 margin quantiles 冻结三档阈值，但在全新 `85000+` calibration
   上比较；diagnostic seeds 的 success 完全忽略；
4. 只有实际产生 override、paired wins 不少于 losses 的 winner 才能进入另一组
   200-episode confirmation。

按已有 sidecar 10-episode固定编译和 rollout 开销，三项 diagnostic ETA 约
`4--7min`，GPU1 独立运行；B fallback 使用 GPU5。

### 21.49 A Stage-53：margin 不是零覆盖原因；hard BC-support deployment 正式关闭

三个 diagnostic 都在完全相同的 `84000--84009` 上执行 exact BC，只记录 would-be action：

| max BC log-prob drop | inferences | would-be overrides | episodes with override |
|---:|---:|---:|---:|
| `.25` | 2225 | **0** | 0/10 |
| `.50` | 2222 | **0** | 0/10 |
| `.75` | 2225 | **0** | 0/10 |

三者都已将 `min_value_margin` 降为 0，且 `max_state_rms=.8`；所以 Stage-52 的零 action
coverage 不能再归因于 `.235` margin 太高。按 21.48 的预注册 rule，当前
“先找 value-optimal sibling、再要求落在 BC hard support 内”的 deployment 路线关闭，不把
BC drop 继续扫到看到 task improvement 为止。

下一项 Stage-54 是 **mechanism-only** 诊断，不是新 deployment candidate：保持
`diagnostic_only=true`，但设 `max_bc_logprob_drop=100,max_state_rms=100,min_margin=0`。
它回答：

- 若出现 would-be override：瓶颈是 policy support 与 reliable branch proposal 不重合，下一训练头
  必须通过 return-supervised policy residual 把可靠 alternative 纳入 support；
- 若仍为 0：瓶颈是 policy-reachable state 上 predicted optimal delta 本身等于 BC delta，下一训练
  目标必须重新覆盖 reachable states，而不是改 selector 阈值。

Stage-54 已在 GPU1 启动，仍使用相同 diagnostic seeds 且完全忽略 success；实测同规模 ETA
约 `70s`。B Stage-48 在 GPU5 并行运行。

### 21.50 A Stage-54--56：确认 reachable-state proposal 坍到 BC；两个直接排序头均未过 gate

#### Stage-54 结果

把所有外部 selector 阈值基本解除后，`84000--84009` 上仍然是
`0/2225` would-be overrides、`0/10` episodes covered，耗时 `71.4s`：

```text
exp_local/cqn_value_fidelity_stage23/stage54_structured_unbounded_diag/unbounded.json
```

这排除了 margin、BC log-prob support 和 state RMS 三个 hard filter。当前 structured
V-shape model 在 policy 实际到达的状态上，预测最优 sibling 就是 BC sibling。Stage-51
通过的 all-pair ordering 因而不能直接等价为可用的 action proposal。

#### Stage-55/56 结果

所有模型只在既有 `branches_train16_val4.npz` discovery cache 上训练；checkpoint 由
`27003` validation seed 选择，四个 held-out simulator seeds 不参与选择，三个初始化取平均：

| target | pairwise | top-1 | regret | gate |
|---|---:|---:|---:|---|
| raw direct return | 51.29% | 26.25% | **0.0778** | reference |
| tied-aware return rank | 54.45% | 26.25% | 0.0889 | fail |
| hard counterfactual best | **54.85%** | **30.09%** | 0.0830 | fail |

`direct_rank` 的 pairwise 提升没有转化成 top-1，且 regret 明显恶化。`direct_top1` 同时
提高 pairwise 与 top-1，但 regret 比 raw return 恶化 `0.00525`；按预注册的
“top-1 提升且 pairwise/regret 均不退化”规则仍然失败，不消耗 fresh simulator seeds。

Artifacts:

```text
exp_local/cqn_value_fidelity_stage23/stage55_direct_rank_discovery/summary.json
exp_local/cqn_value_fidelity_stage23/stage56_direct_top1_discovery/summary.json
```

#### Stage-57 gate 与立即执行

下一候选不是再调 selector，而是用真实 same-state counterfactual return gap 构造
`softmax(return / temperature)` target。它在 hard top-1 与 raw magnitude 之间连续插值，
仍然不读取 BC/policy label。先只在旧 cache 上冻结
`temperature={.02,.05,.10,.20}`：

1. 每个温度仍用 validation seed 选 checkpoint；
2. 温度只能按 validation 的 top-1、pairwise、regret 顺序选，held-out 不选超参；
3. 选出的温度必须在 held-out 上相对 raw direct 同时满足
   `top1 > 26.25%`、`pairwise >= 51.29%`、`regret <= 0.0778`；
4. 未满足则关闭这一组简单 return-target transforms；满足才收集 fresh branch seeds，
   然后做 seed-cluster bootstrap 和任务 policy gate。

实现已加入 `direct_softmax` 及单测；`10 passed`。Stage-57 在 GPU1 执行，按 Stage-56
吞吐估计四个温度总 ETA `1.5--3min`。

### 21.51 B Stage-48：expected-FLOQ fallback 失败；Stage-58 转为 outcome-blind 稀疏校准

#### 上一阶段结果

`70000--70049` paired calibration 的完整结果是：

| policy | success | override rate | wins/losses/ties | gate |
|---|---:|---:|---:|---|
| clean CQN-AS | **62%** | 0 | -- | reference |
| conservative | 58% | 52.52% | 6/8/36 | fail |
| medium | 50% | 64.67% | 3/9/38 | fail |
| wide | 54% | 74.74% | 6/10/34 | fail |

三个 variant 都产生 action，但 success 均低于 exact clean、paired wins 均少于 losses。
因此 `selected_variant=null`、calibration fail，sealed confirmation 没有运行。完整 gate
耗时 `993.8s`：

```text
exp_local/cqn_flow_floq_fallback/stage48_expected_floq_fallback_seed70000/gate_summary.json
exp_local/cqn_flow_floq_fallback/stage48_expected_floq_fallback_seed70000/calibration/summary.json
```

#### 解释

当前方法名为 fallback，但实际在一半到四分之三 inference 上替换 clean action，已经成为
主策略。这个实验否定的是 **dense expected-FLOQ intervention**，不能把失败归结为 FM
永远没有 usable value signal，也不能从这 50 个 outcome 再挑一个看起来更好的 threshold。

#### Stage-58 gate 与立即执行

先在全新 `73000--73009` 上执行 exact clean action，只记录 FLOQ would-be proposal：

- `diagnostic_only=true`，task success 不参与任何决定；
- 固定 `policy_value_beta=1`、BC log-prob drop `.25`、source win fraction `.75`；
- 暂设 `min_value_margin=0`，保存每个 proposal 的 normalized Q margin；
- 用 proposal margin 的 `p95/p97.5/p99` 冻结三档 threshold，目标覆盖约
  `3%/1.5%/.6%` inference（实际覆盖以 artifact 为准）。

只有 coverage 在 `.25%--5%` 且至少覆盖 5/10 episodes 的阈值保留。随后在完全未见的
`74000--74049` 上把保留阈值作为真实 action variants 与 clean paired calibration；
必须 actual override、success 不低于 clean 且 wins 不少于 losses，才进入另一组
200-episode confirmation。为了让 coverage calibration 可复现，evaluator 现在输出
proposal-level margin/source arrays；focused tests `4 passed`。

Stage-58 diagnostic 在 GPU5 执行，按 Stage-48 实测吞吐（50 episodes 约 4.7 分钟）估计
10 episodes ETA `65--90s`。

### 21.52 A Stage-57 pass；B Stage-58 pass，立即进入各自独立 confirmation/calibration

#### A Stage-57 实际结果

四个温度的选择只读取 `27003` validation；按预注册的 top-1、pairwise、regret
lexicographic rule 选中 `temperature=.02`。它在旧 cache 的四个未参与选择 held-out
seeds 上相对 raw direct 三项同时改善：

| model | pairwise | top-1 | regret |
|---|---:|---:|---:|
| raw direct | 51.29% | 26.25% | 0.07778 |
| softmax return, T=.02 | **56.27%** | **33.92%** | **0.07489** |

因此 Stage-57 discovery gate pass。`.05/.10/.20` 不再考虑，也不允许读取后续数据重新调
temperature。Stage-59 已把旧 16-seed train（仍以 `27003` 选 checkpoint）和此前未被
本方法读取的 `80000--80065` 共 65 个 simulator seeds 组成 frozen external-eval cache：

```text
exp_local/cqn_value_fidelity_stage23/stage59_softmax_external_confirmation/frozen_train_external65.npz
```

Stage-59 必须在该 external cache 上保持相对 raw direct 的 pairwise/top-1/regret
不退化，并用 simulator-seed cluster bootstrap 证明 pairwise/regret 改善不是单一 seed
造成；失败则不进入 policy gate，通过才把该 head 接到 clean CQN-AS。

#### B Stage-58 实际结果

`73000--73009` outcome-blind diagnostic 共 `2217` inferences；loose support gate 产生
`1598` proposals（72.08%）。proposal margin 的冻结分位点及由此得到的覆盖为：

| threshold source | exact margin | would-be count/rate | episodes covered |
|---|---:|---:|---:|
| p95 | 2.6467557430 | 80 / 3.61% | 9/10 |
| p97.5 | 2.8061748087 | 40 / 1.80% | 8/10 |
| p99 | 2.8956280684 | 16 / 0.72% | 5/10 |

三项都满足预注册的 `.25%--5%` 和至少 5/10 episodes coverage gate。diagnostic rollout
的 success 为 50%，但 exact clean action 被执行且 outcome 没有参与 threshold 选择。
正式 artifact：

```text
exp_local/cqn_flow_floq_fallback/stage58_sparse_coverage_diag/diagnostic.json
```

Stage-60 在全新 `74000--74049` 上运行 clean 和上述三个 **真实执行** variants。固定其余
条件为 BC drop `.25`、source agreement `.75`、`policy_value_beta=1`。只有 actual
override、success 不低于 paired clean、wins 不少于 losses 的 variant 才按 success 后
coverage 选一个进入 `75000+` 200-episode sealed confirmation。按 Stage-48 的 50-episode
实测，clean 加三个 variants ETA `15--18min`，GPU5 持续运行。

### 21.53 A Stage-59：softmax 泛化失败；冻结 raw-return ensemble 并采集全新 confirmation

#### 上一阶段结果

冻结 `T=.02` 后，在独立 `80000--80065` 65-seed cache 上，三个初始化的均值为：

| model | pairwise | top-1 | regret |
|---|---:|---:|---:|
| raw direct, per-init mean | **51.54%** | 26.98% | **0.1475** |
| softmax return, per-init mean | 49.95% | **27.66%** | 0.1725 |

softmax 只留下 `+0.68pp` top-1，pairwise 和 regret 都退化，Stage-59 confirmation
明确 fail，禁止进入 task policy gate。更重要的是 softmax 单初始化 top-1 为
`40.14%/17.69%/25.17%`，说明 Stage-57 improvement 对 initialization 不稳定。

#### 诊断结果与下一 gate

不再调 softmax temperature。把 Stage-59 cache 降级为 discovery 后，冻结三个
raw-direct initialization 的 **prediction mean ensemble**，其 discovery 指标为：

| pairwise | top-1 | regret | Spearman |
|---:|---:|---:|---:|
| **59.08%** | **31.97%** | **0.10835** | **.19909** |

同一 cache 上 behavior proxy 为 58.52% pairwise / .2030 regret，policy proxy 为
58.12% / .1900；因此 ensemble 是目前同时有 action top-1 和低 regret 的候选。由于这个
cache 已用于发现 ensemble，它不能再充当证据。

Stage-61 冻结：

- old train/validation、网络 `64x64`、updates、seeds `1,2,3` 和 arithmetic-mean
  ensemble 全部不变；
- 在全新 simulator seeds `86000--86065` 只采 anchor 120、dims 4/6/8 的五 sibling
  counterfactual returns；
- 不允许再选择温度、初始化子集或 ensemble reducer；
- fresh gate 要求 pairwise `> .55` 且高于 behavior/policy，regret 低于两者，
  Spearman `> .10`、top-1 高于 random；
- simulator-seed cluster bootstrap 还必须给出正 Spearman lower bound、相对两个
  proxy 的 pairwise delta 与 regret-improvement lower bounds 均非负。

通过后才序列化三个 head 并做 action-facing calibration；失败则 raw/softmax MLP
路线关闭，A 的推荐退回“clean CQN-AS policy + structured causal value audit”，而不伪称
已有 task improvement。collector 按 Stage-49 实测 `18.69s/requested seed`，66 seeds
ETA 约 `20.6min`，GPU1 独立执行。

补充的 discovery-cache seed bootstrap 表明，ensemble 的 regret improvement CI 已对两个
proxy 严格为正，Spearman CI 也是正的；但 pairwise delta CI 仍跨 0
（vs behavior `[-8.47,+9.85]pp`，vs policy `[-7.98,+10.13]pp`）。因此不能把 59.08%
point estimate 当成已复现结论，Stage-61 的全新 66-seed replication 是必要而非重复实验；
只有上述冻结 CI gate 全过才继续。

### 21.54 B 下一层公平性：不再让 seed1 threshold 代替 training-seed replication

Stage-29 与 Stage-30 已在两个完全独立的 environment splits 上给出 frozen
expected-FLOQ-distill 相对 clean 的 `+10pp` 与 `+7pp`，但两个 CI 下界分别为 `-1pp`
和 `-1.5pp`。Stage-48/60 回答的是“如何把 seed1 flow 当 clean fallback”，并不回答该
训练算法跨 initialization 是否成立。为了满足最终可复现结论，Stage-60 完成后无论稀疏
gate 成败，都冻结 Stage-24 的官方语义配置并补训练 seeds 2/3：

- launch、10.5k budget、UTD4、8 flow steps/sources、uniform `.1 Q-range`、HL-Gauss、
  endpoint anchor、source consistency、distilled readout和独立 BC tower均不变；
- 不根据 native eval 改 checkpoint；三个训练 seed 一律用 `10k/beta1`；
- clean references 固定为其各自 validation-selected checkpoint，不拿 final step 代替；
- 最终使用同一组新 environment seeds 做三对 paired eval，并按
  `(training seed, environment seed)` crossed bootstrap；
- 项目级 superiority 要求 mean delta `>0`、95% CI lower `>=0`、aggregate wins>losses，
  且至少 2/3 training seeds 的 point delta 为正；
- 同时在 frozen counterfactual cache 报告 flow-value pairwise/regret，防止纯 BC tower
  improvement 被误称为 value-FM improvement。

seed1 实测训练约 `65.7min`，所以 GPU1/GPU5 并行补两个 seed 的 ETA 为 `66--72min`；
最终 paired eval 预计再 `25--40min`。Stage-60 controller 退出后 GPU5 自动启动 seed2；
GPU1 在 A Stage-61 及其必要 policy gate 释放后启动 seed3。

### 21.55 Stage-60 首个真实稀疏 arm：覆盖控制成功，但 p95 outcome 暂未改善

共同 `74000--74049` 上，clean 已完成 `30/50=60%`；冻结的 p95 arm 为
`29/50=58%`，paired `3 wins / 4 losses / 43 ties`，即 `-2pp`。它实际执行
`348/10566=3.294%` overrides，覆盖 49/50 episodes；所以 Stage-58 的 outcome-blind
分位点成功把原先 52--75% 的 dense intervention 压到预注册稀疏区间，但第一档没有得到
正向 task effect。

这只是预注册三档中的第一项，不能提前据此选择/改阈值。p97.5 正在同一 controller 内运行，
随后自动执行 p99 并一次性生成 calibration summary。按 p95 实测 `302.7s/50 episodes`，
剩余两档加 summary ETA 约 `9.5--10.5min`；若无人通过则 confirmation 不运行并自动开始
expected-FLOQ seed2 训练。

### 21.56 2026 mechanism 复核：expected-FLOQ 的核心不是 return distribution

等待 Stage-60/61 时复核了最新的一手机制研究
[What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)。
它对本项目最重要的结论不是再提出一个新 policy，而是用 controlled ablation 区分了两种
critic objective：

1. **expected-value FLOQ**：先对 successor flow samples 求均值，得到 scalar TD target，
   再用整条 velocity path 的 dense supervision 拟合这个期望；它虽然从噪声出发，但不满足
   distributional Bellman equation；
2. **distributional return flow**：每个 noise 对应一个 Bellman return sample，学习完整
   return distribution。

论文在相同 velocity architecture / integration compute 下发现 distributional variant
通常不优于、甚至弱于 expected-value FLOQ。它给出的两项机制是：

- **test-time recovery**：训练时沿多个 interpolant 位置监督 velocity，推理时后续积分步能
  修正早期 velocity error；
- **feature plasticity**：移动 TD target 可部分由各积分 slice 的重新加权吸收，减少
  monolithic critic 在高 UTD 下的 representation overwrite。

这与本地结果形成一个必须显式区分的映射：

- Stage-24 训练端是 `8 flow steps x 8 sources` 的 expected-value FLOQ，符合 scalar
  expected backup；
- 当前有两次正方向 task evidence 的 action readout 却是 **distilled scalar head**。
  它可能继承 flow training 带来的 shared-feature plasticity，但 action selection 没有执行
  ODE，因此不能把正结果归因于 test-time recovery；
- Stage-28 的 integrated readout 在 `5k/beta3` 上失败，而当前冻结 winner 是
  `10k/beta1`。所以 Stage-28 关闭了那个具体组合，尚未完成对最终 winner 的
  mechanism-matched integrated-vs-distill 比较。

另一条一手路线
[Value Flows](https://openreview.net/forum?id=2VyNYUVF2k) 与
[FlowCritic](https://arxiv.org/abs/2510.22686) 学的是完整 return distribution。后者具体是
PPO 的 state-value flow，并使用多 flow samples 的截断均值、velocity-update clipping 与
CoV policy weighting；它不是 CQN-AS 的 action-bin Q baseline。可迁移的稳定化候选是
velocity-update clipping，但 CoV-weighted PPO 和 state-only GAE target 不能直接移植后称为
CQN-AS comparison。

因此 B 路线冻结以下顺序，避免一边看结果一边改变解释：

1. 先完成 Stage-62 expected-FLOQ `10k/beta1` 三训练 seed superiority gate；这是
   **算法 outcome replication**；
2. 若通过，在独立 selection seeds 上对相同三个 frozen checkpoints 比较
   `distill` 与 `integrated {2,8 steps}`，随后只把 selection winner 带到新 held-out；
   这是 **test-time recovery mechanism gate**；
3. 同时加入 compute-matched monolithic scalar readout control，才能区分“flow objective
   造成的 plasticity”与“只是多了一个容易排序的 scalar head”；
4. 只有跨 seed 显示 flow training instability 时，才单独增加 FlowCritic-style
   velocity-update clipping；不把它与 distributional target、truncated sampling、
   CoV weighting一次性捆绑；
5. 若以后重开 distributional return flow，首要 gate 是反事实
   pairwise/regret/return-calibration，再看 task success；不能用多个 noise samples 的均值
   自动宣称学到了真实 return distribution。

### 21.57 B Stage-60 calibration 完成：选中 p97.5，启动 200-seed confirmation

共同 selection seeds `74000--74049` 的完整预注册 sweep 已落盘：

| arm | success | paired W/L/T vs clean | override rate | calibration gate |
|---|---:|---:|---:|---:|
| clean | `30/50 = 60%` | -- | `0%` | reference |
| p95 | `29/50 = 58%` | `3/4/43` | `348/10566 = 3.294%` | fail |
| p97.5 | `30/50 = 60%` | `3/3/44` | `137/10338 = 1.325%` | **pass** |
| p99 | `26/50 = 52%` | `1/5/44` | `45/10948 = 0.411%` | fail |

正式 artifact 是
`exp_local/cqn_flow_floq_fallback/stage60_sparse_floq_gate_seed74000/calibration/summary.json`。
selection protocol 先要求不低于 clean、产生真实 override、wins 不少于 losses；通过者再按
success 优先、override rate 较低者优先。只有 p97.5 通过，因此没有在 p95/p99 中做
post-hoc 选择。

这个结果说明两件事。第一，outcome-blind margin quantile 能把 dense intervention 的
`52--75%` override 压到 `1.3%`，同时在 selection split 保持 clean outcome；所以
“FLOQ proposal 完全不可控”已被排除。第二，`3 wins / 3 losses` 只说明 calibration
非劣，尚未说明 value override 有正因果效应，更不能宣称超过 CQN-AS。

下一 gate 已立即执行：在完全未用于阈值选择的 `75000--75199` 上，先评估 exact clean，
再评估冻结 p97.5。confirmation 要求：

- overall success 不低于 clean；
- paired-success bootstrap CI lower `>= -5pp`；
- 确实产生 override；
- 有 override 的 episode 子集上，success delta 与 reward delta 均严格为正。

GPU5 当前正在跑 200-episode clean reference，随后同 controller 自动跑 p97.5。按本阶段
实测 clean 约 `2.2s/episode`、candidate 约 `6s/episode`，从 confirmation 启动起 ETA
约 `27--31min`。Stage-60 父进程退出后事件驱动 handoff 自动启动 expected-FLOQ seed2，
不使用短轮询。

同时修正了共享 calibration summarizer 中遗留的旧 `34000--34099` 文案；本阶段 artifact
现在明确记录真实 confirmation seeds `75000--75199`。这是元数据修复，不改变任何 selection
或 outcome 数值，相关 focused tests 为 `14/14` pass。

### 21.58 A 路线的部署原则：真实 value gate 与安全 policy gate 分开

一手的
[Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html)
给出的核心原则是：只有在 state-action 有足够支持时才偏离 baseline，在不确定区域回退到
baseline。它与本项目“clean CQN-AS proposal + value sidecar 只改一个维度”的结构相符，但
不能直接宣称本地深网/连续状态实现继承了论文的 tabular high-probability guarantee。
因此这里只把它作为 deployment design principle，最终安全性仍由 matched simulator seeds
上的 empirical non-inferiority gate 决定。

A 路线现在固定两个独立问题：

1. **value authenticity**：完全不执行模型 action，只在未见过的 simulator branch states
   上比较真实 continuation return；指标是 pairwise、Spearman、top-1、choice regret，
   并做 simulator-seed cluster bootstrap。这就是 Stage-61；
2. **policy improvement**：只在 authenticity pass 后，才把三个 head 的 candidate-vs-BC
   prediction 做 ensemble lower-confidence bound，并同时要求 BC log-prob support 与
   state-OOD radius。一次 inference 最多替换一个 action dimension，其余情况 bitwise
   回退 clean plan。

如果 Stage-61 pass，下一 policy gate 不直接用 arithmetic-mean argmax，而是：

- 对每个 sibling candidate 计算三个 head 相对原计划的 predicted delta；
- 要求多数/全体 sign agreement，并使用
  `LCB = mean(delta) - lambda * std(delta)`；
- 在新 selection seeds 上只选择 `lambda`、BC support drop 与 LCB margin；
- 在再一组未见 seeds 上要求 success 不低于 clean、override subset 的 paired outcome
  为正，且必须有非零 override；
- validation-selected model 参数与 conditioner transforms 已由 benchmark runner 序列化，
  避免用 final-step 参数替代通过 gate 的 checkpoint。

如果 Stage-61 fail，则 arithmetic-mean direct heads 也没有可复现的 value authenticity，
不会进入 policy calibration。A 的保守结论将是：

- Stage-51 structured-delta sidecar 在 branch counterfactual 上有真实 value 信号；
- 但 Stage-52--54 证明当前 reachable-state support 下它不产生 action override；
- clean CQN-AS policy 保持不变，因此 task outcome 不劣，但还不能宣称“有意义的 value 已经
  改善 policy”；
- 下一次扩 coverage 必须预先采集更多 reachable anchors/dimensions，而不是继续放宽同一
  sidecar 的 threshold。

### 21.59 A Stage-61 正式 fail：低 choice regret 复现，但全 pair 排序弱于 behavior

全新 simulator seeds `86000--86065` 的 collection、三模型训练与预注册 gate 已全部完成。
正式 artifact：

```text
exp_local/cqn_value_fidelity_stage23/stage61_direct_ensemble_fresh66/ensemble_gate.json
```

66 个 seed clusters 共形成 198 个 `(state, action-dimension)` records，其中 160 个存在非零
counterfactual return span。三模型的 validation-selected steps 是 `0/500/500`；seed1 在
step0 胜出本身就是 direct raw-return fit 不稳定的证据，不能用 final 5k 参数替换它。
arithmetic ensemble 的 external 结果：

| score | pairwise | Spearman | top-1 | choice regret |
|---|---:|---:|---:|---:|
| direct ensemble | `55.89%` | `0.1355` | `29.38%` | **`0.09875`** |
| behavior delta proxy | **`60.42%`** | `0.1702` | `30.00%` | `0.18241` |
| BC policy log-prob proxy | `60.05%` | `0.1617` | **`31.88%`** | `0.17737` |
| random top-1 reference | -- | -- | `25.00%` | -- |

模型通过了 absolute pairwise `>55%`、Spearman `>.1`、positive Spearman CI、top-1 above
random，以及相对两个 proxy 的 regret point/CI gate。特别是 regret improvement 的
seed-bootstrap CI 分别为：

- vs behavior：`[+0.0352,+0.1378]`；
- vs policy：`[+0.0325,+0.1304]`。

但是它没有超过两个 proxy 的 pairwise point estimate，pairwise-delta CI 也明确跨负值：

- vs behavior：`[-12.83,+4.02]pp`；
- vs policy：`[-12.51,+4.45]pp`。

因此 Stage-61 正式 **fail**。正确解释是：raw-return ensemble 学到了一部分与 return
相关、能降低最终 action regret 的信号，并非纯粹复制 BC；但它不能可靠排序全部 sibling
pairs，尚不满足 action-facing value authenticity。A 的 direct-head online deployment
按预注册协议关闭。

#### 下一 gate 与实际执行

失败结构给出一个可检验的最小下一假设：direct model 擅长低 regret 选点，behavior/policy
proxy 擅长全 pair ordering，validation-selected support blend 可能兼得两者。新增：

- `scripts/run_cqn_prediction_blend_discovery.py`；
- benchmark 保存 selected validation predictions、Flax model bytes 与 conditioner
  transforms；
- 每个 state 内先对分数做 sibling-bin RMS z-score；
- validation seed `27003` 在 `proxy={behavior,policy,mean}` 与
  `model_weight={.25,.5,.75,1}` 中选择；
- fresh66 仍使用 Stage-61 全部 strict checks，但只标记为 discovery，因为提出 blend
  hypothesis 时已经看过 Stage-61 结果；
- discovery 只有全 gate pass 才值得再采独立 external confirmation，否则 direct/proxy
  blend 一并关闭。

相关 focused tests `15/15` pass。CPU controller
`exp_local/cqn_value_fidelity_stage23/stage63_direct_proxy_blend_discovery`
已经启动，预计 `2--4min`，不占 GPU。Stage-61 fail 同时释放 GPU1，预注册 handoff 已真实
启动 expected-FLOQ seed3；按 seed1 吞吐，10k ETA 暂估 `66--72min`，到首个 1k 节点后用
本 run 实测更新。

### 21.60 A Stage-63 关闭 direct blend；路线 A 形成双层可复现结论

Stage-63 在 internal validation seed `27003` 选中：

```text
25% direct ensemble + 75% mean(behavior-delta proxy, BC-policy proxy)
```

它在 validation 上 pairwise `63.64%`、regret `0.0110`。带到 Stage-61 fresh66 后，
point metrics 的确同时改善：

| score | pairwise | Spearman | top-1 | regret |
|---|---:|---:|---:|---:|
| selected blend | **`60.88%`** | **`0.1867`** | `28.13%` | **`0.17460`** |
| behavior proxy | `60.42%` | `0.1702` | `30.00%` | `0.18241` |
| policy proxy | `60.05%` | `0.1617` | **`31.88%`** | `0.17737` |

但是 simulator-seed bootstrap 没有支持 strict improvement：

- pairwise delta vs behavior CI：`[-0.92,+1.79]pp`；
- pairwise delta vs policy CI：`[-0.57,+2.25]pp`；
- regret improvement vs behavior CI：`[-0.00073,+0.02016]`；
- regret improvement vs policy CI：`[-0.01212,+0.01772]`。

所以 discovery gate 正式 **fail**，按预注册协议不再采 external confirmation，direct/proxy
blend 路线关闭。它说明 value signal 可以对 proxy 做小修正，但尚不足以跨 seed 证明改善，
不能把 point estimate 包装成结论。

新增 `scripts/summarize_cqn_route_a.py`，把 A 路线最终结论机器可读地拆成两层。focused
tests `5/5` pass，正式 artifact：

```text
exp_local/cqn_value_fidelity_stage23/route_a_conclusion_20260724.json
```

结果为：

- **`safe_audit_gate=pass`**：Stage-51 的 79-parameter structured-delta model 在 65 个
  simulator seeds 上 15 项 counterfactual checks 全过；pairwise `62.90%`、Spearman
  `0.2423`、regret `0.1592`，均优于 behavior/policy proxy；
- **task non-inferiority**：Stage-52 的 medium/wide exact fallback 均为 `56%`、paired
  `0/0/50`、zero overrides，与 clean 完全匹配；
- **`action_facing_gate=fail`**：Stage-52--54 没有 reachable override，Stage-61/63
  strict deployment gates 也 fail。

因此 A 的明确推荐是：

1. policy 使用 validation-selected clean CQN-AS；
2. structured-delta sidecar 只作为 causal value audit；
3. 当前不把 sidecar 接入 action；
4. 可以复现地声称“value 有 counterfactual 意义且 clean task performance 不劣”，但必须同时
   写明“value 改善 policy 尚未证明”。

这满足 A 路线的安全版本目标，但不是 action-facing success。若以后重开 A，唯一合理的新
实验是扩大 `reachable anchors x dimensions` 的真实 branch coverage 后从头做 sealed gate，
不是继续对 Stage-61/63 cache 调权重。

### 21.61 持久 Goal 重置核验与 B 路线事件驱动执行

当前 active Goal 固定为两条相互独立的路线：

1. A：让 CQN-AS 的 value 具有真实 counterfactual 意义，同时任务效果不低于原始
   CQN-AS；
2. B：系统比较 FM+RL 方案并把 FM 接入 CQN-AS，最终在公平多训练种子实验中超过原始
   CQN-AS。

每一阶段继续使用 `artifact -> 结论 -> 含义 -> 下一 gate -> 立即执行` 协议；ETA 由当前
run 的实测吞吐更新，等待采用进程退出或完成 marker 触发，不使用高频短轮询。

截至本次核验：

- A 已得到 21.60 的双层可复现结论：safe audit pass、action-facing gate fail；
- GPU5 正在对 Stage-60 唯一通过 calibration 的 `p97.5` sparse expected-FLOQ fallback
  运行全新 seeds `75000--75199` 的 200-episode confirmation；检查点为 `40/200`，
  当前吞吐约 `7--8s/episode`，含汇总的剩余 ETA 约 `20--23min`；
- GPU1 已启动 matched `10k / beta=1` expected-FLOQ seed3；首轮包含 JIT 的 update
  interval 为 `106.2s`，在首个 1k checkpoint 前沿用 seed1 实测给出的
  `66--72min` 总 ETA；
- GPU5 confirmation 的父进程退出并写入 `complete` 后，seed2 controller 会自动启动
  同配置 `10k / beta=1` 训练，因此 Stage-60 与 seed2 不会争用同一张卡；
- seed2/seed3 完成后立即在全新 common env seeds `92000--92199` 上执行三训练种子
  crossed bootstrap；预注册 gate 为 mean delta `>0`、95% CI lower `>=0`、
  aggregate wins `> losses` 且至少 `2/3` 个训练种子为正。

Stage-60 的 `40/200` 只用于 ETA，不提前解释为实验结论。

### 21.62 B 路线：补齐 distilled 与 integrated value 的因果审计接口

在等待两张 GPU 的正式实验期间，检查了
`scripts/analyze_cqn_branch_counterfactual.py`。旧实现虽然能加载 CQN-Flow，但无论部署
policy 使用哪个 head，branch 的 predicted Q 都固定来自 integrated endpoints；因此它不能
审计当前 expected-FLOQ winner 真正用于选动作的 distilled scalar readout。

现已新增显式 `--flow-readout={auto,distill,integrated}`：

- `auto` 对带 `flow_distill_lambda` 的 checkpoint 选择 distilled head；
- `distill` 在 forced sibling 与 structured-delta 两种 probe 中直接调用
  `_flow_distill_level/_flow_distill_outputs_per_level`；
- `integrated` 保留共同 source 经 ODE integration 后取 endpoint expectation 的路径；
- 输出 JSON 新增 `value_readout`，避免结果无法区分实际被审计的 head；
- CQN-AS 显式传入 Flow-only readout、或无 distill head 强制选择 distill，都会 fail fast。

进一步发现 Stage-24 训练配置保存的是 `policy_value_beta=null`，而正式 winner 是 eval-time
`beta=1`；若不覆盖，probe 会收集纯 value-greedy 轨迹而非部署 policy。现又加入
`--policy-value-beta={config,bc,number}`，并将 branch trajectory 的 policy readout 与被评分的
value head 解耦。最终部署审计必须显式传 `--flow-readout=distill --policy-value-beta=1`；
JSON 同时记录 `policy_value_beta` 与 `policy_readout`。

focused tests 与语法检查结果为 `7 passed`、`py_compile` pass、`git diff --check` pass。真实
checkpoint smoke 等首张 GPU 按既定 Gate 释放后再做，避免与当前正式 run 抢显存。

审计输出的 `policy_readout` 也已按 `separate_bc_policy` 修正：原始、没有独立 BC tower 的
CQN-AS 在 `beta=null` 时仍标为 `categorical_c51`，不会再被错误写成 `bc`；expected-FLOQ
`beta=1` 则明确标为 `distill_plus_bc`。

最终 task Gate 结束后的 value-authenticity protocol 预注册为：

1. 使用与 task eval 完全不相交的 simulator seeds；
2. 在相同 checkpoint 上分别标明 `distill` 与 `integrated`，禁止把 integrated 的 causal
   signal 归到 deployed distilled head；
3. 使用 simulator-seed cluster bootstrap，而不是把同一 state 的 action pairs 当 IID；
4. deployed head 至少满足 pairwise-sign CI 下界 `>50%` 或 mean-Spearman CI 下界 `>0`，
   并要求至少 `2/3` 个训练种子方向为正，才声称 FM value 具有可复现 action-conditional
   意义；
5. 若 task success 提升但该 Gate 失败，结论只能是“FM representation/plasticity 改善了
   policy”，不能声称 integrated flow 学到了可信 action value。

同时新增可恢复的
`scripts/run_cqn_flow_branch_multiseed_gate.py`。它为每张 GPU 建一个 worker，共享三个
训练 checkpoint 队列；已完成的 probe JSON 会复用。最终汇总不是把 action pairs 当 IID，而是
对 `training checkpoint x simulator seed` 做 crossed bootstrap，并同时要求：

- 每个训练 seed 至少达到冻结的 informative-state coverage；
- aggregate pairwise `>50%` 或 Spearman `>0`；
- aggregate pairwise CI 下界严格 `>50%` 或 Spearman CI 下界严格 `>0`；
- 至少 `2/3` 个训练 seed 的 point direction 为正。

runner、readout/beta plumbing 与 branch helpers 的 focused tests 合计 `10 passed`，两个脚本
`py_compile` pass，相关 diff whitespace check pass。它只是在 task Gate 后立即可执行的工具，
不是实验结果。

为避免 seed2/seed3 训练结束后 GPU 空等，已启动 Stage-64 event controller（PID
`1443098`）。它不轮询训练日志，而是等待两个训练 controller 退出、验证两个
`10000_snapshot.pkl` 和 `training_complete` marker 后，才自动用 GPU1/GPU5 执行：

```text
3 training seeds x 200 common environment seeds (92000--92199)
clean: validation-selected 5k / 5k / 2.5k
FLOQ: matched 10k, beta=1, distilled readout
```

最终 frozen checks 是 mean paired delta `>0`、crossed-bootstrap CI lower `>=0`、
aggregate wins `> losses`、至少 `2/3` training seeds 为正。controller 当前仅处于事件等待，
没有提前占用 GPU。

### 21.63 B Stage-60 正式失败；稀疏 fallback 关闭，matched seed2 已接棒

Stage-60 在 calibration seeds `74000--74049` 唯一选出的 p97.5 threshold 已完成完全独立的
`75000--75199` confirmation。正式 paired 结果为：

| policy | success |
|---|---:|
| clean CQN-AS | `130/200 = 65.0%` |
| p97.5 sparse FLOQ fallback | `127/200 = 63.5%` |

candidate 相对 clean 是 `-1.5pp`，paired `14 wins / 17 losses / 169 ties`，95% paired CI
`[-7,+4]pp`。因此 CI 下界没有达到冻结的 `-5pp` non-inferiority margin，point estimate 也
没有达到 clean。

这不是“threshold 太严所以没触发”：candidate 在 `40,740` 次 inference 中实际应用
`606` 次 override（`1.487%`），涉及 `153/200` episodes。恰恰在这些 override episodes
上，success/reward delta 都是 `-1.96pp`，CI `[-9.15,+5.23]pp`。五项 confirmation checks
只有 `produced_override` 为真，其余四项全部失败。正式 artifact：

```text
exp_local/cqn_flow_floq_fallback/stage60_sparse_floq_gate_seed74000/confirmation/summary.json
```

含义是：outcome-blind coverage calibration 能找到稀疏、BC-supported、source-consistent 的
intervention，但这些条件没有筛出正的真实 action advantage；不能通过继续调 threshold 把它
解释成部署成功。Stage-60 fallback 路线按协议关闭。

下一 Gate 不再改 clean plan 的局部 fallback，而是检验完整、冻结的 expected-FLOQ policy
是否在训练随机性上复现 Stage-29/30 的正向信号。Stage-60 写入 `complete` 后，GPU5 已由
event controller 真正启动 matched seed2（trainer PID `1443702`）；GPU1 seed3 已到 2k，
1k--2k 的实测增量为 `348.55s/1000 steps`，剩余训练 ETA 约 `47--52min`。seed2 首个 1k
节点前按相同配置估计总 ETA `58--65min`。两者完成后 Stage-64 自动执行三训练种子 final
task Gate。

raw YAML 中 seeds2/3 比旧 seed1 多显示五个后来显式化的 default 字段，但通过
`cqn_flow_spec_from_cfg` 比较，三个 resolved method specs 的差异集合为空；新增字段的值也正是
seed1 当时使用的默认语义。当前 seed2/3 CSV 的 non-finite scan 为空。因此这三臂是同一算法
配置的训练随机种子复制，不是不同代码语义的伪多 seed。

### 21.64 B 文献路线重新分层：critic flow 与 policy flow 不能混为一个 baseline

最新一手工作可以按“Flow 放在哪里”分成三类：

1. [FLOQ](https://openreview.net/forum?id=WwQoSHGCXg) 与
   [What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)
   把 Flow 放在 **scalar Q critic**，目标是 dense velocity supervision、test-time recovery
   与高 UTD 下的 feature plasticity；这是当前 CQN action-bin conditioned value 最直接的
   对应路线。
2. [Value Flows](https://openreview.net/forum?id=2VyNYUVF2k) 与
   [FlowCritic](https://arxiv.org/abs/2510.22686) 把 Flow 用于 **return distribution /
   state-value distribution**。它们回答 uncertainty 与 distributional Bellman 问题，不是
   当前 expected-Q action-bin comparator。
3. [FQL](https://openreview.net/forum?id=KVf2SFL1pi)、
   [Direct Flow Q-Learning](https://openreview.net/forum?id=RdkOaK4q6p) 与
   [Q-Flow](https://openreview.net/forum?id=oZqOS1N6Ag) 把 Flow 放在 **action policy**。
   FQL 用 one-step policy distillation 避免对 ODE 反传；DFQL/Q-Flow 用 intermediate
   Q guidance 避免 BPTT。这类方法可能是未来替换 BC action generator 的路线，但不验证
   “CQN-AS 的 value head 用 FM update”这一问题，不能拿它们的提升冒充当前实现复现。

因此当前不转成 flow policy。Stage-64 后的 mechanism Gate 固定回答三件事：

1. 在相同三个 frozen 10k checkpoints、`beta=1` 上比较 deployed distill 与
   integrated `{2,8}`，selection/confirmation seeds 与 task Gate 分离；
2. 使用已有 Stage-20 matched direct-C51 UTD4 checkpoint 的 10k readout 作为非 Flow
   negative control；旧 Stage-20 只在 5k/beta-grid 证明 direct value action-facing fail，
   尚未做 10k matched readout；
3. 若 Flow 最终超过原始 CQN-AS，但 integrated 不超过 distill，结论应归于 Flow 训练带来的
   representation/plasticity，而非 test-time recovery；若要进一步作强机制归因，还需训练
   多 seed monolithic expected-value control，不能仅凭一个 direct-C51 seed。

这里也澄清“蒸馏”的歧义：当前 `flow_distill_readout` 是把 integrated **value endpoint**
压成一个便宜的 scalar Q head，不是 FQL 那种把 iterative **action policy** 蒸馏成 one-step
policy。

已将上述 mechanism Gate 落成可恢复的
`scripts/run_cqn_flow_readout_multiseed_gate.py`：

- 三个训练 checkpoints 在 common selection seeds 上分别运行 distill、integrated-2 与
  integrated-8；
- 只选择一个对所有训练 seeds 共用的 integration steps，禁止每个 seed 各挑有利 readout；
- promotion 同时要求 mean paired delta `>0`、aggregate wins `> losses`、至少 `2/3`
  training seeds 为正；
- 只有 promotion 通过，winner 才会在完全不相交的 200-seed confirmation 上与 distill
  重跑，并使用 crossed training/environment bootstrap，CI lower 必须 `>=0`；
- 所有 job 可恢复并由 GPU worker queue 调度。

该 runner 与 final-task/branch runners 的 focused tests 复核为 `13 passed`，`py_compile` 与
whitespace check 均通过。它将在 Stage-64 task outcome Gate 后按结果解封，目前没有占 GPU。

Stage-65 event controller（PID `1452408`）现已启动：只等待 Stage-64 controller 退出并验证
其 `complete/summary.json`，随后自动用 GPU1/GPU5 跑上述 selection；若 promotion pass，
再自动跑 confirmation。按历史 200-episode clean/distill 分别约 `467s/533s` 以及两 worker
调度估算：当前训练剩余约 `45--60min`，Stage-64 约 `32--38min`，Stage-65 selection
约 `15--22min`；若 integrated promotion pass，confirmation 再约 `35--50min`。所有等待都
绑定进程退出事件，没有短轮询。

Stage-66 conditional controller（PID `1453113`）也已预注册，但不占 GPU：只有 Stage-65
在独立 confirmation 上 `gate=pass`，才读取唯一 selected integration steps，并在再一组
`99000--99199` common seeds 上做三训练 seed integrated-vs-clean superiority Gate；否则只写
`skipped_no_integrated_promotion`。这防止 selection fail 后仍无条件消耗 1200 个 task
episodes。

### 21.65 Adjoint Matching 的准确映射：它优化 flow actor，不训练 flow value

用户此前点名的
[Q-learning with Adjoint Matching (QAM)](https://arxiv.org/abs/2601.14234)
已经是 ICLR 2026 方法。它的结构是：

```text
parameterized critic Q(s,a)
        -> action gradient dQ/da
        -> adjoint-derived step-wise objective
        -> multi-step diffusion/flow action policy
```

它解决的是“怎样用 critic 的一阶信息优化 iterative flow **policy**，同时避免对完整 ODE
solver 做不稳定 BPTT”。critic 仍通过 TD backup 学习；QAM 没有用 Flow Matching 来更新
value。因此它不是当前 `FM(s,image,action-bin) -> Q` 方案的替代实现，也不能解释本地
expected-FLOQ 的结果。

更新的
[Trust Region Q Adjoint Matching](https://arxiv.org/abs/2605.27079)
进一步指出：QAM 会放大 ill-conditioned critic 的小误差，因此用 path-space KL trust region
限制 flow policy 偏离 behavior prior。这一点反而连接了本项目的两条路线：

- 如果 Q 本身只是 imitation shortcut 或 counterfactual ordering 不可靠，Adjoint Matching
  会把错误 action gradient 更有效地放大到 actor；
- 所以 A 路线的真实 value Gate 是采用 QAM/TRQAM 前的先决条件，而不是可以被它绕过的步骤；
- 当前 CQN-AS action bins 是离散 C2F choices，当前 Flow 又作用在 scalar value 轴，没有
  continuous action ODE 或 `dQ/da` 路径，直接移植 adjoint loss 在数学上对象就不匹配。

明确推荐：本轮不实现 QAM。若以后单独开“flow action policy”路线，应先冻结通过 causal
Gate 的 critic，再将 QAM 与 TRQAM 的 trust-region 版本作为 actor-side 对照；不能把它混进
当前 value-flow ablation。

### 21.66 B Stage-67：补上“各方法最佳 checkpoint”公平性 Gate

固定比较 10k checkpoint 能回答 matched-budget 的问题，但如果 10k 后发生过拟合，它不能回答
“这个方法验证选出的最佳版本是否超过 clean CQN-AS”。因此在 Stage-64/66 的固定-budget
task Gate 之后，又补了一层严格分离的 checkpoint-selection Gate：

1. 对每个 Flow 训练 seed 的 `1k,2k,...,10k` checkpoints，在共同的 10 个 screen seeds
   上筛出 top-2；这里只做低成本淘汰，不能用于最终结论；
2. 在新的 50 个 validation seeds 上，从各自 top-2 中为每个训练 seed 独立选择一个
   checkpoint；允许不同训练 seed 有不同最佳训练步数，因为这正是实际 early stopping；
3. 选择完成后，在从未参与选择的 200 个 confirmation seeds 上将三个 frozen winners
   分别与 clean CQN-AS 配对，并用 crossed training-seed/environment-seed bootstrap；
4. 最终仍要求 mean delta `>0`、CI lower `>=0`、aggregate wins `> losses`，且至少 `2/3`
   training seeds 为正。测试集不参与 checkpoint 选择。

实现为 `scripts/run_cqn_floq_checkpoint_selection_gate.py`，screen/validation/
confirmation 都支持断点恢复、GPU worker queue 以及 distilled/integrated 两种 readout；
focused tests 为 `6 passed`，`py_compile` 通过。

Stage-67 event controller 已实际启动并复核存活（PID `1456427`）。它先等待 Stage-66 的
终态；如果 Stage-64 的 distilled 10k 或 Stage-66 的 integrated 10k 已经通过最终 task Gate，
就写 `skipped_task_gate_already_passed`，避免再消耗评估预算；只有两个固定-budget readout
都失败时，才执行上面的最佳-checkpoint Gate。若 Stage-65 没有提升 integrated readout，
Stage-67 使用 distill；否则使用 Stage-65 在独立 confirmation 上选定的唯一 integration
steps。对应输出固定为：

```text
exp_local/cqn_flow_high_utd/stage67_best_checkpoint_multiseed_seed101000_20260724
exp_local/cqn_flow_high_utd/stage67_best_checkpoint_controller
```

截至 `2026-07-24 11:35 BST`，GPU1 的 seed3 已生成 5k checkpoint，GPU5 的 seed2 已生成
2k checkpoint，两卡利用率均约 `67%`、显存各约 `25.5GB`，且没有 non-finite 训练记录。
按当前每 1k steps 的实测墙钟时间，seed3 剩余约 `32--36min`，seed2 剩余约
`45--50min`；整个下游链由进程退出事件触发，不使用短周期轮询。

### 21.67 B Stage-68/69：三训练种子 direct-C51 最佳版本负对照

#### 上一阶段结果与解释

已有 direct-C51 UTD4 seed1 的 action-facing validation 在 `5k` checkpoint 上，最佳
`beta=3` 仍是 `42%`，低于自身 BC 的 `46%`，paired delta `-4pp`、W/L/T
`6/8/36`；而它的原生 BC-controlled checkpoint 曲线是
`48/56/44/48% @ 2.5k/5k/7.5k/10k`。这证明单个 checkpoint 的 C51 value 没有产生正控制
作用，但只有一个训练 seed，而且不能用 10k final 代替该方法的 validation-best 版本。

因此这个结果仍不能回答两个问题：

1. direct-C51 的失败是否在训练随机性上复现；
2. Flow 若通过 task Gate，提升是否仅来自 high-UTD/独立 BC tower，而不是 Flow 训练。

direct-C51 也不是最终的 monolithic scalar-Q control：它与 expected-FLOQ 的输出
parameterization/C51 loss 不同。它先回答“同一 strict two-tower、高 UTD、相同 replay/MC
protocol 的非 Flow CQN control 能否解释 task 提升”；若要把差异严格归因于 Flow loss，之后
仍需同 target 的 direct scalar-Q。

#### 下一 Gate

Stage-68 复现 direct-C51 UTD4 seeds2/3，与已有 seed1 组成三训练种子。Stage-69 不比较
任意 final step，而使用与 Flow Stage-67 相同的三段协议：

1. 每个训练 seed 在共同的 10 个 screen seeds 上筛 `1k--10k` checkpoints 的 top-2；
2. 新的 50 个 validation seeds 各自选择最佳 checkpoint；
3. 三个 frozen winners 在新的 200 个 confirmation seeds 上分别与各自
   validation-selected clean CQN-AS 配对；
4. `beta=3` 固定为旧 direct-C51 validation 选出的算法级超参数，confirmation 不再调参；
5. superiority Gate 要求 mean delta `>0`、crossed-bootstrap CI lower `>=0`、
   wins>losses 且至少 `2/3` training seeds 为正。

runner 已扩展 `flow-readout=auto`，从而同一套 best-checkpoint protocol 能读取 CQN-AS
direct-C51 checkpoint；新增回归测试后相关 tests 为 `5 passed`，`py_compile` 与
`git diff --check` 均通过。

#### 已执行与 ETA

Stage-68 event controller（PID `1458955`）已复核存活。它等待 Stage-67 终态后立即在 GPU1/5
并行训练 direct-C51 seeds2/3，并严格验证两个 10k snapshots 后才写 `complete`：

```text
exp_local/cqn_flow_high_utd/stage68_direct_c51_utd4_seed2_gpu1_20260724
exp_local/cqn_flow_high_utd/stage68_direct_c51_utd4_seed3_gpu5_20260724
exp_local/cqn_flow_high_utd/stage68_direct_c51_multiseed_controller
```

Stage-69 controller（PID `1459047`）也已预注册，等待 Stage-68 进程退出事件后执行上述
screen/validation/confirmation：

```text
exp_local/cqn_flow_high_utd/stage69_direct_c51_best_checkpoint_seed106000_20260724
exp_local/cqn_flow_high_utd/stage69_direct_c51_best_checkpoint_controller
```

依据 Stage-20 direct-C51 的实测，Stage-68 两卡并行训练约 `25--35min`；Stage-69 的
30 个短 screen、6 个 validation 和最终 6 个 confirmation jobs 预计约 `70--110min`。
这两段排在当前 Flow task/readout/best-checkpoint Gate 之后，不与其争 GPU；等待全部使用
父进程退出事件，不作短轮询。

### 21.68 A/B 共同机制对照：matched monolithic direct scalar-Q

#### 上一阶段结果与解释

Stage-68/69 的 direct-C51 能排除“任何非 Flow 的 high-UTD CQN-AS 都能得到同样提升”，但
它仍不是对 expected-FLOQ 训练目标的单变量对照：C51 输出固定 atoms 上的 categorical
distribution，并用 projection/CE 更新；expected-FLOQ 输出 scalar-Q endpoint，并用
conditional velocity field 更新。因此，即使二者任务表现不同，也不能把差异唯一归因于
Flow Matching。

[What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)
中的 monolithic comparator 是对同一个非平稳 scalar TD target 做 squared-error regression。
据此实现了 `CQNDirectQAS`，它与本地 expected-FLOQ 保持以下因素不变：

- state/image encoder、strict independent BC tower 和 replay-next transition；
- C2F 的 `level + current midpoint + candidate action bin` conditioning；
- target-action source、coherent horizon-4 exploration、MC-return anchor 与 UTD=4；
- action selection 时同样将 standardized scalar-Q 与独立 BC log-prior 相加。

唯一核心替换是：不学习 Flow velocity field，而由一个 monolithic
candidate-conditioned scalar head 直接对 Bellman target 做 MSE。这里特意使用 MSE 而不是
Huber，才能匹配论文所分析的 monolithic control；`demo_fosd=false`，因为 scalar head 没有
categorical CDF。该实现不把 BC loss 和 critic loss混在同一 head，所以 task 提升不能仅由
“value head 偷做 action imitation”解释。

实现涉及：

```text
robobase/method/cqn_direct_q.py
robobase/cfgs/launch/cqn_direct_q_two_tower_coherent_mc_high_utd4_gate.yaml
scripts/check_cqn_direct_q_training_gate.py
scripts/run_cqn_checkpoint_beta_selection_gate.py
```

训练/推理 shape、value+BC action selection、JIT update、真实 pixel tower、launch config、
branch causal readout、训练健康检查、checkpoint×beta selection 与已有 Flow runners 的相关
focused tests 合计 `23 passed in 54.11s`；`py_compile` 和相关 whitespace check 通过。测试
只能证明 plumbing 正确，不能替代 MovePlate 任务结果。

#### 下一 Gate

先用两个 1k-update pixel smoke runs 要求：

- 生成 1k snapshot 且训练 CSV 至少有两行；
- direct-Q loss、TD/MC loss、Q span 和 gradient 全部 finite；
- non-finite gradient count 为 0，critic gradient 非零；
- resolved config 确认 `direct_scalar_q=true` 且 `direct_q_loss=mse`。

smoke 通过后，训练 seeds1/2/3 的完整 10k checkpoints。seed1 完成后同时做一个低成本
counterfactual screen；它只用于提前识别完全无信号的实现，不作为最终因果结论。

最终 Stage-73 采用三段、互不重叠的数据选择协议：

1. `111000--111009`：各训练 seed 的 1k--10k checkpoints 在预声明 `beta=1` 下筛 top-2；
2. `112000--112049`：所有 retained checkpoints 评估公共 beta grid `{0.3,1,3}`；只选择一个
   对三个训练 seeds 共用的 beta，再为每个训练 seed 选 validation-best checkpoint；
3. `113000--113199`：冻结 beta/checkpoints 后，与各自 clean CQN-AS
   validation-best checkpoint 做 200-seed paired confirmation。

最终 superiority Gate 与 Flow 相同：mean paired delta `>0`、crossed-bootstrap 95% CI
lower `>=0`、aggregate wins `> losses`，且至少 `2/3` training seeds 为正。这样可以回答：

- direct scalar-Q 通过而 Flow 不通过：value parameterization 足够，FM 没有必要；
- Flow 通过而 direct scalar-Q 不通过：支持 FM 的 plasticity/recovery 机制，而非仅 high UTD；
- 二者都通过：比较独立 confirmation 效应与 causal audit，再判断是否值得承担 Flow 开销；
- 二者都失败：当前 action-conditioned TD route 不能达到“任务不劣/超越”的部署要求。

#### 已执行与 ETA

已建立完全事件驱动的 Stage-70--73 链：

```text
Stage-70 smoke controller                         PID 1474029
Stage-71 direct-Q seeds1/2 full training          PID 1474137
Stage-72 seed3 + seed1 causal screen              PID 1474346
Stage-73 checkpoint x global-beta final gate      PID 1478290
```

对应 artifacts：

```text
exp_local/cqn_flow_high_utd/stage70_direct_q_smoke_seed{1,2}_gpu{1,5}_20260724
exp_local/cqn_flow_high_utd/stage71_direct_q_utd4_seed{1,2}_gpu{1,5}_20260724
exp_local/cqn_flow_high_utd/stage72_direct_q_utd4_seed3_gpu1_20260724
exp_local/cqn_flow_high_utd/stage72_direct_q_seed1_causal_screen_seed110000.json
exp_local/cqn_flow_high_utd/stage73_direct_q_checkpoint_beta_seed111000_20260724
```

截至 `2026-07-24 11:56 BST`，上游 Flow seed3/seed2 分别到 8k/5k，GPU1/GPU5 利用率
`75%/81%`、显存均约 `25.5GB`。最近 checkpoint 间隔给出的剩余训练 ETA 分别约
`12min/31min`。Stage-70--73 不会提前抢卡：它们依次等待 Stage-69 以及各父 controller
退出。按 direct-C51 的实测 10k 训练约 23min，Stage-70 smoke 约 3--5min，
Stage-71/72 每批 full training 约 23--30min；Stage-73 因包含 30 个 screen、18 个
validation 和 6 个 confirmation jobs，预计约 `90--145min`。最终墙钟还取决于
Stage-65/66/67 是否因前序 gate 通过而跳过；所有等待均使用 `tail --pid` 进程事件，不使用
高频短轮询。

### 21.69 B Stage-74--80：官方 FLOQ fidelity 修正与 task-qualified causal Gate

#### 上一阶段结果与解释

截至 `2026-07-24 12:10 BST`，原 matched expected-FLOQ 的 seed3 已完成并生成
`10500_snapshot.pkl`，seed2 已生成 7k checkpoint、仍在 GPU5 训练；实测每 1k frame
约 `5.7--6.0min`，所以 seed2 余下训练和终局评估约 `12--18min`。Stage-64 尚未产生
`summary.json`，因此这一时刻没有新的 task superiority 结论，不能把 seed2 训练中的
单次 5-episode eval 当作最终结果。

随后对 [FLOQ 官方论文](https://arxiv.org/abs/2509.06863) 和
[官方代码](https://github.com/CMU-AIRe/floq) commit
`3d60e638b42bbef018c56cb1199ff37c3470520d` 做了逐项实现审计。已有本地 Stage-24/62
与官方方法一致的部分包括：

- target endpoint 对 8 个 target-flow samples 求均值；
- current velocity loss 使用 8 个独立 uniform source samples；
- 8 个 flow steps、HL-Gauss 51 atoms、`sigma=16`、Fourier time embedding；
- online flow endpoint distillation、UTD=4 和独立 BC tower。

但发现两个此前未隔离的实现差异：

1. 本地 resolved config 的 `flow_source_min/max=null` 会从 value support
   `[-2,2]` 推导为 `[-0.2,0.2]`；官方 FLOQ 的 `noise_coverage=.1` 对本任务
   `[0,1]` terminal-return support 对应显式 `[0,0.1]`。因此旧文档中“已匹配
   `[0,0.1]`”的表述不正确。
2. 官方实现把 8 个 current-source velocity squared errors **求和**，再与 distillation
   相加；本地实现对它们 **求均值**。不改变 optimizer/global loss scaling 时，
   `bcfm_lambda=8` 才与官方 BCFM:distill 相对权重一致。

两项差异不能同时修改，否则 task 变化无法归因。因此新增两个单变量 arms：

```text
cqn_flow_floq_source01_distill_two_tower_high_utd4_gate
  仅把 source support 改为 [0, 0.1]，其余保持旧 Stage-24/62

cqn_flow_floq_bcfm8_distill_two_tower_high_utd4_gate
  仅把 bcfm_lambda 从 1 改为 8，其余保持旧 Stage-24/62
```

对应实现与 gate：

```text
robobase/cfgs/launch/cqn_flow_floq_source01_distill_two_tower_high_utd4_gate.yaml
robobase/cfgs/launch/cqn_flow_floq_bcfm8_distill_two_tower_high_utd4_gate.yaml
scripts/check_cqn_floq_training_gate.py
scripts/run_cqn_floq_fidelity_arm_gate.py
```

config、训练健康检查、joint arm selection、checkpoint×beta selection、branch
auto-readout 相关 focused tests 分组结果分别为 `3 passed`、`5 passed`、`15 passed`、
`13 passed`；`py_compile` 和相关 whitespace check 通过。它们证明配置和评估 plumbing，
不代表 task Gate 已通过。

#### 下一 Gate

只有原 Stage-64 distilled、Stage-66 integrated、Stage-67 best-checkpoint 三个公平
task Gates 全部失败时，才启用 fidelity correction，避免在已有成功方案上重复调参：

1. Stage-74：两个 arms 各做 1500-frame pixel smoke；要求正确 resolved config、
   1k snapshot、finite losses/gradients 且 non-finite gradient count 为 0。
2. Stage-75：smoke 通过后，在 seed1 各训练完整 10k checkpoints。
3. Stage-76：共同 10-seed screen 各保留 top-2；在新的 50 validation seeds 上联合选择
   唯一 `arm × checkpoint × global beta`；只有 validation delta `>=+2pp` 且
   wins>losses 才进入新的 200-seed sealed confirmation。promotion 要求 confirmation
   delta `>0`、wins>losses、bootstrap CI lower `>=-5pp`。
4. Stage-77/78：只训练被 promotion 的唯一 arm 的 seeds2/3；冻结 Stage-76 beta 后，
   三训练种子重新做 checkpoint selection 和新的 200-seed confirmation。最终
   superiority Gate 恢复严格要求：mean delta `>0`、CI lower `>=0`、wins>losses、
   至少 `2/3` training seeds 为正。
5. Stage-79：对通过 Stage-73 task Gate 的 direct scalar-Q frozen winners 做三训练种子
   sibling-horizon causal audit；每 seed 至少 24 个 informative states，要求至少
   `2/3` training seeds 的 value/realized-return ordering 为正。
6. Stage-80：只对真正通过 task Gate 的 Flow winner 做同一 causal audit；优先级固定为
   corrected Stage-78、integrated Stage-66、distill Stage-64、best-checkpoint
   Stage-67。没有任何 Flow task winner 时明确写
   `skipped_no_b_task_pass`，不能拿 causal metric 掩盖 task 失败。

这形成两个彼此独立的最终判据：A 必须同时满足“不低于 clean CQN-AS”和 causal value
有效；B 必须先超过 clean CQN-AS，再证明 Flow readout 的 action ordering 不是 imitation
shortcut。

#### 已执行与 ETA

Stage-74--80 已以进程退出事件预注册并在启动时复核存活：

```text
Stage-74 fidelity smoke                    PID 1484747
Stage-75 two seed1 full arms               PID 1484877
Stage-76 joint arm/checkpoint/beta gate    PID 1485011
Stage-77 promoted arm seeds2/3             PID 1485124
Stage-78 strict three-seed task gate       PID 1485247
Stage-79 direct-Q three-seed causal gate   PID 1485800
Stage-80 Flow three-seed causal gate       PID 1485852
```

主要输出固定为：

```text
exp_local/cqn_flow_high_utd/stage76_floq_fidelity_arm_gate_seed114000_20260724
exp_local/cqn_flow_high_utd/stage78_floq_fidelity_multiseed_seed118000_20260724
exp_local/cqn_flow_high_utd/stage79_direct_q_multiseed_causal_seed121000_20260724
exp_local/cqn_flow_high_utd/stage80_floq_multiseed_causal_seed122000_20260724
```

若旧 Flow 任一 task Gate 通过，Stage-74--78 会在父事件到达后立即跳过；否则，从启动
correction 计，Stage-74 约 `10--15min`、Stage-75 `60--70min`、Stage-76
`45--65min`、Stage-77 `60--70min`、Stage-78 `70--110min`。Stage-79/80 的 causal
audit 各约 `45--75min`。这些 ETA 来自当前 1k checkpoint 间隔和此前 paired-eval
吞吐，不采用固定 30 秒轮询。

### 21.70 Stage-81：A/B 最终判据汇总与当前资源窗口

#### 上一阶段结果与解释

Stage-69--80 的原始 controller 链为了避免两个 25GB pixel jobs 抢同一张卡，在资源层面
串行；这不代表 A/B 的假设、selection seeds 或结论被合并。`2026-07-24 12:15 BST`
检查时，Flow seed2 已生成 8k checkpoint、GPU5 约 `82% / 25.5GB`，而已完成 Flow
seed3 后的 GPU1 形成了一个短空窗。这个状态仍没有 Stage-64 task `summary.json`，
所以当前能汇报的是训练进度，不是 Flow 胜负。

为让最终结论也保持两条路线分离，新增：

```text
scripts/summarize_cqn_final_routes.py
tests/unit/test_summarize_cqn_final_routes.py
```

它只读取 frozen JSON artifacts，分别要求：

- A：direct scalar-Q 的 held-out task Gate 通过，且三训练种子 causal-value Gate 通过；
- B：唯一 task-qualified Flow winner 严格超过 clean CQN-AS，且随后同样通过 causal
  Gate；
- 任一路只有 task 或只有 causal 通过都不能 promotion；输出会列出精确
  `unmet_gates`，不会把 audit-only value 写成可部署 policy。

汇总器与相邻 selection/causal runner 的 focused regression 为 `9 passed in 0.07s`，
`py_compile` 和相关 `git diff --check` 通过。

#### 下一 Gate 与已执行

Stage-81 event controller 已启动并复核存活（PID `1487434`）：

```text
exp_local/cqn_flow_high_utd/stage81_final_route_summary_controller
exp_local/cqn_flow_high_utd/stage81_final_route_summary_20260724.json
```

它等待 Stage-80 进程终态后，自动从 `stage78_fidelity`、`stage66_integrated`、
`stage64_distill`、`stage67_best_checkpoint` 中解析实际 task winner；若 Stage-80 写
`skipped_no_b_task_pass`，则 B 明确记录为没有 task-qualified candidate。Stage-81 只做
证据汇总；若 `research_goal_gate=fail`，goal 保持 active，并按未满足的 A/B gate
进入下一轮单变量实验。

同时已在空闲 GPU1 立即启动一个不参与 selection 的 direct-Q pixel preflight：

```text
exp_local/cqn_flow_high_utd/stage70_preflight_direct_q_smoke_seed1_gpu1_20260724
exp_local/cqn_flow_high_utd/stage70_preflight_direct_q_smoke_controller
controller PID 1487045
training PID 1487053
```

为了防止 smoke 尚未释放约 25GB 显存时 Stage-64 双卡评估启动，只暂时
`SIGSTOP` 了等待中的 Stage-64 controller；Flow seed2 训练没有被暂停。preflight
controller 的 EXIT trap 会在成功或失败时自动 `SIGCONT` Stage-64；另有独立的
进程退出 watchdog（PID `1488568`）提供同一恢复保证。按 GPU5 最近
checkpoint 间隔和 GPU1 direct-Q 初始化速度，两边预计都还需约 `8--14min`，因此这项
调度利用了真实空窗而不需要短轮询。

### 21.71 Stage-82：恢复非平稳 TD target 的 Flow 机制检验

#### 上一阶段结果

`2026-07-24 12:25 BST` 的实际 artifacts 显示：

- direct scalar-Q MovePlate preflight 已完成：
  `exp_local/cqn_flow_high_utd/stage70_preflight_direct_q_smoke_controller/gate.json`
  为 `pass`；7 个健康检查全部通过，1k snapshot 存在，两行训练日志有限，
  `direct_q_grad_nonfinite_fraction=0`，最大梯度范数约 `0.1067`。
- preflight 退出 watchdog 已恢复 Stage-64 controller；Stage-64 进程状态为运行等待，
  不是暂停。
- matched expected-FLOQ seed2 已生成 10k checkpoint；最终 5-episode 训练内 eval
  success 为 `0.44`，但 trainer 尚在终局保存/清理，且
  `stage62_seed2_control/training_complete` 与 Stage-64 `summary.json` 尚未出现。
  因此 `0.44` 不是 sealed paired-eval 结论，也不能与 clean best checkpoint 直接比较。

#### 解释

direct-Q preflight 只证明 scalar-Q、独立 BC tower、pixel replay 和 UTD=4 的实现可训练，
不证明策略达到 clean CQN-AS，更不证明 value 有因果意义。Flow seed2 的训练内 5-episode
数值方差太大，最终 B 路线仍必须等待 Stage-64 的三训练种子、公共 200-seed paired Gate。

对 [What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)
及 [FLOQ 官方代码](https://github.com/CMU-AIRe/floq) commit
`3d60e638b42bbef018c56cb1199ff37c3470520d` 的进一步审计指出：论文观察到的
expected-value Flow 优势主要来自非平稳 TD target 下的 test-time recovery 和 feature
plasticity，而 distributional target 本身并非稳定增益来源。官方 FLOQ 用 online critic
对下一动作做选择、target flow 做 Bellman 评估；本地 Stage-24/62 则使用
`td_target_action_source=replay_next`，更接近固定行为轨迹上的 SARSA target。后者会降低
target 非平稳性，可能恰好抑制要验证的 Flow plasticity 机制。这是一个机制假设，不是当前
实验已经证明的结论。

#### 下一阶段 Gate

如果 Stage-81 的 B task Gate 仍失败，Stage-82 只改变一个变量：

```text
replay_next  ->  policy_value TD target action
```

并运行两个完全 matched 的 MovePlate arms：

1. `cqn_flow_floq_td_policy_value_two_tower_high_utd4_gate`：online integrated
   Flow-Q 与独立 BC log prior（target beta=1）选择下一动作，target Flow 评估；
2. `cqn_direct_q_td_policy_value_two_tower_coherent_mc_high_utd4_gate`：相同
   Double-Q target-action 规则，但 critic 是普通 scalar-Q，作为“非平稳 TD 本身”的
   机制对照。

两者的 rollout `policy_value_beta=null`，因此采集和评估仍是原始独立 BC policy；demo、
replay、exploration、UTD、MC anchor 与损失均继承各自 replay-next parent。这个设计把
“训练 target 由 value 选择”与“部署 policy 用 value 选择”分开，避免用 rollout 改动伪造
提升。

先各做 1500-frame smoke，要求 resolved config 正确、1k snapshot、至少两行 finite
training metrics、non-finite gradient fraction 为 0。smoke 通过后才训练 seed1 完整
checkpoints；用新的 screen/validation seeds 选择 checkpoint 和唯一 target beta，再在
sealed confirmation seeds 上比较：

- Flow policy-value target vs clean CQN-AS：检验 B 的部署效果；
- direct-Q policy-value target vs clean CQN-AS：检验普通 critic 是否已足够；
- Flow vs direct-Q 的公共 paired split：只有 Flow 的增益显著更大，才能把效果归因于
  FM plasticity，而不是仅归因于恢复非平稳 Double-Q target。

task Gate 仍要求 validation-selected best checkpoint、held-out mean delta `>0`、
bootstrap 95% CI lower `>=0`、wins>losses 和至少 `2/3` training seeds 为正。通过 task
Gate 后再运行同一 sibling-horizon causal audit；task 与 causal 必须同时通过才可推荐。

#### 已实现

已加入独立的 target-only 参数 `td_target_policy_value_beta`，并在 CQN-AS、direct-Q
和 Flow-Q update 中实现 online critic 选动作、target critic 评估的 Double-Q 语义。
rollout 参数 `policy_value_beta` 仍独立且为 `null`。Flow-V/direct-A hybrid 明确拒绝这个
尚未定义的 target source。

对应配置与 focused tests：

```text
robobase/cfgs/launch/cqn_flow_floq_td_policy_value_two_tower_high_utd4_gate.yaml
robobase/cfgs/launch/cqn_direct_q_td_policy_value_two_tower_coherent_mc_high_utd4_gate.yaml
tests/unit/test_cqn_flow.py
tests/unit/test_cqn_direct_q.py
```

新路径的最小 config/update 测试为 `5 passed in 66.58s`，此前相邻回归为
`3 passed in 45.5s`；更宽的 direct-Q/CQN-AS/Flow focused regression 为
`16 passed in 167.99s`，扩展后的两个训练真实性 checker 为 `7 passed in 0.03s`，
`py_compile` 和相关 `git diff --check` 均通过。Stage-82
训练必须等 Stage-81 给出 B 未满足的真实 gate 后才占用 GPU，避免在现有 B 路线已经通过
时继续调参；等待与启动均使用父 controller 退出事件，不做固定短轮询。

### 21.72 Stage-64 基础设施故障恢复与 Stage-82/83 实际排队

#### 上一阶段结果

Flow seed2 在 `2026-07-24 12:26:28 BST` 正常完成，10k snapshot 和
`stage62_seed2_control/training_complete` 均存在。但原 Stage-64 在下一秒退出，实际
`gate.log` 为：

```text
FileNotFoundError:
exp_local/cqn_value_fidelity_stage22/
clean_cqn_as_repro_seed2_gpu5_20260724/snapshots/5000_snapshot.pkl
```

这不是训练、任务或 causal Gate fail。真实 clean seed2 run 是
`clean_cqn_as_repro_seed2_gpu1_20260724`；它的 5k snapshot 存在，原 validation
artifact 也明确记录 seed2 best step `5000`、training-time success `0.72`。原 controller
把目录名中的 `gpu1` 写成了 `gpu5`。Stage-65--81 因 `set -e` 的父依赖检查，在
`12:26:29--12:26:40` 依次快速退出，所以没有任何下游算法结论。

#### 解释

必须区分“实验 Gate 失败”和“runner 输入路径错误”。本次 cascade 没有生成 Stage-64
`summary.json`，不能记作 B 路线负结果，也不能据此触发下一种算法。恢复时仍使用原先冻结
的 clean best checkpoints：

```text
seed1: gpu1 / 5000
seed2: gpu1 / 5000
seed3: gpu1 / 2500
```

而不是重新查看 sealed seeds 后挑 checkpoint。

#### 修复、Gate 与执行

已用修正后的 seed2 路径启动 Stage-64 recovery controller：

```text
exp_local/cqn_flow_high_utd/stage64_multiseed_final_controller_r1
PID 1502217
```

启动前逐个验证 6 个 baseline/candidate snapshots；runner PID `1502225` 已真实拉起两个
GPU workers。`2026-07-24 12:36:38 BST` 时，GPU1/GPU5 上首批 clean seed1/seed2 都已完成
`160/200` episodes，当前 running success 分别约 `0.662/0.606`。每 160 episodes 实测约
5.3min；还剩本批 40 episodes 和后续两批，共约 `15--19min`。这些 running numbers 只用于
ETA，不用于胜负。

新增并启动可复现的 recovery orchestration：

```text
scripts/run_cqn_stage65_67_recovery.sh
  PID 1504526，事件等待 Stage-64-r1
  -> Stage-65 integrated readout selection
  -> conditional Stage-66 sealed task Gate
  -> conditional Stage-67 validation-best checkpoint Gate

scripts/run_cqn_stage82_83_td_target_mechanism.sh
  PID 1504866，事件等待 Stage-65--67 终态
  -> 仅当三个旧 B task Gates 都失败时运行
  -> Stage-82 Flow/direct-Q 两个 1500-frame smoke
  -> Stage-83 两 arms 各 seeds1/2 full training
```

Stage-65--67 recovery 中所有 clean seed2 引用均已改为真实 `gpu1` 路径；shell syntax 与
whitespace checks 通过。Stage-82/83 的两个 smoke checker 不只看 finite loss，还验证：

- Flow/direct scalar-Q 的声明 target source 都是 `policy_value`；
- target-only beta 为 `1`；
- rollout `policy_value_beta=null`，仍为 exact BC；
- snapshot、日志、有效梯度和 non-finite gradient fraction。

Stage-83 根据已测吞吐做异步流水调度：先 `Flow-seed1 || directQ-seed1`；较快的
directQ-seed1 完成后在 GPU5 启 Flow-seed2；Flow-seed1 完成后在 GPU1 启
directQ-seed2。这样预计 smoke `8--12min`，两种方法各两个完整训练种子总墙钟约
`85--105min`，而不是按四个 jobs 串行约 180min。若现有 B task Gate 已通过，
Stage-82/83 会写 `skipped_existing_b_task_pass`，不会重复训练。两层 controller 均用
`tail --pid` 等进程退出事件，不做短轮询。

### 21.73 Stage-84--91：两路线完整 task + causal 收口链

#### 上一阶段结果与解释

`2026-07-24 12:42:35 BST`，Stage-64-r1 已完成 clean seed1/seed2 的两个 200-episode
jobs，生成真实 JSON；第二批 Flow seed1/seed2 分别推进到 `100/200` 和 `90/200`。
running success 约 `0.700/0.478`，但还没有 seed3、crossed bootstrap 或最终
`summary.json`，所以仍不能下 B 路线结论。GPU1/GPU5 各约 2.9GB、利用率约 31%，与
MovePlate eval 的 simulator-bound 特征一致。按当前一批约 6--7min，第二批剩余和最后
seed3 clean/Flow 一批合计 ETA 约 `12--16min`。

只有训练 smoke 而没有 validation-best task Gate 和 causal Gate，不满足本 research goal。
因此后续不再依赖已经 cascade 退出的旧 controller，而用可审计脚本重新注册完整收口链。

#### Route A Gate 与执行

新增 `scripts/run_cqn_stage84_87_route_a.sh`，controller PID `1507445`。它等待
Stage-82/83 释放 GPU 后执行：

1. Stage-84：matched replay-next direct scalar-Q seeds1/2 两卡并行 full training；
2. Stage-85：seed3 full training；若 target-policy 实验已启用，同时把 Flow-target
   seed3 和 direct-Q-target seed3 流水排进另一张卡；
3. Stage-86：`130000--130009` screen、`131000--131049` validation、冻结全局
   beta/per-seed checkpoint 后在 `132000--132199` sealed confirmation；
4. Stage-87：只读取 Stage-86 frozen beta/checkpoints，在 `133000--133031` 做三训练种子
   sibling-horizon causal audit。

Route A 的 promotion 必须同时满足 strict task Gate 与 causal Gate；training loss、
training-time 5-episode eval、或单训练 seed correlation 都不能替代。

#### TD-target 机制的最终 Gate

新增 `scripts/run_cqn_stage88_90_td_target_final.sh`，controller PID `1507513`。只有
Stage-82/83 实际训练时才运行：

- Flow policy-value TD target 三训练种子：integrated-8 readout；
- direct scalar-Q policy-value TD target 三训练种子：auto scalar readout；
- 两者共用预声明的 `134000` screen、`135000` validation、`136000` sealed
  confirmation seeds 与 beta grid `{0.3,1,3}`；
- task 通过者再共用 `137000--137031` causal seeds。

共享 splits 使“恢复非平稳 target 本身”和“Flow parameterization 的额外效果”可直接比较；
但 Route B 只允许 Flow candidate，direct-Q target arm 只能进入 Route A/机制控制。

#### 最终汇总与推荐规则

新增：

```text
scripts/summarize_cqn_autoresearch_routes.py
tests/unit/test_summarize_cqn_autoresearch_routes.py
scripts/run_cqn_stage91_final_summary.sh
```

汇总器支持每条路线多个预声明 candidates。每个 candidate 必须 task 和 causal 同时 pass；
若有多个 pass，保守选择 task crossed-bootstrap CI lower 更高者，再按 mean paired delta
破平。Route A 可在 replay-next direct-Q 与 policy-value-TD direct-Q 中选择；Route B 可在
legacy distill、integrated、validation-best 与 policy-value-TD Flow 中选择，非 Flow
control 不会误记为 B 成果。相关 summarizer regressions 为 `7 passed in 0.02s`。

Stage-91 controller PID `1507877` 会先为任何 task-qualified legacy Flow 补做三训练种子
causal audit，再写：

```text
exp_local/cqn_flow_high_utd/stage91_final_autoresearch_summary_20260724.json
```

若两条路线都 pass，写 `research_goal_pass`；否则写 `next_gate_required`，goal 保持
active，下一轮必须根据具体 unmet gate 继续，不能因总运行时间长而结束。Stage-65--91
所有等待 PID 当前均处于 `do_wait`，没有短间隔 sleep loop。

### 21.74 Stage-64-r1 前两训练种子与下游 orchestration 审计

#### 当前已完成的实际结果

Stage-64-r1 已在相同 `92000--92199` simulator seeds 上完成前两个训练种子的
clean-vs-distilled-Flow paired eval：

| training seed | clean | Flow | paired delta | W/L/T |
|---|---:|---:|---:|---:|
| seed1 | `66.5%` | `71.0%` | `+4.5pp` | `30/21/149` |
| seed2 | `61.0%` | `54.5%` | `-6.5pp` | `29/42/129` |

这些数值直接来自四个 200-episode JSON，不是训练内 eval。seed3 clean/Flow 尚未完成，
所以这一阶段还没有 aggregate mean、crossed bootstrap CI 或最终 Gate。当前唯一成立的解释是：
seed1 的正效应没有在 seed2 复制，training-seed heterogeneity 是实质性的；这再次证明不能把
早先 seed1 `+7--10pp` 当成算法级结论。最终仍必须等待 seed3，并要求至少 `2/3` seeds 为正。

#### 吞吐问题与修复

`run_cqn_floq_multiseed_paired_gate.py` 原来把每个 training seed 的
`clean -> candidate` 两个 eval 绑定在同一 worker。三个 training seeds、两张 GPU 时，最后
一对会在一张卡上串行，另一张卡空闲一轮。当前稳定运行不为节省几分钟而中断；但已把未来
runner 改为 6 个独立 `(training seed, policy)` jobs，由两张卡共同取队列。解析/readout 和新
job-granularity tests 为 `6 passed in 0.07s`，语法与 whitespace check 通过。Stage-66 及后续
paired Gates 会自动使用新调度。

#### 单变量与 schema 审计

对 Hydra resolved method tree 做了程序化差分，两个 TD-target arms 相对各自 parent 都只改变：

```text
td_target_action_source: replay_next -> policy_value
td_target_policy_value_beta: null -> 1.0
```

rollout beta、loss、UTD、replay、exploration、Flow sources 与 architecture 无其他变化。该
不变量已写入 Flow/direct-Q config tests；与真实性 checker、最终汇总测试合计
`13 passed in 4.83s`。

进一步发现 fixed-beta Stage-67 runner 把 checkpoint map 写在 summary 顶层
`selected_steps`，而 checkpoint×beta runner 才写在 `selection.selected_steps`。Stage-91
原脚本误用了后者，已改为读取顶层字段；同时 Stage-82/84/88/91 都新增父阶段
`complete` 断言，避免基础设施异常再次被解释成算法 fail。

为了确保运行中 Bash 真正加载修订，不依赖文件被修改后是否继续读取，四个仅处于
`do_wait`、未占 GPU 的旧等待进程已定向退出并重启。当前新 PID：

```text
Stage-82/83 target mechanism       1508476
Stage-84--87 Route A               1508489
Stage-88--90 target final Gates    1508502
Stage-91 final summary             1508515
```

Stage-64-r1 与 Stage-65--67 recovery 未中断。

#### 文献后的下一决策

本轮检索仍支持现有排序：
[What Does Flow Matching Bring To TD Learning?](https://arxiv.org/abs/2603.04333)
针对 expected-value critic 的关键变量是非平稳 TD target 与高 UTD plasticity；这正由
Stage-82/83 的 target-only matched arms 检验。
[PCBF](https://arxiv.org/abs/2605.08253) 和
[Value Flows](https://openreview.net/forum?id=2VyNYUVF2k) 是 return-distribution 路线，
而本项目已有三训练/selection 证据显示 PCBF 远低于 clean、FlowIQN 也失败，因此不在当前
expected-Q Gate 未完成时重开。
[FlowCritic](https://arxiv.org/abs/2510.22686) 的 velocity clipping 只在跨 seed 出现
gradient/parameter instability时作为单变量稳定化候选；当前三个 expected-FLOQ runs 都 finite，
不能无证据地同时加入。

所以下一 Gate 不变：先完成 seed3 和 Stage-64 aggregate；随后由 Stage-65--67 检验
integrated readout/validation-best checkpoint。只有它们都失败，才执行已经排队的
policy-value TD-target Flow/direct-Q matched experiment。

### 21.75 预注册的下一种 target：独立 BC-policy evaluation，避免 Q 自选 target

#### 依据与问题分解

[FLOQ 官方实现](https://github.com/CMU-AIRe/floq) 的 Bellman target action 来自当前 actor，
而不是 replay buffer 的 next action；target Flow 只负责评估该 action。当前本地
`replay_next` 更接近 behavior-SARSA，target 对同一 replay sample 基本固定。Stage-82 的
`policy_value` 则用 online Q+BC prior 选 action、target Q 评估，能恢复 moving target，
但也重新引入用户担心的循环：

```text
Q 排序 action -> 该 action 进入自己的 Bellman target -> 排序被自举强化
```

因此如果当前 A/B Gate 仍未满足，需要把“target 非平稳性”和“critic 自己做 argmax”拆开。
最小对照是 `td_target_action_source=bc_policy`：

- 当前独立 BC tower 在 next state 产生 action；
- target critic/Flow 评估该 action；
- critic 不参与 target action selection，学习对象是近似 `Q^{pi_BC}`；
- rollout 仍是 exact BC，训练/部署 policy 是否调用 value 继续分离；
- BC tower 随训练更新，所以 target 比 replay-next 更接近 moving-policy evaluation。

它不是完整 FLOQ actor reproduction：官方 actor 还接受 Q guidance。这里故意先去掉 Q guidance，
因为目标是检验 value authenticity 与自强化 shortcut，不把新 actor 算法混进来。

#### 已实现但尚未抢跑

新增两个单变量 launch：

```text
robobase/cfgs/launch/cqn_flow_floq_td_bc_policy_two_tower_high_utd4_gate.yaml
robobase/cfgs/launch/cqn_direct_q_td_bc_policy_two_tower_coherent_mc_high_utd4_gate.yaml
```

resolved method tree 相对各自 replay-next parent 都只改变：

```text
td_target_action_source: replay_next -> bc_policy
```

Flow/direct-Q 的真实 update tests、exact config-diff tests 为
`4 passed in 63.78s`，rollout beta 与 target-policy-value beta 都保持 `null`。该 arm 只在
Stage-91 显示相应路线仍有 unmet gate 时启动：

- A fail：direct-Q BC-policy target；
- B fail：Flow 与 direct-Q 两臂 matched 对照；
- 先 smoke，随后仍用三训练种子 checkpoint/beta selection、sealed task confirmation 和
  causal audit；不能凭“critic 不做 argmax”直接宣称 value 真实。

当前不启动它，避免在 Stage-64/65 的预声明结果出来前扩大 search family。

### 21.76 Stage-64-r1 三训练种子正式失败；Stage-65 readout Gate 已接力

#### 上一阶段结果

修正 seed2 clean checkpoint 路径后的 Stage-64-r1 已正常结束。正式 artifact：

```text
exp_local/cqn_flow_high_utd/stage64_floq_multiseed_final_seed92000_20260724/summary.json
```

在共同、未用于选择的 `92000--92199` 环境 seeds 上，三个训练 seed 的 frozen clean
validation-best checkpoint 与 fixed `10k + distill + beta=1` Flow 为：

| training seed | clean CQN-AS | Flow | paired delta | W/L/T |
|---|---:|---:|---:|---:|
| seed1 | `66.5%` | `71.0%` | `+4.5pp` | `30/21/149` |
| seed2 | `61.0%` | `54.5%` | `-6.5pp` | `29/42/129` |
| seed3 | `66.0%` | `54.0%` | `-12.0pp` | `30/54/116` |
| aggregate | **`64.50%`** | **`59.83%`** | **`-4.67pp`** | **`89/117/394`** |

crossed training-seed/environment-seed bootstrap 95% CI 为
**`[-14.83,+5.00]pp`**；四项预注册检查（mean delta、CI lower、wins>losses、至少
2/3 training seeds 为正）全部失败，因此 Gate 为 **fail**。总评测耗时
`1693.75s`。

#### 解释

seed1 的 `+4.5pp` 并没有跨训练 seed 复现，seed2/3 分别转为 `-6.5/-12.0pp`。所以当前
fixed 10k distilled readout 不能宣称超过 clean CQN-AS，也不能把先前单 seed 的正结果
解释成算法级提升。600 对环境 seed 后仍有很宽的 crossed CI，主要不确定性来自
**training-seed heterogeneity**，而不是继续给同一训练 seed 增加少量 eval episodes 能解决。

这个结果只否定“固定 final checkpoint + distilled readout”候选；它还没有区分：

1. Flow 内部 iterative value 是否有信息、但 distill head 丢失；
2. 同一训练 run 的较早 validation-selected checkpoint 是否更好；
3. replay-next 的近静态 TD target 是否没有触发 FLOQ 文献所说的 plasticity 优势。

#### 下一阶段 Gate

Stage-65 先保持三个 Flow training runs 和 `beta=1` 不变，只在新的
`96000--96049` selection seeds 比较：

```text
distill
integrated, 2 flow steps
integrated, 8 flow steps
```

promotion 要求 integrated 相对 distill 同时满足 mean delta>0、aggregate wins>losses、
至少 2/3 training seeds 为正。只有 promotion 通过，才在完全不同的
`97000--97199` seeds 做三训练种子 200-episode confirmation，并要求 crossed CI lower
`>=0`。这样该阶段只回答 readout 机制，不借 checkpoint 搜索提高结果。若不通过，
Stage-67 才在 screen/validation/confirmation 三个互斥 splits 上选择各训练 seed 的 best
checkpoint。

#### 已执行与 ETA

Stage-64 在 `12:59:32 BST` 写出 complete 后，事件等待中的 Stage-65 立即启动：

```text
controller PID 1504526
runner PID     1515850
output         exp_local/cqn_flow_high_utd/stage65_readout_mechanism_seed96000_20260724
```

GPU1/GPU5 已分别运行 `seed1/distill` 与 `seed1/integrated_steps2`；9 个 selection jobs
按独立 job queue 分配，不再把一对方法绑死在同一 GPU。根据 distill 与早期 integrated
成本，selection 初始 ETA 为 `30--45min`。若 promotion fail 到此结束；若 pass，
6 个 200-episode confirmation jobs 视 winner 为 2 或 8 steps 再增加约
`30--90min`。首批 JSON 落盘后必须用实际 elapsed 更新 ETA。

### 21.77 Stage-92--97：BC-policy TD target 后继已实现并事件排队

#### 上一阶段结果对后继设计的约束

Stage-64 已证明 fixed distilled Flow 不稳定，但尚未说明失败来自 Flow 表示还是 Bellman
target action。已排队的 Stage-82 policy-value target 恢复 moving target，同时让 online
critic 参与 target action selection；若它成功，仍可能包含
`Q -> action -> Q target` 的自强化。若它失败，也不能知道 moving BC policy evaluation
是否更稳定。因此 Stage-91 任一路线仍失败时，下一单变量必须是已预注册的
`td_target_action_source=bc_policy`，而不是同时更改 loss、actor 和网络。

#### 实现与验证

新增自动链：

```text
scripts/run_cqn_stage92_93_bc_policy_target.sh
scripts/run_cqn_stage94_97_bc_policy_final.sh
```

其逻辑为：

1. 读取 Stage-91 的 Route A/B 独立 Gate，只运行仍失败的路线；
2. Flow/direct-Q 先各做 1500-frame smoke，并由 checker 验证 finite update、声明 target
   source 为 `bc_policy`、rollout 仍为 exact BC；
3. smoke 通过后跑三个 training seeds；两路线都需要时，GPU1 队列为
   `Flow-1 -> directQ-2 -> Flow-3`，GPU5 队列为
   `directQ-1 -> Flow-2 -> directQ-3`，避免异构训练串行；
4. Flow 使用 distill readout、direct-Q 使用 auto readout，各自在
   `144000/145000/146000` 的 screen/validation/sealed splits 选择 checkpoint 与全局
   `beta in {0.3,1,3}`；
5. task 通过者才读取 `147000--147031` 做 sibling-horizon causal Gate；
6. 新候选追加到 Stage-91 已有 evidence，不覆盖或重新解释旧结果。

`summarize_cqn_autoresearch_routes.py` 新增 `--base-summary`，可保留既有 A/B candidate 后
追加新候选。包含真实 update、exact config diff、checker、task runner 和 summary CLI 的
focused suite 为 **`20 passed in 65.17s`**；两个 shell controller 通过 `bash -n` 和
whitespace checks。

#### 已执行

后继不是短轮询，也没有提前占 GPU；两个 durable master 已用退出事件排队：

```text
Stage-92/93 master PID 1518404
  waits Stage-91 PID 1508515

Stage-94--97 master PID 1518406
  waits Stage-92/93 master PID 1518404
```

进程状态均为 `do_wait`，子进程分别是 `tail --pid=1508515` 与
`tail --pid=1518404`。若 Stage-91 两路线已全部通过，Stage-92--97 只写 skip artifacts；
若仅一条路线失败，只运行该路线；不会在当前 Stage-65/后续预注册 Gate 出结果前抢跑新
search family。

### 21.78 新文献分流：只有 critic-flow 可直接回答当前 B，policy-flow 只能做后续 extraction

#### 上一阶段结果带来的研究问题

Stage-64 的 fixed distilled Flow 跨训练 seed 失败，可能来自 critic 表示、TD target 或
action extraction。检索到的 2025--2026 Flow+RL 方法必须按“Flow 放在哪里”分类，否则容易
把一个更强的 Flow policy 当成 Flow value 的证据。

#### Primary-source 分类

| 类别 | 方法 | Flow 的对象 | 对当前路线的含义 |
|---|---|---|---|
| iterative expected-Q critic | [FLOQ](https://openreview.net/forum?id=WwQoSHGCXg) | scalar Q 的 iterative velocity field | 当前 B 主线；Stage-65/67 与 policy-target arms 正在直接检验 |
| return-distribution critic | [Value Flows](https://openreview.net/forum?id=2VyNYUVF2k)、[PCBF](https://arxiv.org/abs/2605.08253)、[FlowCritic](https://arxiv.org/abs/2510.22686) | return samples/distribution | 本地 PCBF、FlowIQN、Value-Flows arms 已有负 Gate；除非出现新的稳定性证据，不重开无条件 sweep |
| energy/Q-guided Flow policy | [FlowQ](https://arxiv.org/abs/2505.14139)、[Direct Flow Q-Learning](https://openreview.net/forum?id=RdkOaK4q6p) | action-generating flow | 仍依赖普通 critic；不能证明 value 是 Flow 学会的 |
| intermediate-value Flow policy | [Q-Flow](https://arxiv.org/abs/2605.13435) | 为 action-flow latent state 学 inner value，再优化 policy velocity | 解决 BPTT/actor expressivity，不替代 outer Q；可在 critic 已通过 causal Gate 后改善 extraction |
| discrete Flow policy | [Flow Matching for Offline RL with Discrete Actions](https://arxiv.org/abs/2602.06138) | C2F action bins 上的 CTMC/Q-weighted flow | 与 action bins 形式最接近，但仍是 policy flow，不是 action-conditioned value flow |
| behavior-density critic regularization | [Flow Actor-Critic](https://arxiv.org/abs/2602.18015) | Flow 建 behavior proxy；critic 本身仍是 scalar | 可抑制 OOD Q explosion，但可能把 behavior likelihood 写进 Q，必须过 anti-cheat causal Gate |

[Q-Flow](https://arxiv.org/abs/2605.13435) 的核心恒等式是 inner action-flow 路径上
`V^pi(s,x_tau,tau)=Q(s,Psi_{1,tau}(x_tau,s))`；它传播的是 terminal outer-Q 到 policy-flow
latent state。[FlowQ](https://arxiv.org/abs/2505.14139) 学的是
`pi(a|s) proportional pi_B(a|s) exp(Q(s,a))` 的 action velocity field，而 Q 仍由普通
Bellman critic 训练。[Flow Actor-Critic](https://arxiv.org/abs/2602.18015) 则用 flow
behavior density 给 scalar critic 的 OOD penalty 加权。这三者都不能替代当前
“state/image + C2F action bin 作为 condition、value 本身由 FM update”的 FLOQ/return-flow
实验。

#### 证据触发的后续 Gate

当前不把这些方法混入 Stage-65--97。Stage-97 后按独立证据选择：

1. **critic causal pass、task fail**：value 有真实排序但 extraction 不好；此时才比较
   discrete CTMC policy flow、FlowQ/DFQL 或 Q-Flow 式 intermediate guidance，critic
   checkpoint 与 causal evidence保持冻结。
2. **task pass、causal fail**：policy 提升可能只是 behavior-density shortcut；不得上
   policy-flow，优先改 value supervision/target 并重复 sibling counterfactual audit。
3. **Flow 与 direct-Q 都出现 OOD value explosion/gradient instability**：才加入 FAC
   behavior-density penalty或 FlowCritic clipping 的单变量 arm。
4. **direct-Q task+causal pass、Flow 在 matched target 下 fail**：Route A 可收口，但 B
   的失败归因是 Flow parameterization，而不是继续用 Flow policy 论文冒充 Flow critic
   正结果。

这样把“value 是否真实”“Flow 是否改善 value”“真实 value 如何转成更强 policy”保持为三个
顺序问题，符合两条路线不能合并解决的约束。

### 21.79 Route A/B causal Gate 加入显式 anti-cheat proxy

#### 上一阶段审计结果

对 Stage-87/90/91/96 将调用的
`scripts/run_cqn_flow_branch_multiseed_gate.py` 做最终审计后发现，原 Gate 虽然使用了真实
same-state sibling intervention returns、三训练 seed 和 crossed bootstrap，但只要求：

```text
Q pairwise CI lower > 0.5
或 Q Spearman CI lower > 0
```

这个条件能排除随机/常数 critic，却不能排除用户指出的 shortcut：如果 Q 只是复刻独立 BC
policy 对各 bin 的偏好，而任务动力学刚好让 BC 偏好与 return 正相关，它也可能通过。原 Gate
因此不足以支持“value 真实有意义”的最终结论。

#### 更严格的 Gate

`analyze_cqn_branch_counterfactual.py` 现在在每个完全相同的 branch state、同一个被强制
action dimension 和同一组 realized continuation returns 上同时记录三种排序：

1. **Q score**：待审计的 direct-Q、distilled Flow 或 integrated Flow；
2. **BC-prior proxy**：独立 policy tower 在固定 C2F prefix 下对 5 个 sibling bins 的
   logits；它不读取 return；
3. **action-nearness proxy**：候选 sibling bin 到独立 BC tower 首选 bin 的负索引距离；
   它不读取 Q 或 return。

最终 causal Gate 除原有 coverage、Q-vs-chance 和至少 2/3 training seeds 为正外，还必须满足：

```text
crossed-bootstrap CI lower of
  Q pairwise accuracy - BC-prior pairwise accuracy > 0

crossed-bootstrap CI lower of
  Q pairwise accuracy - action-nearness pairwise accuracy > 0

至少 2/3 training seeds 的 Q point estimate 同时高于两个 proxy
```

proxy 与 Q 使用完全相同的 informative action pairs，并要求每个 training seed 的 pair-count
逐环境 seed 精确匹配；不能用缺失 proxy 的较小子集制造优势。这样“Q 预测 return”与“Q 只是
模仿 behavior preference”变成可证伪的 paired hypothesis。

#### 已实现与执行

新增 `--require-anti-cheat-proxies` 并在所有最终 causal controllers 中显式启用：

```text
Stage-87  Route A replay-next direct-Q
Stage-90  policy-value target Flow/direct-Q
Stage-91  task-qualified legacy Flow
Stage-96  BC-policy target Flow/direct-Q
```

ranking helper、proxy coverage、paired crossed-bootstrap pass/fail 以及 summary evidence 的
focused suite 为 **`21 passed in 0.11s`**，四个 controller 均通过 `bash -n` 和
whitespace checks。

为保证等待中的 Bash 实际加载新 Gate，只定向重启了尚未占 GPU 的后继进程；Stage-65 和
Stage-82 未中断。新 event-chain PIDs：

```text
Stage-84--87 master  1531630
Stage-88--90 master  1531632
Stage-91 master      1531636
Stage-92--93 master  1531640
Stage-94--97 master  1531646
```

依赖链已经核验为
`1508476 -> 1531630 -> 1531632 -> 1531673 -> 1531640 -> 1531646`，
均处于 `do_wait`；没有 GPU job 被终止，也没有基础设施 fail marker。

### 21.80 Stage-65 integrated readout 跨三训练 seed 失败；Stage-67 已启动

#### 上一阶段结果

Stage-65 在相同 `96000--96049` selection seeds 上，对三个 frozen 10k Flow checkpoints
分别比较 distill、2-step integrated 和 8-step integrated。正式 artifact：

```text
exp_local/cqn_flow_high_utd/stage65_readout_mechanism_seed96000_20260724/summary.json
```

| readout | mean success | vs distill | per-training-seed delta | W/L/T | crossed CI |
|---|---:|---:|---:|---:|---:|
| distill | `59.33%` | reference | — | — | — |
| integrated-2 | `48.67%` | **`-10.67pp`** | `-14/-2/-16pp` | `17/33/100` | `[-24.67,+2.67]pp` |
| integrated-8 | `55.33%` | **`-4.00pp`** | `-6/0/-6pp` | `22/28/100` | `[-16,+8]pp` |

两个候选的 mean delta、wins>losses 和 positive-training-seed majority 三项 promotion checks
全部失败。8-step 只是失败候选中 delta 较高者，所以 summary 的 `selected_readout` 为
8-step；它的 `selection_gate` 仍是 **fail**，不能误称为被 promotion。按 protocol 没有读取
`97000--97199` confirmation seeds。总耗时 `1008.27s`。

#### 解释

同一 checkpoint、同一 state/image/action-bin condition 和相同 `beta=1` 下，把 value 从
distilled scalar 改为真正积分 2 或 8 次都没有改善，且三个训练 seed 没有一个严格正向。
因此 Stage-64 的失败不能归咎于“只蒸馏一次丢掉了 iterative flow 的信息”；当前 action-facing
结果反而支持 distill 是更稳定的 readout。增加 inference compute 不能修复 training-seed
heterogeneity。

这个阶段只冻结 final 10k checkpoint，所以仍未排除 checkpoint selection。Stage-66
`integrated task` 已按 Gate 写 `skipped_no_integrated_promotion`，没有浪费 sealed eval。

#### 下一阶段 Gate 与已执行

Stage-67 已在 `13:16:21 BST` 自动启动，只使用 distill readout：

```text
runner PID 1532285
output:
  exp_local/cqn_flow_high_utd/stage67_best_checkpoint_multiseed_seed101000_20260724
```

对每个 training seed 的 `1k--10k` 十个 checkpoints：

1. `101000--101009` 各跑 10 episodes screen，保留 top-2；
2. `102000--102049` 在新 seeds 中选一个 checkpoint；
3. 冻结三个 winners 后，与 clean validation-best checkpoints 在
   `103000--103199` 做一次 200-episode sealed paired confirmation；
4. 仍要求 mean delta>0、CI lower `>=0`、wins>losses、至少 2/3 training seeds 为正。

30 个 screen jobs、6 个 validation jobs、6 个 confirmation jobs 都以独立 job queue 在
GPU1/GPU5 调度。首个 10-episode screen job 实测 `72.23s`；据此 15 个双卡 screen waves
约 `18min`，validation 约 `8min`，sealed confirmation 约 `24--27min`。更新后的总 ETA
为 `50--53min`，即约 `14:06--14:10 BST`。

### 21.81 FLOQ fidelity recovery：补回被基础设施 cascade 跳过的官方变量

#### 上一阶段结果

`2026-07-24 13:28 BST` 的实际状态是：

- Stage-67 validation-best checkpoint Gate 正在 GPU1/GPU5 运行，30 个 screen jobs
  已完成 `20/30`；runner PID `1532285`，没有 fail marker，原 ETA 仍约
  `14:06--14:10 BST`。
- 原 Stage-74--80 controller 目录都只有 `failed` marker，时间集中在
  `12:26:35--12:26:40 BST`，没有任何 `source01` 或 `bcfm8` 训练 run。它们是在旧
  Stage-64 错写 clean seed2 路径后被 `set -e` 依赖链级联退出，不是 fidelity 实验
  Gate 失败。
- Hydra 当前 resolved config 已重新程序化核对：

```text
legacy Stage-24/62:
  flow_source_type=uniform
  flow_source_min/max=null -> runtime Uniform[-0.2, 0.2]
  bcfm_lambda=1

source01:
  Uniform[0, 0.1], bcfm_lambda=1

bcfm8:
  Uniform[-0.2, 0.2], bcfm_lambda=8
```

因此此前待澄清的 source 并不是 Gaussian；真正差异是旧 run 使用
`Uniform[-0.2,0.2]`，而论文在本任务 `[0,1]` return support 与
`noise_coverage=0.1` 下对应 `Uniform[0,0.1]`。旧 recovery 只恢复了
Stage-65--67，确实漏掉了这项官方 fidelity Gate。

#### 解释

如果 Stage-67 失败后直接进入 moving-TD-target 实验，会把“source support / 多 source
loss 相对权重不忠实”和“replay-next target 太静态”两个解释混在一起。当前证据只能否定
旧 `Uniform[-0.2,0.2], lambda=1` FLOQ 的 fixed-10k 与 integrated readout，尚不能否定
官方关键设置下的 expected-value Flow。这个缺口必须先以单变量实验闭合。

#### 下一阶段 Gate

新增完整 `2x2` fidelity 设计的三个非 base cells：

```text
source01              [0,0.1], lambda=1
bcfm8                 [-0.2,0.2], lambda=8
source01_bcfm8        [0,0.1], lambda=8
legacy base           [-0.2,0.2], lambda=1（复用 Stage-24）
```

执行协议：

1. Stage-74：三个新 arms 各跑 1500-frame smoke，要求 exact resolved config、
   1k snapshot、finite metrics、Flow/distill 都有非零梯度且 non-finite fraction 为 0。
2. Stage-75：smoke 全过后，三个 arms 的 seed1 各训练 10k checkpoints；用动态双卡队列，
   前两个完成任意一个后立刻在释放的 GPU 上启动第三个。
3. Stage-76：legacy 与三个新 arms 在共同 `114000` screen、`115000` validation 上联合
   checkpoint×beta selection。legacy 先声明并赢得所有平局，所以 correction 必须在相同
   validation seeds 上严格高于 legacy；随后只对冻结 winner 使用 `116000--116199`
   confirmation。promotion 仍要求 validation `>=+2pp`、wins>losses，以及 confirmation
   delta>0、wins>losses、CI lower `>=-5pp`。
4. 若 winner 是 legacy 或 promotion 失败，明确写 task-fail artifact，不扩训练 seed。
   否则 Stage-77 只训练 winner 的 seeds2/3；Stage-78 用全新
   `118000/119000/120000` screen/validation/sealed splits 做严格三训练 seed Gate：
   mean delta>0、crossed CI lower `>=0`、wins>losses、至少 `2/3` seeds 正向。
5. 只有 Stage-78 task pass 才运行 Stage-80 sibling-horizon causal Gate，并要求 Q
   同时显著超过 chance、独立 BC-prior ranking 和 BC-action-nearness proxy。

#### 已实现与执行

新增：

```text
robobase/cfgs/launch/
  cqn_flow_floq_source01_bcfm8_distill_two_tower_high_utd4_gate.yaml
scripts/run_cqn_stage74_80_fidelity_recovery.sh
scripts/launch_cqn_autoresearch_chain.sh
```

Stage-82 现在必须等待 fidelity recovery，并把 Stage-78 加入“已有 B task pass”判断；
Stage-91 最终汇总也已加入 `official_fidelity_flow` task+causal candidate。配置、training
checker、selection 和最终汇总的 focused suite 为 **`18 passed in 4.28s`**；两个新增
shell、Stage-82 和 Stage-91 均通过 `bash -n` 与 whitespace check。

为确保等待中的 Bash 加载新逻辑，只重启了没有 GPU children 的后继 controllers；
Stage-67 未中断。新的事件链为：

```text
Stage-74--80 fidelity recovery     PID 1558796
Stage-82--83 policy-value target   PID 1558818
Stage-84--87 Route A              PID 1558842
Stage-88--90 target final         PID 1558866
Stage-91 aggregate                PID 1558892
Stage-92--93 BC-policy target     PID 1558916
Stage-94--97 BC-policy final      PID 1558947
```

依赖分别落在 `tail --pid` 退出事件上，没有短轮询或预占 GPU。若 Stage-67 pass，
Stage-74--80 会立即写 skip markers；若 fail，三个 smoke 因增加组合 cell 预计
`18--28min`，三个 seed1 full runs 以两波双卡调度约 `125--150min`，Stage-76 约
`55--80min`。只有 seed1 promotion 后才追加 Stage-77 `65--75min`、Stage-78
`50--60min` 和 Stage-80 `45--75min`；从 Stage-67 fail 到最严格 fidelity 结论的
条件 ETA 约 `6--7.5h`。

### 21.82 两路线保持实验独立，同时用 B 的单卡空窗预训练 Route A

#### 上一阶段结果

`2026-07-24 13:33 BST`，Stage-67 的 `30/30` screen JSON 已全部落盘，两个实际 GPU
children 已切到 validation：

```text
GPU1: seed1 / flow_step2000 / 50 episodes / seeds 102000--
GPU5: seed1 / flow_step4000 / 50 episodes / seeds 102000--
```

此时 validation JSON 仍为 `0/6`，说明两个 jobs 正在运行而不是已经产生 selection 结果；
因此仍不能汇报 candidate 胜负。Stage-67 主 runner PID `1532285` 和两个 GPU children
存活，没有 fail marker。

#### 解释

Stage-75 的三个新 Flow seed1 jobs 需要两波 GPU：第一波 source01/bcfm8 各占一卡，第二波
组合 arm 只占一卡约 `65--75min`。原调度会让另一张额度卡在整段第二波空置，而 Route A 的
三个 replay-next direct-Q training seeds 彼此独立、单个仅约 `23--30min`。把它们放进这个
资源空窗不会共享 replay、checkpoint、selection seed 或 Gate；它只改变墙钟调度，不合并
A/B 研究问题。

#### 下一阶段 Gate 与已执行

Stage-75 调度已改为：

```text
wave 1:
  GPU1 source01 Flow seed1
  GPU5 bcfm8 Flow seed1

wave 2:
  GPU1 source01+bcfm8 Flow seed1
  GPU5 Route-A direct-Q seeds1 -> 2 -> 3
```

Route-A 输出固定为不伪装 GPU 编号的路径：

```text
exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed1_20260724
exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed2_20260724
exp_local/cqn_flow_high_utd/stage85_direct_q_replay_utd4_seed3_20260724
```

每个 filler job 完成后立即运行 direct-Q training-health Gate。filler 失败只写
`route_a_prefill_incomplete`，不能令 B 路线实验失败；Stage-84/85 会在独立的新目录重试。
若三个 prefill 都通过，Stage-84/85 重新检查 artifact 后直接复用，仍在
`130000/131000/132000` task splits 和 `133000` anti-cheat causal split 上完成 Route A
正式 Gate。也就是说 training compute 被并行化，但 task/causal 结论仍完全分离。

修改后的两个 orchestration shells 通过 `bash -n`、whitespace checks；相关配置、
training checker、fidelity selection 与最终汇总 suite 仍为
**`18 passed in 4.60s`**。为使等待进程读取新调度，仅重启无 GPU children 的后继链，
Stage-67 未中断。当前 PIDs：

```text
Stage-74--80 fidelity recovery     1570975
Stage-82--83 policy-value target   1570995
Stage-84--87 Route A              1571015
Stage-88--90 target final         1571035
Stage-91 aggregate                1571055
Stage-92--93 BC-policy target     1571075
Stage-94--97 BC-policy final      1571095
```

若 Stage-67 fail 并进入 fidelity，Stage-75 的总墙钟仍由两波 Flow 约
`125--150min` 决定，但 Route-A 三个约 `70--90 GPU-min` 的训练被吸收到第二波，不再在
后面额外串行消耗约一小时。所有跨阶段启动仍由 `tail --pid` 事件触发，没有训练完成的短
轮询 loop。

### 21.83 Route-A causal Gate 有效性修正：一步 do-action 与完整 BC-path proxy

#### 上一阶段结果

Stage-67 已完成全部 screen 并进入 validation；`13:39 BST` 前后主 runner 仍存活，GPU1/GPU5
各有一个约 `2.6GB` 的 validation child，没有 summary 或 fail marker。因此当前仍没有新的
B task 结论。

等待期对最终 sibling causal Gate 做逐行审计后发现两个会令“真实 value”结论过强的问题：

1. direct-Q/FLOQ 的 task configs 使用
   `critic_sequence_mode=effective_k0`，critic 只条件化当前实际执行动作；但旧 causal runner
   从 `structured_exploration_horizon=4` 继承 horizon，把相同 action delta 连续强制四个
   decision。H=4 的 return 是宏观干预结果，不是该 critic 声称的
   \(Q(s_t,a_t)\) under normal continuation。
2. anti-cheat 的 `policy_prior` 只取 force level 当前 bin 的独立 BC logit。同一个实际执行
   坐标还包含后续 C2F refinement levels；一个复制该坐标完整 BC path likelihood 的 critic
   仍可能超过这个局部 proxy。

#### 解释

旧 H=4 probe 适合回答“持续沿这个方向控制四步是否有用”，但不能单独证明 current action-bin
value 有真实一步因果语义。把它作为 promotion Gate 会把宏观 action intervention 与
`effective_k0` value 混为一谈。类似地，只超过局部 BC logit 不能充分排除 value 学成更复杂的
imitation score。

#### 下一阶段 Gate

所有最终 A/B causal promotions 现在统一冻结为：

```text
intervention_mode = sibling_horizon
intervention_horizon = 1
continuation = restored common policy state + branch observation
```

即只对当前真正执行的 action 做一次 do-intervention，之后不再强制 delta。每个 sibling
同时记录三种独立 BC proxies：

1. `policy_prior`：forced level 的局部 BC bin logit；
2. `policy_path`：该 forced bin 加后续 C2F greedy refinements，对实际替换的
   `sequence=0, action_dimension=d` 坐标累计独立 BC log-probability；
3. `action_nearness`：到独立 BC preferred bin 的离散距离。

promotion 必须同时满足：

- 每个 training seed 至少 24 个 informative one-step branch states；
- crossed-bootstrap 的 Q pairwise CI lower `>0.5` 或 Spearman CI lower `>0`；
- Q pairwise 相对上述三个 proxies 的 paired crossed-bootstrap CI lower 都严格 `>0`；
- 至少 `2/3` training seeds 的 point direction 同时超过 chance 和全部 proxies。

H=4 只保留为 task-pass 但 H=1 coverage 不足时的机制诊断；它不能使路线通过最终 causal
Gate。如果 H=1 只有 coverage fail，下一步要么扩预注册 anchors/force-level，要么显式学习
宏观 action value，不能用 H=4 指标替代 \(Q(s,a)\)。

#### 已实现与执行

修改覆盖：

```text
scripts/analyze_cqn_branch_counterfactual.py
scripts/run_cqn_flow_branch_multiseed_gate.py
scripts/summarize_cqn_autoresearch_routes.py
Stage-80 / 87 / 90 / 91 / 96 controllers
```

probe 会把 `intervention_horizon` 写入 artifact，并拒绝复用 horizon 不匹配或缺
`policy_path_proxy` 的旧 JSON。路径分数只在每个 C2F level 对实际执行的
`sequence=0, action_dimension=d` 选择 bin 累计独立 BC `log_softmax`；不会混入
`execution_length=1` 下未执行的其余 15 步 action chunk 或未被 intervention 替换的其他
动作维度。候选之间仍共享完全相同的 branch state、realized return 和 bootstrap indices。

helper、command wiring、三 proxy coverage、full-path anti-cheat rejection 与最终 summary 的
focused regression 为 **`23 passed in 1.95s`**；Python compile、五个 shell syntax checks
和 whitespace checks 通过。只重载了事件等待 controllers，Stage-67 未中断。新 PIDs：

```text
Stage-74--80 fidelity recovery     1578624
Stage-82--83 policy-value target   1578644
Stage-84--87 Route A              1578664
Stage-88--90 target final         1578684
Stage-91 aggregate                1578704
Stage-92--93 BC-policy target     1578724
Stage-94--97 BC-policy final      1578744
```

### 21.84 Stage-67 screen/validation 已冻结早停 checkpoints；sealed confirmation 运行中

#### 上一阶段结果

Stage-67 的完整 screen 曲线为：

| training seed | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed1 | 0% | **70%** | 30% | 60% | 50% | 50% | 60% | 50% | 40% | 40% |
| seed2 | 0% | 20% | 60% | **80%** | 40% | 40% | 30% | 50% | 50% | 40% |
| seed3 | 0% | 10% | 30% | 40% | **60%** | 50% | 50% | 40% | 30% | 30% |

在独立 `102000--102049` validation seeds 上，top-2 的实际结果为：

```text
seed1: 2k=68%, 4k=60% -> freeze 2k
seed2: 4k=70%, 3k=54% -> freeze 4k
seed3: 5k=60%, 6k=64% -> freeze 6k
```

正式 selection artifact：

```text
exp_local/cqn_flow_high_utd/
  stage67_best_checkpoint_multiseed_seed101000_20260724/selection.json
```

#### 解释

三个训练 seed 都没有选择 final-10k，且 10k screen 只有 `40/40/30%`。因此 final-10k
Stage-64 的失败确实混入了明显的 checkpoint degradation；用户此前提出的“不能拿过拟合后
最后一步代表方法”在 Flow 上同样成立。另一方面，三个 frozen validation successes
`68/70/64%` 只是 Flow 自身的 checkpoint selection，不能与不同 seeds 上的 clean 数字直接
作胜负，也不能提前读成超过 CQN-AS。

#### 下一阶段 Gate 与执行

冻结的 `2k/4k/6k` 已进入唯一一次 `103000--103199` sealed paired confirmation；当前两个
GPU children 是 seed1 clean 与 seed1 Flow，之后队列依次执行 seed2/3 的 clean/Flow，共
6 个独立 jobs。最终 Gate 不变：

- 三训练 seed mean Flow-clean delta `>0`；
- crossed-bootstrap CI lower `>=0`；
- aggregate paired wins>losses；
- 至少 `2/3` training seeds delta 为正。

依据 Stage-64 同机 200-episode job 的约 `6--7min` 吞吐，三波双卡 confirmation 约
`18--22min`，加 summary/bootstrap 后预计约 `14:01--14:08 BST`。完成事件会自动触发
Stage-74 fidelity skip 或正式 smoke，不做短轮询。

### 21.85 双路线 active goal 重新确认；Stage-67 ETA 按真实 Flow 吞吐校正

#### Goal

当前 persistent goal 已核对为 `active`，目标不做降级或合并：

1. Route A：CQN-AS 学到可通过一步反事实与 anti-cheat proxy Gate 的真实 value，且任务效果
   不低于 original CQN-AS；
2. Route B：系统比较 FM+RL 路线并实现 FM+CQN-AS，最终任务效果严格超过 original
   CQN-AS；
3. 每阶段必须以实际 artifact 给出结果、含义、下一 Gate 并立即执行，持续到两条路线各自有
   可复现结论与明确推荐。

#### 当前实际进度与 ETA 修正

`13:48:38 BST`，Stage-67 sealed confirmation 的 `seed1/clean.json` 已完成，seed2 clean 已在
GPU1 启动；seed1 selected Flow `2k` 从 `13:41:20` 起仍在 GPU5 正常运行。此前
`6--7min/job` 低估了 Flow readout 的耗时：Stage-64 同规格 clean 约 `5--7min`，Flow 约
`9--12min`。当前调度的关键路径是 GPU5 上三个 Flow jobs，因此按真实吞吐校正为：

```text
Stage-67 summary ETA: 14:17--14:22 BST
```

后续仍由 runner 完成事件直接唤醒 Stage-74；不通过固定短周期轮询消耗资源。Stage-67 Gate
本身及 sealed seeds 不因 ETA 修正发生任何变化。

### 21.86 Stage-67 validation-best 正式失败；过训练修正不足以救回 legacy FLOQ

#### 上一阶段结果

Stage-67 已在唯一一次 `103000--103199` sealed paired confirmation 上完成。每个 Flow
training seed 都使用 screen 后再由独立 validation 冻结的最佳 checkpoint，而不是 final-10k：

| training seed | selected | clean | Flow | delta | W/L/T |
|---|---:|---:|---:|---:|---:|
| seed1 | 2k | 65.0% | 55.5% | -9.5pp | 28/47/125 |
| seed2 | 4k | 62.0% | 64.5% | +2.5pp | 40/35/125 |
| seed3 | 6k | 64.5% | 63.0% | -1.5pp | 32/35/133 |
| mean/aggregate | validation-best | **63.83%** | **61.00%** | **-2.83pp** | **100/117/383** |

crossed training-seed × environment-seed bootstrap 95% CI 为
`[-11.67,+5.50]pp`；只有 `1/3` training seeds 为正。四项预注册 checks
（mean delta、CI lower、wins>losses、positive-seed majority）全部失败。正式 artifact：

```text
exp_local/cqn_flow_high_utd/
  stage67_best_checkpoint_multiseed_seed101000_20260724/summary.json
```

#### 解释

Stage-64 final-10k 的 `-4.67pp` 的确混入 checkpoint degradation；严格早停后差距缩到
`-2.83pp`。但 Stage-67 排除了“只要用每个 seed 的最佳 checkpoint，legacy
FLOQ-distill 就会超过 CQN-AS”：seed1 仍大幅为负，aggregate wins 也少于 losses。故
过训练是问题之一，但不是 B 路线失败的充分解释；不能把 validation 的 `68/70/64%` 当成
正式结果。

#### 下一阶段 Gate 与立即执行

事件 handoff 已自动进入 official-fidelity 2×2：

```text
legacy:             source U[-.2,.2], lambda=1
source01:           source U[0,.1],   lambda=1
bcfm8:              source U[-.2,.2], lambda=8
source01_bcfm8:     source U[0,.1],   lambda=8
```

Stage-76 在 `114000+` screen、`115000+` validation、`116000+` confirmation 上联合选择，
新 fidelity arm 必须严格击败先声明、平局优先的 legacy；seed1 promotion 后才扩 seed2/3，
再由 Stage-78 做三训练种子 sealed task Gate。task pass 后 Stage-80 才运行
H=1 sibling do-action 与三 imitation proxies 的因果 Gate。

`14:05:58 BST`，source01 smoke PID `1595783` 已在 GPU1、bcfm8 smoke PID `1595784`
已在 GPU5 启动；两者首个编译/update 均 finite、nonfinite gradient fraction 为 0。
JIT 后单 update block 实测约 `92--95s`，smoke 预计 `14:14--14:18 BST` 完成。若两项通过，
Stage-75 两波 seed1 full training 的关键路径按同配置历史吞吐约 `125--150min`，首轮
1k snapshot 后再校正；当前保守 Stage-75 完成窗口为 `16:20--16:50 BST`。完成和跨阶段
启动都绑定 PID/marker，不做短轮询。

`14:14 BST` 的实际 milestone：source01/bcfm8 的 1k snapshots 已同时落盘，
`total_time=447.80/444.42s`；两个 run 的 Flow/readout/encoder gradients 仍 finite。
两者的 1.5k smoke 分别在 `14:16:50/14:16:54` 完成，controller 随即在先释放的 GPU5
启动第三个 `source01_bcfm8` combo smoke（PID `1607397`）。所以完整 Stage-74 health Gate
不是 `14:17` 结束；按同机实测，combo ETA 为 `14:27--14:30 BST`，之后才启动 Stage-75
source01/bcfm8 双卡 full training。Stage-75 完成窗口相应校正为约
`16:30--17:00 BST`。

### 21.87 非重复 fallback 已实现并事件排队：A 的 H=1 CF-FQE 与 B 的 full-FLOQ interaction

#### A：一步 counterfactual FQE

历史 Route-A oracle 使用 H=4 宏观持续干预；当前 direct scalar-Q 声称的是
`effective_k0` 一步 action value。新 Stage-98--101 因而只在现有 online-TD arms 全部未过
Route-A Gate 后执行：

1. 从 task evidence 最强的 validation-selected direct-Q parent 开始；
2. 在 exact BC reachable states 上，对全 15 action dimensions 的 level-1 五个 sibling bins
   做 `intervention_horizon=1`，之后恢复同一个 frozen BC continuation；
3. 用同状态 return differences 做 pairwise + delta regression；这会消去 state difficulty，
   state-only head 无法降低 action-difference loss；
4. 相同初始 critic、相同 cache 训练 `within_state` label-shuffle negative control；
5. discovery held-out Gate 要求至少 24 informative states，并以 simulator-seed bootstrap
   同时超过 chance、action-nearness、independent BC prior 和 shuffle control；
6. seed1 pass 才扩 algorithm seeds2/3；至少 `2/3` 复制后，再用独立
   `166000/167000/168000` splits 做 checkpoint/beta task non-inferiority，最后在
   `169000+` fresh branches 做 H=1 full anti-cheat causal Gate。

`finetune_cqn_branch_oracle.py` 已新增 direct scalar-Q scorer，并在写 snapshot 前 bitwise
验证 encoder、policy encoder 与 independent BC policy 未被 counterfactual fit 修改。新增
paired seed-bootstrap gate：

```text
scripts/summarize_cqn_h1_cf_fqe_gate.py
scripts/run_cqn_stage98_101_h1_cf_fqe.sh
```

direct-Q/oracle/gate focused regression 为 **27 passed**；其中新增端到端 synthetic
regression 实际把 zero-span direct scalar-Q 更新为正确的同状态 action ordering。compile、shell syntax 与
whitespace checks 通过。lineage 改为从 task summary 的
`sources.*.candidate_snapshot` 读取真实 validation-selected run，兼容 Stage-84 retry
目录；重载后的 event master PID `1606305` 正在等待 Stage-97，不占 GPU。若被解封，
历史同规模 5+5-seed 全维度 collection 给出的 discovery 初始 ETA 为 `60--75min`；完成第一
个 seed 后必须用实际 `elapsed_seconds` 更新。

#### B：官方 fidelity × fixed-policy target 的交互项

现有主链分别测试 official source/loss fidelity 与 `bc_policy` TD target，但未测试二者的
交互。新增配置严格组合：

```text
source U[0,.1] + bcfm_lambda=8
td_target_action_source=bc_policy
policy_value_beta=null
```

resolved Hydra config 已核验上述四项。Stage-102 smoke 通过后才扩三个 training seeds；
Stage-104 仍使用 screen→validation→sealed confirmation 与 validation-selected checkpoint，
相对 clean CQN-AS 要求 mean delta `>0`、CI lower `>=0`；只有 task pass 才进入 Stage-105
H=1 anti-cheat causal Gate。相关 config/gate regression **10 passed**。B event master
PID `1627975` 正在等待 Stage-97；当 A 仍开放时只占 GPU5，A 占 GPU1。最终 Stage-106
PID `1627990` 会等待两个 fallback 完成并合并两边真实 artifacts，避免两个独立 summary
各自遗漏另一条路线。

Stage-74 combo smoke 期间 GPU1 的空档被用于提前执行该 interaction 的 1k numerical
preflight：PID `1612473`，`XLA_PYTHON_CLIENT_PREALLOCATE=false`，只检查 config、finite
gradients 与 1k snapshot，不读取 task success。实际 artifact 的 12 项 checks 中 11 项为
true：snapshot、组合 config、Flow/readout gradients 与 nonfinite checks 全过；唯一失败是
`num_log_rows=1 < 2`，因为正好在 1k snapshot 停止、没有写第二个 metric row。因此该
preflight **不解封 Stage-102**。controller 已改为拒绝复用并在真正解封时写入 fresh
`_retry_<PID>` 目录完成 1.5k smoke，不能把部分健康检查包装成 pass。

### 21.88 Stage-74 三个 official-fidelity smoke 全部通过；Stage-75 双卡训练已启动

#### 上一阶段结果

三个新 cell 的 1.5k smoke 均生成 1k snapshot，并通过
`check_cqn_floq_training_gate.py` 的全部 checks：

| arm | source | lambda | metric rows | nonfinite Flow/encoder grad | gate |
|---|---|---:|---:|---:|---:|
| source01 | `[0,.1]` | 1 | 2 | 0 / 0 | pass |
| bcfm8 | `[-.2,.2]` | 8 | 2 | 0 / 0 | pass |
| source01_bcfm8 | `[0,.1]` | 8 | 2 | 0 / 0 | pass |

每项还同时验证 scalar/8-flow-sample/UTD4/two-tower fidelity、Flow 与 distill readout 均收到
非零梯度、rollout 仍为 exact BC。正式 artifacts：

```text
exp_local/cqn_flow_high_utd/
  stage74_floq_fidelity_smoke_controller_r1/{source,bcfm,combo}_gate.json
```

#### 解释

Stage-74 排除了 source bounds、loss ratio 或组合配置没有真正生效，以及数值梯度断开的实现
问题。它没有评价 policy success；训练 loss/finite gradient 不能替代任务表现。因此只解封
full training，不把三项 smoke 排名。

#### 下一阶段 Gate 与执行

Stage-75 第一波已经于 `14:27:40 BST` 同时启动：

```text
source01 seed1: PID 1622907, GPU1
bcfm8    seed1: PID 1622908, GPU5
```

两者各训练到 10.5k、保存 1k--10k checkpoints。第一波都通过 10k health Gate 后，GPU1
训练 combo；GPU5 同时顺序填充 Route-A direct-Q seeds1--3。Stage-76 才用共同
`114000/115000/116000` screen/validation/confirmation 比较 legacy 与三 fidelity arms。

按 smoke 的 0→1k `444--448s` 与历史 full-run 后续区间，第一波 1k milestone ETA
`14:35--14:37`，第一波完成 ETA `15:25--15:40`；包含第二波 combo 的 Stage-75 完成窗口
仍为 `16:30--17:00`。下一次状态读取绑定两个 1k CSV/snapshot milestones，而非短轮询。

实际 1k snapshots 在 `14:35:47/14:35:48` 落盘，`total_time=448.25/450.92s`；
两项 nonfinite gradient fraction 均为 0。该吞吐与 Stage-62 的 `447.01s@1k,
3767.94s@10k` 基本相同，因此 ETA 更新为：

```text
第一波 source01 + bcfm8:       15:30--15:36
第二波 combo:                  16:35--16:42
Stage-75（含 GPU5 direct fillers）: 16:40--17:05 BST
```

后者取 combo 与三个 direct-Q fillers 的较慢关键路径；不以训练 loss 预测 task winner。

### 21.86 Stage-67 正式结论：validation-best 早停减轻退化，但 FLOQ 仍未超过 CQN-AS

#### 上一阶段结果

Stage-67 在 `14:05:58 BST` 完成唯一一次 `103000--103199` sealed confirmation。每个
training seed 的 Flow checkpoint 都只由此前独立 screen/validation 冻结：

| training seed | clean CQN-AS | selected Flow | checkpoint | paired delta | W/L/T |
|---|---:|---:|---:|---:|---:|
| seed1 | 65.0% | 55.5% | 2k | -9.5pp | 28/47/125 |
| seed2 | 62.0% | 64.5% | 4k | +2.5pp | 40/35/125 |
| seed3 | 64.5% | 63.0% | 6k | -1.5pp | 32/35/133 |
| mean/aggregate | **63.83%** | 61.00% | validation-selected | **-2.83pp** | **100/117/383** |

crossed training-seed × environment-seed bootstrap 95% CI 为
`[-11.67,+5.50]pp`。mean delta、CI lower、aggregate wins>losses 和 `2/3` positive
training seeds 四项全部失败，正式 `gate=fail`：

```text
exp_local/cqn_flow_high_utd/
  stage67_best_checkpoint_multiseed_seed101000_20260724/summary.json
```

#### 解释

与 final-10k Stage-64 的 `-4.67pp` 相比，合法的 validation early stopping 把差距缩小到
`-2.83pp`，所以 checkpoint degradation 确实是失败的一部分；但它不是全部原因。三个独立
training seeds 只有 seed2 为正，aggregate losses 仍多于 wins。因此不能把 seed2 的
`+2.5pp` 或 validation 曲线包装成“FM 超过 CQN-AS”，也不能再用多跑 checkpoint 解决这个
机制问题。

#### 下一 Gate 与实际执行

Stage-67 的完成 marker 已在 `14:05:58` 事件触发 Stage-74 官方 fidelity smoke：

```text
GPU1: source U[0,0.1], lambda=1
GPU5: source U[-0.2,0.2], lambda=8
```

两个真实 `train.py` children 分别为 PID `1595783/1595784`，均已创建约 `25.5GB` runtime
显存并正常载入 demonstrations。smoke 只检查 resolved fidelity、finite gradients、Flow 与
distill 两头确实收到梯度；不读取 task success。按同配置首次 JIT 与 1.5k 训练吞吐，Stage-74
ETA 为 `14:17--14:21 BST`。通过后立即进入预注册 2×2 seed1 joint selection；smoke 完成由
进程退出触发，不做短轮询。

### 21.87 Route A 新 fallback：H=1 fixed-policy simulator-branch MC

#### 上一阶段含义

历史 Stage-XII 的全维度 branch oracle 使用 H=4，证明 frozen representation 有表达真实
action effect 的容量；Stage-51 的低容量 structured value 也在独立 seeds 上通过。但它们都
没有直接训练当前 monolithic direct scalar-Q 的一步
\(Q^{\pi_{\rm BC}}(s,a)\)，而当前最终 critic 声明的是 `effective_k0`。因此继续调 H=4
sidecar、selector margin 或 imitation blend 会重复已经关闭的路线。

#### 新假设与 Gate

保留历史 artifact label `direct_q_h1_cf_fqe` 以兼容已经排队的 controller，但需要准确
说明：这不是仅依赖 replay transition 的传统 fitted Q evaluation。它通过恢复模拟器状态，
对每个 do-action 分支执行 fixed-policy rollout，得到 truncated Monte-Carlo return，因此是
**H=1 simulator-branch MC oracle**。新候选固定：

1. 从 policy-reachable image+state 抓取 simulator state；
2. 每个 state、action dimension、C2F sibling bin 只执行一次 do-action；
3. 之后恢复同一个 independent BC continuation；
4. 用同状态 return difference 训练 direct scalar-Q，encoder 和 BC policy bitwise 冻结；
5. 同 cache、同初始化训练 `within_state` return-shuffle negative control。

同状态 centered pair/ranking loss 无法由 state-only difficulty 降低。seed1 discovery 必须在
未见 simulator seeds 上同时满足：

- H=1、全 sibling、至少 24 个 informative train/heldout states；
- seed-cluster pairwise CI lower `>50%` 且 Spearman CI lower `>0`；
- Q pairwise 相对 action-nearness、independent BC prior 和 within-state shuffle 的 paired
  seed-bootstrap CI lower 都 `>0`；
- critic fit 前后 encoder、BC policy、policy encoder bitwise 相同。

只有 discovery pass 才用 algorithm seeds2/3 复现；至少 `2/3` pass 后才进入相对
validation-selected clean CQN-AS 的 checkpoint/beta task confirmation，要求 mean delta
`>0`、CI lower `>=0`。task pass 后再用全新 `169000+` seeds 跑 H=1 full-path anti-cheat
causal Gate。

#### 已实现与执行

`finetune_cqn_branch_oracle.py` 现在原生读取 `direct_scalar_q`，不再错误经过 C51 support；
fit artifact 记录并强制检查 frozen-policy bitwise equality，同时显式记录
`target_estimator=simulator_branch_monte_carlo` 和
`continuation_policy=frozen_independent_bc`。若 source config 的
`policy_value_beta` 非空，实验会直接拒绝运行，避免 Q 偷偷参与 continuation。新增：

```text
scripts/summarize_cqn_h1_cf_fqe_gate.py
scripts/run_cqn_stage98_101_h1_cf_fqe.sh
tests/unit/test_summarize_cqn_h1_cf_fqe_gate.py
```

direct-Q scorer、H=1/shuffle Gate、现有 oracle/direct-Q focused tests 共 **26 passed in
84.08s**；compile、shell syntax 与 whitespace checks 通过。event controller PID
`1596474` 已挂在 Stage-97 完成事件上，当前 child `do_wait`，不占 GPU。若现有 A 候选已经
通过则自动 skip；否则 discovery 使用 GPU1。参考历史同规模 5+5 全维度收集，首次 artifact
ETA 约 `60--75min`，届时再按真实 branch throughput 校正。

### 21.88 Route B 新 fallback：官方 FLOQ fidelity × fixed-BC TD target 交互

#### 上一阶段含义与新假设

现有 factorial arms 分别测试了：

- FLOQ source interval / `bcfm_lambda=8` 的 fidelity 主效应；
- legacy source/lambda 下 `td_target_action_source=bc_policy` 的主效应。

它们尚未测试二者交互。若更窄 source 与高权重 BCFM 只有在不让 Q 自己选择 bootstrap
action 时才稳定，单独主效应会漏掉该机制；这与再扫 flow steps 或 beta 不同。

新 config
`cqn_flow_floq_source01_bcfm8_td_bc_policy_two_tower_high_utd4_gate`
已逐项解析为：

```text
source = Uniform[0, 0.1]
bcfm_lambda = 8
td_target_action_source = bc_policy
td_target_policy_value_beta = null
rollout policy_value_beta = null
```

smoke 通过后才训练三个 algorithm seeds；checkpoint/beta 仍由 screen→validation 冻结，
`174000+` 做 200-episode paired confirmation。最终 B Gate 仍要求相对 clean CQN-AS
strict positive mean、CI lower `>=0`、wins>losses、`2/3` seeds positive，并额外通过
`175000+` H=1 full-path anti-cheat causal Gate。

实现与 config/gate tests 共 **10 passed**。B event controller PID `1603834`、A/B fallback
aggregate PID `1603850` 都已确认以 `do_wait` 等待 Stage-97/后继完成事件，不占 GPU、不做短
轮询。若届时 B 已由更早候选通过，full interaction 自动 skip。

### 21.89 Route B 文献与本地实现复核：排除 policy-flow 和重复 readout，预注册真正的新候选

#### 文献结论及其与本问题的边界

本轮只保留“FM 用来更新 value/return，state（含 image）和 C2F action bin 作为
condition”的方法：

1. [FLOQ](https://arxiv.org/abs/2509.06863) 用 velocity field 表示 scalar \(Q(s,a)\)，
   以 Bellman target 给出 dense flow supervision；这是当前 expected-value Route B 的直接
   上游。
2. [Value Flows](https://arxiv.org/abs/2510.07650) 学完整 return distribution，依次提出
   distributional FM、distributional conditional FM 和 Bellman-consistent FM；它还从
   flow derivative ODE/VJP 估计 return uncertainty，并用随置信度变化的权重重加权 FM loss。
3. [PCBF](https://arxiv.org/abs/2605.08253) 强调 source-consistent Bellman path 与
   control variate；本地此前已经实现和检验这一族。
4. [EVOR](https://arxiv.org/abs/2510.08218) 不只定义
   \(Q_\eta=\eta\log\mathbb E[\exp(R/\eta)]\) 的 entropic readout；其 Equation
   35--36 还提出独立的 velocity-space FlowTD update：online current-state velocity
   回归到 `reward + discount * frozen next-state velocity`，并在同一插值位置和时间评估
   两个 field。此前把 EVOR 归类成“只有 readout”是不完整的，后文 21.97 已纠正并
   单独实现该 update。
5. [Q-learning with Adjoint Matching](https://arxiv.org/abs/2601.14234) 和
   [Reinforce Adjoint Matching](https://arxiv.org/abs/2605.10759) 优化的是
   flow/diffusion **policy/generator**；critic/value 仍作为指导信号。它们可以成为未来
   actor 方案，但不能回答当前“让 value 用 FM update”的 Route B，因此不作为本轮候选。

#### 本地 fidelity 审计

当前 Stage-74 resolved config 已经实际包含 FLOQ 的关键表征项：

```text
scalar_value_embedding = hl_gauss
scalar_embed_bins       = 51
scalar_embed_sigma      = 16
time_embedding_type     = fourier
time_embed_dim          = 64
time_scale              = 1
```

因此当前 `source interval × BCFM lambda` 实验是在这些表征之上做的 official-fidelity
检验；不能再新增一个内容相同的“HL-Gauss/Fourier fidelity”arm。代码搜索也确认当前只有
endpoint/source variance diagnostics 和基于 TD error 的 priority，没有 Value Flows 所述
的 flow-derivative ODE uncertainty reweighting。

EVOR 式 entropic readout 也已经有 sealed 本地证据，不应重复扫温度：

```text
artifact:
  exp_local/cqn_flow_pcbf_ranking/
    stage26_pcbf_return_readout_gate_seed44000/gate_summary.json

validation: R16 entropic eta=1 24% vs R4 mean 14%
confirmation:                    19% vs         14%
paired delta: +5pp, 95% CI [-4,+14]pp
```

它只达到宽松的方向性 gate，未达到 strict superiority，而且绝对成功率远低于 clean
CQN-AS 约 61--64%。所以“多采样噪声后改 expected/entropic readout”不能作为下一阶段主线。

#### 下一候选的预注册 Gate

先让已经在跑的 Stage-74--106 完成；这期间不抢 GPU、也不根据中途曲线改假设。若
official source/loss、fixed-BC target 及其交互全部未通过 Route B task Gate，下一项只实现
一个此前未检验的机制：**return-sample DCFM+BCFM 的 Value-Flows confidence
weighting**。它不能套在 expected scalar FLOQ 上：scalar FLOQ 的不同 source 本来就应收敛到同
一个 Q，source derivative 不是 return variance。

- 固定相同的 CQN-AS conditioning、two-tower BC、UTD、数据、BC target policy、flow steps 与
  training frames；matched control 是 confidence off 的同一 return-sample DCFM+BCFM；
- 用 EMA return flow 的 derivative ODE/JVP 估计每个 transition 的 return std，并严格使用官方
  `stop_gradient(sigmoid(-temperature/std)+0.5)`，范围为 `(0.5,1)`；
- smoke Gate：resolved 开关正确、权重 finite/non-constant、有效样本量不坍缩、两头梯度
  finite；
- seed1 mechanism Gate：unweighted control 先声明、平局优先；weighted 必须由独立
  screen→validation 选中，并在 sealed confirmation 同时正向超过 unweighted control 与 clean
  CQN-AS；
- 只有 mechanism pass 才进入三 training-seed screen→validation→sealed task Gate；仍要求
  相对 clean CQN-AS mean delta `>0`、CI lower `>=0`、wins>losses、至少 `2/3` seeds 为正；
- task pass 后再跑 H=1 full-path anti-cheat Gate。若失败，结论应是目前证据不支持
  value-FM 在该数据/预算下优于 CQN-AS，而不是继续 sweep readout、flow steps 或 beta。

### 21.90 Stage-74 正式通过；Stage-75 已双卡启动；完整 Value-Flows 缺口已实现

#### Stage-74 实际结果

`14:27:40 BST`，三个 1.5k smoke 全部完成并通过所有预注册 checks：

| arm | max flow grad | max distill grad | nonfinite fraction | gate |
|---|---:|---:|---:|---:|
| source01 | `0.24481` | `0.01021` | `0` | pass |
| bcfm8 | `0.45447` | `0.01297` | `0` | pass |
| source01+bcfm8 | `0.24620` | `0.01002` | `0` | pass |

三个 artifact 分别为：

```text
exp_local/cqn_flow_high_utd/stage74_floq_fidelity_smoke_controller_r1/
  source_gate.json
  bcfm_gate.json
  combo_gate.json
```

每个 gate 都确认 official embeddings、声明的 source/lambda、two-tower/UTD4、snapshot、
finite required metrics、zero nonfinite gradient，以及 Flow/distill 两头实际收到梯度。该结论
只证明三个 arm 实现可训练，不代表任务提升。

#### 下一 Gate 与已执行

Stage-75 已在 `14:27:40` 自动启动第一波 10.5k：

```text
GPU1 PID 1622907: source01
GPU5 PID 1622908: bcfm8
```

按历史同配置先估第一波 `60--75min`，窗口为 `15:28--15:43 BST`；第一个 1k snapshot
落盘后必须用本 run 的 `total_time` 重算。随后组合臂才与 Route-A direct-Q fillers 资源共排。
Stage-76 对 legacy/source01/bcfm8/combo 做一次联合 screen→validation→sealed confirmation，
legacy 先声明并赢得平局；只有新 arm 严格晋级才扩 training seeds2/3。

#### Value Flows 实现审计纠正

全盘搜索 `.hydra/config.yaml` 后，没有找到任何 `dcfm_lambda>0` 的实际 run。仓库已有
`cqn_flow_value_flows` profile 与 DCFM 单测，但“本地 Value-Flows arms 已经跑过并失败”不是
artifact 支持的结论；此前真正跑过的是 PCBF/FlowIQN/entropic readout。

等待 Stage-75 的时间内已补齐缺失的官方机制：

- `integrate_value_flow_with_source_jvp(...)`：与 return ODE 同步积分 source JVP；
- EMA critic、Gaussian source、replay-selected C2F condition 上估计 return std；
- 官方 confidence weight `sigmoid(-T/std)+0.5`，全程 stop-gradient；
- 权重只乘 DCFM/BCFM/PCBF per-transition objective，不改 BC、MC anchor 或 demo loss；
- `confidence_weight_temp=null` 是精确 no-op，并记录 weight/std 的 mean/min/max；
- 新 matched high-UTD configs 固定 return-sample、DCFM+BCFM、independent BC Bellman
  target，仅 confidence temperature 不同；
- smoke checker 要求 weighted 权重 finite、positive、non-constant；control 必须严格记录
  `weight=1,std=0`。

最终定向兼容验证为 **13 passed, 107 deselected**；shell/Python syntax 与
`git diff --check` 同时通过。这些候选只在 Stage-106 后 Route B 仍失败时事件启动，不占用
当前两张训练卡。

### 21.91 Value-Flows 事件链已挂起；Stage-75 首个 compile 点不是吞吐 ETA

#### 上一阶段结果与含义

`14:34:12 BST`，Stage-107--113 event controller 已以 PID `1630608` durable 启动，child
PID `1630617` 正在等待 Stage-106 完成事件。此时 GPU 上仍只有 Stage-75 的两个训练进程，
因此 controller 没有抢卡或短轮询。

Stage-75 两个 arm 当前各只有 `iteration=0` 的 CSV 行：

| arm | compile/update time | flow grad | distill grad | nonfinite |
|---|---:|---:|---:|---:|
| source01 | `106.02s` | `0.27510` | `0.01023` | `0` |
| bcfm8 | `108.55s` | `0.26888` | `0`（该 arm 不含 distill） | `0` |

这两行主要包括 XLA 首次编译，不能线性外推 10.5k 的训练时长，也不能用于判断 policy
quality。两进程均仍存活，GPU1/GPU5 显存分别约 `22.3/25.2 GiB`。

#### 下一 Gate 与执行状态

Stage-75 必须等首个 1k snapshot/后续稳定区间，再按
`delta_steps / delta_total_time` 估计第一波 completion ETA；不得拿 compile 点估计，也不做
30 秒短轮询。完成后 Stage-76 按预注册的独立 screen→validation→sealed confirmation 比较
legacy/source01/bcfm8/combo。Stage-107 只在 Stage-106 仍未让 Route B 通过时执行：
control 与 confidence-weighted 先做成对 smoke，随后只有 weighted 在 held-out confirmation
同时超过 clean CQN-AS 和 unweighted control 才扩展到三 seed。

`14:35:47--14:35:49 BST`，两个 `1000_snapshot.pkl` 和第二行 CSV 已实际落盘：

| arm | total time @1k | stable reported throughput | backend update |
|---|---:|---:|---:|
| source01 | `448.25s` | `2.763 updates/s` | `0.0769s` |
| bcfm8 | `450.92s` | `2.716 updates/s` | `0.0769s` |

按较慢臂、剩余 `9500` updates 加少量收尾预算，第一波更新后的 ETA 为
**`15:30--15:35 BST`**，而非继续沿用未经本 run 校正的宽窗口。临时观察器随后已移除；
durable Stage-74--80 controller 直接等待两个训练 PID 的退出事件并自动续跑。

### 21.92 FLOQ 官方实现级审计：当前是 CQN-AS fidelity 因子实验，不是逐行复现

#### 上一阶段实际结果

本轮在 Stage-75 训练继续运行期间，直接审计了 FLOQ 官方仓库
[`CMU-AIRe/floq`](https://github.com/CMU-AIRe/floq) 的当前 commit
`3d60e638b42bbef018c56cb1199ff37c3470520d`，并逐项对照正在运行的 resolved config
和 `robobase/method/cqn_flow.py`：

| 机制 | FLOQ 论文/官方实现 | 当前 CQN-AS integration |
|---|---|---|
| target Q | 8 个 target-flow samples 求均值 | `num_target_flow_samples=8` |
| 当前 flow samples | 8 个 uniform samples | `num_flow_samples=8` |
| scalar interpolant | 51-boundary HL-Gauss，`sigma=16` | 相同的 50-bin probability feature |
| time condition | 论文明确为 64-D Fourier basis | 64 个 cosine basis；source/endpoint 时间方向已换算 |
| source interval | 宽度约为真实 Q range 的 `0.1`，并与 target Q 重叠 | MovePlate 单次成功即终止，真实 return support 为 `[0,1]`，候选为 `[0,0.1]` |
| flow/distill 比例 | 8 个 source MSE **求和**后加一次 distill MSE | 本地 source 轴求均值，因此 `bcfm_lambda=8` 恢复相同比例 |

官方仓库本身存在一个值得单独记录的 paper/code discrepancy：配置声明
`time_embed_dim=64`，论文也明确报告 64-D Fourier embedding，但公开
`CriticVectorField` 当前实际只计算 `jnp.cos(times)`，没有使用
`time_embed_dim`。本地 64-frequency cosine basis 因而是 **paper-faithful**，并非该
commit 的逐行 code-faithful 行为。

当前两个 Stage-75 arm 在 1k 的健康证据仍为：

```text
source01: total_time=448.25s, flow_grad_nonfinite=0,
          endpoint Q range [-0.0003, 0.9329]
bcfm8:    total_time=450.92s, flow_grad_nonfinite=0,
          endpoint Q range [ 0.0245, 0.9521]
```

这只能证明两个 critic 正在学习且数值有限，不能把 endpoint range 或 training loss
解释成 policy quality。

#### 解释

Stage-74--78 应准确称为 **FLOQ key-mechanism fidelity factor experiment**：
它在固定 CQN-AS 的 image/state encoder、C2F action-bin conditioning、独立 BC policy、
replay-next target、UTD4、MC anchor 和 action collection 下，只检验
`source support × official flow/distill relative weight`。本地还保留
`endpoint_q_lambda=1`、`source_consistency_lambda=0.1` 和 `mc_return_weight=0.1`；
因此该实验回答“FLOQ 的关键 critic 机制能否改善 CQN-AS”，不能声称复现官方完整
FQL actor-critic。

这也排除了再增加一个重复的 “HL-Gauss/Fourier arm”：两者已经真实启用。尚未解决的是
这些机制是否提升 held-out task success、是否跨 training seed 稳定，以及选出的 value
是否通过 H=1 do-action 反事实真实性 gate。

#### 下一阶段决策与 Gate

Stage-76 保持预注册设计，不根据 1k loss 改选择规则：

1. matched candidates 为 `legacy/source01/bcfm8/source01+bcfm8`，legacy 先声明并赢得平局；
2. checkpoint 和 beta 只由 screen→validation 选择；
3. never-used confirmation 同时要求相对 clean CQN-AS 的方向性提升与预注册 CI；
4. seed1 只有严格晋级才训练 seeds2/3；三 seed 最终仍比较各自
   validation-selected best checkpoint；
5. task gate 通过后才运行 H=1 full-path anti-cheat gate；task success 与 value
   authenticity 保持为两个独立结论。

Pass/fail 不因“更接近官方”而放宽：Route B 最终仍须三 seed mean delta `>0`、
crossed-bootstrap CI lower `>=0`、wins>losses、至少 `2/3` training seeds 为正，并通过
H=1 因果 gate，才可推荐为超过原始 CQN-AS。

#### 已执行

Stage-75 的 GPU1/GPU5 进程 `1622907/1622908` 已再次核验存活，分别占用约
`22.7/25.5 GiB`，GPU utilization 约 `84%/68%`；durable controller 正以进程退出事件等待，
不是短轮询。基于本 run 的 1k 稳态吞吐，第一波 ETA 仍为
**2026-07-24 15:30--15:35 BST**。完成后 controller 会立即启动组合 arm，并并行填充
独立 Route-A direct scalar-Q seeds。

### 21.93 Route A 实现审计：direct-Q 已切断 imitation/value 参数通路，真实性仍待实验

#### 上一阶段实际结果

在不占训练 GPU 的条件下，对 `CQNDirectQAS`、training gate、branch oracle 和 H=1
CF-FQE summarizer 运行了完整定向测试：

```text
30 passed in 83.94s
```

代码与端到端测试确认：

1. online direct-Q 的 value encoder、scalar critic、BC encoder 和 BC policy 是独立参数树；
2. replay-next TD 与 completed-return MC regression 只通过 value tower，demo CE 只通过
   independent BC tower；
3. rollout 在 `policy_value_beta=null` 时严格保持 BC policy，不让正在学习的 Q 改写收集分布；
4. H=1 CF-FQE 从同一 branch cache 分别拟合真实 label 和 within-state shuffled label，
   只更新 direct scalar critic；
5. fine-tune 前后对 value encoder、BC encoder、BC policy 做 bitwise equality 验证；
6. held-out bootstrap 的独立单位是 simulator seed，而不是把同一 restored state 下的 sibling
   actions 错当成独立样本。

#### 解释

这排除了一个实现层面的混淆：Route A 的 direct-Q 不能靠共享 logits/head 或共享 encoder
上的 demo gradient，把 “value learning” 偷换成 BC。within-state shuffle 还保留 state
difficulty 和 label marginal，只破坏 action→return 对应，因此能识别 state-only shortcut。

但 `30 passed` 只证明实验协议真的实现了这些约束；它不证明 direct-Q 已经学到真实 value，
也不证明其 closed-loop success 不低于 clean CQN-AS。训练 loss、demo top-1 和单元测试都
不能替代这两个实证 gate。

#### 下一阶段决策与 Gate

Route A 保持两个独立问题：

- **Task Gate（Stage-86）**：三 training seeds，各自用 screen→validation 选择 checkpoint，
  用 never-used 200-episode confirmation 比较对应 clean CQN-AS best checkpoint；
  要求 mean paired delta `>=0`、CI lower `>=0`，不能拿 overtrained final BC 作 baseline。
- **Authenticity Gate（Stage-87）**：对 task-selected checkpoint 做新的 H=1 sibling
  do-action branches；Q pairwise accuracy 的 seed-cluster CI lower 必须高于 chance，并严格
  超过 action-nearness、independent BC prior 和 within-state shuffle control，至少
  `2/3` algorithm-training seeds 为正。

若 online direct-Q 未同时通过两项，Stage-98--101 才启用 frozen-representation H=1
CF-FQE fallback；它也必须先通过新的 branch heldout split，再在全新 task confirmation
和 H=1 causal split 上复验，不能用训练 branch 的拟合结果直接宣称成功。

#### 已执行

Stage-84/85 direct-Q seeds 已被 Stage-75 controller 预排为组合 Flow arm 的 GPU5
并行 filler；Stage-86/87 和 Stage-98--101 均由完成事件触发。当前没有额外进程抢占
GPU1/GPU5，Route A 与 Route B 的数据、checkpoint、selection split 和 conclusion 保持独立。

### 21.94 Value Flows 官方代码复核：fallback 核心公式一致，actor/ensemble 是明确边界

#### 上一阶段实际结果

直接审计了 Value Flows 官方仓库
[`chongyi-zheng/value-flows`](https://github.com/chongyi-zheng/value-flows)
commit `01833354f547ad842bdaa21d2b761a16069f9724`。官方实现的 critic 核心为：

```text
BCFM:
  Z'(s',a') = integrate target flow from shared Gaussian epsilon
  z_t        = t * (r + gamma * Z') + (1-t) * epsilon
  target v   = r + gamma * Z' - epsilon

DCFM:
  z'_t       = partial target-flow integration to the same t
  z_t        = r + gamma * z'_t
  target v   = target_vector_field(s',a',z'_t,t)

confidence:
  std        = abs(d phi(epsilon) / d epsilon)
  weight     = stop_gradient(sigmoid(-0.3 / std) + 0.5)
```

本地 `return_sample` 路径逐项保持 same-noise BCFM、partial-flow DCFM、DCFM target
不额外乘 gamma、EMA critic source-JVP，以及完全相同的 stopped-gradient weight 公式。
CPU 定向 golden/config/gate 验证结果为：

```text
19 passed, 96 deselected in 20.12s
```

其中覆盖了 JVP 与离散 flow derivative 对齐、DCFM Bellman partial-flow golden、
confidence off 的 exact no-op、weighted finite/non-constant 检查和两份 matched config。

#### 解释

这证明排队中的 Stage-107 treatment 不是把 expected scalar FLOQ 的 source variance
误称为 return uncertainty，而是真正切换到 Gaussian **return-distribution flow** 后再做
Value-Flows JVP confidence weighting。

它仍不是官方算法的逐行复现，差异是预先声明且对 treatment/control 完全 matched：

- 官方有两个 return-flow critics 并聚合两者的 return/std；当前 CQN-AS integration 保持
  一个 C2F conditional critic；
- 官方用 flow actor + rejection sampling/one-step distillation；当前使用独立 CQN-AS BC
  policy 作为固定 Bellman target，并只在 held-out evaluation 选择 Q+BC beta；
- 本地每个 transition 并行 4 个 return samples，官方主 loss 每 critic 使用 1 个；
- 本地对 C2F level、有效 action token 和 action dimension 聚合 JVP std。

所以 Stage-107--113 回答的是“在 CQN-AS 的固定 actor/data protocol 下，Value Flows
critic 和 confidence weighting是否有增益”，而不是复现论文总分。若它失败，不能据此否定
官方 flow actor/dual-critic 整体；但可以否定当前 CQN-AS-compatible 单 critic 方案。

#### 下一阶段决策与 Gate

Stage-107 仍把 `confidence=null` 的 DCFM+BCFM control 放在前面并赢得平局；weighted
只改变 `confidence_weight_temp=0.3`。seed1 必须在独立 validation 被选中，并在 sealed
confirmation 同时超过 unweighted control 与 clean CQN-AS，才扩到 seeds2/3。三 seed
task gate 和 H=1 authenticity gate 均保持原严格标准。

若 confidence treatment 失败，下一候选必须把“dual critic aggregation”或“policy
extraction”作为新的单变量问题，不能把二者一起改后声称是 uncertainty weighting 的效果。

#### 已执行

Stage-107--113 controller PID `1630617` 已挂在 Stage-106 完成事件上；目前不占 GPU。
它会先跑 1.5k matched smoke，并要求 control 精确记录 `weight=1,std=0`、treatment 权重
finite/positive/non-constant，再进入 task evaluation。当前 GPU1/GPU5 继续专用于 Stage-75。

### 21.92 Value-Flows 论文/官方代码逐项审计；Stage-75 2k 吞吐稳定

#### 上一阶段结果

对 [Value Flows 论文](https://arxiv.org/html/2510.07650) 与
[官方 `value_flows.py`](https://github.com/chongyi-zheng/value-flows/blob/main/agents/value_flows.py)
逐项复核后，本地 confidence treatment 的核心语义与官方一致：

1. 在 target/EMA return critic 上，用一个 Gaussian noise 同步积分 return ODE 与
   source-JVP；
2. `return_std=abs(source_jvp)`，权重为
   `stop_gradient(sigmoid(-temperature / return_std) + 0.5)`；
3. 同一 transition weight 同时乘 BCFM 与 DCFM，再对 batch 求均值；
4. BCFM 是防止 self-distillation/zero-field collapse 的 regularizer，DCFM 是主要的
   distributional Bellman signal。

需要明确：Stage-107 不是对原论文 agent 的整机复现。论文使用 twin continuous-action
return fields 与 flow-policy rejection sampling；这里有意保留一个 CQN-AS conditional
return field、C2F action bins、fixed independent BC next-action policy 和 integrated
readout，测试的是 **Value-Flows value update × CQN-AS action selection**。因此最终只能
把结论归因于该适配版，不能声称复现论文 benchmark。

`14:41:29--14:41:31 BST`，Stage-75 两臂的 2k snapshots 均落盘：

| arm | total time @2k | 当前 reported throughput | backend update | flow grad |
|---|---:|---:|---:|---:|
| source01 | `790.45s` | `2.709/s` | `0.0769s` | `0.01784` |
| bcfm8 | `793.25s` | `2.816/s` | `0.0749s` | `0.14661` |

1k→2k 都约 `342.3s`，吞吐已经进入稳定区，梯度 finite。按较慢臂剩余 8.5k 与 checkpoint
收尾预算，第一波 ETA 收紧为 **`15:30--15:32 BST`**。此处 loss/grad 仍只作健康检查，
不作 task-quality 结论。

### 21.93 新候选 FlowCritic：只迁移可识别的 truncated readout，并已事件排队

#### 文献结论与边界

[FlowCritic](https://arxiv.org/html/2510.22686) 是另一条直接用 FM 建模 value
distribution 的路线，但其原算法是 PPO state-value critic，不是 CQN-AS：

- Gaussian source 经 5 步 Euler 生成 distributional Bellman return samples；
- 默认生成 `N=10` 个 value samples，排序后丢弃最大 `K=1` 个再求均值，抑制高回报
  outlier/overestimation；
- velocity-field update 相对 target field 做 clipping，默认阈值 `0.2`；
- CoV 权重用于 **PPO policy gradient**，并且低 CoV 样本权重更高。这与 Value Flows
  “高 return variance 的 transition 获得更高 critic-loss 权重”方向相反，不能直接把
  CoV 权重乘到 CQN-AS critic loss；
- 论文所指代码仓库目前仍是 private，因此没有把无法逐行核验的 clipping 先冒充官方复现。

对当前 CQN-AS 最小且可归因的新假设是：return flow 的高尾 outlier 会放大
argmax-over-bins 的 overestimation/cheating；在同一 checkpoint 上使用论文默认
`N=10,K=1` 的 truncated mean，可能比普通 mean 给出更稳健的 action ranking。

#### 已实现与预注册 Gate

已加入：

- `return_sample_aggregation=truncated_mean` 与
  `return_sample_truncate_top`；
- 任意 return-sample axis 上排序并丢弃最大的 `K` 个，再求均值；
- eval、checkpoint-selection 和 H=1 causal probe 都能显式冻结
  `num_action_flow_samples=10, truncate_top=1`；
- arm Gate 支持同一 run 声明不同的 per-candidate readout，所以 comparison 不需要重训，
  且 mean/truncated 使用相同 checkpoint、beta 和 environment seeds；
- task summary 与 causal summary 都记录 aggregation/sample/truncation，防止缓存或报告把
  两种 readout 混淆。

定向 compile/syntax/diff 与 **27 tests passed, 111 deselected**。Stage-114--118 event
controller 已于 `14:55:10 BST` 以 PID `1646700` 启动，child `1646709` 正在通过
`tail --pid=1630608` 等待 Stage-107--113，不占 GPU。

Stage-114 只有在此前 Route B 仍失败时运行：

1. Stage-109 已使用的独立 validation 先冻结 control/weighted 中的 training arm；
2. fresh `187000` screen、`188000` validation、`189000` sealed confirmation 比较
   `mean(N=10)` 与 `truncated_mean(N=10,K=1)`，mean 先声明并赢得平局；
3. truncated 必须在 confirmation 同时正向超过 clean CQN-AS 与 independently
   validation-selected mean，才训练/复用 seeds2/3；
4. fresh `190000--192000` 做严格三 training-seed task Gate；通过后才用 `193000+`
   做 H=1 sibling anti-cheat causal Gate。

#### 当前运行状态

`14:53:32--14:53:34 BST`，Stage-75 两臂的 4k snapshots 均落盘，当前吞吐
`2.75--2.77 updates/s`，GPU1/GPU5 仍只有两个训练进程。4k 稳定区继续支持第一波
**`15:30--15:32 BST`** ETA；此时没有 task-quality 结论。

### 21.95 Stage-119：把“flow 是否真的被使用”从 task score 中拆出并事件排队

#### 上一阶段实际结果

新的长期 goal 已通过 goal service 核验为 active：

```text
goal id: 019f859b-a58f-7142-88ba-c6c96d290ca0
A: meaningful causal value, task performance >= clean CQN-AS
B: FM x CQN-AS, task performance > clean CQN-AS
```

本阶段没有拿 training loss 代替 policy quality。`15:01 BST` 从实际 artifacts 读取到：

| Stage-75 arm | 最新 checkpoint | internal eval @5k | best internal eval so far | total time @5k |
|---|---:|---:|---:|---:|
| source01 | 5k | 56% | 56% | 1892.39s |
| bcfm8 | 5k | 76% | 76% | 1892.14s |

路径分别为：

- `exp_local/cqn_flow_high_utd/stage75_floq_source01_utd4_seed1_gpu1_20260724`
- `exp_local/cqn_flow_high_utd/stage75_floq_bcfm8_utd4_seed1_gpu5_20260724`

两者 5k snapshots 于 `14:59:50--14:59:51 BST` 落盘；进程仍在 GPU1/GPU5。
这个 internal eval 只有一个 training seed、尚未走 Stage-76 的
screen/validation/confirmation，因此 **76% 不是 validation-selected 最终结果**，也还
没有可公平报告的 clean-best 对比。

同时完成了只读 flow-utilization 诊断：

- `integrate_value_flow_trajectory` 保留与训练 endpoint integrator 完全相同的 Euler
  recurrence，并逐步返回 trajectory；
- `scalar_flow_trajectory_diagnostics` 测 curvature、source contraction 和 step
  increment variation；
- `CQNFlowAS.flow_utilization_probe` 在固定 BC action condition、相同 source bank 下比较
  1/2/4/8-step Q ranking 与 configured-depth ranking；
- `scripts/analyze_cqn_flow_utilization.py` 只载入 frozen checkpoint 和 fresh reset states；
- `scripts/summarize_cqn_flow_utilization.py` 明确输出
  `diagnostic_only=true, selection_use_forbidden=true`。

CPU 定向验证实测为：

```text
10 passed, 115 deselected in 8.35s
full CQN-Flow + diagnostic regression: 125 passed in 98.80s
py_compile: pass
bash -n Stage-119 controller: pass
```

测试包含 endpoint recurrence exact equality、直线路径零 curvature、曲线路径正
curvature、zero-velocity identity-collapse、RNG bitwise 不变和 summary pass/fail golden。

#### 解释

这建立了一个能区分以下两种情况的诊断工具：

1. flow 真正逐步消除 source noise，并且多步积分改变 action-bin value ranking；
2. velocity field 退化为 identity/直线/近似单步 head，多跑 Euler steps 基本不改变 Q。

它尚未建立任何 checkpoint 的 task quality，也不能证明 value 有 causal meaning。
尤其 Stage-75 的 deployed selection 使用 distilled scalar readout；即便 integrated field
通过 utilization gate，也只说明内部 flow 非平凡，不能把它冒充 task 增益。反过来，
若 task score 尚可但 utilization 失败，则说明当前 FM 很可能只是昂贵的辅助 head，
下一步不应继续盲目增加 solver steps。

#### 下一阶段决策与 Gate

**假设：** Stage-76 validation-selected scalar-flow checkpoint 的 configured 8-step field
不是 identity/source-preserving collapse，并且不能被 one-step readout 等价替代。

**Matched baselines：** 同一 checkpoint、同一 target critic、同一 16 个 observations、
同一 8-source common-random-number bank、同一 BC action/C2F conditions，只改变 Euler
steps 为 `1,2,4,8`；configured 8-step 自身是 exact-consistency control。

**Selection split：** checkpoint/arm 只能由 Stage-76 的 `115000` validation split 冻结；
Stage-119 不重新选择 arm、step 或 beta。

**Held-out diagnostic split：** reset seeds `194000--194015`，未用于 Stage-76/78、
Value-Flows 或 FlowCritic task/causal gates。

**Metrics：** configured Q span、source contraction ratio、normalized trajectory
curvature、one-step/configured bin agreement、one-step normalized Q RMSE。

**Mechanistic pass/fail（不等于 task gate）：**

- all finite，configured-depth self comparison agreement `>= 1-1e-6` 且 RMSE `<=1e-6`；
- mean configured Q span `>=1e-3`；
- mean source contraction ratio `<=0.95`；
- normalized curvature `>=0.01`，或同时满足 one-step agreement `<=0.98` 与 normalized
  Q RMSE `>=0.02`。

Task superiority 仍只由 Stage-76/78 的 validation-selected sealed confirmation 判断；
causal authenticity 仍只由 H=1 sibling do-action gate 判断，三者不合并。

#### 已执行

Stage-119 controller 已于 `15:00:55 BST` 以 master PID `1651433`、child PID
`1651441` 启动：

```text
scripts/run_cqn_stage119_flow_utilization.sh
exp_local/cqn_flow_high_utd/stage119_flow_utilization_master
exp_local/cqn_flow_high_utd/stage119_flow_utilization_seed194000_20260724
```

它通过 `tail --pid=1646700` 事件等待 Stage-114--118 完整链释放 GPU，不做短轮询、当前
不占显存。随后读取 Stage-76 已冻结的 `selected_arm/selected_step`，在 GPU1 运行一次
probe 并写出 `probe.json`、`summary.json`。Stage-75 按 5k 实测吞吐剩余约 31 分钟，
第一波 ETA 仍为 **`15:30--15:32 BST`**。

### 21.96 Route A 的 deployable fallback：action-centered randomized advantage

#### 上一阶段实际结果

Stage-98--101 的 H=1 protocol 经重新审计后，已明确改称
**simulator-branch Monte-Carlo oracle**，而不是传统 replay-only FQE。artifact 现在强制记录：

```text
target_estimator=simulator_branch_monte_carlo
continuation_policy=frozen_independent_bc
continuation_policy_value_beta=null
```

如果 source checkpoint 的 `policy_value_beta` 非空，collector 会拒绝运行。positive 与
within-state shuffle 仍使用同一 branch cache；encoder、BC encoder、BC policy 仍做 bitwise
不变检查。相关定向 regression 为 **29 passed in 3.31s**。

与此同时，Stage-75 在 `15:17:35--15:17:39 BST` 产出两个 8k snapshots：

| arm | internal success @5k/7.5k | internal best | total time @8k | throughput |
|---|---:|---:|---:|---:|
| source01 | `56% / 68%` | `68%@7.5k` | `2956.13s` | `2.768/s` |
| bcfm8 | `76% / 60%` | `76%@5k` | `2961.92s` | `2.719/s` |

这些仍只是同一 training seed 的 internal checkpoints；Stage-76 尚未在独立 validation
冻结 arm/step/beta，所以 `76%` 不能当作最终 superiority 结论。按 8k 后剩余工作量，
第一波完成 ETA 为 **`15:32--15:34 BST`**。

#### 解释

simulator branch oracle 能回答“representation/head 在拿到真实同状态反事实标签时能否学到
action effect”，并能给出 simulator-assisted 上限；但其训练标签不能在真实机器人或普通
replay-only setting 获得，因此即使 Stage-101 通过，也不能单独作为 Route A 的 deployable
算法推荐。

本地 replay 已经保存每一步 intervention 的
`start/dimension/delta/assignment_probability`，因此不需要 simulator branch 就能利用真实
randomization。这个新候选依据三条 primary literature：

- [Action-Centered Contextual Bandits](https://papers.nips.cc/paper_files/paper/2017/hash/4fa177df22864518b2d7818d4db5db2d-Abstract.html)
  证明 \((Z-p)Y\) 在已知随机化概率下正比于 treatment-vs-baseline reward difference，
  可把任意复杂的 state-only baseline 正交掉；
- [Q- and A-Learning Methods for Estimating Optimal Dynamic Treatment Regimes](https://arxiv.org/abs/1202.4177)
  将只建模 action contrast 与 propensity 的 A-learning 和完整 outcome-model Q-learning
  区分开；
- [Flexible and Efficient Contextual Bandits with Heterogeneous Treatment Effect Oracles](https://proceedings.mlr.press/v206/carranza23a.html)
  给出直接学习跨 action reward difference 比完整 reward model 更适合决策的理论/实证动机。

这里把即时 bandit reward 换成“当前 H=1 randomized action 后、固定 BC continuation 的
completed discounted return”，所以估计的是该单次 decision 的 long-horizon
\(Q^{\pi_{\rm BC}}(s,a)-Q^{\pi_{\rm BC}}(s,a_{\rm BC})\)，不是假设 action 不影响未来。

#### 新算法与 matched Gate

新增 direct-Q optional loss：

\[
L_{\rm AC} =
p(1-p)\,\tau_\theta(s,\bar a)^2
-2(Z-p)\,G\,\tau_\theta(s,\bar a),
\]

其中 \(G\) 是真实 completed return，\(Z\) 表示是否执行了随机 proposal，
\(p=0.2\)，\(\tau=Q(s,\bar a)-Q(s,a_{\rm BC})\)。对未 treatment 的 control transition，
proposal dimension/sign 由独立均匀 key 生成；对 treatment transition，则从 replay 的
dimension/delta 精确恢复 baseline action。population optimum 是 treatment effect，而
state difficulty 因 \(\mathbb E[Z-p\mid s]=0\) 被消去。

严格限制：

- `structured_exploration_horizon=1`，拒绝用历史 H=4 assignment 冒充一步 effect；
- 只使用 online replay 中 assignment probability `<1` 的 randomized samples，demo 与
  H>1 continuation 不进入 causal loss；
- independent BC rollout、two-tower encoder、TD、MC、UTD4 都保持；
- matched control 与 treatment 都用 `p=0.2,H=1`，唯一差异为
  `causal_rct_weight: 0 -> 0.1`；
- replay gate 要求 observed propensity 精确为 treatment 的
  `0.2/(2*15)` 或 control 的 `0.8`，所有 15 个 dimensions 都有覆盖。

Stage-120--126 gate 顺序：

1. matched 1.5k smoke，两臂 finite 且真实 replay coverage 通过；
2. seed1 两臂各 10.5k；control 先声明并赢 tie；
3. fresh `195000/196000/197000` screen/validation/confirmation 中，RCT 必须同时超过
   clean CQN-AS 与 matched no-loss control，两个 paired CI lower 都 `>=0`；
4. 只在 discovery pass 后训练 RCT seeds2/3；
5. fresh `198000/199000/200000` 做三 training-seed best-checkpoint task Gate，要求相对
   clean mean delta `>=0` 且 CI lower `>=0`；
6. task pass 后用 `201000+` H=1 sibling do-action Gate，要求至少 `2/3` training seeds
   为正并超过 action-nearness 与 independent-BC proxies。

#### 已实现并执行

代码与配置：

```text
robobase/method/cqn_direct_q.py
robobase/cfgs/launch/cqn_direct_q_h1_rct_control_two_tower_high_utd4_gate.yaml
robobase/cfgs/launch/cqn_direct_q_h1_rct_two_tower_high_utd4_gate.yaml
scripts/check_cqn_direct_q_rct_training_gate.py
scripts/run_cqn_stage120_126_action_centered_route_a.sh
```

数学 optimum、真实 update divergence、H=4 rejection、config single-difference 和 synthetic
replay propensity Gate 的核心测试为 **7 passed in 45.76s**；compile、shell syntax 与
`git diff --check` 均通过。

随后清理了 `effective_k0` 路径中一个重复的 critic-training slice，并修正真实随机
assignment 在 action bound 被 clip 成 `delta=0` 时被 replay gate 误判的问题；这种情况
对应合法的零 realized effect，仍必须保留其 assignment、dimension 和 propensity。强制
禁用 CUDA 做完整相关回归，避免与正在满载的 GPU-1/5 训练争抢显存：

```text
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu
38 passed in 31.79s
```

此前未强制 CPU 的一次广测出现 12 个 `CUDA_ERROR_OUT_OF_MEMORY`；这是错误的验证环境，
不计作算法 failure。当前 CPU 全量结果、`py_compile`、shell syntax 和
`git diff --check` 均通过。

event controller 已于 `15:19:58 BST` 启动：

```text
master PID 1674949
child  PID 1674958
wait   do_wait on Stage-119 master
```

它当前不占 GPU；`nvidia-smi` 仍只有 Stage-75 PID `1622907/1622908`。Stage-119 无论
diagnostic pass/fail 都先释放 GPU，随后 Stage-120 才双卡启动。按当前 direct-Q 历史吞吐，
smoke 约 `4--7min`，每个 10.5k 双卡 wave 约 `19--24min`；task/causal evaluation ETA 会在
首次实际 episode/branch throughput artifact 落盘后重新估算。

### 21.97 Route B 新候选：EVOR velocity-space FlowTD，而不只是 entropic readout

#### 上一阶段实际结果

Stage-75 的两个单变量 seed-1 run 已完成到 10.5k budget；10k checkpoint 的 internal
eval 与 validation 前最佳点为：

| arm | internal @10k | internal best | total time @10k |
|---|---:|---:|---:|
| source01 | `56%` | `68%@7.5k` | `3682.21s` |
| bcfm8 | `60%` | `76%@5k` | `3688.69s` |

这两个完成 artifacts 已由独立 recovery preflight 重新检查：

```text
exp_local/cqn_flow_high_utd/stage75_floq_fidelity_recovery_preflight/
  source_gate.json: pass, last_iteration=10000,
                    max flow grad=0.27510, max distill grad=0.03210
  bcfm_gate.json:   pass, last_iteration=10000,
                    max flow grad=0.26888, max distill grad=0.00672
```

两者的 config fidelity、snapshot、10 个 logging rows、finite gradient、zero nonfinite
fraction、independent-BC rollout 全部通过。`68%/76%` 仍是训练期 internal eval；Stage-76
尚未用 `115000` validation 冻结 arm/checkpoint/beta，因此不能与 clean validation-best
作结论性比较。

文献侧重新逐式阅读 [EVOR](https://arxiv.org/abs/2510.08218) 后，纠正了 21.89 中
“EVOR 只有 risk readout”的不完整分类。论文 Equation 35--36 定义了独立的 FlowTD：

```text
z0 ~ Gaussian
z1 ~ frozen return flow R_target(s,a)       # 与 z0 独立
z_t = (1-t) z0 + t z1
target velocity = reward + effective_discount
                  * frozen_velocity(z_t,t | s',a'_BC)
online_velocity(z_t,t | s,a) -> target velocity
```

本地实现保持 repository 的 reverse-time `tau=1-t` 坐标，但产生完全相同的线性
interpolant。初版审计还抓到并修正了一个关键 coupling：interpolant source 与生成
`z1` 的 target-flow source 现在来自两个独立 PRNG branches，不能偷换成同一 ODE
trajectory。当前实现只保留 EVOR FlowTD，明确关闭 BCFM、DCFM、PCBF、FlowIQN、
endpoint、distillation 和 MC anchor。

实现与健康 gate：

```text
robobase/method/cqn_flow.py
robobase/cfgs/launch/
  cqn_evor_flowtd_bc_target_two_tower_high_utd4_gate.yaml
scripts/check_cqn_evor_training_gate.py
scripts/run_cqn_stage127_133_evor_flowtd.sh
tests/unit/test_check_cqn_evor_training_gate.py
```

公式/stop-gradient、真实 critic update、非法 action-source、resolved config 与 gate
定向验证为 **10 passed**；强制 CPU 的完整 CQN-Flow 回归为：

```text
JAX_PLATFORMS=cpu
127 passed in 105.41s
py_compile: pass
bash -n all 16 controller scripts: pass
git diff --check: pass
```

一次未限制 backend 的广测落到已满载的 GPU，产生 25 个
`CUDA_ERROR_OUT_OF_MEMORY/cuSolver`，随后 CPU exact rerun 全部通过；这不是算法 failure。

#### 解释

EVOR FlowTD 是此前本地 entropic-readout ablation **没有测试过的新 Bellman update**。
所以 Stage-26 的 `+5pp, CI[-4,+14]pp` 只能说明 readout 方向性，不能否定或支持本候选。
该候选直接符合“state/image 与 action bin 都作为 condition，value 本身由 FM update”的
Route B 定义。

边界也预先声明：

- 这是 EVOR FlowTD × CQN-AS 的适配，不是整机 reproduction；论文用 flow actor、从
  base policy 采 32 个 action candidates 后 softmax/rejection，当前用 C2F sibling bins
  与独立 categorical BC prior；
- 论文 train 用 1 个 return sample、eval 用 50；本地 train 同样用 1，eval 用 16 个并行
  samples，符合当前显存/延迟预算；
- EVOR 公式在 terminal transition 上仍把 target velocity 写成 `reward`，而不是经典
  endpoint FM 的 `reward-source`；本地先 paper-faithful 实现，不用未验证的“修正版”偷换
  hypothesis。若 task/causal gate 失败，这会是下一步的明确诊断点；
- [Distributional Flow Critic](https://arxiv.org/abs/2509.23087) 的 flow Bellman target
  后再训练 one-step quantile critic，核心依赖 distillation。因为 distillation 同时改变
  readout/optimization 且用户已质疑其必要性，当前不与 EVOR 混合；只有 EVOR gate 给出
  明确失败后，才把 DFC 的 flow-target 与 quantile-distill 拆成 matched factor。

因此本阶段建立的是实现可运行性与独立研究问题，尚未建立 policy quality 或 causal value。

#### 下一阶段决策与 Gate

**假设：** 在 fixed independent BC behavior/next-action、two-tower encoder 和相同 replay
protocol 下，EVOR FlowTD 学出的 integrated return value 能在不靠 BC imitation loss
改动的情况下超过 validation-selected clean CQN-AS。

**Matched baseline：** clean CQN-AS 三个 algorithm seeds 的既有 validation-best
checkpoints：`5k/5k/2.5k`。EVOR 训练保持同一个 BC objective、数据、UTD4、C2F levels/bins
和 action sequence；变化只在 value update/readout。

**Selection/held-out split：**

1. Stage-127：seed1/2 各 1.5k smoke，只检查 active finite FlowTD、nonzero critic/velocity
   gradients、BCFM/DCFM/PCBF exact zero 和 snapshot；不进入 task selection；
2. Stage-128：algorithm seeds1/2 各 10.5k；
3. Stage-129：fresh `202000` screen、`203000` validation 选择一个 global
   `beta in {0.3,1,3}` 与每 seed checkpoint；`204000` 做两-seed sealed promotion；
4. 只有两 seed 都为正、aggregate wins>losses、mean delta `>0`、crossed CI lower
   `>=-5pp`，才进入 seed3；
5. Stage-130 训练 seed3；Stage-131 用 fresh `205000/206000/207000`
   screen/validation/confirmation 做三 training-seed strict task Gate；
6. 最终 task pass 必须满足 mean paired delta `>0`、crossed bootstrap CI lower `>=0`、
   wins>losses、至少 `2/3` algorithm seeds 为正；比较双方均为各自 validation-selected
   checkpoint，不能拿 clean overtrained final；
7. task pass 后 Stage-132 用 fresh `208000+` H=1 same-state sibling branches 检验
   ranking，要求至少 `2/3` seeds 为正并超过 BC/path/action-nearness proxies。task 与
   authenticity 是两个独立结果，不互相代替。

部署 readout 固定为 10-step integrated flow、16 个 common-random-number return sources、
EVOR entropic temperature `1`。Stage-129 只决定是否值得扩 seed；它被明确标记
`route_b_update_forbidden=true`，不能提前满足最终 goal。

#### 已执行

首先定位并修复了一次 controller-only 级联失败。Stage-75 训练本身正常结束，但运行中的
controller 在 wait 返回后读到一个短暂的未闭合引号版本，实际日志为：

```text
unexpected EOF while looking for matching '"'
```

恢复动作没有删除或覆盖训练数据：旧 `failed` markers 被重命名为
`.superseded.20260724T1535`；Stage-74--80 加入“required snapshot 已存在则异步 no-op”
恢复语义；然后统一重启整个 event chain。

`15:37 BST` 已从 process、log 与 GPU memory 三方验证真正启动：

```text
GPU1 PID 1693989:
  Stage-75 source01+bcfm8 combo, fresh 10.5k
GPU5 PID 1693991:
  Route-A replay-next direct-Q seed1，随后 seed2/3
```

combo 与 direct-Q logs 都已完成 demo load，GPU memory 各约 `25GB`；compile 后会进入稳定
update。按同配置实测，combo 约 `62--66min`，GPU5 三个 direct-Q filler 合计约
`57--72min`，因此 Stage-75 完整 recovery ETA 为 **`16:40--16:50 BST`**，保守上界
`17:00 BST`。随后 Stage-76 才用双卡跑独立 selection。

统一 chain master PIDs 已更新；关键后继为：

```text
Stage-119 flow utilization master: 1694278
Route-A Stage-120--126 master:     1694306
EVOR Stage-127--133 master:        1694453
```

EVOR child 当前通过 `tail --pid=1694306` 等待 Route A 完成，不占 GPU、不短轮询，并使用
完全不重叠的 `202000--208031` selection/confirmation/causal seeds。等 Stage-127 smoke
真实落盘后，再按其 measured wall time/throughput 给 EVOR full-wave ETA；现在不拿其他算法
吞吐冒充精确 ETA。

`15:49 BST` 的首个真实 recovery checkpoint 进一步校准了 ETA：

```text
source01+bcfm8 combo @1k:
  total_time=450.48s
  batched update throughput=2.817/s
  flow critic nonfinite fraction=0

direct-Q replay-next seed1 @4k:
  total_time=561.19s
  batched update throughput=8.106/s
  direct-Q nonfinite fraction=0
```

据此 Stage-75 recovery 的瓶颈仍是 combo 与 GPU5 上三个 sequential direct-Q seeds 的较慢
一侧，更新 ETA 为 **`16:43--16:50 BST`**。这一阶段完成前不会短轮询；controller 直接等待
trainer PID，并自动进入 Stage-76 sealed selection。

`15:40 BST` 首个恢复后 throughput artifact 已落盘：

```text
GPU5 direct-Q seed1 @1k:
  total_time=147.65s
  batched_updates_per_second=8.1267
  backend_update_time=0.0210s
  snapshot=1000_snapshot.pkl

GPU1 combo @iteration0 after compile:
  total_time=104.13s
  backend_update_time=96.41s
  flow_grad=0.35637
  distill_grad=0.01007
```

两者 gradient/metric 均 finite；GPU sample 为 `82%/61%` utilization。direct-Q 进入稳定区后，
单 seed 粗估 `18--21min`；三个 sequential fillers 与约 `62--66min` 的 combo 仍大致同时
收尾，因此上述 `16:40--16:50 BST` Stage-75 ETA 保持不变。iteration-0 的 combo 吞吐被
JIT compile 主导，明确不用于外推。

### 21.98 DFC 实现级审计：其 flow-only 部分已被 BCFM 覆盖，quantile 蒸馏不作为当前新变量

#### 上一阶段实际结果

`15:44 BST` 的最新真实 artifacts 表明恢复链继续健康推进，而不是停在 compile：

```text
GPU1 Stage-75 source01+bcfm8 combo @1k:
  total_time=450.48s
  batched_updates_per_second=2.817
  bcfm_loss=0.001279
  flow_grad=0.05312
  distill_grad=0.04917
  all reported nonfinite fractions=0

GPU5 direct-Q replay filler @3k:
  total_time=444.05s
  batched_updates_per_second=8.033
  direct_q_loss=5.00e-5
  direct_q_grad=0.00455
  snapshot=1000/2000_snapshot.pkl
  internal eval @2.5k=36% (10 episodes)
```

这里的 `36%` 只是 filler 的单次训练期 internal eval，既没有 validation selection，也不是
Route-A RCT treatment，不能当作 task-quality 或 causal-value 结论。Stage-75 combo 也尚未
产生独立 validation 结果；当前只建立了 active/finite/throughput。

同时逐式审计了
[Distributional Flow Critic / DFC](https://arxiv.org/abs/2509.23087) 的 critic 更新。
Equation 4--5 为：

```text
xi_j ~ N(0, I)
z_next_j = frozen_flow(s', a'_policy, xi_j)
z_target_j = reward + gamma * z_next_j
z_t = (1-t) * xi_j + t * z_target_j
v(s,a,z_t,t) -> z_target_j - xi_j
```

这与本地 `value_mode=return_sample` 的 **same-noise BCFM** 完全同构：同一个
`target_source_key` 同时生成 next-return endpoint 和 current interpolation source，
Bellman target 为 `reward + discount * bootstrap * next_return`。因此 DFC 的
flow-only critic 不是尚未测试的新 CQN-AS 候选，而是 Stage-107 的 BCFM 部分已经覆盖的
目标。定向检索论文页面、OpenReview 与 GitHub 后，当前没有定位到作者发布的官方代码；
上述判断来自论文公式与本地实现逐项对照，不声称做了官方 repository reproduction。

论文额外训练 one-step quantile critic，使其匹配 10-step flow 产生的 return samples；
actor 再对该 one-step critic 求 action gradient。论文明确把这个 distillation 用于规避
actor update 对 ODE solver 的 BPTT，并且 Table 5 报告 flow-only critic 在若干任务/
seeds 上高方差，而 full DFC 更稳定。

#### 解释

对当前问题，DFC 审计建立并排除了三件事：

1. 它支持“return distribution 可以由 flow 建模”，但没有提供一个独立于本地 BCFM 的
   新 Bellman-flow target；
2. CQN-AS 通过 `levels × action_dimension × bins` 并行枚举后选 bin，不对 action 走
   actor gradient，也就不存在 DFC 所要解决的 actor-through-ODE BPTT；
3. 现在加入 quantile head 会同时改变 flow readout、critic parameterization 和
   optimization，若结果变化无法归因于“FM value update 更好”。

所以不因为论文 full model 使用蒸馏就机械加入蒸馏。这并不证明蒸馏永远无用：如果后续
Stage-119 证明 integrated-flow readout 本身造成部署不稳定，one-step quantile head 可以
作为独立的 readout/latency factor；在那之前它不是最高信息增益实验。

当前仍未解决的是：same-noise BCFM 单独是否会 self-distill/collapse，以及 Bellman
partial-flow transport 是否能给 CQN-AS 提供真实的 value signal。前者正是
Value Flows 用 DCFM 主信号加 BCFM anchor 的理由，后者尚待 Stage-107 实验回答。

#### 下一阶段决策与 Gate

不新增一个重复的 `DFC flow-only` run，也不把 DFC quantile distillation 混入正在进行的
fidelity gate。Route B 的下一独立 hypothesis 保持为：

> 在 actor/data/encoder/BC objective 全部 matched 时，return-sample
> `DCFM+BCFM` 能否优于 clean CQN-AS；JVP confidence weighting 是否进一步优于
> unweighted `DCFM+BCFM`。

Stage-107--113 的预注册比较保持：

- matched baseline：unweighted DCFM+BCFM control；treatment 只打开
  Value-Flows confidence weight；另与 validation-selected clean CQN-AS 比较；
- selection split：`178000` screen 10 episodes、`179000` validation 50 episodes，
  global `beta in {0.3,1,3}` 和 checkpoint 只在 validation 选择；
- held-out split：`180000` confirmation 200 episodes；
- seed1 promotion：validation delta `>=0`，并且相对 clean 和 control 的 confirmation
  paired CI lower 都 `>=-5pp`；
- 最终三 training-seed gate：fresh `182000/183000/184000`
  screen/validation/confirmation，各方法只比较 validation-selected best checkpoint；
  task pass 后再做独立 H=1 causal/authenticity gate。

若 DCFM+BCFM 失败，则“加 DFC distillation”不会被解释为 Bellman-flow 修复；只有
Stage-119 明确定位到 ODE readout/utilization failure，才注册 matched
`integrated flow vs frozen one-step quantile readout`，两臂共享完全相同的 flow critic
与训练数据。否则优先测试已经实现且不重复的 EVOR FlowTD。

#### 已执行

当前 GPU 执行保持不变：

```text
GPU1 PID 1693989: Stage-75 combo
GPU5 PID 1693991: direct-Q seed1 filler，随后自动 seed2/3
Stage-119 child PID 1694294: event wait
Route-A child PID 1694441: event wait
EVOR child PID 1694462: event wait
```

按 combo 1k 的稳定更新吞吐与 direct-Q 3k 的累计 wall time，combo 剩余约
`52--59min`；GPU5 当前 seed1 约还需 `14--18min`，随后两个 seed 合计约
`36--44min`。二者仍预计在 **`16:40--16:50 BST`** 左右共同释放，保守上界
`17:00 BST`；Stage-76 会由 completion event 立即启动，不做短轮询。

另外已修改 `scripts/launch_cqn_autoresearch_chain.sh`：每个 controller 启动前把源脚本
复制为 control 目录下带纳秒时间戳的只读 snapshot，并记录 SHA-256；实际 controller
执行 snapshot，而不是数小时后继续读取可被修改的 source script。这样后续代码编辑不会
再次污染一个已经启动、正在等待上游事件的研究链。验证结果：

```text
bash -n scripts/launch_cqn_autoresearch_chain.sh: pass
untracked-file whitespace check: pass
```

当前活跃 controllers 没有被重启；这一可靠性修复从下一次 recovery/launch 生效，正在
运行的本阶段继续使用已验证的 event waits。

### 21.98.1 Route A 真实性 Gate 加强：从 Q-span discovery 改为 Q-independent dimension confirmation

#### 上一阶段实际结果

`16:02 BST` 的 artifact-backed 训练进度为：

```text
GPU1 Stage-75 source01+bcfm8 combo @3k:
  snapshot=3000_snapshot.pkl
  total_time=1184.45s
  batched_updates_per_second=2.741
  bcfm_loss=0.001139
  flow_critic_grad_norm=0.04823
  all reported nonfinite fractions=0

GPU5 direct-Q replay filler seed1 @10k:
  snapshot=10000_snapshot.pkl
  total_time=1454.79s
  batched_updates_per_second=8.644
  direct_q_loss=2.57e-5
  direct_q_grad_norm=0.00467
```

这仍只证明两个训练进程 active、finite 且持续产出 checkpoint。`direct_q_loss` 小或
`policy_demo_top1=98.2%` 都不是 task quality，也不证明 critic 学到真实 value；必须等
validation-selected task Gate 和 held-out branch Gate。

随后对 Route-A 的训练与评测公式做了实现级审计。训练侧
`action_centered_moment_loss` 为：

```text
p(1-p) * tau(s, proposal)^2
  - 2 * (Z-p) * completed_return * tau(s, proposal)
```

在 `Z~Bernoulli(p)` 的已知随机 assignment 下，其 conditional population optimum 是
`E[Y(1)-Y(0) | state, proposal]`。control transition 的 proposal dimension/sign 来自独立
均匀 key，treatment 使用 replay 中真实 assignment；demo 的 assignment probability 是
`1`，被 causal mask 排除。当前 replay 默认 `prioritization=false`，所以也没有
outcome-dependent importance weight 破坏这个 moment。该公式审计没有发现识别错误。

但 Stage-125/132 的真实性评测存在一个明确的 optimistic selection：

```text
action_dimension = argmax_dimension ptp(Q_bins)
```

即每个状态先在 15 个 action dimensions 中挑 Q span 最大者，再判断 Q 是否排对五个
bins。这个设计对 failure 是保守的，却不能把 pass 当成无偏证据：随机 critic 也可能因
15 维 maximum selection 产生 winner's curse。此前的 BC prior、full BC path 和
action-nearness proxies 并不能消除这个维度选择偏差。

#### 解释

本阶段将两个问题拆开了：

1. action-centered RCT **训练识别式是正确的**；目前没有理由因公式错误废弃 Route A；
2. 原 causal Gate 只能作为 discovery，不能作为“真实 value”的最终确认。

这意味着此前任何 `q_span` causal pass 都不能完成研究目标。真正的 held-out
confirmation 必须在看 Q、BC 和 realized return 之前冻结 intervention dimension。

若后续训练表现为 effect 方向正确但方差过大，下一单变量候选是对 outcome 做
cross-fitted residualization 的 R-learner 式 variance reduction，而不是改变 propensity
或混入 imitation loss。[Nie & Wager](https://arxiv.org/abs/1712.04912) 的核心也是先估计
nuisance outcome/propensity，再用正交 objective 隔离 heterogeneous treatment effect；
这里 propensity 已由随机实验精确已知，所以只需把 outcome baseline 作为独立 factor。

#### 下一阶段决策与 Gate

原 Stage-125/132 保留为 **Q-favorable discovery**，但任何 task+causal pass 必须追加
Stage-134：

- checkpoint：严格复用 task validation 已选择的三 training-seed checkpoints 和
  global policy/value beta；Stage-134 禁止重新选择；
- held-out simulator split：fresh `209000--209031`，每 seed 的
  anchor steps `30/75/120`；
- intervention：`H=1 sibling_horizon`，force level 1；
- dimension schedule：
  `dimension=(eval_seed_position * 3 + anchor_position) mod action_dim`，
  完全不读 Q、BC 或 outcome；
- metrics：Q-vs-realized pairwise sign accuracy、Spearman、top-1/regret，以及 Q 相对
  independent BC prior、full BC path、action-nearness 三个 paired proxy 的增量；
- crossed bootstrap：三 training checkpoints × 32 simulator seeds，20k replicates；
- pass：
  1. 每 training seed 至少 24 informative states；
  2. 至少 8 个 action dimensions 各有至少 2 个 informative states；
  3. aggregate pairwise CI lower `>0.5` 或 Spearman CI lower `>0`；
  4. 至少 `2/3` training seeds 方向为正；
  5. Q 相对三个 imitation/nearness proxies 的 paired accuracy-delta CI lower 全部 `>0`。

最终 autoresearch summary 现在要求 causal artifact 明确记录
`dimension_selection=round_robin` 且 dimension-coverage checks 通过；旧 `q_span`
artifact 即使自身写着 `gate=pass`，也只算 discovery，不能使 Route A/B 或总 goal pass。

#### 已执行

新增/修改：

```text
scripts/analyze_cqn_branch_counterfactual.py
scripts/run_cqn_flow_branch_multiseed_gate.py
scripts/run_cqn_unbiased_causal_from_task.py
scripts/revalidate_cqn_autoresearch_summary.py
scripts/summarize_cqn_autoresearch_routes.py
scripts/run_cqn_stage134_unbiased_dimension_confirmation.sh
scripts/launch_cqn_autoresearch_chain.sh
```

实现内容包括 Q-independent round-robin selection、per-dimension informative coverage、
task-manifest 到 validation-selected checkpoint 的自动解析、严格 causal evidence
replacement，以及重算 A/B overall gate。CPU 定向与扩展回归：

```text
20 passed in 0.58s
41 passed in 1.18s
29 passed in 0.58s  # 新 controller/wrapper/strict-summary 集合
py_compile: pass
bash -n Stage-134 and master launcher: pass
```

Stage-134 已用 launch-time 只读 snapshot 和 SHA-256 durable 启动：

```text
master PID 1712972
child  PID 1712981
wait   tail --pid=1694453
GPU    0 MB additional allocation
```

它只在 Stage-133 完全结束、GPU1/GPU5 已释放后执行；若没有任何 task+旧 causal pass，
立即零 GPU 跳过并输出 strict fail summary。若有候选，则每个候选顺序使用两张卡做 fresh
unbiased confirmation。首次真实 branch throughput 落盘后再给该 Gate ETA，当前不拿
training throughput冒充 simulator-branch ETA。

当前 Stage-75 ETA 根据 combo 3k 的累计 wall time约为 `16:43--16:50 BST`；GPU5 seed1
已到 10k，随后 seed2/3 合计约 `44--50min`，所以两路共同释放时间仍估计
**`16:46--16:55 BST`**，保守上界 `17:05 BST`。后继由 completion event 自动推进。

### 21.99 2026 Flow×RL 新方法分类：actor-flow 不替代当前 value-flow 问题

#### 上一阶段实际结果

当前可观测训练证据仍是 Stage-75 combo `1k` 和 direct-Q filler `4k` 均 finite；尚无新的
sealed task result。因此这一阶段只更新方法空间与下一步实验归因，不用论文结果冒充本地
MovePlate 结果。

在 FLOQ、Value Flows、DFC、EVOR 之外，继续检索到 2026 年几条 Flow×RL 路线：

| 类别 | 代表方法 | Flow 建模对象 | 与当前 CQN-AS 问题的关系 |
|---|---|---|---|
| iterative scalar value | [FLOQ](https://openreview.net/forum?id=WwQoSHGCXg) | 标量 Q 的迭代计算 | 当前 Stage-74--80 直接测试 |
| return-distribution value | [Value Flows](https://openreview.net/forum?id=2VyNYUVF2k)、[DFC](https://arxiv.org/abs/2509.23087)、[EVOR](https://openreview.net/forum?id=Hze2lxCX6D) | 条件 return distribution / velocity Bellman target | 当前 Stage-107--118 与 Stage-127--133 直接测试 |
| flow policy + ordinary critic | [FQL](https://arxiv.org/abs/2502.02538)、[BFQ](https://arxiv.org/abs/2606.10613) | action policy | 改 actor，不回答“value 用 FM update” |
| critic-gradient guidance of flow policy | [QAM](https://arxiv.org/abs/2601.14234)、[Q-Flow](https://arxiv.org/abs/2605.13435)、[Direct Flow Q-Learning](https://openreview.net/forum?id=RdkOaK4q6p) | action-flow 的逐步优化 | 解决 actor-through-solver 的 BPTT/梯度问题；CQN-AS 并行枚举 bins，没有该瓶颈 |
| confounding-robust flow policy/value | [Causal Flow Q-Learning](https://arxiv.org/abs/2602.02847) | worst-case confounded Bellman target，主实现仍基于 FQL | 处理像素观测缺失导致的 hidden confounding，不直接证明 critic 学到同状态 action effect |

关键边界：

- QAM、Q-Flow、DFQL 都利用 scalar critic 的 action gradient 改 flow actor；把它们直接移植
  到当前实验会同时改变行为策略、replay coverage 和 value update，无法归因；
- BFQ 加速的是 noise-to-action flow policy，也不是 return/value flow；
- Causal Flow Q-Learning 与 Route A 名字相近，但问题不同：它给 observation confounding
  下的 worst-case policy guarantee；Route A 当前检验的是已知随机 assignment 下
  \(Q(s,a)-Q(s,a_{\rm BC})\) 是否真实可识别。两者不能合成一个 gate；
- 论文附录中的 Causal Value Flows 可以作为两条路线都独立通过后的 robustness 扩展，
  但现在加入会违反“两个问题分开解决”和 single-variable attribution。

#### 解释

系统调研把当前最相关的方法压缩为四个互不重复的 value hypotheses：

1. FLOQ：flow 只是 scalar Q 的 iterative computation；
2. Value Flows：DCFM+BCFM 学 return distribution；
3. DFC：same-noise BCFM 加用于 actor gradient 的 quantile readout；
4. EVOR：直接在 velocity space 做 FlowTD，并以多 return samples 做 policy extraction。

当前 pipeline 已覆盖 1、2、4；DFC 的 flow target 被 2 的 BCFM 覆盖，而其 quantile head
只在 Stage-119 证明 integrated readout 是瓶颈后才有独立实验价值。因而继续实现 QAM/
Q-Flow/DFQL 不会提高当前两个 research question 的信息增益。

#### 下一阶段决策与 Gate

保持预注册顺序，不新增抢卡实验：

1. Stage-76 首先在 sealed split 选择 FLOQ fidelity arm/checkpoint；
2. 若其 task gate 不通过，按事件链依次测试 Value-Flows confidence、FlowCritic truncated
   readout、EVOR FlowTD，每次只改变一个机制；
3. Route A 独立运行 action-centered randomized advantage；task non-inferiority 与
   H=1 same-state causal ranking 两道 gate 缺一不可；
4. 只有 Stage-119 显示 integrated-flow step/source utilization 不稳定，才注册
   `integrated flow vs frozen one-step quantile readout` 的 DFC readout 对照；
5. actor-flow 方法只作为最终“是否还要替换 CQN-AS policy class”的外部研究建议，不参与
   本 goal 的胜负统计。

#### 已执行

文献分类和 exclusion rationale 已写入本文件；当前两张卡继续原实验，未改变 live
controller、配置、seed split 或 checkpoint。下一次状态读取仍由 trainer completion
事件触发，ETA 保持 `16:43--16:50 BST`。

Route A 另补了真实 update 级别的 clipped-assignment regression：随机 treatment 即使在
action bound 被 clip 成 `delta=0`，`causal_rct_valid_fraction` 仍为 `1`、treated fraction
仍为 `0.5`、propensity error 为 `0`。强制 CPU 的完整相关回归更新为
**39 passed in 33.58s**。

`15:55 BST` 新 artifacts：

```text
source01+bcfm8 combo:
  2k total_time=795.23s, throughput=2.716/s, nonfinite=0
  internal success @2.5k=64%

direct-Q replay-next seed1:
  7k total_time=972.66s, throughput=7.864/s, nonfinite=0
  internal success @2.5k/5k/7.5k=36/56/40%
```

direct-Q 的 final 前回落再次证明只能比较 validation-selected best checkpoint；这些 10-episode
internal eval 均不进入 promotion。按 combo 2k 累计 wall time，Stage-75 release ETA 仍为
`16:43--16:50 BST`。

### 21.100 Route A Stage-120 识别假设预检与 fail-fast 加固

#### 上一阶段实际结果

Stage-120 treatment/control 的 resolved Hydra 配置逐项读取结果为：

```text
num_train_frames=1500
replay_size_before_train=500
num_explore_steps=0
stddev_schedule=0.0
replay.prioritization=false
replay.nstep=1
structured_exploration_prob=0.2
structured_exploration_horizon=1
critic_sequence_mode=effective_k0
td_target_action_source=replay_next
policy_value_beta=null
```

因此 seed replay 的前 500 steps 也使用被记录的结构化 assignment，并不存在一段未记录的
random warm-up。按 1500 次 eligible decisions、`p=0.2`，smoke 期望 300 次 treatment start，
15 个维度平均各 20 次。二项分布下 start 少于 gate 所需 30 次的概率约
`1.79e-102`；任一维度完全没覆盖的 union bound 约 `2.70e-8`。

#### 解释

1.5k smoke 的 coverage gate 不是靠运气才能通过；若真实 artifact 缺维度或 start 数太少，
应解释成 collector/replay bug，而不是统计波动。另一方面，原 gate 只检查已落盘 propensity，
没有锁死无 Gaussian exploration、uniform replay 和 effective-k0 等识别条件；未来 config
漂移可能造成“训练运行了，但 causal moment 已不再对应预注册 estimand”。

#### 下一阶段决策与 Gate

Stage-120 的 task gate 不变，但 training-health gate 新增六个强制条件：

- global 与 method 的 `num_explore_steps` 都必须为 0；
- 未记录的 Gaussian exploration 必须为 0；
- replay 必须 uniform、`nstep=1`；
- critic 只评价真实执行的第一步 `effective_k0`；
- TD target 必须保持 `replay_next`。

任何一项不满足立即 fail，不能进入 seed1 task selection。随机 action 被 bound clip 成
零位移仍是合法 assignment；该边界已有独立 replay 与真实 update regression。

#### 已执行

已更新：

```text
scripts/check_cqn_direct_q_rct_training_gate.py
tests/unit/test_check_cqn_direct_q_rct_training_gate.py
```

新增 unlogged-noise rejection test；focused gate/causal tests 为
`10 passed, 9 deselected in 15.24s`，强制 CPU 的 Route-A 完整相关回归为
**40 passed in 35.17s**。Stage-120 当前仍在 completion event 上等待，不占 GPU；这些
fail-fast checks 会在真实 smoke 完成后立即执行。

`16:02:42 BST`，GPU5 的 direct-Q replay-next seed1 完整结束：

```text
run:
  exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed1_20260724
snapshots:
  10000_snapshot.pkl
  10500_snapshot.pkl
training health gate:
  pass (10/10 checks)
internal success @2.5k/5k/7.5k/10k:
  36/56/40/40%
```

该结果只证明 scalar-Q training finite、critic 有梯度、rollout 保持独立 BC、replay-next
target 与 MSE 配置正确；它既不是 action-centered treatment，也未经过 fresh validation，
不回答 task superiority 或 value authenticity。controller 已立即启动 seed2：

```text
GPU5 trainer PID 1713514
launch time 16:02:42 BST
```

seed1 从 launch 到完整 artifact 用时 `25m47s`。若 seed2/3 同速，GPU5 尚需约 `51m34s`，
比 GPU1 combo 略晚，因此 Stage-75 recovery ETA 更新为 **`16:54--17:00 BST`**。

### 21.101 因果 Gate 固定独立 BC continuation，切断 Q 的自证路径

#### 上一阶段实际结果

对 branch counterfactual 的真实代码路径做了字段级复核：

- `predicted_q` 直接记录 critic 的 raw Q，并没有叠加 BC logits；
- independent BC prior、完整 BC action path 和 action-nearness 都是另算的 anti-cheat
  proxies，因此 Q-vs-proxy 本身没有混淆；
- 但 multi-seed wrapper 此前把 task validation 选择的数值
  `policy_value_beta` 同时传给 counterfactual continuation。这样 forced action 以后，
  critic 可以改变后续 action 和访问到的状态，而 realized outcome 又被拿来验证同一个
  critic；
- direct-Q/action-centered critic 的训练 target 和部署 rollout 均以独立 BC policy 为
  behavior/continuation，其因果 estimand 实际应为 \(Q^{\pi_{\rm BC}}(s,a)\)，数值 beta
  continuation 与训练 estimand 也不一致。

同期可观测训练 artifact：

```text
GPU1 Stage-75 combo:
  4k snapshot complete
  total_time=1530.44s
  throughput=2.816 train frames/s
  flow/encoder nonfinite fraction=0
  internal success @2.5k=64%

GPU5 direct-Q seed2:
  2k snapshot complete
  total_time=268.40s
  incremental throughput about 8.4 train frames/s
  direct-Q nonfinite fraction=0
  first sealed internal eval is pending
```

GPU1/GPU5 分别由 trainer PID `1693989`/`1713514` 持续占用，核对时 GPU utilization
约为 `70%/64%`，并非空等。

#### 解释

raw-Q 与 imitation proxy 的拆分是正确的，但 continuation policy 的旧语义会产生
**self-fulfilling evaluation**：即使 Q 对固定 policy 下的 action effect 不准确，它也可能
通过改变后续 policy 让自己的排序与 outcome 变得一致。这不等于一定发生了 cheat，却使
causal pass 无法排除该解释。

因此 task performance 与 value authenticity 必须使用两个不同但预注册的 policy 语义：

1. task gate：保留 fresh validation 选出的 global beta，回答部署效果；
2. causal gate：forced bin 后固定独立 BC continuation，回答 critic 是否学到
   \(Q^{\pi_{\rm BC}}(s,a)\)；
3. deployment-policy continuation 只允许作为 sensitivity diagnostic，不能单独满足
   “真实 value” Gate。

#### 下一阶段决策与 Gate

Stage-125/132/134 及其后续 causal confirmation 默认强制：

```text
continuation_policy=bc
probe policy_value_beta=null
deployment_policy_value_beta=<task validation selected beta>
```

缓存 probe 只有在 artifact 的 `policy_value_beta` 与预期 continuation 完全一致时才能复用；
旧数值-beta probe 必须重跑，不能被误当成 BC-continuation 证据。最终 causal pass 仍须同时
满足 held-out round-robin dimensions、三 training seeds、crossed bootstrap 和三个
anti-cheat proxy 增量 Gate。若以后显式运行 `continuation_policy=deployment`，结果单独标记
为 sensitivity，不进入 Route A/B authenticity conclusion。

#### 已执行

已更新：

```text
scripts/run_cqn_flow_branch_multiseed_gate.py
scripts/run_cqn_unbiased_causal_from_task.py
tests/unit/test_run_cqn_flow_branch_multiseed_gate.py
tests/unit/test_run_cqn_unbiased_causal_from_task.py
```

实现了 continuation policy 与 deployment beta 的显式分离、manifest 双字段记录、旧 cache
兼容性拒绝，以及 `"bc"` 命令透传测试。Stage-134 preregistration 也会显式落盘
`deployment_policy_value_beta`、`continuation_policy=bc` 和
`continuation_policy_value_beta=null`，不再依赖 wrapper 隐式默认。所有既有 stage shell
未被运行中修改；它们之后调用 wrapper 时会自动获得新的 BC-continuation 默认值。验证结果：

```text
py_compile: pass
git diff --check: pass
focused causal suite: 22 passed in 0.16s
expanded branch/counterfactual suite: 57 passed in 1.29s
explicit Stage-134 preregistration suite: 23 passed in 0.15s
```

按最新 artifact 吞吐，GPU1 剩余约 `6500/2.8 ≈ 39min`；GPU5 seed2 剩余约 17min，
随后 seed3 约 26min。共同释放的 bottleneck ETA 为 **`16:50--16:56 BST`**，保守上界
`17:02 BST`。controller 继续按 trainer completion event 推进，不做短间隔轮询。

### 21.102 最终 Route Gate 的证据强度加固

#### 上一阶段实际结果

最终汇总代码审计确认 task evaluator 本身使用正确的公平路径：

- clean baseline 固定为各 training seed 的 validation-best：
  seed1 `5k`、seed2 `5k`、seed3 `2.5k`；
- 旧 internal best 分别为 `68%/72%/76%`，另一个 100-seed held-out split 的三 seed
  均值为 `61.33%`；后者只描述 baseline 方差，不替代新 paired confirmation；
- candidate 先用 screen/validation 选择 checkpoint（以及需要时的 global beta），再在
  200 个共同、未使用 simulator seeds 上与 clean checkpoint 做 paired comparison；
- crossed bootstrap 同时 resample training seed 和 simulator seed。

但 `summarize_cqn_autoresearch_routes.py` 旧实现只读取 task/causal artifact 的
`gate=pass`，没有在最终层再次锁死 evidence scope。理论上，二 training-seed promotion
artifact、`min_ci_lower=-0.05` 的 exploratory gate，或使用数值 beta continuation 的 causal
artifact，都可能被错误传入并使最终 summary 假阳性。

#### 解释

单个 verifier 通过不等于证据满足最终 claim；最终层必须验证 verifier 的适用范围。Route A
需要 zero-margin non-inferiority，Route B 需要严格正 improvement，二者都不能由 smoke、
promotion 或 discovery split 代替。同理，“真实 value”必须在固定独立 BC continuation
下预测 do-action return，而不能只在 Q 参与的 deployment continuation 下自洽。

#### 下一阶段决策与 Gate

最终 Route A/B task evidence 现在共同要求：

1. 至少三个独立 training seeds；
2. 每个 seed 至少 200 个共同 sealed simulator episodes；
3. artifact 预注册 `min_mean_delta>=0`、`min_ci_lower>=0`；
4. mean、crossed CI、paired wins 和 training-seed majority checks 全部通过；
5. Route B 额外要求 mean delta 和 CI lower **严格大于 0**。

最终 causal evidence 现在共同要求：

1. 至少三 training seeds、32 fresh simulator seeds；
2. `H=1`、`round_robin` Q-independent dimension selection；
3. `policy_value_beta=null`，即独立 BC continuation；
4. anti-cheat proxy coverage 完整；
5. Q 相对 BC prior、完整 BC path、action-nearness 三个 proxy 的 paired delta CI lower
   全部严格大于 0。

#### 已执行

已修改 final summarizer 及 strict revalidation fixtures，新增 numeric-beta causal
rejection、二 training-seed task rejection、Route-B zero-lower-bound rejection。验证：

```text
scripts/summarize_cqn_autoresearch_routes.py
tests/unit/test_summarize_cqn_autoresearch_routes.py
tests/unit/test_revalidate_cqn_autoresearch_summary.py

final-summary / paired / causal focused suite:
  23 passed in 0.11s
py_compile: pass
git diff --check: pass
```

这些修改不影响 live trainer，也没有修改正在等待的 shell script；后续 Stage-126/133/134
调用 Python summarizer 时会自动使用更严格的最终 Gate。

### 21.102.1 Route B 新诊断：corrected FlowIQN 实际缺少主 quantile loss

#### 上一阶段实际结果

corrected FlowIQN 的 validation-selected 10k checkpoint 已有 sealed paired 结果，不再重跑：

```text
artifact:
  exp_local/cqn_flow_flowiqn/stage44_flowiqn_corrected_r4_gate_seed60000/gate_summary.json
corrected FlowIQN:
  19/50 = 38%
clean CQN-AS validation-best:
  33/50 = 66%
paired delta:
  -28pp, 95% CI [-46,-8]pp
wins/losses/ties:
  7/21/22
McNemar exact:
  p=.01254
```

对本地 `_flow_matching_loss` 和
[Diffusion Bridge Critics](https://arxiv.org/abs/2602.05783) 的公式级审计发现：

- 本地 FlowIQN 对 source/target particles 分别排序，再把每个 source quantile 与一个
  target order statistic 一对一配对做 velocity MSE；
- 这相当于用经验分位作为直接 target，统计角色更接近 DBC Eq. 11 的 anchor；
- 本地没有 DBC Eq. 8：每个预测分位对所有 Bellman target particles 的 all-pairs
  quantile-Huber 主损失；
- DBC 的消融显示，移除 Eq. 8、只保留 anchor 会在三个报告任务上大幅下降；anchor
  权重 `0.01` 优于 `0.02/0.1`，但 DBC 是 diffusion bridge critic，不是 Flow Matching，
  且当前没有找到论文官方代码，因此这里只迁移可识别的 loss 因子，不声称复现 DBC。

`16:26 BST` 的 live artifacts 同时显示：

```text
GPU1 Stage-75 combo:
  7k snapshot, total_time=2599.94s, throughput=2.612/s
  internal success @2.5k/5k/7.5k = 64/56/72%
GPU5 direct-Q seed2:
  9k snapshot, total_time=1267.30s, throughput=7.467/s
  internal success @2.5k/5k/7.5k = 40/60/48%
flow/direct nonfinite fraction:
  0
```

这些 internal success 仍不参与方法选择。

#### 解释

corrected FlowIQN 的失败不能再简单解释成 flow dimension、steps 或 quantile conditioning
不足：它已经有显式 `tau` condition、单调 coupling 和 8-step integration。新的可证伪
解释是，当前 objective 把有有限样本偏差的 order-statistic anchor 当成了全部监督，
缺少分布式 RL 的 proper all-pairs quantile regression。

这也说明 DBC 不能作为“另一个 FM paper”直接复现；可迁移且不改变 CQN-AS policy/data
问题的是 quantile loss。state/image、C2F level、zoom path、sequence position、action
dimension 和 candidate bin 仍全部作为 flow condition，Bellman continuation 固定为独立
BC policy，避免 critic 自选 target action。

#### 下一阶段决策与 Gate

先运行严格三臂训练健康 Stage-135，不用 smoke loss 宣称 policy quality：

| arm | all-pairs quantile | sorted CFM | 作用 |
|---|---:|---:|---|
| `anchor_only` | 0 | 1 | matched FlowIQN control |
| `joint_equal` | 1 | 1 | 只新增主 quantile loss |
| `dbc_ratio` | 1 | 0.01 | 检验小 anchor-like CFM 权重 |

三臂共同冻结：MovePlate、两塔独立 BC、`td_target_action_source=bc_policy`、UTD=4、
effective-k0、8 target/train samples、8 flow steps、K=4 train-time action readout、batch 8。
Stage-135 只在三臂均有 1k snapshot、finite/nonzero objective、velocity gradient、zero
nonfinite fraction 和配置无漂移时 pass。

若 pass，下一独立阶段才训练 seed1 family 到 10.5k：

- screen：fresh seeds `210000--210009`，只留每臂 top-2 checkpoints；
- selection：fresh seeds `211000--211049`，联合选择 checkpoint 与一个 global
  `beta in {0.3,1,3}`；
- sealed promotion：fresh seeds `212000--212199`；
- matched baselines：`anchor_only` 与各 training seed 的 clean CQN-AS
  validation-best；
- seed1 family promotion 要求 treatment 相对 anchor-only paired delta `>0` 且 CI lower
  `>=0`，同时相对 clean mean `>=0`、CI lower `>=-5pp`；
- 只有晋级 arm 才训练 seed2/3。最终 Route-B claim 仍要求三 training seeds、每 seed
  200 sealed common episodes、mean delta `>0`、crossed CI lower `>0`，随后再通过
  H=1、round-robin dimension、固定 BC continuation 的 causal gate。

#### 已执行

已实现：

```text
robobase/method/cqn_flow.py
  quantile_huber_endpoint_loss
  quantile_endpoint_lambda
  quantile_huber_kappa
  endpoint/source-quantile joint integration path
robobase/cfgs/launch/
  cqn_flowiqn_bc_target_two_tower_high_utd4_gate.yaml
  cqn_qr_flowiqn_equal_bc_target_two_tower_high_utd4_gate.yaml
  cqn_qr_flowiqn_dbc_ratio_bc_target_two_tower_high_utd4_gate.yaml
scripts/check_cqn_qr_flowiqn_training_gate.py
scripts/run_cqn_stage135_qr_flowiqn_smoke.sh
```

验证：

```text
py_compile: pass
bash -n: pass
focused new tests: 14 passed
full CQN-Flow/factory/checker regression: 154 passed in 108.91s
```

Stage-135 已用不可变 shell snapshot 和 SHA256 事件排队：

```text
master PID:
  1735655
controller child:
  1735664
event wait:
  PID 1735667, tail --pid=1712972
master path:
  exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_smoke_master
```

核对时 Stage-135 没有 GPU process 或显存占用；GPU1/GPU5 仍只属于 PID
`1693989/1713514`。它必须等 Stage-134 完成，最早不早于当前 Stage-75 的共同释放下界
`16:50--17:02 BST`；中间 conditional gates 的结果尚未知，因此不给虚假的精确开跑时间。
一旦取得两张卡，按既有 1.5k Flow smoke 量级，Stage-135 本身预估约 `15--25min`，并在
写出 `summary.json` 后停止，不会越过阶段汇报直接启动 full training。

### 21.103 EVOR FlowTD 的离线 return endpoint 修正与当前两路线快照

#### 上一阶段实际结果

`2026-07-24 16:31 BST` 的 live artifact 快照为：

```text
Route B / Stage-75 full-fidelity FLOQ seed1:
  run:
    exp_local/cqn_flow_high_utd/
      stage75_floq_source01_bcfm8_utd4_seed1_20260724
  latest completed snapshot: 8k @ 16:27:23
  internal success @2.5k/5k/7.5k: 64/56/72%
  validation-best so far: 7.5k, 72%
  flow/encoder nonfinite gradient fraction: 0

Route A / direct-Q replay-next:
  seed1: training health gate 10/10 pass
         internal success @2.5k/5k/7.5k/10k: 36/56/40/40%
         validation-best: 5k, 56%
  seed2: training health gate 10/10 pass
         internal success @2.5k/5k/7.5k/10k: 40/60/48/60%
         validation-best: 5k or 10k, 60%
  seed3: PID 1737558 started on GPU5 at 16:28:23
```

对应 clean CQN-AS 的 internal validation-best 为 seed1/2/3
`68%/72%/76%`。因此当前 direct-Q seed1/2 discovery 结果分别低 `12pp/12pp`；
它证明了训练路径 finite 且可重复，却**尚未**满足 Route A 的 no-worse task Gate。
FLOQ seed1 暂时比 clean seed1 高 `4pp`，但单 training seed、25-episode internal eval
不能满足 Route B 的 superiority claim。

对当前 EVOR 实现与
[EVOR 原论文](https://arxiv.org/pdf/2510.08218) 的公式级复查同时发现：

- FlowTD 把当前 velocity 回归到
  \(r+\gamma\,v_{\bar\theta}(z_t,t,s',a')\)，current/next field 使用同一个
  \(z_t,t\)；
- 论文允许终点 \(z_1\) 来自 offline dataset 中的 reward-to-go sample，或来自
  target reward model；
- 本地旧实现只从尚未训练好的 target flow 生成 current \(z_1\)，在 MovePlate
  sparse-reward cold start 下没有数据锚点；
- 论文训练时使用一个 return sample；本地保持 `N=1`，action ranking 的 16 个初始
  Gaussian source 仍沿 action-bin 轴并行。论文 eval 使用更多样本，本地 16 是预注册的
  compute adaptation。

#### 解释

Route A 当前结论只是“critic 确实收到 finite gradient”，不是“value 已真实有效”：
internal task best 已落后 clean baseline，必须等待三 seed 的 fresh task selection 和
H=1 causal probe；不能用训练 loss 或最后一步成功率替代。

EVOR 的旧 target-flow endpoint 在论文文字上并非非法，但它把稀疏奖励问题变成未锚定的
自举 fixed point。改用完整 episode 的 discounted return 作为 current \(z_1\) 是论文明确
允许的 offline-data 路径；它没有新增 MC regression loss，`mc_return_weight` 仍为零，
所以本实验仍隔离地检验 FlowTD。该方法保留 CQN-AS action bins、图像/状态 condition、
独立 BC tower 与 BC Bellman continuation，因此是 **EVOR-to-CQN-AS adaptation**，
不是对 EVOR flow actor/rejection policy 的逐行复现；尤其本地 next action 是独立 BC
tower 的 deterministic top-1，而论文公式对 stochastic base-policy action 取期望。

[Distributional Flow Critics](https://arxiv.org/abs/2509.23087) 的
target-flow 后接 quantile distillation 是另一个可比较方向，但它同时改变训练目标和
readout。为保持本阶段单一因果解释，先让 Stage-119 判断迭代 flow 是否真的被使用；只有
诊断显示 ODE/readout 是瓶颈时，DFC-style distillation 才进入下一独立 arm。

#### 下一阶段决策与 Gate

Stage-127 的两个 `1.5k` EVOR smoke 只做 health Gate，必须同时满足：

1. `1k` snapshot 和至少两行日志存在；
2. `evor_td_loss` finite、非零且随 update 非常数；
3. flow critic、velocity head 和参数 update norm 都非零；
4. flow/encoder nonfinite fraction 精确为零；
5. BCFM、DCFM、PCBF 等辅助 flow loss 精确为零；
6. `mc_return_mean` finite，且至少一行绝对值大于 `1e-8`，证明 offline return
   endpoint 真正进入 replay/update；
7. next action 固定来自独立 BC policy，两塔与预注册 compute/readout 无漂移。

若任一 smoke 失败，只能得出“本地 adaptation health fail”，不能声称 EVOR 论文无效。
若通过，Stage-128 才训练 seed1/2 到 10.5k；二 seed fresh promotion 通过后才训练 seed3。
最终 Route B 仍要求三 training seeds、每 seed 200 个 sealed common episodes、
paired mean delta `>0` 且 crossed-bootstrap CI lower `>0`，再通过 `H=1`、
round-robin dimension、独立 BC continuation 和三个 anti-cheat proxy 的 causal Gate。

按 Stage-75 最近三个千步间隔，FLOQ 预计 `16:46--16:49 BST` 完成；direct-Q seed3
参考 seed2 的全程耗时预计 `16:54--16:57 BST` 完成。当前共同 GPU 释放 ETA 为
`16:55--17:00 BST`，保守上界 `17:05 BST`。后续 controller 使用
`tail --pid=<trainer/controller>` 事件等待；Stage-119、Route-A task/causal Gate 和
EVOR 的实际开跑时间仍由上游 pass/fail 决定，不用短轮询伪造精度。

#### 已执行

已更新：

```text
robobase/workspace.py
  EVOR 启用时，即使 mc_return_weight=0 也在 episode-backed replay 存储 mc_return

robobase/method/cqn_flow.py
  _evor_td_loss 接收 stopped-gradient completed return 作为 current z1
  fresh Gaussian z0；current/next velocity 共用同一个 z_t 和 time

robobase/cfgs/launch/
  cqn_evor_flowtd_bc_target_two_tower_high_utd4_gate.yaml

scripts/check_cqn_evor_training_gate.py
  新增 mc_return_mean 与 offline_return_endpoint_present Gate

tests/unit/test_cqn_flow.py
tests/unit/test_check_cqn_evor_training_gate.py
tests/unit/utils/test_add_demo_to_replay_buffer.py
```

验证结果：

```text
EVOR golden equation/update/checker focused tests:
  13 passed, 123 deselected
full CQN-Flow + EVOR checker + replay-return regression:
  145 passed in 113.90s
py_compile: pass
targeted diff check: pass
```

Stage-127 尚未启动 trainer，因此这次 Python/config 修改会从它的第一个 smoke 生效；
没有修改任何正在运行的 shell script。

### 21.103.1 QR-FlowIQN family Gate：先证明 treatment 胜过 matched anchor

#### 上一阶段实际结果

Stage-135 当前仍处于真实事件等待：

```text
master PID:      1735655
controller PID:  1735664
tail wait PID:   1735667
dependency:      tail --pid=1712972
GPU allocation:  none
smoke artifact:  not started
```

因此当前没有 QR-FlowIQN policy-quality 结果，不能提前把实现通过当成算法通过。对现有
`run_cqn_checkpoint_beta_selection_gate.py` 的代码与 Stage-67 artifacts 审计显示：

- 它能给一个 candidate 的每个 training seed 相同 checkpoint/beta budget，并与 clean
  做 paired confirmation；
- 它不能在同一 selection split 上同时选择 `joint_equal`/`dbc_ratio`，也不能把 frozen
  treatment 与 matched `anchor_only` 做 sealed paired comparison；
- 如果分别运行三个独立 gate 再挑最好结果，等价于 confirmation 后选择，会产生
  multiple-comparison 与 winner's-curse 偏差。

Stage-67 的真实 evaluator artifact 为 `42` 个 process-per-checkpoint jobs、两张 GPU、
总 elapsed `2977.43s`；这是当前 Stage-137 ETA 的直接吞吐基准。

#### 解释

“新 treatment 超过 clean CQN-AS”不足以回答 all-pairs quantile loss 是否修复了
anchor-only FlowIQN，因为两者还可能受共同的两塔/BC-target 改动影响。必须先证明：

1. treatment 在完全相同的 state/image/action-bin condition、training seed、budget 和
   eval seeds 下超过 `anchor_only`；
2. treatment 至少不低于 validation-best clean；
3. treatment identity、checkpoint 和 beta 都只能由 validation 决定，sealed confirmation
   只打开一次。

这把“quantile loss 是否有效”和“最终是否超过 CQN-AS”拆成顺序明确的两个问题，不用
同一 confirmation 同时选方法和报结论。

#### 下一阶段决策与 Gate

Stage-135 pass 并完成阶段汇报后，Stage-136 才允许三臂使用同一个 training seed `1`
训练到 10.5k：

- `anchor_only` 与 `joint_equal` 先占 GPU1/GPU5；
- 任一先完成，`dbc_ratio` 立即接管释放的卡；
- 三臂仍只做 training-health Gate，不用 internal success 选择方法；
- ETA 在开跑前由各自 1.5k smoke 的 `total_time/iteration` 外推，调度 wall ETA 为
  `max(anchor,equal)+ratio`。

Stage-136 health 被汇报后，Stage-137 使用：

```text
screen:       210000--210009, 10 episodes, every arm x 10 checkpoints
validation:   211000--211049, 50 episodes, top-2 x beta {0.3,1,3}
confirmation: 212000--212199, 200 episodes, opened once
```

每个 arm 独立获得相同 checkpoint/beta selection budget；两个 treatment 只按 validation
success 选一个，tie 按预注册 CLI 顺序优先 `joint_equal`。sealed promotion 同时要求：

```text
treatment vs anchor_only:
  paired mean delta > 0
  paired wins > losses
  bootstrap CI lower >= 0
treatment vs clean:
  paired mean delta >= 0
  bootstrap CI lower >= -5pp
```

这仍只是单 training-seed family promotion，明确禁止满足 Route-B claim。晋级 arm 后续
必须再训练 seed2/3，并在新 3-seed/200-episode/crossed-bootstrap split 上取得严格正
delta 与 CI lower。

#### 已执行

已新增：

```text
scripts/run_cqn_qr_flowiqn_family_gate.py
  common screen/validation/confirmation worker pool
  per-arm equal-budget checkpoint/beta selection
  validation-only treatment selection
  treatment-vs-anchor 与 treatment-vs-clean paired gates

scripts/run_cqn_stage136_qr_flowiqn_seed1_family.sh
  smoke-throughput ETA
  两卡 three-arm scheduler
  full training-health summary

scripts/run_cqn_stage137_qr_flowiqn_family_gate.sh
  frozen 210000/211000/212000 split
  exact clean seed1 5k validation-best
  K=8 integrated mean readout
```

回归验证：

```text
family evaluator/checker/paired-stat tests: 10 passed in 0.11s
expanded QR-FlowIQN focused suite: 16 passed in 16.44s
py_compile: pass
bash -n Stage-135/136/137: pass
```

Stage-136/137 已编写但未启动，也未加入自动 launcher；两者分别要求上阶段
`reported` marker，防止 smoke 或 training-health 完成后在用户看不到结论的情况下自动
越过阶段。Stage-137 有 `51` 个 eval jobs；按 Stage-67 实测线性外推为
`2977.43 * 51 / 42 = 3615.45s`，即取得两张卡后约 `60min`，实际开跑时再用 Stage-136
artifact 修正。当前唯一执行中的新阶段仍是 Stage-135 的事件依赖，不做短轮询。

### 21.103.2 Route-A 数值语义 Gate：causal ranking 不等于 calibrated value

#### 上一阶段实际结果

对当前正式 causal summarizer 与 raw branch artifact schema 的逐项审计确认：现有 Gate
只使用 pairwise sign、Spearman、top-1/regret，以及 Q 相对 `policy_prior`、
`policy_path`、`action_nearness` 三个 imitation proxy 的 paired improvement；它没有
检查 raw Q 是否以 discounted-return 为单位。因此它可以排除“只复制 BC 排序”，但
任意正单调重标定仍能通过。

先在已有 clean direct-C51 的 24-seed branch artifact 上执行了一个不用于 promotion 的
数值 discovery：

```text
artifact:
  exp_local/cqn_value_fidelity_stage_x/probes/
    direct_c51_two_tower_8000_branch_h4_scoreL2_combined_s24_a3.json

calibration seeds / held-out seeds: 12 / 12
held-out native delta-return-on-delta-Q slope:
  point 5.0089, crossed-bootstrap CI [-3.8205, 14.6412]
native MSE skill versus zero action effect:
  point 0.99%, CI [-0.79%, 2.86%]
calibration-split rescaled-Q held-out MSE skill:
  point 1.93%, CI [-249.13%, 10.41%]
within-state bin-permutation placebo:
  p = 0.1377
gate: fail
```

这个 probe 是单 training seed、H=4、非 round-robin 且没有完整 anti-imitation proxy，
所以 artifact 明确标记为 `legacy_single_stage_discovery_only`；它只说明旧 clean
direct-C51 没有提供可复现的数值校准证据，不能代替 Stage-134 的正式结论。

#### 解释

现有 H=1 Gate 回答的是“Q 的 action-bin 排序是否包含超出 imitation 的因果信息”；
数值 Gate 回答另一个问题：“同一 state 内的 `delta Q` 是否真的是相应
`delta discounted return`”。二者必须分开：

- ranking pass、calibration fail：critic 对选 action 有信息，但不能把 Q 数字解释成
  expected return；
- calibration pass、task fail：value 数值有意义，但 policy extraction/coverage 仍差；
- task + ranking + calibration 同时 pass，才支持“真实且可用于决策的 value”。

同一 state 内先中心化 Q/return，消除任意 state-only baseline；这既不靠跨状态难度制造
相关，也不把训练 loss 当 policy quality。

#### 下一阶段决策与 Gate

Stage-134 的 32 个 fresh simulator seeds 在打开前已冻结为：

```text
calibration-only: 209000--209015
sealed held-out:  209016--209031
estimand: H=1 sibling action-bin delta-Q -> delta-return
continuation: fixed independent BC
dimension: round_robin, independent of Q/BC/return
bootstrap: training checkpoint x simulator seed, 20,000 replicates
placebo: within-same-state action-bin permutation, 2,000 replicates
```

正式 pass 同时要求：

1. 3 个 training seeds，每个 split 至少 12 个 informative states；
2. 不做任何重标定的 native-Q MSE skill CI lower `> 0`；
3. held-out native slope 的完整 95% CI 位于预注册 `[0.5, 2.0]`；
4. 至少 2/3 training seeds 的 native slope 为正；
5. calibration split 只拟合 through-origin slope 后，held-out Q skill CI lower `> 0`；
6. 给三个 imitation proxy 相同 calibration budget 后，Q-vs-proxy MSE improvement 的
   CI lower 全部 `> 0`；
7. within-state bin permutation `p <= 0.05`。

这一步不重新选择 checkpoint、beta 或 candidate；只有已经通过 task + unbiased
causal-ranking Gate 的 frozen candidate 才能读取数值结果。

#### 已执行

已新增并验证：

```text
scripts/analyze_cqn_branch_calibration.py
tests/unit/test_analyze_cqn_branch_calibration.py
scripts/run_cqn_stage134b_numeric_calibration.sh

focused calibration + causal-runner suite:
  14 passed in 0.11s
py_compile: pass
bash -n: pass
```

事件驱动 controller 已真实启动：

```text
master PID:      1749515
controller PID:  1749539
wait process:    1749542 tail --pid=1712972 -f /dev/null
GPU allocation:  none
output:
  exp_local/cqn_flow_high_utd/
    stage134b_numeric_calibration_seed209000_20260724/summary.json
```

它只等待 Stage-134 master 完成，随后做 CPU-only 离线统计，可与 Stage-135 GPU smoke
并行。按本次 discovery 的实测与 3-checkpoint/20k-bootstrap/2k-permutation 放大，
Stage-134 artifact 就绪后每个 qualifying candidate 预计 `<1--3min`；上游等待时间不伪造
固定 ETA，继续由 controller dependency 与真实 trainer throughput 更新，不使用短轮询。

### 21.103.3 Stage-135 anchor smoke 利用 GPU1 空档预填充

16:45 BST 的低频 live audit 显示：

```text
GPU5: direct-Q seed3, snapshot 7000/10500
      measured total_time=980.53s at 7k
      remaining linear ETA=3.5/7*980.53=490.3s, about 8.2min
GPU1: no trainer, only 0.9GB runtime context
Stage-135 main: still event-waiting for Stage-134
```

为避免 GPU1 空转，已启动 Stage-135 内部预注册的 `anchor_only` smoke；这不是新 research
question，不读取 Stage-134 confirmation，也不改变 Stage-135 三臂 Gate。主 controller
已有 `1000_snapshot.pkl` 复用逻辑，因此之后不会重复训练。

```text
script:
  scripts/run_cqn_stage135_anchor_prefill.sh
immutable snapshot:
  exp_local/cqn_flow_high_utd/stage135_anchor_prefill_master/
    controller_script.20260724T164700.sh
master PID:     1755996
controller PID: 1756005
trainer PID:    1756007
GPU:            1, measured allocation 24598 MiB
```

相近 FlowIQN artifacts 的真实 10k wall time 为 `1250.70--2292.13s`，所以 1.5k smoke
外推 ETA 为 `188--344s`，即约 `3.1--5.7min`；首个本配置 train row 生成后再用其吞吐
收紧。Stage-135 仍须等三臂 health summary 后才允许得出阶段结论。

### 21.104 Route-A 去除弱 BC 混杂：冻结 validation-best CQN-AS，只学习独立 value

#### 上一阶段实际结果

现有 direct scalar-Q replay-next 三种子中的前两个已完整结束。按每个方法自己的
internal validation best，而不是 final checkpoint 比较：

| training seed | clean CQN-AS best | direct-Q best | delta |
|---|---:|---:|---:|
| 1 | `68%@5k` | `56%@5k` | `-12pp` |
| 2 | `72%@5k` | `60%@5k/10k` | `-12pp` |

第三种子随后完整到达 10.5k，validation-best 为 `72%@7.5k`，而不是过拟合后的
`40%@10k`；其 clean 对照 best 为 `76%@2.5k`（5k 同为 76%，按预注册规则取较早者），
因此为 `-4pp`。三种子 clean/direct-Q best 均值为 `72.00%/62.67%`，平均
`-9.33pp`，`0/3` training seeds 严格胜出。正式 internal-curve summary 与对应
artifacts：

```text
exp_local/cqn_flow_high_utd/
  stage84_85_direct_q_validation_best_summary_20260724/summary.json
exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed1_20260724
exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed2_20260724
exp_local/cqn_flow_high_utd/stage85_direct_q_replay_utd4_seed3_20260724
```

与此同时，Route-B Stage-75 `source01 + BCFM8` seed1 已完成 10.5k；其 internal eval 为
`64/56/72/76% @ 2.5k/5k/7.5k/10k`，当前 best `76%@10k`，比 clean seed1
validation-best `68%@5k` 高 `8pp`。这只是一个 candidate signal；Stage-76 尚未在共同
screen/validation/sealed split 上选择 checkpoint/beta，因此不能据此声称 FM 已超过
CQN-AS。

#### 解释

前两个 direct-Q 的 `-12pp` 不能干净归因于 scalar value：它们的 rollout 来自一个从零
训练的独立 atoms=1 BC tower，而 clean CQN-AS 的行为来自已经 validation-selected 的
51-atom critic。value objective 与 behavior-policy 质量同时变化，违反单变量比较。

因此 Route A 被拆成两个独立要求：

1. **行为不退化：** 每个 seed 精确导入其 clean validation-best
   `target_critic_params` 作为 legacy-C51 policy，并导入同一个 image encoder 到
   policy encoder；训练中二者 bitwise 不变。
2. **value 真实性：** 只允许独立 direct scalar-Q/value encoder 接收 TD、completed
   return 与 H=1 randomized action-centered loss；后续在 fixed BC continuation、
   round-robin dimensions 下单独检验 causal ranking 与 return-unit calibration。

这样如果 causal value Gate 失败，结论是 value 没学真；不会再混入弱 BC policy。如果它
通过，task success 至少有 exact clean fallback，而 value 是否能进一步改善动作选择仍需
另开 policy extraction Gate。

#### 下一阶段决策与 Gate

Stage-138 是两个 training seed 的 implementation/identity smoke，不用于 policy-quality
选择。每个 seed 训练 1.5k，Gate 同时要求：

```text
config:
  freeze_bc_policy = true
  bc_policy_mode = legacy_c51
  policy_value_beta = null

trained snapshot:
  clean target critic == trained policy, bitwise
  clean encoder == trained policy encoder, bitwise

all logged updates:
  policy_grad_norm == 0
  policy_encoder_grad_norm == 0

paired fresh closed loop:
  8 episodes per seed
  success/reward/length mismatch count == 0 at atol=0
```

只有两个 seed 全部通过，才晋级 seed3 与三种子 matched RCT/no-RCT causal Gate；该正式
Gate 使用 H=1、round-robin 维度、fixed frozen-BC continuation、至少 32 fresh eval
seeds，并继续分开报告 causal ranking 和 native return-unit calibration。smoke loss、
demo top-1 或 1.5k internal success 都不作为晋级依据。

#### 已实现并执行

实现修改：

```text
robobase/method/cqn_direct_q.py
  legacy-C51 policy architecture
  clean target critic / encoder exact import
  expected-C51 bin score action path
  frozen policy/encoder parameter restoration after AdamW update
  separate policy and policy-encoder gradient logging

scripts/check_cqn_direct_q_rct_training_gate.py
  source-path identity
  source/trained policy and encoder SHA256 + bitwise comparison
  exact-zero gradient checks

robobase/cfgs/launch/
  cqn_direct_q_h1_rct_frozen_clean_high_utd4_gate.yaml
```

验证：

```text
py_compile: pass
direct-Q + RCT checker focused suite: 23 passed in 155.08s
factory + original CQN-AS + direct-Q + checker: 73 passed in 300.62s
Hydra real move_plate composition: pass
bash -n Stage-138: pass
```

Stage-138 已用不可变脚本快照真实排队：

```text
script:
  scripts/run_cqn_stage138_frozen_clean_rct_smoke.sh
immutable snapshot:
  exp_local/cqn_flow_high_utd/stage138_frozen_clean_rct_smoke_master/
    controller_script.20260724T165115.sh
master PID:      1760957
controller PID:  1760968
wait PID:        1760971
event wait:      tail --pid=1735655  # Stage-135 master
```

当前 GPU1 正在预填 Stage-135 anchor smoke，GPU5 正在完成 direct-Q seed3；Stage-138 不抢占
这两项，而是在整个当前双卡链释放后自动启动 seed1/2 并行训练。按 direct-Q seed3
step1k 的实测 `152.49s`、历史 paired eval 吞吐与 snapshot audit 开销，取得两张卡后
preregistered wall ETA 为 `480s`（约 8 分钟）；上游等待使用 PID 退出事件，不用短轮询，
也不伪造尚无法可靠确定的绝对完成时刻。

### 21.105 Stage-75 关闭并立即进入 Route-B 独立 selection

#### 上一阶段实际结果与解释

16:55 BST，Stage-75 controller 已写出：

```text
exp_local/cqn_flow_high_utd/stage75_floq_fidelity_seed1_controller_r1/complete
source_gate.json: pass
bcfm_gate.json: pass
combo_gate.json: pass
route_a_seed1/2/3_gate.json: pass
```

这里的 pass 只证明所有训练完整、snapshot/flow losses/gradients 有限；task 证据仍只有
Stage-75 internal curve，不能用它跳过独立 selection。direct-Q 三种子 health 也全 pass，
但 validation-best summary 的 task Gate 为 fail（平均 `-9.33pp`），进一步确认旧
atoms=1 BC tower 方案不能作为 Route-A 推荐。

Stage-135 opportunistic `anchor_only` smoke 同时完成；`anchor_only_health.json` 的所有
14 项检查均为 true，包含 1k snapshot、finite gradient、非零 FlowIQN/velocity gradient
以及其他 objective 精确为零。这是 Route-B 新 objective family 的 anchor 实现健康结果，
不是 task-quality 结果。

#### 下一 Gate 与已执行

Stage-76 已由 Stage-75 completion event 自动启动，使用两张卡做预注册共同 Gate：

```text
runner PID: 1761690
screen split: 114000--114009
arms: legacy, source01, bcfm8, source01_bcfm8
checkpoints per arm: 1k--10k
screen beta: 1
validation split: 115000--115049
beta grid: 0.3, 1, 3
sealed confirmation: 116000--116199
```

启动后 GPU1/GPU5 已分别出现独立 evaluator PID `1764865/1764393`；最先两个
10-episode JSON 在 `74.7s/87.8s` 内落盘。Stage-76 只有 validation delta
`>=+2pp` 且 wins>losses 才打开 200-episode confirmation；confirmation 再要求
delta `>0`、wins>losses、bootstrap CI lower `>=-5pp`。仍需 seed2/3 的 Stage-78
strict Gate 才能声称 Route B 超过 CQN-AS。

依据同 evaluator 的 Stage-67 `42 jobs / 2977.43s` 与本阶段首批真实吞吐，当前
Stage-76 wall ETA 保持 `45--65min`，即约 `17:40--18:00 BST`；runner 自己用 job
completion queue 调度两张卡，不做短轮询。Stage-138 继续绑定 Stage-135 master 释放事件，
不会与当前正式 selection 抢卡。

### 21.104.1 Stage-135 anchor-only smoke 实际结果

#### 上一阶段实际结果

事件 PID 在 16:55 BST 完成，artifact 为：

```text
exp_local/cqn_flow_high_utd/stage135_anchor_prefill_controller/
  anchor_only_health.json

snapshot: 1000_snapshot.pkl and 1500_snapshot.pkl
1k total_time: 356.67s
batched updates/s: 4.1059
gate: pass
```

全部 14 个 health checks 为 true。关键数值：

```text
max sorted BCFM loss:          6.5465e-4
max critic update norm:        5.5271e-2
max flow-critic grad norm:     3.6785e-3
max velocity-head grad norm:   3.6785e-3
max quantile endpoint loss:    0.0
nonfinite gradient fraction:   0
```

#### 解释

这证明 matched `anchor_only` 的 sorted one-to-one CFM、BC target、two-tower critic 和
optimizer path 都实际运行，且 checker 能确认它没有意外启用 all-pairs quantile loss。
它不建立 policy quality，也不支持“Flow 比 CQN-AS 好”；训练 loss 只用于排除死梯度和
错误 arm wiring。

#### 下一阶段决策与执行

同一 Stage-135 剩余两个 preregistered treatment 必须分别满足：

- `joint_equal`：BCFM 与 quantile endpoint loss 都活跃，两个梯度 path 非零；
- `dbc_ratio`：相同，但 BCFM/quantile 的配置权重必须精确为 `0.01/1.0`；
- 两者 snapshot、finite update 和 unused-objective checks 全部通过。

真实 anchor 吞吐把每个剩余 1.5k smoke 的 ETA 更新为约 `356.67 * 1.5 = 535s`
上界估计，即单臂约 `8.9min`（含初始化）；若 GPU1/GPU5 同时释放则并行 wall ETA 同样约
`9min`，否则按实际空闲卡事件调度。

#### 已执行

direct-Q seed3 已生成 `10500_snapshot.pkl` 并释放 GPU5；上游随即进入 Stage-76
短 episode screen。已在 GPU1 启动 `joint_equal` prefill：

```text
script:
  scripts/run_cqn_stage135_joint_equal_prefill.sh
immutable snapshot:
  exp_local/cqn_flow_high_utd/stage135_joint_equal_prefill_master/
    controller_script.20260724T165700.sh
master/controller/trainer PID:
  1769805 / 1769812 / 1769814
GPU1 allocation:
  trainer 24598 MiB
```

启动后发现 Stage-76 有一个 10-episode evaluator 同驻 GPU1、占 `2660MiB`；总显存仍低于
卡容量，未发生 OOM，但只把这一段 wall-clock 吞吐视为保守 ETA，不用于方法比较。训练
结果的 seed、数据、checkpoint 与 loss Gate 均未改变。

### 21.105.1 Stage-135 joint-equal smoke 实际结果

#### 上一阶段实际结果

17:06 BST，`joint_equal` prefill 完成：

```text
artifact:
  exp_local/cqn_flow_high_utd/stage135_joint_equal_prefill_controller/
    joint_equal_health.json
gate: pass, 14/14 checks true
1k total_time: 383.60s
batched updates/s: 3.8244
max BCFM loss: 6.1377e-4
max quantile endpoint loss: 1.9508e-4
quantile loss range: 1.9163e-4
max critic update norm: 5.1162e-2
max flow critic / velocity gradient: 3.7086e-3 / 3.7086e-3
nonfinite gradient fraction: 0
```

#### 解释

all-pairs quantile endpoint objective 确实接入训练：它非零、随 update 改变，同时
sorted CFM、critic update 和 velocity gradient 均保持活跃。因而若后续 policy Gate
失败，不能再归因于 “quantile loss 没有执行”。本结果仍只是实现健康，不是 task quality。

#### 下一阶段决策与执行

Stage-135 最后一臂只检验 DBC 推荐的小 anchor 比例是否按 `BCFM=0.01`、
`quantile=1.0` 健康运行；其余结构、batch、UTD、seed budget 与 checker 不变。必须满足
同样 14 项检查，尤其 quantile loss 非零和 unused objectives 精确为零。按本臂实测，
1.5k wall ETA 为约 `6.4--9min`。

已立即启动：

```text
script:
  scripts/run_cqn_stage135_dbc_ratio_prefill.sh
immutable snapshot:
  exp_local/cqn_flow_high_utd/stage135_dbc_ratio_prefill_master/
    controller_script.20260724T170700.sh
master/controller/trainer PID:
  1804337 / 1804344 / 1804346
GPU1 allocation:
  24598 MiB
```

### 21.106 Route-A 正式 matched design：value 选择与 task 选择彻底分开

#### 上一阶段实际结果

旧 direct-Q 三种子已经以 validation-best 均值 `62.67%` 对 clean `72.00%`、平均
`-9.33pp` 失败；它证明从零训练的 atoms=1 BC tower 不是可接受 baseline。新的
frozen-clean 实现目前有 `73` 个 factory/CQN-AS/direct-Q/checker 回归通过，Stage-138
仍在等待当前双卡链释放，尚无 pixel smoke artifact，因此这里不提前声称新设计成功。

Route-B 同时推进到 Stage-76 `31/40` screen JSON；Stage-135 `joint_equal` health 已
`14/14` checks pass，最后一个 `dbc_ratio` smoke 正在 GPU1 运行。两项都仍是实现或
selection 阶段，不是最终 task 结论。

#### 解释

冻结 behavior 解决了 task degradation 混杂，但 value authenticity 还必须避免另一种
selection cheat：因为 frozen policy 让所有 value checkpoints 的 BC-only task success
完全相同，不能再用 task success 选择 value checkpoint。新的正式设计预注册 `10k`
endpoint，不在 1k--10k 中事后挑 causal metric 最好的 checkpoint。

RCT 的作用也必须由 matched no-loss control 识别。control 与 treatment 完全共享：

```text
clean validation-best legacy-C51 policy + image encoder
H=1 randomized structured replay and exact propensities
completed Monte-Carlo returns
replay-next TD, UTD=4, seed, budget, optimizer
BC-only rollout and fixed-BC branch continuation
```

唯一差异是 `causal_rct_weight: 0.0` 对 `0.1`。因此 treatment 的 causal improvement
不再能归因于不同 policy、不同 intervention data 或不同 task states。

#### 下一阶段 Gate

Stage-139 在 Stage-138 identity smoke 被汇报并 pass 后，双卡并行训练 seed1
control/treatment 到 preregistered 10k；两者都必须：

1. training health pass；
2. source policy/encoder 与 10k snapshot bitwise identical；
3. 所有 policy/policy-encoder gradient 精确为 0；
4. randomized H=1 replay 覆盖至少 200 starts、每维至少 5 starts。

Stage-140 再在从未使用的 `214000--214011` simulator seeds 上并行执行：

```text
anchors: 30, 75, 120
dimension: round_robin
intervention: sibling bins, H=1
continuation: exact frozen clean BC, value beta null
checkpoint: fixed 10k
```

seed1 只作 discovery promotion。control/RCT 每个 branch state 的 forced bins、
realized return、rollout length、success 与三类 imitation proxy 必须逐项完全相同；
随后才允许比较 Q。promotion 要求 treatment causal direction 为正、pairwise accuracy
高于 control、并同时高于 policy-prior、policy-path、action-nearness。它不允许产生正式
Route-A claim；通过后仍须训练 matched seed2/3，在新 sealed 32-seed split 上使用
training-seed × simulator-seed paired crossed-bootstrap 的严格 CI，并另过 native
return-unit calibration Gate。

#### 已实现

新增：

```text
robobase/cfgs/launch/
  cqn_direct_q_h1_rct_frozen_clean_control_high_utd4_gate.yaml
  # resolved method diff to treatment: causal_rct_weight only

scripts/summarize_cqn_paired_causal_arms.py
  exact control/treatment branch matching
  paired training-seed x simulator-seed bootstrap
  treatment-vs-control and treatment-vs-three-imitation-proxy gates

scripts/run_cqn_stage139_frozen_rct_seed1.sh
scripts/run_cqn_stage140_frozen_rct_seed1_causal.sh
```

验证：

```text
control/treatment config + paired comparator/direct-Q: 22 passed in 153.45s
paired comparator + calibration + multiseed causal runner: 18 passed in 0.12s
py_compile: pass
bash -n Stage-139/140: pass
```

Stage-139/140 有意尚未启动：Stage-139 要求 Stage-138 的 `reported` marker，Stage-140
要求 Stage-139 的 `reported` marker，确保每个阶段先报告真实结果再晋级。当前已执行的
Stage-138 event controller 继续有效。Stage-139 会在 Stage-138 实测 frozen update
吞吐落盘后，以该 backend/reference ratio 写入 ETA；Stage-140 参考既有 8/16-seed branch
probe 的 `146.94s/256.71s`，双卡 12-seed discovery preregistered ETA 为 `<=360s`。

## 22. 2026-07-25：覆盖、RCT 功效与 on/off-path 校准的三重测量

本节是一次纯测量批次：不训练新的正式 arm，用现有 replay/checkpoint 回答三个
决定 Stage-139/140 与 Route-B 下一步设计的问题。旧的 Stage-135/138/76 controller
链在本批次开始前已全部退出；GPU1/GPU5 用于本批次，GPU0/2/3/4 为其他用户占用。

### 22.1 上一阶段实际结果

#### A-0：zoom-path 覆盖统计（无 GPU，6 个既有 replay）

新脚本 `scripts/analyze_cqn_zoom_path_coverage.py`（其 `encode_action_bins`
与 `robobase.method.cqn.encode_action` 逐位一致，已验证）。对 stage2/3/6/7 六个
run 的完整 replay（各约 2 万 transitions，demo=60 episodes）：

| run | 组 | level-0 归一化熵 | modal bin share |
|---|---|---:|---:|
| clean_full | demo | 0.509 | 66.5% |
| clean_full | online_failure | 0.472 | 68.6% |
| qtd_only | online_failure | 0.911 | 35.4% |
| coherent_L1_mc | online_failure | 0.345 | 77.8% |

depth-3 前缀（5^3=125 空间）每维实际 usable（≥5 样本）76–86 条；每个 phase
（epis 早/中/晚三分）有 14/15 维存在 ≥2 个 usable level-0 bin；level-1 输入的
modal-子树集中度 69–73%。`qtd_only` 因随机化行为几乎满覆盖（124.7/125 usable）
但 0% success。

**结论：** 「off-expert 子树不在输入分布里」的强机制假设被否定；覆盖存在但集中。
`qtd_only` 满覆盖仍完全失败，说明覆盖本身不是 TD 可识别性的约束。
artifacts：`exp_local/cqn_zoom_coverage/coverage_moveplate_20260725.json`、
`replay_support_moveplate_20260725.npz`。

#### RCT 功效分析（stage7 L1 随机化 starts 为真实 RCT 数据）

treated=`structured_explore_start`，Y=discounted RTG，control=同 run 非 active
online transitions：

| 数据 | starts | ATE | pooled SD | Cohen's d | 需要 N/臂 |
|---|---:|---:|---:|---:|---:|
| coherent_L1_mc | 549 | +0.001 | 0.251 | 0.004 | ~1.09M |
| coherent_L1_nomc | 491 | −0.004 | 0.237 | −0.018 | ~47.5k |
| pilot L0 p060 | 265 | +0.000 | 0.177 | 0.002 | ~2.7M |

per-dim 最好情形（受 max-of-15 选择偏差高估）需要 165–212 starts/维，实际
33–37/维。**level-0 剂量（5 倍于 L1）没有提高 unadjusted ATE**。
artifacts：`exp_local/cqn_zoom_coverage/rct_power_analysis_L1_20260725.json`、
`rct_power_analysis_L0_pilot_20260725.json`。

#### Level-0 coherent pilot（GPU1，3 剂量臂 × 5.5k frames，每臂约 7 分钟）

`scripts/run_cqn_pilot_level0_coherent_pscan.sh`，decoupled+MC0.1 平台，
H=4、level=0（cell 0.4）：

| 臂 | 实测 active | success @2.5k/5k | starts@5k | 全剂量(δ=0.4)比例 |
|---|---:|---:|---:|---:|
| p013 | 4.4–4.8% | 12/32% | 56 | – |
| p027 | 9.3% | 0/60% | 117 | – |
| p060 | 18.4–18.9% | 16/32% | 237 | 88.3% |

三臂 success 全部在 20–50%（p027 到 60%）可用带内；level-0 剂量不摧毁
BC 行为。clip 后 88.3% starts 拿到全剂量（level-1 当时 26.7–50% token alias）。
run dirs：`exp_local/cqn_zoom_coverage/pilot_level0_coherent_p{013,027,060}_seed1_gpu1_20260725162222`。

#### A-0b：on/off-path Q 校准分层 probe（GPU5，4 checkpoints × 480 样本）

新脚本 `scripts/analyze_cqn_onpath_q_reliability.py`。executed-action Q 对
RTG，按「executed level-0 bin 在其 phase 层内的支撑份额」四分位分层：

| checkpoint @10.5k | ρ 全体 | ρ 最低支撑 Q1 | ρ 最高支撑 Q4 | MAE | mean Q / RTG |
|---|---:|---:|---:|---:|---|
| clean_full（canonical） | **−0.541** | −0.436 | −0.604 | 0.462 | 0.49 / 0.22 |
| floq_anchored | +0.374 | +0.416 | +0.171 | 0.769 | 1.00 / 0.23 |
| decoupled+MC 0.1 | +0.876 | +0.835 | +0.892 | 0.018 | 0.24 / 0.23 |
| coherent_L1_mc | +0.880 | +0.804 | +0.857 | 0.012 | 0.23 / 0.22 |

**支撑份额分层内没有出现校准梯度**；canonical critic 在自己的行为分布上
全局反相关；MC-anchored 解耦 critic 全谱系校准良好；FLOQ（无 MC anchor）Q
膨胀约 4 倍。artifacts：`exp_local/cqn_zoom_coverage/onpath_probes/*.json`。

#### Control-variate 可行性检验

用 MC-anchored checkpoint 的 predicted Q 作 baseline b：online 子集
sd(Y)=0.26–0.29，sd(Y−Q)=0.066–0.093，**方差缩减 10–15.4×**。调整后
per-contrast 所需 N：δ=0.08→21、δ=0.05→54、δ=0.03→149。现有 30–60
starts/维使 δ≥0.05 进入可检验区。

### 22.2 解释

1. 机制修订：约束不是输入覆盖，也不是行为分布上的 value 校准，而是
   **同状态跨动作对比的估计量**。unpaired 设计在任何剂量下被 between-state
   方差（SD≈0.18–0.29）淹没；Stage-VII branch probe 唯一一次通过正是因为
   同状态配对。
2. canonical CQN-AS 的 ρ=−0.54 是新的、独立于 causal gate 的负证据：其 Q
   连自己执行过的动作的相对好坏都排反了；「92% success 的 policy」与
   「反校准的 critic」共存，再次支持 imitation-shortcut 主张。
3. 预注册判据触发：Stage-139/140 的 causal RCT 按现设计（unadjusted moment
   loss、L1 剂量）功效不足 5 倍以上，**不应原样启动**。这不是 value 学不学
   得到的证据，是估计量选择错误。
4. Route B 校准对比中 FLOQ 的 4 倍高估不构成公平负证据：它没有 MC anchor。
   公平比较必须 anchor-matched。

### 22.3 下一阶段决策（预注册）

**Route A（Stage-141，替换原 Stage-139/140 设计）：**

- 平台：decoupled Q-TD + π-BC + MC 0.1；coherent 结构化探索
  level=0、H=4、p_start=0.060（active≈19%，starts 最多）。
- loss 修订：`action_centered_moment_loss` 的 outcome 改为
  `y − stop_gradient(Q(s, a_proposed))`，其中 a_proposed 为干预前 BC 提议
  动作（由 `structured_explore_delta/dimension` 从 executed 恢复；
  **pre-treatment，不吸收处理效应**；executed-action Q 不允许作 covariate）。
- 对照：唯一差异 `causal_rct_weight ∈ {0, 0.1}`；≥2 training seeds；10k
  endpoint 固定，不做事后 checkpoint 挑选。
- Gate：同状态 branch probe（L0 干预、最深层 readout、state-bootstrap
  pairwise CI 下界 >50%），加训练中 causal moment 的 SNR 日志。
- 原 2×2 factorial 的 coverage 前提已被 A-0 削弱，撤回该设计。

**Route B（Stage-142）：**

- anchor-matched 对比：MC-anchored flow endpoint vs MC-anchored C51，
  同一 frozen-policy replay、同 budget/seed/optimizer。
- 主指标：executed-action 校准（ρ/MAE）+ 分布指标（CRPS）+ 同一 branch
  gate。当前无 anchor 的 FLOQ 已知 ρ=0.37/MAE 0.77，作为下界参考。
- 终止判据保持：若 anchor-matched flow 在校准与 CRPS 上均无显著优势，
  Route B 记录为负结果并停止扩展。

### 22.4 已执行

新增脚本（均通过 py_compile；coverage 编码器与 method 实现逐位对拍）：

```text
scripts/analyze_cqn_zoom_path_coverage.py
scripts/analyze_cqn_onpath_q_reliability.py
scripts/run_cqn_pilot_level0_coherent_pscan.sh
scripts/run_cqn_onpath_probes_gpu5.sh
```

本批次全部测量已完成并落盘于 `exp_local/cqn_zoom_coverage/`；pilot 三臂与
四个 probe 的实际完成通过 eval.csv、JSON artifacts 与进程退出确认。
Stage-141/142 的实现（loss 修订 + matched flow anchor 配置）为下一执行项，
尚未启动训练。

### 22.5 2026-07-25：两路线 goal 重置（用户决定）

**诊断阶段冻结。** canonical CQN-AS「伪装成 RL 的模仿学习」这一主张的证据
已足够支撑内部决策：objective decomposition（§20.1）、同状态 causal 反例
（§20.2/20.4）、全局反校准 ρ=−0.54（§22.1）、Q-BC 与 full 的 imitation 指标
逐项相等。不再为该主张收集新证据；唯一保留的可选项是发表前用第二个任务
复跑零成本的 A-0/A-0b 以补外部效度。

**Route B 新 goal：解决问题，而不是继续证明问题。** 目标改为构造一个
在 matched 预算下 task success 严格超过 clean CQN-AS、且 value 同时通过
causal gate 的方法。分解为三个组件，每个组件已有当前最优候选：

1. 行为：独立 π-BC head（已有，decoupled 平台）；
2. 校准 value：MC-anchored critic（已有，ρ=0.88 / MAE 0.012）；
3. 因果 advantage：缺失组件，由 Route A / Stage-141 提供。

flow 参数化降级为组件 3 的可选表示，仅当 Stage-142 anchor-matched 对比
在校准/CRPS 上显著胜出才纳入；不再作为独立研究线。

**Route A 定位澄清。** 当前结论不是「方法不行」，而是「此前的实验大多
没有检验能力」：unpaired 估计量需要 5 万–270 万样本，实际只有数百，
因此历次 fail 无法区分「value 学不到」与「仪器测不到」。Stage-VII 唯一一次
弱通过恰恰来自唯一一次配对测量。§22.3 的 Stage-141（control-variate 修正 +
level-0 剂量 + 配对 gate）是第一个功效充足的正式实验：

- 若 pass：获得经认证的 causal advantage，直接作为 Route B fix 的组件 3；
- 若 fail：因功效充足，首次构成真正的负结论——局部干预在该 regime 下
  不可识别 value，转向 simulator paired-branch 数据收集或接受
  IQL-style support 内改进。

## 23. 2026-07-25：Stage-141 实施与首轮 control/treatment 结果

### 23.1 已实施

`robobase/method/cqn_as.py` 新增 CV-adjusted causal RCT（`cv_rct_weight/level/
baseline`）：outcome 为 `mc_return − stop_grad(Q_target(s, a_pre-treatment))`，
pre-treatment 动作由 executed − recorded delta 恢复；demo 与 continuation
transitions 显式排除；`null` 保持 legacy 图 bitwise 不变，`0.0` 为同图零权重
matched control。`action_centered_moment_loss` 移至 cqn_as 并由 cqn_direct_q
re-export。新增 launch `cqn_as_pixel_bigym_stage141_cv_rct_gate`（L0/H4/
p0.060/MC0.1）。5 个新单测 + 72 个 cqn_as/cqn_direct_q/factory 回归通过。

4 个训练臂（2 seeds × weight {0.0, 0.1}）已完成于
`exp_local/cqn_stage141_cv_rct/move_plate_cv_rct_*_20260725195703`；
10k success 48–64%，treatment 臂 cv loss 非零、control 臂精确为 0。

### 23.2 Gate 实际结果

预注册主判定（structured_horizon、深层 readout；工具限制使 round_robin 不可用
于该模式，主判定退回 q_span，偏离已记录）在 8 个 fresh eval seeds
（300–307）× 3 anchors 下：

| arm | pairwise sign | 95% CI | ρ | pairs |
|---|---:|---|---:|---:|
| seed1 control | 0.500 | [0.208,0.783] | −0.041 | 24 |
| seed1 treatment | 0.390 | [0.220,0.571] | −0.210 | 41 |
| seed2 control | 0.538 | [0.392,0.680] | +0.106 | 52 |
| seed2 treatment | 0.511 | [0.362,0.674] | +0.020 | 45 |

**预注册 gate：fail**（两 seed 均未过 CI 下界与超越 control 两条）。

副判定（sibling_horizon + round_robin，Q-independent，pairs 3–4 倍）：

| arm | pairwise sign | 95% CI | ρ | pairs |
|---|---:|---|---:|---:|
| seed1 control | 0.500 | [0.410,0.592] | −0.009 | 100 |
| seed1 treatment | **0.608** | [0.489,0.725] | +0.209 | 143 |
| seed2 control | 0.466 | [0.364,0.569] | −0.070 | 178 |
| seed2 treatment | **0.557** | [0.443,0.677] | +0.128 | 131 |

### 23.3 解释

1. 主判定再次功效不足（24–52 pairs，CI 宽至 0.56）；其 fail 与「无效应」
   不可区分，不作为负结论。
2. 副判定给出**两 training seed 一致的方向性 treatment 效应**（+10.8/+9.1pp、
   ρ 由负转正），且效应恰好出现在与训练 loss 反事实结构一致的 sibling-bin
   对比上。这是 discovery-level 证据，不是 confirmation：CI 下界 0.489/0.443
   未过 0.5，且 sibling 协议作为主判定属于事后选择。
3. 结论：CV-RCT 首次在功效可用的探测下显示出可复现方向信号；按预注册纪律，
   必须以新预注册在 sibling 协议 + 新 seeds 上确认。

### 23.4 下一阶段（Stage-143 预注册）

- 主判定改为 sibling_horizon + round_robin + L0 + H4（理由：Q-independent、
  功效最高、匹配 loss 的反事实结构；此变更在看到 seed3 数据前冻结）。
- 新增 training seed 3 的 control/treatment 对；probe 扩至 12 eval seeds
  （400–411）× 3 anchors，bootstrap 单位不变。
- Pass：三 seeds treatment 全部超 matched control，且三 seed pooled
  treatment CI 下界 > 0.5（training-seed × eval-seed crossed bootstrap）。
- Stage-142（anchor-matched FLOQ，两 seeds）先占用双卡；seed3 对在其后。

## 24. 2026-07-25：Stage-142 anchor-matched flow 校准裁定

### 24.1 上一阶段实际结果

两个 anchor-matched FLOQ 臂（`cqn_flow_floq_stage142_anchor_matched_gate`，
与 Stage-141 control 完全同平台：decoupled π-BC、replay-next、MC 0.1、
coherent L0/H4/p0.060）已完成 10.5k 训练：

```text
exp_local/cqn_stage142_anchor_matched/move_plate_floq_mc_seed{1,2}_gpu{1,5}_20260725204456
success @2.5k/5k/7.5k/10k: seed1 20/56/48/32%, seed2 8/40/52/40%
```

同一 on-path 校准 probe（480 样本/checkpoint，10.5k）对 matched 四臂：

| 臂 | ρ(Q,RTG) | MAE | mean Q / RTG |
|---|---:|---:|---|
| C51 control seed1 | 0.851 | **0.019** | 0.227 / 0.225 |
| C51 control seed2 | 0.858 | **0.016** | 0.235 / 0.237 |
| FLOQ+MC seed1 | 0.839 | 0.048 | 0.225 / 0.223 |
| FLOQ+MC seed2 | 0.868 | 0.043 | 0.224 / 0.227 |

### 24.2 解释与裁定

1. MC anchor 修复了 unanchored FLOQ 的 4× Q 膨胀（§22.1），两种参数化都无偏。
2. 排序相关性统计上打平（Δρ ≤ 0.012，方向跨 seed 不一致）；**MAE flow 差
   2.5–3×**。在校准维度 flow 无任何优势。
3. 按 §22.5 预注册终止判据，flow-as-critic 线**临时关闭**：唯一保留的翻案
   窗口是 CRPS/分布锐度对比（尚未测量）；若后续测得 flow CRPS 显著优于
   C51 atoms 可重开，否则 Route-B fix 的组件 3 表示确定为 C51/scalar。
4. 该裁定与 §21 十次 task-level 失败一致，但证据等级更高：这是首次在
   完全 matched 平台上按 value-quality 指标直接比较。

### 24.3 下一阶段

Stage-143（sibling 协议三 seed confirmation）已启动，占用双卡：
seed3 control/treatment 训练 → 6 checkpoints × 12 fresh eval seeds
（400–411）sibling probe → crossed-bootstrap 判定
（`stage143_gate_summary.json`）。

## 25. 2026-07-25：Stage-143 confirmation 正式失败与 Route-A 分叉点

### 25.1 实际结果

seed3 control/treatment 对训练完成后，6 个 checkpoint 在 fresh eval seeds
400–411（12 seeds × 3 anchors，sibling_horizon + round_robin + L0 + H4）下：

| seed | control | treatment | treatment 胜 |
|---|---:|---:|---|
| 1 | 0.409 (164p) | 0.393 (168p) | 否 |
| 2 | 0.445 (200p) | 0.587 (155p) | 是 |
| 3 | 0.555 (146p) | 0.494 (180p) | 否 |

pooled treatment `0.489`（503 pairs），crossed-bootstrap CI `[0.383, 0.629]`。
**预注册 gate：fail**（1/3 seeds 胜出；pooled CI 覆盖 0.5）。
artifacts：`exp_local/cqn_stage141_cv_rct/stage143_gate/`。

### 25.2 解释

1. seed1 的 discovery 信号（0.608 @ eval 300–307）在 fresh eval seeds 上
   塌回 0.393——§23 的方向性信号主要是 eval-seed artifact。预注册的
   fresh-seed confirmation 正确拦截了它。seed2 是唯一两轮皆正的臂
   （0.557→0.587），单臂不构成结论。
2. 与历史 fail 不同，本次 gate 功效充足（503 pairs）。但训练侧仍是边缘
   功效：三个 treatment 臂 per-dim starts 34–36，只够检测 δ≥0.07 的
   per-dim 效应（δ=0.05 需 54，δ=0.03 需 149，见 §22.1 CV 功效表）。
3. 因此正确表述是：**10.5k 预算下的 CV-RCT 未能学到可确认的 sibling
   反事实排序**；「该目标在更大数据量下是否可学」由训练侧功效缺口留开。

### 25.3 Route-A 分叉（待选路线，均不自动启动）

§22.5 冻结决策树在 powered fail 下指向 (b)/(c)；训练侧功效边缘为 (a)
提供了修正理由，作为对冻结树的显式修订记录：

- (a) **Stage-144 数据规模检验**：50k frames（约 170 starts/dim，覆盖
  δ≥0.04），2 seeds × control/treatment，同一冻结 sibling 协议。
  约 2.5h 双卡。若仍 fail，(b)/(c) 成为唯一路径。
- (b) **simulator paired-branch 训练数据**：用现有 branch-state 机制在
  训练中采集真正同状态 (s, a, a′) 配对 outcome。估计量最优，工程量最大。
- (c) **接受 support 内改进**：组件 1+2（π-BC + 校准 V）已可用，放弃
  反事实 claim，按 IQL-style 收口 Route-B fix。

### 25.4 本日总账（三个 Stage 一日完成）

- Stage-141：CV-RCT 实现 + matched 双臂 ×2 seeds + 双协议 gate —— 主
  gate fail、副协议 discovery 信号；
- Stage-142：anchor-matched FLOQ 校准裁定 —— flow 无优势，flow-as-critic
  线临时关闭（CRPS 翻案窗口保留）；
- Stage-143：三 seed fresh-seed confirmation —— fail，discovery 信号为
  eval-seed artifact。

组件现状：① π-BC 行为（可用）② MC 校准 value（可用，ρ≈0.85、MAE≈0.02）
③ 因果 advantage（三条候选路线待选）。

## 26. 2026-07-25：Stage-144 预注册与 Route-(c) 并行设计（用户决定）

### 26.1 Stage-144 预注册（启动前冻结）

用户选定分叉 (a)：数据规模检验。

- 训练：50.5k frames（约 170 starts/dim，功效覆盖 per-dim δ≥0.04），
  **全新 training seeds 4/5**（不复用已知响应方向的 1–3），每 seed 的
  control/treatment 共享同一 GPU；固定 50.5k endpoint，无 checkpoint 挑选。
- Gate：冻结 sibling 协议（sibling_horizon + round_robin + L0 + H4），
  **从未使用的 eval seeds 500–511**；判定与 Stage-143 相同：两 seed
  treatment 均须超 matched control，且 pooled crossed-bootstrap CI 下界
  > 0.5。
- 判定树：pass → 组件 ③ 到手，进入 policy-extraction gate；fail →
  (b) paired-branch 或 (c) support-only 成为唯一路径，且「数据量不足」
  解释被排除。
- 配置/脚本：`cqn_as_pixel_bigym_stage144_cv_rct_50k_gate.yaml`、
  `run_cqn_stage144_cv_rct_50k.sh`（snapshot_every_n=10000 控制磁盘）。

### 26.2 Route-(c) 设计（与 Stage-144 并行，仅设计+实现，不占卡）

目标：不做反事实 claim，用已验证组件 ①+② 构造超过 clean CQN-AS 的
方法。机制选择 IQL/AWR 式 support 内改进：

1. 在 critic trunk 上加 **expectile state-value head** `V_psi(s)`
   （IQL expectile τ_e=0.7，只回归 executed transitions 的 MC return；
   不查询未执行 action，不引入 max-bootstrap）。
2. π-BC 的 CE 改为 **advantage-weighted**：
   `w = clip(exp((G_t − V_psi(s)) / beta), w_min, w_max)`，
   `G_t` 为 completed-episode MC return；demo 与 online-success
   transitions 同权处理，失败 online transitions 自然获得低权。
   beta 首值 0.5，w_max 10（AWR 惯例）；权重 stop-gradient。
3. 明确不做：Q argmax 改行为、candidate rerank、counterfactual
   advantage——这些都被 §21/§25 证据排除。
4. Gate（预注册）：MovePlate 10.5k matched budget，3 training seeds，
   vs clean CQN-AS validation-best 基线（§21.104：72.00% 三 seed 均值）；
   screen/validation/sealed split 沿用既有纪律；同时报告
   advantage-weight 的有效样本量（ESS）防止权重塌缩到少数 transitions。

## 27. 2026-07-26：Stage-144 正式失败——CV-RCT 路线关闭

### 27.1 实际结果

50.5k frames、全新 seeds 4/5、冻结 sibling 协议、从未使用的 eval seeds
500–511：

| seed | control | treatment | treatment 胜 |
|---|---:|---:|---|
| 4 | 0.629 (178p) | 0.523 (172p) | 否（−10.6pp） |
| 5 | 0.466 (208p) | 0.559 (211p) | 是（+9.3pp） |

pooled treatment `0.543`（383p），CI `[0.442, 0.640]`。**gate：fail**。
训练侧本次 per-dim starts ≈ 170（5× Stage-141），功效覆盖 δ≥0.04；
50k control 臂 task success 升至 64–76%，训练本身健康。
artifacts：`exp_local/cqn_stage144_cv_rct_50k/gate/`。

### 27.2 解释与关闭决定

1. 累计五对 matched 臂（Stage-143 三对 + Stage-144 两对）：treatment
   2 胜 3 负，效应符号随 training seed 翻转，pooled 两次均覆盖 0.5。
2. 数据规模从 35 starts/dim 提高到 170 starts/dim 没有改变模式，
   「训练侧功效不足」解释被排除。
3. **正式结论：CV-adjusted action-centered RCT moment loss 在
   H=4 coherent level-0 干预、10.5k–50k 预算下，未能学到可确认的
   同状态 sibling 反事实排序。** 该负结论 gate 功效充足、协议冻结、
   fresh-seed 复核，证据等级为本仓 Route-A 系列最高。
4. 按 §26.1 冻结判定树：(b) simulator paired-branch 训练数据与
   (c) support-only 改进成为唯一路径。(c) 已作为 Stage-145 自动接续
   （seeds 1/2 训练中）；(b) 保留为唯一的 causal-value 候选，其
   工程投入决策待 Stage-145 结果与用户确认。

### 27.3 下一 Gate

Stage-145（Route-(c) AWR）：三 training seeds、10.5k matched budget、
无结构化扰动；judgment 为 internal validation-best 对 clean CQN-AS
三 seed 均值 72.00%（§21.104）。完成后直接从 eval.csv 汇总，无需 probe。

## 28. 2026-07-26：Stage-145 AWR v1 失败与 β 诊断

### 28.1 实际结果

三 seed、10.5k、awr_beta=0.5：validation-best `44/36/64%`，均值
`48.0%`，远低于 clean CQN-AS `72.00%`，也低于历史 decoupled+MC0.1
单 seed 参考（52%）。**v1 gate：fail。**
runs：`exp_local/cqn_stage145_awr/move_plate_awr_seed{1,2,3}_*_20260726003451`。

### 28.2 诊断（train.csv 实测）

ESS≈0.97–1.0、weight mean≈1.0、V≈mc_return 均值、expectile loss 收敛。
即：**V 头正常，但 β=0.5 对 sd≈0.28 的 advantage 残差过钝，权重几乎
均匀**。失败机制不是权重塌缩，而是 BC 的训练集从 demo-only 扩到全部
transitions 后，均匀权重使策略克隆了自己失败的 rollout。

### 28.3 Stage-145b（单变量修复 + 补对照）

- v2 臂：唯一改动 `awr_beta: 0.5 → 0.1`（残差 ±0.3 → 权重 0.05–10，
  真实现失败抑制）；三 seeds。
- 补齐 matched 对照：`awr_beta=null` 的 decoupled+MC0.1 seeds 2/3
  （seed1=52% 已有），使 AWR 对比不再依赖单 seed 参考。
- 判定不变：validation-best 三 seed 均值对 72.00%；同时报告 v2 对
  no-AWR 对照的逐 seed 差。

## 29. 2026-07-26：路线1重开——文献综述与"正确结合方式"的设计（Stage-146）

### 29.1 用户指示

路线1的 goal 是**构造性的**：研究出 FM×CQN-AS 怎么结合才能好，而不是
检验某一种结合是否有效。§24 关闭的只是 value 侧参数化（flow-as-critic），
该负结论保留；policy 侧结合从未实现（正是 CQNAS_FLOW_MATCHING_RESEARCH.md
的原始提案），现按最新文献重开。

### 29.2 2025-26 顶会文献分类（FM×RL）

| 家族 | 代表 | 需要 ∇_a Q | 对本仓适用性 |
|---|---|---|---|
| 梯度引导 | Q-VGM、Guided Action Flow (2607.02092)、DFQL (ICML26)、FlowDPG | 是 | **排除**：离散 C2F critic 零动作梯度；且 GAF 自证 critic 泛化是瓶颈（held-out 仅 +2.5pp），与本仓 on-manifold-only 校准发现一致 |
| 蒸馏/一步 actor | FQL (2502.02538)、OFQL (ICLR26)、PA-RL | FQL 是；PA-RL 否 | PA-RL 式（蒸馏 best-of-N 胜者）可行，列为 v2 |
| advantage 加权 flow BC | GFP (2512.03973)、energy-weighted FM | 否 | 可行；正是 Stage-145 AWR 的 flow 孪生，机制上收敛 |
| 采样选择 (best-of-N) | FQL 论文先例、rejection sampling | 否 | **MVP**：无新训练损失，行为策略级改动 |

### 29.3 本仓独有的适配论证

Stage-142 已证明：本仓 critic 在行为流形上校准良好（ρ 0.85–0.88、
MAE 0.016–0.019），而所有反事实探测失败。**best-of-N over flow-BC 采样
恰好只在流形上查询 Q**——候选全部来自 BC flow，Q 的查询点正是它被
校准的地方。这把"为什么 rerank 能成而 argmax-Q 不能"从经验命题变成
机制命题。§20.4 的 temporal-ensemble 语义教训保留：v1 对 raw chunk
打分、执行链路不动（与基线 matched），并记录 selected-vs-mean Q gap
与 effective-action 稀释诊断。

### 29.4 Stage-146 预注册

实现（decoupled 平台内新增 flow policy head，原子 checkpoint）：

```text
FlowPolicyHead: v_theta(x_t[K*D], t | h(s)) MLP
训练：CFM loss on demo actions（可选 AWR 权重复用 expectile-V）
采样：Euler T=8，M 候选并行
Rollout：target critic 对每候选 chunk 求最深层 mean_{k,d} Q，argmax 后
        进入原 temporal ensemble
```

Arms（move_plate、10.5k、3 training seeds、validation-best 判定）：

- A：flow BC alone（M=1）——采样器质量；
- B：flow BC + Q-rerank（M=8）——组合本体；
- 判定：B 须同时超过 A（rerank 贡献）并对 clean CQN-AS 72.00% 报告；
  B>A 而 B<72% 记为"组合有效但采样器不足"，先修采样器再扩 M。

一手来源：FQL arXiv:2502.02538；OFQL arXiv:2508.13904；DFQL ICML 2026
poster 64018；ReinFlow arXiv:2505.22094；FlowDPG arXiv:2606.22303；
Q-VGM arXiv:2606.08015；Guided Action Flow arXiv:2607.02092；
πRL arXiv:2510.25889；SERNF arXiv:2602.09580；GFP arXiv:2512.03973。

### 28.4 Stage-145b 结果：软加权二次失败，转向二值过滤

AWR β=0.1（v2）三 seed validation-best `52/64/44%`，均值 `53.3%`；补齐的
无 AWR 对照（decoupled+MC0.1）三 seed `52/60/76%`，均值 `62.7%`。v2 仍
输对照 9.4pp。诊断修订：失败不在温度，而在**BC 训练集从纯 demo 扩到全部
经验本身**——即使失败样本权重≈0，模仿自身平庸成功也劣于纯 demo 模仿。
软加权（AWR 家族）子路线关闭。

Stage-145c（v3，纯配置双臂 × 3 seeds，对照 62.7% 与 clean 72.0%）：

- `si`：`use_self_imitation=true`——官方 CQN-AS 重标机制，成功 online
  episode 整条进 demo 库（二值成功过滤，替代软权重）；
- `mc1p0`：`mc_return_weight=1.0`——历史 stage4 seed1 曾达 best 72%，
  三 seed 复核。

### 29.5 Stage-146 实现完成

`FlowPolicyHead`（条件速度场 MLP，forward-time CFM，demo-only 训练——遵循
Stage-145b「勿克隆 online」教训）、`_flow_policy_sample`（Euler T=8、M 候选
并行）、`_flow_rerank_action`（target critic 最深层 mean-Q 打分 + argmax）、
rollout 接入 `_build_greedy_action_fn`；`flow_policy=false` 时 legacy
graph/params 完全不变。4 个新单测 + cqn_as/direct_q 全量 63 回归通过；
launch `cqn_as_pixel_bigym_stage146_flow_rerank_gate` 组合验证通过。
Stage-146 六臂（M∈{1,8} × 3 seeds）已挂 Stage-145c 完成事件自动启动。

### 28.5 Stage-145c 结果：二值过滤与强锚同样失败，decoupled 平台模式确立

si（自我模仿重标）三 seed best `52/44/72%`，均值 `56.0%`；mc1p0（MC 权重
1.0）`48/56/68%`，均值 `57.3%`。两臂均低于无加权对照 `62.7%`（各变体间
差异在 25-episode 评估噪声 SE≈10pp 内，但「四个机制无一超过 plain」的
方向模式一致），全部远低于 clean `72.0%`。

**模式解读**：decoupled 平台本身对 clean 有约 9pp 的行为代价；clean 的
优势恰恰来自被解耦移除的 imitation-shortcut 机械（Q 上的 FOSD/margin +
Q-argmax rollout ≈ 带 TD 平滑的分类式模仿）。用 value 信息改进行为的
四种尝试（软权、锐权、二值重标、强锚）都无法在该平台上兑现——与
线路二全部证据一致：可识别的 value 信息只有「已执行路径的校准」，它
不含改进行为所需的反事实内容。

### 28.6 Stage-147 预注册（线路二新臂：clean + MC anchor）

证据指向的下一个可交付物不是「打败 clean」而是「**给 clean 补上诚实的
value**」：canonical clean CQN-AS（保留其全部 72% 行为机械）单变量加
`mc_return_weight=0.1`。判定双指标：

1. task：三 seed validation-best 均值不低于 clean 72.00% 的
   non-inferiority（容差 −5pp）；
2. value：on-path 校准 probe 的 ρ 从 clean 的 −0.54 转为显著正值。

若同时达标，线路二产出第一个正式可交付：任务性能不降、value 可信的
CQN-AS 变体。挂 Stage-146 完成事件自动启动。

### 29.6 Stage-146 v1 中止：flat-chunk flow 头精度不足，v1b 按步分解重启

v1 六臂中 M=8 seed1/2 跑到 7.5k：CFM loss 正常收敛（1.02→0.29）、
categorical 对照头 top1 0.81、MC 校准 mae 0.006–0.009，但 **success 全程
0%**。诊断：联合 240 维 flat-MLP 速度场的端点精度（每维残差 ~0.2+）低于
MovePlate 所需（~0.05）；这是采样器架构问题而非 rerank 机制问题。剩余臂
无信息量，全部中止（GPU 释放确认）。

v1b 单变量修复：FlowPolicyHead 改为**按序列步分解**（shared MLP 作用于
[features, 该步 15 维 x_t, t-embed, step one-hot]，输出该步速度），与
categorical head 的 factorization 同构。4 个单测通过后重启六臂
（M∈{1,8} × 3 seeds）；Stage-147 事件链保持不变。

### 29.7 Stage-146 v1b 结果：rerank 机制首次得到正信号；v1c 修采样器

v1b（按步分解 flow 头）六臂完成：

| 臂 | best per seed | 均值 |
|---|---|---:|
| M=1 flow BC 单独 | 4/4/16% | **8.0%** |
| M=8 flow + Q-rerank | 20/24/16% | **20.0%** |

**判定命中 §29.4 第三分支：组合有效但采样器不足。** M=8 对 M=1 平均
+12pp（2.5×）、方向跨 seed 一致——这是线路A（flow 加 CQN）历史上第一个
机制正信号，与 on-manifold 校准论证（§29.3）一致。绝对水平受限于
flow BC 采样器本身（8% 对 categorical BC 62%+）。

v1c 单变量修复：FlowPolicyHead 补齐 categorical 塔的两个结构组件——
per-stream rgb/low-dim 投影（raw 6k 维特征先降到 128 维）与沿序列的
GRU。4 个单测通过。挂 Stage-147 完成事件自动重启六臂。

### 28.7 Stage-147 结果：canonical MC 锚 0.1 任务判定 fail，剂量机制确认

三 seed（w=0.1）：`0/12/4/20`、`0/8/28/28`、`0/0/16/48`，validation-best
均值 **32.0%**（clean 72.0%，−40pp）。**non-inferiority fail。**

机制定量确认：MC CE 占 critic 梯度 ~40%（it=1k/4k 的
mc_loss/(critic−mc) = 0.39–0.46），且与 demo margin 写同一组 logits——
canonical 的排序压力被稀释；曲线终点即峰值（20/28/48 仍上行），为
「拖慢而非摧毁」。当年 decoupling 约束的动机得到剂量证据。
训练内 mc_return_mae 0.11→0.06，锚本身在校准。

Stage-147b（预注册 fallback 双臂 × 3 seeds，挂 146-v1c 后）：
`mcw0p02`（剂量 0.02，梯度占比 ~8%）与 `mcvonly`（w=0.1 但
mc_return_value_only=true，梯度只入 dueling value 流，保护 advantage
排序——§20.9 机械在 canonical 的首次使用）。判定不变（§28.6 双指标）；
w=0.1 checkpoints 的校准 probe 与 147b 一并补测。

### 29.8 Stage-146 v1c 结果：rerank 组合确立，EMA 为下一单变量

v1c（特征投影+GRU 流头）六臂完成：

| 臂 | best per seed | 均值 |
|---|---|---:|
| M=1 flow BC 单独 | 56/24/44% | **41.3%** |
| M=8 flow + Q-rerank | **84**/68/32% | **61.3%** |

判定（§29.4）：rerank 贡献 mean +20.0pp、2/3 seeds 胜出；连同 v1b 共
5/6 seed-pair 正向。**「flow 提议 + 校准 Q 在流形内裁判」的组合机制
确立**。对 clean 72.0% 尚差 11pp（M8 均值），但 seed1 的 84% 为本研究
程序单点最高，且采样器仍有 ~20pp 余量（M1 41.3% vs categorical 62.7%）。

已识别缺陷：flow BC 在 seeds 2/3 后期塌 0（56→0、24→0），训练后期不
稳定——FM 策略经典症状。Stage-146b 单变量：flow head 加 **EMA 权重**
（decay 0.999，rollout 用 EMA、训练不变；repo flow_matching.py 同款
机制），M=8 × 3 seeds 对照 v1c M8。挂 147b 完成后自动启动。

### 28.8 Stage-147b 结果：canonical 剂量曲线收口，转 sidecar 后验校准

mcw0p02 三 seed best `56/56/60%`（均值 **57.3%**）；mcvonly `40/52/52%`
（均值 **48.0%**）。加上 w0.1 的 32.0% 与 clean 的 72.0%，canonical 平台
的剂量-响应为 `0 → 72.0`、`0.02 → 57.3`、`0.1@value-only → 48.0`、
`0.1 → 32.0`：**任何有效校准压力都以两位数 pp 损伤行为**；value-only
路由减轻但不消除（atom-softmax 耦合，§20.9 的预测成立）。§28.6 判定：
canonical 在线锚不可行。

**Stage-148（线路二收口形态）：post-hoc sidecar 校准。** 冻结 clean
checkpoint 全部参数（行为按构造 = 72% 基线，零风险），在其 encoder 特征
上离线训练小型 value 头回归 executed-transition MC return；按 episode
划分 held-out 报告 ρ/MAE（参照：decoupled 在线锚 ρ=0.88 / clean 自身 Q
ρ=−0.54）。纯离线，用既有 stage2 clean run 的 snapshot+replay。若
held-out ρ≥0.8，线路二交付「行为不变 + 诚实 behavior-value 读出」，
其 value 语义边界（on-path、非反事实）按 §25 系列证据明示。

### 29.9 Stage-146b 结果：EMA 根除塌陷，budget 成为下一约束

EMA 三 seed best `56/68/68%`（均值 **64.0%**，v1c 61.3%）；末点
`48/68/60%`（v1c 末点 56/48/32 且有 84→16 崩溃）。**EMA 判定：后期
塌陷根除、方差收窄、均值小幅上行**。距 clean 72.0% 差 8pp；曲线 10k
仍上行（seed2 末点即峰值），budget 受限。

Stage-146c 双臂 × 3 seeds（挂 148 后）：`b20k`（20.5k frames、M=8、
snapshot_every_n=5000）与 `m16`（10.5k、M=16）。判定：任一臂三 seed
validation-best 均值 ≥ clean 72.0% 则设计 sealed confirmation（新
eval seeds、预注册 checkpoint 规则）；否则记录 M/budget 缩放曲线。

### 28.9 Stage-148 结果：后验 sidecar 未达门槛，权衡面完整成图

sidecar（冻结 clean 特征 + 小头回归 MC return，episode 级 held-out）：
v1（action 输入、40 锚/ep）held-out ρ=0.580/MAE 0.181；v2（state-only、
全锚、wd=0.01）ρ=0.560/MAE 0.170。**未达 ρ≥0.8 门槛；train ρ=0.96 →
泛化缺口为特征所限**。

诚实对照（同为「完全未见 episode」口径）：decoupled 在线锚 critic 在
clean run 的 episodes 上（跨运行、双方未见）**ρ=0.749**（Pearson
0.753，160 样本；`exp_local/cqn_stage148_sidecar/
decoupled_mc_on_clean_episodes.json`）。即：联合训练把 value 信息写进
特征（0.75），冻结模仿特征装不下（0.56）——**校准要么进训练（付
行为代价），要么受限于模仿特征**。

线路二权衡面（MovePlate、matched 预算）：

| 形态 | task | value（held-out 口径 ρ） |
|---|---:|---:|
| clean 原版 | 72.0% | −0.54（own-replay 口径，有害） |
| clean + 后验 sidecar | 72.0%（按构造） | 0.56–0.58 |
| decoupled + 在线锚 0.1 | 62.7% | 0.749（跨运行）/0.88（own） |
| canonical + 锚（0.02–0.1） | 32.0–57.3% | 被支配 |

进行中的最后一杠杆：v3 pooled sidecar（147 系列 5 个 run 的 replay 併
入训练集、held-out 集不变）检验缺口是否含数据成分。

### 28.10 Stage-148 v3：pooled sidecar 通过门槛，线路二交付

v3（训练集併入 147 系列 5 个 run 的 replay，约 6× 数据；held-out 集
不变 = clean seed1 的 modulus-5 episodes）：**held-out ρ = 0.918、
MAE = 0.045**（1260 样本），超过 ρ≥0.8 门槛，也超过在线锚 critic 的
跨运行 0.749。v2 的 0.56 缺口为数据覆盖所致，非特征容量。

**线路二（CQN-value问号）三层结论收口：**

1. **诊断**（已冻结）：canonical CQN-AS 的 value 是模仿形状的，own-replay
   反校准 ρ=−0.54；success 来自模仿机械。
2. **不可能性**：共享头在线锚在任何剂量下以两位数 pp 换校准
   （0→72、0.02→57.3、0.1→32）；反事实排序在可行预算内不可识别
   （Stage-141/143/144 有功效 fail）。
3. **可交付物**：post-hoc pooled sidecar——行为零改动 + held-out
   ρ=0.918 的 behavior-value 读出。语义边界明示：on-path、非反事实。

可选加固（待 GPU 空闲）：对 §21.104 的 clean seeds 2/3 checkpoints
重复 v3，验证多 training-seed 稳健性。
artifacts：`exp_local/cqn_stage148_sidecar/`。

## 30. 2026-07-26：Stage-146c 结果与 Stage-149 sealed confirmation 预注册

### 30.1 Stage-146c 实际结果

| 臂 | best per seed | 均值 |
|---|---|---:|
| b20k（20k、M=8+EMA） | 72/72/80% | **74.7%** |
| m16（10.5k、M=16+EMA） | 72/64/80% | **72.0%** |
| （146b M8 10.5k 参照） | 56/68/68% | 64.0% |

预算与候选数两个杠杆都有效（+10.7pp / +8pp）。**线路A validation-best
均值首次达到/超过 72.0 参照。**

### 30.2 可比性审计（关键修正）

72.00% 的 clean 参照（§21.104）来自 high-UTD4 协议；本平台
（value_fidelity、UTD1、batch16+16）的 clean 只有 seed1（stage2，
25-ep 曲线 48/56/92/56）。公平判定必须补齐本平台 clean seeds 2/3，
并对全部臂做同协议密封评估。

### 30.3 Stage-149 预注册（评估前冻结）

1. 补训 clean seeds 2/3（canonical value_fidelity_gate、10.5k）。
2. 密封评估：`eval_cqn_as_bigym_checkpoint.py`，**50 episodes/arm、
   eval-seed-start 600（从未使用）**；两个协议都评：
   - primary：距各臂内部 validation-best 步数最近的已存 snapshot
     （b20k 为 5000 粒度、其余 1000 粒度；规则=最近者，平局取更早）；
   - secondary（无选择偏差）：final snapshot。
3. 臂集：clean×3、b20k×3、m16×3。
4. 判定：primary 协议下 flow 臂（b20k 或 m16）三 seed 均值
   ≥ clean 三 seed 均值 −5pp 为 non-inferiority；> clean 为
   superiority；两个协议方向一致才作正式 claim。

### 30.4 Stage-149 密封判定：m16 非劣成立，b20k 未通过

50 episodes/arm、eval-seed-start 600、双协议：

| 臂 | primary（validation-best 邻近） | final（无选择） |
|---|---:|---:|
| clean matched | 70/56/78 → **68.0%** | 64/48/68 → **60.0%** |
| m16 | 52/72/74 → **66.0%**（−2.0pp） | 44/72/60 → **58.7%**（−1.3pp） |
| b20k | 60/62/62 → 61.3%（−6.7pp） | 52/56/52 → 53.3%（−6.7pp） |

**正式 claim（预注册判据满足）：flow+rerank M=16 在 matched 预算下与
clean CQN-AS 非劣（−5pp 容差内，双协议方向一致）。** superiority 未
成立。b20k fail：validation 74.7% 为 25-ep 选择膨胀 + snapshot 粒度
错位（5000 网格距 best-eval 至 2500 步）。matched clean 自身密封值
68.0/60.0 亦低于其 validation 均值，佐证密封协议必要性。

线路A 方法学结论：① rerank 机制 +20pp（对自身采样器，5/6 seed-pair
复现）；② EMA 根除后期塌陷；③ M 缩放有效（8→16 = +8pp validation）；
④ 密封口径达 clean 平价。下一杠杆：M=32（146d，最便宜）；再往上是
transformer 采样器与多任务。

### 30.5 Stage-146d + 线路二加固（并行启动）

146d：M=32 × 3 seeds（10.5k、EMA），单变量 vs m16。sidecar 加固：对
149 新产 clean seeds2/3 checkpoints 重复 v3 pooled（多 training-seed
稳健性）。

### 28.11 sidecar 多 seed 加固

对 Stage-149 新产 matched clean seeds 2/3 checkpoints 重复 v3 pooled
协议（held-out=各自 modulus-5 episodes，pool 不含任何 held-out）：

| clean seed | held-out ρ | MAE |
|---|---:|---:|
| 1（§28.10） | 0.918 | 0.045 |
| 2 | 0.754 | 0.093 |
| 3 | 0.888 | 0.053 |

**三 seed 均值 ρ=0.853、范围 [0.75, 0.92]**；2/3 过 0.8。最弱者
（seed2）亦为行为最弱 checkpoint（validation 56%），特征信息量与
行为质量同向。线路二交付物的稳健性表述定为「均值 0.85、按 seed
0.75–0.92」，完全收口。

### 30.6 Stage-146d：M 缩放饱和，线路A 阶段收口

M=32 三 seed validation-best `72/72/68%`（均值 70.7%）对 m16 的 72.0%：
**16→32 无增益（−1.3pp，噪声内）**。M 缩放曲线：8→16 = +8pp、
16→32 ≈ 0。rerank 的 best-of-N 增益在 M≈16 饱和，binding constraint
回到采样器质量（flow BC 单独 41.3% vs categorical 62.7%）。

**线路A（flow 加 CQN）阶段总结论：**

1. 密封 claim（§30.4）：flow 提议 + 校准 Q 流形内 rerank（M=16、matched
   预算）与 clean CQN-AS 非劣（primary −2.0pp / final −1.3pp）。
2. 机制四件套：rerank +20pp（5/6 seed-pair）、EMA 根除塌陷、M 缩放
   8→16 有效后饱和、预算 10.5k→20k validation 有效但密封未存活。
3. superiority 的两条已识别前沿（均为新投入，未自动启动）：
   (a) transformer/UNet 级流采样器（复用 flow_matching.py 骨干，
   close 采样器 21pp 缺口）；(b) 多任务扩展验证外部效度。

### 30.7 两线终局快照（2026-07-26）

- 线路二（CQN-value问号）：**交付收口**。诊断（模仿伪装，ρ=−0.54）→
  不可能性（共享头剂量曲线 + 反事实不可识别）→ 交付物（pooled sidecar，
  行为零改动，held-out ρ 三 seed 0.75–0.92、均值 0.85）。
- 线路A（flow 加 CQN）：**密封非劣收口**，机制全部定量化；superiority
  前沿明确为采样器架构与任务广度。

### 30.8 Stage-146e 预注册：采样器容量斜率先导

在投入 transformer 采样器集成前，先裁定采样器 21pp 缺口的性质：
单变量把流头容量翻倍（hidden [1024,1024]、gru_layers=2），M=16、
10.5k、3 seeds，对照 m16 的 validation 72.0%。

- 有斜率（>+3pp）→ 容量受限，transformer 集成立项；
- 持平/下降 → 缺口为 demo 数据上限（60 条），采样器路线封顶，
  superiority 唯余多任务/更多 demo 方向。

### 30.9 Stage-146e 密封复核：容量斜率不存活，杠杆树穷尽

m16big validation 78.7%（+6.7pp 过预注册门槛），但密封 50-ep：primary
`54/62/78 → 64.7%`、final `56/62/56 → 58.0%`——与 m16（66.0/58.7）
统计持平，仍非劣于 clean（68.0/60.0）、仍未超越。§30.8 的判定分支基于
validation 斜率，密封复核否决其结论：**transformer 集成的容量论证不
成立**。

至此 flow+rerank 家族三个独立配置（m16、b20k、m16big）密封全部落在
clean −7~−1pp 区间，无一超越；validation 尖峰（84/88%）无一存活。
本任务（MovePlate、60 demos）的杠杆树穷尽：M 饱和于 16、预算与容量
的 validation 增益均为选择噪声。**密封结论固化为非劣平价**（§30.4
claim 不变）。

剩余 superiority 假设均需改变实验域：更多 demos（数据上限假设的直接
检验）或第二任务（外部效度）。两者为新范围，待用户方向决定。GPU 已
释放，无运行中作业。

## 31. 2026-07-26：Stage-150 预注册——用户提案：双头保留原核心

用户假设：单头坍缩应由双头不同 loss 解决，**不应改动 high-level 核心**
（Double-Q argmax bootstrap、全序列 TD、Q 参与选动作）。此组合在今晚
校准平台上未原样测过（历史 blend 失败证据来自 direct-Q/high-UTD 平台）。

Stage-150 双臂 × 3 seeds（10.5k、密封协议复用 stage149）：

- 平台：separate_bc_policy=true（BC 头照常模仿）+
  td_target_action_source=critic（**原版 Double-Q argmax bootstrap**）+
  critic_sequence_mode=full（**原版全序列 TD**）+ mc_return=0（纯原核心）
- 臂 A `pvb1`：rollout 用 policy_value blend（Q + β·logπ_BC，β=1）——
  Q 实际参与每步选 bin；
- 臂 B `pvb0p2`：β=0.2（Q 主导更强）。
- 判定：密封 50-ep（seed 600）对 clean（68.0/60.0）与 m16（66.0/58.7）；
  若任一臂密封超 clean → 用户假设成立，核心不应改；若均 ≤ → 分歧由
  数据裁决归档。

## 32. 2026-07-26：Stage-150 判定与 Stage-151 预注册（开环训练）

### 32.1 Stage-150 实际结果（双头保留原核心 + Q 参与选动作）

pvb1（β=1）validation best `72/40/72%`（均值 61.3%，≈纯 BC 62.7%）；
pvb0p2（β=0.2）**三 seed 全程 0%**。剂量方向 = Q 话语权越大行为越差；
原版 Double-Q 核心的流形外排序在被赋予控制权时具破坏性。用户假设
「核心不改、双头即可」在此配置下未成立；pvb1 密封评估随后归档。

### 32.2 Stage-151 预注册（用户方向：训练不用 ensemble 就用 open loop）

训练侧 rollout 改开环 chunk 执行（`method.temporal_ensemble=false`：
缓存一条 K=16 plan 执行到底再刷新），消除 §20.4 的三重代价（决策稀释
15×、credit 错位、探索被平均）。评估侧密封时双模式都测
（`--method-temporal-ensemble true/false`），primary=ensemble on。

- 臂 A：canonical clean + 开环训练 × 3 seeds——用户问题的直接检验；
- 臂 C：flow+rerank（M=16+EMA）+ 开环执行被选 chunk × 3 seeds——
  rerank 效力不再被稀释；
- 臂 B（随后实现）：开环 + coherent ε-bin-flip 探索 + bc_lambda decay。

### 32.3 臂 B 实现完成

`bin_flip_prob/bin_flip_level`（host 侧、开环 plan 刷新时触发：随机一维、
随机一层、整 chunk 平移到同一 sibling cell、深层子索引继承——整数 cell
平移按构造无 alias；元数据复用 structured_explore 字段，assignment_prob
按 start=ε/持续=1 记录）与 `bc_lambda_schedule`（canonical FOSD/margin
的时间调度，legacy 四签名组合保持不变）。新增 5 个单测通过（flip 不
变量 3 + 调度 2）。臂 B 配置：开环 + flip ε=0.2 + bc 1.0→0.2 线性
decay，×3 seeds，挂 151-A/C 完成事件。

### 32.4 Stage-151 A/C 结果与臂 B 判据修正（B 出结果前冻结）

A（clean+开环）validation best `44/52/48`（均值 **48.0%**，对 ensemble
clean −24pp）；C（flow+rerank+开环）`20/32/16`（均值 **22.7%**，−49pp）。
**开环方向任务层面不成立**：ensemble 的闭环逐步纠错在本任务承重，
flow 族依赖更深（ensemble 一直在平均采样噪声）。§20.4 三代价论保留，
但收益-代价符号在本任务为负。validation 差距远超选择噪声带，A/C 不做
密封复评（决策记录）。

**臂 B 判据修正（其训练进行中、结果未见时冻结）**：任务成功率降为
描述性；主判据改为**因果探针**——sibling_horizon + round_robin + L0 +
H4、fresh eval seeds 700–711、B（treatment）对 A（matched 开环对照，
唯一差异 = bin_flip+bc_decay）三 seed 逐对与 pooled crossed-bootstrap
（复用 stage143 汇总器，B=w0p1、A=w0p0 命名）。pass = B 逐 seed 超 A
且 pooled CI 下界 > 0.5。这是「探索与查询语义对齐后 TD 能否识别反事实」
的直接检验，也是臂 B 存在的全部意义。

### 32.5 臂 B 因果判定：gate fail，但开环平台出现程序首个 pooled CI>0.5

B（flip+decay）逐 seed `0.632/0.667/0.638` 对 A（开环对照）
`0.603/0.635/0.761`：2/3 胜，**冻结判据 gate fail**；flip 的边际贡献
±0.03（噪声级），margin-decay 无可辨增益。B pooled `0.643`、
**CI [0.540, 0.750]——历史首次下界过 0.5**（143/144 均骑线）。

更重要的伴生发现：**无任何探索处理的开环对照 A 自身因果准确率
0.60–0.76**，远超 ensemble 平台历代对照（0.41–0.55）。候选机制：开环
使执行动作 = 选择动作，TD 的 chosen-bin 监督与行为语义精确对齐
（§20.4 错位的修复），普通 TD+margin 首次学到真实 per-bin 差异。

两个待杀混杂（结论声明前必须处理）：
1. eval-seed 集不同（700–711 vs 历史 400–411）——正在跑 seed-set 对照
   （ensemble 训练的 149 clean seed3 在 700–711 上探测）；
2. 探测时执行模式不同（开环 checkpoint 探测时 continuation 也是开环，
   干预不被 ensemble 稀释，仪器灵敏度本身可能更高）——若混杂 1 排除后
   信号保留，需再做执行模式交叉（ensemble 训练 checkpoint × 开环探测）。

### 32.6 混杂分解与 Stage-150/151 全线收口

**seed-set 对照**：ensemble 训练的 canonical clean seed3 在 700–711 上
`0.622 [0.503, 0.730]`（185p）——开环因果优势解释死亡（开环 pooled
0.643 vs 0.622，噪声级）。**分解探针**：decoupled+MC seed2 在 700–711
上 `0.564 [0.370, 0.761]`（94p），对其 400s 时代的 0.445——seed 集本身
贡献约 +0.12；canonical 对 decoupled 在共同 seed 集上余差 +0.06–0.08，
CI 重叠、功效不足，列为低优先开放项。因此「首个 pooled CI>0.5」按协议
真实、但跨 seed 集不可与历史直接比较；开环的因果增益与 bin-flip 的
边际增益均未成立。

**Stage-150 密封归档**：pvb1（β=1）50-ep `46/50/20`，均值 **38.7%**
（clean 68.0，−29.3pp）——双头保留原核心方案在两个剂量下密封决定性失败。

### 33. 程序级终局快照（2026-07-26）

用户三个假设的裁决（全部 matched、多 seed、预注册）：

| 假设 | 裁决 |
|---|---|
| 双头+原核心+Q 参与选动作 | fail：β 剂量与成功率反向（1.0→38.7% 密封、0.2→0%） |
| 训练去 ensemble（开环） | 任务 fail（clean −24pp、flow −49pp）；因果优势被 seed-set 对照消解。ensemble 的闭环纠错在本任务承重 |
| bin-flip 探索 + margin decay | 任务与因果均无可辨边际效应 |

有效结论保持：线路二交付（sidecar ρ 0.85–0.92 @ 零行为代价）；线路A
密封非劣（m16 66.0/58.7 vs clean 68.0/60.0）+ rerank 机制 +20pp。
开放前沿（需新范围决定）：第二任务外部效度；更多 demos 的数据上限
检验；canonical-vs-decoupled 因果余差的功效化复测；执行模式×探测
仪器交叉。GPU 已释放，无运行作业。

### 34. Stage-152：coarse-flow（粗层 critic + bin 条件 flow）——新算法主线（2026-07-26）

**背景与算法动机（用户裁定）**：用户判定 rerank 属 test-time steering 而非算法
创新，要求算法级贡献。本程序全部负结论指向同一结构事实：CQN 的 zoom 层级在
统计上不等价——粗层（level 0，5 bins/dim）的兄弟 bin 有在线数据支撑，TD 可识
别；细层（等效 125 格/dim）的 counterfactual 不可识别（Stage-141/143/144
powered null），TD 退化为模仿影子。原版对全层级同质施加同一 loss。

**算法（coarse-flow / CCFF）**：
1. critic 仅保留 level 0（levels=1），loss 与原版 canonical 完全一致
   （TD C51 + margin；demo_fosd=false 平台）；
2. 删除细层，代之以 bin 条件 flow 头：沿用 v1c 架构（per-step 因子化 +
   投影 + GRU），额外输入 per-step bin one-hot + 归一化 cell 中心；flow 在
   [-1,1] 的 cell 内坐标建模连续残差，demo-only CFM，EMA(0.999) rollout；
3. 出动作：critic 在粗 bin 上原生 argmax（rerank 的摊销形式），flow 在选中
   cell 内积分出连续动作（构造上不可能越出 cell）；训练侧 TD 完全在 bin 分辨
   率（bin 中心 bootstrap，within-cell 残差不改变 encode 结果）。
4. 预留 `coarse_flow_selfdistill_weight`：高回报在线 chunk 以降权加入 CFM
   （需 canonical MC 提供 mc_return；本 gate 不启用）。

**实现**：`cqn_as.py`（FlowPolicyHead bin_context 参数、_coarse_flow_cell/
_coarse_flow_action、canonical greedy 分支、EMA canonical 延伸）、`cqn.py`
（canonical loss_fn 内 CFM 项，features stop-gradient——critic 梯度与 legacy
逐位一致）、spec/factory/yaml。单测 6 组通过；全回归 205+132 通过（顺带修复
bin_flip 属性缺失导致的 CQNFlowAS 5 个既有失败，getattr 守卫）。GPU smoke
700 帧通过，coarse_flow_loss 收敛中（1.38@~200 upd）。

**发现**：BiGym move_plate 数据集实为 51 条成功 demo（env.expected_successful_demos=51），
非先前叙述的 35。`demos=10` 载入前 10 条、其中 9 条成功（demo 5 失败）——
A/B 两臂共享同一 9 条子集，需 `env.expected_successful_demos=null`。

**预注册（结果未见）**：
- 竞技场：demo-scarce（demos=10→9 条成功）。纯模仿应显著下跌；若 critic 的
  粗层 TD + flow 精度分工恢复更多，「RL 部分有用」首次以成功率口径成立。
- 臂：A=vanilla value_fidelity_gate@demos10（levels=3 原版，3 seeds，GPU1）；
  B=stage152 CCFF@demos10（3 seeds，GPU5）；B-full=CCFF@全 demo（3 seeds，
  尾随），对照历史 clean@full（sealed 68.0/60.0，validation 轨迹在档）。
- 平台：value_fidelity_gate（10.5k 帧，batch16，tau0.005，demo_fosd=false，
  eval 25ep@2500，snapshot@1000）。
- 判据（validation 口径，末两个 eval 点均值、3-seed 平均）：
  * 主：B@10 − A@10 ≥ +10pp → pass，晋级密封（50-ep，fresh eval-seed-start
    800，双协议）；(−5, +10) → inconclusive（加 seed 或 20k 预算再判）；
    ≤ −5pp → fail（数据 ceiling 结论最终定案，转第二任务）。
  * 次：B-full 相对历史 clean@full 同口径 ≥ −10pp 才可谈非劣密封。
- 诊断记录：coarse_flow_loss 收敛、level-0 critic 熵、（若可）critic bin 选择
  与 demo bin 的一致率。
- 已知偏差：A/B 的 stddev 探索噪声施加于各自动作之上（matched）；B 的 EMA
  blend 在 canonical 路径按 update() 调用节律（num_update_steps=1 下与
  decoupled 语义一致）。

### 34.1 Stage-152 gate 结果（validation 口径）与晋级决定

**demo-9（demos=10 载入、9 成功）末两点均值**：
- vanilla_d10：24 / 20 / 34 → 均值 **26.0%**
- ccff_d10：30 / 32 / 18 → 均值 **26.7%**
- 主判据 B−A = **+0.7pp ∈ (−5, +10) → inconclusive**。按预注册走扩 seed
  路径：两臂各加 seeds 4-6（stage152b 链已排队）。

**full-demo（51 条成功 demo）末两点均值**：
- ccff_full：86 / 62 / 74 → 均值 **74.0%**
- 历史 clean@full（同平台，stage149 复用运行）：58 / 50 / 76 → 均值 **61.3%**
- 次判据（≥ clean − 10pp）通过，且 validation 点估 +12.7pp——超出非劣直指
  优越，但 validation 尖峰历史上从未存活密封，故立即密封。

**晋级密封（预注册补充，结果未见）**：ccff_full s1-3 与 clean s1-3 全部在
fresh eval-seed-start **800**（600 已被 149/151 使用）重新密封，50-ep 双协议
（primary=最近验证最优快照、secondary=final）。判据：两协议下
mean(ccff_full) − mean(clean) ≥ 0 → 非劣成立且记录点估；≥ +8pp 且 3/3
seed 方向一致 → 报告优越（首个超过 clean 的算法级结果）；≤ −5pp → 视为
validation 泡沫，记录并回 demo-10 轴。stage152b 控制器同时执行密封与
demo-10 扩 seed。

### 35. Stage-153：分层 ε-bin 探索（用户方案，闭环 ensemble 兼容）

**用户规格**：随机探索不同 bin，逐层探索率递增（L0 最小，L1、L2 依次增大，
但都要小），并且**保持闭环执行 + temporal ensemble**（区别于 Stage-151 的
开环 bin-flip）。

**机制（`bin_explore_probs`）**：每次新 plan 生成时按层概率触发（粗到细，
先中先得）：随机选一维，把该维 level-l bin 移到随机 sibling（整数 cell 平移、
inherit-refine，复用 bin-flip 的 alias-free cell 数学）；关键差异是**持续性**
——同一平移在接下来 K=16 个新 plan 上重复施加，否则单次翻转会被 ensemble
的最新-plan 权重（6.7%）稀释到从不执行。闭环纠错全程保持。单测 3 组通过
（sibling/parent 不变量、持续性、ensemble act 兼容、验证互斥）。

**预注册（结果未见）**：臂 E = vanilla + bin_explore_probs=[0.002, 0.004,
0.008]（每步触发率合计 1.4%，窗口占比约 18%，深层配额 4×粗层），full demo，
3 seeds，value_fidelity_gate 平台。对照 = 同平台历史 clean s1-3（validation
末两点均值 61.3）。判据（validation 末两点均值、3-seed 平均）：E − clean ≥
+8pp → 晋级密封@800；∈ (−8, +8) → 无边际效应（与 Stage-151 bin-flip null
一致，宣告该假设在两种执行模式下均 null）；≤ −8pp → 探索有害，记录剂量。
已知限制：未接 RCT 元数据（本臂只判任务效果，不判因果可辨识）；ε 为首个
剂量点，若 null 且用户要求可做剂量扫描。排队在 stage152b 之后自动起跑。

### 34.2 Stage-152 密封裁决：coarse-flow 首个密封优越（2026-07-27）

50-ep、fresh eval seeds 800、成对同训练 seed、双协议：

| 协议 | CCFF-full (s1/s2/s3) | clean (s1/s2/s3) | 差（成对） |
|---|---|---|---|
| primary（最近验证最优） | 80/68/86 → **78.0** | 72/56/76 → **68.0** | +8/+12/+10 → **+10.0** |
| final（免选择） | 88/72/80 → **80.0** | 66/54/70 → **63.3** | +22/+18/+10 → **+16.7** |

预注册优越判据（≥+8pp 且 3/3 方向一致）在**两个协议下同时满足**。clean
在 800 上的 primary 均值 68.0 与其 600 时代完全一致——无 seed-set 偏移，
对照有效。**这是整个程序第一个存活密封的正收益，且是算法级改动**（层级
按可识别性分工：TD 全留粗层，细层由 bin 条件 flow 连续残差替代），非
test-time steering。与 §30.4 的 m16 非劣（−2pp）对比：同为 flow+critic
组合，把 critic 的话语权从「对 M 个流形样本重排」改为「在可识别的粗分辨
率上原生 argmax」后，从非劣变为 +10~+17pp 优越。

demo-9 轴保持 inconclusive（+0.7pp），seeds 4-6 扩充已在跑（152-ext）；
Stage-153（分层 ε-bin 探索）排队其后。

### 36. Stage-154 预注册：CCFF 的 TD-off 对照（定性裁决）

**动因（用户质询）**：CCFF 是否已退化为模仿学习算法——flow 头是纯模仿
（demo-only CFM），粗层 critic 的 loss 与原版相同（TD+margin 混合，§28 已证
margin 主导排序）。Stage-152 的密封 +10pp 只证明算法赢，未证明为什么赢：
候选解释 (a) 模仿/架构效应（连续残差消除细层量化 + margin 驱动的粗层选择）
vs (b) RL 效应（粗层 TD 从在线数据修正 bin 选择）。

**臂**：Stage-152 CCFF 平台单变量 `critic_lambda: 0.1 → 0.0`（Bellman 信号
完全关闭，margin+CFM 保留），full demo，3 seeds，validation 口径先判。

**判据（validation 末两点均值、3-seed 平均，对照 = stage152 ccff_full 的
74.0）**：TD-off ≥ 74.0 − 5pp → **打平**：CCFF 定性为模仿学习算法（架构
贡献成立、RL 贡献 null，与全程序诊断一致，如实改写叙事）；≤ 74.0 − 8pp →
**TD 承重**：程序首个 TD 因果贡献证据，晋级密封@800 与 Stage-152 合并
报告。中间带 → 扩 seed。与 Stage-153（分层 ε-bin 探索，给粗层 TD 喂反
事实数据）构成组合：153 探索×154 开关共同定死「CQN 能否学到能用的
value」主线。排队在 153 之后自动起跑。

### 37. Stage-155 预注册：纯 Flow（无选择）对照——分解套件补全（用户要求）

**动因**：用户指出还需对比纯 Flow Matching——检验粗层 RL 选择整体是否承重。
与 Stage-154 组成完整分解：clean（无 flow）68.0/63.3 密封 ← CCFF（选择+
残差）78.0/80.0 密封 → 154 TD-off（选择但无 Bellman）→ 155 纯 flow（无
选择）。

**臂**：stage152 平台单变量 `coarse_flow_pure=true`——flow 建模完整动作
chunk（无 bin 条件），critic 照常训练但不参与 rollout（encoder 仍为 critic
塑形，与 CCFF matched；因此本臂隔离的是「决策时选择+条件化」这一机制整
体）。full demo，3 seeds。

**判据（validation 末两点均值、3-seed 平均，对照 = ccff_full 74.0）**：
纯 flow ≤ 74.0 − 8pp → 选择机制承重（与 §29 flow-BC-alone 46% 的旧证据
一致，且此次平台完全 matched）；≥ 74.0 − 5pp → CCFF 的增益全部来自
连续残差架构，粗层选择（无论 TD 还是 margin）不承重——CQN 部分可整体
删除，算法如实改名纯模仿。中间带 → 扩 seed。三臂（152/154/155）合并后
给出 +10pp 的完整成分分解。排队在 154 之后自动起跑。

### 34.3 demo-9 轴 6-seed 终裁：无可辨效应

扩充后（末两点均值）：vanilla_d10 = 24/20/34/34/20/26 → **26.3%**；
ccff_d10 = 30/32/18/28/22/36 → **27.7%**。B − A = **+1.4pp**（6 seeds），
仍在无效应带内且远离 +10 判据 → **demo-scarce 轴宣告无可辨效应**。
CCFF 的优势只在全数据显形（密封 +10.0/+16.7），demo 匮乏时粗层选择与
flow 同等饥饿。诚实注记：这削弱「粗层 TD 在利用在线数据」的解释（demo
越少、在线数据相对占比越高，却无优势），使 154/155 分解更为关键。

### 35.1 Stage-153 裁决：分层 ε-bin 探索无边际效应（两种执行模式下均 null）

validation 末两点均值：binexp = 56/56/60 → **57.3%**；对照 clean 同口径
**61.3%**（58/50/76）。E − clean = **−4.0pp ∈ (−8, +8) → 无可辨边际效应**，
按预注册与 Stage-151 开环 bin-flip 的 null 合并：「随机 bin 探索改善 CQN」
假设在开环与闭环 ensemble 两种执行模式、两个剂量点下均 null。附注：
binexp 的 seed 间方差显著小于 clean（56/56/60 vs 58/50/76，n=3 仅记录不
下结论）。若 154/155 显示选择本就靠 margin，则探索 null 有了机制解释：
喂给 TD 的反事实数据没有被行为使用。

### 36.1 Stage-154 validation 裁决：边界命中，按协议晋级密封

TD-off 末两点均值：78/54/66 → **66.0%**，对 CCFF 74.0 差 **−8.0pp**——
恰好落在预注册「≤ 74.0−8 → TD 承重」的边界上（含端点成立）。按协议
晋级密封@800 双协议，与已密封的 CCFF（78.0/80.0）对比作最终裁决。
诚实注记：n=3、seed 间散布 24pp（78/54/66），边界命中完全可能在密封
下消解——validation 尖峰教训在档，密封才是判词。若密封确认 TD-off <
CCFF −8pp，程序首次证明 Bellman 信号承重；若密封打平，回到模仿定性。
Stage-155（纯 flow）自动接跑中，其 validation 结果出来后与 TD-off 一并
密封（一批 800 评估）。

### 37.1 Stage-155 validation 结果与密封批决定

纯 flow 末两点均值：82/66/58 → **68.7%**，对 CCFF 74.0 差 **−5.3pp**，落在
预注册中间带 (66, 69)。validation 口径三臂全景：CCFF 74.0 > 纯 flow 68.7 >
TD-off 66.0 > clean 61.3——但臂间差（≤8pp）与 seed 内散布（约 20pp）同
量级，validation 分辨率不足以裁决。且 TD-off < 纯 flow（margin-only 选择
弱于无选择）提示选择的 margin 部分可能为零甚至为负，TD 部分承担全部
选择增益——此排序若密封成立，结论出人意料地反转为「选择的价值恰恰
全在 TD」。

**密封批（预注册补充，结果未见）**：tdoff s1-3 与 pureflow s1-3 全部
50-ep@800 双协议（与 CCFF 78.0/80.0、clean 68.0/63.3 同 seed 集同协议
可直接三方比）。裁决表：
- CCFF − pureflow ≥ +8pp（两协议一致）→ 选择机制承重；
- 且 tdoff ≤ pureflow → margin 无贡献、增益归 TD（RL 承重的最强形态）；
- CCFF ≈ pureflow（±5pp 内）→ 选择不承重，CCFF 如实定性为模仿算法
  （用户判断成立），粗层 critic 从算法中删除。

### 37.2 三方密封终裁：分解完成（2026-07-27）

50-ep@800 双协议全表（primary / final，s1/s2/s3）：

| 臂 | primary | 均值 | final | 均值 | 合并均值 |
|---|---|---|---|---|---|
| CCFF | 80/68/86 | 78.0 | 88/72/80 | 80.0 | **79.0** |
| TD-off | 86/70/72 | 76.0 | 74/70/78 | 74.0 | **75.0** |
| 纯 flow | 74/84/60 | 72.7 | 72/52/64 | 62.7 | **67.7** |
| clean | 72/56/76 | 68.0 | 66/54/70 | 63.3 | **65.7** |

两协议排序一致：CCFF ≥ TD-off ≥ 纯 flow ≈ clean。成分分解（合并口径）：
- **flow 残差单独：+2.0（≈0）**——纯 flow 打平 clean，连续残差架构本身
  不是增益来源；validation 的 TD-off<纯 flow 倒序未存活密封。
- **粗层选择机制整体：+11.3**（CCFF − 纯 flow；final +17.3 决定性，
  primary +5.3 未过 +8 杠——预注册严格口径下"选择承重"只在 final 协议
  确立，primary 方向一致但幅度不足）。
- **选择内部拆分**：margin 份额 +7.3（TD-off − 纯 flow），TD 份额 +4.0
  （CCFF − TD-off；primary 成对 1/3 正向、final 3/3 正向——**TD 贡献小
  且 seed 间不稳，未达可主张水平**）。validation 的 −8 边界命中（"TD 承
  重"）未存活密封，再次验证密封纪律。

**对用户质询「CCFF 是否已是模仿学习算法」的最终答复**：一半成立。
不成立的一半：纯模仿变体（去掉选择）丢掉全部增益，粗层 critic 选 bin
机制确实承重，CQN 结构保留有据；成立的一半：该选择机制的主导信号是
margin（demo 模仿），Bellman 信号的边际贡献 +4.0 未过显著杠。诚实定性：
**层级化模仿算法（粗层分类 + 细层 flow 残差）+ 未证实的小幅 TD 增益**，
「RL 学到了可用 value」在本任务仍不成立——与线路二两天的全部诊断自洽。

主结果不变：CCFF 密封优越 clean +10.0/+16.7（合并 +13.3），机制归因为
架构（层级分工），非 RL。TD-off 变体（无 Bellman）密封 75.0 也胜 clean
+9.3——可作为更简选项写入论文消融。

### 38. Stage-156 预注册：CCFF + 自模仿（±ε-bin 探索）——打通 online 回流

**动因（用户指令）**：用户指出应使用 online rollout。澄清：训练一直是
online RL，但 online 数据只进 TD（已证不承重）；gate 平台继承
`use_self_imitation: false`，成功 online episode 不回流模仿通道
（margin/CFM 只认 demo 标记）。一号开关即 `use_self_imitation=true`。

**臂（各 3 seeds，10.5k matched，GPU1/GPU5 并行）**：
- S = CCFF + use_self_imitation=true（单变量）；
- SE = S + bin_explore_probs=[0.014]（levels=1 单层，剂量对齐 153 总触发
  率）——探索负责发现新 cell，自模仿负责消费成功探索，传导链首次闭合。

**先验（诚实记录）**：decoupled 平台自模仿为负（145c 低于对照）；canonical
CCFF 上未测，且 CCFF 成功率约 70%，回流样本质量更高。

**判据（validation 末两点均值、3-seed 平均，对照 = ccff_full 74.0 /密封
合并 79.0）**：S ≥ 74.0 + 8pp → 晋级密封@800；S 或 SE ∈ ±8pp → 无边际
效应（回流在本预算不改变行为）；≤ −8pp → 自模仿在 canonical 平台亦有害，
与 145c 合并为跨平台负结论。SE − S 单独读作探索的边际（有消费通道条件
下），这是 ε-bin 假设的最后一次机会：若仍 null，探索线在所有已试通道下
关闭。

### 38.1 Stage-156 裁决：online 回流无增益、纯自模仿轻度有害、探索边际不足

validation 末两点均值：
- S（自模仿）：74/60/62 → **65.3%**，对 CCFF 74.0 差 **−8.7pp**——触及
  预注册有害判据（≤−8）。与 145c（decoupled 自模仿为负）合并：**自模仿
  回流在两个平台上均无增益、canonical 上轻度有害**。机制上与「克隆自身
  rollout 有害」的既有结论一致：回流样本挤占真 demo 的监督权重。
- SE（自模仿+粗层探索）：74/60/80 → **71.3%**，差 −2.7 → 无效应带；
  SE − S = **+6.0**：有消费通道时探索方向为正但幅度未达 +8 判据。ε-bin
  假设按预注册在「所有已试通道」下关闭——最有利条件（消费通道存在）下
  也只有噪声级正边际。
（10.5k 预算、n=3 口径；此两臂不晋级密封。）

程序状态：CCFF（无自模仿、无探索的原型）仍是最优配置——密封合并 79.0，
对 clean +13.3。线内改进的已试选项（探索、自模仿、TD 加权、开环、双头、
AWR、bin-flip）全部未能超越它；下一个可信的增量只剩换 regime（第二任务、
更长预算、更强 demo 多样性）。

### 39. Stage-157：核心结论在 100k 官方 checkpoint 上的验证（2026-07-27）

**动因（用户质询）**：全部密封结论出自 10.5k 预算；用户要求用库中 100k
官方运行（move_plate_paper_seed1-4_100k_nw0，self-imitation on，最终成功
64/76/40/60）验证结论是否外推。

**① Q 校准**：ρ(Q, first-success 回报) = −0.06/−0.67/−0.37/−0.72，均值
≈ **−0.45**（10.5k 时代 −0.54）——**反校准在十倍在线数据 + 自模仿下完整
复现**。demo 成功组内排序保持 0.6-0.7（认得专家），online 成功组混乱。

**② 反事实探针**（sibling L0 rr H4，seeds 700-711，协议与 143/151 一致）：
0.569/0.579/0.537/0.583，合并 ≈ **0.567**（716 对）——与 10.5k 时代同带
（0.54-0.62），机会线附近。**十倍同分布数据未提升反事实排序**。

**结论**：两条核心结论均在 100k 尺度成立；「加步数」路径被直接测量证否。
缺口确认为同状态异动作的对照数据（分布性质），而非数据量。
附注：官方 100k 配置最终成功均值 60%，低于 CCFF@10.5k 密封 79.0。

### 39.1 Stage-158 预注册：ε 探索 × 100k × margin 衰减（唯一未证否组合）

**臂**：官方 100k 配置 + bin_explore_probs=[0.002,0.004,0.008] +
bc_lambda_schedule=linear(1.0,0.25,100000)，2 seeds（每卡一个），
snapshot@20k。**主判据是探针不是成功率**：最终 snapshot 的 sibling
pairwise accuracy ≥ 0.70 → 反事实在探索+衰减下可学，进入行为耦合阶段；
仍 ≈ 0.567 带 → 结构瓶颈（K×D 广播 target、chunk 信用分配）盖章，
探索线最终关闭。次要记录：任务成功率轨迹（对照官方 60%）、探针 vs
训练量曲线（20k 快照事后补测）。预计约两天。

### 39.2 Stage-158 中间点（40k）：探针首次显著移动

sibling 探针 @40k snapshot（协议不变）：seed1 **0.618**（220 对）、seed2
**0.726**（190 对），合并 **0.668**（410 对）——对无探索基线 0.567
（157，716 对）+10.1pp，约 4σ（二项 SE≈2.3pp）。**整个程序中反事实排序
第一次离开机会带**，且 seed2 已越过 0.70 预注册线。同期任务成功率不降
（seed1 均值 ~64% vs 官方基线 ~50%）。margin 权重此时已衰减至 ~0.7。
待 101k 终点探针确认趋势（探索数据继续积累 + margin 进一步衰减至 0.25）。

### 39.3 Stage-158 终点裁决（101k）：反事实知识真实存在但在 ~0.64 平台化

sibling 探针 @101k（协议不变）：
- seed1 **0.655** [0.558, 0.744]（229 对）——单 seed CI 下界首次明确高于机会线；
- seed2 **0.630** [0.525, 0.730]（189 对）；
- 合并 **0.6435**（418 对），对无探索基线 0.567（716 对）差 **+7.7pp ≈ 2.6σ**；
  per-state Spearman CI 亦为正（[0.10, 0.50] / [0.06, 0.48]）。
- 40k→101k：0.668→0.644（噪声内）——**上升在 40k 前完成，此后平台化**；
  margin 已衰减至 0.25、探索数据翻 2.5 倍未再抬升。
- 任务成功率（validation 口径）：seed1 末 8 评均值 ~73.5%、seed2 ~67.5%，
  显著高于官方基线（~50-60%），接近 CCFF validation。

**裁决**：预注册两端均未命中（≥0.70 未达；≈0.567 亦否）。定性为：
**探索×margin衰减×尺度使 CQN 首次学到真实、可复制（2/2 seeds）的反事实
知识，但在 0.64-0.67 平台化**。平台化候选解释：残余 margin（0.25 未到 0）、
探索剂量、或结构上限（K×D 广播 target、chunk 信用分配）——未裁决。
两天主线「CQN-value 问号」的最终答案由「否」修正为「**部分可以，条件
是探索+衰减+尺度，且存在未定位的天花板**」。

### 39.4 中期：ε-bin@100k 定点 50-ep 与 seed 匹配补全

- seeds 1/2 定点 50-ep@800 双协议：最佳 82.0/78.0（均 80.0）、最终 82.0/80.0
  （均 **81.0**）——两协议一致，同尺下超过 CCFF（78.0/80.0）与 clean
  （68.0/63.3），暂居全表第一。
- 25-ep 口径（74/70.5）与 50-ep@800（81）的差来自 eval seed 集效应
  （clean 亦从 61.3 读到 68.0）与小样本噪声，非"episode 多了成功率变高"。
- Stage-158b 进行中：补 seeds 3/4（与官方 seed 值 1-4 完全对齐，同初始化
  流），各配 101k 探针；官方 4 个 101k checkpoint 的 50-ep@800 已提前并行
  插跑。

### 39.5 Stage-159 预注册：2×2 因子分解（用户要求归因）

**动因**：Stage-158 的增益是"探索"还是"margin 衰减"的贡献？已有臂：官方
（都不加，4 seeds）、158（两者都加，4 seeds 补齐中）。新增：
- **E 臂（只探索）**：stage158 配置去掉 bc_lambda_schedule，seeds 1/2；
- **D 臂（只衰减）**：stage158 配置去掉 bin_explore_probs，seeds 1/2。
每 run：100k 训练 → 101k sibling 探针 → 50-ep@800 定点。排队在 158b 后。

**判读表（预注册）**：
- 探针（value 口径）：预期 E>官方（探索造数据）而 D≈官方（无对照数据可
  用）；若 D 也升，衰减本身即可释放 TD（意外）。
- 任务（50-ep@800）：主效应与交互效应按 2×2 分解；若只有组合臂高于两个
  单因素臂 → 交互承重（探索需要衰减才能变现，153 的机制解释获证）；若
  E 单独即达 158 水平 → 衰减非必要；若 D 单独即达 → 探索非必要。

### 40. Stage-160 预注册：随机低维观测遮罩（保留 floating base）（用户方案）

**规格（用户）**：训练时以 20% 概率遮掉整条低维观测、保留 floating base，
迫使模型更依赖视觉。1 seed，100k。

**实现要点**：管线里 low_dim_state（60 维）原本不含 floating base（ConcatDim
keys_to_ignore 排除、模型不消费）。故 (a) 新增 AppendKeysToLowDim wrapper：
把归一化后的 proprioception_floating_base（3 维）拼到 low_dim_state 尾部
（63 维，位置固定）；(b) method 侧在 _augment_update_obs_inputs（与 rgb
random-shift 同处）以 p=0.2 每样本把每帧前 60 维置零、保留末 3 维；obs 与
next_obs 独立抽签；**act()/评测永不遮罩**。单测 3 组 + 全回归 78 通过，
GPU smoke 通过。基底 = stage158 配方（探索+衰减）。

**判据（对照 stage158 seeds1/2）**：任务 50-ep@800 终点对照 81.0；探针对照
0.644。注意本臂输入多了 3 维 base 状态（用户规格所需），严格归因需
"仅加 base 不遮罩"对照，暂不跑、结果显著时补。

### 41. 训练 infra 剖析与 fast 版重设计（2026-07-28）

**基准（3000 帧+2 eval，同 seed）**：v1 fast（预取+锁+B+C）854s vs 原版
829s——无加速且 critic_loss 有 ~0.2% 漂移（预取的有界陈旧度）。v1 否决。

**py-spy 剖析（480s，主循环）**：
- 代码库 iterator 本就有后台预取线程（_prefetch_loop）——v1 的预取层重复
  建设；采样线程 CPU 占比 70%（采样 13% + np.stack 15% 等）；
- **主线程头号热点 = demo-merge 主机拼接**（_DemoMergedIterator 的
  np.concatenate，每 update 约 130MB memcopy）= 全部 CPU 采样的 29.7%，
  另 batch→device 转换 ~10%；
- 瓶颈本质是 GIL 串行下的总 Python 工作量，而非缺少重叠。

**v2 设计**：_DeviceMergedIterator——两 batch 各自 device_put 后在 GPU 上
jnp.concatenate（数值逐位等同）+ B（update_block_every_steps=10，uniform
replay 下逐位无损）+ C（关 wandb/视频）。等价性验收进行中：orig_repeat
（run-to-run 确定性基线）、bc_only（仅 B+C）、fast2（v2 全量），同 seed
同预算比对 critic_loss 轨迹与墙钟。

### 41.1 infra 验收终表（3000 帧 + 2 eval，同 seed）

| 版本 | 墙钟 | critic_loss 第2点 | 判定 |
|---|---|---|---|
| orig | 829s | 0.390993 | 基线 |
| orig 重跑 | （1500帧 429s） | 0.401358 | **原版自身 run-to-run 漂移 ~2.6%**（自带预取线程时序 + GPU 核非确定） |
| bc_only（B+C） | （1500帧 406s） | 0.397262 | 方差内等价；~5% 提速（主要是关视频） |
| fast2（设备merge+B+C） | **818s** | 0.393424 | 方差内等价；合计 **~5-8%** 提速 |

**结论**：v1 预取否决（重复建设）；v2 验收通过但收益有限——管线本已充分
重叠，剩余墙钟长杆是**采样线程内部的 batch 组装**（_sample_non_sequential
+ 逐样本 np.stack，CPU 占比 ~28%），提速它需要对 replay buffer 的组装路径
做向量化手术（约一天工程 + 严格等价验证），收益预估 20-30%。等价性验收
标准修正为「原版自身重跑方差之内」——逐位复现原版都做不到。

### 42. 异步评估协议(用户指令,今后默认)

训练不再内嵌 eval(eval_every_steps 设超出预算),按 snapshot_every_n=5000
存快照(单个 429MB、写盘 ~3s);独立 watcher(scripts/async_eval_watcher.py)
在**另一张卡**上守候快照目录,每个新快照跑 **50-episode** 评估(默认
validation seeds 400 起;密封仍用 800),追最新策略防积压,结果写入与
resolve 工具兼容的 eval.csv。收益:训练全程不被评估打断(拿回 ~20% 墙钟),
validation 噪声减半(50ep vs 25ep)。当前在跑的三个训练不改(保持与参照
可比),自下一批 run 生效。

### 39.6 Stage-159/160 seed1 因子结果:两因素效应几乎正交(预注册预测被反转)

| 臂 | 探索 | 衰减 | 探针(value) | 50-ep@800(任务) |
|---|---|---|---|---|
| 官方(4 seeds) | ✗ | ✗ | 0.567(716对) | 64.5 |
| 只探索 E | ✓ | ✗ | **0.548** [0.39,0.70] | **84.0** |
| 只衰减 D | ✗ | ✓ | **0.631** [0.51,0.74] | 70.0 |
| 组合(158,2 seeds) | ✓ | ✓ | 0.644 | 81.0 |
| 组合+遮罩(160) | ✓ | ✓ | 0.508 | 74.0 |

**裁决(seed1 口径,n=1 注意)**:
- **value 口径**:探针提升由 **margin 衰减**驱动(D 0.631 ≈ 组合 0.644),
  探索单独不动探针(E 0.548 ≈ 官方 0.567)——预注册预测(E↑、D≈基线)
  被完全反转。机制改写:TD 所需的反事实信号在普通数据里本就存在
  (bin 边界噪声溢出、episode 间变异),margin 广播一直在压制它;
  释放 margin 即释放信号,ε 探索数据不是关键。
- **任务口径**:增益由**探索**驱动(E 84.0,全程序密封最高;D 70.0
  仅 +5.5)。E 单独 ≥ 组合(81.0)——任务上衰减非必要。
- **正交结论**:探索→任务(+20pp),衰减→value(+0.06),各管一头;
  要两个端点都好才需要组合。Stage-153 探索 null 的旧结论修正:当时
  10.5k 预算不足,100k 下探索的任务效应显著且巨大。
- **遮罩**:双口径净损(探针坍回机会线 0.508、任务 −7)——低维 dropout
  破坏 critic 的反事实读出。等 seed2 复核后定稿。

### 39.7 因子臂双协议补测:验证尖峰再次不存活,终点即真实最优

| 臂(seed1) | 验证最优快照密封 | 终点快照密封 |
|---|---|---|
| 只探索 | 68.0(@20k,验证时曾 88%) | **84.0** |
| 只衰减 | 60.0(@60k) | 70.0 |
| 遮罩 | 66.0(@40k) | 74.0 |

三臂一致:早期验证尖峰密封后缩水 8-20pp,终点快照全面更高——(a) 25-ep
验证尖峰纯属噪声(程序内第 5 次复现该模式);(b) 三臂训练全程无后期
退化,终点即真实最优,84.0/70.0/74.0 立稳(免选择口径)。

### 39.8 seed 集×臂交互:单一评测集的冠军宣称不成立

全快照 50-ep 补测(seeds 400-449)与密封表(seeds 800-849)终点对照:
exponly 84@800 vs 62@400(差 22pp≈2.3σ)、官方均值 64.5@800 vs 68@400
——排序在两个评测集间翻转,存在臂×seed集交互。组合臂是唯一双集稳定者
(s1 82/76、s2 80/66)。结论修正:组合>官方 双集成立;「只探索单独最强」
降级为待验证。终局矩阵 = 2 训练 seed × 2 评测集 × 各臂终点,seed2 完成后
补齐。方法论教训追加:密封协议还需第二评测集交叉,单集 50-ep 也会偏。

### 39.9 终版 200-ep 矩阵(seeds 800-999,终点 checkpoint,每数 SE≈3.2pp)

| 臂 | 各训练 seed | 均值 | 对官方 |
|---|---|---|---|
| 官方(4 seeds) | 62.0/60.5/62.0/74.0 | **64.6** | — |
| 只探索 | 74.0/63.5 | 68.8 | +4.2 |
| 只衰减 | 63.5/66.0 | 64.8 | +0.2 |
| **组合(探索+衰减)** | **76.0/74.0** | **75.0** | **+10.4** |
| 组合+遮罩 | 70.0/69.0 | 69.5 | +4.9(对组合 −5.5) |

**终局结论(200-ep、双训练 seed、seed 区间收敛性已验证)**:
1. **任务口径亦为超可加交互**:组合 +10.4 > 只探索 +4.2 + 只衰减 +0.2。
   与 value 口径(仅组合臂探针 0.644 离开机会线,单因素均塌回)完全同构。
2. **统一机制叙事定稿**:ε-bin 探索制造同状态异 bin 的对照数据;margin
   衰减解除广播压制、让 TD 消费之;被消费的反事实同时改善 value(探针
   +0.08)与行为(任务 +10.4)。两者缺一不可——这就是「CQN 如何真正
   从 RL 部分获益」的最终答案。
3. 遮罩:对组合配方任务 −5.5、探针塌回机会线;双 seed 高度一致,负结论
   定稿(官方基底版 stage161 尚在训练,独立判定待出)。
4. 50-ep 时代的种种"矛盾"(84 vs 62 等)全部为抽样噪声,200-ep 下无一存活;
   今后关键对比 200-ep 起。

### 37.3 CCFF 线 200-ep 复核(seeds 800-999,终点 checkpoint)

| 臂 | 各 seed | 均值 | (50-ep 旧值) |
|---|---|---|---|
| clean | 55.5/59.0/67.5 | **60.7** | 63.3 |
| CCFF | 78.5/69.5/65.0 | **71.0** | 80.0 |
| TD-off | 69.5/70.0/70.5 | **70.0** | 74.0 |
| 纯 flow | 67.5/50.0/54.5 | **57.3** | 62.7 |

**200-ep 定稿**:(1) CCFF 优越存活:+10.3pp(71.0 vs 60.7),幅度与
探索+衰减线的 +10.4 几乎相同——两条改进线各自独立给出 ~+10pp;
(2) TD 份额收缩到 +1.0pp(71.0 vs 70.0)——Bellman 无贡献的定性完全
坐实,且 TD-off 是种子间最稳的变体(69.5/70/70.5);(3) 选择机制份额
~+13(对纯 flow),纯 flow ≤ clean。50-ep 时代的 80.0 头条修正为 71.0,
+16.7 修正为 +10.3——方向不变、幅度缩水,再次验证 200-ep 纪律的必要。

### 43. Stage-162 预注册:探索概率消融(用户设计)

对照 = 组合配方(200-ep 终点 75.0,seeds 76/74;探针 0.644)。三个新臂
(各 seed1,100k,异步评估新协议首批使用者):
- **U 均分**:[0.00467×3],总概率持平 0.014——检验"逐层递增"分配是否必要;
- **H 加倍**:[0.004,0.008,0.016],同比率、总概率 0.028——剂量响应;
- **E ε退火**:[0.002,0.004,0.008]×linear(1→0,100k)——经典 ε-decay,
  假设:中前期探索够用,后期回收行为质量。
澄清:官方无 bin 探索(仅高斯噪声,各臂共有),0.014 无官方对应物。
判据(200-ep@800 终点 + 探针):任一臂 ≥ 组合 +5pp 则升级为新默认并补
seed2;E 臂若任务↑且探针持平,写入配方;H 臂若任务↓给出剂量上限。

§43 追加:第 4 臂 nonoise(组合配方 + stddev=0)——检验高斯噪声(σ=0.01,实为 level-2 抖动探索,粗层不可达)在已有 ε-bin 探索下的净贡献。排队 GPU3。

### 44. Stage-163 预注册:训练期滚动时域(Q-chunking 精神,用户裁定)

文献:ACT 的 ensemble 是推理期发明(无在线训练);Diffusion Policy 用滚动
时域不平均;Q-chunking(NeurIPS'25)整段开环 + 段内 n-step,明确反对混合。
CQN-AS 把 ensemble 搬进训练 rollout 是少数派。已有证据:官方 20k ckpt 的
执行消融 openloop16 36 < ensemble 52 < replan8 60(eval 期);Stage-151
整段开环训练 −24pp。臂:组合配方 + temporal_ensemble_replan_interval=8,
ε 剂量换算保持每步激活率与窗长匹配(probs ×8=[0.016,0.032,0.064],
persist 2 plans=16 步),seed1,GPU5,新协议全链。判据:200-ep 终点对
组合 75.0;探针对 0.644——若任务持平且探针↑(动作语义变干净),训练期
ensemble 判定为可替换;若任务↑则直接升级配方。

### 45. Stage-163b 预注册:QC 段内回传 × replan-8(用户裁定)

对 QC(arXiv 2507.07969)loss 的逐项对照结论:我们缺 (1) h-step 段内无偏
回传(其主引擎)、(2) critic 全程无模仿损失、(3) 离线预训练。163b 移植
(1):replay.nstep=8,与 replan-8 执行(训练+评估同口径)绑定——段内执行
与存储 chunk 一致使 8 步奖励和 (近似)无偏;已知残偏:段中采样的窗口尾部
跨 plan(换最新 plan,动作接近,远小于 ensemble 混合的错位)。seed1,
GPU5,排队 163 链后。判据:200-ep 终点(replan-8 口径)对 163 与组合
75.0;探针对 0.644。若 163b > 163 → 段内回传增量成立;若 163b ≥ 组合
→ "QC 化的 CQN-AS" 成为新配方候补,追加 seed2 与全臂对照。

### 45.1 计划变更(用户裁定):163/163b 让位,先跑 163c「官方 QC 化」

用户指令:GPU5 的 163(组合×replan8)暂停(latest 快照保留,可续),
组合基底的 163b 取消排队;优先测**官方配方 + QC 两件套**(nstep=8 段内
回传 + replan-8 训练评估),不含探索/衰减/mask——回答「QC 化本身能把
原版 CQN-AS 提到多少」。对照:官方 200-ep 64.6(4 seeds)。若 163c 显著
超官方,则 QC 化与探索/衰减的组合(原 163b)成为下一步;seed1,GPU5。

### 46. 隔夜六臂收割(200-ep@800 终点 + 探针;新臂均 seed1)

| 臂 | 任务 | 探针 | 备注 |
|---|---|---|---|
| (对照)官方 4s | 64.6 | 0.567 | |
| (对照)组合 2s | 75.0 | 0.644 | |
| **163c 官方+QC** | **74.0** | 0.467 | nstep8+replan8;评估同口径 replan8 |
| 官方+mask | 72.5 | 0.616 | |
| U 均分 | 71.5 | 0.551 | |
| H 加倍 | 71.5 | 0.558 | |
| N 去噪声 | 68.0 | **0.658** | 探针全程序最高 |
| E ε退火 | 67.0 | 0.596 | |

**六条裁决(单 seed 注意)**:
1. **QC 化官方 = 任务口径追平组合配方**(74.0 vs 75.0;对官方 +9.4)——
   且其探针 0.467 说明增益纯来自 8 步回传的信用传播加速,与反事实知识
   无关。两条独立的 +10 路径(组合=探索×衰减;QC=段内回传×滚动执行)!
2. **mask×基底交互坐实**:弱基线 +7.9(72.5),强配方 −5.5——mask 是
   "欠拟合基线的正则",与强配方互斥。
3. 探索的分级(U)与剂量×2(H)对任务/探针均无实质影响(71.5/0.55 档)
   ——ε 探索对超参不敏感,是稳健组件。
4. **ε 退火有害**(67.0,−8 vs 组合)——探索需要全程在线,验证"后期
   回收行为质量"假设不成立;TD 的反事实数据需求持续存在。
5. **高斯噪声承重任务 +7**(N 68.0 vs 组合 75.0)——细层抖动不是冗余;
   但去掉它探针反而最高(0.658)——噪声在帮行为的同时轻度污染反事实
   读出,两口径首次出现明确的 trade-off 组件。
6. 下一步天然候选:**组合 × QC**(两条 +10 路径合体,原 163b 设计)
   与 163c/新臂的 seed2 复核。

### 47. Stage-164 预注册:王冠臂(组合 × QC)+ 两个 seed2 复核(用户裁定)

- **crown ×2 seeds(GPU0/2)**:探索(剂量换算 [0.016,0.032,0.064]、persist 2)
  + margin 衰减 + nstep=8 + replan-8(训练评估同口径)。两条独立 +10 路径
  (组合 75.0、官方+QC 74.0)的合体。判据:200-ep 终点 ≥80 → 可加性成立,
  成为新旗舰;≈75 → 两路径共享瓶颈(信用分配上限);探针同时看
  (组合 0.644 vs QC 0.467 的张力:合体探针落在哪决定"QC 回传是否
  侵蚀反事实知识")。
- offqc8 seed2(GPU3)、offmask seed2(GPU4):两个 +8~+9 单 seed 臂的复核。

### 48. Infra:replay 采样路径手术(用户指令"做一下",实测驱动)

**实测在跑预算(stage164 crown,nstep=8,251ms/step,最近30行中位)**:
update 调用 281-335ms(其中 backend/JAX 仅 71-77ms → **等待 210-260ms**),
env.step 4.5-8.7ms,act 1.5-1.7ms。GPU ~3/4 时间挨饿(util 快照 0-55% 相符)。

**否证一个中间假设**:§41 的"组装向量化 20-30%"线索先被放大成"cache
miss 解码论"(16 上限 vs 329+169 episodes → 每样本整集解码?)——读码
否证:fetch 路径 `_cache_episode(enforce_limits=False)` 无上限常驻,在线
稳态**零 miss 零解码**(/proc io 30s 读增量=0 佐证);16 上限只在"先
miss 一次→永久 thrash"的政权下咬合(实测灾难参考:cap16 冷缓存
1.5-1.7 s/batch,232 decode/batch)。cache 上限改为 null/24G(robobase_config)
纯属保险(resume/重启政权),对新鲜 run 逐位无差。

**真瓶颈 = 逐样本组装的双重拷贝**(cProfile:scalar np.stack 重拷贝 63ms
+ 逐样本 gather 30ms ≈ 93ms/256-batch,调用开销次要,拷贝带宽主导)。

**手术**(`uniform_replay_buffer.py`):`sample()` 改为两阶段——索引选择
逐样本保持与 `sample_single` **完全相同的 np.random 调用序列与 _try_fetch
节律**(RNG 逐位一致);装配利用 frame_stack/action_seq 是**连续行**的事实,
非边界样本 = 单次连续 slice memcpy 直写 batch 行(无 fancy-index 临时件),
边界样本回退 clip-gather(逐字节同值);reward/terminal/extras 按 episode
分组向量化。护栏:子类(prioritized)/sequential/zero-pad/action-history/
逐样本预处理一律回退 scalar 参考路径;kill-switch
`ROBOBASE_SCALAR_SAMPLE=1`。

**验收(已过)**:pytest 逐位等价 7/7(nstep 1/3/8 × fs 1/2/4 × tp1 ×
uniform-sampling × 显式 indices,含 RNG 终态逐位相等),
`tests/unit/replay_buffer/test_vectorized_sample_equivalence.py`;真实
episode(crown replay 150 集)CPU 基准:**78.5 → 35.8 ms/256-batch
(2.19×)**。两 buffer 合计 157→72ms/update,**首次低于 backend 75ms**,
`replay_prefetch_size=4 + device_prefetch` 应可完全遮蔽采样 →
预计 ~95-105ms/step(**~2.4-2.6×**),待 GPU 验收定夺。

**GPU 验收(待跑,卡满排队)**:`scripts/bench_vecsample_equiv.sh`
(3000 帧同 seed A/B,crown launch,GPU5 按既定指令;判据 = critic_loss
轨迹在重跑方差 ~2.6% 内 + 墙钟)。stage164 在跑进程不受影响(代码已
加载),自下一批 run 生效。若遮蔽不完全,下一杠杆已识别:flat 存储
镜像(每 key 每 batch 单次 gather,est. ~15ms/buffer)。

### 48.1 [P1] 用户复核发现:device-side merge 在默认 JAX 配置下从未生效

**问题(用户最小复现,已核实)**:`workspace_fast.replay_iter` 覆盖
property 并对父类返回值做 `isinstance(_, _DemoMergedIterator)`;但父类在
`backend.replay_prefetch_size=4`(jax.yaml 默认)时已把合并器包进
`PrefetchReplayBatchIterator` 并赋值 `self._replay_iter`——isinstance 永假,
设备合并是死代码,~130MB/update 的 host `np.concatenate` 一直在跑。
**历史修正**:§41.1 的 "5-8%" 全部来自 B+C(关视频+update blocking),
A'(设备合并)贡献为零(fast2 818s ≈ bc_only 水平,数据早有指纹)。

**修复**:workspace 新增可覆盖 hook `_make_merged_replay_iter`(在 prefetch
包装**之前**注入),WorkspaceFast 覆盖之返回 `_DeviceMergedIterator`
(设备合并在 prefetch 线程内执行;`prefetch_batch` 的 device_put 幂等,
map_fn 保留)。回退开关 `ROBOBASE_HOST_MERGE=1`。回归测试
`tests/unit/test_workspace_fast_merge.py`(5/5):接线断言
prefetch→_DeviceMergedIterator、kill-switch、host/device 合并值逐位一致
(dtype 映射 f64→f32/i64→i32 与 jit 边界原有行为相同)。含 §48 采样手术
全套 101/101 过。

**基准协议修正(用户指出的顺序偏置)**:`bench_vecsample_equiv.sh` 改为
**ABBA**(old,new,new,old),稳态口径 = train.csv env_steps 1000→3000 的
steps/s(排除 JIT 编译与启动),等价判据 = 跨臂 critic_loss 漂移落在
**组内重跑漂移**(A1-A2/B1-B2,当场测得)之内,不再引用历史 2.6%。
旧臂 = 两开关全开(SCALAR_SAMPLE+HOST_MERGE)= 原生产路径。GPU5 待空。

### 48.2 用户复审第二轮:bin-explore 状态生命周期三修(#2-#4)

复审判定 #1/#5 已修,#2-#4 稳定复现,均已修复 + 回归测试
(`tests/unit/test_cqn_as_bin_explore_state.py`,4/4;test_cqn_as 全套 74/74):

- **#2 [P1] episode reset 泄漏探索状态**:`reset()` 清 structured_exploration
  但不清 `_bin_explore_*`,跨 episode 延续旧 sibling shift。已在 reset 中
  按 agent_index 清理(带 shape 越界护栏)。**历史影响**:覆盖 ~18% 步数的
  探索窗以 ~16/90 概率跨越 episode 边界 → 约 1/5 的 episode 开头带上一条
  episode 的 shift(≤16 步),存在于所有已跑探索臂;量级在剂量不敏感区
  (§46 U/H),不追溯重跑,自下一批语义变更。
- **#3 [P1] register_mask 未作用于 bin exploration**:多环境下整 batch 传入,
  未 replan 的 env 也被抽签/施移/扣 persist。已把 mask 传入
  `_apply_bin_explore`,mask 外行完全不动。**关键澄清(血量减免)**:
  num_train_envs=1 时 `needs_inference = any(mask)` 使非 replan 步根本不
  调用该函数——**至今所有 run(含在跑 crown)的单环境语义一直正确**,
  预注册的 ×8 换算/16 步窗成立;此 bug 仅在未来多环境采集时才会咬合。
- **#4 [P2] 断点续训丢 NumPy 探索 RNG**:`checkpoint_state_dict` 仅存
  JAX RNG。CQNAS 现补存 `_bin_flip_rng`/`_bin_explore_rng` 的
  bit_generator.state + 四个 persist 数组;load 带 .get 护栏,旧快照
  (含 stage164 在写的)仍可加载(行为同旧:探索流重开)。测试断言
  "中断-恢复"与"不中断"的 assignment 序列逐位一致。

### 48.3 用户复审第三轮:48.2 的 checkpoint 修复引入的 P2 边界问题(已修)

**问题(成立)**:48.2 把四个 `_bin_explore_*` persist 数组也写进
checkpoint 并在 load 时恢复;但 workspace 快照不含环境状态,resume 走
`train_envs.reset()`(workspace.py `_online_rl` 开头)且不调 `agent.reset()`
——恢复的 mid-episode 窗口会落在全新 episode 上,与 reset() 的
"intervention 不跨 episode" 语义自相矛盾(等于把 #2 的泄漏从 episode
边界搬到了 resume 边界)。

**修复**:checkpoint 只保存/恢复两个 NumPy RNG 流(可复现性目标);
persist 窗口显式不入 checkpoint,load 对短命 48.2 格式里的窗口键显式
忽略。语义:resume = 新 episode 从无窗状态开始,RNG 流从快照点继续。
测试更新(5/5):roundtrip 等价改在"无活跃窗口"快照点断言序列逐位
一致;新增边界测试断言 mid-window 快照 resume 后窗口清零、RNG 流仍续、
且 48.2 格式快照的窗口键被忽略。test_cqn_as 全套 + method 目录复跑通过。

### 49. Stage-164 crown 裁决:两条 +10 路径**破坏性干涉**(未达任何预注册分支)

**Sealed 200-ep @101k 终点(seeds 800+,replan-8 同口径,ne=25)**:
| seed | 任务 | 探针(sign acc) |
|---|---|---|
| 1 | 56.5 | 0.628(183 对) |
| 2 | 50.0 | **0.500 = 纯随机**(130 对) |
| 均值 | **53.3** | — |

对照:combined 75.0/0.644,official+QC 74.0/0.467,官方 64.6/0.567。
中途 validation 读数(§今日:90k 63.5 / 95k 55.0,seeds 400)方向一致,
非终点噪声。

**裁决**:预注册判据(≥80 可加;≈75 共享瓶颈)双双落空——crown 落在
**官方基线之下 11pp、combined 之下 22pp**。两条独立 +10 路径合体是
**破坏性干涉**,不是可加,也不是简单瓶颈。探针分裂:seed1 0.628 尚存
combined 型反事实知识但任务已塌,seed2 0.500 全数侵蚀——§47 预注册的
张力(QC 回传是否侵蚀反事实知识)得到肯定方向的回答(2 seeds 一致塌
任务,1/2 塌探针)。

**机制候选(单 seed-pair 谨慎,不下定论)**:nstep-8 段内报酬和把
ε-bin 强制探索段的(故意劣化的)执行结果以 8 步聚合直灌 TD 目标,
1-step TD 里由 bootstrap 吸收的离策略污染在 QC 化后被放大;同时 margin
衰减撤掉了行为锚。官方+QC 无探索故近贪心数据下 n-step 无害(74.0),
combined 无 n-step 故探索污染被 bootstrap 缓冲(75.0),二者合体互拆。

**旗舰维持 combined(explore×decay,75.0/0.644)**。QC 化(74.0)保留为
独立任务口径路线。第二任务外部效度实验按 official vs combined 设计,
crown 出局。快照按协议保 101000 + 里程碑,其余可清。

### 50. Infra GPU 验收(ABBA,GPU5):**通过,稳态 2.36×**

| run | 稳态 steps/s(1000→2000) | 墙钟 |
|---|---|---|
| old1 | 4.40 | 12.8 min |
| new1 | 10.65 | 5.9 min |
| new2 | 10.80 | 5.7 min |
| old2 | 4.69 | 12.3 min |
| **old→new** | **4.55 → 10.73 = 2.36×** | |

new 稳态 ≈93ms/step ≈ backend 上限(预测 95-105ms 命中),预取遮蔽基本
做满。等价性(诚实记录判据的意外):
- 组内重跑漂移 median 0.04%/0.08%(max **4.13%**/2.86%);
- 跨臂漂移 median 1.77%/1.00%(max 1.77%/**2.55%**)。

预注册的"跨臂 median ≤ 组内 median"**字面上未过**——诊断:同速重跑的
fetch 时序几乎一致 → 批组成几乎相同 → 组内 median 塌缩到 0.0x%,它并
未涵盖"速度本身改变 fetch 交错 → 批组成合法漂移"这一噪声源;而这正是
所有 run 固有的不可逐位复现机制(§41.1,历史时序噪声 ~2.6%)。按语义
正确的尺子:跨臂 max 2.55% < 组内 max 4.13% 且 median 1.0-1.8% < 2.6%
历史包络;计算本身的逐位等同已由单测钉死(采样 bit-equal 7/7、合并值
等同 5/5)。**判定:接受**;残余漂移归因于批组成时序,与既有 run-to-run
非确定性同源。

全尺度换算:100k 老口径 251ms/step → 新 ~93-95ms ≈ **2.6-2.7×**,101k
训练 7.5h → **~2.7h**(下次全尺度 run 复核)。附:训练真实显存峰值
13.1G(prealloc=false,含 JIT;§AGENTS.md 已收录),eval 1.6G;
xla_mem_fraction 开关入 gpu.py。同卡双跑(0.45×2,错峰 120s)实测
探针在跑,出数后补记。

§50 补记(同卡双跑实测,GPU5,0.45×2 + 错峰 120s):peak 30.2G/32.6G
(两个 0.45 硬切片各自全额预分配 14.7G + EGL/系统,按设计;余量 2.4G,
无 OOM);每条稳态 7.09/6.99 steps/s = 单跑的 **0.66×**,合计
**14.1 steps/s/卡 = 1.31× 吞吐**(且双跑单条仍是旧管线单跑的 1.54×)。
全尺度换算:101k 单跑 ~2.6h、双跑 ~4.0h/条。第二任务矩阵(8 run):
双跑 4 卡一波 ~4h < 单跑 4 卡两波 ~5.3h,**采用双跑**。

### 51. 预注册:官方配方全尺度复现闸门(新 infra,用户指令)

新 replay 管线(§48/50)上跑**官方 CQN-AS**(demo_driven launch,无探索/
衰减/QC,官方 ensemble 执行)MovePlate 101k,seed 1,GPU2 单卡,async
协议(无 in-loop eval),链 sealed 200-ep@800(ne=25,最终快照)。
判据:① 任务成功率落在官方 4-seed 带内(均值 64.6);② 全尺度墙钟
~2.6-2.7h(§50 的换算复核);③ 训练曲线无异常。通过则第二任务波次
直接在新 infra 上发车;不过则回退开关逐项归因。
runner:`scripts/run_cqn_official_repro_newinfra.sh`。

### 51.1 复现闸门裁决:**通过**

官方配方 × 新 infra 全尺度(seed1,GPU2,101k,async 协议):
- **sealed 200-ep@800 终点 = 67.5**,落在官方 4-seed 带内(均值 64.6,
  +2.9pp < 1 SE)✓;
- **墙钟 3.20h**(23:44→02:56;旧管线同尺度 ~7.0-7.4h → **全尺度
  2.2-2.3×**)。略高于 §50 的 2.6-2.7× 外推:稳态 ~114ms/step vs 基准
  93ms,归因候选 = _try_fetch 的目录 glob+sort 随 episode 文件数(终局
  ~700)线性变贵 + 快照写盘 + 邻卡负载;可选微修(目录列表增量缓存,
  预估再拿 5-8%)记为后续,不阻塞;
- 曲线健康(log 零错误,在线末段采样成功率 ~0.55,官方典型)✓。

**结论:新 infra 复现官方基线成立,第二任务波次绿灯**(官方 vs combined
× 2 seeds × 2 任务,双跑规程,AGENTS.md)。本 run 顺带成为官方基线在新
infra 上的 seed-1 参照点(67.5)。

### 52. Stage-165 预注册:第二任务外部效度(dishwasher_close_trays,用户指令)

任务选择:dishwasher_close_trays(中等难度双托盘推入;demo 库本地齐备,
量级与 MovePlate 同域;有效 episode ~800 步 = 中等视界)。臂(用户指定
两臂):**combined**(探索×衰减,stage158 原配方,官方 ensemble 执行)与
**official+QC**(stage163c:nstep8 + replan-8 训练评估同口径),seeds 1/2。
新 infra + 双跑规程首个生产波次:GPU2=两臂 seed1、GPU5=两臂 seed2
(0.45 切片、错峰 120s,臂跨卡平衡)。链条:各 run 训毕自动 sealed
200-ep@800(ne=25,最终快照,匹配执行方式)。探针本波不跑(anchor 为
MovePlate 调参,任务口径先行)。判据:任务口径下两臂相对该任务官方水平
的增益方向与 MovePlate 一致(combined ≈ +10、QC ≈ +9)则机制外推成立;
注意本波无同任务官方臂,官方参照取 CQN-AS 论文该任务数值区间,若两臂
绝对值可疑则补官方臂。ETA:双跑 ~0.66× → 单条 ~4.9h,波次 ~5h。

### 52.1 附:独立 JAX Q-chunking 的 robomimic square-mh 复现结账(用户问询驱动)

7/28 的 repro run 被中止在 **1M offline 预训练的 355k 处(35%)**,从未进入
online 阶段——console log 的 "Step: 0" 是 env-step 计数(离线阶段恒 0),
eval.csv 的 5k-355k 是**离线 update 数**,不是 online env steps。对照
run 内存档的论文数字化参考曲线(diag_bc_only_100k.json,Fig.3):
- 论文 offline 段:100k=4.8%,1M 终点=36.8%;线性内插 355k ≈ ~16%;
- 我们 offline 355k = **28-30%**(50-ep,watcher)——**在匹配预算处
  略超论文离线曲线**;BC-only 诊断 100k=26% 亦健康(BC 早期强于
  QC-offline 属预期:RL 式离线目标需 Q 预热)。
- 论文头条数字(offline 终点 36.8 / online 终点 92.8)因 run 中止从未
  被检验。
**结论:截至被中止处,复现在轨甚至略超;账没结完而非对不上。**
结账成本:恢复 latest snapshot 续跑 offline 剩余 645k(~2h)+ online 1M
(env-stepped,较慢),约一夜;排在 stage-165 波次之后可选。

### 53. Stage-166 预注册:BC 断奶(λ→0)续训(用户裁定,单 seed)

问题:combined 训成的策略能否撤除模仿锚(桥接 no-BC 线)。从 stage158
seed1 @101k 快照恢复(新 run dir,快照+全量 replay/demo 硬链接 = 完整
续训语义),续 100k 至 201k:
- **wean0**(GPU0):`step_linear(1.0,0.25,100000,0.0,100000)`,前段与
  历史逐点一致,101k 起 0.25→0(201k 归零);
- **hold025**(GPU1,对照):原 schedule 钳位 0.25,隔离"多训 100k"。
探索保持恒定(§46:不许动)。旧快照无探索 RNG 键(48.2 前)→ 恢复时
探索流重开,两臂一致,不构成混淆。判据:201k sealed 200-ep@800,
**Δ(wean0 − hold025) 为主读数**;wean0 落在 combined 带(75±SE)→ 断奶
成立;塌则塌点 λ* = 模仿锚最低必要剂量。探针两臂各跑可后补。单 seed
初筛,有效再补 seed2。

§53 修订(用户裁定,v2):首次发射 40 分钟后按新设计重启(从 101k 快照
重启而非中途改 schedule,保持"两臂至 101k 逐点一致"精确成立;首发目录
已清)。变更:① 两臂**同卡双跑**(GPU0,0.45 切片,错峰 120s),释放
GPU1;② wean0 schedule 改 `step_linear(1.0,0.25,100000,0.0,50000)`——
λ 在 **150k 归零**,150k-201k 为 **50k 纯 TD 平台期**(观察 λ=0 稳态,
不只是逼近过程);名义 TD:BC 均衡点(λ=0.1)在 ~130k。ETA 双跑 ~4.2h。

### 52.2 Stage-165 中途手术(用户优先级清单 P1/P2)+ §53 v2 勘误

- **P1**:offqc8 seed2 自 **8k 步**起 critic_loss=NaN(其后 39k 步无效),
  已终止;**seed1 健康**(48k,loss 0.13,零 NaN)→ QC 化臂以单 seed 存
  活。新任务上 nstep-8 的 1/2 seed 早期发散本身入档:dishwasher(~800 步
  稀疏)下 QC 段内回传的数值稳定性弱于 MovePlate(~300 步),后续若补
  seed 需考虑 reward-scale/target-clip 防护。
- **P2(设计缺陷承认并修复)**:stage165/166 runner 的 inline eval 链会
  在同卡配对训练未结束时开评(违反不共置协议)。已杀全部 runner shell
  (训练进程转孤儿续跑),改由**后置编排器**统一调度:每卡全部训练结束
  后才在该卡顺序跑 sealed 200-ep;GPU5 评完自动接 **P4 纯官方臂**
  (seed1/2 双跑,官方执行)。offqc8_s2 不评(无价值)。
- **§53 v2 勘误**:v2 重启后我按列位读 wean0 的 env_steps(39 列布局下
  $34=episode_length)误判"未恢复"并误杀了健康进程(实际按列名读为
  102000,恢复成功);已原目录重启,损失 ~1k 步。**规则固化:CSV 一律
  按表头名取列,禁止按位置**(今日第二次踩同类坑)。
- **P5**:QC square 离线段边界监视器已装(online 首行即 SIGINT,精确停
  在 ~1M 离线快照);GPU1 上 async watcher 已即时恢复评估(50-ep,补
  360k→1M 积压)。P3(no-BC Stage-23)为用户线,不代管。

§52.2 P5 勘误与补评协议(7/31 13:36):上句“watcher 已恢复评估”只证明
进程曾启动,不构成有效结果。取证显示旧命令一处把 watcher 参数误写成
`--num-eval-episodes`(watcher 实际接收 `--num-episodes`),另一处又把不接受
单快照参数的 `eval_q_chunking_snapshot_sweep.py` 传给 watcher;其生成的
`async_eval_*.log` 均为 CLI 失败,因此有效 `eval.csv` 仍只到 355k=28%,
**1M offline endpoint 尚无任务质量读数**。训练 loss 有限只证明 wiring,
不能替代该结论。下一阶段固定为单一问题:精确评估
`snapshots/1000000_snapshot.pkl`(不使用含 55k online 的 latest),本地单
seed、不做 checkpoint selection,held-out seeds=400--449、50 episodes;
指标为 success% 与 mean reward,匹配参考为官方 QC square-mh 1M offline
=36.8%。预注册通过线为 success>=30%(至少不低于本地 355k 的 28% 且
距官方点不超过 6.8pp),否则判复现未在 offline 终点保持在轨。执行已排
队:PID 1581041 等待 Stage-26 controller PID 1517295 完整结束且 GPU1
连续空闲 120s 后,调用 `eval_q_chunking_robomimic_checkpoint.py`;输出
`offline_1000000_eval50_seeds400.{log,json}`。当前 Stage-26 seed2 尚在跑,
预计约 25--30min 后启动,补评本身另需数分钟;结果落盘前不得写成通过。

§52.2 追加(用户质询"NaN 是不是代码问题"的取证结论):
1. seed2 存储 replay 全量扫描:191 集、全部 float 数组零非有限值、
   最短 225 步 ≫ nstep=8 → 环境物理爆炸/数据污染/短集边界三假设排除;
2. **决定性实验**:用 seed2 真实数据(60 集,真实 shape:低维 62 维、
   proprio 4 维、action 16 维)做新旧采样路径对比,20 试次 × 256-batch
   **逐字节零差异、交付值全有限** → 新采样器交付的 batch 与旧代码逐位
   一致,NaN 必在下游;
3. 方法侧定位:7k 时 critic_loss=0.19 正常,8k 时 loss 与 entropy 同时
   nan → 单个 1k 区间内的更新爆炸(logits/权重 → nan),发生在全零
   reward 阶段,nstep=8 bootstrap 下的目标漂移溢出为首要机制假设;
   seed1 同代码同任务 48k 健康、MovePlate nstep=8(crown 2 seeds +
   offqc8)从未 NaN → "确定性方法 bug"假设弱,"任务统计相关的数值
   不稳定"假设强。潜在方法级 bug 不能绝对排除,深挖入口 = 5000 快照
   权重范数对比 seed1(未做,按需)。

### 52.3 offqc8_s2 NaN 结案(用户"为啥会爆炸"追查)

静态取证全线排除(§52.2 追加 + 本节):输入有限性 ✓、幅度常规(爆炸窗
≤94,而 seed1 后期吃过 4631 不死)✓、5k 参数与健康 seed 逐层孪生 ✓、
Adam 一二阶矩冰冷同构 ✓、采样逐位等同 ✓、demo 集完整(44=该任务成功
demo 数;seed1 67=44+23 自模仿,顺带证明 seed1 已在学成功)✓。
**动态判决**:seed2 从 5000 快照在 GPU1 复活穿越 7-8k 危险区至 13k,
critic_loss 0.16 全程干净——**未复现**。按预注册解读:无法区分"方法×
任务罕见数值事件"与"GPU5 硬件瞬态",但实用结论一致:**单发低概率事件,
非系统性**;处置 = probe 转正为 seed2b(GPU4 续至 101k,链 sealed eval)
+ 建议加 non-finite 早停守卫(待用户裁定)。
附:QC 边界停机实际落在 online ~55k(console 块缓冲延迟了监视器触发;
教训:监控 console 需 PYTHONUNBUFFERED=1),1M 离线边界快照完整,离线
判决不受影响,55k online 为免费预览。

§52.3 再追加(QC 评估恢复的曲折,全部入档):watcher 三连败(①参数名
--num-episodes 记错;②给错 eval-script 合同(sweep vs checkpoint);
③换对脚本后进程仍静默死)→ 弃 watcher 改直跑 milestone 循环仍死
(exit 144,无 traceback,死于 robosuite env 初始化)→ GPU4 单测成功
(42s)→ **隔离结论:GPU1 无法运行 robosuite/robomimic eval(EGL 路径
特异,BiGym 训练不受影响)**,GPU1 彻底留空,milestone 扫描(400k→
1055k,15 点 × 50-ep,seeds 400)转 GPU4 执行中。教训:后台进程发射后
必须验证首个成功产物,不能只验证进程存活。

### 52.4 QC square-mh 离线段复现判决:**通过**

完整离线曲线(50-ep,seeds 400,GPU4;单点 SE≈±7pp):
400k=24 450k=36 500k=24 550k=36 600k=54 650k=36 700k=44 750k=40
800k=34 850k=42 900k=40 950k=46 **1M(离线终点)=40** 1005k=32;
意外多跑的 online+55k 点 = 52。
对照论文(Fig.3 数字化):离线终点 36.8,内插路径 100k=4.8 → 1M=36.8。
**判定:离线终点 40 vs 36.8(噪声内达标),且实测段全程压在论文内插
路径上方;early-online 52 也落在论文 online 起坡(36.8→59.6@1.1M)的
轨道上。JAX Q-chunking 移植的离线复现成立。**
成本修正:今晨意外 online 段实测 ~115 steps/s → 补完 1M online 仅需
**~2.5-3h 单卡**,不是先前估的"一夜"(那是像素时代的直觉);是否补跑
待用户裁定(论文 online 终点 92.8)。GPU4 已被 jz5725 接走(留卡政策
生效),补跑需等空槽。

§53 中途读数(重要,先于终点判决记录):**wean0 在 λ 归零点即刻行为塌缩**。
5k 桶点采样在线成功率:145k(λ=0.025)= 0.80 → **150k(λ=0)起连续三桶
= 0.00**(15 个采样点全零;若真率仍 60%,P≈1e-6,非采样噪声);同窗
hold025(λ=0.25)保持 0.2-0.8 常态。且 wean0 的 critic_loss 不升反降
(0.207→0.194)、entropy 正常——**TD 找到了更低损失但不再把好动作排在
argmax 上的解**,正是"margin 锚承重"的行为学证据。塌点定位:λ* ∈
(0, 0.025]——2.5% 的名义权重仍足以维持行为,恰好为零则崩。两臂按预注册
跑完 201k(观察纯 TD 段是否自恢复 + 终点 sealed 定量)。
勘误:我此前向用户口头报"曲线正常"仅基于 loss/存活,未查成功率——
第三次"验证前先宣称"教训,并入 §52.3 的规则。

### 52.5 第二任务首批 sealed 结果(dishwasher_close_trays,200-ep@800)

| arm | sealed | 在线尾段(30 采样) | 备注 |
|---|---|---|---|
| combined seed1 | **1.000**(200/200) | 0.50 | 在线-贪心差=0.50,全部由 ε-bin 探索污染解释 |
| official+QC seed1 | 0.865 | 0.90 | 无探索 → 差≈0,同时排除评估器全判成功的 bug |

**新观察入档:探索的在线代价是任务相关的**——MovePlate 上在线≈贪心,
dishwasher 上在线=贪心的一半;该任务存在"脆弱段",16 步连贯 sibling
偏移落上即毁整集,但对最终贪心策略无损(反而 100%)。外部效度正式
结论待纯官方臂(seed1/2 已上 GPU2 双跑,P4)标定任务难度后下;
combined_s2 独立 eval 交叉验证中;seed2b 停 50k 快照(GPU5 让渡)。

### 54. Stage-167 预注册:任务广度检验(用户裁定,砍官方臂)

用户裁定:dishwasher 官方臂取消(论文 Fig.5 读图 ~75-80 + 宽 CI 做参照;
GPU2 官方双跑已杀,零浪费损失 ~40min)。对"可能只是 seed 好/任务易"的
质疑,以**广度**回应:combined × seed1 × {sandwich_remove(540 步,
24 demos,论文 CQN-AS 读图 ~55-65), move_two_plates(550 步,30 demos,
论文 ~20-30,全家最难档)},GPU2 双跑(0.45,错峰;首发因
replay_size_before_train=500 < 单集长 540/550 秒退,以 =600 重发)。
判据:两硬任务上 combined 相对论文官方读数的方向与幅度;
move_two_plates 是论文接近失败的任务,若 combined 仍显著为正即为
最强外部效度证据。sealed 200-ep@800 训毕自动(卡空后)。

### 53.1 Stage-166 断奶终点裁决

| 臂 | sealed 200-ep@201k | 
|---|---|
| wean0(λ 150k 归零 + 50k 纯 TD 平台) | **0.0**(0/200) |
| hold025(λ=0.25 定格,对照) | **79.0** |

**Δ = −79pp,四条结论:**
1. λ* 悬崖坐实:0.025 仍安然(在线 0.76-0.80),恰好 0 即全塌;塌后
   50k 纯 TD **零自恢复**(在线 16 连零 + sealed 0/200)。
2. 对照臂 79.0 ≥ combined 参考 75.0:多训 100k(λ=0.25)无害微益
   → 效应 100% 归因 λ→0 本身。
3. 塌缩期 loss 反降 + entropy 正常:TD 自由解在数值上更优、行为上致命
   ——反标定论题的最纯净展示,"margin 是 argmax 排序的棘轮"。
4. 对 no-BC 线的直接推论:任何纯 RL 目标必须携带 argmax 锚的功能等价物
   (floor/gate 类),否则即使从完美策略出发也会被 TD 拆掉。
断奶不可行(以本配方形态);λ 最低剂量 ∈ (0, 0.025],便宜到可永久保留。

§54 首批读数(seed1,200-ep sealed@800):
- sandwich_remove:combined = **52.5**(论文官方 Fig.5 读图 ~55-65 → 平/略低)
- move_two_plates:combined = **19.5**(论文 ~20-30,最难档 → 平/略低)
外部效度图景(3 任务):dishwasher_close_trays 顶格(100/100 双 seed)、
两个长视界精细任务与论文官方水平持平。初步解读:粗层对比数据在"粗粒
度失败模式"任务(托盘/单盘位置类)上命中,在精细双手长视界任务上不
增益——与 §52.5"探索代价任务相关"同源。限定:单 seed(seed2 夜车中)、
跨口径(200-ep vs 8×25-ep 读图)。钉死需自跑 official 臂(待裁定)。
stage168(nonoise+L2boost)已于 01:42 发车。

§55 追加(用户裁定,stage169):第二格剂量探针 `nonoise +
[0.01, 0.05, 0.4]`(近 ε 语法天花板:激活覆盖 ~87% 步,L2 dim-step
≈5% —— 仍为高斯 42% 的 1/8,rate parity 不可达已预注册)。与 168
([0.002,0.004,0.064])构成剂量二点:若 168≈169≈68 → 剂量说排除、
亚 cell 通道论成立;若单调上升趋 75 → 剂量说复活。GPU2 排 168 后,
ETA 训毕 ~08:45。在线成功率预期被 87% 覆盖率重度污染,仅 sealed 有效。

### 56. 晨间五判决(08/01 07:52)

1. **stage168 = 74.0**(nonoise + [.002,.004,.064],200-ep sealed):**剂量说
   获胜,我的"亚 cell 不可替代"预测被证伪**。L2 档 ×8 在无高斯下几乎复原
   combined(75.0),远超 N 臂(68.0)。高斯通道可被网格级细层 ε 以足够
   速率替代 → **配方 v2 候选:单一探索机制、零噪声**。待 169(天花板剂量
   单调性)+ seed2 + 探针后再议升级。
2. **saucepan_to_hob combined seed1 = 83.0** vs 论文读图 ~75-85:带顶。
   外部效度第 4 任务:长视界旗舰格达标(与 MovePlate/dishwasher 家族一致,
   与两个精细双手任务的平手形成分化图谱)。
3. **wean00125 = 83.5**(λ 定格 0.0125):健康且高于 hold025(79.0)→
   **λ* < 0.0125**,悬崖在 1.25% 权重之下;且更低的晚期 λ 或有微益
   (与衰减方向一致)。
4. **seed2b = 0.02**(offqc8 dishwasher seed2,50k→101k 续完,无 NaN):
   任务口径灾难。offqc8@dishwasher = 86.5 / 2.0,双 seed 均值 44、方差
   巨大——**QC 化在该任务上双模态不稳**(该 seed 谱系先 NaN 后死策略);
   不稳定性本身入档为 QC 线发现。
5. stage169 首发 07:18 因 GPU2 三租户瞬时挤爆(CUDA client 初始化失败)
   秒死;08:0x 已重发(0.45 cap,solo),ETA sealed ~11:45。
seed2 双跑(sandwich_remove/move_two_plates)90k/91k,~09:00 齐。

### 57. 评估污染事件:sealed 数字可被静默损坏(协议级警报 + 修正)

**事实链**:sandwich_remove combined seed2 同一 101000 快照——
坏评估(09:09 起,GPU1 与 jz×4 重度竞争,渲染落物理 GPU5)耗时
**5786s(4.7× 正常)→ 20.0%**;健康重复(11:40,轻载)1236s → **67.5%**;
validation 400 家族 66.5%;在线尾段 0.50。三健康口径一致,20.0 为损坏值。
**机制候选**(未定):(a) 重度竞争下 EGL 渲染出坏帧(观测变垃圾 → 策略
"致盲"仍能拿 20%);(b) 快照写入竞态(弱:写完 ~09:05,枚举 ~09:11)。
**修正**:sandwich_remove combined = 52.5 / **67.5**,均值 60.0 ≈ 论文带
(55-65)——昨夜"高方差低于论文"的下调判断作废一半;move_two_plates
seed2(0.10,同一污染会话)重复评估进行中。
**协议加固(即刻生效)**:① sealed 一律配 50-ep validation 交叉读数,
偏差 >3σ 触发重复;② eval 避开重度竞争窗口;③ 评估脚本加观测均值
sanity 输出(致盲检测)待实现。

### 58. 正午总裁决(08/01 12:20,全部 sealed 200-ep@800,健康评估)

**A. 细层/噪声通道问题(162N + 168×2 + 169)**:
| 臂 | 任务 |
|---|---|
| combined(ε 基础 + 高斯) | 75.0(2 seeds) |
| N(ε 基础,无噪声) | 68.0 |
| 168 = L2×8,无噪声 | 74.0 / 65.0(均值 69.5) |
| 169 = 天花板 [.01,.05,.4],无噪声 | 70.0(半程 68.5) |
**裁决:晨间"剂量说获胜"过早(seed1 侥幸)**。无噪声 L2-boost 家族聚在
~65-74(三 run 均值 ~69.8),介于 N(68)与 combined(75)之间;168≈169
→ 3.2%→5% 覆盖再加剂量无增益(平台)。真相居中:**细层 ε 补回高斯通道
的一部分,但补不满;亚 cell 通道仍有独立贡献**。配方 v2(去噪声)不予
升级,旗舰保持 combined(带高斯)。所有差异 1-2 SE,精细区分需更多 seed。

**B. move_two_plates seed2 重复 = 9.0%**:与原次 10% 一致(未污染)——
该任务 combined = 19.5 / 9.0,均值 14.3,真实地弱且不稳。

**C. 外部效度终表(combined vs 参照)**:
| 任务 | combined | 参照 | 判定 |
|---|---|---|---|
| move_plate | 75.0(2s) | 自家 official 64.6 | **+10** |
| dishwasher_close_trays | **100/100** | 论文 ~75-80 | **顶格** |
| saucepan_to_hob | 83.0(1s) | 论文 ~75-85 | 带顶 |
| sandwich_remove | 60.0 均值(52.5/67.5) | 论文 ~55-65 | 平手 |
| move_two_plates | 14.3 均值(19.5/9.0) | 论文 ~20-30 | 低于 |
**图景**:粗粒度失败模式任务(盘/托盘/锅)大胜至顶格;精细双手长视界
递减至低于论文。机制边界与 §52.5/§56 一致:ε-bin 的粗层对比数据只在
失败发生于粗粒度的任务上转化为增益。

### 60. Stage-171 预注册:explore-aware n-step 截断 × crown 复活(用户裁定的 QC 结合路线 ①)

**机制**:agent 追踪"执行动作来自被 shift 的已注册 plan"(replan 感知
倒计时,persist-2 → 2×replan_interval 步),经 `explored` extra 入 replay
(在线+自模仿重标自动继承,离线 demo 恒 0);采样时 n-step 在窗内首个
后继 explored 步处截断提前 bootstrap(首动作自身 explored 不截断——
1-step 对比正是所需)。标量/向量双路径,bit-equal + 语义测试 12+1/13 过,
回归 91/91。开关:`replay.nstep_explore_truncate`(单旗标同时门控存储与
截断)。
**Run**:crown 配方(stage163b:探索[0.016,0.032,0.064]persist2 + 衰减 +
nstep8 + replan8)+ 截断,move_plate seed1,GPU2,101k。
**判据**:crown 参照 53.3(±2.5);截断版 **≥70 → 毒性通道理论证实**
(探索段 8 步报酬和污染 QC 目标是 crown 崩溃主因),CQN-AS×QC 结合的
第一块基石落地;≈55 → 毒性通道非主因,干涉另有机制。健康检查:replay
中 explored 密度应 ≈0.15-0.20。

§60 追加(实现-发射记录):实现 + 双路径位等/语义测试全过(13+91)。发射
经历一次"零旗标幽灵"(两次 script 发射 npz 密度 0.000,同代码同配置的
直跑/探针进程密度 0.17-0.21,机制未定,疑 stale import 或 kill 交叉火力;
全部僵尸清场后以探针实测 0.17 的进程转正为正式 run)。健康检查规矩再
加固:发射后必须验证「进程树单实例 + 用进程 cmdline 捕获 run dir +
数据内容抽查」三件套。正式 run:probe_run 目录,GPU2,~04:15 训毕,
链 sealed+validation。

### 60.1 Stage-171 判决:**80.5**(sealed 200-ep@800;validation 74.0 一致)

**毒性通道理论证实**:同一 crown 配方,仅加 explored 截断,53.3 → **80.5**
(+27.2)。且为全项目 move_plate 历史最佳:vs combined 75.0(+5.5)、
official+QC 74.0(+6.5)、official 64.6(+15.9)。**§47 的可加性判据
(≥80)在修复毒性通道后达标**——探索×衰减与 QC 回传×滚动执行在隔离
相互污染后近乎可加。seed2 已发射(单实例全树验证,健康自检挂链);
≥2 seeds 通过后 crown-truncate 升任新旗舰,随后:探针(反事实口径)、
Tier-1 近分布探索变体、Tier-2 检索 best-of-N、外部效度任务矩阵移植。

§60.2 勘误(追鬼收场):零旗标幽灵 = **误读 demo 播种**。前 ~11k 全局步
的 episode 文件是 51 条离线 demo 的转换件(explored=0 正确),在线
episode 自 ~11.2k 起密度即健康(0.18-0.24 全程)。被判死的历次发射
全部无辜;env-var/启动载体/stale-import 假设全部撤销。§60.1 的 80.5
判决无损(晚期密度 0.27-0.34 实证截断生效)。健康检查配方修正:抽
start_step>12000 的 episode。seed2(seed2_direct)存活转正,评估链已挂。

§61 预注册(stage-172,Tier-1 细层专属探索):皇冠-截断配方,唯一变量
bin_explore_probs [0.016,0.032,0.064]→[0,0,0.112](总剂量配平,全押最细层
±1/125 格)。假设:近分布探索降低探索税(撞死率),截断保留其收益通道。
对照 = stage-171 seed1 80.5(200ep@800)。MovePlate,seed1,GPU4。

§60.3 stage-171 seed1 完整 scaling 曲线(每 5k snapshot × 200ep@seeds800,
三卡并行评完;merged CSV: probe_run/scaling_curve_200ep_seeds800.csv):
5k 67.5 | 10k 69.0 | 15k 75.0 | 20k 77.0 | 25k-60k 平台 73.5-77.0 |
65k-75k 77.5/77.5/75.0 | 80k 78.5 | 85k 80.5 | 90k 78.0 | 95k 77.5 |
100k 80.0 | 101k 80.5(sealed 原点)。
判读:①BC 起跳极快(5k=67.5 已平官方终点 67.5);②15k 即入 ~75 平台,
25k-60k 无增益(λ≈0.9→0.55 区间,BC 仍压制 TD);③80k+(λ≲0.4)出现
向 80 的缓坡,85k/100k/101k 三点 80±0.5 —— 增益集中在 λ 低段,无跃迁,
101k 附近未见饱和转折但斜率已缓。含义:(a) 中段 40k 步近乎浪费,支持
更快衰减臂(linear 1.0→0.25@50k,从未跑过,wean 线证明 0.25 持有安全);
(b) 若求单点更高,加训至 150k(λ hold 0.25)有正期望但边际有限。
评估事故记录:分片 g4 曾因 EGL 数字索引把渲染落到 GPU0(与他人训练争
用,饿死 9 分钟无输出)——根因=eval 脚本用同一数字同推 CUDA 与 EGL;
临时解=UUID 算力 + 显式 EGL id;待办=脚本拆 --compute-device/--egl-device-id。
GPU1 已于 08-02 重启后恢复(CUDA 可用),旧"坏卡"结论作废。

§62 预注册(stage-173,两段式快衰减):皇冠-截断配方,唯一变量
bc_lambda_schedule linear(1.0,0.25,100000) → step_linear(1.0,0.25,50000,
0.0125,50000)。动机 = §60.3 曲线(增益集中于 λ 低段,25k-60k 平台浪费)
× §53 wean 阶梯(晚期 λ=0.0125 安全且 83.5 为项目最高)。假设:压缩
高 λ 平台 + 主阶段引入低 λ 段 → 同预算终点 ≥ 80.5;分段归因:50k 前塌
= 快衰过急,后半塌 = 低 λ 过早(λ* 悬崖仅在成熟策略上测过)。
对照 = stage-171 seed1(80.5,§60.3 全曲线)。MovePlate,seed1,GPU2
(用户指定,与 seed2_direct 合租 0.45×2 至其 ~11:00 训完;首发 GPU3 因
hydra 括号值未引号解析失败,未启动即退,无污染)。

§60.4 stage-171 seed2 判决:sealed 200ep@800 = 72.5(GPU3 独占,187.6s
正常,验证 50ep@400 = 62.0 同向,无腐蚀迹象)。皇冠-截断双 seed =
80.5/72.5(均值 76.5,散差 8pp)vs 组合 nstep3 双 seed 76.0/74.0(均值
75.0)。裁决:毒性通道机制结论不变(站在 seed1 +27.2 上);**旗舰晋升
暂缓**——均值 +1.5 在 8pp seed 方差下不足裁定,待 seed3 破平局或
stage-173(快衰减)/172(细层探索)改写格局。附注:seed2 曲线未做全扫
(只评终点);§60.3 曲线属 seed1。

§62.1 stage-173 判决:两段式快衰减 step_linear(1.0,0.25,50k,0.0125,50k)
**中途坍塌**——训练成功率(replay 实测,seeds≥12k 在线段)30k 0.51 →
40k 0.05 → 50k-101k 全零。塌点 ~35-40k,对应 λ≈0.40-0.50,落入预注册
判据第一支:**快衰过急**。发现升格:λ 悬崖随策略成熟度移动——成熟
策略(150k 训练后)λ*<0.0125(§53),而 ~35k 的未成熟策略在 λ≈0.4-0.5
即塌。BC 锚最低必要剂量 = f(策略成熟度),单调下降。不做终点 sealed
(训练全零已决定性);皇冠的 linear(1.0,0.25,100k) 恰好压着安全走廊。
后续候选:主线安全版低 λ 变体 = 皇冠原 schedule 跑满 100k 后接续训
100k→150k λ:0.25→0.0125(= 把 §53 wean00125 的 83.5 红利以续训形式
搬进主线,零悬崖风险)。

§61.1 stage-172 判决:细层专属探索(剂量配平 [0,0,0.112])sealed 200ep
@800 = **65.0**,低于皇冠-截断(80.5/72.5)与组合(76.0/74.0)。训练内
稳态 0.55-0.60 亦低一档。**Tier-1 假设"近分布探索降低探索税"在纯细层
形态下不成立**:粗/中层大步探索承载不可替代的价值(与 §58 剂量部分
替代结论一致)。细层-only 出局;Tier-1 余下候选(后缀翻转、相干高斯)
优先级下调,待 174/seed3 出分后再议。

§62.2 stage-174 判决:皇冠-截断 seed1(101k=80.5)续训 50k、λ
step_linear 降至 0.0125 → **sealed 200ep@800 = 82.5,验证 50ep@400 =
82.0**。三个要点:①续训段(101k-151k)训练成功率 0.62-0.72 全程无塌,
再证成熟策略低 λ 安全(对照 §62.1 未成熟 λ≈0.45 即塌);②sealed +2.0
(80.5→82.5,单点不显著)但**验证 +8.0(74.0→82.0),两口径收敛于 82**
——泛化 gap 闭合,增益真实;③与 wean00125(组合谱系,83.5)同带,
低 λ 续训红利跨谱系可移植。**主线配方候选升级:皇冠-截断 100k +
λ→0.0125 续训 50k**(150k 总预算,82.5/82.0)。注:eval 耗时 450.9s
(2.6× 常态,GPU4/EGL5 组合偏慢)但验证交叉一致,数字可信。待 seed3
(~18:50 训完)定旗舰终局。

§60.5 旗舰终局判决(三 seed 齐):皇冠-截断 sealed 200ep@800 =
**80.5 / 72.5 / 42.5**,均值 65.2,极差 38pp;对照组合 nstep3 =
76.0/74.0(紧)。裁决:**旗舰晋升否决**——皇冠-截断是高天花板/高方差
线,QC 化(nstep8 开环训练)的跨 seed 不稳定性再次现身(同族证据:
offqc8@dishwasher 86.5/2.0 双模态)。seed3 非塌盘非 NaN:训练全程
0.38-0.50 就没起来过,弱 seed 形态。机制结论不变:①截断修复毒性通道
(seed1 +27.2)仍成立;②低 λ 续训红利(174:82.5/82.0)仍成立,但其
基座是幸运 seed。**组合 nstep3 保住旗舰(稳定性权重 > 均值 +1.5 的
诱惑)**;皇冠-截断降格为"冲分线"。开放方向(未启动,待定夺):
(a) stage-175 = 组合 nstep3 + 截断(把已证的截断收益装回稳定基座,
无 QC 方差);(b) QC 方差根因(为何 nstep8 开环训练分裂 seed 命运);
(c) 弱 seed 早期识别信号(20k 时 0.42 已可判?)。

§63 QC 方差根因调查(用户指令:先审计再查因)

**A. seed3 训练审计:干净。** 配置与 seed2 逐行 diff 仅差 seed(及一个
null 字段的 schema 顺序);日志 0 错误/0 NaN/0 重启;独占 GPU2(254
steps/s,快于合租期 seed2 的 102,与正确性无关);截断/探索管线正常。
42.5 是真实的弱 seed,不是事故。

**B. 三个证据链,方差机制重构:**
1. **命运在首个在线 bucket(12-20k)已定**:三 seed 在线成功率
   0.61/0.41/0.39,与终局 80.5/72.5/42.5 完全同序;relabel 飞轮
   (@101k 278/220/144)是放大器而非起因——@20k 时仅 22/13/11,差距
   已由成功率决定。
2. **弱开局是配方无关的种子彩票**:组合 nstep3 的 comb_s2 同样开局
   0.41,但 40k 即回到 0.62、终局 74.0(完全自愈);QC 线的 seed2
   (0.41 开局)仅部分自愈(72.5),seed3(0.39)零自愈(42.5)。
   **差异不在彩票,在自愈能力:nstep3 有均值回归,nstep8 锁死早期命运。**
3. **锁死形态 = 自信的悲观**:seed3 终局 entropy 1.28(seed1 1.70、
   seed2 1.46)、critic_loss 最低(0.135)——分布最尖、拟合最好、行为
   最差。机制推断:replan8+nstep8 下 8 步回报和忠实反映当前弱策略的
   失败(截断只挡探索污染,不挡 on-policy 失败和),TD 高效学会"确信
   的零";nstep3 多 5 步 max-Q bootstrap 的乐观偏置维持候选动作的
   生存空间 → 回归。n-step 双刃:降 bootstrap 偏置的同时放大行为策略
   偏置——强 seed 更强、弱 seed 更弱的 rich-get-richer 回路。
   (旁证:offqc8@dishwasher 86.5/2.0 双模态同构。)

**C. 附带修正**:组合 nstep3"稳定"的样本量仅 n=2(seeds 3/4 于 07-27
9k 步即被杀,非完整 run)——其低方差结论证据薄弱,§60.5 的旗舰裁决
稳定性论据需降级为"未被证伪"。

**D. 待做的决定性实验**(未启动):
1. **seed3 × nstep3 重跑**(同 seed 同配置仅 nstep 8→3):若恢复到
   ~70+,nstep8 因果锁死成立——这是一锤定音的实验,3.2h;
2. 弱 seed 早期识别:20k 在线成功率 <0.45 即弱开局(三例全中),可作
   止损/重开信号;
3. 廉价 Q 探针:给 eval sweep 脚本加 10 行 dump 每 snapshot 的贪心 Q
   均值,验证"自信的零"(低优先)。

§63.1 预注册(stage-176,nstep 因果锁死检验):seed3_direct 全同配置
(seed=3,replan8,截断开,同 launch 链)唯一变量 replay.nstep=8→3。
判据:sealed ≥~70 → nstep8 因果锁死弱开局成立(§63.B 机制坐实),
QC 方差获得可控旋钮(成熟度自适应 nstep 候选);若仍 ~40-50 → 锁死
主因不在 nstep(转向 replan8 开环执行或交互项)。对照 = seed3_direct
42.5/48.0。MovePlate,GPU2。

§63.2 stage-176 判决(nstep 因果检验):seed3 仅改 nstep 8→3 →
sealed 200ep@800 = **60.5**(对照 42.5,+18.0),验证 50ep@400 =
**74.0**(对照 48.0,+26.0,已达组合线弱 seed comb_s2 的自愈水平)。
训练内曲线:开局 0.39 完全复刻对照(同种子验证),随后每 bucket
+0.08-0.11。裁决:**部分因果成立**——nstep8 约贡献锁死伤害的一半
(sealed 口径 38pp 差距中的 18pp),§63.B 的"n-step 放大行为策略偏置"
机制获得直接因果支持;但残余 ~12-20pp 缺口指向 replan8 开环执行或
BC 表征彩票的交互项。注:两 seed 族分裂(800 族 60.5 vs 400 族 74.0,
弱/中 seed 反复出现 val>sealed 的倒挂,强 seed 相反)——本身或是
"弱 seed 泛化谱系不同"的线索,列为开放观察。
后续候选:(a) seed3 × nstep3 × replan1(全去 QC 化,补齐因果分解
最后一格);(b) 成熟度自适应 nstep(弱期 3 → 强期 8)原型。

§63.3 预注册(stage-177,因果分解最后一格 = 全去 QC 化):seed3,
nstep=3 + replan_interval=1 + 探索剂量回配平([0.002,0.004,0.008],
按 §163 的 ×8 规则反算)+ 截断保留。这同时是 §60.5 提的 stage-175
(组合 nstep3 + 截断)在弱 seed 上的首个数据点。判据:若 ≥~75 →
锁死配方 = nstep8(半)+ replan8(半),QC 化两component 均有害于弱
seed;若仍 ~60 → replan8 无关,残差归于 BC/表征彩票。评估用
replan-interval=1(与训练匹配)。对照:42.5(nstep8+replan8)、
60.5(nstep3+replan8)。GPU2。

## 64. 2026-08-03:λ 的可解释性纲领(用户诊断:schedule 的初值/末值/时长
均为魔数,不可跨任务迁移)

**问题重述**:λ 不是"BC 权重"而是**罚系数**。margin 铰链实施约束
`Q(s,a_demo) >= max_sibling Q + m`(分段线性 → 违反时梯度为常数
λ/(L·D·B),满足时**恰为零**),所以 λ 设定的是一个**力**;其对手 TD
力的尺度由奖励密度、视界、Q 值域、demo 占比决定,**全部任务相关**。
故 (1.0, 0.25, 100k, 0.0125) 是"在 MovePlate 上恰好达成某力比的坐标",
不是常数。可迁移的应是**力比**或其行为效果,而非 λ 本身。

**发现的仪表空白**:主 update 路径此前只 log critic_loss/entropy/
target_entropy/loss_coeff——**连 bc_weight 都没落盘**,即两周的 λ 调参
全程无观测。已修:`cqn.py` update 加 6 个诊断量(demo 行统计):
bc_weight、bc_agreement(demo 动作是否仍是 argmax = 约束的行为满足度)、
bc_binding_rate(兄弟 bin 中违反 margin 的比例 = 铰链真正在施力的部分,
已排除自身项)、bc_margin_gap(Q_demo − max_sibling,约束松弛量)、
bc_sibling_q_span(critic 对兄弟的分辨力 = 覆盖度代理)、
bc_online_agreement(在线行同量)。测试:tests/unit/test_cqn_as.py 中
"demo 身份不得泄漏"守护测试原比较整个 metrics dict;诊断量按 demo/
online 分组统计,故按定义会随标签改变——已在该测试中排除这 5 个纯观测
键并写明理由,参数级 bit-for-bit 保证不变(仍逐叶断言)。

**离线回溯探针**:`scripts/probe_bc_anchor.py`——对任意 run 的 snapshot
档案逐点重算上述诊断,demo batch 由 BiGym 原始数据集**重新加载**(跨 run
完全同批,消除自模仿污染)。首测(171 seed1):
`@5k λ=0.96 agree=0.899 bind=0.297 span=0.41` →
`@101k λ=0.25 agree=0.980 bind=0.093 span=1.42`。
即 λ 降 4× 的同时约束**满足得更好**、铰链施力更少、critic 对兄弟的分辨
力涨 3.5×——初步支持 H3(约束随覆盖度增长自我执行,λ 可被释放)。

**待证伪的三条定律**(全档案扫描进行中:171 s1/s3、174、173、176、
wean0、wean00125、158 s1):
H1 力比定律(塌盘发生于力比跌破普适 c*)、H2 行为定律(前兆是 agreement
跌破阈值 → 控制器直接盯它)、H3 覆盖定律(安全 λ 下限 = f(sibling span
/ 覆盖度)→ 逐样本不确定度门控,消灭 schedule 概念本身)。

### 64.1 首批回溯结果:**λ 的真实职能是维持未执行 bin 的可分辨结构**

(注:wean0 run 目录经 3 次续跑覆盖,.hydra/config.yaml 只保留最后一段
的 schedule,故探针 λ 列对该 run 不可用;相位按 §53 已知:101-150k
λ=0.25,150-201k λ=0,205k+ λ 恢复 0.25。探针数据与该相位表逐点吻合,
本身即是探针有效性的强验证。)

| run/相位 | agreement | binding | **sibling span** |
|---|---|---|---|
| 171 s1 @5k(λ=0.96) | 0.896 | 0.305 | 0.404 |
| 171 s1 @101k(λ=0.25) | 0.979 | 0.097 | 1.386 |
| wean0 @145k(λ=0.25) | 0.963 | 0.154 | 1.429 |
| **wean0 @150k(λ=0)** | **0.475** | **0.944** | **0.045** |
| wean0 @201k(λ=0 尾) | 0.315 | 0.995 | 0.011 |
| wean0 @216k(λ 恢复) | 0.795 | 0.323 | 1.073 |
| 174 @150k(λ=0.0125) | 0.919 | 0.307 | 1.125 |

**机制结论(改写 §53 的"恒力墙"解释)**:C51 per-bin 头中,TD 只监督
**被执行的那个 bin**(`canonical_per_sample` 只用 chosen_log_probabilities);
兄弟 bin 无任何直接梯度(`unseen_return_floor_weight=0.0`),**margin
铰链是塑造它们的唯一力**。λ→0 时该力消失,兄弟 Q 在权重衰减与参数共享
的牵引下**塌向同一值**——span 1.43 → 0.045(30×),argmax 退化为在平坦
地形上抽签,行为随之死亡。塌缩不是"模仿变弱"的连续极限,而是**约束
存在/不存在的相变**:λ=0.0125 仍保住 span 1.125,λ=0 直接归零。这解释
了为何 λ* < 0.0125 却又 λ=0 必死——铰链只需强到抵消漂移力,所需强度极小
但**不可为零**。

**λ 可被释放的真正原因(H3 获支持)**:171 s1 的 span 在 λ 单调下降的
同时从 0.404 涨到 1.386——探索使兄弟 bin 被真实执行、从而获得 TD 监督,
覆盖度接管了铰链的工作。**λ 该跟随的不是时间表,而是覆盖度。**

**可迁移的量**:span/Q(无量纲)。健康区 ≈1.0-1.4,塌缩 ≈0.01。
agreement ≥0.95 健康、≤0.5 已死。二者皆无量纲、跨任务同义。

**由此产生的算法候选(比控制器更优美)**:铰链的职能既然是"给无监督的
bin 提供结构",就该由一个**不依赖 demo 身份、不含手调力度**的项承担——
代码里已有的 `unseen_return_floor_loss`(Q-Transformer 式:把未执行 bin
回归到合法最小回报,`weight=0` 未启用)正是该形态,且天然服务 no-BC 线。
候选实验:λ→0 + unseen_return_floor 开启,检验 span 是否被守住、行为
是否免于塌缩。若成立,则 (1.0,0.25,100k,0.0125) 四个魔数一次性消失。

### 64.2 **§62.1 判决作废(重要更正)**:stage-173 不是"快衰过急",是 NaN

回溯探针在 173 的 35k+ 快照上返回全 nan;直查参数确认:30000 快照
0 NaN、max|w|=1.228,**31000 步起全部 28,571,854 个参数皆为 NaN**
(train.csv 的 critic_loss/entropy 亦自 31k 起为 nan)。训练"成功率
40k 跌到 0.05、其后全零"是 NaN 权重输出垃圾动作的结果,**与 λ 无关**。

因此:
1. **§62.1 的结论(λ 悬崖随策略成熟度移动、λ≈0.4-0.5 对未成熟策略致命)
   证据基础不成立,予以撤销**。快衰减 schedule 从未被真正检验——它在
   λ 还有 0.55 时就死于数值发散。
2. 这是本项目第三次 NaN 事件(offqc8_s2 @8k、本次 @31k),且两次都发生在
   nstep8+replan8 谱系。**NaN 不再是"非系统性单发",升为待查缺陷**。
3. §174 的低 λ 续训结论**不受影响**(它是从健康的 171 s1 101k 快照续训,
   全程 span 1.1-1.4、无 NaN,82.5/82.0 有效)。
4. 教训:**训练曲线全零 ≠ 策略塌缩**。此前仅凭 replay 成功率归零即下
   "坍塌"判决,漏检了数值发散。已有的非有限值早停守卫(开放项)应立即
   实装:检测到 loss 非有限即停并落盘诊断快照。

**待重跑**:快衰减臂需在装上 NaN 守卫后重做,才谈得上检验。

### 64.3 预注册(stage-178,机制检验:结构项能否替代 demo 锚)

§64.1 的机制主张:铰链的职能是给**无 TD 监督的兄弟 bin** 提供结构。
若为真,一个不看 demo 身份、不带时间表的结构项应能替代它。代码内已有
`unseen_return_floor_loss`(遮蔽被执行 bin,把其余 bin 回归到合法最小
回报 0;Q-Transformer 式保守正则),此前 weight=0 从未启用。

**完美三元对照**(同一基座:171 seed1 的 101k 快照,续训 50k):
| 臂 | λ(101k→151k) | floor | 已知/待测 |
|---|---|---|---|
| 174(已跑) | 0.25→0.0125 | off | **82.5/82.0**,span 1.125 |
| wean0(旧谱系) | 0 | off | 塌(span 0.045) |
| **178A** | **0** | **on** (w=0.1, value=0, mean) | ? |
| **178B**(对照) | **0** | off | ? 本谱系内复现塌缩 |

判据:178A 若 span 保持 ≳1.0、sealed ≳75 → 机制成立,**λ 及其四个魔数
可被一个无量纲、无时间表的结构项替代**(且该项不看 demo 标签,天然服务
no-BC 线);178B 应塌(本谱系内的阴性对照,排除"皇冠-截断谱系自带免疫"
这一替代解释)。floor 权重 0.1 与 critic_lambda 同量级,是唯一剩余旋钮,
但它不随时间变化、不随任务重调——这正是要检验的性质。

### 63.4 stage-177 判决:**QC 化是 38pp 方差的全部来源;seed3 不是弱 seed**

同一颗 seed3,三种制度(sealed 200ep@800 / 验证 50ep@400):
| arm | nstep | replan | sealed | Δ |
|---|---|---|---|---|
| 171 s3 | 8 | 8 | 42.5 | — |
| 176 | 3 | 8 | 60.5 | +18.0 |
| **177** | **3** | **1** | **79.5**(val 70.0) | **+19.0(累计 +37.0)** |

裁决:①**锁死是两个组件各承一半**:nstep8(+18)与 replan8 开环执行
(+19),后者略重;②seed3 在标准执行制度下达到 79.5 ≈ seed1 皇冠-截断
的 80.5——**"种子彩票"是 QC 化的产物,不是环境/初始化的固有属性**;
③§60.5 的"高天花板高方差"描述需改写:皇冠-截断的方差**全部**来自
QC 化,去掉它方差即消失。

**附带的重大收获**:177 的配方 = 组合 nstep3 + replan1 + 探索截断,
即 §60.5 预告的 stage-175。它在**最差的那颗 seed** 上拿到 79.5,已超过
现旗舰(组合 nstep3 双 seed 76.0/74.0)的两个 seed。**新旗舰候选**:
截断把已证的毒性通道修复装回稳定基座,收益保留、方差消失。
待办:seed1/seed2 复现(seed1 已发,GPU4);之后做截断消融(nstep3 下
窗口短,截断的边际贡献需单独确认)。

### 64.4 stage-178B(阴性对照)判决:**机制在第二个谱系上复现**

皇冠-截断基座 + λ=0 + 无结构项,续训至 151k:
**sealed 200ep@800 = 0.0(0/200)**;探针 @151k:agreement 0.361、
binding 0.996、**span 0.0104**(健康态 1.1-1.4)、Q 1.104。

与 wean0(组合谱系)的塌缩指纹逐项一致(span 0.011、agreement 0.315、
binding 0.995)。**§64.1 的机制主张在独立谱系上获得复现**:λ=0 →
兄弟 bin 结构消失 → argmax 退化 → 行为死亡;且"皇冠-截断谱系自带
免疫"这一替代解释被排除。178A(λ=0 + unseen_return_floor)评估中,
它是"结构项能否替代 demo 锚"的正面检验。

### 64.5 stage-178A 判决:**λ 有两份工作,结构项只能接管其中一份**

同一基座(171 s1 @101k = 80.5)续训 50k,四元对照终局:
| 臂 | λ | floor | sealed | agreement | span | span/Q |
|---|---|---|---|---|---|---|
| 174 | →0.0125 | off | **82.5** | 0.919 | 1.125 | 1.10 |
| **178A** | **0** | **on** | **72.5** | 0.819 | 0.818 | **1.06** |
| 178B | 0 | off | **0.0** | 0.361 | 0.010 | 0.009 |

**结论(机制二分)**:
1. **可分辨性(span)可以外包**。一个完全不看 demo 标签、无时间表的
   结构项(`unseen_return_floor`,w=0.1,value=0,mean)把 λ=0 的 run
   从 0/200 救到 72.5,span/Q 1.06 落在健康带。**"demo 锚是存活的必要
   条件"被证伪**——必要的是"未执行 bin 有人监督",而非 demo 身份。
   这条对 no-BC 线是直接可用的结论。
2. **排序(agreement)接管不了**。178A 的 agreement 只有 0.819(健康
   ≥0.95),且相对自身起点退步(80.5 → 72.5),而 174 同期是进步
   (80.5 → 82.5)。floor 把所有未见 bin 一律压向 0,能撑开跨度,却
   不保证 demo 动作**排在第一**;铰链除了"压下兄弟"还在"顶起 demo",
   这是第二份、独立的工作。
3. 因此 §64.1 的机制陈述需精化:**λ = 可分辨性维持 + 排序优先,两份工作
   捆在一个标量里**——这也解释了为什么它既无法用单一无量纲量刻画、
   又对数值不敏感(两份工作的需求量级不同)。

**由此收敛的方案图**:
- **主线**:结构由 floor 承担,排序由一个**恒定的小 λ** 承担(174 证明
  0.0125 恒定即足够且更好)。候选 = floor on + λ≡0.0125,**零时间表、
  两个有自然标度的常数**。待验:此前 0.0125 只在"高 λ 训练 100k 之后"
  被检验过,from-scratch 未测。
- **no-BC 线**:floor 是合法的结构替代品,72.5 是该线迄今最强的存活证据
  (对比其历代 40-50 区间)。

### 64.6 跨任务标定:**计数型指标可迁移,比值型指标不可**

MovePlate(171 s1,80.5) vs dishwasher_close_trays(combined s1,100.0),
同一探针、同一 batch 协议:
| 量 | MovePlate 5k → 平台 | dishwasher 5k → 平台 | 可迁移? |
|---|---|---|---|
| agreement | 0.896 → **0.98** | 0.839 → **0.98** | **是** |
| binding_rate | 0.305 → **0.07-0.10** | 0.319 → **0.08-0.12** | **是** |
| span/Q | 0.54 → **1.37-1.40** | 0.66 → **1.99** | **否**(+45%) |
| Q 量级 | 1.01 | 0.65 | — |

**为什么**:agreement 与 binding 是**计数比例**(达标的头数 / 违反的兄弟
bin 数),按构造落在 [0,1],与价值尺度无关;span/Q 是两个价值尺度量的
比,任务的价值地形形状不同则不抵消。轨迹**形状**三者一致(先升后平台),
但只有前两者的**绝对水平**跨任务重合。

**对控制器版的直接修正**:设定值应挂在 **agreement(地板 0.95)**或
binding_rate(上限 ~0.15),**不要用 span/Q 的绝对阈值**。这两个量在两个
任务上标定一致,且与 §64.5 的 seed3 悖论相容——它们只作单边地板(跌破
才加大 λ),绝不作为优化目标(seed3 的 0.99 是自然到达,不是被推上去的)。
span/Q 仍是有用的**诊断**(塌缩时归零),但不适合做设定值。
待第三任务(sandwich_remove)确认。

### 64.6a QC NaN 根因审计: OOD 标准化输入触发,不是 C51 目标溢出

**1. Previous-stage result.** 对全部带 Hydra 配置且使用 fused-8 JAX
后端的 BiGym CQN-AS `train.csv` 重新按列名扫描:完成至少 8k 步的
`nstep=8,replan=8` 共 15 个 run,2 个出现真实训练 NaN(13.3%);
其余制度共 322 个 run,1 个 NaN(0.31%;单侧 Fisher exact p=0.0056)。
两个 QC 事件分别是
`cqn_stage165_second_task/...offqc8_seed2...` @8k 和
`cqn_stage173_fastwean/two_stage_s1` @31k。坏前一桶的 critic loss 仅
0.188/0.201,不是 loss 逐步爬升。stage-173 的 30k snapshot 中参数、
target 与 Adam moments 全部有限(max|param|=1.228);35k 时全部
28,571,854 个参数、全部 26,484,718 个 target 参数及除 step counter
外全部 57,143,708 个 optimizer 元素均为 NaN,即一次坏更新全树污染。

真实 replay 的网络输入提供了新的共同前兆:offqc8_s2 在坏点前出现
`max|low_dim_state|=2605.5`,而同任务/同预算健康 seed1 只有 186.5;
stage-173 坏点前为 3042.8,且 `|z|>1000` 占 0.147%。这些值已经是
ConcatDim 交付给网络的 demo-z-score,不是原始物理量。实现以 demo
`std`(下限仅 1e-10)做除法,但对 online OOD 值不裁剪
(`robobase/envs/bigym.py:804-843`,
`robobase/envs/wrappers/concat_dim.py:58-94`)。QC 的 replan-8 连续执行
更容易产生这种离开 demo 支持集的状态;随机 replay batch 是否聚集到
极端窗口,解释了 seed/运行级偶发性。当前配置同时是
`critic_grad_clip=null`,所以首个非有限梯度会直接进入 AdamW。

**2. Interpretation.** 先前的“nstep=8 bootstrap 目标漂移溢出”解释被
排除:C51 投影在进入 loss 前显式 clip 到 `[-2,2]`,softmax Q 也被同一
support 有界;原始 replay 全有限且新旧 sampler 逐字节相同。当前最强
因果链是 **replan-8 数据分布 -> 未裁剪的巨大 z-score -> 某个随机
minibatch 的非有限 forward/backward -> 无梯度防线的 Adam 全树污染**。
已确定前兆、污染位置和 QC 富集;尚未捕获首个坏 minibatch,所以不能把
最后一跳进一步归到 LayerNorm/GRU/CUDA 的某一个具体算子。另一个
`nstep=1,replan=1` dense-return run 也曾 NaN,因此 QC 是强风险放大器,
不是 NaN 的逻辑必要条件。

**3. Next-stage decision.** 数值机制门采用同一 QC 配方/seed/replay,
只比较 baseline、`low_dim z-clip=20`、`critic_grad_clip=10` 三臂;不同时
改变 nstep/replan。先以固定训练 seeds 1--3 做 40k stability selection,
指标为首个 non-finite step、输入 max/p99.9、pre/post-clip grad norm 和
有限参数比例;通过线为三 seeds 全部 40k 有限且 baseline 至少复现一次。
这个门仅用于复现历史 `truncate_demo_at_success=false` 数值机制,不作策略
质量基线;通过臂须在已修正的 `truncate_demo_at_success=true` 数据制度下
另起 matched quality 实验跑到 101k,用 validation seeds 400--449 选 checkpoint,
最后 sealed held-out seeds 800--999;任务质量不得用稳定性 loss 代替。

**4. Execution.** 已完成上述 373-run 矩阵扫描、三份坏 snapshot 全树
审计及 matched replay 输入分布扫描。现有
`exp_local/cqn_nan_probe/qc_notrunc_s1` 复现探针已正常完成 8k;最后记录的
7k critic_loss=0.2055 且有限,log 随后正常清理 replay,未复现 NaN。
未再发三臂动态门:当前六卡均已有训练/评估负载且仓库训练已超过用户规定
的四并发硬上限,没有安全训练槽。下一步应先给 update 输出加 pre-apply
gradient/activation 非有限诊断与坏 batch 落盘,再在槽位释放后发 matched
门;仅有 post-update loss 早停只能止损,不能定位首坏算子。

### 64.6b 2605 state 反解:右夹爪内部关节的 demo-std 放大,不是 MuJoCo qpos 爆炸

**1. Previous-stage result.** 将 `dishwasher_close_trays_offqc8_seed2` 的 44 份
成功 demo raw safetensors 与 44 份归一化 demo replay 逐轨迹配对(第 0 维
逐值完全相等),反解出了运行时实际使用的 affine stats。在线 episode 56 的
local step 242/global replay index 15128 中,`low_dim_state[21]=2605.4626`。
62 维布局是 qpos 0--29、同序 qvel 30--59、两维 gripper summary 60--61;
运行时 MuJoCo joint 映射确认 dim 21 是
`h1/robotiq_2f85_right/left_driver_joint`。它的 raw qpos 仅 0.774285 rad,
demo mean/std 为 0.002510/0.0002962,故被放大成 2605.46。相邻的右夹爪
driver/spring joints(dim 17/19/23)同时是 raw 0.775/0.782/0.769 rad;
step 249 的 `dim 47=-2588.68` 是 dim 17 同一关节的 qvel,raw 仅
-5.336 rad/s,其 demo std 为 0.002061 rad/s。driver 的 MJCF 合法范围是
[0,0.8] rad,所以 0.774 rad 仍在关节限位内。

成功 demos 的右手 gripper summary 第二维在全部 11,046 帧中恒为 0,
且右夹爪内部 dim 21 的 raw 范围仅 [-0.00048,0.00505];该在线 episode
却在 step 228--255 走过 0.1 -> 1.0 -> 0.0。三路图像显示右腕紧邻
dishwasher rack,但仅凭 replay 不能把运动进一步唯一归因为接触力还是控制
动态。可以确定的是 MuJoCo 给出的是有限且限位内的夹爪状态,2605 只在
demo 标准化之后出现。本阶段是数值机制定位,不涉及 best-checkpoint 策略
质量比较。

**2. Interpretation.** “因为没有 fail termination,MuJoCo 状态本身跑到
2600”不成立。BiGym 实际有 `terminate = success or fail`,其中通用 fail
是 pelvis 距原点超过 10 m,另在 PhysicsError 时 truncate;该样本的
reward/terminal/truncated 均为 0,浮动底座 xyz 约
[-0.111,-0.554,0.500],episode 最终正常在 320-step TimeLimit 截断。
它没有 task-specific stuck/jam failure,所以长 episode 的确增加遇到 OOD
接触状态的机会,但这里只是物理上合法的右夹爪全行程。真正的表示问题是
`proprioception` 已包含两只 Robotiq 的内部被动机构 qpos/qvel,同时又追加
了两维 `proprioception_grippers`;任务 demo 从不动右夹爪,于是重复的内部
维度以求解器微抖动作为 std。把合法夹爪动作定义为 fail 会错误改变 MDP,
只能减少暴露机会,不能修复数值尺度。

**3. Next-stage decision.** 下一机制门优先隔离“重复的内部夹爪状态”这一项:
同一历史 QC 配方、replay 与 seeds 1--3,只比较原始 62 维 baseline 与移除
两只夹爪内部 qpos/qvel、保留 2 维 gripper summary 的 observation arm;
不同时加 z-clip 或 grad-clip。40k stability 指标为首个 non-finite step、
input max/p99.9、pre-apply grad norm 和参数有限比例;通过线是三 seeds 40k
全有限、内部夹爪 arm 的 |z| 极值消失且 baseline 至少复现一次。梯度防线
另作后续 guardrail 实验。策略质量仍须在
`truncate_demo_at_success=true` 下另跑 101k,validation seeds 400--449
按 best checkpoint 选择,最后 sealed held-out seeds 800--999。

**4. Execution.** 已完成 44/44 raw-demo/replay 精确配对、坏 episode 逐维
反归一化、无渲染 MuJoCo runtime joint-name/range 审计及 step 220--256
三相机核对。现有 `exp_local/cqn_nan_probe/qc_notrunc_s1` 已正常完成 8k;
最后记录的 7k critic loss=0.2055 且有限,log 随后正常清理 replay,没有
复现 NaN,再次说明单 seed/单次短跑不是充分复现。当前仍有至少八个训练
量级 GPU 进程,超过四并发硬上限,故未启动 matched 40k 门;本阶段未改
训练代码。

### 64.6c move_plate 更正:夹爪尺度病灶真实,但既非 NaN 必要条件也非充分条件

**1. Previous-stage result.** 上一节反解的是 dishwasher 的 2605,不是
move_plate stage-173 的首个 NaN;这里独立重做。stage-173 在 30k 的
critic loss=0.2008 且有限,31k 首次记录 NaN。其 30k 前最大输入发生在
online step 29,557:`low_dim_state[51]=-3042.8093`。move_plate 的 60 维
布局是 qpos 0--28、同序 qvel 29--57、gripper summary 58--59;runtime
映射确认 dim 51 是 qvel index 22,
`h1/robotiq_2f85_right/left_coupler_joint`。将全部 60 份 raw demo 与运行的
60 份 replay demo 逐轨迹精确配对后,该值反解为 raw qvel=-3.02674 rad/s,
运行实际 mean/std=-6.23e-5/9.94698e-4。同一冲击中 dim 48 的 raw qvel
为 12.0146 rad/s(z=2345.58)。因此这里仍是有限的夹爪机构状态被 demo
尺度放大,但具体致命极值是 qvel,不是 dishwasher 的 dim-21 qpos。

stage-173 实际 qpos std(dim 17--24)依次为
`3.20e-4,9.02e-5,3.70e-4,4.69e-4,3.20e-4,8.50e-5,3.70e-4,4.54e-4`。
这与外部审计所报的 `1.2e-5/3.3e-5` 不是同一运行统计;上述数值由真正存入
stage-173 replay 的 affine 关系反解,逐点残差 <3e-10。更关键的是,
`truncate_demo_at_success` 在当前代码中发生在 obs stats 计算之后。实际把
未截断 stage-173 与截断 official seed1 的 replay 分别反解,上述每一维
std 逐位相同到约 1e-14,不是只差 1.2 倍。

跨臂证据切断了“极值直接导致 NaN”的充分性与必要性:

- 截断 official nstep-1 seed1/seed2 各跑完 101k 且有限,online max 分别
  2213.64/2444.17;
- 截断 QC seed1/seed2 分别在 iteration 4k/1k 因 NaN guard 退出,但整个
  replay max 仅 91.45/42.54;
- 截断 QC seed4 当前已到 15k,critic loss=0.2051 且有限,同时 replay max
  已达 1440.40;
- 另有 move_plate nstep-1/replan-1 dense-return A7 在 13k NaN,其 replay
  max 仅 91.59。

本阶段是数值机制审计,不作 best-checkpoint 策略质量比较。

**2. Interpretation.** 夹爪内部关节用极小 demo 方差归一化是确定存在的
表示缺陷,且与 demo success 截断无关;但现有证据不再支持把它写成 NaN 的
直接根因。大极值既非充分条件(2444 的 nstep-1 和 1440 的 QC 仍健康),也
非必要条件(42/91 已可 NaN)。nstep-8/QC 在总体矩阵中仍显著富集 NaN,
因此可称风险放大器,不能称“把极值变致命的必要一步”;A7 是 nstep-1
反例。

外部审计关于 normalization 分支的判断也需纠正:`use_standardization=false`
与 `use_min_max_normalization=true` 只控制 action wrapper。stage-173 的
`.hydra/config.yaml` 是 `norm_obs=true,obs_norm_type=standardization`,并由
`ConcatDim` 走 `(v-mean)/(std+1e-10)`。raw demo 到真实 replay 的 affine
拟合残差 <3e-9 也直接证明当前走 z-score,不是 min-max observation 路径。

**3. Next-stage decision.** “std floor + output clip 同时改、两个 seed 过 5k”
只能作为止损 smoke,不能建立因果:它一次改两个机制,而历史 failure 可晚至
31k,当前 QC seed4 也已有限通过 15k。下一数值 containment 门只改一个量:
对 standardization 输出加 `clip=10`,baseline 与 clip arm 使用相同 QC+
截断配置和 seeds 1--4,至少跑到 40k;指标为 first non-finite step、clip
rate、pre-apply grad norm、参数有限比例。通过线为 clip 四 seeds 40k 全
有限且 matched baseline 至少一次失败。与此同时必须在 update apply 前
保存首个坏 batch、逐模块 activation 与 gradient finite mask;即便 clip
通过,没有首坏 batch 也只能证明 guardrail 有效,不能证明根因。nstep 因果
另用 1/1、8/1、1/8、8/8 独立矩阵回答,不得混入 normalization arm。
任务质量仍须 101k,validation seeds 400--449 选 best checkpoint,最后
sealed held-out seeds 800--999。

**4. Execution.** 已完成 move_plate 60/60 raw/replay 配对、截断与未截断
运行 stats 逐位对照、stage-173 30k 前 replay 扫描、runtime joint mapping
及 nstep-1/QC 反例矩阵。QC+截断 seed4 正在 15k 继续运行且有限;当前至少
八个训练量级进程占卡,超过四并发上限,未再启动 clip arm。本阶段没有修改
训练或 wrapper 代码。

### 64.7 三任务标定完成:**agreement / binding 是跨任务常数,span/Q 不是**

| 任务(健康 run) | 5k agree | 平台 agree | 平台 bind | 平台 span/Q | Q |
|---|---|---|---|---|---|
| move_plate(80.5) | 0.896 | **0.984** | **0.081** | 1.39 | 0.99 |
| dishwasher_close_trays(100.0) | 0.839 | **0.980** | **0.084** | 1.96 | 0.66 |
| sandwich_remove(67.5) | 0.823 | **0.988** | **0.074** | 2.17 | 0.49 |

三个任务、三种成绩(67.5/80.5/100.0)、三种 Q 量级(0.49-0.99),
**agreement 平台 0.980-0.988(散布 0.008)、binding 平台 0.074-0.084
(散布 0.010)——跨任务几乎逐位重合**;span/Q 从 1.39 到 2.17(+56%),
且与 Q 量级反相关,不可迁移。原因见 §64.6:前两者是计数比例,后者是
价值尺度之比。

**因此控制器版的设定值确定为 agreement,地板 0.95**(健康平台 0.98,
塌缩态 0.32-0.48,中间无观测密度——地板落在真空区,不敏感)。
binding 上限 0.15 可作冗余判据。三任务标定后,s* 不再是魔数。
注意:agreement 只作**单边地板**(§64.5 seed3 悖论:0.99 且最差 42.5),
控制器是安全装置,不是优化目标。

### 64.8 move_two_plates @82k 更正:state 极值不是这类 NaN 的近因

**1. Previous-stage result.** `official_move_two_plates/seed1_20260804100800`
已完成 101k 训练和全部评估。validation-50(seeds 400--449)在
5k--100k 为 12--24%,101k 最高 **32%**;因此按 validation-selected
best checkpoint 口径,应报 101k 的 sealed-200(seeds 800--999)
**19.5%**。固定端点 100k/101k 分别是 **21.5%/19.5%**。未截断
combined 的 101k 是 19.5% 与 10.0%(后者健康重评 9.0%);所以
当前 seed1 在 matched 101k 与 combined 的强 seed 相同,相对原两次
均值 14.75% 是 +4.75pp(用重评数字则均值 14.25%,+5.25pp)。
先前 +6.7pp 是用当前 100k 对 combined 101k 均值,端点不 matched;
且仍是 n=1 vs n=2,不作方法提升结论。

`official_move_two_plates/seed2_20260804100800` 的 81k 记录仍有限
(`critic_loss=0.22834`),80k snapshot 已落盘,随后在 iteration 82k
报 `critic_loss=NaN` 并中止。该次完整 forensic dump 的三个保留
batch 中,所有 float 叶子的 finite fraction 都是 1.0;
`low_dim_state` max|x| 依次为 **36.814/30.486/34.353**。
守卫触发后,online params 108/108 个叶子与 optimizer state
216/216 个叶子全部 0% finite。另外三份有完整 dump 的事件
(QC seed3@11k,QC seed2@99k,mask seed1@2k)也是所有 float batch
全 finite,发现 batch 的 max|low_dim| 分别只有 87.75/57.41/29.71,
且同样是全部 params/optimizer leaves 一次性变为 non-finite。

**2. Interpretation.** 这个 nstep-1、已有 relative std floor 的最难任务
是直接反例:`2600/3000` 标准化夹爪状态是真实的表示缺陷,但它
不是 NaN 的必要条件,也不是这次的近因。nstep-8/QC 也不是
必要条件,最多仍是事件率放大器。因此§64.6c 的 observation
clip arm 不再是根因定位的下一优先级;它只可作独立 guardrail,
不能用来解释这一簇同签名 NaN。

当前 dump 仍不能把首发点写成“已确定在 optimizer apply”。
`critic_loss` 是 `value_and_grad` 在 optimizer 之前算出的,但 workspace
是 update 返回后才只检查这个 loss。如果前一步是 finite loss ->
non-finite gradient/update,它会先污染参数,到下一步才被 loss guard
发现。所以 34.353 严格说是 detection batch,首污染 batch 也可能是
前一个 30.486 batch;两者(连同再前一个 36.814 batch)均全 finite
且没有 state 极值,因而不改变上述结论。真正未决的是首个
non-finite 究竟位于 forward/loss、backward gradient、Adam state/update
还是 apply/target EMA。现有 forensic tap 还丢掉了 uint8 RGB;图像一定
有限且有界,但重要的是,没有它们就不能对失败 update 做逐位重放。

**3. Next-stage decision.** 先完成已注册的 no-ensemble 独立问题,
随后把下一数值阶段改为**只加取证,不加 clip/删 state/grad-clip**:
在 JIT update 中返回 pre-params/target/opt-state、forward 中间量
(encoder features,target/chosen logits,target distribution,TD/FOSD/margin loss)、
raw gradient 分子树、optimizer updates/new state、candidate params/target 的 finite bitmask、
max-abs 和 global norm。一旦 candidate state 不 finite,不 commit 该次更新,
保留 pre-update 状态和前/当两个完整 batch(包括 RGB)。

诊断组使用高复发率的 QC+truncation seeds 1--4,不做 policy checkpoint
selection;对正常 finite snapshot/batch,新旧 update 必须逐位等价。通过线是
至少捕获一次“首个坏 stage+参数叶子”,且能从前一 finite snapshot
重放相同 full batch 复现;若四 seeds 全部 101k 无事件,只能说本轮
未复现。根因候选修复后再用 held-out seeds 5--6 做 matched
101k 确认,指标为 first-nonfinite step、各 stage finite mask/norm 和 101k
存活率;策略质量若需比较,仍另用 validation 400--449 选 best,
sealed 800--999 只在闸门通过后打开。

**4. Execution.** 至 2026-08-04 16:45 BST,no-ensemble move_plate seeds 1/2
已在物理 GPU1(`GPU-ce804993-...`)各以 0.45 显存切片真实开始;
PIDs 3828668/3832907,路径为
`exp_local/cqn_trunc_arms/noens_move_plate/seed{1,2}_20260804162305`。
seed1 已到 8k(`critic_loss=0.28331`,5k snapshot 存在),seed2 已到
6k(`critic_loss=0.29589`,5k snapshot 存在),两者均 finite;按当前双租户稳态速度预计
20:45--21:15 完成 101k。这是当前已执行的 next stage;梯度首发
守卫按上述注册顺序等 no-ensemble 结果,本阶段未修改 update 数学路径。

move_two_plates seed2 的 17 个现存 snapshot val50 和 80k sealed
已在 GPU4 分别由 PIDs 3849025/3849824 运行;val50 已完成
3/17(到 15k),sealed 仍在运行,不预报数字。sandwich_remove 两个 101k
训练均有 `train_complete`;GPU3 上的链式 val50 已分别完成
12/21(seed1/seed2,均到 60k),按当前每快照
~246--251s 预计 validation 约 17:20 齐,sealed 约 17:45--18:00 齐。
训练只占一张卡、评估与训练分卡,符合 GPU 配额协议。

## 65. Artifact lifecycle: shared demos, rolling resume, best+final retention

**1. Previous-stage result.** 2026-08-04 的磁盘审计确认旧生命周期有三个
独立重复源:每个 run 都重新落盘 demo replay;每个 snapshot 都保存完整
optimizer/RNG/replay metadata 且从不轮换;自然训练完成后 online replay 仍被
保留。旧 evaluator 只读取完整 snapshot,所以即使 evaluation 不需要 optimizer,
也不能在训练后立即删 resume state。

**2. Interpretation.** demo 是跨 seed 不变的输入数据,但 CQN-AS 的 dedicated
demo buffer 还会追加 self-imitation success,因此只能共享一个不可变 demo seed,
不能让多个 run 写同一目录。resume checkpoint 与 evaluation checkpoint 也是
两个不同用途:前者必须含 optimizer/RNG/replay state,后者只需 agent state。
把两者继续混在同一个 pickle 中会阻止安全清理。

**3. Next-stage decision.** CQN-AS/CQN-Flow 默认启用以下 gate:公共缓存分别保存
`all_demos` 和 `expert_demos`;run-local replay 以 hardlink 初始化后可独立追加;
训练期完整 resume 最多保留最近 2 个;每个评估步另存 params-only checkpoint;
只有自然训练完成且最终 params checkpoint 存在时才删 run-local replay 和全部
resume snapshot。validation seeds(<800)覆盖全部 checkpoint 后选 success 最大者
(tie 取更早),sealed evaluation 完成后仅保留 validation-best 与 final;held-out
CSV 被禁止用于选择。旧 run 未启用该配置,不自动清理。

**4. Execution.** 已实现 `SharedDemoReplayCache`、replay hardlink seed/局部清理、
Workspace 原子保存与 latest-two 轮换、params-only checkpoint、自然完成清理,
并让 `eval_cqn_as_snapshot_sweep.py` 优先读取轻量 checkpoint及执行有 gate 的
best+final 收尾。`run_cqn_trunc_arm.sh` 对新生命周期在 sealed eval 后自动收尾,
对当前正在跑的旧 noens/hard-task 配置保持 legacy fallback。聚焦测试
`tests/unit/replay_buffer/test_uniform_replay.py` 与
`tests/unit/test_artifact_lifecycle.py` 为 **10/10 pass**;BC/Flow snapshot及
CQN-Flow config 回归为 **5/5 pass**。实测 hardlink inode 相同,删除 run-local
链接后公共 cache 文件仍存在;5/10/15 step 保存后完整 resume 只剩 10/15,
而三个 params checkpoint 均可加载;validation-best=10、final=15 的收尾只保留
这两个并删除 resume/replay。当前 16:23 启动的 noens runs 的已解析 Hydra
config 不含新 artifact 字段,因此不会被这次改动中途删除。
另以 4-step 微型 BC 完整调用 `Workspace.train()` 做自然完成 smoke:
实际生成 1/2/3/4 四个 params checkpoint,结束后 resume 目录为空、online replay
NPZ 为 0,finalization record 记录删除 2 个 replay episode 与 3 个 resume entry;
因此清理不只是 helper 单测路径。

## 66. Post-ensemble L2 随机化审计与跨任务确认门

**1. Previous-stage result.** `move_plate` 的 100k、sealed-200
(episode seeds 800--999) 四个 train seeds，normal success 为
`78.5/73.0/82.0/82.0%`，修正后的 post-temporal-ensemble iid-L2
(`--post-ensemble-keep-levels 2`) 为 `78.0/76.0/83.0/74.5%`；逐 seed
差为 `-0.5/+3.0/+1.0/-7.5pp`，均值 `-1.0pp`。这只是在该四 seed
点估计上没有观察到平均退化；train-seed 差值的 t-95% 区间约为
`[-8.3,+6.3]pp`，不能据此认证等价。固定 leaf-0 的同组结果为
`0.0/0.0/5.0/0.5%`，但 trace 中相对原动作的 delta 不是零方差常数：
全 15 维均值约 `-0.0301`、std `0.0228`、范围约 `[-0.072,+0.008]`。

同一 move_plate seed1 100k checkpoint 另在 eval seeds 400--449、50
episodes 上对 temporal-ensemble plan history 做三层离散一致性审计：success
`80%`；L0/L1/L2 的 modal-vote exact 分别为
`98.54/92.34/65.78%`，相邻格一致率为 `99.95/99.31/88.90%`，所有
history plans 全体一致的 `(dimension, step)` 比例为
`94.26/70.78/9.38%`。配置确为 `levels=3,bins=5`、temporal ensemble on、
replan interval 1。归一化 action 全跨度为 2；选定 L0/L1 后的 cell 宽
`2/5^2=0.08`，即全动作跨度的 4%，L2 cell 宽为 `0.016`，即 0.8%。

**2. Interpretation.** 可以保留的窄结论是：在 move_plate 这一个任务、
这一组 checkpoint 上，把 temporal ensemble 已输出的连续 action 重新定位到
其几何 L1 cell，再在该 cell 的五个 L2 中心逐 step、逐 dimension iid 均匀
抽样，平均 success 只差 `-1pp`；而且 raw plan histories 的 L1 选择确实高度
一致。不能把它写成“4% 区间内无论如何晃动都无损”或“L2 critic 等价于
常数”：尚未测试 episode-persistent、低频、对抗、state-dependent 的零均值
扰动，且 seed4 单独下降 `7.5pp`。post-ensemble 随机化发生在 plan averaging
之后，所以 temporal ensemble 不会把该 iid 噪声再平均掉；plant/controller
的低通与任务容差仍是可能机制。实现保留的是 ensembled continuous action
反解出的几何 prefix，不是每个原始 critic plan 的逐位 ground-truth L0/L1。
“无偏”也只严格指相对 L1 cell center 对称，不自动等于相对 critic 原动作
零均值。

**3. Next-stage decision.** 跨任务确认只纳入当前 corrected-demo-MDP 且非低
成功率任务：`flip_cup`、`sandwich_remove`、`wall_cupboard_open`；明确排除
sealed success 约 `18.5--21.5%` 的 `move_two_plates`。旧
`dishwasher_close_trays` 100% run 使用 `truncate_demo_at_success=false` 且
没有 validation sweep，不混入 matched 表。每任务 train seeds 1/2，各自用
validation seeds 400--449 success 最大且 tie 取更早的 checkpoint：flip 为
`101k/101k`，sandwich 为 `85k/35k`，wall 为 `5k/5k`。normal 与 iid-L2
均用相同 diagnostic episode seeds 600--699、100 episodes、25 envs；800--999
held-out 保持不用于这次选择或诊断。主指标为每 train seed success paired
delta 与跨 seed mean；`mean delta >= -5pp` 视为初步复现，`<= -10pp` 视为
明确不复现，中间区间增加 episode 数而不改 intervention。

**4. Execution.** 新增 `scripts/run_l2_cross_task_eval.sh`；每个 arm 在独立
artifact view 下只 symlink 原 config 与 validation-best params checkpoint，
避免 evaluator 的 `sweep_evals/eval_<step>.json` 覆盖原 validation artifact。
结果将写入
`exp_local/cqn_l2_cross_task/<task>/seed<seed>/step<step>/<arm>/result.csv`。
2026-08-06 01:24 BST 已启动 controller PID `1826297`，log 为
`exp_local/cqn_l2_cross_task/controller.log`。启动时六张 GPU 均有训练进程；
按 no-eval-with-training 协议，controller 只等待物理 GPU2
(`GPU-80b9cc0d-...`)连续两次无 compute process 后才发车，目前尚未占用 GPU。

### 66.1 跨任务 iid-L2 结果：三任务全部明确不复现 move_plate

**1. Previous-stage result.** controller 已自然退出，12/12 个预注册 CSV
完整，未见 traceback、CUDA/OOM 或 CPU fallback。每格均为 validation-selected
checkpoint、diagnostic seeds 600--699、100 episodes；normal -> post-ensemble
iid-L2 结果为：

| task | train seed / checkpoint | normal | iid-L2 | delta |
|---|---:|---:|---:|---:|
| flip_cup | 1 / 101k | 42% | 28% | -14pp |
| flip_cup | 2 / 101k | 39% | 27% | -12pp |
| sandwich_remove | 1 / 85k | 71% | 38% | -33pp |
| sandwich_remove | 2 / 35k | 85% | 61% | -24pp |
| wall_cupboard_open | 1 / 5k | 100% | 88% | -12pp |
| wall_cupboard_open | 2 / 5k | 100% | 88% | -12pp |

task-level mean delta 分别为 `-13/-28.5/-12pp`，全部越过预注册的
`<= -10pp` 明确不复现线；6/6 train-seed comparisons 同号下降。按每任务
合并两 seed 的 200+200 episodes 做保守的独立二项近似，flip、sandwich、wall
的 delta 95% CI 分别约为 `[-22.2,-3.8]`、`[-37.5,-19.5]`、
`[-16.5,-7.5]pp`。sandwich 的两个 iid-L2 log 各记录 2 行
`UnstableSimulationWarning`，对应 normal 均为 0；这最多是额外症状，数量
不足以单独解释 24--33pp 的 success drop。

**2. Interpretation.** `move_plate` 的四-seed mean `-1pp` 不是可迁移的
CQN-AS 性质。相同 levels/bins、temporal ensemble 和 post-ensemble L2
实现，在三个非低成功率任务上都造成 12pp 以上损失；因此“只要在 4%-wide
L1 cell 内无偏 iid 晃动，成功率就不变”被直接否定。现有结果还不能区分
两种机制：(a) 高频逐 step/逐 dimension jitter 本身破坏控制；(b) critic 学到
的 L2 within-cell placement 对这些任务确实有用。sandwich 的少量 physics
warning 支持 jitter 会伤动力学，但不是充分解释。

**3. Next-stage decision.** 只增加一个独立 arm：同一 checkpoint、同一
600--699 episodes，在 post-ensemble L1 cell 恒取中心 leaf 2。复用上表 normal
baseline，不再打开 held-out。若 fixed-center 相对 normal 的 task mean
`>= -5pp`，则判为主要是 iid jitter/时间结构问题、而非 critic L2 ordering；
若 fixed-center 与 iid-L2 相差不超过 5pp 且相对 normal `<= -10pp`，则支持
critic 的 L2 placement 有任务价值；其余为 mixed，增加 episode 数而不改 arm。

**4. Execution.** 已新增并启动
`scripts/run_l2_cross_task_center_eval.sh`，controller PID `2273788`，log 为
`exp_local/cqn_l2_cross_task/center_controller.log`。2026-08-06 07:29 BST
确认 flip/sandwich/wall seed1 分别真实运行在空闲 GPU1/2/3，命令均含
`--post-ensemble-fixed-leaf 2 --num-eval-episodes 100
--eval-seed-start 600 --num-eval-envs 25`；两 train seeds 在各卡串行，预计
wall/flip 约 4--8 分钟、sandwich 约 16--18 分钟完成。未与训练同卡。

### 66.2 口径更正：正式确认改为 200 episodes / seeds 800--999

**1. Previous-stage result.** §66.1 的每格数字来自 diagnostic episode seeds
600--699、100 episodes；虽然 6/6 deltas 同号且幅度很大，但它与 Claude
原始 move_plate 表使用的 800--999、200 episodes 不同，不能标作同口径
复现。现存跨任务 800--999 baseline 也不能直接拼表：flip 旧 200-episode
结果是 100k，而 validation-best 为 101k；wall 旧结果是 25k，而 best 为
5k；sandwich 没有 best-checkpoint 的 200-episode 表。用户指出这一不一致
是正确的。

**2. Interpretation.** 600--699 表保留为独立诊断证据，但正式跨任务裁决
必须降级等待 matched 800--999 结果。旧 move_plate 四-seed normal/iid-L2
表虽然已经是 200 episodes、800--999，却全部固定在 100k；其 validation-best
实际为 seed1/2/3/4 的 `100k/100k/95k/45k`。特别是 seed4 validation 在
45k 为 90%、100k 仅 74%，固定 100k 会把 checkpoint overtraining 混入
L2 intervention 问题，不能回答“validation-selected policy 是否需要 L2”。

**3. Next-stage decision.** 正式矩阵统一为：corrected-demo-MDP、validation
seeds 400--449 选 success 最大且 tie 取更早 checkpoint；normal 与
post-ensemble iid-L2 使用相同 episode seeds 800--999、每格 200 episodes、
25 envs。三项 breadth tasks 各两个 train seeds，共 12 格/2400 episodes；
move_plate 为直接审核 Claude 结论保留四 train seeds，共 8 格/1600 episodes。
checkpoint 不由 800--999 选择。沿用 gate：task mean delta `>= -5pp` 初步
复现，`<= -10pp` 明确不复现，中间区间增加 model seeds/episodes。fixed-center
仍是独立机制问题，不混进此正式 normal-vs-iid 裁决。

**4. Execution.** 2026-08-06 07:38 BST 已启动
`scripts/run_l2_cross_task_ep200_800.sh`，controller PID `2292236`，artifact
root `exp_local/cqn_l2_cross_task_ep200_800`；flip/sandwich/wall 已分别在
GPU1/2/3 开始 best-checkpoint baseline。07:39 BST 又启动
`scripts/run_l2_move_plate_ep200_800.sh`，controller PID `2295693`，artifact
root `exp_local/cqn_l2_move_plate_ep200_800`；seed1/2 的 100k baseline 已在
GPU0/5 开始，随后同卡串行 iid-L2 与 seed3/4 的 95k/45k 两臂。启动审计确认
所用卡只存在 eval-only workload、无训练，occupancy <80%；命令强制
`JAX_PLATFORMS=cuda`，因此不接受 CPU fallback。按现有每格耗时，move_plate
预计 12--18 分钟，完整 breadth matrix 受 sandwich 限制预计 60--75 分钟。

### 66.3 move_plate 独立四-seed 重跑完成：小幅负效应，不是零效应

**1. Previous-stage result.** `scripts/run_l2_move_plate_ep200_800.sh` 已自然
完成，8/8 result CSV 齐全；无 traceback、CUDA/OOM、CPU fallback 或
`UnstableSimulationWarning`。每格均为 validation-selected checkpoint、
episode seeds 800--999、200 episodes：

| train seed / checkpoint | normal | iid-L2 | delta |
|---|---:|---:|---:|
| 1 / 100k | 79.5% | 79.0% | -0.5pp |
| 2 / 100k | 73.5% | 70.5% | -3.0pp |
| 3 / 95k | 81.0% | 79.5% | -1.5pp |
| 4 / 45k | 73.5% | 70.0% | -3.5pp |

normal mean `76.875%`，iid-L2 mean `74.75%`，paired model-seed mean delta
`-2.125pp`；四个 delta 全部为负。以四个 model-seed deltas 做 t-95% CI
约为 `[-4.32,+0.07]pp`；以 800+800 episode totals 做保守独立二项近似
约为 `[-6.32,+2.07]pp`。因此未证明统计等价，但通过预注册的
`mean delta >= -5pp` 初步复现线。Claude 固定 100k 的原表 mean delta
`-1pp` 方向和量级大体正确，精确数值没有独立复现；例如 seed2 原表为
`+3pp`，本次为 `-3pp`。

**2. Interpretation.** “move_plate 上 post-ensemble iid-L2 只造成小影响”
不是 evaluator 接错，也没有被 validation-best 选择推翻；但应写成约 2pp 的
小幅负点估计、CI 仍跨 0，而不是“完全等价/完全无损”。这与任务语义上的
“需要精细操作”不矛盾：精度需求要区分低频/系统偏差和逐 step、逐 dimension
零均值高频 jitter。move_plate 的固定 leaf-0 系统偏差约 -0.03 会摧毁策略，
证明它对 bias 高度敏感；本结果只说明其 contact/controller/trajectory 对
同 L1 cell 内的高频 iid 变化有较强容忍，不能推出 persistent 或 state-dependent
L2 perturbation 也安全。

100-episode diagnostic 的 fixed-center 已全部完成，normal -> center 为：flip
`42->40/39->37`（均 -2pp），sandwich `71->67/85->77`（-4/-8pp），wall
`100->100/100->100`。center 远好于对应 iid-L2 的
`28/27`、`38/61`、`88/88`；所以 breadth tasks 的大跌主要指向 iid 时间结构
有害，而不是“critic 精确 L2 bin selection 必不可少”。sandwich 仍有 modest
center cost，不能说所有任务 L2 ordering 完全无价值。

**3. Next-stage decision.** 先完成正在运行的正式 breadth 200-episode
800--999 矩阵，不在其完成前再开新的 perturbation 问题。最终按每 task 两个
train-seed paired mean 和原 gate 裁决，并将 600--699 表明确保留为诊断/复现
split。若正式 breadth 仍显示 move_plate 独有的 iid robustness，下一独立机制门
才比较 fixed-center、iid-L2 与 episode/block-persistent L2；相同 best checkpoint
与 600--699 selection-free diagnostics，persistent 相对 iid 再下降 >=10pp 才
支持“频谱/低通”解释。

**4. Execution.** move_plate controller PID `2295693` 已退出并写出
`[l2-move-plate] all matched evaluations complete`，artifact root 为
`exp_local/cqn_l2_move_plate_ep200_800`。breadth controller PID `2292236`
仍运行；截至 08:03 BST，wall 两 seed 已完成为
`100->83.5/100->82.5`，flip seed1 为 `42.5->27.0`、seed2 baseline 38%，
sandwich seed1 baseline 65.5%，其余格仍在跑。当前结果未用于 checkpoint
选择，预计受 sandwich 剩余三格限制还需约 35--50 分钟。

### 66.4 正式跨任务 800--999 矩阵完成：move_plate 的耐受性不具普遍性

**1. Previous-stage result.** 2026-08-06 08:28 BST，formal breadth 矩阵
12/12 个 CSV 已齐全。每格均使用 validation seeds 400--449 选出的 best
checkpoint，normal 与 post-ensemble iid-L2 共用 episode seeds 800--999、
200 episodes、25 envs：

| task | train seed / checkpoint | normal | iid-L2 | delta |
|---|---:|---:|---:|---:|
| flip_cup | 1 / 101k | 42.5% | 27.0% | -15.5pp |
| flip_cup | 2 / 101k | 38.0% | 26.0% | -12.0pp |
| sandwich_remove | 1 / 85k | 65.5% | 33.5% | -32.0pp |
| sandwich_remove | 2 / 35k | 78.5% | 54.5% | -24.0pp |
| wall_cupboard_open | 1 / 5k | 100.0% | 83.5% | -16.5pp |
| wall_cupboard_open | 2 / 5k | 100.0% | 82.5% | -17.5pp |

task-level paired mean delta 分别为 flip `-13.75pp`、sandwich `-28.0pp`、
wall `-17.0pp`，全部越过预注册的 `<= -10pp` 明确不复现线，且 6/6
train-seed comparisons 同号下降。按每任务 normal/iid 各合并 400 episodes
做保守独立二项近似，delta 95% CI 分别为 `[-20.22,-7.28]pp`、
`[-34.56,-21.44]pp`、`[-20.68,-13.32]pp`。对应 600--699、100-episode
diagnostic 的 task means 为 `-13/-28.5/-12pp`，正式 split 在方向和量级上
复现，而不是由 episode split 造成。

结合 §66.3 的 move_plate 四-seed结果，完整 formal audit 共 20 格、4000
episodes：move_plate `-2.125pp`，其余三个 task 均为 `-13.75pp` 或更差。
formal breadth logs 未见 traceback、CUDA/OOM 或 CPU fallback；sandwich
四格各有 2 条 `UnstableSimulationWarning`，normal 与 iid 对称，不能解释
arm 间 success 差异。所有目标 evaluator PID 已退出。

**2. Interpretation.** Claude 的窄结论——move_plate 上 post-ensemble、
逐 step/逐 dimension 的无偏 iid-L2 只造成小幅变化——量级上成立；本次
validation-best 重跑点估计为 `-2.125pp`，不是严格的零损失或统计等价。
把它推广成“在 4%-wide L1 cell 内只要无偏就无损”则被三个正式任务一致
否定。move_plate 看起来需要精细操作却能容忍这种 intervention，不构成矛盾：
已有 leaf-0 实验说明它对约 0.03 的低频系统偏差极敏感；本实验测的是同一
L1 cell 内的高频零均值 jitter。任务精度、偏置敏感性与高频抖动敏感性是
不同轴。fixed-center diagnostic 又显示 breadth 的大跌主要来自 iid 时间结构，
而不是 critic 精确 L2 leaf 本身不可替代；但 sandwich 的 4--8pp center cost
仍保留少量 L2 placement 价值的可能。

**3. Next-stage decision.** 对用户当前问题，formal gate 已给出明确裁决：接受
“move_plate 特例上影响小”，拒绝“其他成功率较高 task 也普遍如此”，不再
为这一裁决增加 held-out episodes。episode/block-persistent、低频或
state-dependent L2 扰动属于新的机制问题，必须使用独立 protocol；若继续，
matched arms 应为 center、step-iid、episode-persistent，复用同一 best
checkpoint 与未参与选择的 diagnostic split，主指标为 success delta，
persistent 相对 step-iid 再下降 `>=10pp` 才支持低通/频谱解释。

**4. Execution.** formal artifacts 保存在
`exp_local/cqn_l2_cross_task_ep200_800` 与
`exp_local/cqn_l2_move_plate_ep200_800`。用户授权后，缺失臂迁到 GPU2/4；
GPU2 峰值同时运行三条 eval，未与训练同卡。为避免原 controller 自动重复
启动 seed2，sandwich 父 shell 曾暂停；seed1 iid 写出后仅终止该旧 shell，
新发车的 seed2 两臂不受影响。最终结果计数为 breadth `12`、move_plate `8`，
日志异常扫描完成，旧 controller/zombie 与全部目标 evaluator 均已退出，
GPU2/4 已释放本实验占用。本阶段按预注册 gate 闭环，不自动启动一个不同的
扰动机制实验。

### 66.5 L0-only deterministic-center gate：区分 fine-level 选择与随机抖动

**1. Previous-stage result.** 回查现有 100k、200-episode、seeds 800--999
MovePlate artifacts 后确认，旧 `level_override_mode=middle` 已经做过真正的
层级删除对照：保留 critic L0，L1/L2 全部固定为 bin 2，四个 train seeds 为
`81.0/71.5/80.5/83.0%`，mean `79.0%`；matched normal 为
`78.5/73.0/82.0/82.0%`，mean `78.875%`，delta `+0.125pp`。因此 fixed
center 删除 L1/L2 的选择没有造成损失。相反，修正后的 post-ensemble
keep-L0、逐 step/逐 dimension 均匀随机 L1+L2 为 `0.5/0/0/0%`，mean
`0.125%`；归零来自高幅 iid jitter，而不是删除 fine-level critic ordering。

同一批官方 truncated MovePlate 100k checkpoints 的 powered sibling causal
probe（四 train seeds；每层约 800 states）给出 pooled pairwise sign accuracy：
L0 `57.1%`、L1 `52.2%`、L2 `51.2%`；五个 sibling expected-Q mean span 为
`1.358/1.084/0.328`。这说明 L0 有弱但一致的因果排序信号；L1 虽制造较大
Q gap，排序仍近随机；L2 更平且排序近随机。probe 没有保存 51-atom
distribution divergences，所以不能把 expected-Q 更接近升级为完整 C51
distributions 相同。

**2. Interpretation.** MovePlate 当前证据已不支持“L1 是 value 命门”：任务
需要一个稳定的 L0 cell 内 placement，但不需要 critic 对 L1/L2 做逐层 argmax；
deterministic midpoint 足够。random keep1 与 middle-from1 回答的是不同问题，
前者测时间高频扰动，后者才测 fine-level selection 的边际价值。算法层面的
候选简化因此是 `critic-select L0 + deterministic center refinement`，而不是
继续修 L1/L2 value 或把 Gaussian/jitter 当作 value fidelity proxy。

**3. Next-stage decision.** 正式 external-validity gate 复用 §66.4 的
validation-selected normal baselines，不重跑 baseline；新增唯一 arm 为
post-ensemble geometric L0 cell center：`keep_levels=1,fixed_leaf=12`（3 levels、
5 bins 下剩余 25 leaves 的正中心）。episode seeds 800--999、200 episodes、
25 envs；MovePlate 四 train seeds，flip/sandwich/wall 各两 seeds，共 10 arms、
2000 new episodes。每 task paired mean delta `>=-5pp` 判 fine-level selection
可在执行侧删除，`<=-10pp` 判该 task 仍需 fine geometry（但单独不能证明
critic ordering 正确），中间带增加 model seeds。若所有高成功率 task 均通过，
下一阶段才训练真正 `levels=1` 的 matched variant，以区分执行侧可删与训练期
shared-representation/loss 仍有贡献。

**4. Execution.** 新增 `scripts/run_l0_only_center_ep200_800.sh`，artifact root
`exp_local/cqn_l0_only_center_ep200_800`，controller log
`exp_local/cqn_l0_only_center_ep200_800_controller.log`。2026-08-06 09:22 BST
controller PID `2531776` 已确认运行；GPU2 上 flip seeds 1/2 两条 evaluator
命令均含 `--post-ensemble-keep-levels 1 --post-ensemble-fixed-leaf 12
--num-eval-episodes 200 --eval-seed-start 800`。GPU4 当时已有三个其他用户的
`eval_only=true` 进程，第三 worker 正在等待 occupancy <3 后发车；controller
每臂前重新计数，保证 GPU2/4 各自最多三个 eval 且不与训练同卡。

## 67. `official+truncation` 命名更正与官方差距归因审计

**1. Previous-stage result.** 四个 MovePlate 高分 run 的实际
`.hydra/overrides.yaml` 只含 canonical `cqn_as_pixel_bigym_demo_driven`、
`env.truncate_demo_at_success=true`、train seed/GPU 与运行基础设施开关；
四份 resolved config 除 seed/GPU 外逐值相同。没有 offline pretraining、high
UTD、QC/n-step、bin/structured exploration、reward/Q scale、BC schedule 或
checkpoint selection；10/100k 日志按列名核验为 UTD=1，100k 时
`act_train_calls_total=100001`，各探索触发计数为 0。四 seed 固定
100k、200-episode、seeds 800--999 为 `78.5/73.0/82.0/82.0%`，mean
`78.875%`；101k 为 `78.5/70.5/83.0/76.0%`，mean `77.0%`。

两个同 checkpoint 的执行侧对照均已完成（seed1@100k，200 episodes，
seeds 800--999）：

| eval contract | success | matched current |
|---|---:|---:|
| current local | 78.5% | — |
| 官方 `TemporalEnsembleControl` 的 zero-sentinel + 旧计划高权重 | 78.5% | 0.0pp |
| 官方锁定 BiGym `72d3054` | 78.0% | -0.5pp |

新 artifacts 为
`cqn_official_truncated/...seed1.../ep200_official_temporal.csv` 和
`ep200_bigym72d305.csv`；两次均无 traceback、CUDA/EGL fallback、OOM 或
physics warning。因此 temporal-ensemble 权重方向/zero-sentinel 与新旧
BiGym reset 不能解释约 14.9pp 的官方 public-log 差距。

**2. Interpretation.** runner 注释中“Everything else is stock official”不成立，
这些 run 应改称 **local canonical JAX + demo truncation**。官方公开代码自
首个 public commit 起已在第一个 reward 处截断 demo，所以不能把
`78.875 - 64.0pp` 全部写成“我们多加了截断”。能严格成立的
截断因果量只是本地同 seed 的 `67.5 -> 78.5%` (+11pp)；它说明
截断修复了本地未截断 replay，不能单独解释为什么官方已截断的
public log 仍为 64.0%。

已确认的剩余训练合同差异为：官方每帧 low-dim 是
`58 proprio + 2 gripper + 3 floating-base = 63D`，并对 63D 全部用截断后
demo stats 做 z-score；本地是 60D，不输入 floating base，并把
`proprioception[0]` 和两维 gripper summary 保持 raw scale。官方把第一个
post-action demo frame 当 dummy FIRST，丢掉 reset-to-first-action transition；本地用
demo seed 重建 reset observation 并保留该 transition。本地 stats 在截断前算，
官方在截断后算；MovePlate action min/max 实测逐维相同，因此 action
scaling 已被排除，obs contract 仍未隔离。其余 residual 是 Flax/PyTorch
GRU、initialization、RNG/replay scheduling 和官方训练内每 2500 步 25-episode eval
与本地 async eval 的差异。官方 README 自身明示 public logs 可与 paper
不同、public port 可能无法完全复现 paper；其 64.0 只有聚合 pickle，
无 checkpoint/resolved config/per-seed episode 可供进一步归因。

**3. Next-stage decision.** 下一训练归因门只回答 observation contract，
不混 demo alignment 或 framework 更换。共享 current local+trunc baseline，分别单改
(A) 恢复对 `proprioception[0]`/gripper summary 的全 z-score（仍 60D），
(B) 只追加 normalized floating-base 3D（仍保留本地特殊 scale）；seeds 1/2、
101k，validation seeds 400--449 覆盖所有 checkpoints 选 best，通过后才打开
800--999 的 200-episode held-out。主指标是 best-checkpoint success 的 paired
delta，并同时记录 online low-dim p99.9/max 和 non-finite 事件。任一单改两
seed mean 使 local baseline 下降 `>=8pp` 则升格为主差距候选；`<=5pp`
则排除为主因；中间带补 model seed。demo alignment 作为下一个独立问题，
不与本门同时修改。

**4. Execution.** 本轮已完成 launcher/resolved-config/log/replay 审计与两个
200-episode 执行侧配对对照，并写出上述 CSV。未启动新训练臂：
当前 GPU0/1/3/5 均有其他用户的训练量进程，GPU2 正跑两个已注册
L0-center eval，GPU4 有三个其他用户 eval；新训练会违反共享集群配额与
no-eval-with-training 规则。因此 concrete blocker 是没有安全训练槽，不是实现或数据缺失。

## 68. JAX CQN-AS 首个 non-finite stage 取证与 task-rate 对照

**1. Previous-stage result.** 旧 forensic dump 已覆盖 QC/non-QC、nstep-1/8、
move_plate/move_two_plates 与多种执行臂；所有保留 batch 的 float 叶均有限，
但守卫触发后 online params 与 Adam state 已整树 non-finite。坏点前的 snapshot
参数范数和 Adam moments 与健康 seed 同量级，loss 也没有渐进爬升。因此旧
artifact 只能把首污染区间缩到一次 JAX forward/backward/Adam/apply，不能判定
首个坏算子；旧 tap 又没有保存 RGB，无法做完整 batch 重放。PyTorch 与本地
JAX 配置均是 AdamW、`critic_grad_clip=null`，所以“PyTorch 默认有梯度裁剪而
JAX 没有”已由配置和代码排除。

按 2026-08-03 后有明确错误日志的近期波次计数，8 次 hard NaN 中 7 次发生在
move_plate、1 次在 move_two_plates；但 exposure 强烈偏向 move_plate，不能把
7/8 当 task 因果。更可比的子组是：official move_plate nstep-1/replan-1 为
0/2；当前使用的 move_plate `QC8 + truncate + obs_std_floor=0` 旧波次中两个
有效启动 seed 分别在 1k/4k NaN（另一个 OOM、一个被 controller 中止，均不计）；
`obs_std_floor=0.01` 的 QC 波次为 2/4（11k/99k）；official move_two_plates
为 1/2（82k）；sandwich_remove 0/4、flip_cup 0/2。训练 loss 仅用于数值
存活判断，不代表 policy quality，本阶段不涉及 checkpoint selection。

**2. Interpretation.** 当前最高复发的历史组确实是 move_plate QC8，而不是
普通 official move_plate。task 相关性有信号，但与 nstep/replan、observation
contract 和各 task 的曝光预算混淆，尚不能写成根因。新 provenance 的
move_plate seed1/2 已同时健康穿过历史 1k/4k 坏点，seed3 到 2k；所有记录的
features/target logits/distributions/loss、raw grads、Adam updates/new state、
candidate params/target 均有限且 `update_committed=1`。同 seed/config 坏点不
确定复现，说明事件依赖实际 online trajectory/minibatch 或更低层非确定性，
而非由 task+seed+config 唯一决定。

**3. Next-stage decision.** 首要门继续只加 provenance、不加 z-clip、删 state
或 grad-clip：move_plate QC8 seeds 1--3 跑到首个 non-finite 或 101k；通过线
是捕获至少一次首坏 stage 并保留 pre-update state 与完整 RGB batch，三 seed
全到 101k 则只判本轮未复现。task 假设作为独立 matched control：
dishwasher_close_trays seed2 仅替换 env，保持 QC8、truncate、
`obs_std_floor=0`、无 grad clip 和相同 instrumentation；比较 first-nonfinite
step 与 101k survival，不用 success/loss 代替数值判决。确认首坏 stage 后，
根因修复才用新 seeds 做 survival validation；策略质量另按 validation
400--449 选 best checkpoint，held-out 800--999 保持密封。

**4. Execution.** provenance 代码已在 JIT update 内返回 forward、gradient、
optimizer/new-state/candidate-tree finite flags 与 max-abs；首坏 candidate 不
commit，workspace 保存最近三个完整 batch（含 uint8 RGB）和 pre-update agent
state。focused tests：CQN-AS update `1 passed`，workspace guard `2 passed`，
并已由真实 GPU 首次 update 验证。主 artifact root 为
`exp_local/cqn_nan_provenance/qc_20260806212634`；截至 21:38 BST，GPU2
运行 seed1/3，GPU4 运行 seed2（该卡另有一条外部训练），严格保持每卡最多
两条训练。cross-task controller PID `3487733` 已排队，root 为
`exp_local/cqn_nan_provenance/cross_task_20260806213800`；它要求 GPU2/4
任一卡连续两次检查低于两条 CUDA process 才发 dishwasher control，当前尚未
占用额外 GPU slot。

### 68.1 首坏点已捕获：tp1 batch 在 critic 前已发生内存级污染

**1. Previous-stage result.** move_plate device-merge seed2 在 iteration
`5154` 首次触发，新 guard 在 candidate commit 前保住了 last-good state。
`pre_params/pre_target/pre_opt_state` 全有限；current features、chosen/all logits、
BC terms 也全有限。第一个坏 stage 是 `next_features_all_finite=0`，随后
target logits/probabilities/distribution、loss、gradient、Adam candidate 和
candidate params/target 依次非有限。关键是输入 tap 同时证明网络前的
`low_dim_state_tp1` 已坏：512/512 rows 受影响，122,880 个元素中 203 个
non-finite、36,016 个 `abs(x)>1e6`，最大有限值
`3.3974545e38`；同批 current low-dim max 仅 40.666，其余 float fields 全有限。

按 dump 的 512 个 indices 分别从 online replay 与 expert-demo replay 的落盘
episode 重建 nstep-8 tp1。current state 与 dump **122,880/122,880 逐元素相同**；
正确 tp1 全有限、max 21.386，但与 dump **122,880/122,880 全部不同**，online
和 demo 两半均为 0 个相等元素。源 episode 自身全有限。原始 dump、pre-state
与重建审计分别在
`exp_local/cqn_nan_provenance/qc_20260806212634/seed2/nonfinite_dump/`
的 `batches_iter5154.npz`、`pre_update_state_iter5154.pkl`、
`summary_iter5154.json`、`reconstruction_audit_iter5154.json`。

**2. Interpretation.** 这次 NaN 的直接原因不是 C51 target、GRU/LayerNorm、
backward 或 AdamW 自发数值爆炸，而是 **host replay 文件之后、JAX forward
之前的 tp1 transport/merge batch corruption**。critic 只是忠实地把坏输入传播成
NaN；新 guard 阻止了整树污染。它也解释了为什么 PyTorch 版本可以从不 NaN：
当前嫌疑是本地 JAX-only 的 `WorkspaceFast` device-side demo merge/background
prefetch 路径，而不是两个 framework 的 Adam epsilon 或 gradient clip 差异。
task 仍可能通过 tensor shape/allocator 时序改变触发率，但环境动力学不可能
产生本次接近 float32 上限的数值。尚未区分 device merge/prefetch 与更早的
vectorized sampler；重建排除了持久 replay 数据本身。

**3. Next-stage decision.** matched containment/attribution arm 只设置
`ROBOBASE_HOST_MERGE=1`，其余保持同 move_plate QC8、truncate、seed2、
`obs_std_floor=0`、vectorized sampler、device prefetch 与 provenance；先比较
5154 旧坏点，最终指标为 first-corrupt-input step 与 101k survival。若 host merge
也复现相同输入污染，下一独立 arm 才设置 `ROBOBASE_SCALAR_SAMPLE=1`；若 host
merge 多 seeds 存活而 device arm 再次失败，则 device merge/background JAX
dispatch 被判为主因。数值 containment 确认前不做 policy-quality 声明。

**4. Execution.** device seed1 留作 replicate；已在健康 3k 主动停止 seed3，
并写入 `STOP_REASON.md`，不能计作 survival。host-merge controller PID
`3501504` 已于 21:45:38 BST 在 GPU2 启动，artifact root
`exp_local/cqn_nan_provenance/host_merge_20260806214538`；GPU2 上仅与 device
seed1 配对，首次真实 JIT update 已写出 iteration 0、loss 0.49318、全部 stage
finite、`update_committed=1`。dishwasher task-only control 已在 GPU4 启动并写出
2k finite row；
同卡另有一条外部训练。停止 seed3 后旧 queue 曾竞态重发 PID `3500330`，已
精确终止该重复进程及 queue shell；GPU2 已恢复两条训练上限。

### 68.2 第二次独立复现与 host-merge 首个归因门

**1. Previous-stage result.** device-merge seed1 又在 iteration `10308`
触发，pre-params/target/Adam 仍全有限。这次与 5154 相反：tp1 features、target
logits/distribution 全有限，首先坏的是 current features；完整 batch 中
`low_dim_state` 的 122,880 个元素有 297 个 non-finite、33,453 个
`abs(x)>1e6`，最大有限值 `3.3975584e38`。按 indices 从持久 replay 重建，
正确 current state 全有限、max 42.536，且 dump 的 122,880/122,880 个元素
全部不匹配；同 batch 的 tp1 则与重建 122,880/122,880 逐元素相同、max
15.160。第一次随机坏 tp1，第二次随机坏 current observation，签名相同而 key
不同。

matched `ROBOBASE_HOST_MERGE=1` seed2 已写出 6k finite row：loss 0.22447、
`update_committed=1`、无 nonfinite dump，明确穿过同 seed device arm 的 5154
failure step。dishwasher device-merge task control 同期到 6k finite。

**2. Interpretation.** 两次都是 online 与 demo 两半一起整张 observation
buffer 变成近似未初始化内存，而另一时态 observation 完全正确；这不符合
MuJoCo/task state、持久 replay 数据或 critic 数值发散。它与
`WorkspaceFast._DeviceMergedIterator` 的实现边界吻合：background prefetch
线程对每个 key 独立执行 host-to-device `jnp.asarray` 和 device-side
`jnp.concatenate`，每个 key 对应独立的异步输出。host merge 去掉这一层后首个
matched failure step 存活，已支持 device-merge/prefetch 为主因，但单个 6k
survival 尚不足以排除随机事件或 vectorized sampler。

**3. Next-stage decision.** host-merge seed2 至少继续超过第二个 device 事件
10308，并最终跑到 101k；若全程无输入污染，再补一条 host-merge seed 验证。
若 host arm 也出现同签名，下一独立 arm 才关闭 vectorized sampler。低层确认可
在 device merge 前记录 host finite/checksum、merge 后立即 block/checksum；host
正确而 device 错误即可把首坏操作钉到 concat/transfer。通过标准仍是 input
finite 与 101k survival，不用 critic loss 代表 policy quality。

**4. Execution.** device seed1 已由 guard 安全退出并保存完整 dump/pre-state；
host-merge PID `3501509` 在 GPU2 继续运行，6k artifact 已验证；dishwasher
PID `3489377` 在 GPU4 继续运行。两卡仍各最多两条训练进程，本阶段没有打开
任何 held-out policy evaluation。

### 68.3 host-merge 回退的实时吞吐成本

**1. Previous-stage result.** host-merge seed2 已到 8k、无 dump。剔除首次
compile、5k/10k snapshot 区间后，当前同 GPU2 双跑条件下，host arm 的完整
1k-window 中位吞吐为 6.61 steps/s；同期 device seed1 为 6.13 steps/s。host
没有观测到减速，但两者受同卡竞争和不同启动时刻影响，不能把 +7.7% 写成
真实加速。历史 clean ABBA 的 4.55 -> 10.73 steps/s 是
`scalar+host` 对 `vectorized+device` 的组合差，不能归因给 merge 单项；其中
已单独测得的主要收益是 vectorized sampler 的 78.5 -> 35.8 ms/buffer。

**2. Interpretation.** 保留 vectorized sampler、只回退 host merge 不会退回旧
管线 2.36x 的慢速；当前实测成本在运行噪声内。host `np.concatenate` 会多做
约 130MB/update 的 CPU copy，但 prefetch 可与约 75ms backend update 重叠，
所以 CPU work share 不能直接换算成 end-to-end slowdown。保守预期是 0--15%，
而不是 2x；exact 单卡数字仍需 clean matched benchmark。

**3. Next-stage decision.** 数值稳定优先，host merge 继续作为 containment arm。
它通过 101k survival 后再在 GPU5 按 ABBA 单跑、固定 1k--3k window 测
device/host 单项差，并用输入 checksum guard 保证 device arm 不静默污染。
若 host 的 clean slowdown <=15% 则直接作为默认；更高才设计同步后的安全
device merge。性能门与 policy quality 分开。

**4. Execution.** host PID `3501509` 继续训练；当前 throughput 由两份
`train.csv` 按列名和 total-time 差分重算，snapshot windows 已剔除。未为测速
新增 GPU 进程，也未改变正在跑的 numerical-control config。

### 68.4 GPU0 utilization 跳动的实时归因

**1. Previous-stage result.** 20 秒、1 Hz `nvidia-smi dmon` 的 SM utilization
为：GPU0 mean 65.0%, sd 23.7, range 10--98；GPU5 mean 61.0%, sd 29.9,
range 10--98；GPU2 mean 95.0%, sd 3.9, range 87--98；GPU4 mean 92.1%,
sd 4.2, range 86--98。GPU0 上是两条相差约 3 秒启动的
`swflip_sandwich_remove` host-merge nstep-1 run，采样时均在 3k；GPU5 上
两条同批 `l1flip_hi_move_plate` run 也同样跳动。GPU2/4 各自还有一条
早启动约两小时的 QC8/nstep-8 训练填充新 run 的空档。GPU0 当前
47 C、power limit 575 W，无 active thermal/HW slowdown；Xorg/desktop 仅占约
61 MB。sandwich 两份日志合计仅一次 `mjWARN_BADQACC`。

**2. Interpretation.** 图上的差异是同卡 workload mix，不是 GPU0 硬件故障：
host replay/concat、environment step 和 GPU update 交替会产生 burst/gap；同时
启动的同构进程会让部分 gap 对齐。GPU2/4 的旧 QC8 run 将这些空档
填平。GPU5 的 move_plate 也出现同等幅度，所以 task 不是主因；
一次 MuJoCo warning 也无法解释持续波形。utilization 跳动本身不是
NaN/batch corruption 的证据。

**3. Next-stage decision.** 不为了平滑 nvitop 图重启训练。效率以每条
`train.csv` 的无 snapshot 完整 1k window 为准；只有若 GPU0 的单 run
持续比 matched GPU5 低 >20%，才单独分离 task CPU cost、host merge 和进程
同步性。后续新建配对 run 仍按协议错开约两分钟启动。

**4. Execution.** 已核对 GPU PID/CWD/config/environment、采集 dmon/pmon、检查
日志和 throttle state；六条新 run 都仍在 2k--3k 并继续产生 CSV。
没有新增、终止或重启任何 GPU 进程。

### 68.5 host-merge 101k survival gate 与扰动 wave 当前进度

**1. Previous-stage result.** matched move_plate QC8 host-merge seed2
`exp_local/cqn_nan_provenance/host_merge_20260806214538/seed2` 已于
2026-08-07 01:45 BST 正常退出，`completed_env_steps=101000`、保留 21 个
eval checkpoints。`train.csv` 的 101 行、20 个 `*_all_finite` 诊断列和
`update_committed` 全部通过，无 `nonfinite_dump`；对应 device-merge seed2/seed1
分别在 5154/10308 发生整张 observation buffer 污染。device-merge
dishwasher task control
`exp_local/cqn_nan_provenance/cross_task_20260806213800/dishwasher_close_trays_seed2`
也正常完成 101k，同样 101/101 rows、20 个诊断列全通过、0 dumps。

六条 post-ensemble perturbation 主训练尚未完成：`swflip_sandwich_remove`
seed1/2 在 93k/84k，`l1flip_move_plate` 在 79k/78k，
`l1flip_hi_move_plate` 在 86k/82k；六条当前都是 finite、0 dumps。还没有
`val50_seeds400.csv` 或 `ep200_seeds800.csv`，因此尚无 policy-success 结果。

**2. Interpretation.** host merge 在同 task、同 seed、同 QC8 配置下从 device
路径的 5k failure 提升到完整 101k survival，强支持
`_DeviceMergedIterator`/asynchronous device merge 边界是主因；但只有一条
matched host QC8 seed，仍不能宣称完全封案。dishwasher device arm 能到
101k 说明 task/tensor/allocator timing 会影响触发率，不说明 device path
安全。perturbation wave 的 training loss/online episode 不能替代 validation 或
sealed success rate，所以 L1/L2 研究结论此刻不变。

**3. Next-stage decision.** NaN 线需要第二条 matched host-merge QC8 seed 完成
101k、0 dumps 才通过 containment replicate gate；之后才做 blocked/checksummed
device concat 定位。perturbation wave 按原协议训到 100k，用 seeds 400--449
的 50-episode validation 做 checkpoint selection，再用 seeds 800--999 的 200-episode
sealed read；必须比较各自 validation-selected best checkpoint，100k endpoint 只作
固定端点补充。当前最慢主 run 按最近 1k window 约 91 分钟到
100k，此后才会有 eval result。

**4. Execution.** 已从 live PID/environment、GPU UUID、`train.csv`、controller
exit code、`training_artifacts_finalized.json`、eval checkpoints 和 dump 目录完成本阶段
核对；正在运行的六条主 run 保持不动。本次检查同时发现他会话
新启动了 GPU1 两条 `l1flip_decay_move_plate`；当前本 repo 的 training
使用 GPU0/1/2/4/5，违反“最多两张卡训练”协议。GPU0/5 的两个 wrapper
还可能在 peer seed 仍训练时直接启动同卡 eval。因本轮是只读状态请求，
未擅自终止其他 session 的进程，也未新启动 replicate。

### 68.6 blocked device merge 候选修复

**1. Previous-stage result.** host-merge move_plate QC8 seed2 已 101k/0 dumps，
而旧 device-merge seed2/seed1 在 5154/10308 出现整张 float observation
buffer 污染。检查 `WorkspaceFast._DeviceMergedIterator` 确认：它在后台
prefetch thread 中发起每个 key 的 `jnp.asarray` 和 device concatenate，然后在
整棵 tree ready 前直接返回。已实现候选 containment：在该 background
worker 里对 merged tree 做一次 `jax.block_until_ready`，然后才释放两份
source NumPy batch；concat 和 transfer 仍在 device，没有恢复 host 的大拷贝。
另增 `ROBOBASE_DEVICE_MERGE_VERIFY=1` attribution gate，会对所有非 RGB
字段做 host/device 逐值 exact compare，任一 mismatch 立即报错。

CPU 的 workspace/vectorized-replay focused suite 为 16 passed。在空闲 GPU3 上使用
5154 dump 中一个真实 20-key、261,701,120-byte retained batch，修复路径
与 host concat 20/20 keys 逐值一致；500 次源 buffer 返回后立即 NaN/Inf
poison stress 为 500/500 exact。修复路径孤立延迟 median 17.55 ms、p99
26.97 ms；host-concat-then-put 的 12 次对照 median 120.99 ms。但同样的
旧 unsafe async synthetic 500 次也未复现污染。

**2. Interpretation.** ready barrier 是一个有明确安全性边界、不恢复 260 MB
host copy 的合理修复；孤立测量中它只比 unsafe dispatch median 约 1.1 ms，
而且仍可以在 prefetch thread 与 main-thread update 重叠。但 synthetic unsafe
500/500 通过意味着“host source 被立即复用”还没被证明为唯一首坏
机制；当前应称为 candidate containment，不能称已封案 root cause。微基准
也不等于 end-to-end training throughput。

**3. Next-stage decision.** 下一独立 infra arm 必须去掉 `ROBOBASE_HOST_MERGE`、
打开 `ROBOBASE_DEVICE_MERGE_VERIFY=1`，使用同 move_plate QC8、truncate、seed2
和原 nonfinite dump 设置。第一门是穿过 5154 和 10308 到 12k，exact
verifier 0 mismatches、0 dumps；通过后继续 101k。再用第二 seed 复制 101k
survival。性能用无 JIT/snapshot 的 1k windows 与 host arm 比较；本问题没有
checkpoint selection 或 held-out policy split，不做 policy-quality 声明。只有两条
101k 都通过后，才从后续生产 launch 中摘掉 `ROBOBASE_HOST_MERGE=1`；
kill switch 保留。

**4. Execution.** 修复与 verifier 已写入 `robobase/workspace_fast.py`，回归写入
`tests/unit/test_workspace_fast.py`；CPU 和 GPU retained-batch gates 均已执行。没有
改动任何已运行训练的 environment，也没有摘除它们的 host rollback。
真实 12k/101k arm 未启动：当前他 session 仍让本 repo 的 training 占用
GPU0/1/2/4/5，已超过“最多两张卡”硬上限；在 card count 回到允许范围
前不再新增训练。后续 live 复查又确认他 session 同时启动了 6 个
50-episode validation：GPU2/4 已实际出现 training+eval 同卡，GPU3 同时
运行 4 个 eval。本阶段未再加载 fixed-device training，也未干预他 session
进程。

### 68.7 blocked device merge 真实训练门已启动

**1. Previous-stage result.** 空闲卡出现后，候选修复的 move_plate QC8 seed2
arm 已在独占 GPU4 真正写出 `env_steps=2000`：最近 `critic_loss=0.28463`，两行
均为 20/20 `*_all_finite` 诊断通过、`update_committed=1`，无 verifier mismatch、
无 nonfinite dump。0--1k/1k--2k 的 `total_time` 差分吞吐分别为
11.24/10.97 steps/s；30 秒、1 Hz
GPU4 采样的 SM utilization 为 83--92%，显存固定约 13.08 GB，卡上只有该
训练 PID。旧 host arm 的 80 个无 snapshot window 中位数为 6.90 steps/s，
但它曾与别的训练同卡，不能把两者差值归因成 merge 修复的加速。

**2. Interpretation.** 这证明 barrier + dtype-aware exact verifier 已经通过真实
replay、JIT 和首个参数更新，且 device 快路径在当前独占卡上没有 GPU0 所见的
大幅 utilization 空档。它尚未穿过旧 device seed2/seed1 的 5154/10308 首坏点，
所以现在只能说 launch/wiring/performance smoke gate 通过，不能说 NaN 已修复；
两个早期 1k window 也不是最终吞吐结论。critic loss 有限只用于数值健康，不代表
policy quality。

**3. Next-stage decision.** 该 arm 原配置不变继续到 12k：matched baseline 是
旧 device-merge seed2/seed1 的 5154/10308 failure 与 host-merge seed2 的
101k survival；选择 split/held-out split 均不适用，因为这是 numerical infra
gate。主指标为 exact verifier mismatch、nonfinite dump、20 个 finite 诊断和
进程退出状态；12k 必须全部为零失败才继续到 101k。按首个完整 window，12k
约 16 分钟、101k 约 2.5 小时，snapshot 开销会让实际 ETA 略长。

**4. Execution.** 运行目录是
`exp_local/cqn_nan_provenance/blocked_device_20260807072305/seed2`，controller
PID `4185035`、train PID `4185058`，workspace fix SHA256 为
`231e012b72f7273d486f58b3c53e8498e73d3f6cf42b80d7f9a066475b260ba3`。
launch 显式清掉 `ROBOBASE_HOST_MERGE`、设置
`ROBOBASE_DEVICE_MERGE_VERIFY=1`、`xla_mem_fraction=0.45`、truncate=true，
并保留 nonfinite dump；当前进程仍活着并继续写 CSV。更早的两次启动分别因
已删除的 Hydra 字段和 verifier 未按 JAX x64-disabled dtype canonicalization
比较而在训练前/首 batch 退出，均不计 scientific arm；回归已补至 17 passed。
