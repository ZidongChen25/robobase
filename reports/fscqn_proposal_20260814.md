# 裁决与合成报告（adversarial judge, 2026-08-14）

## 1. 逐案攻击

### Design A（SC-CQN：learned mask + in-mask exploration）

**暗中重跑了谁**：它的 per-episode information 预算有一半押在 in-mask ε-bin exploration 上——这正是已被官方判死的 exploration 家族（A4/A14 +1.0 FAIL、combined −2.1/−1.5、coarse-only −36）。"那些都是在 BC veto 下测的"是假说，不是数据；§49 的 contrast 价值本身就出自这条死线。其次，behavior head 用 success-gated online 数据持续做 CE = Stage-41 growing-replay 的近亲——且这里 drift 的是 suppressor 本体，比 buffer drift 更致命。τ 0.85→0.2 的长 schedule 落在 anneal-A 家族的高方差区（85.0 vs 69.5，n=2 无 declaration）。

**anchor 是否成立**："off-manifold 时 head 的 entropy 升高、relative threshold 自动放宽"是零测量断言；softmax 网络在 off-distribution 上的常态是 confident-wrong 而非 entropy 上升。且这个 head 一直在被 online 数据训练，suppressor 自己回到了 training dynamics 里——与"frozen weights don't collapse"这条唯一直接证据背道而驰。

**improvement channel**：真实（无 hinge veto + in-mask reorder，headroom 52–58 vs 66–85 论证成立），但扣除 exploration 后它退化为"B + gated floor"；不扣除则 mask、gated floor、exploration 改造、τ schedule、online CE 五个 lever 同时上——正是 owner 不信任的 kitchen sink，四 seed 预算下无法归因。

### Design B（Stage-42/43 truncated port + frozen mask）

**暗中重跑了谁**：primary floor 变体在 truncated regime 给**所有** unseen bins（包括 mask 内 siblings）打 0-floor，executed bin 拿 γ^k∈[0.2,1.0]——这是在小尺度上手工重建 saturation crutch：electable 集合内部仍是"executed vs 全 0"的 uniform margin，没有可 reorder 的细结构。这正是 Stage 43 flat 的机制本身，所以 primary 形态是 parity-shaped；§4 的上升故事悄悄依赖被降级为 rider 的 mask-gated floor。同时整个 dense 组合在 graded-target regime 的最近亲属是 A18/A19（endpoints 34/36、26/26）——mask 挡得住 decode，挡不住 floor 梯度与 TD 在 loss 内的斗争。

**anchor**：与 A 同一个 off-manifold calibration 空洞，但 freeze 换来唯一的直接证据（collapse 是 training-dynamics 现象，frozen 权重免疫）；代价是 coverage 永不增长，τ 是仅有的 dial。

**improvement**：primary 弱、rider 强而未证。collapse-immunity 三案最强（双 suppressor + target-side BCQ）；实现最便宜（40–70 行）。

### Design C（CQN-RC：retrieval candidates）

**证据挪用（最重一击）**：0→52–58 来自 qselect，其候选是 checkpoint policy 的 **forward pass**（c65/c50，state-conditioned generalizer），不是 kNN 检索的 chunk——这个数字不能移植给 nonparametric retrieval。**暗中重跑**：(i) Stage 22–25 instance-candidate null（selection 0–9%）；(ii) 在 nearby-but-not-identical state 上开环执行别处录的 chunk = Stage-12 open-loop penalty + erratum-#4 的病理（cross-state replay 的 divergence 正是 v1 fence 的 off-manifold 分布，median distance 7.1–8.0），且从未有通过 zero-perturbation control 的先例；(iii) nstep3（−3.25）凭论证复活。63-dim proprioceptive 距离 ≠ 行为距离，precision-sensitive 任务会定价（L2 audit：sandwich −28）。improvement channel 与 instrument（non-1NN pick rate）是三案最干净的；infra 代价最大（DB + s′ 检索穿两个 replay buffer + jit 边界）。

### 评分（/5）

| | collapse-immunity | improvement | reference solved | impl-cost | novelty vs catalogue |
|---|---|---|---|---|---|
| A | 4 | 3.5 | 3.5 | 3 | 3 |
| B | **5** | 2 | 3.5 | **5** | 3 |
| C | 3.5 | **4.5** | 2 | 2.5 | **5** |

## 2. 最终提案：FS-CQN（Frozen-Support CQN-AS）

= B 的骨架 + A 的 mask-gated floor 升为 primary，删掉 A 的 exploration 与 online CE，删掉 C 的 retrieval。六个组件，无新 loss 项。上升通道 = mask 内 failure demotion（真实 0 return）+ success grading（γ^(T−t) 分档），sparse label 花在每 factor 少数 admissible candidates 上而非 5^45；headroom = restriction-only 52–58 → baseline 76+。

1. **平台不动**：official CQN-AS dueling C51 [−2,2]/51、K=16、nstep=1、batch 256+256、bc_lambda=0、bc_margin=0、Gaussian 0.01。pin：hinge ≥0.0125 = veto；nstep3 gap tax −3.25（仅作 mask 生效后的 v2 旋钮）；exploration 通道官方死亡。
2. **Offline 10k**：demo-only dense reward-Q（executed ← max(forced-SARSA Bellman, MC γ^(T−t))，unseen ← 0）+ 同期 CE 训练 mask head，10k 时 **freeze**。pin：无 offline = sealed 28.5（Stage 37）；无 dense 的 offline 全 0（Stage 30–36）；offline dense 独立 52/50；frozen weights don't collapse。早期 behavior shaping 由此相位承担（rfloor −26 的坑由 Phase 1 gate 在花 env 步前验掉），不引入 τ schedule。
3. **Frozen support mask**：per-level {b: π_b ≥ τ·max}，τ=0.3 固定，作用于 decode **和** target argmax。pin：restriction 贡献 0→52–58 的 100%、ranking 贡献 0；BCQ target-mask 切断 +64% fake up-slope；R-line must-do "fix the decoder"。
4. **Online positive-only dense，floor 只打 out-of-mask bins**（in-mask unexecuted siblings 留给 TD reorder）；failure samples 走 canonical chosen-bin、真实 0 return。pin：always-dense erode vs positive-only scale（Stage 38 vs 41）；178A 证明 demo-blind floor 是充分 suppressor（λ=0 下 72.5）；absolute-0 target 避开 A13 相对 floor 灾难；"real low returns, never clamped"。**这是唯一偏离已证组合的编辑**；fallback 预注册：decay fingerprint 出现即回退 Stage-42-faithful 全 floor。
5. **Frozen expert replay**（9,253 transitions，use_self_imitation=false）。pin：Stage 41→42 单变量消融（44/40 erosion vs 80/68）。
6. **critic_replay_max** 候选 {masked-greedy, recorded next chunk}。pin：worst case 退化为 in-sample SARSA；22–25 证明至少无害。

**reference 问题的答案**：在任意 online state s，anchor 是 frozen head π_b(s) 的一次 forward pass——与 critic 共用 (features, one_hot, midpoint) 的 C2F zoom conditioning，逐 level 输出 admissible bin 集合。它在所有状态上有定义：无 demo lookup、无 nearest-neighbor、无生成数据；泛化质量继承 BC（rollout 上 imitation top-1 0.991 是它在 rollout-reachable 状态上工作的直接测量）；freeze 把它放在 training dynamics 之外（collapse 的已证边界）；τ 是唯一的 pessimism dial。发射前零 GPU 校准：held-out demo states 上 demo-bin recall ≥0.99；已存 baseline rollout states 上统计 admission width。探针覆盖不了的 distribution shift 部分，由 Phase 1 的 continuation 实验兜底。

**两阶段预注册验证**

- **Phase 1（探针 + continuation，≤20k env）**：(a) mask recall/width 探针，gate recall ≥0.99；(b) offline-only masked-decode dev（100 eps），gate ≥45%；(c) ordering 探针：masked-Q vs random-within-mask 各 50 eps，gate Q ≥ random +5pp（否则 = rerank-null，停，零 env 花费）；(d) 178 协议第三臂：crown-truncate s1@101k continuation，λ=0 + mask + gated floor，20k 步。gate：+20k dev ≥60%；kill：dev <40%，或 span/Q <0.3，或 violation→1.0（collapse 5k 内可检出）。
- **Phase 2（from-scratch scaling）**：先 2 seeds 101k。**主指标 = online-curve slope**：固定 checkpoints {20k,40k,60k,80k,100k} 各 100-ep dev eval，对 success-vs-steps 做 least-squares slope；promote 条件：两 seed slope>0 且 100k−20k ≥ +10pp。kill-only 中点：30k dev < BC-at-same-budget −5pp 即杀；divergence instrument（masked argmax ≠ π_b top-1 比率）20k 时 <5% 且曲线平 = parity-shaped，关线。通过后 4-seed sealed（200 eps，seeds 800–999，固定 100k endpoint）：paired mean ≥ baseline 76.0 +3pp、≥3/4 non-negative，并 sealed 复核 rising claim（固定 20k vs 100k checkpoints ≥ +10pp）。

**Implementation checklist（真实代码）**：cqn_as.py `_greedy_action`（2804–2918）在 2856 注入 `jnp.where(mask_l, q, -inf)`，tie-break（2858–2863）与 `random_levels_from`（2892–2904）限制到 admissible bins，以 `_policy_value_action`（3150–3244）为模板改硬 mask；target 侧经 `_greedy_action_for_update`（3605）/`_td_target_action_for_update`（3612）免费继承，`critic_replay_max`（3633–3687）不动；mask head 复用 `policy_model` + demo-masked CE（2417–2435, 4854–4918），解开 1566 的 `bc_lambda>0` guard 与 1573/1843/1901–1938 的 dense/MC 互斥 guard，freeze = stop_gradient + 移出 optimizer；floor gating 在 cqn.py `unseen_return_floor_loss`（388）/`dense_return_distributional_loss`（461）把 unseen_mask（412, 572–596）与 ¬support_mask 相与；config：`truncate_demo_at_success=true`、`dense_return_positive_only=true`（cqn.py:1074）、`use_mc_lower_bound=true`、`use_self_imitation=false`、`bc_lambda=0`。合计约 100 行。

## 3. 最大的风险（诚实）

两条线的全部历史里，**没有任何一个实验测到过上升的曲线**。FS-CQN 的每根 evidence pin——0→52–58、178A 72.5、frozen-no-collapse、38 vs 41、41 vs 42——钉的全是"地板"和"不塌"，没有一根钉"会涨"。上升要求 in-mask reorder 信号仅靠 cross-trajectory generalization 产生（无 exploration、无 hinge、floor 不碰 in-mask siblings），而这件事没有活体先例：A18/A19 有 graded targets 没有 mask，Stage 43 有 mask 级稳定没有 gradation。最可能的失败不是 collapse，而是 mask 兜出一个 55–65% 的平台——divergence rate 在涨、success 不涨：一个 collapse-proof 的 IL，恰好不是 owner 要的东西。次级隐患同源：frozen mask 在 policy 驶入的新分布上 confident-wrong，stored-state 探针无法完全排除。Phase 2 的 slope 主指标与 divergence instrument 的意义就是让这个失败在 30k、而不是四个 sealed seed 之后被看见。

---
# 附:三个候选设计原文


## 候选设计 1

# Extraction: research_paper.md — algorithm-level design for no-BC, no-augmentation, critic-only discretized-action online improvement

Source: `/home/zc1525/robobase_jaxflat/research_paper.md` (72-paper survey, §§1–5; appendix S1 is MILES/IL-recovery material, not relevant to this question). Failure-mode numbering (F1–F5) is the survey's own: F1 = BC-margin critic encodes demo-membership not return ordering; F2 = removing BC collapses off-manifold top1−top2 margins (0.356→0.066) across ~240 factored bin decisions; F3 = C51 support saturation vs. its role as stabilizer; F4 = fragile offline→online seam; F5 = crude training regime (nstep=1, UTD=1, single critic).

## 1. BCQ and "discrete BCQ" (imitation-head threshold masking)

**Mechanism.** BCQ names extrapolation error: the bootstrap max evaluates actions absent from data, whose fabricated values propagate through TD. Continuous BCQ trains a conditional VAE to generate candidate actions near the data plus a perturbation net, and restricts both policy and bootstrap max to those candidates. The survey's transplant is the discrete form: train a small per-(timestep, dim, level) bin classifier π_b on demos + successful replay, and **mask the bootstrap argmax to bins with π_b ≥ τ·max**, with τ annealing toward 0 online.

**On the reference-action problem.** This is the survey's direct answer to "reference action at arbitrary states": the generative/classifier head gives *state-conditioned support generalization* — a masked candidate set exists at states never literally seen in replay, where instance-based candidates (the recipe's existing "candidate backup from replay argmax") do not. π_b is a proposal/masking prior only; argmax-Q still decides, so it stays inside the critic-only constraint. It composes with (does not replace) the pessimism floor. Weakness: the mask quality depends on the classifier generalizing off-manifold — exactly where it has no labels — so it degrades gracefully only via the τ anneal.

## 2. IQL / expectile in-sample targets (+ IDQL's critic half)

**Mechanism.** Train V(s) by expectile regression (τ≈0.7–0.9) against Q at *dataset actions only*; bootstrap Q with the SARSA-like r + γ·V(s′). No training target ever queries Q at a counterfactual or policy-chosen action — the max is replaced by an expectile statistic over seen actions.

**On the reference-action problem.** It **dissolves the problem inside training**: no loss term asserts anything about unexecuted sibling bins, so the critic cannot invent fake gaps (F1) and off-manifold bins are simply never dragged up by bootstrapping (a different fix for F2 than pushing them down). The survey's factored transplant: per head, expectile-V on the executed bin's Q, bootstrap all heads with r + γ·V(h′). **But the problem reappears at act time**: argmax still runs over all bins, whose values are purely generalization-determined — so the survey flags that IQL-style targets *require* candidate restriction at decision time (QC/IDQL-style) to be complete. Expectile τ doubles as a tunable pessimism knob replacing the positive/negative-return asymmetry in the F4 recipe. C51 caveat: expectile regression on a categorical head is nontrivial; start with a scalar V head.

## 3. IDQL / AQL / QMLE / EMaQ — candidate-proposal decoding

Convergent thesis (§2, backfill synthesis): CQN-AS's fragility is not big-action-space Q-learning per se but **how the max is computed**. Composing ~240 independent per-factor argmaxes makes the decision margin a product of fragile per-factor margins; all four methods instead evaluate a small set of *complete joint candidates* and take one argmax, so the margin is a single joint Q-gap.

- **EMaQ**: bootstrap target = E[max_{i=1..N} Q(s′, a_i)], a_i sampled from an autoregressive behavior model. N is the single conservatism knob (N=1 pure behavior value → N→∞ support-restricted Q-learning). Coarse-to-fine bin selection *is* an autoregressive generative process, so a lightweight bin prior slots in near-isomorphically; N anneals pessimism online without a hard floor.
- **AQL**: sample N≈100 joint actions from an autoregressive proposal trained by *supervised amortization* (NLL of best-found action + entropy bonus — no policy gradient), plus cheap uniform actions; argmax-Q over the union for both acting and the TD target. Ablation: discretized proposals beat continuous Gaussians (validates the bin structure). Free candidates: replay action and demo action.
- **QMLE**: adds (a) **replay-cached best-argmax** — store per-transition the best candidate found so far, update whenever any candidate scores higher, use as guaranteed TD-target and decision candidate (makes the approximate max monotone; formalizes the recipe's ad-hoc replay-argmax backup); (b) proposal *ensembles* + uniform mixing against proposal collapse; (c) the action-in argument — add a joint chunk-scoring head Q(s, full chunk), since action-out factored heads can't score joint candidates coherently.
- **IDQL**: the clean role decomposition — a pure-imitation diffusion/BC model says *what is plausible*; an in-sample-trained, return-only critic says *which is best* (best-of-N, hard argmax at eval). The survey reads F1 as a feature here: the BC-margin critic *is* a demo-manifold membership model — exactly the proposal role. Keep it as proposal; let the no-BC return critic judge.

**On the reference-action problem.** The proposal *is* the reference-action generator at arbitrary states, and it is trained by supervised likelihood, never touching the critic loss. Standing caveat (stated in both AQL and QMLE entries): **candidate ranking only helps if Q carries return ordering** — this whole family must ride on the no-BC return-trained critic; on a BC-margin critic it just re-selects demo-frequent chunks. Repo prior art (Route R4): flow-proposal + Q-rerank already validated mechanically (61.3% vs 41.3% flow-alone), but the judge then was the F1-poisoned critic; new plan uses the categorical policy's own top-k beam as the proposal set (`top2_joint_beam`, cqn_as.py:79), bypassing the weak flow sampler.

## 4. Q-Transformer — conservative floor details

**Mechanism.** Per-dim autoregressive Q (256 bins/dim, each dim its own "timestep", within-timestep discount 1.0); sigmoid-normalized Q in [0,1] with terminal 0/1 reward; conservative regularizer **α/2·E_unseen[(Q−0)²]** — every bin not taken in the dataset is regressed toward the minimal attainable return (0), α=1.0 untuned, **always on, never annealed, applied to all data including 20k failed episodes** (which improved policy 70% over IL). TD target additionally floored by MC return-to-go: max(MC, Bellman). They explicitly reject softmax CQL — its ablation collapses success to ~the demo fraction of data (~8%), i.e. the critic degenerates into a behavior-density model (**published reproduction of F1**). Dimension-skipping n-step is safe under the floor and 4× cheaper.

**On the reference-action problem.** Answers it *negatively and everywhere*: at any state, every unseen bin is pinned to worst-case return, so dataset actions win by exactly their advantage over "do nothing useful" — Q keeps return semantics, not density semantics. This is published validation of the lab's sealed F4 recipe (floor ≈ their regularizer; max(MC,Bellman) ≈ their MC floor). **One direct conflict flagged**: Q-Transformer says always-on floor on all data; local evidence says positive-return-only flooring is a required stabilizer. Survey hypothesizes the difference lies in their cross-dim autoregression (no independent-factor fragility) or sigmoid head, and prescribes a cheap decisive 2×2 (autoregression × floor regime).

## 5. Cal-QL — calibration at the offline→online seam

**Mechanism.** One-line CQL fix: replace the pushdown E_π[Q] with E_π[max(Q, V^μ(s))] — conservative pushdown is **masked whenever Q has already fallen below what the behavior policy actually achieves**, with V^μ = dataset MC return-to-go (no extra network on sparse terminal tasks). Result is provably a lower bound on the learned policy's value *and* an upper bound on the behavior policy's ("calibrated"). Their measured failure of naive conservatism — over-pessimistic offline Q makes new online actions spuriously outrank the initialization, causing an unlearning dip — is F2's mechanism in factored-argmax form.

**On the reference-action problem.** The reference is not an action but a **per-state value scale**: at any state with a behavior MC estimate, no sibling bin may be pushed below V^behavior(s). Transplant: floor(s) = max(task-minimum, V^behavior(s)) projected onto atoms — one line, reusing the already-computed MC return-to-go; keeps sibling gaps commensurate with true return gaps (0.009, not 0.89). Limitation: silent at states with no behavior estimate — it calibrates the seam, it doesn't generate references off-manifold. Pairs with WSRL warmup rollouts (seed the online buffer with the frozen policy's own rollouts before any gradient) as the tested seam recipe.

## 6. SDQN — joint-consistency head

**Mechanism.** The origin of per-dim autoregressive decomposition: an inner zero-discount "lower MDP" selects one discretized dim at a time, plus a separate **upper Q^U(s, full action)** trained by ordinary Bellman TD, with an MSE consistency loss tying the factored per-dim Q values of executed bins to the joint head.

**On the reference-action problem.** Instead of supplying reference actions, it supplies a **shared return-grounded anchor**: the joint head only ever sees real return targets, and consistency propagates that calibration into all ~240 factored decisions — attacking F1 (joint head can't learn density) and F2 (shared anchor resists per-factor drift) at any state the critic evaluates. Bonus: its ordering ablation shows dimension order is empirically benign — don't spend experiments on it.

## 7. Advantage-learning gap operators — and the project's prior-art verdict

**Mechanism.** AL: subtract α·(V−Q) from non-greedy executed actions' targets (optimality-preserving, gaps widen, 37/60 Atari wins). Munchausen: add α·τ·log π(a|s) (π = softmax of own Q) + soft backup — implicit KL between successive policies, provable (1+α)/(1−α) gap amplification. Clipped AL: apply the penalty only to near-top actions, because gap-increasing is *unsafe when the ranking is untrustworthy* — and F1 says this critic's ranking is coin-flip.

**On the reference-action problem.** These need no reference action at all: they amplify the critic's **own** ordering at every visited state, including off-manifold ones BC can't reach — the cleanest conceptual replacement for the margin's F2 function without F1 poisoning. **Project verdict (§5 header note, decisive):** the pre-execution prior-art review (cqn-no-bc.md Stage 20/21, cqn-flow.md Stage 163/164/171/177) **falsified/exhausted the R1a gap-operator forms** (and R2a's atomic K-step form); the actually-launched wave-1 arms were floor+λ0.0125 and nstep3, adjudicated in `cqn-rline.md`. The R1a card survives as design archive only. Architectural cousin still standing: the Mean-Expansion Layer (factor out the sibling baseline so the learned residual *is* the action gap; C51 adaptation untested).

## 8. Trust-region / proximal constraints for discrete factored actions

No dedicated section; the survey's implicit-trust-region inventory: **QC's Best-of-N** over a behavior prior enforces KL(π‖behavior) ≤ log N − (N−1)/N with no policy gradient — a *tunable, annealable* trust region replacing the BC margin's hard anchor (N is the dial); **EMaQ's N** is the same knob in the bootstrap; **BCQ's τ** is a proximal support mask; **Munchausen** is an implicit KL to the critic's own previous policy (temporal, not behavioral, proximity); **Peng's Q(λ)** converges optimally only if behavior slowly tracks greedy (conservative-policy-iteration flavor); **AWAC** shows implicit KL beats explicit behavior-density models, which go stale as the online mixture shifts. Common thread: for factored discrete actions, express the trust region as a *candidate set restriction at the joint-chunk level*, never as a per-factor penalty.

## 9. The survey's own final recommendations

Priority order (§3 table): **R2** prefix/chunk multi-horizon targets (SEAR beat CQN-AS head-to-head >20% IQM; the paid-for sequence critic is unused at nstep=1) → **R1** gap operators (since falsified in this project, see §7) → **R5** craft package (critic LayerNorm — RLPD's ablation is a published F2 reproduction — small ensemble subset-min, UTD>1, seam resets, WSRL warmup) → **R4** proposal + Q-gated execution (§3's F1+F2 fix; must ride the no-BC critic) → **R6** two-head decouple (clipped decision head for argmax vs calibrated head for judging; stop-gradient boundary mandatory, Stage 147/148 behavior-tax evidence) → **R3** margin surgery (L0-only / Nair-style per-factor Q-filter gate with hard-zero on correctly-ranked factors — diagnostic value) → **R7** RFCL reverse+forward reset curriculum (zero imitation gradient, demo influence via start-state distribution only — makes F1 structurally impossible; costliest engineering). **Endgame conjecture**: no-BC dense recipe (value-space pessimism + MC anchor) × R2 multi-horizon targets × R5 craft package × R4 decision gating — each critic-only, mutually orthogonal, each with published evidence. Negative list to respect: label smoothing in the distributional bootstrap (−31pp), relative floor (−7.3), de-saturation online, BC annealing (margin collapse), Retrace on demo transitions (cuts traces exactly where needed).

## 候选设计 2

# INVENTORY: CQN-AS machinery reusable for a support-constrained / proposal-based redesign

All line numbers: `robobase/method/cqn_as.py` unless prefixed `cqn.py:`.

## 1. BC/imitation heads that predict actions separately from the critic

- **`separate_bc_policy` head** — `self.policy_model = C2FSequenceDistributionalCritic(atoms=1, use_dueling=False)` (2417–2435). Output: **per-level, per-(seq-step, action-dim) bin logits** `[B, levels, K*D, bins]` via `_policy_logits_per_level` (2697–2739; also returns encoded expert bins from `encode_action`). Greedy decode: `_policy_action` (2741–2797), autoregressive over C2F levels with `zoom_in`. Trained with demo-masked CE (or AWR-weighted CE via `ExpectileValueHead`, 657–686 / 4877–4917) inside the decoupled update path (4854–4918). Constraints: requires `bc_lambda>0` (1566), incompatible with `mc_lower_bound_target`, `dense_return_q_target`, twin critic (1843, 1573, guard block 1901–1938). `distinct_policy_encoder` duplicates encoder leaves (2528–2536).
- **FOSD is NOT a separate head** — `demo_fosd` is a critic-side first-order-stochastic-dominance loss on demo transitions plus `bc_margin` expected-Q hinge, both on the critic's own logits (cqn.py:1526–1554).
- **`FlowPolicyHead`** (689–837): continuous velocity field over full chunks. Two modes: `flow_policy` (decoupled sampler `_flow_policy_sample` 3246 + critic rerank `_flow_rerank_action` 3278 — an existing **proposal-based decode**: M candidates, deepest-level Q rerank) and `coarse_flow` (bin-conditioned within-cell residual, `_coarse_flow_cell` 3311, `_coarse_flow_action` 3374, EMA params 5670–5680).

## 2. Decode paths

- **Greedy per-level argmax**: `_greedy_action` (2804–2918). Per level: expected C51 Q `jnp.sum(softmax(logits)*support, -1)` (2851–2855), `index = jnp.argmax(q_values, axis=-1)` (**2856**), random tie-break under `tie_break_delta` (2857–2872), then `zoom_in` (2907). Level-override diagnostics: `random_levels_from` + `level_override_mode` ("middle" = center bin, else uniform random) at 2882–2904.
- **Autoregressive dims**: `AutoregressiveActionCorrection.greedy_bins` (1127–1186), causal per-dimension argmax at 1164.
- **Twin pessimistic**: `_pessimistic_greedy_action` (2920–2994), argmax of `pessimistic_categorical_q` (56) at 2967.
- **Beam**: `_joint_beam_action` (3015–3148) with `top2_joint_beam` (79–154), width `twin_rollout_beam_width`; final rerank by `_score_action_sequence_for_backup` (3694–3712, deepest-level mean expected Q — the universal chunk scorer).
- **`_policy_value_action`** (3150–3244): per-level score = variance-normalized centered Q + `beta * log_softmax(policy logits)` (3208–3216) — **an existing soft support-constrained decode** (BC log-prior as soft mask).
- **Dispatcher**: `_build_greedy_action_fn` (3427–3603) selects among all of the above; jitted as `_greedy_action_impl` (2553–2558), called in `act()` at 5681–5688 with args `(params, target_critic_params, obs_inputs, use_target, key, twin_head_indices)`.
- **Post-ensemble overrides** (numpy, after `_ensemble_current_action` 5189–5238): `_post_ensemble_bin_flip` (5295, train exploration at L1/L2 granularity), `_post_ensemble_randomize` (5364, eval diagnostic: keep true C2F prefix `post_ensemble_random_keep_levels`, randomize/fix leaf), consensus diagnostic `_accumulate_ensemble_consensus` (5240). Pre-ensemble plan exposed as `_last_plan_chunk` (5693) for external Q-selection scripts.

## 3. Target options

- **Hook chain (canonical no-policy path)**: parent `CQN._build_update_fn` `loss_fn` calls `self._td_target_action_for_update` (cqn.py:1228); CQNAS override at **3612–3692** implements:
  - `replay_next`: `shift_replay_action_sequence` (40–53) — SARSA on executed chunk.
  - `critic_replay_max` (**3633–3687**): candidate-set backup — `{greedy chunk, replay next-action chunk}` scored by `_score_action_sequence_for_backup`, `behavior_selected = behavior_score >= greedy_score` (3659), optional `demo_behavior_force_probability` forcing (3660–3671). **This is the existing template for any proposal-based target max**: extending the candidate set is a local edit here.
  - default `critic`: `_greedy_action_for_update` (3605) → `_greedy_action`.
- **Decoupled path** (`separate_bc_policy`, 4504–4534): adds `bc_policy` (`_policy_action` on next features) and `policy_value` (`_policy_value_action` with `td_target_policy_value_beta`) sources.
- **Twin path**: `horizon_target` inside `_build_pessimistic_twin_update_fn` (4013–4148): `_pessimistic_greedy_action` (4022) + behavior candidate + clipped-twin target selection (4104–4109) + elementwise MC lower bound (4121–4133).
- **Parent CQN loss machinery** (cqn.py): `project_categorical` (220), MC lower bound `use_mc_lower_bound` (1334–1381), `episodic_success_returns` (757/1321), `sequence_aligned_sparse_returns` (363/1345), **dense-return targets** `dense_return_distributional_loss` (461, all-bins floor + finest-neighbor kernel + label smoothing + floor-satisfaction margin + relative floor margin) and `dense_return_expected_q_loss` (710), **advantage shift** `advantage_learning_target_shift` (338) + `shift_categorical_distribution` (312), `return_gated_margin_loss` (677), **`unseen_return_floor_loss`** (388, mean/max/topk over unseen bins — a soft conservatism penalty that is the loss-side sibling of a hard support mask), canonical MC anchor (1616–1635), token-split horizon targets (1260–1316).

## 4. Exploration

- **`bin_explore`** (Stage-153/162): spec `bin_explore_probs/schedule/persist_plans` (235–237, validation 2134–2188); state arrays `_bin_explore_{remaining,dimension,level,sibling}` + RNG (2152–2164); `_apply_bin_explore` (**5524–5629**): per fresh plan (`register_mask` = replan mask), per level coarse-to-fine first-fire, sibling-cell shift with inherit-refine, persists `bin_explore_persist_plans` plans; schedule scale via `utils.schedule` (5706–5709); executed-step `_last_bin_explored` flags for explore-aware n-step truncation (5722–5749); episode reset clears windows (6193–6201); RNG streams checkpointed (5908–5927).
- **`bin_flip`** `_apply_bin_flip` (5475): open-loop only (`temporal_ensemble=false`), whole-plan sibling shift; relabels structured-exploration extras (5831–5864).
- **Structured exploration**: `_structured_exploration_action` (3718) / `_coherent_structured_exploration_action` (3761): one coordinate ± one C2F cell width, horizon-persistent; per-env assignment state feeds the CV-RCT causal loss (4657–4851, `action_centered_moment_loss` 619).
- **Episodic twin-head** (`select_episodic_twin_actions` 68, `_resample_episodic_twin_heads` 6158, act 5653–5669).
- **Base noise**: uniform for `num_explore_steps`, then Gaussian `stddev_schedule` (5774–5795).

## 5. Where a per-level bin MASK inserts

**(a) Greedy decode** — single choke point: line **2856** in `_greedy_action` (`q_values = jnp.where(mask_l, q_values, -inf)` before argmax). The level loop is a static Python loop, so a per-level mask list (each `[B, K*D, bins]` or `[K*D, bins]`) plumbs cleanly. Caveats: (i) the tie-break `random_index` (2858–2863) and `random_levels_from` overrides (2892–2904) must also be restricted to allowed bins; (ii) the mask is zoom-path-dependent — if it comes from a network (e.g., `policy_model` top-k / thresholded logits), compute it inside the loop with the same `(features, one_hot, midpoint)` signature `policy_model` already uses (`_policy_value_action` is the exact structural template — replace its soft log-prior with a hard `-inf` mask); if it is a static table, add one traced argument through `_build_greedy_action_fn`'s `action_fn` (3428) and the `_greedy_action_impl` call (5681) — mechanical. Parallel edits needed only if those platforms are in play: `_pessimistic_greedy_action` (2967), `_policy_value_action` (3217), autoregressive `greedy_bins` (1164); the beam is the awkward one — `top2_joint_beam` hard-requires ≥2 bins/factor (100) and its regret math assumes the top-2 are admissible.

**(b) Target max** — masking `_greedy_action` is **inherited for free** by the canonical no-BC platform, because the Bellman argmax is the same function: `_td_target_action_for_update` (3612) → `_greedy_action_for_update` (3605) → `_greedy_action`, called from cqn.py:1228. The twin path needs the same edit inside `horizon_target` at 4022, and the decoupled path at 4530. Alternatively, skip masking the argmax entirely and use the **`critic_replay_max` candidate-set pattern** (3633–3687): a support-constrained backup = max over {replay next action, N proposal-head samples}, scored by `_score_action_sequence_for_backup` — no `-inf` surgery, one localized edit. A third insertion: gate the all-bins terms (`dense_return_*`, `unseen_return_floor_loss`) with the support mask via their `chosen_mask`/`unseen_mask` (cqn.py:412, 572–596) so out-of-support bins are floored while in-support unseen bins are left alone.

**Size verdict**: small (≈20–40 lines) for the canonical platform (mask in `_greedy_action` + tie-break guard, target constraint inherited); small for a proposal-based backup via the `critic_replay_max` template; moderate only if the twin/beam/autoregressive decode variants and the dense-return loss family must all honor the mask.

## 候选设计 3

# Design: SC-CQN — Support-Constrained CQN-AS (discrete-BCQ over the coarse-to-fine factored head)

## 0. One-paragraph summary

Keep the CQN-AS critic (C2F factored bins, C51, chunks, critic-only) but make it **no-BC-on-the-critic**: `bc_lambda=0, bc_margin=0`. Add a separately trained **behavior head** (the existing `separate_bc_policy` C2F model, atoms=1) trained by pure CE on demo + successful-episode actions. At every decode — greedy act AND Bellman target max — per level, per (step, dim), **mask Q to bins with π_b ≥ τ_l·max π_b** before argmax. Out-of-mask bins additionally get the 178A return floor. Improvement channel: **in-mask ε-bin exploration** (existing `bin_explore` machinery, sibling sampled from in-mask siblings) + explored-truncation n-step, feeding pure TD with MC lower bound on real sparse returns. τ starts tight (decode ≈ BC) and anneals to a permanent floor τ_min>0.

## 1. Components and their justifying facts

**(a) Critic loss = canonical per-sample C51 TD + `use_mc_lower_bound` (max(MC, Bellman)), λ=0.** Truncated regime returns γ^(T−t) sit on interior atoms — the saturation crutch of Stage-42/43 is gone (A18/A19: unclipped calibrated-small-margin geometry decays online), so we do not rely on it; the mask is the explicit replacement margin source. MC floor is Q-Transformer-validated and already load-bearing in the survivor recipe.

**(b) Behavior head as a MODEL, not a loss.** IDQL's decomposition, and the survey's reading of F1 as a feature: the BC-shaped object is a demo-manifold membership model — exactly the proposal role. Code exists: `C2FSequenceDistributionalCritic(atoms=1)` + demo-masked CE (cqn_as.py:2417–2435, 4854–4918). Trained on the **frozen expert replay + success-gated online additions** (Stage-41 vs 42: unfiltered growing replay is a seed-specific feedback loop; success-gating + a rehearsal cap bounds drift). Critic never sees a BC gradient.

**(c) Hard per-level relative-threshold mask** (discrete BCQ). Probability threshold, not radius: C2F cells are already spatially local; the threshold is the state-conditioned quantity. Mask both decode and target max — BCQ's core point is that extrapolation error enters through the bootstrap max; masking the target makes the critic **never trained against out-of-support fabricated values** (EMaQ at N→∞ = support-restricted Q-learning). Per-level: coarse levels tight (unified law + §58C: coarse decisions price behavior and carry all exploration information), finest level loose (Gaussian σ=0.01 = level-2 jitter = exactly zero information, so restricting it buys nothing).

**(d) 178A floor on out-of-mask bins only** (`unseen_return_floor_loss`, weight 0.1, target 0, mean). 178A proved a demo-blind floor is a sufficient sibling-supervisor at λ=0 (72.5 vs 0.0). Gating it to out-of-mask bins keeps in-mask unexecuted siblings **free** for TD to reorder — avoiding rfloor's mistake of flooring everything and A13's relative-floor disaster.

**(e) In-mask ε-bin exploration** (`bin_explore` with sibling drawn from in-mask siblings, weighted by π_b) + `nstep_explore_truncate`. The dividend-bearing datum is the first-action same-state different-bin contrast (§49); truncation machinery already surgically keeps it. In-mask restriction keeps online cost low (candidates are all plausibly-good; cf. dishwasher: heavy exploration online yet sealed 100%). Exploration stays on all run (ε-anneal −8).

**(f) τ schedule: τ≈0.85 for ~15k steps (mask≈top-1 ⇒ decode ≡ BC policy), anneal to τ_min≈0.2, never 0.** At τ→1 the algorithm *is* BC — this supplies λ's "early behavior shaping" job (rfloor's −26 proved a floor alone can't). The anneal is EMaQ's N / BCQ's τ: a tunable trust region expressed as **candidate-set restriction, never a per-factor penalty** (survey §8 common thread). τ_min>0 forever: the suppressor is permanent.

**(g) Offline phase**: 10k offline (critic TD-on-demos with `replay_next` SARSA backup + MC floor; behavior-head CE), then online. Offline-first is load-bearing (Stage 37 sealed 28.5 without it). No dense all-bin target — its job (suppress unseen bins) is taken by mask+floor, and always-dense eroded online (Stage 38).

## 2. The reference/anchor problem, solved concretely

At an arbitrary online state s (never in any demo), the reference is **π_b(s): a state-conditioned distribution over bins at every level, produced by a network forward pass** — no demo lookup, no nearest-neighbor, no generated data. It generalizes exactly as well as BC generalizes, and BC demonstrably generalizes to rollout-reachable states (official BC 64.6%, baseline imitation top-1 0.991 along 200-ep rollouts). The mask {b: π_b ≥ τ·max} is therefore defined *everywhere*, including off-manifold — where it degrades gracefully: the head's entropy rises off-manifold, the relative threshold then admits more bins, and the out-of-mask floor plus MC-anchored TD decide among them. This is the survey's stated answer (§1): support **generalization** where instance-based candidate backup (`critic_replay_max`) has no candidate. Resurrection probe v1 is the measured warrant that this restriction is the operative ingredient: restriction to data-supported candidates contributed 100% of the 0→52–58 recovery; ranking contributed 0.

## 3. Why no collapse at λ=0, and why improvement is expressible

**No collapse.** The corrected phase law (R-line §2.7): the phase variable is "does any unseen-action suppressor exist", not λ. Collapse mechanism is sea-level rise + peak melting → argmax drifts on a sea of ties (69% of factored decisions leave the demo bin) → off-manifold → +64% fake up-slope pulls it away. Under the mask: (i) argmax **cannot leave the in-mask set**, at act time and in the bootstrap, so tie-drift is confined to 2–4 in-support bins per factor; (ii) the worst case of total in-mask tie-melt is "random over data-supported candidates" = **52–58% measured on a fully collapsed critic** — the collapse floor of this design is the resurrection number, not 0; (iii) out-of-mask bins keep the 178A floor, which alone held 72.5 at λ≡0; (iv) the fake up-slope never enters targets because the target max is masked (BCQ). Two independent suppressors, both demo-blind, both permanent.

**Improvement expressible.** The gap-tax law existed because margin was the safety mechanism: buying ranking (Spearman 0.87) thinned the execution-token gap and cost behavior (nstep3 −3.25, rfloor −26). Here **safety is decoupled from margin**: thin in-mask gaps are harmless because every admissible choice is in-support. So TD can reorder in-mask bins freely, and — critically — there is no BC hinge veto (established fact: λ≥0.0125 vetoes behavioral improvement; mechanism-behavior decoupling, 3 reproductions). Headroom is real: restriction-only on a collapsed critic = 52–58 while healthy runs reach 66–85 — within-support reordering spans ≥20pp. The channel from better Q to better behavior is one argmax, immediately.

## 4. Why the curve should RISE over ~300 episodes

Per-episode information under replan-1/K=16: ~34 chunk decisions/episode; explored density 0.18–0.34 gives ~6–11 first-action in-mask contrasts per episode, ~2–3k over the run — same order as the ~10 real coarse events the old schedule carried, but now (i) each contrast is between *plausible* actions (informative comparisons, not junk devaluation), and (ii) consuming them changes behavior because nothing vetoes the argmax. ~100 successful episodes deposit graded interior-atom returns γ^(T−t) at executed in-mask bins (real value ordering, first time it's both learnable and actionable); ~200 failures deposit **genuine zero returns at in-support executed bins** — Q-Transformer's measured result that failed episodes improve policy 70% over IL, and it is real online data, not augmentation. The τ anneal converts learning into behavior monotonically: early = BC (no dip), later = critic-adjudicated in-support choice. The prior exploration-dividend-zero verdicts (combined −1.5/−2.1) were all measured with the veto on; the coverage law (§64.1: sibling span grew 3.4× as λ fell — coverage takes over the hinge) says executed-sibling TD supervision self-enforces the constraint, which is exactly what in-mask exploration manufactures.

## 5. Implementation delta

- `_greedy_action` (cqn_as.py:2804–2918): inside the level loop, compute behavior logits with the same `(features, one_hot, midpoint)` signature (`_policy_value_action` 3150–3244 is the structural template — replace soft log-prior with hard `-inf` where π < τ_l·max); mask at 2856 before argmax; restrict tie-break `random_index` (2858–2863) and `random_levels_from` overrides (2892–2904) to in-mask bins.
- Target max: **inherited free** via `_td_target_action_for_update` (3612) → `_greedy_action_for_update` (3605).
- Floor gating: pass the mask as `unseen_mask` restriction into `unseen_return_floor_loss` (cqn.py:388, mask plumbed via chosen/unseen masks at 412, 572–596).
- Behavior head: reuse `policy_model` + demo-masked CE (2417–2435, 4854–4918); lift the `bc_lambda>0` guard (1566); extend CE batch to success-gated online transitions.
- Exploration: `_apply_bin_explore` (5524–5629): sample sibling from in-mask siblings (export per-level in-mask sets alongside `_last_plan_chunk`, 5693). Keep `nstep_explore_truncate` (§60) unchanged.
- τ via `utils.schedule`. ~150–250 lines total; no new networks, no new replay machinery.

## 6. Predicted failure modes + cheapest preregistered tests

1. **In-mask tie-melt** (collapse-lite to ~52–58). Cheapest decisive test, before any from-scratch run: **continuation from crown-truncate s1 @101k, λ=0, mask on, 20k steps** — the exact 178A/178B protocol, third arm. Preregistered: dev ≥60% at +20k, span-within-mask and agreement tracked; 178B collapsed within 5k, so 20k detects it. Kill line: dev <40% or span/Q <0.3.
2. **Behavior-head miscalibration** (mask drops the demo bin at drifted states). Zero-GPU probe first: mask recall of demo action on held-out demo states (require ≥0.99 at τ=0.85) and mask width/entropy on stored online-rollout states.
3. **Flat curve — dividend zero again** (combined-flipcup cousin). Discriminating instrument, logged from step 0: **divergence rate** = fraction of decisions where masked argmax ≠ behavior-head top-1, and its success-conditioned delta. Prediction: rises from ~1% toward 10–25% and correlates with success. If divergence rises but success doesn't, the channel is dead → close (30k midpoint kill only, per statistics discipline).
4. **Early from-scratch failure** (rfloor cousin). Gate: 10k–20k dev must be ≥ BC-at-same-budget − 5pp; if not, τ early schedule too loose.
5. **Success-gated head drift** (Stage-41 cousin). v1 ablation guard: run one seed with demo-only frozen head; if it matches, freeze wins by parsimony.
Declaration only via four-seed same-seed paired sealed evals, mean ≥+3pp, ≥3/4 non-negative.

## 7. Closest already-failed cousins and the discriminating difference

- **zero70 / weansw / 178B (λ→0 collapse)**: they removed the suppressor; SC-CQN never has a suppressor-free step (mask + floor permanent, τ_min>0), and its worst case is measured at 52–58, not 0.
- **rfloor (floor+λ0.0125 from scratch, −26)**: early shaping came from nothing; here τ≈0.85 makes the early decode *identical to BC* without any critic-side hinge.
- **Rerank/judge line (full-matrix null)**: that was decision-time harvesting on an unconstrained proposal set with an F1-poisoned judge, changing nothing about training. The mask changes the **TD target and the data distribution** — a training-loop intervention; and the "nothing left to harvest" law was proved for a BC-margin critic, which this critic is not.
- **`_policy_value_action` soft prior (exists in code)**: soft per-factor penalty — exactly the form §8 warns against; the hard set-restriction is the trust-region form with published support (BCQ/EMaQ/QC).
- **Exploration-dead verdicts (combined −2.1/−1.5, coarse-only −36)**: all measured with the BC veto on and unconstrained sampling; discriminator is the divergence-rate instrument in §6.3 — if behavior can't move, we see it in weeks-cheaper telemetry, not sealed seeds.
- **Stage 30–36 offline-all-zero (calibration ≠ extraction)**: their failure was extraction from a flat field; the mask *is* the extraction fix (resurrection 0→52–58), applied continuously rather than post-mortem.
- **nstep3/b2 gap-tax arms**: they bought ranking by spending the margin that was doing safety duty; here margin has no safety duty. (Corollary prediction: nstep3 becomes net-positive inside the mask — v2 knob, not v1.)

No arm in either line ever ran a learned support mask on decode+target: the family is genuinely unexplored here, its two load-bearing ingredients (restriction 0→52–58; non-hinge suppressor survival 72.5) are both locally measured, and its outer form is published (BCQ, EMaQ, IDQL).