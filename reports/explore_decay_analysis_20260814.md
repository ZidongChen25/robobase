# 为什么"探索 + BC 退火"还是不行 — 证据链综合

**标签约定**(全文强制):
- `[实测:…]` — 本线 log 里有 sealed 数字的测量
- `[引用:…]` — 跨线/跨文档引用的数字(178A/178B/wean0 等),本线未审计其 lifecycle
- `[声称:…]` — 出现在 summary 里但 forensic audit 判定 **log 中无记录或与 log 冲突** 的数字,按未验证处理
- `[文献:…]` — 发表文献
- `[假设:…]` — 未测量的推理环节,附上能关闭它的测量

---

## 1. 因果链

### (a) 为什么 exploration + anneal-to-0.0125 打不过 baseline

**链条:探索的信息带宽窄 → 同类证据 offline 免费可得 → 但只要 hinge 在,任何数据质量都不换算成行为 → 换算成 value 的部分反而抽 decision margin 的税 → 全部效应落在 seed noise 带内。**

1. **探索的 outcome 信息通道只在 coarse level。** eval-time dose-response(baseline s1 @100k):Gaussian σ=0.01 → 68%=68%(零效应);as-trained bin probs [.002,.004,.008](17.2% explored steps)→ 58%;4x rate(46.7% steps,2.7x events)→ 仍 58%;coarse-only [.03,0,0] → 32% `[实测:上述全部数字]`。fine-level 事件物理上不改变 outcome,加 2.7 倍事件不加伤害——说明只有 coarse 事件携带 outcome 信息。**但**"eval-time 伤害 = training-time 信息量"是恒等式假设,log 自己下修过一次("基本无信息"说法)`[假设:训练期按 level 记录 explored-event 数 vs 其 replay 中 realized TD-error / outcome diversity]`。

2. **coarse 事件 online 很贵,offline 免费。** coarse-direct 探索 eval 代价 −36pp(68→32)`[实测]`;而同类证据(off-manifold failure + true return)可由 fence injection 零 online 成本注入,且确实产生 true devaluation:held-out branches −19%~−23%,对照 baseline 的 +64% fake upslope `[实测]`。所以探索作为"信息采集器"是被 fence 支配的方案。

3. **但在 λ ≥ 0.0125,数据质量不换算成分数——行为被 hinge 钉死。** 三次独立复现:cfaug-v1 四 seed +1.25(2/4 正,gate failed);fence-v2(true near-manifold,median manifold distance 0.000)n=4 平均 −0.125;recovery-teach n=2 −3.5 `[实测:全部]`。rerank 矩阵 {frequency, ranking, counterfactual judge} × 3 tasks 全 null `[实测]` ——已有的 value 知识在 argmax 里已被完全表达,decision time 无剩余可收割。("corr=0.10"这个数字 log 里没有,decoupling 本身有三处定性记录 `[声称:corr=0.10]`。)文献口径一致:小权重 anchor 是 projection 不是 teacher,plateau 上表现不随 λ 缩放 `[文献:TD3+BC arXiv:2106.06860;ReBRAC arXiv:2305.09836]`。

4. **真正进到 target 的 value 改动抽 decision margin 的税,税率按任务 precision-sensitivity 定价。** Spearman 0.53–0.66 → 0.87–0.90(nstep3/rfloor)同时 top2 gap 0.37 → 0.15–0.21,行为随 gap 单调 `[实测]`;税率排序对齐 L2-randomization sensitivity audit(sandwich −28 / flip_cup −13.75 / move_plate −2):b2-sandwich −18/−20 vs b2-move_plate ≈ 平 `[实测]`。**诚实备注**:summary 说 value-into-target"全部有害"是过头的——b2-move_plate sealed +2.0(−1.5/+5.5)是第一个双指标同向的 arm,nstep3 s2 sealed +3.0(Spearman 0.878)是"ranking 与行为可共存"的本线存在证明 `[实测]`;MC lower bound 的 −18.75/−15.25 不在本线 log `[声称]`。理论对应:greedy policy 对 value error 的鲁棒性由 action gap 决定,gap 变薄则同样的 error 翻更多 argmax `[文献:Farahmand 2011 action-gap]`。

5. **组合配方的直接测量。** exploration × decay:move_plate 四 seed −2.1,flip_cup −1.0/−2.0(均值 −1.5)——log 原话"探索通道正式出局" `[实测]`。anneal-to-0.0125(带 v1 fence):n=2 均值 +2.0,seed spread −8.0/+12.0,s1 的 69.5 低于全部四个 baseline seed(71.0–82.5)`[实测]`。baseline 自身 seed spread 11.5pp(move_plate n=4)/ 13.5pp(flip_cup)`[实测]`。**按本线自己三次处决 n=2 结论的四 seed 纪律,"0.0125 安全且单 seed 线最优"是同款 over-read;"安全"成立,"更好"不成立。**

6. **合账。** 探索提供的是 fence 免费提供的同类信息的低带宽高成本版本;hinge 在位时两者都不换算成行为;换算进 value 的机制在薄 gap 任务上倒扣。文献侧的封底:buffer 里的 passive 数据修不了 untaken actions 上的 extrapolation `[文献:tandem effect, Ostrovski arXiv:2110.14020]`。所以"探索 + 退火到 0.0125"在结构上没有一个环节能产生超出 seed noise 的增益——这不是执行失败,是配方两个旋钮都拧在错误的轴上。`[假设:探索是否贡献 fence 没有的 state coverage——diff combined run 与 cfaug run 的 replay state-visitation 可关闭]`

### (b) 为什么 λ=0 必崩(且探索救不了)

1. **相变本体。** zero70(move_plate):65k(λ≈0.003)68/64 → 70k(λ=0)18/18 → 75k 0/0,双 seed 对称;weansw(sandwich):65k = 66(该 run 新高)→ 70k = 0 `[实测]`。名义权重 0.3% 仍锚得住,零点后 5k 步内归零。**flip_cup 的"x2 seeds 崩溃"log 里没有**——weanfc2 只有 preregistration,早先 weanfc 的崩溃被 log 明确归因为 params-only warm-start artifact("与 λ 机制无关")`[声称:flip_cup 复现]`。"same speed"也不准确:sandwich 更快更彻底(70k 已 0 vs move_plate 的 18%)`[实测]`。

2. **为什么 0.003 够用而 0 不行(理论环)。** sparse reward + demo post-success tail(96% demo targets clip 到 top atom `[实测:demo 审计]`)意味着大量状态的 true action gap ≈ 0;factored argmax 在 15 dims × 3 levels 每个维度独立取 max,winner's curse 偏差 √(C/(m−1)) 按维度按层各收一次 `[文献:Thrun & Schwartz 1993;Double DQN Theorem 1, arXiv:1509.06461]`;deep network 的 greedy action 每个 gradient step 在 ~10% 状态上翻面,且 artificially 压低 suboptimal action(即 gap-increasing)可显著抑制 churn `[文献:policy churn, arXiv:2206.00730]`;一个只需超过 per-update jitter 的 anchor 即是 gap-increasing 且 optimality-preserving 的 operator `[文献:Bellemare 2016 Theorem 1, arXiv:1512.04860]`。所以"极小非零 vs 恰好零"行为不同有理论形状:anchor 是边界起效的 constraint,不是按权重连续起效的 loss `[文献:TD3+BC 梯度几何;DQfD margin ablation, arXiv:1704.03732]`。

3. **崩溃引擎。** argmax 是在 5^45 组合空间上搜索 critic error 的 optimizer,TD 把搜到的 overestimate 写回上游 target——iterative error exploitation,自增强故快而全 `[文献:one-step RL, arXiv:2106.08909;BEAR, arXiv:1906.00949;BCQ extrapolation error, arXiv:1812.02900]`。**但注意:这条机制的正面证认在本线是缺位的。**"hallucinated corners"是 log 的 interpretation;没有人测过崩溃后 argmax 是否真落在低 support 组合上且 Q 虚高、demo action 的 Q 是否仍正常 `[假设:eval-only——decode 崩溃 checkpoint 的 argmax bins,对 replay support 做 histogram,读 Q(argmax) vs Q(demo)]`。summary 的 fingerprint 数字(span 1.1→0.01,agreement 0.95→0.31–0.36,mean Q 持平 1.10)恰好就是这个测量的答案——**但它们不在 log 里,且量纲对不上 log 的健康值(top2 gap 0.15–0.38)** `[声称:全套 fingerprint 数字]`。

4. **探索不能替代,suppressor 能。** 178B:8x explore + λ=0 → 0/200 崩 `[引用]`;178A:同 8x explore + unseen-bin return floor + λ=0 → 72.5 不崩 `[引用]`;qselect3:崩溃的 75k critic 从 0% 被 decode-domain restriction 拉回 52%(random 58,solos 54/50,同噪声带——restriction 贡献 100%,judge ranking 贡献 0)`[实测]`。三件事共同指向:**相变变量不是 λ 本身,是"是否存在任一 unseen-action suppressor"(hinge / floor / decode restriction)。** 这直接修正 claim 1 的绝对化表述——λ=0 本身是可存活的 `[引用:178A]`。文献侧完全同构:anchor-free 而稳的方法全部带替代 constraint(RLPD 的 LayerNorm + min-ensemble + 50/50 sampling;PEX 的冻结 anchor;IDQL/BCQ 的 decode restriction),**没有任何一篇演示裸退火到零而不崩** `[文献:RLPD arXiv:2302.02948;PEX arXiv:2302.00935;IDQL arXiv:2304.10573;Adaptive BC Reg arXiv:2210.13846 明言 decay-to-zero brittle]`。

5. **为什么探索数据在 buffer 里也没用(结构原因)。** TD 只修 buffer 里出现过的 (s,a);argmax 每个 gradient step 都在向剩余 overestimate 移动,coverage 每个 env step 才涨一点,这场赛跑在任何现实数据率下都输 `[文献:tandem effect——bit-identical 数据流,passive twin 照样崩,>75% 状态 greedy 不一致;错误集中在 behavior policy 没采的 action 上]`。sparse reward 下崩溃期探索流全是 return≈0 的失败 rollout,只能逐点压已访问的 OOD bin,重建不了 on-support 排序。sibling bins 无自己的 TD target、仅靠 hinge 压制的说法与此吻合,**但这是从未在 log 验证的 code-level claim,且对 floor 类配置(rfloor/178A)明确不成立** `[假设:gradient probe——λ>0 vs λ=0 时测到达 sibling logits 的 gradient norm]`。

6. **崩了回不来。** wean0 的 50k 纯 TD 尾巴零自恢复 `[引用]`;文献预测同向:sparse reward 下 bootstrapped target 序列造成 target-fitting capacity 损失,累积且不自愈——崩溃是 absorbing state `[文献:capacity loss, arXiv:2204.09560;forked tandem——从好 snapshot 出发 passive 训练照样快速衰减]`。

7. **tie-breaking 理论已死,但尸体上有一道没解释的口子。** Wave-8 的可证伪预测(崩溃形态应随 gap 结构变化,14x sensitivity spread)被 sandwich 的同样-或-更狠崩溃杀死 `[实测]`——hinge 的承重不是 demo-sibling tie-breaking。但残余的任务差(sandwich 更快)方向与 sensitivity 排序**相反**,corner 理论也不解释它 `[实测,未解释]`。

---

## 2. 文献对位表

| 我们的发现 | 文献对应 | 新颖性判定 |
|---|---|---|
| λ plateau(1.0→0.0125→0.003 皆安全)+ cliff at zero `[实测]` | TD3+BC 的小权重带 + 零点崩溃;ReBRAC:系数是唯一要紧超参但宽带可用、零不可用;DQfD margin ablation | **已知定性现象**。我们的增量:threshold 压到名义 0.003、崩溃速度 ≤5k steps、双任务复现——比文献中任何数据点更极端地支持"constraint 不是 loss" |
| 崩溃机制(unseen 组合被 argmax 选中)`[假设,未正面证认]` | BCQ extrapolation error;deadly triad off-policy 是最强失稳因子(prioritized 采样把 soft-divergence 从 52% 推到 77%);one-step RL 的 iterative error exploitation | **已知**。注意 C51 bounded support 把我们保在 van Hasselt 的"soft"域——崩的是 ordering 不是 magnitude,这与 mean Q 不炸的(未验)指纹一致 |
| 崩溃速度(5k 步内 68→0)`[实测]` | policy churn:单次 update 翻 ~10% 状态的 greedy action;AL-式 gap 抑制 churn | **已知机制,我们提供了带中间态(18%)的干净时间切片** |
| resurrection:decode restriction 0%→52–58%,judge 贡献 0 `[实测]` | BCQ/IDQL:collapse 是 action-selection pathology,restrict argmax 即恢复 | **恢复本身是 BCQ 的预测复现**。但 q(52) ≤ random(58) 比 BCQ 预测更悲观——BCQ 预期 on-support ordering 存活,我们连"存活"都没建立(log 自己承认 confound)。**若 v2 probe 的 inversion(healthy 3.7% vs 33% chance)坐实,是文献中没有直接对应物的新发现——但它现在是 `[声称]`,log 里只有 preregistration** |
| 探索/replay 数据不能替代 anchor(178B 0/200;combined −2.1/−1.5)`[实测+引用]` | tandem effect:passive 数据结构性修不了 untaken actions;RLPD 明言承重的是替代 constraint 不是数据混比 | **已知**,178B 相当于 tandem 结论在 demo-driven 机器人设定下的 field replication |
| floor 防崩但收 −8 税;178A 72.5 `[引用]` | CQL 家族:pessimism-by-loss 防 OOD escape,代价是 over-pessimism tax(MCQ 一文即为降税而生) | **已知**,含税额都对得上量级 |
| +64% fake upslope / counterfactual 注入 → −19~−23% true devaluation / 行为纹丝不动 `[实测]` | extrapolation error(定性);"anchor 在位时 performance 不随数据/λ 缩放"的共识 | **devaluation 与行为的三次复现 decoupling 作为直接测量,相对新颖**;文献只有定性版本 |
| coarse-only 信息通道,fine-level 探索物理不可见 `[实测]` | 未找到对应物 | **疑似新颖**,特定于 coarse-to-fine factored action space;发表前需先关闭 §1(a)-1 的恒等式假设和 harness plumbing 疑点 |
| 崩溃 task-independent,残余方向反着 sensitivity 排序 `[实测]` | action-gap 理论预测 gap 依赖 | **与 Farahmand 表面张力**:可用"per-update jitter >> 全部任务的 gap 尺度"自洽,但这是 `[假设]`,且 sandwich-更快无解释 |
| 崩溃不自愈(wean0 50k 尾巴)`[引用]` | capacity loss / forked tandem / primacy bias | **已知预测,我们的数据一致** |
| rfloor from-scratch −26:小 λ 的安全是 path-dependent(需早期高 λ 塑形)`[实测,双变量混杂]` | Cal-QL 的 handoff 敏感性、Adaptive BC Reg 的 schedule 脆弱性,方向一致 | **部分新颖但证据不干净**(floor 与 λ÷80 同时变,log 自己归因 λ 侧) |

---

## 3. 未排除的替代解释(按可信度排序)

1. **"threshold 被穿越"而非"零点极限不连续"。** schedule 在 65k–70k 连续走完 0.003→0,之间零测量;"极限不连续"整个压在一个 65k checkpoint 上,而 65k 的 68/64/66 是单次 50-ep 读数(±7pp noise,30k lookpoint 的教训同样适用于安全侧)。*判别测量:constant λ=1e-3 / 1e-4 arm,或对现有 run 65–70k 做 1k 间隔 dense snapshot eval(纯 eval)。1e-4 也崩 ⇒ 是 threshold 不是 limit,整个"geometric constraint"叙事要改写成"最小有效权重"。*

2. **optimizer / target-network shock。** 突然移除一项 loss 会把梯度尺度砸进 stale 的 Adam second moments;5k 步崩溃可以是 optimization transient + TD 正反馈,与 argmax hallucination 无关。**这类失败在本 codebase 已被证明存在**:weanfc 的 params-only warm-start(等效 optimizer reset)单独就造成 41.5→2–6%、4–9k 步、λ-independent `[实测]`。*判别测量:零成本——现有 train.csv 画 60–80k 的 critic_loss / grad norm / mean & max Q。Q 发散 ⇒ dynamics 故事;Q 稳定但排序重排 ⇒ corner 故事。*

3. **λ=0.0 的 code-path 不连续。** step_linear 到精确 0 可能走不同分支(hinge 计算被 skip、JIT graph 变化、0×loss 的退化梯度)。*判别测量:λ=1e-12 一个 arm。存活 ⇒ implementation artifact;崩 ⇒ 真机制。与 #1 的 1e-4 arm 合并即一次三方判别。*

4. **历史锚点(178A/178B/wean0)的 lifecycle 卫生未审计。** 这三个 run 是"探索不能替代/floor 能替代/不自愈"三根支柱,全是跨线引用;weanfc 证明 warm-start 伪影能独立复刻"崩溃"。*判别测量:审计这三个 run dir 的 optimizer/sidecar state(true resume vs params-only)。*——排位靠前是因为 §4 的结论直接压在 178A 上。

5. **replay composition 死亡螺旋。** 70k 后 online buffer 被崩溃中 policy 的失败灌满,18%→0% 可能是 data-driven 而非纯 value pathology(但解释不了 68→18 的起跳)。*判别测量:log 里读 65–80k 的 online-buffer success rate 与 demo_batch_fraction;或一个 freeze-online-buffer-at-70k rerun。*

6. **C51 atom-distribution 退化。** 96% demo target 打 top atom,hinge 可能是 categorical head 分布不退化的唯一支撑;崩溃可能是 distribution-support pathology,特定于 C51。*判别测量:dump 65k vs 75k 在 demo states 的完整 atom 分布,纯 eval。*

7. **dose-response harness plumbing。** 该 harness 中途修过 num_train_envs=batch bug,per-level realized 探索率从未对账;§1(a) 的"coarse-only 通道"结论(疑似新颖发现)依赖它。*判别测量:对账 logged per-level explored-step counts vs [.002,.004,.008] 与 [.03,0,0] 的期望值。*

8. **0% 读数的 eval artifact。** 双 seed 双任务加 18% 中间态使其可能性低,但没有任何 rollout 被肉眼确认过。*判别测量:render 崩溃 checkpoint 一集(顺带直接检验"frozen corner action"预测)。*

9. **qselect3 的 proposal-quality confound。** 52–58% 可能只测了"健康 checkpoint 的动作本来就好"。log 自己已让步;v2 probe(garbage + self-hijacked candidates)**按 log 仅 preregistered 未跑**。*判别测量:跑 v2。*

---

## 4. 结论

### 瓶颈是什么(按证据说话)

**瓶颈不是信息,是 constraint 的不可移除性,以及 constraint 在位时信息的不可兑换性——两个旋钮拧的都是错误的轴。**

- 探索旋钮:提供的是 fence 免费提供的同类证据的贵版本 `[实测:−36pp vs 零成本]`,且带宽只有 coarse level `[实测,恒等式假设未闭合]`;
- 退火旋钮:在 plateau 上(≥0.0125)hinge 不是 teacher 而是 projection,松它不释放任何被压制的能力(三次注入实验 + rerank 矩阵全 null `[实测]`),而拧到零等于移除唯一的 unseen-suppressor,factored argmax + churn 在 ~5k 步内解体 `[实测+文献]`;
- 唯一在两个方向上都被测到"起作用"的变量是 **suppressor 的有无**(hinge / floor `[引用:178A]` / decode restriction `[实测:qselect3]`),不是 λ 的数值,不是数据的量或质。

诚实的封底:除 collapse 本身外,本线全部行为效应都在 baseline 的 11.5–13.5pp seed spread 之内 `[实测]` ——log 自己的结论"全部行为效应居于种子噪声带"仍然是对 criterion 1 最准确的一句话。且以上结论有三处软肋:λ=0 机制未做正面证认(§3.2/3.3)、178A 支柱未审计(§3.4)、"安全侧"证据薄于"崩溃侧"(§3.1)。

### 最高价值的下一步实验(按性价比排序)

**E1. λ 微尾判别 + 免费 forensics(一张卡 + 零成本读盘)。** 现有 zero70/weansw 的 train.csv 画 60–80k critic_loss / grad norm / mean-max Q;同时开 constant λ ∈ {1e-3, 1e-4, 1e-12} 三个尾巴 arm(或至少 1e-4 与 1e-12)。预测:
- *geometric-constraint 理论*:1e-3 存活,1e-4 边缘,1e-12 崩(梯度低于 jitter 即等效于零),train.csv 中 Q 稳定但排序重排;
- *code-path artifact*:1e-12 存活(loss 分支仍在),只有精确 0.0 崩;
- *optimizer-shock 理论*:凡 schedule 有突变的都崩、constant 的都活,train.csv 中 grad norm / Q 出现 transient;
- *threshold 理论*:存在 λ* ∈ (1e-4, 3e-3),两侧连续变化。
一组实验四方判别,直接决定结论的措辞是"limit"还是"threshold"还是"bug"。

**E2. 崩溃 checkpoint 尸检套件(全 eval-only,数据在盘上)。** 对 75k 崩溃 checkpoint:decode argmax bins 对 replay support 做 histogram;读 Q(argmax) vs Q(demo-action);dump demo states 的 atom 分布(65k vs 75k 对照);render 一集。预测:
- *corner 理论*:argmax 落在低 support 组合、Q(argmax) 虚高、Q(demo) 正常、atom 分布健康——即把 `[声称]` 的 fingerprint 变成 `[实测]`;
- *dynamics 理论*:Q 全域畸变;
- *C51 退化理论*:demo states 的 atom 分布本身塌缩;
- *eval artifact*:render 出来行为正常。

**E3. fence-v2 × λ→0 组合 arm(pending approval 的那个)+ v2 judge probe。** 唯一未测的实质性问题:**true near-manifold 数据(v2,manifold distance 0.000)能否替代 hinge**——目前"fence 不能替代"只对 v1 垃圾数据成立 `[实测:zero70/weansw 带的是 v1,erratum #4 判其 off-manifold]`。预测:
- *support-constraint 文献共识*:照样崩(数据不在 RLPD 列举的替代机制清单上)——那么结论闭合为"必须换 constraint 类机制"(decode restriction / calibrated floor / LayerNorm+ensemble);
- *信息论替代假说*:存活——那么瓶颈叙事翻转为"是 v1 数据质量问题,不是 constraint 问题",本文 §1(b) 需要重写。
v2 judge probe 顺带把"崩溃 critic 的 surviving 结构是 dead 还是 inverted"从 `[声称]` 收进 `[实测]`,决定 resurrection 一节能不能写。