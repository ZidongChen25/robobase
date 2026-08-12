# cqn-rline.md — R 路线执行日志（文献驱动的 CQN-AS 改进线）

## 0. 当前状态总览（每次更新盖写此节，2026-08-12 10:40）

- **四种子终审：cfaug 未过声明门**（配对 +10.5/+1.0/−0.5/−6.0，均值
  +1.25，2/4 正；基线自身散布 13.5pp）。四种子纪律第三次拦下 n=2 超读。
- **criterion 2 已成立**：复权探针定案——基线假上坡 +64%（净），cfaug
  held-out 真贬值 −19%；机制三层证据完整。**criterion 1 未达成**：全部
  臂的行为效应居于种子噪声带。
- **待用户裁决的分叉**：剂量升级 / cfaug+MC 锚组合终局臂 / 机制级定位
  收官。o2o 阶梯（cfaug-o2o → noBC）在队列，премise 待终局臂结果。
- **已关闭**：rfloor、nstep3、分视野×3任务、b8、flip03、AL/原子K步、
  重排（实测判官只会贬值不会排序）、双头（依附重排，未启）。
- **方法论资产**：统一定律（gap 税×任务敏感度）、复权口径、四种子
  纪律、三份 HANDOFF 勘误、进程/评估纪律全套。
- **文档**：文献路线 → `research_paper.md`；执行判决 → 本文件。

---

开线 2026-08-09。上游文档：`research_paper.md`（72 篇文献 + 路线 R1–R7 +
预注册卡）。目标（用户设定）：(1) move_plate 上显著超过原版（非噪声）；
(2) 展示真实 value learning 行为，而非"伪装成 RL 的 IL"。

对照制度：**当前默认制度**（`truncate_demo_at_success=true` +
`append_floating_base_to_low_dim=true` 63 维，均为 config 默认），
canonical launch `cqn_as_pixel_bigym_demo_driven`，101k 预算，5k snapshot，
val50 seeds 400–449 · ne=25，密封 200ep seeds 800–999 @ 100k 整。
**基线参照（已存在，不重跑）**：`exp_local/cqn_trunc_arms/
official_basestate_move_plate` seeds 1/2 —— 密封 100k = **77.5 / 73.0**
（均值 75.25），完整 5k 间隔 val 曲线在 `val50_seeds400.csv`。

## Wave-1 预注册（2026-08-09，发射前写定）

### 设计裁定记录（为什么不是 research_paper.md §5 原卡）

1. **R1a（AL gap 算子）撤销**：cqn-no-bc.md Stage 20/21 已在 dense-offline
   语境正面证伪 gap-operator 家族（constant α=0.5 → 2% vs 对照 47%；
   clipped c=0.9 → 11%；判决原文"rules out the tested gap-operator
   family"，机制 = early wrong-greedy lock-in）。按"已证伪不再审"铁律，
   本线不从头重跑 AL；gap 需求转由架构类方案（Mean-Expansion Layer，
   wave-2 候选）与 floor+小 λ 组合承接。
2. **R2a-原子 K 步 backup 撤销**：stage-163c/164/171/177 已穷尽测试
   （nstep8+replan8 旧制度 +9.4 / 新制度 qc 臂 73.0 未超基线 / 38pp
   种子方差全源 / NaN 高危组）。R2 的新颖成分只剩 per-prefix 多地平线
   target（SEAR 式）——实现成本高（需前缀边界特征），排 wave-2。
3. Wave-1 改为两个**零代码、有直接实验室先验**的臂（见下）。

### 臂 α "rfloor"：floor + 恒定小 λ（§64.5 收敛方案 from-scratch 首测）

- **假设**：`unseen_return_floor`（w=0.1, value=0, mean；demo-agnostic 的
  兄弟 bin 可分辨性来源）+ **恒定** `bc_lambda=0.0125`（排序锚，80× 低于
  官方的 1.0）从头训练，密封不低于基线，且 critic 的价值语义显著改善
  （模仿压力 ÷80 → F1 直接缓解）。
- **先验**：stage-174（λ→0.0125 续训 82.5，旧制度）；stage-178A（floor
  救 λ=0 从 0% 到 72.5，续训）；§64.5 明示 from-scratch 未测。
- **launch**：`cqn_as_pixel_bigym_rline_floor_lambda_gate`（resolved diff
  vs canonical 已审计 = method.bc_lambda 1.0→0.0125 +
  method.unseen_return_floor_weight 0→0.1 + 与基线 CLI 相同的 infra 键）。
- seeds 1/2，GPU3（UUID `GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`，
  EGL 3，本日实测放置验证通过），0.45 mem fraction，120s stagger，
  `JAX_PLATFORMS=cuda ROBOBASE_HOST_MERGE=1`。

### 臂 γ "nstep3"：canonical + replay.nstep=3（单变量）

- **假设**：3 步不修正回报窗（DQfD/R2D3/Rainbow 标配；stage-177 在旧制度
  最差 seed 拿 79.5 的配方成分）在新制度、标准 replan-1 执行下，
  密封超基线。
- **先验**：163c（nstep8 新制度未超基线）、177（nstep3+replan1 方差消失、
  79.5）；nstep3+replan1 新制度未测。
- **launch**：`cqn_as_pixel_bigym_rline_nstep3_gate`（resolved diff =
  replay.nstep 1→3 + infra 键）。
- seeds 1/2，GPU4（UUID `GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08`，
  EGL 0，放置验证通过），其余同 α。

### 门（发射前写定）

- **wiring 阳性对照**（解释任何结果前必须过）：α 的 train.csv 首批行
  `bc_weight=0.0125` 且 `unseen_return_floor_loss>0`；γ 的 launch log
  `nstep: 3` 且首批 reward 统计与 3 步和一致。两臂 resolved-config diff
  已在发射前审计（见上）。
- **30k 中途看点**（信息性，不选点）：从空卡评 25k/30k snapshot 副本，
  与基线同 seed 同步点（基线 s1: 25k=74, 30k=74）比较。**kill 线**：
  某臂两 seed 都 <40% → 停臂止损，记录。
- **密封主门（100k, 200ep, seeds 800–999, 无选点）**：
  - 臂均值 ≥ 基线均值 (75.25) + 3pp 且两 seed 配对 Δ ≥ 0 → 晋级
    seeds 3/4（四种子声明制度）；
  - (−2, +3) 区间 → 机制探针裁决是否进组合 wave；
  - ≤ −2pp → 此形态关臂。
- **机制门（criterion 2，两臂都做，α 为主）**：100k checkpoint 对基线同
  seed checkpoint 的配对 value-fidelity 探针（固定 demo 状态集、
  `analyze_cqn_value_fidelity.py`、CPU）：Q-vs-RTG Spearman、expert-bin
  top-1、`Q(expert)−Q(greedy)`。α 预测：Spearman ↑（模仿压力 ÷80 后价值
  语义恢复）。**A21 仪器警告在案**：跨 run 比较必须同批配对跑探针。
- **最终声明门（goal criterion 1）**：需 4 seeds：held-out 均值超基线
  >3pp 且 ≥3/4 配对为正（比 no-BC 线的 +1.75 未显著更严）。criterion 2
  终审 = qselect3 竞技场（q > max(solo) 且 > random）+ 探针改善。
- **停止规则**：NaN → 按 §68 取证保留现场再相机重启；EGL 启动崩 →
  重试/换卡；训练进程消失 → 顺序重启该 seed。

### 发射记录

2026-08-09 01:08 BST，stamp `20260809rline1`，控制器 PID 2715416
（`scripts/launch_rline_wave1.sh`）。seed1 双臂 01:08 起跑，seed2 各
+120s。进程树单实例验证通过（每臂每 seed 恰一个 train_fast.py，
cmdline 捕获 run dir 一致）。GPU3/GPU4 各 ~4.9GB（JIT 期）。

Wiring 证据（首批）：
- γ/nstep3 seed1 launch log：replay 构造打印 **`nstep: 3`** ✓（demo 与
  online 两个 buffer 均为 3）；
- α/rfloor seed1 launch log：`nstep: 1` 如常 ✓；bc_weight/floor loss 列
  待 train.csv 首批行（见下条追记）。

监控：Monitor 挂 NaN/进程死亡/30k snapshot 三类事件（5 分钟轮询）。

**Wiring 阳性对照通过（01:25）**：α seed1 train.csv @1k 行：
`bc_weight=0.0125` ✓、`unseen_return_floor_loss=4.1e-05 > 0` ✓（零初始化
下所有 Q≈floor=0，违反量小是预期的；随价值展开会咬合）、critic_loss
有限；γ seed1：replay `nstep: 3` ✓、train.csv 正常写行。两臂干预确证
生效，结果可以被解释。

## Wave-2 预注册："tokensplit"（实现完成 2026-08-09 02:2x，待 GPU 槽位）

- **假设**：per-token 地平线分裂（SEAR 移植的 CQN-AS 形态）——token 1..b
  保持精确 legacy 1-step backup，token b+1..K 回归 auxiliary_nstep 地平线
  backup（同起点、episode 末端 clamp 的 aux 窗）——兼得早 token 精度与晚
  token 的稀疏奖励传播，且不改执行（replan-1 ensemble 不动，与
  163c/171/177 的区分点）。move_plate：K=4，aux=4，b=2。
- **实现**：`method.token_split_horizon_targets` + `token_split_boundary`
  （四层链全动：cqn_as.yaml / CQNASpec+reader / factory 表 / __init__
  验证+赋值）；核心在 `cqn.py` canonical loss——aux 五元组经 adapter 尾部
  传入，aux 状态经同管线增广，target 用 `jnp.where(token_mask)` 按 token
  混合两个 projected 分布；replay 零改动（复用 auxiliary_nstep 机件，
  scalar+vectorized 都支持）。launch
  `cqn_as_pixel_bigym_rline_tokensplit_gate`（resolved diff 已审计 =
  恰 3 语义键）。
- **测试**：`tests/unit/test_cqn_as_token_split.py` 10/10——含四层链
  round-trip、五类拒绝、**aux==primary 时与 legacy 参数树逐位等价**（发现
  并规避了"零初始化头下首步 loss 与 target 无关"的假阳性坑：断言改为
  一步更新后参数树）、aux 地平线真实消费（reward_aux +1 → 参数树分歧）、
  wiring 指标（aux_fraction=2/3、aux_reward_mean 跟随）。
- **门（预注册）**：与 wave-1 相同协议（同 seed 对基线 val 曲线 30k 中途
  看点 + 密封 100k 配对主门 + 机制探针）。发射条件：wave-1 任一卡完成。
- **回归备注**：test_cqn_as.py 全量 223 过 / 11 个 `*demo_agnostic*` 失败
  为**先前会话遗留**（§64 λ-仪表把 `bc_agreement` 等诊断无条件并入
  metrics，测试的 metrics 字典全等断言必然破；参数相等断言——真正的
  反模仿契约——全部通过；qc 臂 08-04 train.csv 已有这些列可证早于本线）。
  不属本线破坏，留待仪表归属方修。

### Wave-1 30k 中途看点（2026-08-09 03:40，GPU2 评估，信息性不选点）

val50 seeds 400–449（±7pp 噪声带；基线数字取其 val50_seeds400.csv 同
合同实测，注意与早前引用的 val50_early.csv 74/74 不一致——以同合同
csv 为准）：

| 臂 | s1 25k/30k | s2 25k/30k | 30k 配对 Δ |
|---|---|---|---|
| 基线 | 70/66 | 60/54 | — |
| nstep3 | 62/68 | 70/70 | +2 / +16（均值 +9，两 seed 全正）|
| rfloor | 54/58 | 44/50 | −8 / −4（均值 −6）|

判读：nstep3 领跑；rfloor 落后但两 seed 均在 kill 线（40%）之上，且
慢启动与 λ÷80 的机制预期一致（§64.5 预测后程反超），按预注册继续。
无 NaN / 无崩溃 / 无 failed 哨兵。25k/30k snapshot 已按 sweep 合同转为
params-only checkpoint（续评时自动跳过）。

## Wave-1b + Wave-2 发射预注册（2026-08-09 ~09:00，用户加拨两张卡后）

用户拨款：训练预算 2 卡 → **4 卡**。新卡取 GPU2（空）与 GPU5（空；
其"infra 基准专用"预留在本次用户明示加卡下让位，冲突时 infra 优先）。
GPU0（显示卡）与 GPU1（他人 5.9GB）不用。

- **Wave-1b**：nstep3 seeds 3/4 → GPU5（UUID `GPU-2f044e6a-…`，EGL 1，
  发射前渲染探针实测通过）。理由：30k 配对 +2/+16 全正，最终声明需要
  4 seeds；现在补种子把"若 100k 过门"的确认周期压缩一天。若 100k 判决
  杀掉 nstep3，损失 = 一卡·半天，可接受。门与 wave-1 相同（同 seed 配对
  基线不存在 s3/s4 → 配对参照改用基线 §67 四种子分布 78.5/73.0/82.0/82.0
  的均值口径，并明记 n=2→n=4 的参照升级）。
- **Wave-2**：tokensplit seeds 1/2 → GPU2（UUID `GPU-80b9cc0d-…`，EGL 2，
  midlook 已实证渲染放置）。预注册见上节；30k 中途看点与 kill 线同
  wave-1。
- 共用：`xla_mem_fraction=0.45`、120s stagger、`JAX_PLATFORMS=cuda
  ROBOBASE_HOST_MERGE=1`、UUID pin。

### Wave-2 tokensplit wiring 阳性对照（发射后首批 train.csv）

`token_split_aux_fraction = 0.875`、`token_split_aux_reward_mean = 0.017`
（有限、非零）。**0.875 = 14/16 揭示实际 K=16**（canonical BiGym
demo_driven 的 action_sequence=16，与 qc 臂 train.csv 的
`env_info_action_sequence_mask0..15` 互证）——HANDOFF §1 写的
"move_plate action_sequence=4" 与现行 config 不符，记为文档勘误，勿沿用。
boundary=2 在 K=16 下恰为"最近地平线分配"规则（可选 {1,4}：token ≤2 →
1-step，≥3 → 4-step），与设计意图一致，臂照预注册继续。后续变体空间
（aux=8 或多段分裂）留待本臂 30k/100k 读数后再议。

### 基建事故 + 修复（2026-08-09 ~09:3x）：runner 的 `$#` sed 炸弹

wave-1 四个 runner shell 在 val50 完成后全体静默死亡（无哨兵、无密封
评估）。根因：`run_cqn_trunc_arm.sh` SKIP 计算里
`sed -n "s#…${CHECKPOINT_SUFFIX}$#\1#p"` 的 **`$#` 在双引号内被 bash
展开为位置参数个数**，sed 收到未终止的 `s` 命令 → `set -e` 杀死 shell。
该分支（eval_checkpoints 存在时）是 §65 artifact-lifecycle 改造后新增，
qc 臂时代未覆盖，首次真实执行即引爆。
修复：`\$` 转义；**用 mv 原子替换**（wave-1b/2 四个 runner 正在执行同
一脚本，就地编辑会损坏运行中的 bash——旧 inode fd 不受 mv 影响，它们
死在同一行之前会先训练 ~6h，届时读的仍是旧 inode……实际上它们 exec 时
已打开旧文件，将同样死在 sed 行：**已预置恢复脚本**，届时直接复用）。
恢复：`scripts/recover_rline_wave1_sealed.sh`——GPU3 补 rfloor ×2 密封、
GPU4 补 nstep3 ×2 密封 + nstep3 两 seed 的 seeds-450 确认档（执行用户
新规则：被引用的 50-ep 点须凑 100 集，已存 memory `val50-confirm-100`）。

### 教训沉淀

1. 曲线引用违反了"同种子配对"铁律一次（拿 nstep3 s2 对比了跨 seed 的
   "基线 ~54-60"——那只是基线 s2；基线 s1 同期 70-84）。已当场纠正，
   后续一切曲线引用必须标明配对 seed。
2. 50-ep 单点 ±7pp：新规则 = 被引用点补 seeds 450-499 凑 100 集。

## 后续臂预注册（2026-08-09 ~10:45，四卡饱和方案）

用户明确 4 卡训练预算后的排程（评估改走 GPU0/GPU1，async 协议归位）：
- **basestate 基线 seeds 3/4** → GPU3（密封收尾后自动接档，
  `launch_rline_next_arms.sh`，与 s1/s2 完全同合同含显式
  `env.append_floating_base_to_low_dim=true` override）。动机：一切
  4-seed 配对声明（nstep3 s3/s4、flip03 s3/s4、未来臂）都缺同制度基线
  s3/s4 参照；这是方法论债，先还。
- **flip03（post-ensemble L1 flip p=0.03 h=4）seeds 3/4** → GPU4。
  HANDOFF §6.2 悬置的功效确认：n=2 均值 78.25 vs 基线 75.25（+3.0
  两 seed 同向），是独立于 R 线机制臂的实用配方候选。门（预注册）：
  四 seed 配对（s1..s4 对基线同 seed）均值 ≥ +3pp 且 ≥3/4 为正 →
  流入最终声明制度；否则 flip 家族关闭（与 HANDOFF §6.2 原门一致）。
- nstep3 s1/s2 确认档（seeds 450-499）→ GPU1（EGL5 探针验证过）。
- 交接编排：`rline_sealed_handoff.sh`（四密封齐 → 杀恢复尾巴 → 发新臂
  → GPU1 确认档）。

### Wave-1 密封判决（完整，200ep seeds 800-999 @100k，2026-08-09 11:4x）

| 臂 | s1 | s2 | 配对 Δ（对 77.5/73.0） | 均值 Δ | 判决 |
|---|---|---|---|---|---|
| rfloor | 51.5 | 46.5 | −26.0 / −26.5 | **−26.25** | **关闭**（两 seed 一致） |
| nstep3 | 68.0 | 76.0 | −9.5 / +3.0 | **−3.25** | **关闭**（≤−2 门） |

纪律动作：nstep3 s3/s4 按 wave-1b 预注册条款处死（GPU5 释放，
`killed_by_gate` 哨兵入 run dir）。101k 点与 nstep3 seeds-450 确认档
仍会落盘（存档用，不改判决）。

**两条结论性教训**：
1. **30k val 看点方向性有限**——nstep3 30k 配对 +9（两 seed 全正）→
   100k 密封 −3.25。中途看点只配当 kill 线，不配当晋级信号；晋级判断
   一律等密封。
2. **§64.5 的 floor+恒定小λ 不能 from-scratch 移植**——旧证据（174 的
   82.5、178A 的 72.5）全部是"高 λ 训满 100k 后续训"制度；从头 λ=0.0125
   下行为学习本身受损（两 seed 46-52 一致）。λ 的第三份隐性工作 =
   **早期行为塑形**，比 §64.5 的"可分辨性+排序"二分多出一项，且不可
   被 floor 接管。此结论对 no-BC 线同样有效：margin 替代方案必须解决
   "早期从 demo 中提取行为"的通道，仅解决 margin 维持是不够的。
   机制尸检探针（在跑）将检验：rfloor 的 critic 排序质量是否至少
   改善了（若是，"行为差但价值好"→ 提取问题；若否，全盘否定）。

### 机制尸检（同批配对探针，6 checkpoint @100k，demo_success 组，n=48/组）

| 臂 | Q-vs-RTG Spearman | 模仿 top1 | agreement | top2 gap |
|---|---|---|---|---|
| baseline s1/s2 | 0.532 / 0.663 | 0.991 / 0.992 | 0.962 | 0.366 / 0.377 |
| nstep3 s1/s2 | **0.865 / 0.878** | 0.989 / 0.983 | 0.955 / 0.935 | 0.267 / 0.213 |
| rfloor s1/s2 | **0.899 / 0.881** | 0.888 / 0.901 | 0.837 / 0.850 | 0.148 / 0.151 |

**判读（本线迄今最重要的机制结果）**：
1. **两个"失败"臂都真的学到了回报排序**——Spearman 从基线的 0.53-0.66
   （"伪装成 RL 的 IL"指纹）跳到 **0.87-0.90**，F1 被实质修复；criterion-2
   的机制通道是打得开的。
2. **但行为按 top2 gap 单调掉**：gap 0.37（基线，行为最好）→ 0.21-0.27
   （nstep3，−3~−10pp）→ 0.15（rfloor，−26pp）。这是 F2 的又一次定量
   显形，并与 §28 的"校准 vs 行为"权衡面、A18/A19 的"抗漂移间隔"结论
   三方互证：**价值信息进 critic 的代价是决策 margin 变薄**。
3. 合成靶点因此变得非常具体：**保住基线级 margin 几何（gap≈0.37）+
   拿到 nstep3 级排序（Spearman≈0.87）**。三条在场路径按此重排：
   - **tokensplit（在跑）**：结构上正是为此设计——早 token 保 1-step
     精度（margin 几何），晚 token 长地平线（排序信号）。它的 100k 探针
     是下一个关键读数；
   - **R6 two-head**（决策头 × 校准头）现在有了直接证据支撑，升优先级；
   - nstep3 变体（如 nstep3 + margin/gap 保持项）备选。
4. nstep3 s2 单 seed 已同时做到"+3.0 密封 & Spearman 0.878"——目标状态
   的存在性证明（n=1，不作声明只作方向）。
5. A21 合规注记：六个探针同一进程批、同 --seed、同 --data-run-dir
   （failed-attempt 目录的共享 demo 状态集），跨 run 可比性达到该仪器
   允许的上限。

### Wave-2 tokensplit 30k 中途看点（n=100/点，双 seed 档）

s1: 25k 73.0 (68/78)、30k 71.0 (64/78)；s2: 25k 65.0 (62/68)、30k 56.0
(56/56)。对基线 400 档同口径配对 ≈ ±2 持平，远离 kill 线，继续。
不做晋级解读（wave-1 教训）。附带发现：同 checkpoint 400 vs 450 档差
最高 +14（s1@30k 64/78）——n=100 规则的直接实证。

### 排程修订（2026-08-09 ~12:3x，用户质询后）

用户质询"为何跑基线和 flip"。裁定：
- **基线 s3/s4 保留**（GPU3）：最终四种子同 seed 配对声明的对照组，
  臂无关基础设施，一切成功路径的必经点。
- **flip03 s3/s4 撤销**（用户正确）：对 criterion-2 零贡献（critic 仍是
  BC-margin 形态），+3.0 即便确认也非本目标交付物，占卡属 HANDOFF 遗留
  惯性。s4 已先行自毙于已知 EGL 启动竞争；s3 处死，双 dir 落
  `cancelled` 哨兵。
- **GPU4 改派 tokensplit-b8**（stamp 20260809rline4，audit diff 恰
  `token_split_boundary: 2→8`）：aux 比例 0.875→0.5 的 margin 保守端，
  与在跑的 b2 一起夹住 aux-fraction 轴——尸检的 gap→行为单调关系预言
  b8 gap 更厚；若 b2 100k 显示 gap 变薄掉分，b8 已半程在训。
- 事故记录：撤 flip 时 `pkill -f` 模式自匹配杀掉了自己的 shell 与
  tokensplit 的 runner 壳（trainer 无恙；评估链本就由 tsfinish 编排器
  接管，零实害）。教训：**杀进程一律先 ps 精确到 pid，禁用 -f 宽模式**。

### HANDOFF 勘误 #2（用户提出并查实，2026-08-09）：63 维追加是纯冗余

实证（直接实例化 move_plate、推底盘 40 步、逐值比对）：
`proprioception_floating_base`（x/y/rz qpos）与 58 维 `proprioception`
的索引 [26,27,28] **逐位相同**（qvel 段同理含底盘速度）。BiGym 的
proprioception = 全关节 29 qpos + 29 qvel，**底盘自由度本来就在里面**
（robot.qpos 遍历 `self._body.joints`，含 base slide/hinge）。
推论：(a) 63 维 = 同一信息的第二份拷贝（另一套归一化），零新信息；
(b) §67 官方差距假设 B（obs contract 缺底盘）降级——官方 63D 同样是
冗余拷贝结构，解释不了 14.9pp；实测 60D 四种子 78.875 vs 63D 两种子
75.25 与"冗余无益"一致；(c) R 线内部配对不受影响（全线同合同）；
(d) 本轮维持 63 维合同不动（保配对性），赢家臂出线后再在 60 维干净
制度复核。HANDOFF §1 的"补上官方底盘信息"叙述作废。`analyze_cqn_value_fidelity.py`
以"正式 seed2 checkpoint × 失败attempt目录 replay 作共享数据源
（--data-run-dir）"模式端到端跑通。基线 s2 @100k（demo_success 组，
小样 n=8/组，仅作仪器验证不作结论）：q_raw_return_spearman **0.515**、
replay_bin_top1 **0.990**、behavior_bin_agreement 0.962、
candidate_top2_gap 0.361——完美模仿 + 平庸排序，正是 F1 指纹的定量形态。
100k 机制门将用同一命令、更大样本、同 seed、同 data-run-dir、同批
对所有臂 + 基线配对施测（A21 合规）。

### 2026-08-10 晨间收割：b8 关闭、基线 n=4 建成、b2 哨兵事故

- **基线四种子建成**：77.5 / 73.0 / **71.0** / **82.5**（s1-s4 @100k
  密封），n=4 均值 **76.0**。种子散布 11.5pp——配对设计的必要性再次实证。
- **tokensplit-b8 关闭**：密封 69.5 / 71.0，同 seed 配对 −8.0 / −2.0，
  均值 **−5.0 ≤ −2**。margin 保守端（aux 0.5）也不行——若 b2 亦负，
  token-split 家族现形态关闭。
- **b2 哨兵事故**：b2 训练 08-09 18:43 即完成（trainer 自身做了 §65
  finalize），但 `train_complete` 哨兵属 runner 壳职责——该壳在撤 flip
  的误杀中连带阵亡，tsfinish 编排器干等哨兵 ~14h。已补 touch 放行，
  评估链恢复运行。教训（进程纪律第三条）：**编排器等待条件应绑定训练
  产物本身**（如 101000 checkpoint 文件）而非中间层哨兵。

### Wave-2 tokensplit-b2 判决（2026-08-10 01:15，密封+同批探针）

密封 200ep @100k：s1 **76.0**（配对 −1.5）、s2 **78.5**（配对 +5.5），
均值 **+2.0** → 预注册 (−2,+3) 探针裁决区。同批探针：Spearman
0.700/0.802（基线 0.532/0.663）、top1 0.984/0.988、top2gap 0.205/0.209。
**判读**：(1) 排序显著改善 + 行为微升——首个同时朝两条 criterion 移动的
臂；(2) 与 nstep3 同档的 gap 税（~0.21）但行为差 +5.25pp——wave-1 的
gap→行为单调律需精化为"**执行主导 token（token-1）的 margin 几何**才是
行为的定价者"（replan-1 只执行 token-1，b2 恰好保它 1-step）；(3) b2
（aux 0.875）> b8（aux 0.5）5-8pp：保住 token-1 后，长地平线 token
越多越好。
**推进（依裁决区条款）**：b2 s3/s4 → GPU3（对基线 s3/s4 配对；101k 点
71.5/74.0 在档，低于 100k 点——固定 100k 报点纪律再次到位）；b1
（boundary=1）预注册为下一格，等卡；R6 保持待命。四种子声明门数值不变。

## Wave-3 预注册：数据通道矩阵（2026-08-10 ~13:3x，用户批准）

用户三问（失败数据/同 objective/配比调度）收束成的三杠杆方案，与 R6
两头架构并行：

| 杠杆 | 修的墙 | 形态 |
|---|---|---|
| D1 反事实增广 | 墙一（近流形反事实洞） | branch-state 恢复到 demo 锚点态 → 执行扰动 chunk → 滚到失败/截断 → 存为普通 episode（demo=0、真实回报），offline 与 demo 混训，objective 不变 |
| D2 配比调度 | 墙二（接缝悬崖→坡道） | demo/online 采样比 75%→25% 式退火 + 常数对照（本地 schedule 前科在案，必须对照） |
| D3 buffer 内容策略 | 漂移反馈环 | 冻结（已验证）vs HIL-SERL 式纠正性生长（后置） |

**注入纪律（本地教训铸成）**：失败数据以"执行动作的真实低 MC 回报"进入
（排序信号），严禁整行压 floor（plain-dense 的 margin 塌缩教训）。
顺序：先 D1 单杠杆（对基线同 seed 配对），D2 单杠杆（对常数对照），
过门再组合。

### D1/D2 实现记录（2026-08-10 下午）

- **D2 配比调度实现完成**：`demo_batch_fraction_schedule` config 键
  （null=精确 legacy）；`Workspace._apply_demo_batch_fraction_schedule`
  每 update 块前按 `utils.schedule` 改双 buffer 采样量（总量恒定保
  JIT 形状；num_workers=0 强制；realized fraction 记入
  `demo_batch_fraction` 指标）。测试 5/5。
- **D1 生成器实现+验证完成**：`scripts/generate_counterfactual_episodes.py`
  ——demo seed 重放到锚点（cache_bigym_pixel_demos 已验证机制）→
  capture/restore 分支 → 单维 L0-bin 尺度扰动（±0.4 tanh 空间）持续 4 步
  → 开环续放 demo 剩余动作 → 按 UniformReplayBuffer schema 落 npz。
  冒烟：2 episode 皆 return 0（扰动确实杀死成功=有据近流形负样本）；
  schema 对真实 replay 文件 11 键逐一 shape/dtype 匹配（mc_return 默认
  不写，canonical 签名无此键）。全量生成在跑（60 demo × 6 锚点，
  上限 360 episode，GPU3 渲染）。
- **D1 注入路线（侦察定案）**：预填 `<run>/replay/` + `replay.reuse_saved
  =true replay.persist=true replay.demo_cache_dir=null`——零代码；文件名
  `{ts}_{eps}_{len}_{global}.npz` 索引连续；master 目录只复制不链接
  （delete_replay_on_train_complete 会清目录）。

### R6+R4 机件盘点与阶段计划（2026-08-10 15:3x）

现成机件：stage-148 侧车全链（`run_cqn_lcb_sidecar_gate.py` 训练、
`eval_cqn_lcb_sidecar.py` 侧车参与动作选择的评估、
`analyze_cqn_sidecar_calibration.py`）+ agent 内建 `_joint_beam_action`
（cqn_as.py:3013）。
**Phase 1（零训练，先行）**：(1a) tokensplit-b2 s2 checkpoint（排序 0.80）
自重排——束搜索评估 vs 普通 argmax 的 val50 配对；可证伪预测：束搜索
帮 b2（会排序的 critic）而不帮基线（频率 critic）——若成立即"价值学习
驱动行为改进"的最小闭环证据；(1b) 基线 + 侧车判官评估
（eval_cqn_lcb_sidecar 路径）。
**Phase 2（训练版两头）**：仅当 Phase 1 有信号才投卡。

### R4 Phase-1a 首发受阻 + 修复路径（2026-08-10 16:0x）

补丁视图目录（beam_width=8 的 config 视图 + 符号链接 checkpoint）构造
即被拒：cqn_as.py:1970 校验将 twin_rollout_beam_width>1 硬绑
pessimistic_twin_critic+episodic_twin_head_exploration；act 路径的束搜索
路由也仅存在于 twin 谱系。`_joint_beam_action(critic2=None)` 本身支持
单 critic——**下一步（明确的小实现）**：(1) 校验放宽为"twin 或
单 critic 均可"；(2) 非 twin act 路径在 beam_width>1 时路由到
_joint_beam_action（单 critic 形态）；(3) 焦点测试（默认 off 位等价 +
beam 改变动作选择的行为证据）；(4) 重跑 Phase-1a：b2 s2 checkpoint
beam8 vs 自身 argmax 78（val50 双档）——可证伪预测不变（帮 b2 不帮
基线）。视图目录法保留（exp_local/cqn_rline/r4_phase1a_b2s2_beam8）。
在跑不受影响：cfaug s1/s2（GPU3，晚间出密封）。

### 用户提出的关键假设：move_plate 的分辨率不敏感性（2026-08-10 16:3x）

§66 实测（用户先前工作）：move_plate 对 post-ensemble L2 随机化近乎
免疫（n=4 定案小幅负效应），跨任务三个任务不复现（它们掉分）。
**假设**：move_plate 成功由 L0/L1 决定、细层是松弛量 → 价值排序的
改进（主要分布在细层/长地平线）在此任务上无从兑现 → 解释五臂平局墙
与"Spearman↑ 分不涨"。
**实验修正（预注册）**：(1) 束搜索实验加候选分歧的层级分布诊断
（分歧集中 L2 → null 归因"任务不出题"而非"判官不行"）；(2) 价值兑现
证明战场增开细层敏感任务（sandwich_remove，判官 checkpoint 现成）。
**战略分叉待用户裁决**：move_plate 夺分主力转数据/探索通道（cfaug/D2）
vs criterion-1 扩任务。建议双轨。

## 第二战场预注册：sandwich_remove（2026-08-10 ~16:5x，用户批准）

依据：§66 跨任务证据显示 sandwich 等任务消费细层精度（L2 随机化掉分），
与 move_plate 相反 → 价值排序改进若能兑现成任务分，应在此显形。
- 臂：**baseline**（canonical，当前 63 维截断制度）s1/s2 → GPU5；
  **tokensplit-b2**（boundary=2, aux=4；K=16 下 aux 比例 0.875）s1 →
  GPU4（s2 等 GPU3 今晚腾卡）。stamp `20260810swch`，
  `replay_size_before_train=540`（长 episode 必需）。
- 可证伪预测：b2 对 baseline 的同 seed 配对在 sandwich 上为正
  （move_plate 上是 −1.1 平局）——若成立，"排序兑现"的任务依赖性假设
  确立，criterion-1/2 的双轨战略定案。
- 协议同 move_plate：val50 双档、密封 200ep @100k、同批探针。
- 注意：旧 official_sandwich_remove（08-04/05）是旧制度工件，不作配对
  参照，仅其 s2 继续当 qselect3 判官。

### R4 Phase-1a 四格判决（2026-08-10 晚，勘误后定稿）：双 null

@100k val50，argmax vs beam8（400/450 档）：
- b2 s2：argmax **68/74**（n=100: 71.0）vs beam8 **70/70**（70.0）→ Δ≈−1，null
- 基线 s2：argmax 68/—（450 档 100k 行缺）vs beam8 70/68 → Δ≈+1，null
（勘误：初稿误把 b2 密封 78.5 当 val 参照写成"掉 8pp"，已废；50-ep
单点噪声 ±7pp，两 cell 均在噪声内。）
**判读**：双 null 正落在"move_plate 分辨率不敏感"假设的预测上——8 个
束候选的差异若集中在任务不消费的细层维度，重排必然无差。但也不能排除
"判官反事实排序不足"（F1：在轨 Spearman 0.80 ≠ 反事实排序，sibling
探针硬币）。两解释的判别实验都已预置：
1. 候选层级分歧诊断（候选间差异的 L0/L1/L2 分布）；
2. **sandwich_remove 上的同款四格**（细层敏感任务；b2-sandwich 60 维
   两种子在训）——若 sandwich 上 beam 为正而 move_plate null → 任务
   假设定案；仍 null → 判官问题，rerank 的第二次机会绑定 cfaug 的
   反事实数据通道（其密封今晚出，加测 sibling 反事实探针）。

## Wave-2c 预注册：b2-sandwich 时间尺度修正臂（2026-08-11 ~01:1x，用户批准）

设计失误承认：b2 原样移植 sandwich 时 aux=4 未做 tick 尺度换算——
move_plate（150 tick）上 aux=4 覆盖任务 ~3%，sandwich（~21 tick）上同
参数覆盖 ~19%，回传目标粗化到糊掉五分之一任务，中段配对两 seed 均
大负（−12~−26）与此吻合。
**修正臂 "swaux2"**：boundary=2 不变，`replay.auxiliary_nstep=2`
（覆盖 ~10%，最小多步窗），60 维合同，seeds 1/2 对旧基线 08-05 配对，
stamp 20260811swaux2。经 runner 透传口 CLI 覆盖，其余与 swch60 一字不差。
门与 sandwich 主预注册相同。GPU5 等位发射（评估清空后）。

## Wave-4 预注册：flip_cup 战场（2026-08-11 ~01:5x，用户裁定换防）

依据：§66 正式审计 flip_cup iid-L2 掉 −15.5/−12.0（均值 −13.75，CI
[−20.2,−7.3]）——真消费细层精度；基线 38-42% 头顶空间大（vs move_plate
76% 饱和带）；种子方差远小于 sandwich。swaux2 sandwich 修正臂撤销
（waiter 已拆），sandwich 原版判决仍走完程序作存档。
- **臂 1 "b2-flipcup"**：tokensplit boundary=2 aux=4（550 tick/episode，
  aux 覆盖 0.7%，无需尺度修正），60 维合同（append=false 透传），
  seeds 1/2 → GPU5 等位。配对参照：official_flip_cup 08-05 对
  **42.5/37.5 @100k 密封**（同合同已核）。
- **臂 2 "cfaug-flipcup"**（排队）：先在空闲评估位生成 flip_cup 反事实
  episode（generator --task flip_cup），数据齐后上卡。
- 门：同主预注册（密封 100k 配对 + 同批探针 + n=100 双档）。

### cfaug 机制判决（2026-08-11 02:0x，复合探针，同批配对）

online_failure 组（30 条注入反事实分支，held-in 制度注记：s1 训练见过
同分布数据）：**baseline predQ=0.621 vs cfaug predQ=0.151**——基线把
近流形歪支估得比 demo 态（0.087）还高 7 倍（F2 闭环致死的价值学显影）；
cfaug 正确贬低 4 倍。demo_success：Spearman 0.524→0.664，top1/gap 无损
（0.988/0.370）。**"失败数据 → 反事实贬值"因果链成立**（用户假设证实）。
行为（s1 密封 75.0，配对 −2.5）未兑现，与 move_plate 钝感叙事一致。
推进：cfaug-judge beam8 四格（零训练）已发 GPU2；cfaug-flipcup 数据
生成中（细敏 + 大头顶空间战场的合流点）。

### HANDOFF 勘误 #3 + sandwich 解释重写（2026-08-11 ~09:2x，用户质疑驱动）

实测：sandwich_remove `episode_length=13500`、down_sample 25 → **540
tick/集**（旁证 replay_size_before_train=540 恰为一集行数）。HANDOFF
"episode 522/25≈21 tick" 作废。由此我先前的"aux=4 覆盖 19% 糊目标"解释
**整体作废**（真实覆盖 0.7%，与 flip_cup 同级），swaux2 修正臂的立论
不成立（该臂已因用户换防撤销，未耗卡）。
**b2-sandwich −18/−20 的修正解释（统一定律）**：gap 税按任务精度敏感度
定价——L2 随机化审计给出敏感度排序 sandwich(−28) > flip_cup(−13.75) >
move_plate(−2)；b2 的 margin 变薄在 move_plate 免税（−1.1 平局）、在
sandwich 重税（−18/−20）。**flip_cup 预测改写（发射后、见数前）**：
b2-flipcup 预计偏负（中等税）；cfaug-flipcup 维持正预测（其 margin
几何无损 gap 0.370 + 反事实贬值已证）。若双向应验，统一定律确立，
cfaug 成为唯一双 criterion 候选。

### flip_cup 密封判决（2026-08-11 20:3x）+ 晋级发射记录

密封 200ep @100k（配对基线 42.5/37.5）：cfaug 53.0/38.5（+10.5/+1.0，
均值 +5.75，**过晋级门**）；b2 40.0/28.5（−2.5/−9.0，关闭，统一定律
应验）。晋级：cfaug-flipcup s3/s4（同 120 注入集）+ 基线 s3/s4 配对
（stamp 20260811flip4）+ cfaug 判官重排双档（rerank_fc_cfaug_s1）。
四种子声明门（预注册不变）：4 seed 配对均值 ≥+3 且 ≥3/4 为正。
待办探针：flip_cup 复合数据（demo + held-out 反事实集 121-267）上的
同批配对探针（cfaug vs 基线，含 held-out 泛化检验）。

### 重排线终审（2026-08-11 22:10）：全矩阵 null，正式关闭

cfaug 判官（已证反事实排序）× flip_cup（出题任务）：beam8 56/56 vs
argmax 58——最后一格也 null。重排矩阵完整战绩：{频率判官, 排序判官,
反事实判官} × {move_plate, sandwich, flip_cup} 无一格显著为正。
**结论**：数据通道训练出的价值知识已完整体现在策略 argmax 中，决策时
无剩余可收割；重排与双头（其可训练形态）一并盖棺。获胜机制唯一且
简洁：失败数据 → 反事实价值 → 更好的策略。

### 重排线二次关闭（2026-08-12 00:0x，这次有实测依据）+ cfaug 机制精化

flip_cup held-out 判官探针（分支 121-150，从未参训；同批配对）：
- 基线 critic：held-out 分支 predQ 0.228 vs demo 态 0.090（2.5× 高估）；
- cfaug critic：0.143 vs 0.125（校准到持平）→ **贬值泛化成立**；
- 但两者 demo 态在轨 Spearman 皆废（0.062 / −0.152）→ flip_cup 上不存在
  "会排序"的判官。
**重排关闭的实测理由**：重排需要排序能力，全线判官只习得贬值能力；
贬值型判官对近似候选只会投回贪心，结构上无增益。（首次关闭是推断，
用户驳回正确；本次以测量落锤。）
**cfaug 机制精化**：赢在"围栏"不在"择优"——策略被失败数据训练成
不入近流形陷阱，margin/模仿几何不动。与 +5.75 密封、重排无残余、
三探针互洽。

## 预注册（排队）："cfaug-o2o" 假设臂（2026-08-12，用户提出）

**用户假设**：cfaug 已在功能上接近 o2o（开局 buffer 含富分布非纯噪声、
后期连续漂成 online）；显式 o2o 化（同注入数据 + 10k 纯 buffer 预训练
+ 标准上线）应与隐式版打平。理论账：cfaug 拆掉墙一（反事实洞），
从而软化墙二（接缝不再触发全局重排）——专家-only o2o 的死因均被
数据完备性移除。
**臂**：cfaug-flipcup + num_pretrain_steps=10000，其余一字不改，2 seeds
对隐式版 cfaug（53.0/38.5）与基线双配对。
**预测**：|Δ| ≤ 3 打平（用户）；显著差 → primacy 主导；显著好 →
早期交互质量主导。机件零新增。排队等四种子终审后上卡。

### 锋利版探针判决（2026-08-12 晨）：机制 A 实锤，"围栏"修正为"填坡"

同 episode 相邻窗对比（成分受控，held-out 分支 121-150）：
基线 critic 跨扰动边界 Q 0.056→0.110（**+96%**，OOD 外推假上坡 =
F2 势能面成像）；cfaug 0.114→0.117（+3%，完全填平）。
机制表述定稿：**基线死于悬崖边的假上坡（外推乐观梯度诱导漂移放大）；
cfaug 用真实失败数据铲平该坡（非挖沟——held-out 上未达显著贬低，
但去除诱导即足以保护策略）**。同时确认抽样成分效应真实存在
（锚前段 0.056 vs demo 组 0.018），昨夜的混合平均数字（0.228/0.143）
按本节口径替换。

### 复权版机制判决（2026-08-12，用户指出折扣混淆后重算，为准）

逐样本以 γ^(t−锚点) 折算到锚点时刻（除净截断制度下 γ^(T−t) 的合法
时间涨幅，参考 γ^-24=1.27）：
- 基线：锚前 0.061 → 锚后 0.100（**+64% 净上跳**）——OOD 外推假上坡
  在复权后仍成立，机制 A 定案；
- cfaug：锚前 0.130 → 锚后 0.105（**−19% 净下跌**）——raw 口径的
  "持平"实为贬值被折扣上升掩盖；held-out 真贬值成立，前节"只填坡
  不挖沟"的降级表述撤回。
方法论记录：raw 段均值在截断制度下含折扣趋势混淆，凡跨时间窗的
Q 对比一律用本节复权口径。（n=48/格；prefix 窗末=锚点、post 窗长
≈24 的近似在案。）

## 预注册（排队）："cfaug-o2o-noBC"（2026-08-12，用户提出的 BC 退役假设）

**假设**：失败数据接管 BC 的第④份工作（off-manifold 保护，已证）与
①②的大部（真实负值 > 人造 floor），o2o 结构接管第③份（早期行为塑形）
→ BC-λ 可全撤。
**臂（第 2 阶梯，cfaug-o2o 过门后启动）**：cfaug 注入 + 10k 离线预训练
+ bc_lambda=0 + mc_lower_bound_target=true + unseen_return_floor（兜底
未覆盖兄弟 bin）+ 标准上线。对照 = cfaug-o2o（带 BC）与隐式 cfaug。
**风险在案**：截断制度下纯奖励通道的正向信号样本效率未验（no-BC 线
的 offline 提取证据来自旧制度）。一步一门，不跳级。
**叙事定位**：no-BC 线证明"无 BC 可打平"；失败数据补其反事实洞；
合体命题 = "无 BC 且更强"。

### 四种子终审（2026-08-12 10:34）：cfaug 未过声明门

密封 @100k 配对：+10.5/+1.0/−0.5/−6.0，均值 **+1.25**（2/4 正）——
门（≥+3 且 ≥3/4 非负）未达。新基线 s3/s4 = 51.0/49.5（旧对 42.5/37.5；
基线自身散布 13.5pp）——n=2 晋级的 +5.75 含基线弱种子运气，四种子
纪律第三次拦下超读（nstep3、分视野、cfaug）。
**盘点**：criterion 2 成立（复权机制链完整、不依赖行为分）；
criterion 1 四天未达成——全部行为效应居于种子噪声带。
**分叉呈报用户**：(1) 剂量/覆盖升级（全量 267 + 多维扰动 + 策略自身
失败采集）；(2) 组合臂 cfaug+MC 锚（补时间结构短板）；(3) 接受机制级
定位收官。建议 1+2 合成终局臂，仍噪则取 (3)。

## Wave-5 预注册：BC 退役三臂（2026-08-12，用户共同设计）

**立论**（逐种子探针的推论）：围栏机制四种子稳定（−20%）但 λ=1.0 时
BC 站岗、价值不掌舵——围栏的行为红利应在 BC 削弱时显形。历史对照全在
move_plate（venue 答辩在案：wean 测的是活/死（20-70pp 粗层效应），
L2 钝感只屏蔽 3-5pp 细层效应；rfloor 51.5/46.5、退火崩、178B 0% 皆
move_plate 数据）。三臂全部 = cfaug 注入 + canonical，唯一变量 λ 形态：
- **lowlam**：λ≡0.0125 从头（对照 rfloor 无围栏 51.5/46.5）——预测：
  围栏显著抬升 rfloor（≥+10）；若达基线带则围栏+微 λ 已足。
- **anneal-A**：λ linear(1.0→0.0125, 50k)（对照旧制度退火崩溃先例）
  ——预测：围栏使退火安全，终点 ≥ 基线带。
- **zero70**（用户提出）：step_linear(1.0,0.0125,50k,0.0,20k)——70k 后
  λ=0，**历史必崩点在围栏保护下的正面对撞**。预测：不崩（70k 后
  val 曲线不跳水）即为"围栏接管 BC 第④职能"的最强证据；崩则围栏
  不足以单独接岗。
门：密封 100k 配对基线 n=4（77.5/73.0/71.0/82.5）+ 各自历史对照；
机制探针照旧。stamp：20260812{lowlam,anneal,zero70}。

### Wave-5 预注册措辞更正（2026-08-12，用户问询触发）

lowlam 的对照 rfloor 并非"无围栏"：rfloor 自带人造 floor（一刀切
假设低值）。lowlam vs rfloor 的准确命题 = **同等微 λ 下，数据学来的
围栏（真实失败回报）对决手造 floor（假设值）**。预测不变（真数据胜），
归因表述以本节为准。

### Wave-5 先例补注（用户记忆核证）：anneal-A 的正面先例是 stage-174

174 = 皇冠-截断底座高 λ 训满 100k 后，续 50k λ 0.25→0.0125 → 密封
82.5/82.0（旧制度当时最高）。与新制度失败退火（训中 1.0→0.25，69.5）
的区别 = 衰减发生在能力成形之后 vs 之中（λ 三份工作账本的直接体现）。
anneal-A = 174 的从头压缩版（前 50k 降到 0.0125 ≈ 早期塑形期 λ 大半
在场）+ 围栏 + 新制度。若进基线带 = 174 结论在新制度复活。

## Wave-6 预注册：combined-flipcup（2026-08-12，用户裁定优先补洞）

**证据洞**：截断制度下 combined（探索×衰减）只在 move_plate 测过
（四种子 −2.1，饱和任务）；硬任务外部效度波全为 official 臂。探索是
仓史仅有的两个 +10 杠杆之一，从未在有头顶空间的任务（flip_cup 基线
~45%，n=4 在档 42.5/37.5/51.0/49.5）上做截断制度测试。
**臂**：runner `combined`（stage158 探索×衰减原方）× flip_cup × 60 维
（append=false）× replay_min 560，seeds 1/2，插队列头部。
**门**：密封 100k 对基线 n=4 同 seed 配对；探索若在头顶空间任务复活
（≥+5），分数主攻转探索通道；若仍平/负，探索杠杆在截断制度正式
全灭，分数线聚焦拓管数据。
队列全序：combined-fc ×2 → anneal-A s2 → zero70 ×2（Wave-5 完整性
保持，仅让位两个槽位）。
