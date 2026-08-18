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

### 优先级清算（2026-08-12 13:1x，用户质询"在跑的还有没有意义"）

裁定：lowlam 双 seed 处死（纯论文题、双变量归因不净、让卡）；
**anneal-A s1 保留**——它是 zero70 的直接对照（50k 前同形，70k 后分岔），
无它则 zero70 不可归因。GPU1 即时转 combined-flipcup ×2（分数线头号，
Wave-6 提前直发）。队列 v3：zero70 ×2 → anneal-A s2。
代码已推送私仓（6f617e5），用户将在另一台机器分担队列臂；
本机保留评估/探针链与恢复数据采集。

## 重大勘误 #4：反事实生成器的重放从未对齐过（2026-08-12 18:1x 发现）

**触发**：实现 ground-truth recovery（用户提议的"扰动后伺服回 demo 参考关节
再续放"）时，伺服诊断显示关节误差在恢复段反而发散到 0.6。逐层排查后发现
问题根本不在恢复，而在生成器的 demo 重放本身。

**Bug 链（两层）**：
1. 生成器重放的是 `ts.executed_action` —— 在 down-sample 后的 demo 里这
   不是逐 tick 的 delta 命令（实测其 dim1 有 94.8% 的步幅超出 env 单步动作
   边界，而真命令的幅度范围只有 [-0.045, 0.015]）。训练管线用的是
   `ts.info["demo_action"]`（action stats 也源于它，bigym.py:795）。
2. 即便换成 demo_action 也不能再过 `transform_to_tanh`：
   `post_collect_or_fetch_demos`（bigym.py:693）已经**原地**把它 rescale 成
   tanh 空间（指纹：gripper 命令恰为 -1.0）。再变换一次 = 双重变换。

**实测后果**：重放从第 0 步就偏（qpos 误差 0.11 → 6.0 单调增长，reset 时
逐维差恰为全 0，排除初始状态错位），重放 return 恒为 0。

**波及面（全部在坏重放下生成）**：
- 注入 cfaug 臂的 120 条 ±0.4 "围栏" episode（move_plate）；
- flip_cup 60d 反事实目录；probe_data_composite 的 30 条分支；
- v1 恢复采集的 ~822 条 ±0.15 episode。

**结论改写**：
- 这些数据不是"demo 旁一步之遥的 near-manifold 分支"，而是**从 reset 起
  就发散的 off-manifold 乱舞轨迹**。cfaug 的密封行为判决与探针测量本身
  仍是对该数据分布的真实测量（假上坡 +64% = 基线 critic 对远离流形轨迹
  的 OOD 乐观；cfaug 复权 −19~−23% = 注入该类失败数据后的真贬值），但
  §3 机制资产里"near-manifold/一格之差"的表述**作废**。
- v1 恢复结论（"开环重放极脆，±0.15 仅 2/356 回得来"）**作废**——那是
  坏重放的产物，不是开环脆性的测量。
- 真正的 near-manifold 围栏实验**从未跑过**，现在才第一次可做。

**修复与验证**：生成器改为逐字使用 rescale 后的 `demo_action`（不再任何
变换）。验证：reset 对齐全 0；重放 return 4/5 demo = 1.00；零扰动分支
（restore+续放）return = 1.00。恢复伺服 = demo 动作前馈 + tanh 仿射比例
反馈（raw 偏差 e → tanh 修正 2e/span，gripper 维不反馈——绝对量；env
单步跟踪增益实测仅 0.1-0.6，纯 offset 伺服追不上移动参考，前馈必需）。
关节误差收敛（0.05→0.04 底噪）。

**新铁律 10**：任何"重放/生成"管线在用于生产数据前，必须先过**零扰动
对照**（重放 return 要能复现 demo 成功）。围栏数据"失败是设计预期"恰好
掩护了坏重放整整两个 wave。

## Wave-7 预注册：正确重放后的第一批真 near-manifold 数据（2026-08-12 18:2x）

**背景**：勘误 #4 后，所有旧反事实数据判为 off-manifold。生成器修复 +
recovery 伺服（前馈+反馈）落地，smoke 实测：
- ±0.15×4 步：开环 9/12 成功；带恢复 11/12，jerr 全收敛。
- ±0.4（一个 L0 bin）paired：开环 5/8，恢复 4/8；同锚点翻转 3 例
  （1 例 0→1 被救回，2 例 1→0 被修正动作本身弄坏）。
- 生产批：move_plate paired ±0.4×4 步 + 6 步恢复，51 demo × 6 锚 × 2 
  变体（`exp_local/cqn_rline/counterfactual_v2_paired_moveplate/` 带
  manifest.jsonl，pid 1841418）。

**臂 A "fence-v2 / 邻域真值"**：注入 120 条开环分支（成败都要，带真实
return）。这是用户"高斯邻域价值监督"的数据实测版——物理容忍的扰动
（~60%）带 return 1 进 replay，不容忍的带 0，critic 学到的邻域形状由
仿真而非人为核函数决定。与 cfaug-v1 唯一差异 = 数据分布（真 near-manifold
vs 从 reset 发散的乱舞），故与 cfaug-v1 四种子直接同门对比可归因数据质量。

**臂 B "recovery-teach"**：只注入同锚点 outcome 相反的配对（开环失败 +
恢复成功），教 critic "扰动态下伺服回参考的动作序列有值，硬续走没有"。
剂量按生产批实际翻转率定（smoke 翻转率 ~12%，预计需全部 306 锚点凑
~30-40 对；不足则加锚点补采）。

**门（不变）**：密封 200ep @100k 整，seeds 800-999，四种子配对 vs 基线
76.0（n=4），声明门 = 配对均值 ≥ +3pp 且 ≥3/4 非负。机制门 = v2 held-out
分支复权跨界贬值 + 兄弟 margin 无损。

**在途臂的解释注记**：Wave-5（zero70/anneal）与 Wave-6（combined-fc）
底座是 v1 fence 数据，判决照常执行，但其 cfaug 成分自此记为 "fence-v1
（off-manifold）"。

## Wave-7 生产批与 check 阴性记录（2026-08-12 19:4x）

v2b 批（带 object-deviation manifest）612 条完成,分类分布与 v2 逐对一致
（both_s 209 / both_f 43 / harm 21 / teach 33）——生成器确定性确认。

**MILES 式场景扰动 check = 阴性**：rejoin 时的 max|Δqpos_full| 不能预测
recovery 是否弄坏任务（harm 中位 0.026 vs 正常 0.023;最优阈值只抓 5/21）。
反直觉信号:开环**成功**分支的 rejoin 偏差(0.134)高于失败分支(0.074)——
max 范数被自由空间无关自由度支配,致命的是接触段 gripper-物体相对位姿的
小偏差。改进方向(未实施):按物体 qpos 切片、按接触相位(gripper 闭合)分段
判定。此 check 不作为 Wave-7 数据门;仅存 manifest 供后续分析。

**勘误 #4 补充证据（2026-08-13,应用户质疑核验）**:v1 分支锚前段到
demo 流形的最近距离（63 维 low_dim,max-abs,同时刻 ±3 窗）中位 7.1-8.0;
v2b 锚前段中位恰为 0.000（逐位重合）。旧探针"跨界贬值"实际跨的是
"浅度跑偏→深度跑偏"界（重放偏差单调增长 0.11→6.9),非"demo→扰动"界。
"锚前 Q≈demo Q"（cfaug 0.114 vs 0.115;baseline 分支 0.110 > demo 0.018）
的正确读法:critic 状态层面分不出浅垃圾与真 demo——即假上坡病理本身。

## Wave-5 zero70 中段观察：λ→0 极限不连续（2026-08-13 凌晨，密封待确认）

val50 中段曲线（两种子对称）:65k（λ≈0.003）s1 68%/s2 64% →
70k（λ=0 时刻）双双 18% → 75k/80k 全部 0%。

**读法**:BC 衰到名义功率 0.3% 仍锚得住 demo 邻域 argmax 排序（hinge
只要违反就全力推回,与 λ 数值几乎无关）;精确零 → 5k 步内相变崩塌。
v1 围栏（远处）在近处支撑力为零 → "缓慢退火到 0"路线在 v1 数据下出局,
不是速度问题,是极限相变。BC 完全退役的必要条件 = 有东西在 demo 邻域
接管排序 → 正是 Wave-7 near-manifold 数据的立论。
anneal s2 中段 50k=80/60k=74/70k=82（λ=0.0125 巡航于基线带上方）——
"退到 0.0125"与"退到 0"的命运分野即本 wave 的核心产出。

## Wave-5 zero70 关臂 + Wave-7 上卡（2026-08-13 ~01:4x）

- **zero70 双种子经用户裁定于 ~90k 处杀**（中段已定案:75k-80k val50
  全 0,两种子对称崩塌;100k 密封无意义）。臂判决:**λ→0 相变崩塌,
  v1 围栏不能接管近处排序**——与 178B 裸退同命,但本臂给出了精确的
  崩塌定位（70k 落零时刻 18%,5k 步内归零）与"0.003 的 λ 仍有效"
  这一不连续性证据。
- **Wave-7 臂 A（fence-v2）发射**:s1/s2 于 GPU0/GPU2（stamp
  20260813fv2,120 条 v2 开环分支预填,λ=1 与 cfaug-v1 同合同）;
  s3 由 queue v6 上 GPU4;s4 + recovery-teach s1/s2（stamp
  20260813rt）在 v6 队列。
- 发射事故记录:第一次 fv2 发射把 hydra.run.dir 当 passthrough 传,
  被 arm 脚本自身的同名覆盖压掉（Hydra 后者胜）——杀掉重发;期间又踩
  一次 pgrep 自匹配(exit 144,铁律 4)。教训:arm 脚本的 RUN_DIR 不可
  从外部改道,区分臂用 stamp。

## Wave-6 判决：combined-flipcup 关（2026-08-13 03:0x 密封齐）

- s1 密封 41.5 vs 基线 s1 42.5 → −1.0；s2 密封 35.5 vs 基线 s2 37.5
  → −2.0。臂均值 **−1.5**,声明门 ≥+5 未过,两种子皆非正。
- 中段曲线亦平（s2 45k=34/65k=36/85k=32,始终贴基线之下）。
- 结论:探索×衰减在截断制度下的红利为零,**跨任务成立**
  （move_plate 四种子 −2.1 + flip_cup 两种子 −1.5)。探索通道正式出局;
  分数主攻全部转数据通道（Wave-7）。

## Wave-5 判决更新（2026-08-13 04:0x anneal s2 密封）

- **zero70**:关(λ→0 相变崩塌,见上节;用户裁定 90k 杀,中段定案)。
- **anneal-A**:s1 69.5(−8.0) / **s2 85.0(+12.0)**。均值 +2.0,n=2
  高方差不可声明。**s2 的 85.0 为全线最高单种子密封分**(基线最高 82.5),
  且在 λ=0.0125 末态取得——"BC 退到 1.25% 功率"不仅安全,单种子可超基线。
  按 50-ep 双档纪律的密封版对应:臂扩 s3/s4 后走 4 种子声明门。
- Wave-5 核心产出定稿:**λ 可退至 0.0125(免费甚至有赚),不可至 0
  (相变)**;v1 围栏在近处支撑力为零。

## Wave-7 中段矩阵（2026-08-13 上午,val50 seeds400,判决以密封为准）

| 种子 | 50k | 70k | 90k |
|---|---|---|---|
| fence-v2 s1 | 78 | 82 | 76 |
| fence-v2 s2 | 76 | 66 | 72 |
| fence-v2 s3 | 70 | 76 | 72 |
| fence-v2 s4 | 66 | 78 | 80 |

recovery-teach:s1 40k=80/60k=84/75k=76;s2 30k=76/45k=64/60k=66。
分布特征:两臂全部点位于 64-84,围绕基线带(76)浮动,无一种子系统性
落带下——与历史各臂中段即显著落后的模式不同。rt s1 峰值 84 系
66 条教学对注入所得。

## Wave-7 臂 A 判决:fence-v2 行为平局(2026-08-13 下午密封齐)

| 种子 | 密封@100k | 基线 | 配对 |
|---|---|---|---|
| s1 | 77.0 | 77.5 | −0.5 |
| s2 | 81.0 | 73.0 | **+8.0** |
| s3 | 70.0 | 71.0 | −1.0 |
| s4 | 75.5 | 82.5 | −7.0 |

均值 **−0.125**,1/4 非负,声明门未过。**行为层与 fence-v1(cfaug +0.5
n=2)同判:λ=1 下围栏数据不动行为**——机制-行为解耦定律在 near-manifold
数据上复现(BC 站岗,价值不掌舵,数据质量差异被 λ 掩盖)。
fence-v2 的真正检验移交两处:(a) 机制探针——near-manifold 围栏是否
修复了决策边界上的假上坡(v2b held-out 分支复权测,fv2 vs cfaug-v1 vs
baseline 三方);(b) **anneal × fence-v2 组合臂**——λ 退到 0.0125 后
近区围栏是否兑现为行为(四门全过的第一个实验,见"探索×衰减为何无效"
四门分析)。组合臂待用户批准后上卡。

## Wave-7 臂 B 判决 + 波次总结（2026-08-13 午后密封齐）

recovery-teach:s1 82.5(+5.0) / s2 61.0(−12.0),均值 −3.5,n=2 方差
巨大,声明门未过。

**Wave-7 波次总结**:fence-v2 均值 −0.125(n=4)、recovery-teach 均值
−3.5(n=2)——修复后的 near-manifold 数据在 **λ=1 行为层同样无信号**,
与 v1 时代 cfaug(+0.5)完全同构。机制-行为解耦定律第三次复现:
**只要 BC 满功率站岗,注入数据的质量差异就不进分数**。
波次遗产移交:(a) 机制探针(fv2 critic 在 v2b held-out 分支上的
决策边界排序——criterion 2 的干净测量,待跑);(b) **组合臂
anneal×fence-v2**(四门全过的第一个实验,待用户批准);(c) 33 教学对
的 rt s1 +5.0 为孤证,组合臂若立,教学对可做第二剂量。

## 探索剂量-反应实验（2026-08-13 午后,用户设计）

协议:baseline s1 @100k checkpoint,每档 50 eps(seeds 400-449),
eval 时以不同探索配置在线运行。信息量代理 = 摧毁的成功率。

| 档 | explored 步占比 | 成功率 | Δ |
|---|---|---|---|
| 贪心 | 0 | 68% | — |
| 高斯 σ=0.01 | 0 | 68% | 0 |
| 现状 [.002,.004,.008] | 17.2% | 58% | −10 |
| ×4 [.008,.016,.032] | 46.7% | 58% | −10 |
| 仅粗层 [.03,0,0] | 32.6% | 32% | −36 |

结论:(1) 高斯噪声零信息(严格);(2) 现状 bin 探索有 ~10 分的真实
信息流(此前"基本无信息"的说法下修);(3) **信息只随粗层事件数走,
与总事件率无关**(×4 档 2.7 倍事件零增伤);(4) 粗层直采 = −36 分
在线代价,围栏注入 = 同类证据的零代价离线获取。校准:greedy 68 ≈
官方 val50@100k 的 70。harness 教训:train 模式批量 act 必须
num_train_envs=batch(per-env 探索状态数组按它分配)。

## Wave-8 预注册：λ→0 跨任务崩塌形态测试（2026-08-13 下午,用户设计）

**立论**（平局理论,今日与用户共同推导）:BC 的第一级工作 = 在近平局
真值上提供破平局滞回;探索/数据治不了第一级(178B:8× 探索强度 λ=0
照样 0/200),但**平局程度是任务性质**(L2 审计:move_plate −2 /
flip_cup −13.75 / sandwich −28)。

**可证伪预测**:λ→0 的崩塌形态应随平局度质变——move_plate(最平)
= 5k 步内吸收态归零(已证);flip_cup/sandwich(真 gap 大)= 衰减更慢
/非零地板/span 指纹保留。若 sandwich 上照样瞬崩 0/200 → 平局理论
证伪,管外假上坡(第二级)全任务主导。

**臂**(探索全开 [0.002,0.004,0.008],无注入):
- **weanfc**:flip_cup combined @101k checkpoint warm-start 续训
  (优化器重置、online buffer 重启——与真 resume 的差异记录在案),
  schedule step_linear(1.0,0.0125,131000,0.0,10000):~101k λ0.24 →
  131k 0.0125 → 141k 0 → 161k 观察。×2 种子,GPU0 双跑。
- **weansw**:sandwich_remove 从头(无 combined 前身,查实 swflip 为
  λ=1 无探索),step_linear(1.0,0.0125,50000,0.0,20000),100k。
  ×2 种子,GPU2 双跑。
判读:val50 曲线过 λ=0 点的形态 + 崩塌指纹探针(span/agreement),
不设行为声明门(诊断性 wave)。

**Wave-8 修订(2026-08-13 晚)**:weanfc 续训臂报废——finalize 的存储
清理删除了 resume sidecar 与 replay 目录,params-only warm-start(优化器
重置 + buffer 重启)在 4-9k 步内把底座从 41.5/35.5 砸到 6%/2%
(λ 尚在 0.17-0.21,与 λ 机制无关;A13 深跌先例的重演)。用户裁定从头
训:weanfc2 ×2(flip_cup,探索开,step_linear(1.0,0.0125,50000,0.0,
20000),100k,GPU0 双跑),与 weansw(sandwich)完全同协议;move_plate
参照 = wean0(探索底座 λ→0 瞬崩)。**教训:finalized run 是评估资产,
不可续训;可能续训的 run 在 finalize 前要说**。

## Wave-8 判读 #1:平局理论跨任务预测证伪(2026-08-14 凌晨)

sandwich weansw s2 衰减腿定位:50k(λ.0125)=50 / 55k=46 / 60k=50 /
**65k(λ.003)=66(新高)** / **70k(λ=0)=0** / 75k=0;s1 @70k 亦 0。
与 move_plate zero70 逐点同构(65k 活跃→落零 5k 内湮灭),且更彻底
(70k 即 0 vs 18)。**λ→0 相变与任务 gap 结构无关**
(sandwich 敏感度 −28 vs move_plate −2)——"真 gap 大的任务 honest
critic 能接管"预测死亡(用户质疑驱动的检验)。

**机制改判(幻觉角落理论)**:argmax 在 5^45 组合空间取 max,函数逼近
对未见组合的乐观被 max 专门选中——λ/floor 的真实工作是**全域压制
unseen 组合空间**,不是 demo-sibling 破平局。解释:任务无关性(逼近器
之病)、探索无效(组合空间打不完地鼠)、178A floor 有效(全域压制)、
统一 λ 阈值。

**复活探针(预注册,eval-only)**:崩塌 checkpoint(75k-80k,行为 0%)
换受限解码(数据支撑候选内选择,qselect3)。若成功率显著回升 →
critic 的 on-manifold 知识未死,死的是解码;no-BC 终点改写为
"修 argmax 而非修 value"(候选集/proposal 路线,IDQL/AQL 族)。

## Wave-8 判读 #2:复活探针 v1 完整表(2026-08-14)

qselect3 harness(锁步,harness 内自对照;judge = sandwich s2 @75k
崩塌 critic,自由 argmax 0%):
q(c65+c50)=52 / random=58 / c65 solo=54 / c50 solo=50——四方案同噪声带。
**判决:限域贡献 100%(0→52-58),判官排序贡献 0(q≈random≈solo)**。
"on-manifold 知识存活"论断降级为未证(纯随机判官也会给出同样的表,
用户指正);判别力终审移交 v2 探针(候选加入垃圾 5k + 自劫持 75k 提议:
q 守住 c65→判别力活;q≈random→死;q<random→假上坡在决策回路现形)。

## 本会话补档(2026-08-14,审计驱动——以下数字此前只在对话未入 log)

- Wave-8 崩塌曲线全量:weanfc2(flip,从头)s1/s2 @50k=22/16,@70k-90k
  全 0(双种子);weansw(sandwich)s1 50k=54,70k-100k 全 0;s2 全曲线
  50:50/55:46/60:50/65:66/70-100k 全 0。flip 双 run 于 ~90k 用户裁定杀。
- 复活探针 v2(判别力):judge=sandwich s2 @75k,候选{健康65k,垃圾5k,
  自劫持75k}:q 0%,picks 自劫持 67.6%/垃圾 28.7%/健康 **3.7%**
  (chance 33%,9 倍回避);random 三选一 0%。**偏好与质量反序**。
- swirl03 MC 双臂密封(跨机未校准注脚):mclb 50.5/62.5(101k 52.5/67.5)
  配对均值 −18.75;mcanl 64.5/55.5(101k 60.5/58.0)−15.25。
- **勘误(审计抓获)**:我此前引用的崩塌指纹数字(span 1.1→0.0104,
  agreement→0.361,Q=1.104)出自**旧制度 178B/wean0 的探针**(cqn-flow),
  被我错误泛化为 Wave-8 新崩塌的实测——Wave-8 崩塌 checkpoint 的指纹
  探针**尚未跑过**。"λ=0 必崩"表述过绝对:178A(floor+λ=0)不崩——
  相变变量是"是否存在任一 unseen-action suppressor",非 λ 本身。
- 审计全文:reports/explore_decay_analysis_20260814.md(含文献对位表:
  现象族大体已知 TD3+BC/BCQ/tandem/churn;新增量=0.003 阈值极端性、
  5k 步崩塌时间切片、及偏好反转 3.7%(文献无直接对应))。

## Wave-8 判读 #3:角落成像 + 真实指纹(2026-08-14,审计指定的测量)

**真实指纹**(Wave-8 runs 的 train.csv,替换此前误引的旧制度数字):
sandwich 落零前(69k,λ≈6e-4) agreement 0.77-0.81 / span 0.32-0.35 /
violation 0.57-0.60 → 落零后 1k 步(71k) agreement 0.39-0.46 /
**span 0.004(塌 70 倍)** / violation 1.000 → 深处(85k+) agreement
0.24-0.28, span 0.0007。flip 的 span 在落零前(69k)已蚀至 0.10
(sandwich 同期 0.32)——趋零走廊侵蚀速度任务相关,悬案线索。

**角落成像**(demo 状态上,96 样本,65k 健康 vs 75k 崩塌):
| | 65k(66%) | 75k(0%) |
|---|---|---|
| Q(demo 动作) | 0.174 | 0.239(**升 37%**) |
| Q(自由 argmax) | 0.176 | 0.242 |
| argmax−demo Q 差 | +0.001 | +0.002(**无幻觉尖峰**) |
| argmax bins=demo | 72.8% | **31.5%**(chance 20%) |

**机制第三版(测量驱动)**:崩塌 = 已知状态上的"**海平面上升 + 山峰
融化**"(全体 Q 微涨、gap 塌平)——argmax 不是被某个高分幻觉角落拽走
的(demo 状态上无尖峰),而是在平局之海上失锚漂流(69% 因子决策离开
demo bin);漂进 off-manifold 后遇到 +64% 假上坡(旧实测)才被定向
拉远。平局理论的**机制部分复活**(平局是崩塌动力学的产物,普适;
其跨任务预测死于"融化由训练动力学制造,与环境真 gap 无关")。
"幻觉角落"修正为:家门口无角落,角落是漂出去以后的坡。

## Wave-9 FS-CQN 判决:预注册 kill 触发,臂关(2026-08-15 晨)

11 seeds(03×6+04×5),100k,dev100(seeds 400,100ep/点):
均值曲线 20k **54.7** → 40k 34.8 → 60k 12.6 → 80k 5.4 → 100k **3.1**;
11/11 单调下跌。斜率主指标强负,kill 判据触发,臂关。

**部分成功(入资产)**:λ=0 无 hinge 下 11/11 无相变崩塌(全历史首批
from-scratch 存活);10k 预训练+mask 解码在 20k 点给出 54.7%。

**尸检(意外)**:侵蚀期间 buffer 状态上的指纹**完好无损**——agreement
0.96-0.99、sibling span 0.49-0.51 全程平稳(iter 10k-89k)。**排除**:
(a) in-mask 融化;(b) buffer 状态排序退化。存留嫌疑:**frozen mask head
+ critic 在 rollout 分布上的联合 OOD 失效**(dev 评估时的状态分布 ≠
buffer 分布;head 冻结在 10k 的 demo 分布上,判官风险 #2 兑现)。
待做判别测量:记录 eval rollout 实际状态,对比 20k vs 100k critic 在
同一状态集上的 masked-argmax 与 mask 宽度。

学费定理更新:**限制解码域推迟(5k→~80k)但不阻止死亡;buffer 指标
看不见 rollout 侧的病**——评估口径必须含 rollout-state 探针。

## Wave-10 预注册:双方案并行,各 2 seeds(2026-08-15,用户裁定)

判别测量定案:同一 100k 崩塌 critic,戴面具 5% / 摘面具 **36%**(20k:
53/56)——**frozen decode mask 在漂移状态上是主动毒药;masked target
保护了价值面**(critic 免疫角落,摘面具不归零)。

- **臂 A "FS-CQN-TM"**(target-mask only):face fscqn 预设 +
  `support_mask_decode=false`——TD target 仍戴面具(head 只在 replay
  分布内被查询,可靠),解码放开。×2 seeds。
- **臂 B "floor-wean"**(纯 critic-only,单网络):官方合同 +
  unseen_return_floor(w=0.1,value 0,mean)+ zero70 schedule
  (step_linear(1.0,0.0125,50k,0.0,20k))。= 178A 的从头+截断制度版,
  λ 落零后 floor 接管。×2 seeds。
判据:dev100 斜率(20k-100k 五点);2-seed 纪律(memory:
seed-budget-policy),有苗头再扩 4。均在 swirl03。

**Wave-10 修订(用户两次质询驱动,最终阵容)**:
- 臂 B 三易其稿:floor+wean schedule(未适配截断,q_reward_scale 被
  dense 护栏拒)→ dense+schedule(hinge 塑形)→ **purefloor(定稿)**:
  fscqn 预设 − 面具 + q_reward_scale=2,即 10k 预训练塑形 + 全程 λ≡0
  + dense(自带 floor/MC)+ 截断适配 scale。= 178A 从头+截断版,
  且与臂 A(fstm)构成 target-面具单变量消融对(modulo scale)。
- 用户修正记录:(1) floor 的截断适配(scale=2,余量翻倍;相对 floor
  = A13 尸体不碰);(2) 178A 本无 λ——塑形应由预训练承担而非 hinge
  schedule(FS-CQN 20k=54.7% 为从头可行性证据)。

## Wave-10 阶段报告与执行（2026-08-15）

### 1. Previous-stage result

Wave-9 FS-CQN 的当前保留原始 artifact 是
`exp_local/cqn_trunc_arms/fscqn_move_plate/seed{7..11}_20260814fs/dev100.csv`。
五个 seed 的 dev100 曲线分别为：

| seed | 20k | 40k | 60k | 80k | 100k | validation-selected best |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 56 | 30 | 14 | 3 | 1 | 56 @ 20k |
| 8 | 44 | 28 | 1 | 0 | 1 | 44 @ 20k |
| 9 | 59 | 46 | 16 | 5 | 4 | 59 @ 20k |
| 10 | 53 | 29 | 16 | 6 | 5 | 53 @ 20k |
| 11 | 60 | 38 | 11 | 2 | 1 | 60 @ 20k |
| mean | **54.4** | **34.2** | **11.6** | **3.2** | **2.4** | **54.4 @ 20k** |

所以 best-checkpoint 与 endpoint 的公平比较是 54.4% @20k 对 2.4%
@100k，而不是把 2.4% 当成该方法的最佳能力。此前完整 11-seed 账本的
对应均值是 54.7→34.8→12.6→5.4→3.1，结论一致。固定同一 100k critic
的解码尸检为 masked 5%、unmasked 36%（20k 为 53%/56%）。本阶段未读取
seeds 800--999 held-out。

### 2. Interpretation

FS-CQN 证明 target support 能避免传统 λ=0 的 5k-step 瞬时相变，却没有
带来正 scaling；冻结在 demo 分布上的 mask 一旦被 rollout 分布反复查询，
从保护变成了执行期 OOD 毒药。摘掉 decode mask 后 100k 从 5% 回到 36%，
同时仍远低于 20k 的 53%，所以 decode mask 是一部分原因，但不是全部原因。
仍未解决的是：没有 action-label 约束时，dense floor/TD 是否能在 online
分布上维持正确相对排序并继续提高，而不只是保存 demo 附近能力。

### 3. Next-stage decision

把两个问题放在独立实验中：

- A / `fstm`：以 FS-CQN 为 matched baseline，只改
  `support_mask_decode=false`；target argmax 继续使用冻结 mask。training seeds
  7/8，直接逐 seed 配对 Wave-9 同 seed。
- B / `purefloor`：10k demo-only reward-Q 预训练、λ=0、self-imitation off、
  无 policy/mask head、positive-only dense return target + MC lower bound，且
  `q_reward_scale=2`。它回答修正 demo-MDP/value scale 后纯 critic floor 能否
  scaling，不与 A 混成一个因果比较。

两臂预算均为 101k；selection split 是 seeds 400--499，每个 20/40/60/80/100k
checkpoint 100 episodes；held-out split seeds 800--999 在二种子 gate 前保持
封存。主指标为五点 OLS success slope、100k−20k endpoint delta、100k
seed-min；报告 validation-selected best（平局取更早 checkpoint）。只有同时
满足 mean slope > 0、mean(100k) >= mean(20k)、两个 seed 的 100k 均 >=40%，
才扩到四 seed。mean slope <=0 且 endpoint 下降超过 10pp 直接 fail；其余为
inconclusive，不开 held-out。四 seed 通过后才用 validation-selected best
checkpoint 做 200-episode held-out，并与同样 validation-selected 的 matched
baseline 比较。

### 4. Execution

已加入 `fscqn_target_mask_only_pixel_bigym.yaml`、
`cqn_as_pixel_bigym_purefloor_gate.yaml`、单 seed runner
`scripts/run_cqn_wave10_arm.sh` 和双卡调度器
`scripts/launch_cqn_wave10.sh`。运行布局预注册为 GPU2/EGL2 seed7 与
GPU3/EGL3 seed8；每卡先 A、120 秒后 B，`xla_mem_fraction=0.45`，并显式
使用与 matched artifacts 一致的 `ROBOBASE_HOST_MERGE=1`。runner 只产生
`dev100.csv`，不包含 held-out 评估。

实际发射 stamp 为 `20260815wave10`。首次 `nohup` 尝试只写了 PID 文件便被
外层执行环境回收，GPU 仍为 3 MiB、无 Hydra artifact，因此未计作启动；随后
用独立 tmux session 在 09:32:42 BST 重发。已验证四个真实 trainer：

- GPU2 UUID `GPU-80b9cc0d-...`：fstm s7 PID 2844845，purefloor s7
  PID 2849606；
- GPU3 UUID `GPU-03f1431f-...`：fstm s8 PID 2844846，purefloor s8
  PID 2849605。

两张卡的第二进程均精确晚 120 秒。09:37--09:40 的 artifact 证明不是空壳
进程：四份 `.hydra/config.yaml` 与 `pretrain.csv` 均已落盘；fstm 两 seed
各写 19 行有限 update（critic loss 0.794/0.788，support CE weight=1），
purefloor 各写 7 行有限 update（critic loss=dense loss 0.404/0.402，
`q_reward_scale=2`，无 support-CE 列），demo buffer 均为 8061，所有记录的
finite guard 为 1。GPU2/GPU3 实测为 28.2/26.1 GiB、97/98% utilization，
四个 PID 到各自 GPU UUID 的映射由 compute-app 表确认，无 failed/nonfinite
artifact。

执行链审计发现单臂 runner 若先结束会与同卡尾部 trainer 共置 eval。训练
进程未暂停；仅把四个等待中的 wrapper shell 置为 STOP，并启动
`scripts/guard_cqn_wave10_eval.sh`（GPU2/GPU3 guard PID 2858551/2858558）。
守卫等同卡两 trainer 都退出后才恢复 fstm eval，20 秒后恢复 purefloor
eval，因此保持“training/eval 不共卡”和 eval 构建错峰合同。按双跑历史
约 7 env-step/s 粗估，training ETA 13:40--14:10 BST，五点 dev100 完成 ETA
14:05--14:40；以首个 5k/20k checkpoint 的实测速率为准。

## Wave-10 完成判决（2026-08-15）

### 1. Previous-stage result

四个训练都自然完成到 101k，四份 `dev100.csv` 都在 seeds `400--499`
完成了预注册的 20/40/60/80/100k 各 100 episode；没有读取 held-out
seeds `800--999`。validation-selected best（同分取更早）与固定 endpoint
如下：

| arm / seed | 20k | 40k | 60k | 80k | 100k | selected best |
|---|---:|---:|---:|---:|---:|---:|
| FSTM / 7 | 37 | 26 | 22 | 14 | 21 | 37 @ 20k |
| FSTM / 8 | 42 | 38 | 23 | 18 | 15 | 42 @ 20k |
| **FSTM mean** | **39.5** | **32.0** | **22.5** | **16.0** | **18.0** | **39.5 @ 20k** |
| purefloor / 7 | 41 | 27 | 25 | 23 | 24 | 41 @ 20k |
| purefloor / 8 | 46 | 16 | 13 | 11 | 13 | 46 @ 20k |
| **purefloor mean** | **43.5** | **21.5** | **19.0** | **17.0** | **18.5** | **43.5 @ 20k** |

FSTM 两 seed 的 OLS slope 是每 20k `-4.4/-7.4pp`，100k−20k 是
`-16/-27pp`；purefloor 是 `-3.8/-7.1pp` 与 `-17/-33pp`。两臂 mean
slope 都为负、mean endpoint 分别下降 `21.5/25.0pp`、seed-min endpoint
仅 `15/13%`，所以都触发预注册 fail，不扩 seed、不打开 held-out。

与同 seed 的 Wave-9 masked-decode FS-CQN 配对，FSTM 的 100k 从
`1/1%` 提到 `21/15%`（均值 `+17pp`），但 20k 从 `56/44%` 变为
`37/42%`（均值 `-10.5pp`）。摘 decode mask 避免了接近归零，却没有得到
正 scaling。

训练日志、finite guard、artifact finalizer 与 completion markers 均正常；
四 run 保存了 21 个 params-only eval checkpoints 并按生命周期合同清除了
在线 replay/resume 文件。原始路径是：

```text
exp_local/cqn_wave10/fstm_move_plate/seed{7,8}_20260815wave10
exp_local/cqn_wave10/purefloor_move_plate/seed{7,8}_20260815wave10
```

### 2. Interpretation

结果排除“只要把 frozen mask 从 rollout decode 摘掉就能恢复 scaling”：它
解释了 Wave-9 的末端归零，但不能解释剩余的持续下降。target-only mask 与
无 mask dense floor 的 100k 均值几乎相同（18.0% vs 18.5%），两者都没有
解决 action-facing value learning。

这也不是 replay batch 上可见的 margin 融化。20k→100k 期间，FSTM 的
demo-action agreement 约 `0.96--0.98`、sibling span 约 `0.47--0.53`；
purefloor 的 agreement 约 `0.97--0.99`、span 约 `0.96--1.02`，但 dev
success 仍显著下降。mini-batch positive fraction 同期从约 `0.78--0.83`
降到 `0.65--0.68`，说明 rollout 正在收集更差的状态/结果；这些量是诊断，
不是 policy quality。共同未解因素是 success-only 10k pretrain、factorized
action extraction，以及缺少 online sibling outcome。因为两臂都共享 10k
pretrain，本结果支持取消它，却不能单独把下降因果归给它。

### 3. Next-stage decision

停止 Wave-10 两臂。下一阶段用 DJCQN 单独检验 joint-chunk value 与 online
counterfactual acquisition，不再做 success-only offline TD：

- `num_pretrain_steps=0`；demo transitions 以固定 128-sample protected
  half-batch 持续进入每次 online update，另一半来自随 rollout 演化的 replay。
- matched arms 是完全相同的 DJCQN greedy 与 finest-level adjacent sibling
  exploration（每步 2%，约 2,000 次/100k）；training seeds `1,2,3`。
- selection split 为 seeds `400--449`，50 episode/checkpoint，报告
  10/25/50/100k、逐 seed validation-best、trapezoid success-AUC 与 100k
  endpoint；seeds `800--999` 继续封存。
- exploration gate 要求 paired mean AUC 至少高 greedy `5pp`、至少 2/3 seed
  的 paired AUC 为正、explore 的 mean 100k 不低于 greedy 超过 `2pp`，且
  explore 自身 10k→100k mean slope 为正。否则不打开 held-out，也不把 loss
  下降解释为 task improvement。

### 4. Execution

`djcqn_pixel_bigym.yaml` 已改为零 offline、100k 纯 online 计步、固定
demo/online `128+128` batch 与 5k params-only checkpoints。单臂 runner
`scripts/run_djcqn_wave1_arm.sh` 和六卡 controller
`scripts/launch_djcqn_wave1_six_gpu.sh` 固化了上述 split、剂量和 artifact
合同；远端启动与首个 artifact 验证记录在 `cqn-viability.md`。

## 容忍度三点谱(2026-08-15,导师论证补强,纯测量未注入)

一格(L0,±0.4×4 步)开环容忍度:move_plate **75%**(306 对) /
flip_cup **80%**(120) / sandwich **73%**(103)。
**跨任务平坦(73-80%)**,与 L2 细尺度敏感度(−2/−13.75/−28,差 14 倍)
完全解耦——粗尺度决策 gap 趋零为全套件普适性质,"gap 低于噪声地板"
论证由单任务升为三任务;细/粗两轴独立获双向证据(剂量实验:细层事件
move_plate 隐形;本谱:粗层偏离 sandwich 亦宽容)。
事故记录:sandwich 首发要 60 demo 但缓存仅 36,9 秒崩;flip_cup 两分钟
跑完 120 集经 npz 逐条核真(空载 EGL ~330 步/秒)。pgrep 自匹配第三次
——进程判定一律 /proc/cmdline 精确匹配入铁律。

## 勘误 #5:剂量测量的 eval 槽位污染 + 三任务修正终表(2026-08-17)

### Bug
`scripts/eval_explore_dose.py`(手写 rollout 循环)只重置了 agent 的 train
act-state 槽(`agent.reset(step, range(B))`,在 `cfg.num_train_envs=25` 下全部
路由到 train 槽);eval_mode acting 使用的 `_eval_open_loop_plan/position/valid`
(位于索引 `num_train_envs+i`)从未重置。第 2 轮起每集开头执行上一轮陈旧
chunk 的尾段(最多 15 步)。greedy(eval 槽)被毒,探索档(train 槽,恰好被正确
重置)是干净测量 —— 产生"加探索成功率反而上升"的假信号。move_plate 容忍度
高,旧 harness 的 greedy 68≈val 70 是巧合性"验证通过",掩盖了 bug。

### 验证链
- 标准工具(walk `workspace.eval()`)当日对照:flip 60(ne1)=60(ne25);
  修复版 harness flip greedy 54(其中 seeds 400-449 段=60,严格吻合);
  sandwich 修复版 70 = 记录 val50 70(在 swirl03 测得,03 环境/checkpoint 无罪)。
- 修复:双槽位重置 + final_info task_success 计数 + 逐轮打印。已提交。

### 三任务修正终表(修复版 harness,100 eps,seeds 400-499,ckpt 100k)
| task/run | greedy | asis 三层现状档 | ×4 档 | coarse-only |
|---|---|---|---|---|
| move_plate basestate s1 | 76 | 70 (−6) | 62 (−14) | 21 (−55) |
| flip_cup flip4 s3 | 54 | 38 (−16) | 37 (−17) | — |
| sandwich s2(0805) | 70 | 60 (−10) | 41 (−29) | — |
(flip/sandwich 探索档为 b1+b2 两批各 50 eps 合并,train 槽路径本就干净)

### 结论修订
1. "现状档免费/负税"作废:三任务均付税 −6~−16。探索事件在现有剂量下已触碰
   outcome 相关方向,"剂量太小行为不可见"不成立。
2. 旧结论"×4 平坦=细层事件不携带成败信息"修订:仅 flip 饱和(−16→−17);
   move_plate 额外 −8,sandwich 额外 −19(与其细尺度 L2 敏感度 −28 一致)。
3. coarse-only 灾难性(−55)维持。

### 附带发现
- 时代退化:official_move_plate 0804 批 checkpoints 今日 eval 仅 36-40%
  (记录 72.5,08-05 sweep)。sandwich 0805 批与 basestate 0806 批完好复现。
  时代断层 ≈08-05 晨;候选原因:当日代码变更或 08-06 optstate strip 作业。
  历史封榜数字不受影响(同代代码同代测量);今日起 re-eval 0804 代 checkpoint
  必须先过时代门控。任务 #16 挂账。
- qselect 系列探针脚本踩同款槽位坑(单环境版):绝对数字系统性压低,大间距
  定性结论(9× 反选)稳;引用绝对值前须修复重测。
- Codex baseline 勘误:其"原版"配置 ≠ 我们官方线(append_floating_base_to_low_dim
  =true、无 obs_std_floor_relative、warmup 540)。我们官方 4-run val-best≤25k
  带 56-62(紧);其 74.5/29.0 是另一配置的性质,seed2 全网格 max 30%(病 run),
  +6.5pp 封榜靠该病 run 的 +32 支撑。"sandwich 25k 天然方差大"的旧归因作废。

## Wave-11 判决:×4 三层探索 + λ 衰减,终点大幅低于官方带(2026-08-18)

配方:官方 + bin_explore_probs=[0.008,0.016,0.032](×4 档) + bc_lambda_schedule
linear(1.0,0.0125,50000)。flip_cup/sandwich_remove 各 2 seeds,swirl03。

### 终点(100k+101k 各 50 eps 合并,seeds 400-449,本地评估)
| run | 100k | 101k | 合并 | 官方带 |
|---|---|---|---|---|
| flip s1(复活链) | 30 | 32 | 31 | 51-64 |
| flip s2(单世代) | 34 | 34 | 34 | 51-64 |
| sandwich s1(单世代) | 38 | 44 | 41 | ~70-80.5 |
| sandwich s2(复活链) | 20 | 36 | 28 | ~70-80.5 |

### 读法
- 中场(25k)还在官方带内(flip 22-36 vs 带 10-38;sandwich 42 vs 带 38-56),
  50k 起官方继续爬升而 x4anneal 停滞 → 不是崩溃,是**平台压低 25-40 个点**。
- λ 地板 0.0125 兑现了"不崩"(相变定律再验证),但 ×4 训练时剂量没有换来
  value 增益 —— 行为税(−17/−29)进入数据回路后压低了成功样本比例,
  探索数据未转化(反选/侵蚀机制群依旧)。
- 单世代与复活链 run 同向(34/41 vs 31/28),排除中断伪影主导。
- 归因缺口(设计时已声明):无 decay-only 对照,×4 与 anneal 的责任拆分待
  Wave-12(+MC)与后续 decay-only 臂;move_plate 的 anneal-A 85 先例暗示
  主要嫌疑在 ×4 剂量而非衰减本身。
- 事故记录:看护 v2 判完成 bug 重启已完成 run,流氓世代污染 flip s1(5-95k)
  与 sandwich s2(5-80k)网格;终点文件抢救完好(w11_rescue/),曲线分析只用
  两个单世代 run。判进度以 eval_checkpoints 顶点为准已固化到 v3。

## Wave-12 中场判决:MC 下界救不回 ×4 配方,提前终止(2026-08-18,用户裁定)
配方 = Wave-11 + mc_lower_bound_target。中场(50 eps,seeds 400-449):
flip s1 25k=44→45k=10;flip s2 25k=30→40k=26;sandwich s2 25k=34→50k=10。
对照 Wave-11 同点(flip 22-36/18-36,sandwich 42):不但未拉起,λ 落地后
直接塌方(两个 run 跌到 10)——与 m1/m2 的 MC 值域压缩机制一致,MC×探索
交互假说被否。用户指示中场停机(预注册标准satisfied:各点 ≤ W11 同点,无上翘)。
剂量线(×4 探索家族)至此关闭:Wave-11 平台压低 + Wave-12 塌方 + m1/m2 无增益。
卡位转予 progress potential shaping 臂(commit 28e88f4)。
