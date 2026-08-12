# HANDOFF — CQN-AS 改进研究（R 线，2026-08-12 交接；当晚补丁见 §0.5）

本文件是**唯一现行交接文档**（旧 HANDOFF/HANDOFF_NOBC_CLAUDE 已废；
它们的结论被本线消化或勘误，见 §7）。完整执行日志与全部预注册在
`cqn-rline.md`（顶部有状态总览），文献与路线设计在 `research_paper.md`。

## 0. 使命与两条 criterion 的现状

目标（用户设定）：(1) move_plate/任务族上**密封显著超过原版 CQN-AS**
（非噪声）；(2) 展示**真实价值学习**而非"伪装成 RL 的 IL"。

- **criterion 2：已达成**——反事实失败数据（cfaug）使 critic 在
  held-out 扰动分支上真贬值（复权 −19~−23%，4/4 种子稳定），基线同处
  为 +64% 假上坡；机制三层证据链齐（held-in / held-out / 复权分段）。
- **criterion 1：未达成**——全部臂的行为效应居于种子噪声带
  （±7pp/seed；n=4 配对检测窗 ≈5pp）。当前分数主攻见 §4。

## 0.5 当晚重大补丁（2026-08-12 晚，详见 cqn-rline.md 勘误 #4 与 Wave-7）

1. **勘误 #4**：反事实生成器的 demo 重放从未对齐过（错用 executed_action
   + 对已 rescale 的 demo_action 双重变换）。**全部旧反事实数据实为
   off-manifold 乱舞轨迹**，"near-manifold 一格之差"的表述作废；cfaug
   密封判决与探针数值仍是真实测量，但解释降级为"off-manifold 失败数据
   的效果"。修复后正确用法：`ts.info["demo_action"]` 逐字使用（已是
   tanh 空间）。铁律 10：重放/生成管线必须先过零扰动对照。
2. **恢复伺服落地**（用户提议）：demo 动作前馈 + 关节偏差 tanh 仿射
   比例反馈（raw 偏差 e → 修正 2e/span,gripper 维不反馈）。生成器支持
   `--paired`（同锚开环/恢复双变体）+ manifest.jsonl（含 jerr 与
   object-deviation 字段）。
3. **v2b 生产批**：move_plate ±0.4 paired,306 对（both_s 209 /
   both_f 43 / harm 21 / teach 33）。开环成功率 75% = 邻域真实形状。
   MILES 式场景偏差 check 实测**阴性**（max qpos 范数不预测 harm）。
4. **Wave-7 注入原料已组装**：`exp_local/cqn_rline/inject_fence_v2_moveplate/`
   （120 条开环带真实 return）与 `inject_recovteach_v2_moveplate/`
   （33 反转对 66 条）。均 gitignore,跨机需 rsync 或重生成（~30min）。
5. **MILES 对照**（用户指定调研）:research_paper.md 附录 S1。核心:
   "demo 邻域自监督扰动数据喂 critic"在 citation graph 是空白格;
   MILES 的密度 ablation 与幅度标定思想值得后续移植。
6. 中期读数（等 s2 配对,勿单独引用）:anneal-A s1 密封 69.5(−8.0);
   combined-fc s1 密封 41.5(−1.0,探索通道濒临出局)。zero70b ×2 与
   anneal/comb s2 在训。arm 脚本 finalize 路径 bug 已修
   (--selection-csv 只传文件名);之前两次 sealed_failed 均已手动补。

## 1. 制度与基线（全部密封 200ep @100k 整，seeds 800-999，无选点）

制度 = 当前 config 默认：`truncate_demo_at_success=true`；move_plate 63 维
（append_floating_base=true 默认），flip_cup/其他外部效度任务用 60 维
（`env.append_floating_base_to_low_dim=false` 透传，与其既有基线合同一致）。

| 任务 | 基线 n=4 | 均值 |
|---|---|---|
| move_plate（63 维） | 77.5 / 73.0 / 71.0 / 82.5 | 76.0 |
| flip_cup（60 维） | 42.5 / 37.5 / 51.0 / 49.5 | 45.1 |
| sandwich_remove（60 维，旧 08-05 对，n=2） | 61.0 / 80.5 | （方差凶，慎用） |

## 2. 已判决臂总表（全部预注册密封门）

| 臂 | 任务 | 配对结果 | 判决 |
|---|---|---|---|
| rfloor（人造floor+λ0.0125） | move_plate | −26.0/−26.5 | 关（发现 λ 第三工作=早期塑形） |
| nstep3 | move_plate | −9.5/+3.0 | 关（30k 领先→密封负，中段欺骗#1） |
| 分视野回传 b2（tokensplit） | move_plate | 四种子 −1.1 | 关（平局；排序↑0.53→0.80） |
| 同 b2 | sandwich | −18/−20 | 关（gap税×敏感度定价） |
| 同 b2 / b8 | flip_cup | −5.75 均 / −5.0 | 关（统一定律三连中） |
| flip03 | move_plate | — | 用户撤销 |
| 重排（beam×3种判官×3任务） | 全部 | 全 null | 关（实测：判官只会贬值不会排序） |
| 双头 | — | 未训 | 依附重排，搁置 |
| **cfaug（失败数据增广）** | move_plate | n=2 均 +0.5 | 存活（机制门过） |
| **cfaug** | flip_cup | 四种子 +1.25（2/4 正） | **未过声明门但机制定案**（§3） |
| combined（探索×衰减） | move_plate（截断） | 四种子 −2.1 | 截断吃掉同一红利；其他任务未测→Wave-6 |

## 3. 核心机制资产（可直接写论文的部分）

1. **假上坡定理**：BC-margin 基线的 critic 给"demo 旁一步之遥的扰动分支"
   打出复权后 +64% 的净上坡（OOD 外推乐观）——F2 闭环崩塌的势能面成像。
2. **围栏/填坡**：注入 120 条反事实失败 episode（objective 一字不改）→
   复权后 −19~−23% 真贬值，4/4 种子稳定，held-out 泛化成立，margin/模仿
   几何无损。生成器 `scripts/generate_counterfactual_episodes.py`。
3. **统一定律**：把价值写进决策 critic 的 target（多步/floor 类）必交
   margin 税（gap 0.37→0.21），行为损失 = 税率 × 任务精度敏感度
   （L2 审计：move_plate −2 / flip_cup −13.75 / sandwich −28）。
4. **λ 的四份工作**：兄弟可分辨性（floor 可接）、排序优先（floor 不可）、
   早期行为塑形（rfloor 发现）、off-manifold 保护（围栏可接）。
5. **机制-行为解耦**：λ=1 时 BC 站岗，价值不掌舵——围栏行为红利被
   BC 冗余掩盖（逐种子 corr=0.10）。→ Wave-5 的立论。
6. **复权口径**：截断制度下跨时间窗的 Q 对比必须乘 γ^(t−参考点) 折算，
   否则折扣趋势冒充机制（raw +96% 实为 净+64%；cfaug raw 持平实为
   净−19%）。

## 4. 在途与排队（2026-08-12 时点）

- **Wave-5 BC 退役三臂**（move_plate，cfaug 注入为共同底座）：
  lowlam（λ≡0.0125，对撞 rfloor 51.5/46.5）在训；anneal-A
  （linear 1.0→0.0125@50k，先例 stage-174 续训版 82.5）s1 在训；
  zero70（step_linear 至 70k 归零，对撞 178B 的 0/200）队列。
  **成功标准 = 安全性**（守住基线带即"BC 可退役"定理）。
- **Wave-6 combined-flipcup**（队列头部）：探索×衰减在有头顶空间任务
  的截断制度首测。≥+5 → 分数主攻转探索；平/负 → 探索出局。
- **恢复数据采集**（±0.15 小扰动）双任务进行中 → "拓管臂"原料
  （教策略从管壁回来，直接对准成功率）。
- 预注册排队：cfaug-o2o（10k 预训练版）、cfaug-o2o-noBC（BC 退役
  终局）、失败分类学工具。
- D2 配比调度已实现未上卡（`demo_batch_fraction_schedule`，测试 5/5）。

## 5. 工具箱

| 用途 | 入口 |
|---|---|
| 训练+评估一条龙 | `scripts/run_cqn_trunc_arm.sh ARM UUID EGL SEED STAMP TASK REPLAY_MIN [覆盖...]`（透传口支持任意 Hydra 覆盖；cfaug 臂需先预填 `<run>/replay/`） |
| 反事实/恢复数据生成 | `scripts/generate_counterfactual_episodes.py`（--perturb-delta 0.4=失败围栏 / 0.15=恢复采集） |
| 价值探针（逐样本记录） | `scripts/analyze_cqn_value_fidelity.py` + 复权后处理（cqn-rline.md 复权节的脚本模式） |
| 密封评估/断点续评 | `scripts/eval_cqn_as_snapshot_sweep.py`（--only-steps/--skip-steps；会转 checkpoint 删 snapshot） |
| 束搜索（单 critic 已解锁） | `method.twin_rollout_beam_width>1`；view-dir 补丁法评估旧 checkpoint |
| qselect3 竞技场（未用于本线终审，待用） | `scripts/eval_policy_qselect3.py` |

## 6. 铁律（每条都有事故编号）

1. **四种子 + 同 seed 配对**才允许声明；n=2 晋级被翻盘三次。
2. **中段/验证曲线永不作判决**（三次被密封翻转）；50-ep 点引用须双档
   凑 100（seeds 400-449 + 450-499）。
3. **复权口径**（§3.6）。
4. 杀进程**精确 pid + 排除自身进程树**；pkill -f 自匹配两次自杀。
5. relaunch 永远换新 stamp（同 stamp 复用目录 → 双写/陈旧哨兵事故）。
6. 发射后必须验证：进程树单实例 + cmdline 抓 run dir + train.csv 出行 +
   wiring 指标（cfaug 看 bootstrap 行与 buffer_size；调度看 bc_weight 列）。
7. 编排器等待条件绑训练产物（checkpoint 文件），不绑中间层哨兵。
8. 评估：空卡 3 进程 + 25s stagger + RAM 护栏（<40GB 降 2 并行，
   评估进程 ~8GB RAM）。
9. 任何用于设计决策的数值必须现场重验（HANDOFF 勘误三连的来源）。

## 7. 对旧文档的勘误（继承者勿再引用旧说法）

1. 旧 HANDOFF"move_plate action_sequence=4"→ 实为 **16**。
2. 旧 HANDOFF"63 维补上官方缺的底盘信息"→ 63 维追加是
   **proprioception[26:29] 的纯冗余拷贝**（实测逐位相同）。
3. 旧 HANDOFF"sandwich episode 522/25≈21 tick"→ 实为
   **episode_length 13500 / 25 = 540 tick**。
4. 旧 no-BC 交接的"饱和=缺陷"叙事已被 A18/A19+复权探针重写：见 §3。

## 8. 新机器起跑清单

1. **环境**：python3.11 venv + repo 依赖；`~/.bigym/demonstrations/0.9.0`
   demo 缓存（首跑自动下载或从本机拷贝）。
2. **bigym 本地补丁**（子模块指向上游仓，补丁不在其历史里）：
   `cd third_party/bigym && git apply ../../patches/bigym_local_20260812.patch`
   （含 demo_converter reset-align 等 85 行——**不打补丁 demo 重放不对齐**）。
3. **EGL 必须重探针**：本机的 EGL id ≠ CUDA 序且随主机而异。用
   渲染探针法（1-geom 场景 + 看显存落点；参考 memory
   `egl-under-cvd-pin.md` 的配方）测出本机映射后再 pin。
4. **环境变量**：训练一律 `JAX_PLATFORMS=cuda ROBOBASE_HOST_MERGE=1
   XLA_PYTHON_CLIENT_PREALLOCATE=false`，按 UUID pin 卡，
   双 run/卡 `xla_mem_fraction=0.45` + 120-150s stagger。
5. **跑一个 cfaug 臂的完整配方**（示例 flip_cup seed5）：
   ```bash
   RUN=exp_local/cqn_trunc_arms/cfaug_flip_cup/seed5_<stamp>
   mkdir -p $RUN/replay
   ls exp_local/cqn_rline/counterfactual_episodes_flipcup_60d/*.npz | head -120 | xargs -I{} cp {} $RUN/replay/
   bash scripts/run_cqn_trunc_arm.sh cfaug <UUID> <EGL> 5 <stamp> flip_cup 560 env.append_floating_base_to_low_dim=false
   ```
   （反事实 npz 目录在 `exp_local/`，被 gitignore——**需从本机同步**或用
   生成器重新生成，~2-6h/任务。注意生成合同要与训练合同一致：60 维任务
   生成时也要传 append=false，否则 low_dim 维度撞车——flip_cup 现有
   60d 目录是裁剪修复过的。）
6. **flip_cup/sandwich 必传** `replay_size_before_train`（560/540），
   否则启动即拒。

## 9. 文档地图

- `cqn-rline.md`——本线全日志（状态总览在顶部；预注册、判决、勘误、
  黑话对照表俱全）。
- `research_paper.md`——72 篇文献 + 路线 R1-R7 + 原始预注册卡。
- `cqn-flow.md` / `cqn-no-bc*.md`——前线历史（只读；引用数值前先验证）。
- memory 目录——GPU/EGL/评估纪律等跨会话规则。
