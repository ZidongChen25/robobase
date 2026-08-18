# Sparse-reward value learning: 调研综合与路线图 (2026-08-18)

## 一、用户问题的直接回答
"是不是要做 state/progress estimation 才能学好 value?" — **不是必要条件**。
RLPD/IBRL/WSRL/In-Sample 系列在 sparse 0/1 + 少量 demo 上不用任何 progress
估计就能学出真 value。但 BiGym 当前 SOTA (AC3, arXiv 2508.11143) 确实加了
对比式 progress 内在奖励——同时它的 actor 只在成功轨迹上更新(制度化
self-imitation, BC:Q 权重 = 1.0:0.1)。即:**SOTA 是朝着 self-imitation
更重的方向赢的**,和我们"增益骑在 self-imitation 上"的诊断一致。

## 二、三个外部复现,重构我们的核心发现
1. **崩溃被外部复现**:Q-chunking 的消融 RLPD-AC(分块 critic、无行为约束)
   "performs poorly" —— 我们的 5k 崩溃是领域已知现象,不是实现 bug。
2. **崩溃是"接缝瞬态"不是"永久依赖"**(WSRL, ICLR 2025):撤锚导致的
   Q 发散源于离线/在线分布错配,可用 warmup(用仍带约束的策略滚几个
   episode → 高 UTD 重校准 → 再撤锚)避免。**永远不要冷撤 hinge**。
3. **我们的复活探针结论 = 领域共识**:QC(best-of-N flow 提案)、IBRL
   (IL 提案进 bootstrap max)、PA-RL(候选重排)、In-Sample Softmax
   (支撑内 bootstrap)全部收敛到同一模式——**不修 critic 在自由空间的
   毛病,而是限制"问 critic 什么"**。我们"限制解码=全部,排序=零"的
   探针结果正是这个模式的独立发现。

## 三、我们的阴性结果不覆盖的方案(关键判别)
- MC lower bound(已falsify)= 对已执行动作的目标钳制;
- Q-Transformer unseen floor(Wave-1 已falsify,−26.25)= 无掩码推向 Vmin;
- **Cal-QL = 带掩码的单侧下压**(仅当 Q 超过 demo 参考值 V^ref 才压),
  保 action gap——我们的两个阴性结果都不覆盖它;
- **In-Sample/IQL = bootstrap 只在已执行 bin 上取 max**——从源头切断
  sibling 乐观注入(病 1),侵蚀注入率≈0,病 2 的赛跑直接不用跑。
  离散 5-bin 结构让它是精确解而非近似。

## 四、可跑的臂(按机制拟合度 × 实现成本排序)
| 臂 | 攻哪个病 | 代码现状 |
|---|---|---|
| A. In-sample 目标 + 活 BC-head 支撑掩码解码 | 病1源头 | FS-CQN 机器 80% 可复用;致命差异:掩码不冻结(FS-CQN 死于冻结掩码在漂移态失效) |
| B. WSRL 阶段撤锚(warmup→高UTD重校→撤hinge) | 崩溃瞬态 | 零新代码(schedule+resume) |
| C. JSRL 尾部探索课程(guide 滚入,探索只在退火尾段) | 病3+病2 | workspace 小改;直接治 Wave-11 的死因(均匀剂量毒化数据流) |
| D. Cal-QL 掩码下压 / ENOTO 集成LCB | 病1校准 | pessimistic_twin_critic 已建但硬门在 no-BC 线(cqn_as.py:1954-2022) |
| E. 自适应 λ 控制器(rollout 成功率反馈,降/回收) | 优雅退火 | 最便宜;anneal-A 85 的可控版 |
| F. 进度势函数 PBRS(demo 时间指数,以 critic 初始化实现,Wiewiora 等价避开 C51 support 问题) | 病2 | 无现成钩子,新代码;二线 |
| G. DrS 式成败判别器 dense 奖励(失败变负样本,tanh 带界防反转) | 病3变废为宝 | 新代码;dense 化路线里最贴合 |

## 五、推荐主攻
统一洞见:**BC 的正确位置不是 loss,是提案集/支撑掩码**。下一波建议:
- W13a = 臂A+B 复合:活掩码 in-sample 化 + WSRL 阶段撤 hinge(2 seeds);
- W13b = 臂C:JSRL 尾部探索替代均匀剂量(2 seeds);
互为对照:A/B 治病1,C 治病3;都不新增剂量。进度估计(F/G)留作二波,
除非 W13 双败。Wave-12(MC+explore)按在跑,预计关闭剂量线。
原始调研全文: reports/sparse_value_survey_20260818_raw.txt
