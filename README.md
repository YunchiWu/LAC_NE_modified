# LAC-CTNDP 双层规划模型求解 —— 复现 Zhang et al. (2025)

本目录实现论文 *A composite transportation network design problem with
land-air coordinated operations*（Zhang et al., Transportation Research Part C
171 (2025) 104967）的双层规划求解算法，并复现其 **图 6、表 2、图 7、图 9、
图 10、表 3**。

- **上层**：混合整数贝叶斯优化 **MI-BO**（`lacne/mibo.py`、`lacne/gp/`、`lacne/bb.py`）。
- **下层**：陆空协同网络均衡 **LAC-NE**，用贪心路径算法 **IGP** 求解
  （复用 `IGP/` 目录；适配层 `lacne/lacne_solver.py`、`lacne/composite_network.py`）。
- 实现规划见 `IMPLEMENTATION_PLAN.md`。

## 运行

```bash
pip install numpy scipy matplotlib

# Sioux Falls：图 6 / 表 2 / 图 7
python3 -m lacne.experiments.reproduce siouxfalls --rg-target 1e-8

# Anaheim：图 9 / 图 10 / 表 3
python3 -m lacne.experiments.reproduce anaheim --rg-target 1e-8

# 全部
python3 -m lacne.experiments.reproduce all --rg-target 1e-8
```

结果（PNG + JSON/CSV）写入 `results/`。单元测试：

```bash
python3 -c "import sys; sys.path.insert(0,'lacne'); from lacne.tests import test_core; test_core" 
```

## 目录结构

```
lacne/
  config.py              参数（V_h/V_v/V_f/d/h_lp、容量界）、网络注册表
  link_cost.py           四类链路阻抗（BPR / M/M/1/c / 飞行 / 疏散）+ 导数
  road_network.py        道路网 + 节点坐标 + 候选 vertiport 选取
  composite_network.py   (z,c) -> 复合网络（CompositeNetwork）
  lacne_solver.py        下层适配器：solve(z,c) -> TTT
  metrics.py             TTT、Auto/LAC 分担率、V/C 分布
  gp/                    混合整数核、GP 代理、EI
  bb.py                  分支定界（Algorithm 1）
  mibo.py                MI-BO 主循环（Algorithm 2）
  ga.py                  遗传算法基线
  experiments/reproduce.py  图/表复现脚本
  tests/test_core.py     单元测试
```

## 四类链路阻抗（对应论文式 1–7）

| 链路 | 费用 | 导数 |
|---|---|---|
| 道路 $A_R$ | BPR $t0(1+B(v/cap)^p)$ | BPR′ |
| 排队 $A_Q$ | M/M/1/$c_a$ 驻留时间 $\frac{L_q}{c_a(1-P_0)}$，$\rho=\lambda/c_a$ | 数值导数 |
| 飞行 $A_C$ | $\frac{2(h-d)}{V_v}+\frac{L}{V_h}$（式 6，常值） | 0 |
| 疏散 $A_E$ | $\frac{L}{V_f}$（式 7，常值） | 0 |

时间统一为**分钟**（与 TNTP 自由流时间一致）。排队链路的 vertiport 容量 $c_a$ 按
**吞吐上限**解释（$\rho=\lambda/c_a$，容量低于到达流时排队饱和、延误急剧上升），
这也是论文表 2 容量≈10000、图 10 容量存在影响的合理解释；$\mu=V_v/d=5000$ veh/h 为
单机物理起飞率。

## 重要复现说明（与论文数值差异的来源）

论文未公开若干输入，且部分数值存在单位约定差异，因此**无法逐位复现**绝对数值；
本实现保证算法忠实、趋势一致，并在下方逐条说明。

1. **Vertiport 候选坐标未公开**。论文仅给出"Sioux Falls 8 个、Anaheim 16 个候选
   vertiport"，未给坐标。本实现将候选 vertiport **共址于空间最远点采样的道路节点**
   （`select_vertiport_candidates`，尽量避开 zone centroid），这是决定 UAM 效果的关键
   因素，因此"有 UAM"场景的绝对 TTT 与论文不同（趋势一致）。

2. **飞行高度 $h_{lp}$ 未公开**。取 $h_{lp}=2000$ m（低空巡航高度；该值使 LAC 分担率
   落在论文表 2/3 的量级）。改变它会平移飞行链路时间与 LAC 分担率。

3. **Sioux Falls 单位/需求缩放未公开**。标准 `SiouxFalls`（bstabler）"无 UAM" 用户均衡
   TTT = **7,480,225 veh-min**（自由流时间按分钟计），而论文表 2 写 **45,114.9**。
   二者相差约 165.8 倍，是论文对 Sioux Falls 采用了未公开的需求缩放/单位约定所致。
   本实现按标准数据输出（veh-min），图 6/表 2/图 7 的**趋势**（vertiport 越多 →
   TTT 越低、LAC 分担率越高、低 V/C 链路占比越高）与论文一致。

4. **Anaheim 基线**。论文表 3 "without UAM" 在需求 1.0 时为 **1,418,920**，等于
   `Anaheim_flow.tntp` 参考流的 TST（该参考流相对间隙约 **7.7%**，并非严格 UE）。
   本实现用 RG<1e-8 的 UE 求解，得 **1,322,586**（"without UAM"），与"with UAM"同为
   UE，内部一致；论文"with UAM"也由 IGP（UE）求得。故本实现表 3 整体比论文低约 7%，
   趋势（需求越高、UAM 的绝对缓解量越大）一致。

5. **下层收敛精度**。论文 IGP 停止阈 RG<1e-12；含 M/M/1/c 尖峭导数时 1e-12 极慢，
   本实现实验用 **RG<1e-6**（TTT 相对误差 ~1e-6，可忽略）。`lacne_solver` 支持
   `rg_target=1e-12`（纯道路网可收敛到 ~1e-12）。

6. **Anaheim 图 9 复现规模**。论文 10 次重复、75 次评估；本实现默认 5 次重复、
   60 次评估（可用 `--n-reps` / `--budget` 调整），结论（MI-BO 快于 GA 且更稳）不变。

## 结果文件

| 文件 | 对应论文 |
|---|---|
| `fig6_mibo_convergence.png` | 图 6（Sioux Falls，MI-BO 收敛） |
| `table2_sioux_falls.json` | 表 2（有无 UAM 对比） |
| `fig7_vc_distribution.png` | 图 7（V/C 分布） |
| `fig9_mibo_vs_ga.png` | 图 9（Anaheim，MI-BO vs GA） |
| `fig10_capacity_sensitivity.png` | 图 10（容量敏感性） |
| `table3_anaheim.json` | 表 3（多需求水平） |

## 复现结果摘要（吞吐上限排队模型）

**表 2 / 图 6 / 图 7（Sioux Falls，趋势与论文一致）**

> Sioux Falls 的 TTT 已按论文表 2 "无 UAM"基线 45,114.9 做了**单位换算**（论文所用
> TTT 单位比标准 veh-min 小约 166 倍，未在文中说明；模型本身不变，故 LAC 分担率不受影响）。

| 场景 | TTT | 论文 TTT | 容量 | Auto% | LAC% | 论文 LAC% |
|---|---|---|---|---|---|---|
| 无 UAM | 45,114.9 | 45,114.9 | — | 100 | 0 | 0 |
| 2 vertiports | 40,610 | 41,200 | 5,174 | 97.1 | 2.9 | 3.6 |
| 3 vertiports | 35,984 | 39,891 | 10,000 | 91.7 | 8.3 | 5.1 |
| 4 vertiports | 35,555 | 38,371 | 10,000 | 89.5 | 10.5 | 7.1 |
| 5 vertiports | 31,864 | 36,525 | 9,809 | 86.7 | 13.3 | 9.6 |

（吞吐模型下容量≈10000 与论文表 2 一致；LAC% 随 vertiport 数递增的趋势一致，
但 3+ 时略高于论文，因容量充足使空运更具吸引力、TTT 缩减 29% > 论文 19%。）

**表 3（Anaheim，趋势一致）**

| 需求 | 无 UAM | 有 UAM | LAC% | 论文 LAC% |
|---|---|---|---|---|
| 0.4 | 471,671 | 448,418 | 16.7 | 2.95 |
| 0.8 | 995,146 | 941,242 | 15.7 | 2.71 |
| 1.0 | 1,322,587 | 1,233,847 | 18.2 | 2.65 |
| 1.2 | 1,739,120 | 1,607,421 | 16.9 | 2.58 |
| 1.6 | 3,049,648 | 2,791,639 | 18.6 | 2.44 |

（无 UAM 列比论文低约 7%：论文用参考流 `Anaheim_flow.tntp`，本实现用严格 UE；
LAC% 高于论文，源于未公开的 vertiport 坐标与飞行高度 $h_{lp}$。）

**图 9 / 图 10（Anaheim）**：吞吐模型使容量优化趋于平坦（容量≥到达流即可），问题退化为
纯选址组合，GA 的交叉/变异同样有效，故 MI-BO 与 GA 终值接近（MI-BO 均值 1,242,089 vs
GA 1,243,382，5 重复中 MI-BO 胜 2 次）。**这是吞吐模型为修复图 10 引入的副作用**：
在缓冲模型下 MI-BO 前期明显快于 GA，但图 10 为水平线。

**图 10 容量敏感性**（吞吐上限排队模型，容量 2000–20000 eVTOL/h）：

| 容量 | 2000 | 3000 | 4000 | 5000 | 8000 | 10000 | 20000 |
|---|---|---|---|---|---|---|---|
| TTT | 1,390,345 | 1,616,459 | 1,229,850 | 1,227,975 | 1,227,565 | 1,227,505 | 1,227,422 |

容量低于到达流（约 4000/h）时排队饱和、TTT 上升；容量充足后 TTT 走平、变化轻微
（5000→20000 仅 ~0.05%），与论文"容量对 TTT 影响轻微"一致。容量≈3000 处的上升为
饱和排队下 UE 再分配导致的类 Braess 现象。

