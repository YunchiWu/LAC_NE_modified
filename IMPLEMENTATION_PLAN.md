# 陆空协同复合交通网络设计双层规划模型求解算法实现规划

> 依据论文：Honggang Zhang, Jinbiao Huo, Churong Chen, Zhiyuan Liu.
> *A composite transportation network design problem with land-air coordinated operations*.
> Transportation Research Part C 171 (2025) 104967（下称"本文/原论文"）。
>
> 上层模型：混合整数贝叶斯优化 **MI-BO**（Mixed-Integer Bayesian Optimization，Algorithm 2）。
> 下层模型：陆空协同网络均衡 **LAC-NE**（Land-Air Collaboration Network Equilibrium，式 13–23），
> 用**改进梯度投影 IGP** 求解（原论文引用 Xie, Nie & Liu 2018 的贪心路径算法）。
> 下层求解器**已在 `IGP/` 目录实现**，本文档重点规划（1）如何复用它并把链路阻抗从纯 BPR 扩展到四类链路，
> （2）上层 MI-BO 如何实现并与下层 IGP 串起来。默认技术栈 **Python 3 + NumPy/SciPy**（与 `IGP/` 一致），预留 C++ 移植接口。

---

## 0. 一句话概括

把"政府选 vertiport 位置 `z` 与容量 `c`"建模为**双层 Stackelberg 博弈**：
上层用 **MI-BO**（混合整数高斯过程代理 + 分支定界最大化整数约束 EI）在黑盒目标上采样寻优；
每次采样，固定 `(z, c)` 构建陆空复合网络，用 **IGP** 求解下层 **LAC-NE** 得到用户均衡流量，
回代得到总旅行时间 TTT 作为该样本的目标值。核心难点有二：
（1）混合整数代理模型与整数采集；（2）把标准 UE 求解器 IGP 适配到含排队/飞行/疏散四类链路的复合网络。

---

## 1. 问题定义与双层模型

### 1.1 网络与符号（对应原论文 Section 2 的 notation）

| 符号 | 含义 |
|---|---|
| $N_R,\ A_R$ | 道路网节点集、链路集 |
| $A_Q$ | vertiport 的**进场排队链路**（queue links） |
| $A_E$ | vertiport 的**离场疏散链路**（evacuation links） |
| $A_C$ | 连接任意两个 vertiport 的**飞行链路**（flight links） |
| $P$ | 候选 vertiport 集合，$p\in P$ |
| $W, K_w$ | OD 对集合、OD 对 $w$ 的复合网络路径集合 |
| $q_w$ | OD 对 $w$ 的总需求 |
| $B$ | 允许建设的 vertiport 数量上限 |
| $C$ | vertiport（排队链路）容量上限 |
| $V_h, V_v, V_f, d$ | 水平巡航速度、垂直起降平均速度、道路/疏散链路自由流速度、飞行安全间距 |
| $h_{lp}, L_{\hat a}, L_{\tilde a}$ | 飞行高度、飞行链路长度、疏散链路长度 |

**决策变量**
- 上层：$z_p\in\{0,1\}$（vertiport 是否建设）、$c_a>0$（排队链路容量，$a\in A_Q$）。
- 下层：路径流量 $f_k^w\ge 0$；辅助 0-1 变量 $X_{np}$（道路节点 $n$ 是否接入 vertiport $p$）、$Y_{lp}$（vertiport $l,p$ 是否互连）。

### 1.2 上层模型（式 8–12）—— 决策者 / 政府

$$
\min_{c,z}\ f(c,z)=\sum_{a\in A_R} v_a t_a+\sum_{a\in A_Q} v_a t_a+\sum_{\hat a\in A_C} v_{\hat a} t_{\hat a}+\sum_{\tilde a\in A_E} v_{\tilde a} t_{\tilde a}
\tag{8}
$$

$$
\text{s.t.}\quad \sum_{p\in P} z_p \le B \tag{9},\qquad
c_a \le C,\ \forall a\in A_Q \tag{10},
$$

$$
z_p\in\{0,1\}\ \forall p \tag{11},\qquad c_a>0\ \forall a\in A_Q \tag{12}.
$$

目标即**全网络总旅行时间 TTT**；$v,t$ 是下层 UE 解隐含给出的流量/时间，因此 $f(c,z)$ 是**隐式、非线性、不可微**的黑盒。

### 1.3 下层模型 LAC-NE（式 13–23）—— 通勤者 / UE

$$
\min Z(f)=\underbrace{\sum_{a\in A_R}\int_0^{v_a} t_a(w)dw}_{\text{道路 Beckmann 项}}
+\underbrace{\sum_{a\in A_Q}\int_0^{v_a} t_a(w)dw}_{\text{排队 Beckmann 项}}
+\underbrace{\sum_{\hat a\in A_C} v_{\hat a}t_{\hat a}+\sum_{\tilde a\in A_E} v_{\tilde a}t_{\tilde a}}_{\text{飞行/疏散为常值，故为线性项}}
\tag{13}
$$

$$
\text{s.t.}\quad
\sum_{n\in N_R} X_{np}=z_p\ \forall p \tag{14},\quad
Y_{pl}=z_p z_l\ \forall p\ne l \tag{15},
$$

$$
v_a=\sum_w\sum_k f_k^w\chi_{a,k}^w\ \forall a\in A_R \tag{16},
\quad
v_a=\sum_w\sum_k\sum_{n}\sum_p f_k^w\chi_{a,k}^w X_{np}\ \forall a\in A_Q \tag{17},
$$

$$
v_{\tilde a}=\sum_w\sum_k\sum_n\sum_p f_k^w\chi_{\tilde a,k}^w X_{np}\ \forall \tilde a\in A_E \tag{18},
\quad
v_{\hat a}=\sum_w\sum_k\sum_l\sum_p f_k^w\chi_{\hat a,k}^w Y_{pl}\ \forall \hat a\in A_C \tag{19},
$$

$$
\sum_{k\in K_w} f_k^w=q_w\ \forall w \tag{20},\qquad f_k^w\ge0 \tag{21},
\qquad X_{np}\in\{0,1\} \tag{22},\qquad Y_{pl}\in\{0,1\} \tag{23}.
$$

> **关键认识**：式 13 仍是 Beckmann 凸规划（道路、排队两项是严格递增可积函数，飞行/疏散是常值线性项），
> 所以 UE 解唯一（链路流唯一），且**"所有被使用路径等代价、未使用路径代价更高"的 Wardrop 条件照旧成立**。
> 差异只在于链路阻抗函数不再是单一 BPR —— 这正是复用 IGP 时唯一需要实质改动的地方。

### 1.4 四类链路阻抗函数（式 1–7）

**(a) 道路链路 $a\in A_R$：经典 BPR**

$$
t_a = t0_a\left(1+\alpha_a\left(\frac{v_a}{cap_a}\right)^{\beta_a}\right)
\qquad
t_a' = t0_a\,\alpha_a\,\beta_a\,\frac{v_a^{\beta_a-1}}{cap_a^{\beta_a}}
$$

> 现有 `IGP/igp/network.py` 已实现（含 Bar-Gera 广义代价 `dist_w·length + toll_w·toll`；对 Sioux Falls/Anaheim 可设权重为 0）。

**(b) 排队链路 $a\in A_Q$：M/M/1/$c_a$ 点排队（式 1–5）**

$$
t_a=\frac{L_q^a}{\mu(1-P_0^a)} \tag{1},\qquad
P_0^a=\frac{1-\rho_a}{1-\rho_a^{\,c_a+1}},\ \rho_a\ne1 \tag{2}
$$

$$
L_q^a=\frac{\rho_a}{1-\rho_a}-\frac{(c_a+1)\rho_a^{\,c_a+1}}{1-\rho_a^{\,c_a+1}} \tag{3},
\qquad
\rho_a=\lambda_a/\mu,\quad
\lambda_a=\sum_w\sum_k f_k^w\chi_{a,k}^w \tag{4}
$$

$$
\mu=V_v/d \tag{5}.
$$

- $\lambda_a$ = 该排队链路的流量（到达率），$\mu$ = 服务率（由 $V_v,d$ 决定），$c_a$ = 容量（**上层决策变量，在下层给定后为常数**）。
- 导数 $t_a'=\frac{dt_a}{d\lambda_a}$ 需用于贪心二次近似 / GP 步长（见 §3.4、§7）。

**(c) 飞行链路 $\hat a\in A_C$：常值（式 6）**

$$
t_{\hat a}=\frac{2(h_{lp}-d)}{V_v}+\frac{L_{\hat a}}{V_h},\qquad t_{\hat a}'=0.
$$

**(d) 疏散链路 $\tilde a\in A_E$：常值（式 7）**

$$
t_{\tilde a}=\frac{L_{\tilde a}}{V_f},\qquad t_{\tilde a}'=0.
$$

---

## 2. 求解算法总体框架

原论文 Figure 4 的流程，翻译成可执行结构如下（**上层 MI-BO 驱动，下层 IGP 被当作"目标函数求值器"调用**）：

```
MI-BO (Algorithm 2)
  │
  ├─ 1. 随机生成 m 个初始可行样本 x_i=(z_i, c_i)
  │     对每个样本:  构建复合网络 → IGP 求解 LAC-NE → 得 f_i = TTT
  │
  └─ 2. while 未达预算(如 75 次评估):
        ├─ 用样本集 D_m=(X_m, F_m) 拟合混合整数 GP 代理 (式 24–32)
        ├─ 构造整数约束 EI 采集函数 (式 33–35)
        ├─ 分支定界最大化 EI (Algorithm 1) → 新样本 x_{m+1}
        ├─ 构建复合网络 → IGP 求解 LAC-NE → 得 f_{m+1}
        └─ D_{m+1} = D_m ∪ {(x_{m+1}, f_{m+1})}
  │
  └─ 3. 返回 x* = argmin_i f_i
```

- **停止准则**：预算制（原论文 Sioux Falls 用 75 次评估），而非收敛阈值。
- **随机性来源**：初始样本随机生成。缓解方式（原论文 Remark 2）：多次重复取最优，或增大样本预算。
- **下层停止准则**：IGP 相对间隙 $\mathrm{RG}<10^{-12}$。

---

## 3. 下层求解器：复用 `IGP/` 并适配 LAC-NE

### 3.1 `IGP/` 现状（已实现，直接可复用）

| 原论文需求 | `IGP/` 现有实现 | 备注 |
|---|---|---|
| 贪心路径算法 Algorithm 1/2（Xie et al. 2018） | `assignment.py::adjust_greedy` + `main_loop`/`inner_loop` | 即原论文所称 IGP |
| 主循环列生成 + 内循环智能调度 | `main_loop()` / `inner_loop()` | `use_inner_loop=True` |
| 相对间隙 RG | `relative_gap()` | 式 23 |
| BPR 代价 + 导数 | `network.py::compute_costs/update_links_costs` | 仅道路 |
| 最短路树 + 路径恢复 | `shortest_path.py` | scipy C dijkstra |
| TNTP 解析 | `network.py::parse_net/parse_trips` | 道路网 + OD |

> **术语澄清**：原论文的 "IGP" 即 Xie et al. 2018 的贪心路径算法（主算法 Algorithm 1+2）；
> `IGP/README.md` 里另列了 "iGP = GP+内循环" 作为对比基线。**接线时下层用 `PathSolver(subproblem="greedy", use_inner_loop=True)`**（即贪心 + 内循环），`subproblem="gp"` 仅作基线对照。

### 3.2 核心洞察：IGP 核心与链路阻抗是解耦的

`IGP/igp/assignment.py` 的贪心/GP/内循环/RG **只通过 `net.t`（链路代价）与 `net.dt`（链路代价导数）** 感知网络：

- 路径代价 $v_h=\sum_{a\in h} t_a$ —— `_path_v_s()` 的 `v`；
- 路径导数和 $s_h=\sum_{a\in h} t_a'$ —— `_path_v_s()` 的 `s`；
- 二次近似常数 $c_h=v_h-s_h g_h$ —— `adjust_greedy()`；
- TST $=\sum_a v_a t_a$、RG $=(TST-SPTT)/TST$ —— `relative_gap()`。

因此 **LAC-NE 适配 = 把 `net.t/dt` 按链路类型换成四类阻抗即可**，贪心/GP/内循环代码**无需改动**：

$$
t_a = \begin{cases}
\text{BPR}(v_a) & a\in A_R \\
W_q(\lambda_a; c_a) & a\in A_Q \\
\text{const} & a\in A_C\cup A_E
\end{cases}
,\qquad
t_a' = \begin{cases}
\text{BPR}'(v_a) & a\in A_R \\
dW_q/d\lambda_a & a\in A_Q \\
0 & a\in A_C\cup A_E
\end{cases}
$$

### 3.3 复合网络构建（原论文 Remark 1）

给定上层决策 $(z,c)$，构造复合网络（一个**扩展有向图**，节点/链路重新编号）：

1. 保留道路网全部节点与链路（$N_R, A_R$）。
2. 对每个 $z_p=1$ 的 vertiport $p$：新建 vertiport 节点 $\nu_p$；把离它最近的**道路节点** $n_p$ 连入：
   - **排队链路**（$A_Q$）：$n_p\to\nu_p$，容量 $c_p$；
   - **疏散链路**（$A_E$）：$\nu_p\to n_p$（与进场相反，构成"两向接入"）。
3. 对任意两个已建 vertiport $l,p$：新建双向**飞行链路**（$A_C$）：$\nu_l\leftrightarrow\nu_p$。
4. OD 需求维持道路网 TNTP 的 OD 不变（通勤者在道路起点出发、可选陆空组合路径）。
5. 输出统一链表：`tail/head/capacity/fftt/B/power/length/...` 之外**附加 `link_type ∈ {ROAD, QUEUE, FLIGHT, EVAC}`**，
   并携带每链路专属参数（飞行高度、长度、$V_h,V_v,V_f,d$、队列容量 $c_a$ 等）。

> 每次上层采样都要**重建**复合网络（拓扑随 $z$ 变化），`ShortestPathEngine` 的 CSR 也随之重建；建图本身是 $O(|A|)$ 级、可忽略。

### 3.4 需要改动的点（对 `IGP/` 的最小侵入式扩展）

| 改动 | 文件 | 内容 |
|---|---|---|
| 1. 链路类型化 | `network.py` | `Network` 增加 `link_type`；`compute_costs/update_links_costs` 按类型分派代价与导数 |
| 2. 排队阻抗 + 导数 | 新增 `lacne/queue_cost.py`（或并入 `network.py`） | 实现式 1–5 的 $t_a(\lambda;c)$ 与 $dt_a/d\lambda$ |
| 3. 常值链路 | `network.py` | 飞行/疏散链路 `t=const, dt=0`（现有 `S_FLOOR` 已防御 $s_h=0$） |
| 4. 复合建图 | 新增 `lacne/composite_network.py` | §3.3 的建图逻辑，输出 `Network` + 需求 |
| 5. 下层求解适配器 | 新增 `lacne/lacne_solver.py` | `solve(z, c) -> (TTT, flows, stats)`：建图→`PathSolver(...).solve(rg_target=1e-12)`→回代 TTT |

### 3.5 与原论文符号的代码映射

| 原论文 | 代码 |
|---|---|
| $v_a$ | `Network.x` |
| $t_a$ | `Network.t` |
| $t_a'$ | `Network.dt` |
| $f_k^w$ | `OD.flows` |
| $K_w$ / 路径集 | `OD.paths` |
| $q_w$ | `OD.demand` |
| $m_w$（最短路代价） | `ShortestPathEngine.dijkstra` 返回的 `dist` |
| RG | `PathSolver.relative_gap()` |
| Algorithm 1/2 | `adjust_greedy` / `main_loop`+`inner_loop` |

---

## 4. 上层求解器：MI-BO 实现设计

决策向量 $x=(z,c)$：$z$ 为 $d_z=|P|$ 维 0-1 向量，$c$ 为连续容量向量（维度 $|A_Q|=|P|$，未建 vertiport 的容量项由 $z$ 掩蔽，见 §4.5）。

### 4.1 混合整数高斯过程代理（式 24–32）

**先验** $f(x)\sim\mathcal{GP}(g(x),\kappa)$，取 $g(x)=0$（拟合前对 $F_m$ 做零均值标准化，预测后再还原）。

**混合整数核**（式 28–30）：离散用 Hamming 距离、连续用欧氏距离，两核相乘：

$$
\kappa_z(z,z')=\sigma_z^2\exp\!\Big(-\frac{\sum_{i=1}^{d_z}(1-\delta_{z_i,z_i'})}{2\eta_z^2}\Big)
,\quad
\kappa_c(c,c')=\sigma_c^2\exp\!\Big(-\frac{\|c-c'\|_2^2}{2\eta_c^2}\Big)
$$

$$
\kappa_{MI}(x,x')=\kappa_z(z,z')\,\kappa_c(c,c')
=\sigma_{MI}^2\exp\!\Big(-\frac{\sum_i(1-\delta_{z_i,z_i'})}{2\eta_z^2}
-\frac{\|c-c'\|_2^2}{2\eta_c^2}\Big)
\tag{30}
$$

其中 $\sigma_{MI}=\sigma_c\sigma_z$，$\delta$ 为 Kronecker 函数。**超参数 $\sigma_{MI},\eta_z,\eta_c$ 用最大似然估计（MLE）**：最大化 $\log p(F_m|X_m)=-\tfrac12 F_m^\top K_m^{-1}F_m-\tfrac12\log|K_m|-\tfrac m2\log2\pi$；优化可用 scipy `L-BFGS-B`（取对数尺度），协方差矩阵加 jitter $\epsilon I$ 保证数值稳定。

**预测**（式 31–32，给定样本集 $D_m$，对新点 $x'$）：

$$
\mu_f(x')=K_*^\top K_m^{-1}F_m,\qquad
\sigma_f^2(x')=K_{**}-K_*^\top K_m^{-1}K_*
$$

其中 $K_*=[\kappa_{MI}(x_i,x')]$，$K_{**}=\kappa_{MI}(x',x')$，$K_m$ 为 $m\times m$ 核矩阵。

### 4.2 整数约束 EI 采集函数（式 33–35）

记 $f^*=\min\{f_1,\dots,f_m\}$。EI 及其闭式：

$$
\mathrm{EI}(x)=\mathbb E[\max(f^*-f(x),0)]
=(f^*-\mu_f(x))\,\Phi\!\Big(\frac{f^*-\mu_f(x)}{\sigma_f(x)}\Big)
+\sigma_f(x)\,\phi\!\Big(\frac{f^*-\mu_f(x)}{\sigma_f(x)}\Big)
\tag{33,34}
$$

带整数约束的采集子问题 **P1**：

$$
\max_x\ \mathrm{EI}(x)\quad\text{s.t.}\quad c_l\le c\le c_u,\quad z_i\in\{0,1\}, i=1,\dots,|P|.
\tag{35,36,37}
$$

> 上层原约束 $\sum_p z_p\le B$ 在采集/初始采样阶段**显式纳入**：采样与 B&B 只在满足该基数约束的 $z$ 上搜索（在 B&B 分支时把"已选数量 ≤ B"作为可行性剪枝）。

### 4.3 分支定界求解整数约束 EI（Algorithm 1）

标准"max-EI 分支定界"：

1. 松弛全部 $z_i\in[0,1]$，搜索空间 $X$；队列 $L=\{X\}$；当前最佳 $f_L^*=-\infty$。
2. 当 $L\ne\emptyset$：
   - 从 $L$ 取子空间 $S$；
   - 在 $S$ 内优化 EI（非凸 → **多起点 + L-BFGS-B** 求局部最优 $x_S^*$，其 EI 值作为该节点**上界**；可参考 Jones et al. 1998 的界估计做更紧剪枝）；
   - 若 $x_S^*$ 整数可行：若 $\mathrm{EI}(x_S^*)>f_L^*$，更新 $x^*=x_S^*,\ f_L^*=\mathrm{EI}(x_S^*)$；
   - 否则若 $\mathrm{EI}(x_S^*)>f_L^*$：随机选一个分数分量 $z_i$ 分支为 $S_1=\{z_i=1\}, S_2=\{z_i=0\}$，$L\gets L\cup\{S_1,S_2\}$；
   - $L\gets L\setminus\{S\}$。
3. 返回 $x^*$（即 $x_{m+1}$）。

> 注意 Algorithm 1 中 `f(x_S*)` 应理解为**该点 EI 值**（论文此处符号与目标函数 $f$ 混用）；实现时按 max-EI 处理。

### 4.4 MI-BO 主循环（Algorithm 2）

```
1. 随机生成 m 个满足 (9)(10)(12) 的初始样本 X_m；每个用 IGP 求下层 → F_m
2. while 未达预算:
   3. 按式(32) 拟合混合整数 GP
   4. Algorithm 1 解 P1 → x_{m+1}
   5. 构建复合网络 → IGP 解 LAC-NE
   6. f_{m+1} = f(x_{m+1}) = TTT
   7. X_{m+1}=X_m∪{x_{m+1}}, F_{m+1}=... , m += 1
3. x* = argmin_i f_i；返回 x*
```

### 4.5 决策变量编码与约束处理（设计决定，需与原论文 C++ 对齐时复核）

- **$z$ 维度固定为 $|P|$**；**$c$ 维度固定为 $|P|$**（每个候选 vertiport 一个容量），只有 $z_p=1$ 的容量进入建图，未建的容量项不进目标。
- 核函数 $\kappa_c$ 对"未建 vertiport 的容量"也参与距离计算——为减弱其影响，可在计算 $\|c-c'\|$ 时**用 $z$ 掩蔽**（只累计 $z_p=z'_p=1$ 的容量差），或直接把未建容量钳到同一参考值；**推荐前者**，并在验证时对比两种方案。
- 容量范围 $c_a\in(c_l,c_u]$，$c_l$ 取小正数（如 1），$c_u=C$（原论文 Sioux Falls 表 2 中容量 ≈ 9511–10000，$C$ 取 ≥10000）。
- 基数约束 $\sum z_p\le B$：初始采样用"随机选 ≤B 个位置"；B&B 分支时剪掉超基数的节点。

---

## 5. 模块划分与目录结构

在 `IGP/` 之外新增上层 + 集成代码（**依赖并 import `IGP.igp`**，不复制）：

```
lacne/
├── IMPLEMENTATION_PLAN.md          # 本文档
├── paper/Zhang et al. 2025.pdf     # 原论文（已存在）
├── IGP/                            # 下层 IGP 求解器（已存在，最小侵入式扩展）
│   └── igp/{network,shortest_path,assignment,run}.py
└── lacne/                          # 新增：上层 MI-BO + LAC-NE 集成
    ├── __init__.py
    ├── config.py                   # 参数：V_h,V_v,V_f,d,B,C,候选 vertiport,预算
    ├── link_cost.py                # 四类阻抗 + 导数（BPR / M|M|1|C / const）
    ├── composite_network.py        # (z,c) → 复合 Network + OD + link_type
    ├── lacne_solver.py             # 下层适配器：solve(z,c)->TTT,flows,stats（封装 PathSolver）
    ├── gp/
    │   ├── kernel.py               # κ_z, κ_c, κ_MI（含 z 掩蔽选项）
    │   ├── model.py                # GP 预测 μ,σ² + MLE 超参数
    │   └── acquisition.py          # EI + 整数约束 EI
    ├── bb.py                       # 分支定界（Algorithm 1）
    ├── mibo.py                     # Algorithm 2 主循环
    ├── metrics.py                  # TTT、V/C 分布、Auto/LAC share
    ├── run.py                      # CLI：跑 Sioux Falls / Anaheim
    └── tests/
        ├── test_link_cost.py       # M/M/1/C 与 BPR 手算对照 + 导数有限差分
        ├── test_gp.py              # 核、预测、EI 手算对照
        ├── test_bb.py              # 小规模 B&B 正确性
        └── test_end_to_end.py      # 小网络 MI-BO + LAC-NE 端到端
```

模块依赖关系：

| 模块 | 输入 | 输出 | 依赖 |
|---|---|---|---|
| `link_cost` | 链路类型、参数、流量 | `t, dt` | — |
| `composite_network` | 道路网、候选 vertiport、`(z,c)` | 复合 `Network`+OD | `IGP.igp.network` |
| `lacne_solver` | 复合 `Network`、OD | TTT、流量、RG、share | `IGP.igp.assignment` |
| `gp.kernel/model/acquisition` | 样本集 $D_m$ | 核矩阵、$\mu,\sigma^2$、EI | — |
| `bb` | GP 代理、$f^*$ | $x_{m+1}$ | `gp.acquisition` |
| `mibo` | 道路网、vertiport 配置、预算 | 最优 $(z,c)$、收敛曲线 | `bb`, `lacne_solver` |

---

## 6. 输入输出与数据

### 6.1 输入

- **道路网 + OD**：TNTP 格式（`IGP` 已支持解析）。
  - Sioux Falls：24 节点 / 76 链路 / 576 OD（论文 §5.1）。
  - Anaheim：416 节点 / 914 链路 / 1406 OD（论文 §5.2）。
  - 数据源：`https://github.com/bstabler/TransportationNetworks`（`SiouxFalls`、`Anaheim`）。
- **候选 vertiport 配置**（YAML/JSON）：候选节点坐标/近邻道路节点、$P$、$B$、$C$，
  $V_h=300\ \mathrm{km/h}$、$V_v=50\ \mathrm{km/h}$、$V_f=60\ \mathrm{km/h}$、$d=10\ \mathrm m$、$h_{lp}$、$L_{\hat a}$、$L_{\tilde a}$。
- **算法参数**：初始样本数 $m$、评估预算（Sioux Falls=75）、IGP `rg_target=1e-12`、多重复次数。

### 6.2 输出

- 最优方案 $x^*=(z^*,c^*)$：vertiport 位置与容量、TTT。
- 指标：Auto share / LAC share（按路径是否含 $A_Q/A_C$ 分类）、V/C 分布（道路链路 $v_a/cap_a$ 分箱 0–0.5/0.5–1/1–1.5/1.5–2/>2）。
- 收敛曲线：MI-BO 的"当前最优 TTT vs 评估次数"（对照论文 Figure 6/9）；下层 IGP 的 RG 轨迹。
- 与无 UAM 基线对比表（对照论文 Table 2/3）。

---

## 7. 关键实现细节与数值注意事项

1. **排队阻抗数值稳定**（最重要）：
   - $\rho=\lambda/\mu$，$\lambda=\max(v_a, 10^{-12})$ 防除零；$\mu=V_v/d>0$。
   - $\rho\to1$ 处式 (2)(3) 出现 $0/0$：用极限 $P_0\to1/(c_a+1)$、$L_q\to c_a/2$（或 $|1-\rho|<\epsilon$ 时用一阶展开）。
   - $\rho>1$（有限容量 $c_a$）公式仍成立（有限状态马氏链平稳分布总存在），无需截断。
   - $c_a$ 非整数时 $\rho^{c_a+1}$ 用 `exp((c_a+1)·log(ρ))`；容量作为连续变量直接代入。
2. **排队导数 $dt_a/d\lambda_a$**：解析求导较繁，**建议解析推导后与中心差分（步长 $10^{-6}$）交叉验证**；
   或直接采用高精度数值导数（排队链路数 = |P| 很小，代价可忽略）。供贪心二次近似（$s_h$）与 GP 步长使用。
3. **$s_h=0$ 边界**：飞行/疏散链路 $t'=0$，全空路径可能 $s_h=0$——沿用 `IGP` 的 `S_FLOOR` 防御；道路 BPR 在 $x=0,\beta=4$ 时 $t'=0$ 亦已处理。
4. **下层目标 Z(f) 与上层目标 TTT 的区别**：下层用 Beckmann 积分目标（道路/排队为积分，飞行/疏散为线性）求 UE 流量；上层回报**只用 $\sum v_a t_a$（TTT）**，二者不能混用。`IGP` 已把 TST 作为 $v_a t_a$ 之和，直接可作上层目标。
5. **复合网络拓扑随 $z$ 变化**：每次评估重建 `Network` 与 CSR；OD 需求不变；RG 的 SPTT 用复合网络最短路树（含飞行链路）。
6. **GP 数值稳定**：拟合前对 $F_m$ 标准化；$K_m^{-1}$ 用 `cholesky` 分解 + 白噪声 jitter（如 $10^{-9}$）；样本重复时去重。
7. **EI 的 $\sigma_f=0$ 情形**：$x$ 恰为已评样本时 $\sigma_f=0$，EI=0，直接跳过（B&B 松弛优化时避免除零）。
8. **初始采样可行性**：随机生成满足 $\sum z_p\le B$、$c_a\in(c_l,C]$ 的样本；保证至少一个样本含 UAM 与纯道路两种形态，避免 GP 退化。
9. **确定性/可复现**：固定 RNG 种子；多重复（原论文 Anaheim 跑 10 次）取均值/方差画 Figure 9。
10. **规模**：Python 对 Sioux Falls（24/76/576 OD）与 Anaheim（416/914/1406 OD）完全够用；`IGP` 每求值秒级。若上 PRISM/Chicago 级需 C++（见 §9 M6）。

---

## 8. 验证与测试计划

1. **单元测试（link_cost）**
   - M/M/1/C：$\rho\to0$ 时 $t\to0$；$\rho=1$ 用极限值；取 $(c_a,\lambda,\mu)$ 手算 $P_0,L_q,t$ 对照。
   - BPR 与排队导数：中心差分相对误差 $<10^{-6}$。
   - 飞行/疏散链路：$t$ 常数、$t'=0$。
2. **单元测试（GP/EI）**
   - $\kappa_z$ Hamming 距离、$\kappa_c$ RBF、$\kappa_{MI}$ 乘法核；预测 $\mu,\sigma^2$ 与手算小矩阵对照。
   - EI 闭式与数值积分对照；$\sigma_f\to0$ 时 EI=0。
3. **单元测试（B&B）**：小规模（$|P|=2\sim3$）整数 EI 与穷举最大值一致。
4. **下层 LAC-NE 独立验证**：固定 $(z,c)$，验证
   - RG 降到 $10^{-12}$；
   - 链路流量守恒、路径流量守恒（$\sum_k f_k^w=q_w$）；
   - Wardrop：被用路径等代价、未用路径代价更高；
   - 与纯道路 UE（`subproblem="greedy"`）在"无 UAM"退化场景下 TTT 一致。
5. **端到端复现（关键验收）**
   - Sioux Falls 无 UAM TTT=45114.9；2/3/4/5 vertiports 时 TTT≈41199.7/39891.0/38371.0/36524.9，share 与容量趋势吻合（表 2）。
   - Anaheim 需求缩放 0.4–1.6，复现表 3 的 TTT（如 demand=1.0：无 UAM 1418920 / 有 UAM 1194270）与 V/C 分布趋势。
   - MI-BO 收敛曲线（Figure 6）与 MI-BO vs GA（Figure 9）定性复现：MI-BO 更快更稳。
6. **鲁棒性**：多重复实验方差、不同 $m$/预算敏感性。

---

## 9. 实现路线图（里程碑）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| M0 | 拉取 Sioux Falls / Anaheim TNTP 数据；用 `IGP` 跑通纯道路 UE | 无 UAM 基线 TTT 与论文/参考解一致 |
| M1 | `link_cost.py`：四类阻抗 + 导数 + 单测 | M/M/1/C 与 BPR 手算对照通过，导数差分 <1e-6 |
| M2 | `composite_network.py` + `lacne_solver.py`：给定 $(z,c)$ 求解 LAC-NE | RG<1e-12，守恒/KKT/Wardrop 检验通过 |
| M3 | `gp/`：混合整数核、MLE、预测、EI | 小矩阵手算对照通过 |
| M4 | `bb.py`：Algorithm 1 分支定界 | 小规模与穷举一致 |
| M5 | `mibo.py` + `run.py`：端到端 Sioux Falls | 复现表 2（TTT 与 share 趋势），收敛曲线符合 Figure 6 |
| M6 | Anaheim + GA 对照 + 敏感性/多重复 | 复现表 3 趋势与 Figure 9 定性结论 |
| M7（可选） | C++/Cython 移植热点（IGP 最短路/贪心 + 排队代价） | 支撑 PRISM/Chicago 规模 |

---

## 10. 参考

- 原论文：`paper/Zhang et al. - 2025 - A composite transportation network design problem with land-air coordinated operations.pdf`
- 下层算法：`IGP/IMPLEMENTATION_PLAN.md`、`IGP/README.md`（Xie, Nie & Liu 2018 贪心路径算法）
- 测试网络：`https://github.com/bstabler/TransportationNetworks`（Sioux Falls / Anaheim）
- 相关方法：MI-BO 混合整数核（Baptista & Poloczek 2018）、EI（Jones et al. 1998 / Zhan & Xing 2020）、M/M/1/C 点排队（Cho et al. 2020; Liu et al. 2022）
