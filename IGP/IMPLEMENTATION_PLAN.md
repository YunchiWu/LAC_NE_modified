# 贪心路径交通分配算法实现规划

> 依据论文：Jun Xie, Yu (Marco) Nie, Xiaobo Liu. *A Greedy Path-Based Algorithm for Traffic Assignment*. Transportation Research Record, 2018, 2672(48): 36–44.
> 本文档给出该算法（静态用户均衡 UE-TAP 的贪心路径算法）的完整实现设计：数学建模、算法流程、数据结构、模块划分、关键数值细节、验证方案与路线图。默认技术栈 **Python 3 + NumPy/SciPy**（先保证正确性），架构上预留 C++ 移植接口。

---

## 0. 一句话概括

在**按 OD 对做 Gauss-Seidel 分解**的经典路径算法框架上，用**贪心法**精确求解每个 OD 对上的二次近似子问题（替代 GP/PG），并引入**“内循环 + 智能调度”**：优先、多次调整代价差（$D_{rs}$）大的 OD 对，跳过已收敛的 OD 对。相比 TAPAS/iTAPAS 等 bush 类算法实现更简单、收敛更快。

---

## 1. 问题定义与数学建模

### 1.1 网络与符号

- 有向网络 $G(N, A)$：$N$ 节点集，$A$ 链路集（假设强连通）。
- 起点集 $R$，终点集 $S$；OD 对 $(r,s)$ 需求 $d_{rs}$。
- $H_{rs}$：$(r,s)$ 之间的简单路径集合；$f_h$：路径 $h$ 的流量。
- 链路 $(i,j)$ 的旅行费用 $t_{ij}(x_{ij})$：**可分、严格正、单调递增**（用 BPR 函数）。
- 关联矩阵 $\delta^h_{ij}=1$ 表示路径 $h$ 经过链路 $(i,j)$，否则为 0。

### 1.2 Beckmann 形式（UE-TAP 的凸规划）

$$
\min z(f)=\sum_{(i,j)\in A}\int_0^{x_{ij}} t_{ij}(w)\,dw \tag{1}
$$

$$
\text{s.t.}\quad \sum_{h\in H_{rs}} f_h = d_{rs},\ \forall r,s \tag{2}
\qquad f_h\ge 0,\ \forall h \tag{3}
\qquad x_{ij}=\sum_{r}\sum_{s}\sum_{h\in H_{rs}}\delta^h_{ij}f_h,\ \forall(i,j) \tag{4}
$$

链路流解唯一，路径流解不唯一。

### 1.3 按 OD 对 Gauss-Seidel 分解（单 OD 子问题）

固定其它 OD 对流量 $x^0_{ij}=\sum_{o\ne r}\sum_{q\ne s}x^{oq}_{ij}$，单个 $(r,s)$ 子问题：

$$
\min z_{rs}(f)=\sum_{(i,j)\in A}\int_0^{x^0_{ij}+x^{rs}_{ij}} t_{ij}(w)\,dw \tag{5}
$$

$$
\text{s.t.}\quad \sum_{h\in H_{rs}} f_h=d_{rs},\quad f_h\ge 0,\quad x^{rs}_{ij}=\sum_{h\in H_{rs}}\delta^h_{ij}f_h \tag{6,7,8}
$$

### 1.4 二次近似子问题（对角 Hessian 的 Taylor 展开）

在当前点 $g$ 处对 (5) 做二阶 Taylor 展开并去掉常数项，得**可分二次规划**：

$$
\min \hat z_{rs}(f)=\sum_{h\in H_{rs}}\Big[(v_h^g - s_h^g g_h)f_h + \tfrac12 s_h^g f_h^2\Big] \tag{10}
$$

$$
\text{s.t.}\quad \sum_{h\in H_{rs}}f_h=d_{rs},\quad f_h\ge 0 \tag{11,12}
$$

其中路径代价与“代价对自身流量的二阶导数”（对角项）为：

$$
v_h^g=\sum_{(i,j)\in h} t_{ij}(x^g_{ij}) \tag{16}
\qquad
s_h^g=\sum_{(i,j)\in h} \frac{\partial t_{ij}(x^g_{ij})}{\partial x^g_{ij}} \tag{17}
$$

> 含义：$v_h^g$ 是路径 $h$ 在当前链路流下的总代价；$s_h^g$ 是该路径所有链路代价导数之和（衡量增加该路径流量时自身代价上升的斜率）。注意此处忽略交叉二阶项（对角近似）。

### 1.5 KKT 条件与贪心求解

令常数 $c_h^g = v_h^g - s_h^g g_h$（式 18）。记 $\hat H_{rs}\subseteq H_{rs}$ 为“有流量”的路径集合，则 KKT 给出：

$$
c_h^g + s_h^g f_h = \bar w_{rs},\ \forall h\in\hat H_{rs}
\qquad \sum_{h\in\hat H_{rs}} f_h = d_{rs} \tag{19,20}
$$

解得均衡路径费用与路径流量：

$$
\bar w_{rs}=\frac{d_{rs}+\sum_{h\in\hat H_{rs}} c_h^g/s_h^g}{\sum_{h\in\hat H_{rs}} 1/s_h^g} \tag{21}
\qquad
f_h=\frac{\bar w_{rs}-c_h^g}{s_h^g},\ \forall h\in\hat H_{rs} \tag{22}
$$

$H_{rs}\setminus\hat H_{rs}$ 的路径流量置 0。**确定 $\hat H_{rs}$ 的贪心过程**见 Algorithm 1。

> 关键正确性性质：若按 $c_h$ 升序，第 $h$ 条路径“入集合”的判据 $c_h < \bar w_{\text{加入前}}$ 恰好等价于加入后 $\bar w_{\text{加入后}} > c_h$（即 $f_h>0$），因此贪心法得到的是二次子问题 (10–12) 的**精确解**，不会出现负流量。

---

## 2. 算法核心

### 2.1 Algorithm 1：单 OD 贪心求解器（求解 (10–12)）

**输入**：当前路径集 $H_{rs}$、当前路径流 $g_h$、当前链路流/代价/导数值。
**输出**：更新后的路径流 $\{f_h\}$ 及受影响的链路流/代价/导数。

```
初始化:
  for h in H_rs:
      v_h = Σ_{ (i,j)∈h } t_ij(x)        # 式(16) 路径代价
      s_h = Σ_{ (i,j)∈h } dt_ij/dx       # 式(17) 导数和
      c_h = v_h - s_h * g_h              # 式(18)
  将 H_rs 按 c_h 升序排序 -> {1,2,3,...}, c_1<c_2<c_3<...
  B = 1/(s_1 * d_rs)
  C = c_1/(s_1 * d_rs)
  w = (1.0 + C)/B                        # 初值 = c_1 + s_1*d_rs（全流量都在路径1）
  h = 2
  used = {1}

主循环:
  while h <= |H_rs| and c_h < w:         # 判据用“加入前”的 w
      C += c_h/(s_h * d_rs)
      B += 1/(s_h * d_rs)
      w  = (1.0 + C)/B                   # 式(21) 的增量形式
      used.insert(h)
      h += 1

流量更新:
  for h in used:         f_h = (w - c_h)/s_h     # 式(22)
  for h in H_rs \ used:  f_h = 0
  for h in H_rs:
      if f_h != g_h:
          for (i,j) in h:
              x_ij += (f_h - g_h)                # 增量更新链路流
              # 重算该链路 t_ij(x) 与 dt_ij/dx
  删除零流量路径:  H_rs = used
```

### 2.2 Algorithm 2：主算法（主循环 + 内循环 + 智能调度）

```
初始化:
  for each OD (r,s):
      h_hat = 自由流下的最短路(r,s)
      把 d_rs 全部分配到 h_hat；H_rs = {h_hat}
  由路径流累加链路流 x_ij = Σ Σ Σ f_h δ^h_ij
  计算所有链路 t_ij(x) 与 dt_ij/dx

主循环 (列生成 + 单次流量调整):
  for each r in R:
      用当前链路代价计算 r 到所有终点 S_r 的最短路树
      for each s in S_r:
          h_hat = 从树恢复的 r->s 最短路
          if h_hat not in H_rs:  H_rs.insert(h_hat)      # 列生成
          else:                  (丢弃该临时路径)
          对 H_rs 执行 Algorithm 1（单次流量调整）

内循环 (智能调度，重复流量调整):
  I = 0; MaxI = 1000; FC = 0
  while I < MaxI:
      I += 1; FC = 0
      for each OD (r,s):
          if I % 100 == 0:
              D_rs = max{v_h} - min{v_h} over h∈H_rs    # 代价差（滞后估计）
          if D_rs >= RG_{k-1} / 2.0:                    # 未达本次收敛精度才调整
              FC += 1
              对 H_rs 执行 Algorithm 1
              更新相关链路 x_ij, t_ij, dt_ij/dx
      if FC == 0: break                                 # 所有 OD 都达到精度

收敛检查:
  计算 RG；若 RG <= 目标精度则停止，否则回到主循环
```

### 2.3 相对间隙 RG（收敛指标，式 23）

$$
RG = 1 - \frac{\sum_{rs} m_{rs} d_{rs}}{\sum_{(i,j)\in A} x_{ij}t_{ij}(x_{ij})}
   = \frac{TST - SPTT}{TST} \tag{23}
$$

- $TST=\sum_{(i,j)}x_{ij}t_{ij}$：总系统旅行时间。
- $SPTT=\sum_{rs}m_{rs}d_{rs}$：最短路径旅行时间之和，$m_{rs}$ 为当前链路代价下 $(r,s)$ 的最短路径代价（来自主循环的最短路树）。
- 论文以 $RG\le 10^{-14}$ 为停止条件（Python 原型可放宽到 $10^{-8}\sim 10^{-10}$）。

---

## 3. 数据模型与数据结构

### 3.1 网络

- 前向星（forward star）邻接：`head[]`, `tail[]`, `from_node_first_edge[]`；或 CSR 稀疏矩阵（便于 `scipy.sparse.csgraph` 最短路）。
- 链路属性数组（NumPy 数组，`float64`）：自由流时间 `t0`、容量 `cap`、BPR 参数 `alpha`、`beta`。
- 运行时状态数组：链路流 `x`、链路代价 `t`、导数 `dt`。

### 3.2 OD 需求

- 稀疏存储（大网络 OD 数量达百万级，不能开满 $|R|\times|S|$ 矩阵）：`list[(r, s, d)]` 或按 `r` 分组的字典。

### 3.3 路径存储（内存关键点）

- 每个 OD 对维护一个路径列表 `H_rs`；每条路径 = **链路下标数组**（`int32`）。
- 论文 C++ 实现在 Chicago Regional（~200 万路径）内存 ~490 MB。**Python 对象开销大**，建议：
  - 用小网络（Chicago Sketch 933 节点 / 126 万 OD）先做正确性验证；
  - 路径用 `numpy.ndarray(dtype=int32)` 或 `array('i')` 紧凑存储；
  - 若需跑 PRISM/Chicago Regional，预留 C++/Cython 扩展接口（本规划第 8 节）。
- 每条路径缓存 `v_h`、`s_h`、`c_h`、`f_h` 用于 Algorithm 1 排序与 $D_{rs}$ 计算。

### 3.4 最短路

- 每主循环、每个起点 `r` 一次最短路树。Python 循环版 Dijkstra 较慢，推荐：
  - `scipy.sparse.csgraph.dijkstra(csgraph, indices=origins, return_predecessors=True)` 一次得到所有起点的树（C 实现，向量化）；
  - 或自写堆优化 Dijkstra，供 C++ 移植对照。

---

## 4. 模块划分

```
igp/
  network.py      # 网络数据模型 + TNTP 解析 + BPR 代价/导数
  shortest_path.py# 最短路树（scipy Dijkstra 封装 / 自写堆 Dijkstra）
  od_demand.py    # 需求稀疏结构
  paths.py        # 路径容器：增删路径、路径代价 v_h、导数和 s_h、c_h
  greedy.py       # Algorithm 1 单 OD 贪心求解器
  gap.py          # RG 计算
  solver.py       # Algorithm 2 主循环 + 内循环 + 智能调度
  run.py          # CLI 入口：读数据 -> 求解 -> 输出流量与 RG 曲线
  baseline.py     # Frank-Wolfe 基线（验证正确性用）
  tests/          # 单元测试 + 小网络端到端测试
```

各模块职责与依赖关系：

| 模块 | 输入 | 输出 | 依赖 |
|---|---|---|---|
| `network` | TNTP 文件 | 邻接、`t0/cap/alpha/beta`、`x/t/dt` | — |
| `shortest_path` | 链路代价 | 前驱树 + 距离 | `network` |
| `paths` | 前驱树、OD | 路径数组、`v/s/c/f` | `network` |
| `greedy` | `H_rs`、链路状态 | 更新后的路径流与链路状态 | `paths`,`network` |
| `gap` | 最短路代价、`x,t` | RG | `network` |
| `solver` | 网络 + 需求 | 收敛轨迹、最终流 | 以上全部 |

---

## 5. 输入输出

### 5.1 输入：TNTP 网络格式（Bar-Gera 标准测试网）

论文 4 个测试网络来自 `https://github.com/bstabler/TransportationNetworks`（原 `bgu.ac.il/~bargera/tntp`），目录内通常含：

- `*_net.tntp`：节点/链路/容量/自由流时间/`alpha`/`beta`；
- `*_trips.tntp`：OD 需求；
- `*_flow.tntp`：参考链路流（**验证用**，其 TST 即期望最优值）。

需实现 TNTP 解析器，兼容不同列头与分隔符。

### 5.2 输出

- 每条链路最终流量 `x_ij`、代价 `t_ij`（写 `_flow` 风格文件，便于和参考解比对）；
- 收敛轨迹：`(主循环迭代 k, CPU 时间, RG)`，用于画收敛曲线（对照论文 Figure 1）；
- 统计：生成路径数、内存占用、各阶段耗时。

---

## 6. 关键实现细节与数值注意事项

1. **BPR 代价与导数**
   $$
   t_{ij}(x)=t0_{ij}\left(1+\alpha_{ij}\left(\frac{x}{cap_{ij}}\right)^{\beta_{ij}}\right),
   \qquad
   t'_{ij}(x)=t0_{ij}\,\alpha_{ij}\,\beta_{ij}\,\frac{x^{\beta_{ij}-1}}{cap_{ij}^{\beta_{ij}}}
   $$
   默认 $\alpha=0.15,\ \beta=4$。对 $x=0,\ \beta=4$：$t'=0$。

2. **$s_h=0$ 边界**：$s_h=\sum_{(i,j)\in h}t'_{ij}$。若路径全部链路流量为 0，则 $s_h=0$，式 (21)(22) 会除零。处理：$s_h\leftarrow\max(s_h,\varepsilon)$（如 $\varepsilon=10^{-12}$），并保证链路容量>0 防除零。初始化 AON 后所有被使用链路 $x>0$，实际很少触发，但必须防御。

3. **链路增量更新**：Algorithm 1 第 26–28 行只在“流量发生变化的路径”所经过的链路上做 $x_{ij}\mathrel{+}=(f_h-g_h)$ 并重算 $t,t'$，避免全网络扫描。注意 $v_h,s_h,c_h$ 基于**调整前**的链路代价计算（二次近似点），调整后链路代价更新供后续子问题/内循环使用——这是 GP/PG 类算法的标准做法，需保持一致。

4. **$D_{rs}$ 的滞后更新**：内循环中 $D_{rs}=\max v_h-\min v_h$ 仅在 `I % 100 == 0` 时重算（其余迭代沿用上次值），以省去每轮全 OD 的路径代价扫描。代价差用当前链路代价下的 $v_h$。

5. **跳过已收敛 OD**：内循环阈值 $D_{rs}\ge RG_{k-1}/2.0$（$RG_{k-1}$ 为上一主循环的相对间隙）。未达阈值的 OD 不做 Algorithm 1，把算力集中到“还能显著降低目标”的 OD。

6. **`FC=0` 提前退出内循环**：若某轮内循环没有任何 OD 被调整（全部达标），跳出内循环进入下一主循环（列生成）。

7. **内循环上限**：`MaxI=1000`。

8. **列生成去重**：主循环中若新最短路已在 $H_{rs}$，不再重复加入（避免路径集合膨胀）。

9. **零流量路径删除**：Algorithm 1 末尾 `H_rs = used`，把流量为 0 的路径移出集合，控制内存。

10. **数值精度**：全程 `float64`；RG 目标 $10^{-14}$ 需高精度链路流更新，Python 原型建议 $10^{-8}\sim10^{-10}$，性能与数值稳定性更均衡。

---

## 7. 正确性验证与测试计划

1. **单元测试**
   - BPR 代价/导数在 $x=0$、$x=cap$ 等点的手算对照。
   - Algorithm 1 在 2–3 条路径的小例子：手算 $\bar w$、$f_h$，断言 KKT 满足（$c_h+s_hf_h$ 对有流路径相等、无流路径 $c_h\ge\bar w$）、流量非负、$\sum f_h=d_{rs}$。

2. **基线对照**
   - 实现 Frank-Wolfe（`baseline.py`），小网络上对比最终 TST 与 RG，确认收敛到同一 UE 解（链路流唯一）。

3. **参考解验证**
   - 用 Chicago Sketch 的 `*_flow.tntp`：算法收敛后链路流与参考流逐链比对（相对误差 $<10^{-4}$），TST 逼近已知最优（~1,730,000 量级）。

4. **守恒性检查**
   - 对每个中间节点断言 $\sum x_{进}-\sum x_{出}=0$；对起终点分别等于发出/到达需求。

5. **收敛行为**
   - 记录 RG 随迭代/时间下降，验证内循环使 RG 下降明显快于“去内循环版”（论文称 s-greedy），复现论文 Figure 2 的定性结论。

6. **压力测试**
   - Chicago Sketch（933/2950/126 万 OD）→ PRISM（14639/33937/61 万 OD）→ 视内存决定是否上 Chicago Regional / Philadelphia（13000 节点级、300 万 OD）。

---

## 8. 实现路线图（里程碑）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| M0 | TNTP 解析 + 网络模型 + BPR 代价/导数 + 单元测试 | 正确读入 Chicago Sketch，链路属性一致 |
| M1 | 最短路（scipy Dijkstra）+ AON 初始化 + RG 计算 + Frank-Wolfe 基线 | 小网络 FW 能收敛，RG 单调下降 |
| M2 | Algorithm 1 贪心求解器 + 单测 | 小例 KKT 精确成立 |
| M3 | Algorithm 2 主循环（仅列生成 + 单次调整，即 s-greedy） | 小网络收敛到参考解 |
| M4 | Algorithm 2 内循环 + 智能调度（$D_{rs}$、MaxI、FC、RG/2 阈值） | Chicago Sketch 收敛，RG 达 $10^{-8}$ |
| M5 | 完整验证 + 收敛曲线 + 与 FW/参考解对比 | 复现论文核心结论（贪心 > TAPAS/iTAPAS 的定性优势） |
| M6（可选） | C++/Cython 移植热点（最短路、贪心、链路更新） | 支撑 PRISM/Chicago Regional 规模 |

---

## 9. 参考与资源

- 论文原文：`paper/Xie et al. - 2018 - A greedy path-based algorithm for traffic assignment.pdf`
- 测试网络：`https://github.com/bstabler/TransportationNetworks`（含 Chicago Sketch / PRISM / Chicago Regional / Philadelphia 及参考流）。
- 关联方法：LUCE（Gentile, 2014，贪心求解节点子问题的出处）、GP（Jayakrishnan et al., 1994）、TAPAS/iTAPAS（Bar-Gera, 2010; Xie & Xie, 2016）。
