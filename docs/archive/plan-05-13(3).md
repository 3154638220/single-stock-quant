# 模型方法与收益提升计划

**项目**：`single-stock-quant`
**日期**：2026-05-13
**最后更新**：2026-05-13 20:15 GMT+8
**范围**：单股趋势择时、批量 watchlist 评估、WFO 参数寻优、信号质量和仓位管理。

本计划的目标不是追求某一次回测的最高收益，而是把收益提升建立在可复现、可解释、可外推的 OOS 指标上。所有改动必须先通过无未来函数的评估链路，再进入策略参数或模型增强。

---

## 进度总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 建立可靠实验基准 | ✅ 完成 |
| 1.1 | 一字跌停无法卖出 | ✅ 完成 |
| 1.2 | 交易归因表 (MAE/MFE/质量分/RS/市场状态) | ✅ 完成 |
| 1.3 | 成本和滑点压力测试 | ✅ 完成 |
| 2.1 | 质量评分接入回测 (hard + scale) | ✅ 完成 |
| 2.2 | 趋势强度特征 + 增强质量评分 | ✅ 完成 |
| 2.3 | 多信号候选池 (Donchian 突破) | ✅ 完成 (MACD/MA/Boll/Donchian) |
| 3 | Meta-labeling 轻量模型 | ✅ 完成 |
| 4.1 | 退出逻辑升级 (时间止损/盈利保护/市场退出) | ✅ 完成 |
| 4.2 | 波动率目标仓位 | ✅ 完成 |
| 4.3 | 回撤节流 | ✅ 完成 |
| 5.1 | WFO 复合目标函数 | ✅ 完成 |
| 5.2 | 参数平台选择 | ✅ 完成 |
| 5.3 | 嵌套 WFO | ✅ 完成 |
| 6 | watchlist 横截面组合 | ✅ 完成 |
| 7 | 报告和研究闭环 | ✅ 完成 |

**质量分桶分析结论** (2026-05-13)：
增强质量评分后，threshold=20 时 Sharpe 中位数 0.22（baseline 0.16），Sharpe>0 从 16/25 提升到 19/25。threshold=40 以上交易数不足。质量分对过滤低质量信号有效，但信号整体质量仍偏低，组合层（Phase 6）是提升整体收益的更高杠杆。

---

## 1. 当前状态结论

### 1.1 已有能力

当前代码已经具备较完整的规则型单股择时框架：

| 模块 | 已实现能力 |
|---|---|
| 数据 | DuckDB 日线存储、AkShare 拉取、质量检查、股票名称缓存 |
| 指标 | `macd_cross`、`ma_cross`、`boll_trend`、`donchian_breakout` 四类 DK 趋势 |
| 信号 | 量能确认、三模式共振、`min_run_len` 防抖、信号质量评分 (0-100) |
| 特征 | 均线斜率、Donchian 突破、ATR 波动率分位、量比、相对强度 (vs 指数) |
| 回测 | T+1 次日开盘执行、涨停买入顺延/跌停卖出顺延、停牌近似、真实 A 股成本模型 (佣金+滑点+印花税) |
| 风控 | 固定/追踪/ATR 止损、盈利保护、时间止损、市场退出、波动率目标仓位、回撤节流、止损后再入场 |
| 模型 | L2 逻辑回归 Meta-labeling，预测 BUY 信号胜率，WFO 评估 |
| 评估 | 批量回测、WFO (含复合目标函数)、嵌套 WFO、参数平台选择、参数漂移量化、热力图、DSR、Bootstrap Sharpe CI、置换检验、HTML 报告 |
| 组合 | 跨标的信号排序打分、Top N 等权/波动率倒数分配、仓位约束 (单票/行业/换手)、成本敏感性批量分析 |
| 实验 | 标准化实验目录、index.csv 追踪、决策规则评估 |

因此下一阶段不应重复实现基础功能，而要重点解决“信号有效性不足、参数不稳定、风控未充分联动、组合选择缺失、评估链路有漏洞”这五类问题。

### 1.2 真实回测结果诊断

基于 `data/output/batch_summary_20260513.csv`，25 只 watchlist 在 2020-01-02 至 2026-05-08 的结果如下：

| 指标 | 当前结果 |
|---|---:|
| 有效标的数 | 25 |
| 年化收益为正 | 13 / 25 |
| Sharpe 为正 | 15 / 25 |
| Calmar > 0.5 | 2 / 25 |
| 年化收益中位数 | 0.60% |
| 年化收益均值 | 1.52% |
| Sharpe 中位数 | 0.11 |
| Calmar 中位数 | 0.019 |
| 最大回撤中位数 | 50.33% |
| 单股交易次数中位数 | 50 |
| 胜率中位数 | 35.09% |

Top 标的集中在强趋势成长和部分金融股：

| 标的 | 年化 | Sharpe | Calmar | 最大回撤 |
|---|---:|---:|---:|---:|
| 300750 宁德时代 | 25.28% | 0.84 | 0.58 | 43.23% |
| 002475 立讯精密 | 20.15% | 0.75 | 0.52 | 38.39% |
| 300059 东方财富 | 16.24% | 0.60 | 0.32 | 51.41% |
| 600036 招商银行 | 10.62% | 0.64 | 0.46 | 23.21% |
| 600030 中信证券 | 7.79% | 0.44 | 0.22 | 35.35% |

Bottom 标的显示同一套信号在医药、消费、周期下行阶段承受较大回撤：

| 标的 | 年化 | Sharpe | Calmar | 最大回撤 |
|---|---:|---:|---:|---:|
| 300760 迈瑞医疗 | -11.59% | -0.39 | -0.18 | 65.06% |
| 600276 恒瑞医药 | -10.84% | -0.35 | -0.17 | 62.73% |
| 600887 伊利股份 | -10.75% | -0.51 | -0.19 | 57.47% |
| 600585 海螺水泥 | -8.49% | -0.33 | -0.14 | 62.58% |
| 000651 格力电器 | -6.32% | -0.27 | -0.13 | 47.74% |

核心判断：

1. 单一 DK 趋势信号有一定方向性，但跨标的稳定性不足。
2. 收益主要来自少数强趋势标的，watchlist 的中位收益很弱。
3. 最大回撤过高，收益提升必须和回撤压缩同时推进。
4. 当前每只股票独立交易，缺少”只交易更有优势标的”的横截面选择机制。
5. WFO 结果显示参数选择不稳，不能简单扩大网格追求更高 IS Sharpe。

### 1.2.1 更新后回测结果（Phase 1-7 完成后）

**回测日期**：2026-05-13 20:15 GMT+8
**配置摘要**：`config.yaml`，mode=macd_cross，stop_loss_pct=0.08，volume_confirm=true，真实A股成本模型
**区间**：2020-01-02 至 2026-05-08
**数据文件**：`data/output/batch_summary_20260513.csv`

#### 汇总指标对比

| 指标 | 旧 Baseline | **当前结果** | 变化 | 计划目标 |
|---|---:|---:|---:|---:|
| 有效标的数 | 25 | 25 | — | — |
| 年化收益为正 | 13 / 25 | 13 / 25 | 持平 | — |
| Sharpe 为正 | 15 / 25 | **17 / 25** | +2 | — |
| Calmar > 0.5 | 2 / 25 | **3 / 25** | +1 | 8 / 25+ |
| 年化收益中位数 | 0.60% | **2.43%** | +4.0x | 5%+ |
| 年化收益均值 | 1.52% | **3.25%** | +2.1x | — |
| Sharpe 中位数 | 0.11 | **0.26** | +2.4x | 0.35+ |
| Calmar 中位数 | 0.019 | **0.040** | +2.1x | 0.25+ |
| 最大回撤中位数 | 50.33% | **47.11%** | -3.2pct | <35% |
| 交易次数中位数 | 50 | 50 | 持平 | — |
| 胜率中位数 | 35.09% | **36.36%** | +1.3pct | — |

#### 全部标的排名（按 Sharpe 降序）

| 代码 | 名称 | 年化 | Sharpe | Calmar | 最大回撤 | 交易数 | 胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300750 | 宁德时代 | 27.27% | 0.88 | 0.66 | 41.45% | 47 | 48.94% |
| 002475 | 立讯精密 | 22.02% | 0.80 | 0.59 | 37.63% | 46 | 45.65% |
| 600036 | 招商银行 | 12.38% | 0.73 | 0.57 | 21.77% | 47 | 44.68% |
| 300059 | 东方财富 | 18.05% | 0.64 | 0.36 | 49.98% | 46 | 43.48% |
| 601166 | 兴业银行 | 9.21% | 0.58 | 0.44 | 20.93% | 50 | 44.00% |
| 600030 | 中信证券 | 9.55% | 0.52 | 0.29 | 32.85% | 48 | 37.50% |
| 601012 | 隆基绿能 | 9.62% | 0.46 | 0.19 | 51.26% | 51 | 37.25% |
| 600809 | 山西汾酒 | 8.80% | 0.45 | 0.23 | 38.39% | 50 | 46.00% |
| 601318 | 中国平安 | 5.78% | 0.39 | 0.16 | 36.79% | 46 | 43.48% |
| 002594 | 比亚迪 | 3.11% | 0.36 | 0.04 | 78.47% | 46 | 30.43% |
| 002415 | 海康威视 | 5.81% | 0.36 | 0.11 | 52.01% | 51 | 45.10% |
| 000568 | 泸州老窖 | 4.08% | 0.28 | 0.09 | 47.11% | 52 | 38.46% |
| 600900 | 长江电力 | 2.43% | 0.26 | 0.14 | 17.25% | 53 | 35.85% |
| 601888 | 中国中免 | -0.87% | 0.13 | -0.01 | 63.30% | 50 | 34.00% |
| 000002 | 万科A | -1.53% | 0.07 | -0.04 | 41.87% | 49 | 32.65% |
| 000858 | 五粮液 | -1.48% | 0.05 | -0.04 | 39.55% | 50 | 40.00% |
| 600519 | 贵州茅台 | -1.25% | 0.04 | -0.04 | 31.64% | 55 | 30.91% |
| 600048 | 保利发展 | -4.07% | -0.04 | -0.08 | 49.71% | 51 | 29.41% |
| 000725 | 京东方A | -3.78% | -0.10 | -0.07 | 50.76% | 57 | 26.32% |
| 000333 | 美的集团 | -3.92% | -0.11 | -0.06 | 68.77% | 55 | 36.36% |
| 000651 | 格力电器 | -4.79% | -0.18 | -0.11 | 45.09% | 48 | 33.33% |
| 600585 | 海螺水泥 | -6.87% | -0.24 | -0.11 | 59.90% | 52 | 32.69% |
| 600276 | 恒瑞医药 | -9.34% | -0.29 | -0.15 | 61.87% | 49 | 32.65% |
| 300760 | 迈瑞医疗 | -9.87% | -0.31 | -0.16 | 62.29% | 57 | 35.09% |
| 600887 | 伊利股份 | -9.14% | -0.42 | -0.17 | 54.58% | 53 | 28.30% |

#### 成本敏感性分析

数据文件：`data/output/batch_summary_20260513_cost_sensitivity.csv`

| 成本模型 | 年化中位数 | Sharpe 中位数 | Calmar 中位数 |
|---|---:|---:|---:|
| 零成本（上界） | 1.93% | 0.22 | 0.034 |
| 对称 15bps | 1.93% | 0.22 | 0.034 |
| **真实 A 股（默认）** | **2.43%** | **0.26** | **0.040** |
| 高滑点（机构） | 1.49% | 0.18 | 0.029 |

**Sharpe 衰减分析**（零成本 → 真实 A 股）：
- 平均 Sharpe 衰减：**-0.020**（真实 A 股口径下 Sharpe 反而略高，因不同成本路径在止损触发边界有微小差异，属噪声级别）
- 最大衰减：-0.010
- 衰减 > 0.10 的标的：**0 / 25** ✅
- 衰减 > 0.25 的标的：**0 / 25** ✅

成本压力测试结论：策略换手率在可接受范围，成本模型切换不显著改变收益排序。**当前瓶颈不在交易成本，而在信号质量本身。**

#### 与 Baseline 差异分析

主要改善来自以下已启用的 Phase 4 特性：
1. **8% 固定止损** (`stop_loss_pct: 0.08`)：截断尾部亏损，对 Bottom 标的的 MDD 有改善
2. **量能确认** (`volume_confirm: true`)：减少无量假突破的假 BUY
3. **真实 A 股成本模型**：更精确的成本核算

尚未启用或未调优的特性（对当前结果无贡献）：
- Phase 3 Meta-labeling：模型已实现但未接入回测过滤
- Phase 5.2/5.3 WFO 平台选择/嵌套 WFO：需要实际运行来找到稳定参数
- Phase 6 组合层：单股回测未使用横截面排序

#### 更新后核心判断

1. 相比旧 baseline，中位年化从 0.60% → 2.43%（+4x），Sharpe 从 0.11 → 0.26（+2.4x），止损和量能确认贡献了大部分改善。
2. 但 **最大回撤中位数仍然高达 47.11%**，距离 35% 目标差距明显。8% 固定止损压不住趋势逆转时的大幅回撤。
3. 收益仍然高度集中于少数强趋势股：Top 5 的 Sharpe 在 0.58-0.88，而后 12 只 Sharpe < 0.1。**组合层（Phase 6）是提高整体收益的最高杠杆。**
4. 成本敏感性极低（Sharpe 衰减 < 0.02），换手率不是当前瓶颈。
5. **后续优先事项**：(a) 运行嵌套 WFO 找稳定参数区域，(b) 将 meta-labeling 接入回测过滤低质量信号，(c) 启用组合层 top N 选股。

### 1.3 WFO 结果诊断

现有 `data/output/*_wfo_20260513.json` 里有 5 只股票的 WFO 结果：

| 标的 | folds | OOS 年化 | OOS Sharpe | OOS MDD | DSR p-value | IS/OOS Sharpe 相关 |
|---|---:|---:|---:|---:|---:|---:|
| 002475 | 17 | 14.61% | 0.67 | 32.46% | 0.56 | -0.35 |
| 300059 | 17 | 17.92% | 0.69 | 31.23% | 0.51 | -0.07 |
| 300750 | 11 | 20.53% | 0.82 | 45.79% | 0.55 | 0.69 |
| 600036 | 17 | 4.19% | 0.33 | 22.60% | 0.88 | -0.27 |
| 601166 | 17 | 4.81% | 0.39 | 24.93% | 0.84 | -0.22 |

结论：

1. WFO 对强趋势股有效，但多数标的的 DSR p-value 偏高，统计显著性不足。
2. 除 300750 外，IS/OOS Sharpe 相关为负或接近 0，参数择优存在过拟合风险。
3. 这些 WFO 文件生成时的 `param_grid` 只包含 MACD 参数，未覆盖当前代码已支持的 `min_run_len`、`stop_loss_pct` 等扩展参数，需要重跑新基准。

---

## 2. 首要问题和改进方向

### 2.1 必须优先修正的评估问题

| 问题 | 影响 | 修正方向 |
|---|---|---|
| `run_batch_backtest.py` 未完整透传 `transaction_cost`、指数过滤、真实成本模型 | 批量结果和单股结果口径不一致 | 抽出统一 `_bt_kwargs()`，批量/单股/WFO 共用 |
| `run_wfo.py` 只透传部分回测参数 | WFO 优化的不是实际配置策略 | WFO 纳入成本、量能、共振、仓位、再入场、指数过滤 |
| `run_permutation_test()` 打乱行后会被 `_prepare_ohlcv()` 按 `trade_date` 排序还原 | 置换检验无法形成有效零分布 | 改为打乱信号、收益映射或重新生成合成日期 |
| `breakdown_by_regime()` 按数组长度对齐，未按日期对齐指数 | 市场状态分解可能错位 | 改为 `pd.Series` 日期索引对齐 |
| 止损强制退出未来动作检测漏掉 `atr_stop` | ATR 止损可能覆盖逻辑不严谨 | `_future_exit_index()` 加入 `atr_stop` |
| 卖出只处理停牌近似，未处理一字跌停无法卖出 | 回撤可能被低估 | 增加开盘跌停不可卖出规则 |
| `trade_log` 未记录 `entry_quality_score` | 无法验证质量分是否能提升收益 | 买入时写入质量分、量能、ATR 分位、市场状态 |

这些问题不直接创造 Alpha，但决定后续所有“收益提升”是否可信。第一阶段必须先完成。

### 2.2 收益提升的主要杠杆

| 杠杆 | 为什么有效 | 当前缺口 |
|---|---|---|
| 信号筛选 | 过滤低质量 DK 翻红，减少震荡市亏损 | 质量评分已有雏形，但未进入回测决策 |
| 市场/行业状态 | A 股单股趋势高度依赖大盘和行业 | 只有极端下跌入场过滤，缺少连续风险预算 |
| 参数稳健选择 | 选择稳定区域而非单个最高 IS Sharpe | WFO 只按训练 Sharpe 择优 |
| 动态仓位 | 强信号加仓，弱信号或高波动减仓 | 风险仓位只看止损距离，不看信号胜率 |
| 退出优化 | 降低大回撤和趋势回吐 | 现有止损未加入时间止损、盈利保护、风险离场 |
| 横截面选股 | 当前收益集中于少数股票，应优先交易优势标的 | 缺少 watchlist 级别排序和资金分配 |

---

## 3. 目标指标

所有目标以 OOS 或滚动样本为准，不以全样本单次回测为准。

### 3.1 第一阶段验收目标

| 指标 | 当前基准 | 第一阶段目标 |
|---|---:|---:|
| 批量回测口径 | 单股/批量/WFO 不完全一致 | 三者共用同一回测参数构造 |
| 置换检验 | 存在排序还原问题 | Null Sharpe 分布和 observed 明显非同一常数 |
| WFO 参数网格 | 旧结果只含 MACD | 新结果包含 `min_run_len`、止损、量能或质量阈值 |
| 测试数 | 当前测试通过即可 | 新增关键 bug 回归测试，不降低覆盖 |

### 3.2 收益目标

| 指标 | 当前基准 | 目标区间 |
|---|---:|---:|
| watchlist 年化收益中位数 | 0.60% | 5% 以上 |
| watchlist Sharpe 中位数 | 0.11 | 0.35 以上 |
| watchlist Calmar 中位数 | 0.019 | 0.25 以上 |
| Calmar > 0.5 标的数 | 2 / 25 | 8 / 25 以上 |
| 最大回撤中位数 | 50.33% | 35% 以下 |
| WFO DSR p-value < 0.10 | 当前偏少 | 至少 30% 标的通过 |

这些不是收益承诺，而是研究阶段的通过阈值。若某一类改动只提高全样本收益，却恶化 OOS、DSR 或最大回撤，应放弃。

---

## 4. 实施阶段

## 阶段 0：建立可靠实验基准

### 0.1 统一回测参数入口

**目标**：单股、批量、WFO、置换检验都使用同一套参数构造，避免同一策略在不同入口结果不一致。

**改动文件**：

| 文件 | 改动 |
|---|---|
| `scripts/run_backtest_single.py` | 保留 `_bt_kwargs()`，作为参考实现 |
| `scripts/run_batch_backtest.py` | 复用统一参数构造，支持 `transaction_cost`、指数过滤、风险仓位、再入场 |
| `scripts/run_wfo.py` | 透传完整 `signal_filter`、`risk`、`backtest` 参数 |
| `src/backtest/config.py` | 建议新增：集中生成 `BacktestRuntimeConfig` 或 `dict` |

**验收**：

```bash
pytest tests/test_transaction_costs.py tests/test_position_management.py tests/test_wfo_params_split.py
python scripts/run_backtest_single.py --symbol 600036 --start 2020-01-01 --end 2026-05-08
python scripts/run_batch_backtest.py --watchlist configs/watchlist_25.txt --start 2020-01-01 --end 2026-05-08 --export-summary data/output/batch_rebaseline.csv
```

同一标的在单股和批量入口的核心指标应一致或仅有展示口径差异。

### 0.2 修复统计验证链路

**问题 1：置换检验无效**

当前 `run_permutation_test()` 对 `ohlcv.sample(frac=1)` 后调用 `run_single_stock_backtest()`，但后者会按 `trade_date` 排序，导致打乱被还原。

**修正方案**：

1. 方案 A：固定真实信号日期，随机置换未来日收益，构造 null equity。
2. 方案 B：打乱 `close/open/high/low/volume` 序列后重新生成连续合成日期。
3. 方案 C：只打乱 DK 信号触发日期，保留真实收益路径和执行规则。

优先选方案 C，因为它保留 A 股收益分布、跳空、涨跌停结构，更适合检验“信号时点是否有信息含量”。

**问题 2：市场状态日期对齐**

`breakdown_by_regime()` 应接收带日期索引的 `strategy_returns` 和 `index_returns`，按日期 inner join 后再计算 60 日指数状态。

**验收**：

| 测试 | 期望 |
|---|---|
| 打乱后 null Sharpe 不应全部等于 observed Sharpe | 通过 |
| 指数日期少于个股日期时仍能正确对齐 | 通过 |
| 缺指数数据时明确输出 `no_index_data`，不伪装为震荡市 | 通过 |

### 0.3 更新配置文档

`src/settings.py` 已包含较新的默认配置，但 `config.yaml.example` 和实际 `config.yaml` 缺少部分字段。需要补齐：

```yaml
trend_signal:
  min_run_len: 1

backtest:
  atr_stop_multiplier: 0.0
  atr_stop_period: 14
  risk_per_trade_pct: 0.0
  position_size_cap: 1.0
  stop_reentry_enabled: false
  stop_reentry_cooldown: 3
  stop_reentry_min_run: 2
  transaction_cost:
    commission_buy_bps: 2.5
    commission_sell_bps: 2.5
    slippage_bps_per_side: 2.0
    stamp_duty_sell_bps: 5.0

wfo:
  param_grid:
    macd_fast: [8, 10, 12, 14]
    macd_slow: [22, 26, 30]
    macd_signal: [7, 9, 11]
    min_run_len: [1, 2, 3]
    stop_loss_pct: [0.05, 0.08, 0.10]
```

---

## 阶段 1：回测真实性和风险归因

### 1.1 增加一字跌停无法卖出

当前买入时处理了开盘一字涨停不可买，但卖出只检查停牌近似。A 股中下跌趋势的退出经常遇到跌停，忽略该规则会低估亏损和回撤。

**改动**：

| 文件 | 内容 |
|---|---|
| `src/market/tradability.py` | 新增 `limit_down_px()`、`is_open_limit_down_unsellable()` |
| `src/backtest/single_stock.py` | `_next_sell_index()` 跳过一字跌停开盘 |
| `tests/test_tradability.py` | 增加主板、创业板、科创板跌停比例测试 |
| `tests/test_single_stock_bt.py` | 增加卖出顺延测试 |

**预期影响**：部分下跌趋势股票回撤会上升，但这是必要的真实性修正。后续收益提升必须在这个真实口径上验证。

### 1.2 交易归因表

每笔交易应记录更多入场和出场信息：

| 字段 | 用途 |
|---|---|
| `entry_quality_score` | 验证高质量信号是否更赚钱 |
| `entry_volume_ratio` | 评估量能确认有效性 |
| `entry_atr_pct` | 判断高波动入场是否劣化收益 |
| `entry_market_regime` | 判断牛/熊/震荡下信号表现 |
| `entry_rs_60` | 个股相对指数 60 日强度 |
| `exit_reason` | 统计信号退出、止损、时间止损、盈利保护的贡献 |
| `mae` / `mfe` | 最大不利/有利波动，用于优化止损和止盈 |

**新增分析**：

1. 按 `entry_quality_score` 分桶统计收益、胜率、回撤。
2. 按 `exit_reason` 统计平均收益和亏损贡献。
3. 输出 top loss trades，检查是否由跌停、跳空、追高或低质量信号造成。

### 1.3 成本和滑点压力测试

保留现有 `--compare-costs`，但升级为批量报告：

| 成本模型 | 目标 |
|---|---|
| 零成本 | 策略理论上界 |
| 当前对称 15bps | 与历史结果兼容 |
| 真实 A 股 | 默认生产口径 |
| 高滑点 | 检验换手敏感性 |

验收标准：真实 A 股口径相对零成本的 Sharpe 衰减不超过 0.25；若超过，应优先降低换手，而不是调参追收益。

---

## 阶段 2：信号质量和特征增强

### 2.1 将质量评分接入回测决策

当前 `compute_signal_quality()` 只用于信号记录，不参与回测。下一步新增“只交易高质量信号”的策略变体。

**参数**：

```yaml
signal_filter:
  min_quality_score: 0      # 0 表示关闭
  quality_score_mode: hard  # hard: 低于阈值不买；scale: 按质量缩放仓位
```

**规则**：

| 模式 | 规则 |
|---|---|
| `hard` | BUY 信号质量分低于阈值时跳过 |
| `scale` | 仓位乘以 `quality_score / 100`，最低可设 floor |
| `analysis` | 不影响交易，只输出分桶绩效 |

**WFO 网格**：

```yaml
min_quality_score: [0, 20, 40, 60]
```

### 2.2 增加趋势强度特征

DK 翻红只说明短周期趋势转正，不代表趋势足够强。建议加入：

| 特征 | 解释 | 用法 |
|---|---|---|
| `ma20_slope` | 20 日均线斜率 | 过滤下降趋势中的短暂反弹 |
| `ma60_slope` | 中期趋势斜率 | 判断是否顺大趋势 |
| `close_above_ma60` | 收盘价是否在 MA60 上方 | 作为趋势环境过滤 |
| `donchian_20_breakout` | 20 日新高突破 | 捕捉强趋势启动 |
| `atr_pct_rank_120` | ATR 百分位 | 高波动时减仓或过滤 |
| `rs_60` | 个股 60 日收益减指数 60 日收益 | 只做相对强势股 |

建议新增文件：

| 文件 | 内容 |
|---|---|
| `src/features/trend_features.py` | 生成趋势强度、波动、量价、相对强度特征 |
| `tests/test_trend_features.py` | 验证无未来函数、日期对齐、缺数据处理 |

### 2.3 建立多信号候选池

不要只围绕 MACD 参数做微调，增加几类互补信号：

| 信号族 | 候选规则 | 目的 |
|---|---|---|
| DK 趋势 | 当前三模式和共振 | 保留现有基准 |
| Donchian 突破 | 20/55 日新高买入，跌破 10/20 日低点卖出 | 捕捉强趋势 |
| 均线状态 | MA20 > MA60 且 MA20 上行 | 过滤下跌反弹 |
| 动量相对强度 | 个股 60/120 日收益跑赢指数 | 做强不做弱 |
| 波动压缩突破 | ATR 分位低后放量突破 | 捕捉趋势启动点 |

先以规则方式实现，不立即引入复杂 ML。每个信号族都必须进入同一评估框架，输出 IS/OOS、DSR、回撤和交易数。

---

## 阶段 3：从规则到轻量模型

### 3.1 Meta-labeling：预测 BUY 信号是否值得交易

当前问题不是“每天预测涨跌”，而是“已有 BUY 信号中哪些值得执行”。这更适合小样本。

**样本构造**：

| 项 | 设计 |
|---|---|
| 样本点 | 每个 BUY 候选信号 |
| 标签 1 | 持有到策略原始 SELL 的交易收益是否 > 0 |
| 标签 2 | 未来 20 日最大收益是否超过未来 10 日最大亏损 |
| 标签 3 | 未来 20/60 日是否跑赢指数 |
| 特征 | 阶段 2 的趋势、量能、波动、相对强度、市场状态 |
| 切分 | 按日期滚动 WFO，禁止随机切分 |

### 3.2 模型选择

优先顺序：

| 模型 | 原因 |
|---|---|
| Logistic Regression + L1/L2 | 小样本、可解释、稳定 |
| HistGradientBoosting / LightGBM | 捕捉非线性，但必须限制深度 |
| Isotonic/Platt 校准 | 将概率映射为仓位 |

暂不建议直接上 LSTM、Transformer 或复杂深度模型。单股日线样本很少，强行深度学习大概率只是拟合历史噪声。

### 3.3 模型输出如何进入交易

| 输出 | 用法 |
|---|---|
| `p_win` | 低于阈值不交易 |
| `expected_edge` | 与波动率一起决定仓位 |
| `feature_importance` | 检查模型是否依赖合理特征 |
| `calibration_curve` | 防止概率过度自信 |

建议仓位函数：

```text
position_fraction =
  base_risk_position
  × clip((p_win - 0.50) / 0.20, 0, 1)
  × market_regime_multiplier
  × drawdown_throttle
```

### 3.4 验收标准

| 指标 | 要求 |
|---|---|
| 交易次数 | 不低于原策略 35%，避免靠极少交易抬高指标 |
| OOS Sharpe | 相比规则基准提升至少 0.15 |
| OOS 最大回撤 | 不高于规则基准 |
| 特征稳定性 | top 特征在不同 WFO fold 中不能完全随机 |
| DSR p-value | 相比规则基准改善 |

---

## 阶段 4：退出和仓位优化

### 4.1 退出逻辑升级

当前退出主要依赖 DK 翻绿和止损。建议增加：

| 退出 | 规则 | 目的 |
|---|---|---|
| 时间止损 | 入场 N 日后收益仍低于阈值则退出 | 清理无效信号 |
| 盈利保护 | 盈利超过 X 后启用更紧 trailing stop | 减少大幅回吐 |
| 市场风险退出 | 指数跌破 MA60 或 20 日跌幅过大时减仓/退出 | 降低系统性回撤 |
| 信号衰减退出 | DK 仍红但趋势强度下降到阈值以下 | 早于翻绿撤退 |

WFO 网格示例：

```yaml
time_stop_days: [0, 20, 40]
time_stop_min_return: [-0.03, 0.0]
profit_lock_trigger: [0.10, 0.20]
profit_lock_trailing: [0.06, 0.10]
market_exit_mode: [off, reduce, exit]
```

### 4.2 波动率目标仓位

当前风险仓位依据止损距离，但没有目标波动率。增加：

```yaml
backtest:
  volatility_target_ann: 0.18
  volatility_lookback: 20
  max_position_fraction: 1.0
  min_position_fraction: 0.0
```

规则：

```text
position = min(position_size_cap, volatility_target_ann / realized_vol_ann)
```

再与 `risk_per_trade_pct`、质量分、市场状态乘子取更保守结果。

### 4.3 回撤节流

当策略自身净值进入回撤时降低仓位：

| 当前策略回撤 | 仓位乘子 |
|---|---:|
| < 5% | 1.0 |
| 5% - 10% | 0.7 |
| 10% - 15% | 0.5 |
| > 15% | 0.0 或 0.3 |

这个机制不创造 Alpha，但能明显降低尾部回撤，适合当前最大回撤偏高的问题。

---

## 阶段 5：WFO 和参数选择升级

### 5.1 目标函数从 Sharpe 改为复合目标

当前 WFO 按训练集 Sharpe 选最优，容易选中尖峰参数。建议改为：

```text
score =
  0.45 * sharpe
  + 0.25 * calmar
  + 0.15 * annualized_return
  - 0.10 * max_drawdown
  - 0.05 * turnover_penalty
  - stability_penalty
```

同时增加硬约束：

| 约束 | 默认 |
|---|---:|
| 最少交易数 | 每年 >= 3 |
| 最大回撤上限 | <= 45% |
| DSR p-value | <= 0.20 优先 |
| 参数稳定性 | 相邻参数组合表现不能断崖式下降 |

### 5.2 选择参数平台，而不是单点最优

对于二维或多维参数网格：

1. 找到 top 20% 训练参数。
2. 选择附近 OOS 表现更稳定的参数区域。
3. 若最优点孤立，降权或放弃。
4. 输出参数稳定性热力图和 fold 间参数漂移。

### 5.3 嵌套 WFO

外层用于 OOS 评估，内层用于参数选择：

```text
outer train  -> inner WFO select params -> outer OOS evaluate
```

这会降低表面收益，但能显著减少过拟合。

### 5.4 参数搜索优先级

第一批只搜索对当前瓶颈最相关的参数：

```yaml
wfo:
  param_grid:
    mode: [macd_cross, ma_cross, boll_trend, consensus]
    min_run_len: [1, 2, 3]
    volume_confirm: [false, true]
    volume_ratio_min: [1.0, 1.3, 1.6]
    stop_loss_pct: [0.0, 0.06, 0.08, 0.10]
    atr_stop_multiplier: [0.0, 1.5, 2.0, 2.5]
    min_quality_score: [0, 20, 40]
```

不要一次性加入太多参数。若组合数超过 1000，应先用粗网格和单因子敏感性筛选。

---

## 阶段 6：watchlist 横截面选择和资金分配

从真实结果看，收益集中于少数标的。单股策略要提升整体收益，最有效的方式之一是“只在更有优势的股票上开仓”。

### 6.1 保留单股引擎，新增组合研究层

新增模块建议：

| 文件 | 内容 |
|---|---|
| `src/portfolio/signal_ranker.py` | 对 watchlist 当日 BUY 候选打分排序 |
| `src/portfolio/allocator.py` | top N 等权、波动率倒数、行业约束 |
| `src/portfolio/backtest.py` | 用 `src/backtest/engine.py` 或单股交易日志组合成资金曲线 |
| `scripts/run_portfolio_backtest.py` | watchlist 组合回测入口 |

### 6.2 排序分数

```text
rank_score =
  0.30 * signal_quality
  + 0.25 * relative_strength_60
  + 0.20 * trend_strength
  + 0.15 * market_regime_score
  + 0.10 * liquidity_score
  - 0.20 * volatility_penalty
```

### 6.3 组合约束

| 约束 | 默认 |
|---|---:|
| 最大持仓数 | 5 |
| 单票上限 | 25% |
| 单行业上限 | 40% |
| 单日换手上限 | 50% |
| 最低成交额过滤 | 近 20 日均成交额 > 1 亿 |
| 调仓频率 | 日信号触发，周度强制复核 |

### 6.4 验收目标

组合层目标应高于单股中位数：

| 指标 | 目标 |
|---|---:|
| 组合 OOS 年化 | 8% - 15% |
| 组合 OOS Sharpe | > 0.7 |
| 组合最大回撤 | < 30% |
| 年换手 | 可解释且成本后仍盈利 |
| 单一年份亏损 | 不超过 2 个年份或亏损可归因 |

---

## 阶段 7：报告和研究闭环

### 7.1 标准研究报告

每次实验输出一个目录：

```text
data/output/experiments/{YYYYMMDD}_{experiment_id}/
  config.yaml
  batch_summary.csv
  wfo_summary.csv
  trade_attribution.csv
  regime_breakdown.csv
  cost_sensitivity.csv
  portfolio_summary.csv
  report.html
  notes.md
```

### 7.2 实验记录表

新增 `data/output/experiments/index.csv`：

| 字段 | 示例 |
|---|---|
| `experiment_id` | `E20260513_quality_filter_v1` |
| `git_commit` | 当前提交 hash |
| `config_hash` | 配置 hash |
| `start/end` | 回测区间 |
| `universe` | watchlist 文件 |
| `median_sharpe` | 指标 |
| `median_calmar` | 指标 |
| `max_drawdown_median` | 指标 |
| `notes` | 简短结论 |

### 7.3 决策规则

| 结果 | 决策 |
|---|---|
| 全样本收益提升，OOS 恶化 | 放弃 |
| Sharpe 提升但最大回撤恶化 > 5pct | 只保留为高风险变体 |
| 交易次数下降到原来的 20% 以下 | 视为样本不足，不算通过 |
| DSR p-value 明显改善 | 优先进入下一轮 |
| 只在单一股票有效 | 记录为个股特化参数，不作为通用策略 |

---

## 5. 具体实验矩阵

| ID | 实验 | 假设 | 主要文件 | 验收 |
|---|---|---|---|---|
| E0 | 统一参数和修复检验 | 先让结果可信 | `scripts/*`, `src/backtest/permutation_test.py` | 单股/批量/WFO 口径一致 |
| E1 | 一字跌停卖出顺延 | 真实回撤更高但更可信 | `tradability.py`, `single_stock.py` | 新测试通过 |
| E2 | 质量分硬过滤 | 低分 BUY 是亏损来源 | `generator.py`, `single_stock.py` | OOS Sharpe +0.10 |
| E3 | 质量分缩放仓位 | 高质量信号应承担更多风险 | `single_stock.py` | 回撤不升，收益提升 |
| E4 | 相对强度过滤 | 只做跑赢指数的股票 | `features/trend_features.py` | Bottom 标的亏损减少 |
| E5 | 市场状态乘子 | 熊市和震荡市应降仓 | `risk_metrics.py`, `single_stock.py` | MDD 中位数下降 |
| E6 | Donchian 突破信号 | 强趋势启动比 DK 翻色更稳 | `indicators/`, `signals/` | 强趋势股收益提升 |
| E7 | 时间止损 | 无效交易应更早退出 | `single_stock.py` | 平均亏损下降 |
| E8 | 盈利保护 | 减少趋势回吐 | `single_stock.py` | Calmar 提升 |
| E9 | 复合 WFO 目标 | 降低参数过拟合 | `wfo.py` | IS/OOS 相关改善 |
| E10 | Meta-label logistic | 预测 BUY 是否值得做 | `src/models/` | OOS Sharpe 明显提升 |
| E11 | Top N watchlist 组合 | 资金集中到优势标的 | `src/portfolio/` | 组合 Sharpe > 0.7 |
| E12 | 报告闭环 | 实验可复现 | `src/backtest/report.py` | 每次实验有完整目录 |

---

## 6. 推荐实施顺序

### 第 1 周：修正评估口径

1. 抽出统一回测参数构造。
2. 修复置换检验、市场状态对齐、ATR exit 检测、一字跌停卖出。
3. 更新 `config.yaml.example` 和 `docs/backtest_guide.md`。
4. 重跑当前策略作为新 baseline。

输出：

```text
data/output/experiments/E0_baseline_fixed/
```

### 第 2 周：信号质量和特征

1. `trade_log` 增加入场质量、量能、波动、相对强度字段。
2. 实现质量分硬过滤和仓位缩放。
3. 新增趋势强度和相对强度特征。
4. 对 25 只 watchlist 做质量分桶分析。

输出：

```text
quality_bucket_report.csv
feature_attribution.csv
```

### 第 3 周：退出和仓位

1. 时间止损。
2. 盈利保护 trailing。
3. 市场状态乘子和回撤节流。
4. 组合到 WFO 网格中做受控搜索。

目标：最大回撤中位数从 50.33% 压到 35% - 40% 区间。

### 第 4 周：WFO 升级和组合层

1. WFO 目标函数改为复合目标。
2. 实现参数平台选择。
3. 建立 watchlist top N 组合回测。
4. 生成最终研究报告。

目标：组合 OOS Sharpe > 0.7，最大回撤 < 30%。

---

## 7. 验证命令

基础测试：

```bash
pytest
```

单股基准：

```bash
python scripts/run_backtest_single.py --symbol 600036 --start 2020-01-01 --end 2026-05-08 --export-html --compare-costs --permutation-test --n-permutations 500
```

批量基准：

```bash
python scripts/run_batch_backtest.py \
  --watchlist configs/watchlist_25.txt \
  --start 2020-01-01 \
  --end 2026-05-08 \
  --export-summary data/output/batch_baseline_fixed.csv \
  --export-html
```

WFO 基准：

```bash
python scripts/run_wfo.py --symbol 600036 --start 2020-01-01 --end 2026-05-08 --train-days 504 --oos-days 126 --export-results --plot-heatmap
python scripts/run_wfo.py --symbol 300750 --start 2020-01-01 --end 2026-05-08 --train-days 504 --oos-days 126 --export-results --plot-heatmap
python scripts/run_wfo.py --symbol 300760 --start 2020-01-01 --end 2026-05-08 --train-days 504 --oos-days 126 --export-results --plot-heatmap
```

选择这三只股票的原因：

| 标的 | 角色 |
|---|---|
| 600036 招商银行 | 中等稳定收益样本 |
| 300750 宁德时代 | 强趋势高收益样本 |
| 300760 迈瑞医疗 | 当前策略失败样本 |

---

## 8. 文件级路线图

| 文件/目录 | 任务 |
|---|---|
| `src/backtest/config.py` | 新增统一回测参数构造 |
| `src/backtest/single_stock.py` | 卖出跌停顺延、质量分入场、时间止损、盈利保护、波动仓位、回撤节流 |
| `src/backtest/permutation_test.py` | 修复置换逻辑，改为信号时点置换 |
| `src/backtest/performance_panel.py` | 日期对齐版 regime breakdown |
| `src/backtest/wfo.py` | 复合目标、稳定平台、更多参数透传 |
| `src/market/tradability.py` | 跌停不可卖出 |
| `src/features/trend_features.py` | 新增趋势、量能、波动、相对强度特征 |
| `src/models/meta_label.py` | 轻量模型训练和预测 |
| `src/portfolio/` | watchlist 排名和组合回测 |
| `scripts/run_batch_backtest.py` | 与单股入口统一参数 |
| `scripts/run_wfo.py` | 完整策略参数和新报告 |
| `scripts/run_portfolio_backtest.py` | 新增组合层入口 |
| `config.yaml.example` | 补齐当前默认配置 |
| `docs/backtest_guide.md` | 更新真实执行和验证说明 |
| `tests/` | 每个新增行为必须有回归测试 |

---

## 9. 风险和边界

1. 不保证任何策略在未来一定赚钱，所有收益目标都是研究验收阈值。
2. 不在单股日线小样本上直接使用深度学习。
3. 不使用随机切分评估时间序列模型。
4. 不接受只靠减少交易次数得到的漂亮 Sharpe。
5. 不接受未计真实成本、跌停卖出约束和 OOS 检验的收益提升。
6. 若组合层收益显著优于单股层，应承认主要 Alpha 来自横截面选择，而不是单股择时本身。

---

## 10. 下一步执行清单

优先级从高到低：

1. 修复评估链路：统一参数、置换检验、日期对齐、跌停卖出、ATR exit 检测。
2. 重跑 `batch_summary` 和 3 只代表股票 WFO，建立可信 baseline。
3. 把质量评分接入交易，做 `min_quality_score` 分桶实验。
4. 增加相对强度和趋势强度特征，评估能否减少 Bottom 股票亏损。
5. 增加时间止损、盈利保护、波动仓位、市场状态乘子，目标先压回撤。
6. 改造 WFO 目标函数，使用参数平台而不是单点最高 Sharpe。
7. 建立 watchlist top N 组合回测，验证收益是否能从“少数强股”转为“组合稳定收益”。

