# 改进与模型效果优化计划

**项目**：`single-stock-quant`
**日期**：2026-05-13
**基于代码版本**：`single-stock-quant-main`（Phase 0–7 全部完成）
**撰写目的**：在现有框架之上识别真实瓶颈，提出可落地、可测量的下阶段改进路径。

---

## 0. 执行进度更新（2026-05-14）

| 阶段 | 状态 | 已落地内容 | 验证 |
|---|---|---|---|
| 阶段 8：Meta-labeling 接入 | 已完成 | `single_stock.py` 支持 `meta_model`、`meta_label_mode`、`meta_label_threshold`；WFO fold 内训练并注入 OOS；CLI/config 已接线 | 全量测试通过 |
| 阶段 9：MA120/RS60 趋势过滤 | 已完成 | 单股回测支持 `require_above_ma120`、`require_positive_rs60`；配置和 CLI 已接线 | 全量测试通过 |
| 阶段 10：Portfolio 层运行 | 已完成（代码层 + E10 首跑） | `run_portfolio_backtest.py` 支持 watchlist、benchmark、summary/weights/scores/html 导出；`rank_signals()` 接入 meta-label `p_win`、MA120、RS60 过滤；组合回测透传 meta 分数和过滤参数；新增扩展窗口 `p_win` 面板构造 | `pytest -q` 通过，当前 161 个测试 |
| 阶段 11：Meta-label 特征工程 v2 | 已完成（首版） | `build_signal_features()` 新增 52 周位置、价量背离、趋势一致性、MACD 柱体方向、换手率分位、Beta、价格加速度、量价相关 8 个特征；新增特征边界和稳定性测试 | `pytest -q` 通过，当前 161 个测试 |
| 阶段 12：WFO 稳定性架构升级 | 已完成（首版） | 新增 `_select_stable_params()`，按跨 fold `mean/(std+0.1)` 选择稳定参数区域；nested WFO 内层参数选择优先使用稳定区域，fold 不足 5 个时自动回退；`run_wfo.py --stability-weighting` 已接线 | `pytest -q` 通过，当前 161 个测试 |
| 阶段 13：多周期信号确认 | 已完成（首版） | 新增 `src/features/weekly_trend.py`，日线聚合周线并输出 bullish/bearish/neutral；`single_stock.py` 支持 `require_weekly_bullish`、`weekly_ma_fast`、`weekly_ma_slow`；config/CLI/WFO 已接线 | `pytest -q` 通过，当前 168 个测试 |
| 阶段 14：动态仓位精细化 | 已完成（首版） | 新增 EWMA 年化波动率估计；波动率目标仓位改用 EWMA 并支持高波动折扣；`position_size_cap` 在无止损风险 sizing 时也生效；ATR 止损距离继续约束风险仓位 | `pytest -q` 通过，当前 168 个测试 |
| 阶段 15：实验闭环与报告完善 | 已完成（首版） | `create_experiment_dir()` 生成阶段 15 产物清单和 `DELTA.md` 占位；新增 `load_experiment_metrics()`、`compare_metric_summaries()`、HTML/Markdown 对比报告渲染；新增 `scripts/compare_experiments.py` 支持实验目录对比、CSV 导出和写回 `DELTA.md` | `pytest tests/test_experiment.py -q` 通过，当前该文件 14 个测试 |

阶段 10 首次真实数据实验已完成：`python scripts/run_portfolio_backtest.py --watchlist configs/watchlist_25.txt --start 2020-01-01 --end 2026-05-08 --n-top 5 --enable-meta-label --require-above-ma120 --export-summary data/output/portfolio_final.csv --export-weights data/output/portfolio_final_weights.csv --export-scores data/output/portfolio_final_scores.csv --export-html`。

输出文件：
- `data/output/portfolio_final.csv`
- `data/output/portfolio_final_weights.csv`
- `data/output/portfolio_final_scores.csv`
- `data/output/portfolio_backtest_20260514.html`

E10 首跑结果：年化收益 -1.79%，Sharpe 0.06，Calmar -0.03，最大回撤 64.34%，平均持仓 4.4。结果未达到组合 Sharpe ≥ 0.75、最大回撤 ≤ 28% 的验收目标；阶段 11/12/13/14/15 的代码层首版已完成，下一步应重跑 E11/E12/E13/E14/E_FINAL 实验，并用 `scripts/compare_experiments.py` 生成 DELTA 对比报告，验证新特征、稳定选参、周线过滤和 EWMA 仓位约束是否改善 WFO OOS 指标，同时复盘组合 ranking 是否过度持有弱势标的。

---

## 1. 现状总结与诊断

### 1.1 已完成能力盘点

Phase 0–7 完成后，项目具备以下完整能力：

| 层次 | 已实现 |
|---|---|
| 数据 | DuckDB 日线 + AkShare 拉取 + 质量检查 |
| 指标 | MACD/MA/Boll/Donchian 四类 DK 趋势 |
| 信号 | 量能确认、三模式共振、防抖、质量评分 (0–100) |
| 特征 | 均线斜率、Donchian 突破、ATR 分位、量比、相对强度 |
| 回测 | T+1 涨跌停处理、真实 A 股成本模型、停牌近似 |
| 风控 | 固定/追踪/ATR 止损、盈利保护、时间止损、市场退出、波动率目标仓位、回撤节流 |
| 模型 | L2 逻辑回归 Meta-labeling（`src/models/meta_label.py`） |
| 评估 | WFO + 嵌套 WFO、复合目标函数、参数平台选择、DSR、Bootstrap Sharpe CI、置换检验、HTML 报告 |
| 组合 | `signal_ranker.py`、`allocator.py`、`portfolio/backtest.py` |

### 1.2 当前回测结果（Phase 0–7 完成后）

区间：2020-01-02 ~ 2026-05-08，25 只 watchlist

| 指标 | 当前值 | 目标值 | 差距 |
|---|---:|---:|---:|
| 年化收益中位数 | 2.43% | 5%+ | -2.57pct |
| Sharpe 中位数 | 0.26 | 0.35+ | -0.09 |
| Calmar 中位数 | 0.040 | 0.25+ | **-0.21** |
| 最大回撤中位数 | 47.11% | <35% | **+12.11pct** |
| Calmar > 0.5 标的数 | 3 / 25 | 8 / 25+ | -5 只 |

**核心矛盾**：Sharpe 已有一定改善（0.11 → 0.26），但 Calmar 依然极低（0.04），最大回撤高达 47%。这说明当前策略能捕捉趋势方向，但**无法控制持仓期间的大幅回撤**，导致风险收益比严重失衡。

### 1.3 代码审查发现的关键空洞

通过逐文件阅读代码，发现以下高优先级问题：

#### 空洞 A：Meta-labeling 模型已实现但从未接入交易

`src/models/meta_label.py` 完整实现了 `build_signal_features()`、`run_meta_label_wfo()` 和逻辑回归模型。**但在 `src/backtest/single_stock.py` 中，`run_single_stock_backtest()` 从不调用这个模型**，模型预测的 `p_win` 完全没有影响任何交易决策。

```python
# 现状：run_single_stock_backtest() 内没有任何如下代码
# meta_model.predict_proba(features)
# if p_win < threshold: skip_trade()
```

这是已完成工作中最大的"最后一公里"缺口。模型已训练，特征已计算，只差最后的接入。

#### 空洞 B：Portfolio 层未产出任何 OOS 数字

`src/portfolio/backtest.py` 的 `run_portfolio_backtest()` 已经写好，但：
- 没有对应的 `scripts/run_portfolio_backtest.py`（目录下只有空的 `.gitkeep`）
- 没有任何 portfolio-level 的 OOS 回测结果文件
- `rank_signals()` 在 `signal_ranker.py` 里打的是"趋势强度分"，**没有使用 Meta-label 的 `p_win` 作为信号排名的核心权重**

#### 空洞 C：WFO 的 IS/OOS 相关性大多为负

5 只有 WFO 结果的标的中，4 只 IS/OOS Sharpe 相关系数为负（-0.35, -0.07, -0.27, -0.22）。原因在于：
- 参数选择依赖 IS 单一最高 Sharpe，而不是"稳定区域中心"
- 即使已实现复合目标函数，`_select_best_params()` 仍然是逐 fold 选最优，没有跨 fold 的稳定性加权
- 嵌套 WFO 存在实现，但没有真正运行并记录结果

#### 空洞 D：Bottom 股票的根本问题未被解决

Bottom 5 标的（迈瑞医疗、恒瑞医药、伊利股份、海螺水泥、格力电器）在 2021–2024 年均处于行业或公司基本面下行趋势中，DK 趋势指标在这种"慢熊"结构中会反复翻红翻绿，产生大量假信号。

当前的"市场退出"和"指数过滤"只关注大盘整体，**没有行业/个股的中长期趋势判断**。一只股票可以在沪深 300 上涨的环境中持续走弱——现有框架完全无法识别这种结构。

#### 空洞 E：质量评分特征维度不足

当前 `compute_signal_quality()` 的评分维度（共 8 项）完全基于技术面短期指标，缺少：
- 个股相对行业指数的强弱（60/120 日）
- 个股在其历史分位的动量（52 周高低位）
- 大盘趋势结构的质量（牛市 vs. 震荡市下质量分的校准）

---

## 2. 根本原因分析

```
问题：Calmar 低、最大回撤高
  ├── 直接原因 1：在结构性下跌股票上持续交易
  │     └── 根本原因：缺少中长期趋势过滤（行业+个股 60/120 日方向）
  ├── 直接原因 2：Meta-label 模型未使用，无法过滤低胜率信号
  │     └── 根本原因：模型实现与回测引擎未完成接线
  ├── 直接原因 3：Portfolio 层未运行，资金分散在劣势标的
  │     └── 根本原因：缺少运行脚本和 OOS 验证
  └── 直接原因 4：参数过拟合，IS/OOS 转移差
        └── 根本原因：WFO 选最高点，未用跨 fold 稳定性评分
```

---

## 3. 改进目标（下一阶段）

所有目标仍以 OOS 或滚动样本为准。

| 指标 | Phase 0–7 结果 | 下阶段目标 |
|---|---:|---:|
| 年化收益中位数 | 2.43% | ≥ 5% |
| Sharpe 中位数 | 0.26 | ≥ 0.40 |
| Calmar 中位数 | 0.040 | ≥ 0.20 |
| 最大回撤中位数 | 47.11% | ≤ 38% |
| Calmar > 0.5 标的数 | 3 / 25 | ≥ 8 / 25 |
| **组合 OOS Sharpe** | （未运行） | ≥ 0.75 |
| **组合最大回撤** | （未运行） | ≤ 28% |
| WFO IS/OOS 相关 | -0.35 ~ +0.69 | 多数标的 > 0 |

---

## 4. 实施阶段

---

### 阶段 8：Meta-labeling 完整接入（最高优先级）

**为什么是最高优先级**：模型已训练好，特征已计算好，只差最后的接入。这是所有改动里实现成本最低、收益最确定的一项。

#### 8.1 在回测中实时预测 p_win

**改动文件**：`src/backtest/single_stock.py`

在 `run_single_stock_backtest()` 中新增 `meta_model` 参数：

```python
def run_single_stock_backtest(
    ohlcv, params, *,
    meta_model=None,           # 新增：训练好的 MiniBatchLogisticRegression 或 None
    meta_label_threshold=0.50, # 新增：p_win 低于此值跳过交易
    meta_label_mode="hard",    # 新增：hard / scale / off
    ...
)
```

在产生 BUY 信号时的执行逻辑中：

```python
if meta_model is not None and str(meta_label_mode) != "off":
    feat = _extract_signal_features_at(df, idx, index_ohlcv=index_ohlcv)
    p_win = meta_model.predict_proba(feat[np.newaxis, :])[0]
    if meta_label_mode == "hard" and p_win < meta_label_threshold:
        continue  # 跳过此次 BUY
    elif meta_label_mode == "scale":
        position_frac *= max(0.3, (p_win - 0.40) / 0.40)
```

需要新增辅助函数 `_extract_signal_features_at(df, idx, index_ohlcv)` 从 `build_signal_features()` 提取单行特征向量，避免重复计算整个特征矩阵。

#### 8.2 WFO 内部自动训练 Meta-label

**改动文件**：`src/backtest/wfo.py`

在 `_run_wfo_fold()` 的 train 阶段，除了参数搜索，还要：

1. 用 `build_training_samples()` 在 train 窗口内构建样本
2. 用 `run_meta_label_wfo()` 或直接实例化 `MiniBatchLogisticRegression` 训练
3. 将训练好的模型传给 OOS 评估的 `run_single_stock_backtest()`

这样 meta-label 就自然嵌入到 WFO 评估链路中，不存在未来函数泄露。

#### 8.3 验收标准

| 测试 | 期望 |
|---|---|
| `meta_label_mode=hard, threshold=0.55` 时交易次数比 baseline 减少 | 减少 15%–40% |
| 减少交易后 OOS Sharpe | 不低于 baseline |
| 减少交易后 OOS Calmar | 高于 baseline |
| 单元测试：`test_meta_label.py` 中增加"接入回测"集成测试 | 通过 |

**改动文件清单**：

| 文件 | 内容 |
|---|---|
| `src/backtest/single_stock.py` | 增加 `meta_model`、`meta_label_threshold`、`meta_label_mode` 参数；实现 `_extract_signal_features_at()` |
| `src/backtest/wfo.py` | 在 fold train 阶段训练 meta-label；fold OOS 时传入模型 |
| `scripts/run_backtest_single.py` | `--meta-label-threshold`、`--meta-label-mode` CLI 参数 |
| `scripts/run_wfo.py` | `--enable-meta-label` 参数，透传到 WFO |
| `tests/test_meta_label.py` | 增加 end-to-end 接入回测的集成测试 |

---

### 阶段 9：行业与个股中长期趋势过滤

**为什么必须做**：Bottom 5 标的（迈瑞医疗、恒瑞医药、伊利股份等）在长达 2–3 年的行业下行中持续接到假信号。只靠大盘指数（沪深 300）无法识别行业性熊市。

#### 9.1 行业相对强度特征

**新增文件**：`src/features/sector_features.py`

```python
def compute_sector_relative_strength(
    ohlcv: pd.DataFrame,
    sector_ohlcv: pd.DataFrame,  # 行业指数日线
    *,
    windows: tuple = (20, 60, 120),
) -> pd.DataFrame:
    """
    计算个股相对行业指数的相对收益率，用于判断个股在行业中的强弱位置。
    返回 rs_20、rs_60、rs_120 列。
    """
```

需要配合 `src/data_fetcher/index_benchmarks.py` 扩展行业指数（医疗、消费、金融、科技、周期）的数据获取。

#### 9.2 中长期趋势过滤规则

在 `run_single_stock_backtest()` 中新增过滤开关：

```yaml
signal_filter:
  require_above_ma120: false        # 收盘价需在 120 日均线上方才允许 BUY
  require_positive_rs60: false      # 60 日相对沪深 300 需为正
  sector_ma_filter: false           # 行业指数 60 日均线需向上
  sector_rs_window: 60              # 行业相对强度窗口
```

**过滤逻辑**（按优先级）：

```
Level 1（最宽松）：收盘价在 MA120 上方
  → 直接在当前 single_stock.py 中实现，不依赖行业数据

Level 2（中等）：个股 60 日收益跑赢沪深 300 60 日收益
  → 需要 index_ohlcv 作为参数（已有）

Level 3（最严格）：行业指数 60 日均线向上 + 个股相对行业 RS 为正
  → 需要新增行业指数数据
```

建议先实现 Level 1 和 Level 2，不强制依赖行业数据。

#### 9.3 预期效果

对 Bottom 5 标的：
- 恒瑞医药：2021 年 2 月股价跌破 MA120 后，所有新 BUY 应被过滤，可避免 2021–2023 年的连续亏损
- 伊利股份：类似，2022 年后低于 MA120，应停止做多
- 迈瑞医疗：2021 年 9 月后持续低于 MA120，应全年空仓

预计 Bottom 5 亏损减少 50%–70%，Calmar 中位数显著提升。

**改动文件清单**：

| 文件 | 内容 |
|---|---|
| `src/features/sector_features.py` | 新建：行业相对强度特征 |
| `src/backtest/single_stock.py` | 增加 `require_above_ma120`、`require_positive_rs60` 参数 |
| `src/data_fetcher/index_benchmarks.py` | 扩充行业指数 symbol 列表和获取逻辑 |
| `config.yaml.example` | 补充 `signal_filter.require_above_ma120` 等字段 |
| `tests/test_signal_filters.py` | 增加 MA120 过滤和 RS 过滤测试 |

---

### 阶段 10：Portfolio 层完整运行与 OOS 验证

**为什么必须做**：Portfolio 层代码写好了但从未产出任何数字。若组合 Sharpe 能达到 0.75，则说明横截面选股是最主要的 Alpha 来源。

**执行状态（2026-05-14）**：代码层已完成，并已运行真实数据首轮实验。入口脚本、meta-label `p_win` 排名权重、MA120/RS60 过滤透传、分数/权重/摘要导出和组合端测试均已落地；首跑结果未达目标（Sharpe 0.06，MDD 64.34%），需继续优化 WFO 稳定性、特征和组合 ranking。

#### 10.1 补全 run_portfolio_backtest.py 脚本

**新建文件**：`scripts/run_portfolio_backtest.py`

```bash
python scripts/run_portfolio_backtest.py \
  --watchlist configs/watchlist_25.txt \
  --start 2020-01-01 --end 2026-05-08 \
  --n-top 5 \
  --max-per-stock 0.25 \
  --index-symbol 510300 \
  --export-summary data/output/portfolio_summary.csv \
  --export-html
```

脚本需要：
1. 从 DuckDB 批量读取所有 watchlist 股票的日线数据
2. 同时读取沪深 300（510300）作为 `index_ohlcv`
3. 调用 `run_portfolio_backtest()` 并输出结果
4. 生成 HTML 报告（复用 `src/backtest/report.py`）

#### 10.2 将 Meta-label p_win 整合进排名评分

**改动文件**：`src/portfolio/signal_ranker.py`

当前 `rank_signals()` 的评分权重：

```python
score = (
    0.20 * trend_strength
  + 0.25 * relative_strength
  + 0.20 * momentum
  + 0.15 * donchian_breakout
  + 0.10 * liquidity
  + 0.10 * low_volatility
)
```

改进后，增加 Meta-label 分数权重：

```python
score = (
    0.25 * meta_label_p_win      # 新增：模型胜率预测（最高权重）
  + 0.20 * relative_strength
  + 0.15 * trend_strength
  + 0.15 * ma120_position        # 新增：是否在 MA120 上方
  + 0.10 * momentum
  + 0.10 * donchian_breakout
  + 0.05 * liquidity
)
```

#### 10.3 WFO-based 组合回测

为防止过拟合，组合参数（`n_top`、权重方案、最小质量分）也应走 WFO 评估：

| 参数 | 候选值 |
|---|---|
| `n_top` | [3, 5, 8] |
| `weighting` | `equal`, `vol_inverse`, `score_weighted` |
| `min_meta_score` | [0.45, 0.50, 0.55] |
| `require_above_ma120` | [True, False] |

**验收目标**：

| 指标 | 目标 |
|---|---|
| 组合 OOS 年化收益 | ≥ 8% |
| 组合 OOS Sharpe | ≥ 0.75 |
| 组合最大回撤 | ≤ 28% |
| 组合 Calmar | ≥ 0.30 |
| 年换手率 | 可解释且成本后仍盈利 |

**改动文件清单**：

| 文件 | 内容 |
|---|---|
| `scripts/run_portfolio_backtest.py` | 新建：组合回测入口脚本 |
| `src/portfolio/signal_ranker.py` | 接入 meta-label p_win 分数 |
| `src/portfolio/backtest.py` | 支持 WFO 模式评估组合参数 |
| `tests/test_portfolio.py` | 增加端到端组合回测测试 |

---

### 阶段 11：Meta-label 特征工程 v2

**为什么做**：当前 `build_signal_features()` 特征全部来自短期技术指标（20日、60日均线、14日ATR等）。Meta-label 的预测能力受限于特征质量。

#### 11.1 缺失的高价值特征

| 特征类别 | 具体特征 | 计算方式 |
|---|---|---|
| **52 周价格位置** | `pos_52w` | `(close - low_252) / (high_252 - low_252)` |
| **价量背离** | `pv_diverge` | 价格上涨但量能连续 5 日下降 = 分歧信号 |
| **趋势一致性** | `trend_consistency_20` | 过去 20 日中价格高于 5 日均线的天数比例 |
| **MACD 柱体方向** | `macd_hist_dir` | MACD 柱体是否连续 3 日扩大 |
| **换手率分位** | `turnover_rank_60` | 当日换手率在 60 日历史中的分位数 |
| **个股 Beta** | `beta_120` | 相对沪深 300 的 120 日滚动 Beta |
| **价格加速度** | `close_accel_10` | MA10 的二阶差分（趋势加速/减速） |
| **量价相关** | `vol_price_corr_20` | 20 日内成交量与价格变化的相关系数 |

#### 11.2 特征稳定性检验

新增 `tests/test_signal_features_stability.py`：

```python
def test_feature_importance_stability_across_folds():
    """
    验证 meta-label 模型在不同 WFO fold 中的 top-3 重要特征
    不能完全随机（至少有 2 个特征在 > 50% 的 fold 中出现在 top-3）
    """
```

如果特征重要性在不同 fold 间完全随机，说明模型没有学到真实规律，应放弃对应特征。

#### 11.3 LightGBM 变体（可选）

当每只股票的 BUY 信号积累到 50+ 条（约 2–3 年数据）后，可尝试用 LightGBM 替换逻辑回归：

```python
# src/models/meta_label_gbm.py
from lightgbm import LGBMClassifier

class GBMMetaLabel:
    def __init__(self, n_estimators=50, max_depth=3, min_child_samples=10):
        """
        限制深度和最小叶节点样本数，防止过拟合。
        单股样本少，不建议超过 50 棵树、深度 > 4。
        """
```

注意：**逻辑回归应保留为 baseline**，GBM 只有在 WFO 验证中明显优于逻辑回归时才切换。

**改动文件清单**：

| 文件 | 内容 |
|---|---|
| `src/models/meta_label.py` | 新增 11.1 中的特征到 `build_signal_features()` |
| `src/models/meta_label_gbm.py` | 新建：GBM 变体（可选） |
| `tests/test_signal_features_stability.py` | 新建：特征稳定性检验 |

---

### 阶段 12：WFO 稳定性架构升级

**为什么做**：4/5 标的 IS/OOS 相关系数为负，当前 WFO 选参数是在过拟合。

#### 12.1 跨 Fold 稳定性加权

**改动文件**：`src/backtest/wfo.py`

当前逻辑：每个 fold 独立选最优参数，fold 间参数可以完全不同。

新逻辑：对参数组合计算"跨 fold 稳定性分"，选择**表现稳定但不一定最高**的参数区域：

```python
def _select_stable_params(fold_results: list[dict]) -> dict:
    """
    对所有参数组合，计算：
    - is_score_mean: 平均 IS 复合分
    - is_score_std: IS 分的标准差（越小越稳定）
    - oos_score_mean: 平均 OOS 分（用于验证，不选择）

    选择标准：
    stability_score = is_score_mean / (is_score_std + 0.1)

    返回 stability_score top-5 的参数中位数（而不是最高 IS 的参数）
    """
```

#### 12.2 参数漂移惩罚

在 WFO 目标函数中加入相邻 fold 参数跳变惩罚：

```python
stability_penalty = abs(current_best_param_set - previous_fold_best_params).sum()
score -= 0.05 * stability_penalty
```

如果相邻 fold 最优参数差异过大（如 macd_fast 从 8 跳到 14），说明参数敏感，应降权。

#### 12.3 参数热力图输出

**改动文件**：`scripts/run_wfo.py`

当前已有 `--plot-heatmap` 选项，但热力图只显示 IS Sharpe。改进为输出三层热力图：
- IS 复合分热力图
- OOS 复合分热力图
- IS/OOS 相关性热力图（颜色越绿说明参数越稳定）

#### 12.4 最小 fold 数量要求

当 WFO fold 数量 < 5 时（数据太短），禁止使用 WFO 参数选择结果，回退到默认参数。

**改动文件清单**：

| 文件 | 内容 |
|---|---|
| `src/backtest/wfo.py` | `_select_stable_params()`；稳定性加权；参数漂移惩罚 |
| `scripts/run_wfo.py` | 三层热力图输出 |
| `tests/test_wfo_params_split.py` | 新增稳定性选择的单元测试 |

---

### 阶段 13：多周期信号确认（减少假信号）

**为什么做**：当前所有信号都是日线维度，没有更大周期（周线）的趋势确认。在日线震荡但周线仍向下时，日线 DK 翻红是假信号的概率极高。

#### 13.1 周线趋势状态

**新增文件**：`src/features/weekly_trend.py`

```python
def compute_weekly_trend_state(
    daily_ohlcv: pd.DataFrame,
    *,
    ma_windows: tuple = (5, 13),  # 周线约 = 日线 25/65 日
) -> pd.Series:
    """
    将日线数据聚合为周线 OHLCV，计算周线 MA5/MA13 方向，
    返回每个交易日对应的周线趋势状态：'bullish', 'bearish', 'neutral'
    """
```

#### 13.2 多周期共振过滤

在 `run_single_stock_backtest()` 中新增：

```yaml
signal_filter:
  require_weekly_bullish: false  # 周线趋势向上时才允许日线 BUY 信号
  weekly_ma_fast: 5              # 周线快均线（周数）
  weekly_ma_slow: 13             # 周线慢均线（周数）
```

**预期效果**：在震荡市中减少 30%–50% 的假 BUY 信号（以历史回测为准）。

#### 13.3 月度趋势状态（可选）

对于 Calmar 极低的标的，还可加入月线 MA3/MA6 状态。但月线信号变化极慢，仅适合作为"允许/不允许"的开关，而不是精确择时工具。

**改动文件清单**：

| 文件 | 内容 |
|---|---|
| `src/features/weekly_trend.py` | 新建：日线聚合周线趋势计算 |
| `src/backtest/single_stock.py` | 接入周线趋势过滤参数 |
| `tests/test_weekly_trend.py` | 新建：聚合逻辑、周线方向测试 |

---

### 阶段 14：动态仓位精细化

**为什么做**：当前最大回撤 47% 的一个重要来源是**入场时机虽对但仓位过满**——特别是在高波动期，波动率目标仓位没有足够压低杠杆。

#### 14.1 EWMA 波动率估计替代简单滚动窗口

**改动文件**：`src/backtest/single_stock.py`

当前波动率估计使用 20 日简单滚动标准差，对波动突变反应慢。替换为 EWMA：

```python
def _ewma_volatility(returns: pd.Series, span: int = 20) -> pd.Series:
    """指数加权移动波动率，对近期波动变化更敏感。"""
    return returns.ewm(span=span).std() * np.sqrt(252)
```

当 EWMA 波动率 > 1.5 倍历史中位数时，仓位自动降至 50%。

#### 14.2 仓位决策树（明确优先级）

将当前散落在不同地方的仓位调整逻辑统一为一棵决策树：

```
base_position = 1.0
  × meta_label_scale          （阶段 8）
  × ma120_position_scale       （阶段 9：MA120 以下 = 0.5）
  × weekly_trend_scale         （阶段 13：周线熊市 = 0.5）
  × volatility_target_scale    （目标波动率 / 当前 EWMA 波动率）
  × drawdown_throttle_scale    （回撤节流，现有）
  × market_regime_scale        （大盘状态，现有）
  clip(0.0, position_size_cap)
```

所有乘子明确定义，不隐式叠加。

#### 14.3 止损距离与仓位联动

当 ATR 止损距离超过 `risk_per_trade_pct` 隐含的仓位时，**强制以风险为约束**而不是以目标仓位为约束：

```python
if atr_stop_multiplier > 0:
    atr_stop_distance = atr * atr_stop_multiplier / close
    if risk_per_trade_pct > 0:
        risk_based_position = risk_per_trade_pct / atr_stop_distance
        position_frac = min(position_frac, risk_based_position)
```

**改动文件清单**：

| 文件 | 内容 |
|---|---|
| `src/backtest/single_stock.py` | EWMA 波动率；统一仓位决策树；ATR 止损联动 |
| `tests/test_position_management.py` | 新增 EWMA 波动率和仓位决策树测试 |

---

### 阶段 15：实验闭环与报告完善

**执行状态（2026-05-14）**：首版已完成。实验目录创建时会写入 `ARTIFACTS.md` 标准产物清单和 `DELTA.md` 占位；新增实验指标读取、指标方向判断、Bootstrap CI 重叠标记、HTML 对比报告和 Markdown delta 输出；新增 `scripts/compare_experiments.py` 作为命令行入口。

#### 15.1 实验目录结构强化

当前已有实验目录结构，补充以下内容，并由 `src/backtest/experiment.py` 维护标准产物清单：

```text
data/output/experiments/{YYYYMMDD}_{exp_id}/
  config.yaml                    # 已有
  batch_summary.csv              # 已有
  wfo_summary.csv                # 已有
  trade_attribution.csv          # 已有
  portfolio_summary.csv          # 新增（阶段 10）
  meta_label_calibration.csv     # 新增（阶段 8）
  feature_importance.csv         # 新增（阶段 11）
  regime_breakdown.csv           # 已有
  stability_heatmap.html         # 新增（阶段 12）
  report.html                    # 已有
  DELTA.md                       # 新增：与上一个实验的差异摘要
```

#### 15.2 自动化对比报告

已新增 `scripts/compare_experiments.py`：

```bash
python scripts/compare_experiments.py \
  --baseline data/output/experiments/E07_phase7_complete \
  --current  data/output/experiments/E08_meta_label_integrated \
  --output   data/output/experiments/comparison.html \
  --write-delta
```

输出标准化对比表：每个指标的变化幅度、方向判断、CSV 明细，以及统计显著性辅助标记（当输入摘要包含 `*_ci_low` / `*_ci_high` 时计算 Bootstrap CI 是否重叠）。`--write-delta` 会把摘要写回当前实验目录的 `DELTA.md`。

---

## 5. 实验矩阵

| ID | 实验 | 主要假设 | 核心文件 | 单股验收 | 组合验收 |
|---|---|---|---|---|---|
| E8a | Meta-label hard 过滤（threshold=0.52） | 过滤低胜率信号减少亏损 | `single_stock.py`, `meta_label.py` | OOS Calmar +20% | — |
| E8b | Meta-label scale 仓位（0.3~1.0） | 高胜率信号满仓，低胜率轻仓 | `single_stock.py` | MDD ↓，收益持平 | — |
| E9a | MA120 过滤（require_above_ma120=True） | Bottom 股票主要亏损在 MA120 下方 | `single_stock.py` | Bottom 5 亏损 -50% | — |
| E9b | RS60 过滤（require_positive_rs60=True） | 只做跑赢大盘的股票 | `single_stock.py` | Sharpe 提升 | — |
| E10 | 运行 portfolio 并报告 OOS 数字 | 组合 Sharpe 应明显高于单股中位数 | `run_portfolio_backtest.py` | — | Sharpe ≥ 0.75 |
| E11a | Meta-label 新增特征（52W 位置、Beta、加速度） | 更好特征提升预测准确率 | `meta_label.py` | 精确率提升 ≥ 5% | — |
| E11b | LightGBM 替换逻辑回归 | GBM 对非线性有更强表达 | `meta_label_gbm.py` | OOS Sharpe > 逻辑回归 | — |
| E12 | WFO 稳定性加权选参 | 减少过拟合，IS/OOS 相关转正 | `wfo.py` | IS/OOS 相关 > 0 | — |
| E13 | 周线趋势过滤 | 周线熊市中日线 BUY 是假信号 | `weekly_trend.py` | 假信号减少，胜率提升 | — |
| E14 | EWMA 波动率 + 统一仓位决策树 | 高波动期减仓控回撤 | `single_stock.py` | MDD ≤ 38% | — |
| E_FINAL | 全部开关组合最优配置 | 各改进叠加效果验证 | 所有 | Sharpe ≥ 0.40 | Sharpe ≥ 0.75 |

---

## 6. 推荐实施顺序

### 第 1 周（最高 ROI，代码已存在）

优先完成 **E8a**（Meta-label hard 过滤接入）和 **E9a**（MA120 过滤）：

1. 在 `single_stock.py` 中增加 `meta_model` 参数和 `_extract_signal_features_at()`
2. 在 WFO 的 fold 内部添加 meta-label 训练和传递逻辑
3. 增加 `require_above_ma120=True` 参数
4. 对 Bottom 5 标的单独跑验证：`run_backtest_single.py --symbol 600276 --require-above-ma120`
5. 重跑 batch_summary，建立新 baseline

输出：`data/output/experiments/E08_E09_baseline/`

### 第 2 周（组合层上线）

完成 **E10**（Portfolio 层运行）：

1. 编写 `scripts/run_portfolio_backtest.py`
2. 把 `p_win` 接入 `signal_ranker.py`
3. 对 25 只 watchlist 运行完整组合回测
4. 生成 `portfolio_summary.csv` 和 HTML 报告
5. 跑 WFO 版本（组合参数 n_top=3/5/8 + weighting 方案）

输出：`data/output/experiments/E10_portfolio_oos/`

### 第 3 周（特征和 WFO 优化）

完成 **E11a**（新特征）和 **E12**（WFO 稳定性）：

1. 在 `build_signal_features()` 中添加新特征
2. 对 5 只 WFO 标的重跑，验证 IS/OOS 相关性是否改善
3. 实现稳定性加权参数选择
4. 输出三层热力图

输出：`data/output/experiments/E11_E12_wfo_stable/`

### 第 4 周（多周期 + 仓位整合）

完成 **E13**（周线过滤）和 **E14**（统一仓位决策树）：

1. 实现 `weekly_trend.py`
2. 统一仓位决策树，消除隐式叠加
3. 用全量 watchlist 跑最终组合

最终目标：`E_FINAL` 组合 Sharpe ≥ 0.75，单股 Sharpe 中位数 ≥ 0.40。

---

## 7. 验证命令

### Meta-label 接入验证（阶段 8）

```bash
# 无 meta-label（baseline）
python scripts/run_backtest_single.py --symbol 600276 --start 2020-01-01 --end 2026-05-08

# 开启 meta-label（hard 过滤）
python scripts/run_backtest_single.py --symbol 600276 --start 2020-01-01 --end 2026-05-08 \
  --meta-label-mode hard --meta-label-threshold 0.52

# WFO 内含 meta-label
python scripts/run_wfo.py --symbol 300750 --start 2020-01-01 --end 2026-05-08 \
  --enable-meta-label --train-days 504 --oos-days 126
```

### MA120 过滤验证（阶段 9）

```bash
# 对 Bottom 5 标的验证 MA120 过滤效果
for sym in 300760 600276 600887 600585 000651; do
  python scripts/run_backtest_single.py --symbol $sym \
    --start 2020-01-01 --end 2026-05-08 \
    --require-above-ma120 \
    --export-html
done
```

### 组合回测（阶段 10）

```bash
python scripts/run_portfolio_backtest.py \
  --watchlist configs/watchlist_25.txt \
  --start 2020-01-01 --end 2026-05-08 \
  --n-top 5 \
  --enable-meta-label \
  --require-above-ma120 \
  --export-summary data/output/portfolio_final.csv \
  --export-html
```

### WFO 稳定性检验（阶段 12）

```bash
# 验证 IS/OOS 相关性是否改善
for sym in 002475 300059 300750 600036 601166; do
  python scripts/run_wfo.py --symbol $sym \
    --start 2020-01-01 --end 2026-05-08 \
    --train-days 504 --oos-days 126 \
    --stability-weighting \
    --plot-heatmap \
    --export-results
done
```

### 周线过滤与 EWMA 仓位验证（阶段 13/14）

```bash
# 周线趋势过滤：只允许周线 bullish 时执行日线 BUY
python scripts/run_backtest_single.py --symbol 600276 \
  --start 2020-01-01 --end 2026-05-08 \
  --require-weekly-bullish --weekly-ma-fast 5 --weekly-ma-slow 13

# EWMA 波动率目标仓位 + 高波动折扣
python scripts/run_backtest_single.py --symbol 300750 \
  --start 2020-01-01 --end 2026-05-08 \
  --volatility-target-ann 0.18 \
  --volatility-lookback 20 \
  --volatility-high-vol-multiple 1.5 \
  --volatility-high-vol-scale 0.5
```

---

## 8. 文件级路线图

| 文件/目录 | 阶段 | 改动内容 |
|---|---|---|
| `src/backtest/single_stock.py` | 8, 9, 13, 14 | meta_model 参数；require_above_ma120；EWMA 波动率；统一仓位决策树 |
| `src/backtest/wfo.py` | 8, 12 | fold 内 meta-label 训练；稳定性加权选参；参数漂移惩罚 |
| `src/models/meta_label.py` | 11 | 新增 8 个特征到 `build_signal_features()` |
| `src/models/meta_label_gbm.py` | 11 | 新建：GBM 变体（可选） |
| `src/features/sector_features.py` | 9 | 新建：行业相对强度特征 |
| `src/features/weekly_trend.py` | 13 | 新建：周线趋势聚合 |
| `src/portfolio/signal_ranker.py` | 10 | 接入 p_win 作为核心权重 |
| `src/portfolio/backtest.py` | 10 | 支持 WFO 模式 |
| `src/data_fetcher/index_benchmarks.py` | 9 | 扩充行业指数 symbol 列表 |
| `scripts/run_backtest_single.py` | 8, 9, 13 | CLI：`--meta-label-*`、`--require-above-ma120`、`--require-weekly-bullish` |
| `scripts/run_wfo.py` | 8, 12 | `--enable-meta-label`、`--stability-weighting`；三层热力图 |
| `scripts/run_portfolio_backtest.py` | 10 | 新建：组合回测入口 |
| `scripts/compare_experiments.py` | 15 | 新建：实验对比报告 |
| `config.yaml.example` | 8, 9, 13 | 补充新增参数的示例配置 |
| `tests/test_meta_label.py` | 8, 11 | 端到端接入回测集成测试；特征稳定性测试 |
| `tests/test_signal_filters.py` | 9, 13 | MA120 过滤；RS60 过滤；周线过滤测试 |
| `tests/test_wfo_params_split.py` | 12 | 稳定性选参单元测试 |
| `tests/test_position_management.py` | 14 | EWMA 波动率；统一仓位决策树 |

---

## 9. 风险和边界条件

1. **Meta-label 样本不足问题**：单股 6 年日线数据中 BUY 信号通常只有 50–60 次，逻辑回归样本勉强够用，GBM 要设置严格正则化。绝对不能用随机切分，必须 WFO。

2. **MA120 过滤可能过滤太多**：对于强趋势股（宁德时代、立讯精密），它们本来就在 MA120 上方，过滤不影响；但对于波动大的股票，可能会错过底部反转。建议先跑验证再决定是否默认开启。

3. **行业数据可用性**：行业指数数据依赖 AkShare，若数据不完整需降级到沪深 300 作为通用过滤。

4. **组合层的"过拟合"风险**：组合参数（n_top、权重方案）也需要 WFO 验证，不能只看全样本收益。

5. **不追求零回撤**：Calmar 目标 0.20 对应的是 25% 最大回撤下 5% 年化，是合理的单股趋势策略目标，不应为了降低回撤而完全空仓。

6. **成功标准仍然是 OOS**：任何改动只有在 OOS 指标改善时才算通过。全样本收益提升但 WFO OOS 恶化的改动一律放弃。

---

## 10. 执行优先级清单

按 ROI 从高到低：

1. **接入 Meta-label**（`single_stock.py` + `wfo.py`）——模型已有，只差最后一步，ROI 极高
2. **MA120 + RS60 过滤**——可直接切断 Bottom 5 的主要亏损来源，实现简单
3. **运行 Portfolio 回测脚本**——填补最大数据空白，验证组合 Sharpe 假设
4. **WFO 稳定性加权**——修正 IS/OOS 负相关问题，让参数选择更可信
5. **新增 Meta-label 特征**（52W、Beta、加速度、量价相关）——提升模型表达力
6. **周线趋势过滤**——进一步减少震荡市假信号
7. **统一仓位决策树 + EWMA 波动率**——精细化风控，压低尾部回撤
8. **实验对比报告自动化**——提升研究效率，不直接创造 Alpha
