# 改造计划：月度选股系统 → 个股多空趋势操盘系统

**日期**：2026-05-12  
**原项目**：`quant-stock-advisor`（A 股月度截面选股 + LTR + 组合优化）  
**新目标**：针对指定个股，在日线级别实时计算「多空趋势」指标，自动生成买入/卖出信号，并提供历史回测验证

---

## 一、改造动机与核心思路

### 1.1 原项目定位 vs 新目标

| 维度 | 原项目 | 新系统 |
|---|---|---|
| **问题类型** | 选股（哪些股票值得持有？） | 择时（这只股票什么时候买卖？） |
| **操作对象** | A 股全市场截面，月度 Top-K 组合 | 单只（或少数几只）自选股 |
| **核心模型** | LightGBM LTR、因子 IC、组合优化 | 技术指标（多空趋势线）的规则信号 |
| **调仓频率** | 月度 | 日线级别，信号变色时触发 |
| **持仓逻辑** | 分散持仓、行业约束、风险平价 | 单票满仓 / 空仓，信号驱动进出 |
| **数据依赖** | 基本面、资金流、股东户数、行业宽度 | 仅日线 OHLCV（复权） |

### 1.2 灵感来源：东方财富「多空趋势」

东方财富 App 日 K 图下方的「多空趋势」柱状图（也称「DK 点」）：
- 柱子**变红**（由绿转红）= 多头趋势启动，触发**买入信号**
- 柱子**变绿**（由红转绿）= 空头趋势启动，触发**卖出信号**
- 同色柱子连续出现是常态（趋势延续），避免高换手

### 1.3 指标逆向工程策略

东方财富的公式为私有实现，但可从外观与行为推断其本质为「趋势强度归零再反转」型指标。新系统实现以下三种等效近似，通过配置切换，回测后保留最优：

| 方案 | 公式核心 | 特点 |
|---|---|---|
| `macd_cross`（**默认**）| `DIFF = EMA(12) - EMA(26)`; `DEA = EMA(DIFF, 9)`; 红 = `DIFF > DEA` | 视觉最接近，换手较低 |
| `ma_cross` | `MA5 > MA20` 为红，加 EMA 平滑去除单日噪声 | 直观，略微领先 |
| `boll_trend` | 收盘 > BOLL 中轨（MA20）为红，< 为绿 | 最简洁，阻力较多 |

三种均可在 `config.yaml` 中一键切换，互为回测对照组。

---

## 二、文件去留决策

### 2.1 直接保留（无需修改）

```
src/
  data_fetcher/
    akshare_client.py          # AkShare 日线拉取、列名规范化 ✅
    akshare_resilience.py      # 超时 / 重试 / 本地缓存回退 ✅
    db_manager.py              # DuckDB 增量写入管理器 ✅
    data_quality.py            # 写入前质量校验 ✅
    stock_name_cache.py        # 股票名称查询缓存 ✅
    __init__.py                ✅
  market/
    tradability.py             # 涨跌停检测、停牌近似 ✅（单股回测时需排除一字涨停）
    __init__.py                ✅
  backtest/
    performance_panel.py       # 年化/夏普/Calmar/最大回撤/胜率 ✅
    transaction_costs.py       # 印花税 + 佣金成本模型 ✅
    risk_metrics.py            # max_drawdown 等风险统计 ✅
    __init__.py                ✅
  notify/
    __init__.py                # 企业微信 Webhook 推送 ✅（信号变色时推送）
  settings.py                  # 配置加载、project_root ✅
  logging_config.py            ✅
  event_log.py                 ✅

scripts/
  fetch_stock.py               # 保留并大幅精简（去掉全量宇宙拉取逻辑）
```

### 2.2 大幅裁剪后保留（保留文件，删除内部不相关函数）

```
src/backtest/engine.py
  保留：BacktestConfig, TieredImpactConfig, 底层日收益计算核心
  删除：walk_forward runner、组合权重输入分支（原来支持 Top-K 多股）
  → 新增：单股信号回测接口 run_single_stock_backtest()

src/data_fetcher/index_benchmarks.py
  保留：拉取沪深 300/中证 500 作为基准对比
  删除：复杂的多基准聚合逻辑

config.yaml.example
  保留：paths / database / akshare 节
  删除：所有月度选股参数（signals.top_k, portfolio.*, label.*, model.*）
  → 新增：trend_signal 节（indicator、params）
```

### 2.3 彻底删除（整目录）

```
src/features/          # 全部截面因子（14 个文件，约 17 万行）— 不需要
src/models/            # LightGBM、LTR、LSTM/TCN 模型 — 不需要
src/portfolio/         # 协方差、权重优化、约束 — 不需要
src/pipeline/          # 月度 pipeline（8千行）— 不需要
src/reporting/         # 月度选股报告 — 不需要
src/research/          # 研究治理、gates、manifest — 不需要
src/monitoring/        # OOS 跟踪 — 不需要
src/analysis/          # benchmark_suite、capacity_report — 不需要
src/cli/               # 月度选股 CLI — 不需要
```

### 2.4 彻底删除（脚本）

```
scripts/run_monthly_*.py       # 全部月度选股入口（13 个文件）
scripts/run_oracle_*.py        # oracle 诊断
scripts/run_factor_*.py        # 因子 IC 审计
scripts/run_portfolio_*.py     # 组合比较
scripts/run_regime_*.py        # 市场状态敏感性
scripts/run_pipeline_all_steps.sh
scripts/generate_m8_*.py
scripts/materialize_prepared_factors.py
scripts/load_lhb.py / load_margin_trading.py / load_northbound.py
scripts/load_concept.py
scripts/fetch_events.py / fetch_fundamental.py / fetch_fund_flow.py
scripts/fetch_shareholder.py / fetch_industry.py
scripts/build_industry_map.py
scripts/apply_factor_audit_results.py
scripts/check_lightgbm_retest_trigger.py
scripts/generate_dashboard.py
scripts/query_run_events.py
scripts/refresh_prepared_fund_flow_cache.py
scripts/research_identity.py
scripts/research_m13b_alternative_apis.py
scripts/run_m6_ltr_failure_attribution.py
```

### 2.5 文档/配置清理

```
删除：
  configs/experiments/         # 实验配置
  configs/promoted/            # 月度选股 promotion registry
  docs/archive/                # 历史计划文档
  docs/reports/                # 历史研究报告
  docs/monthly_selection_*.md  # 月度选股研究记录
  config.yaml.backtest         # 旧回测配置

保留并更新：
  configs/README.md            → 改写说明多空趋势配置示例
  docs/README.md               → 改写指向新文档
```

---

## 三、新增模块设计

### 3.1 `src/indicators/` — 多空趋势指标核心

```
src/indicators/
  __init__.py
  dktrend.py      # 多空趋势指标实现
  utils.py        # EMA / 滚动高低点等工具函数
```

#### `src/indicators/utils.py`

```python
def ema(series: pd.Series, period: int) -> pd.Series:
    """标准 EMA，min_periods=period。"""

def highest(series: pd.Series, period: int) -> pd.Series:
    """滚动 period 日最高价。"""

def lowest(series: pd.Series, period: int) -> pd.Series:
    """滚动 period 日最低价。"""
```

#### `src/indicators/dktrend.py`

```python
class TrendMode(str, Enum):
    MACD_CROSS = "macd_cross"   # DIFF vs DEA
    MA_CROSS   = "ma_cross"     # MA5 vs MA20（EMA平滑）
    BOLL_TREND = "boll_trend"   # 收盘 vs BOLL 中轨

@dataclass
class DKTrendParams:
    mode: TrendMode = TrendMode.MACD_CROSS
    # macd_cross 参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # ma_cross 参数
    ma_fast: int = 5
    ma_slow: int = 20
    ma_smooth: int = 3    # 对 (MA_fast - MA_slow) 再做 EMA 平滑
    # boll_trend 参数
    boll_window: int = 20

def compute_dktrend(
    df: pd.DataFrame,          # 含 close / high / low，index 为 trade_date
    params: DKTrendParams,
) -> pd.DataFrame:
    """
    返回 DataFrame，新增列：
      dk_value   : float，指标数值（DIFF-DEA / MA差 / close-中轨）
      dk_color   : str，'red'（多）或 'green'（空）
      dk_signal  : str，'buy'（绿→红）/ 'sell'（红→绿）/ ''（延续）
      dk_run_len : int，当前同色连续天数（1 表示刚变色当天）
    """
```

**颜色判定与信号逻辑：**

```
dk_color[t] = 'red'   if dk_value[t] > 0 else 'green'
dk_signal[t]:
  'buy'  当 dk_color[t] == 'red'  且 dk_color[t-1] == 'green'
  'sell' 当 dk_color[t] == 'green' 且 dk_color[t-1] == 'red'
  ''     其他（颜色未变）
```

### 3.2 `src/signals/` — 信号生成与状态管理

```
src/signals/
  __init__.py
  generator.py   # 从指标输出提取信号序列
  types.py       # Signal / Position 枚举
```

#### `src/signals/types.py`

```python
class Signal(str, Enum):
    BUY  = "buy"
    SELL = "sell"
    HOLD = "hold"   # 无新信号

class Position(str, Enum):
    LONG  = "long"
    FLAT  = "flat"
```

#### `src/signals/generator.py`

```python
@dataclass
class SignalRecord:
    trade_date: pd.Timestamp
    signal: Signal           # buy / sell / hold
    close: float
    dk_color: str            # red / green
    dk_run_len: int          # 同色已连续天数
    position_after: Position # 该信号发出后的仓位状态

def generate_signals(
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
) -> list[SignalRecord]:
    """对给定日线数据计算多空趋势并提取所有信号列表。"""

def get_current_signal(
    symbol: str,
    db_path: str,
    params: DKTrendParams,
) -> SignalRecord:
    """从 DuckDB 读取最近 N 天数据并返回最新信号状态。"""
```

### 3.3 `src/backtest/single_stock.py` — 单股回测

复用现有 `engine.py` 的成本模型与 `performance_panel.py` 的绩效统计，增加单股专属逻辑：

```python
@dataclass
class SingleStockBacktestResult:
    symbol: str
    stock_name: str
    period: str                      # "2020-01-01 ~ 2025-12-31"
    n_trades: int                    # 完整买卖周期数
    win_rate: float                  # 盈利交易占比
    avg_hold_days: float             # 平均持仓天数
    avg_return_per_trade: float      # 每次完整交易平均收益
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_return: float
    annualized_return: float
    buy_hold_return: float           # 同期持有不动的收益
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    trade_log: pd.DataFrame          # 每笔交易明细

def run_single_stock_backtest(
    symbol: str,
    ohlcv: pd.DataFrame,
    params: DKTrendParams,
    *,
    cost_bps: float = 15.0,         # 单边成本（印花税 10 + 佣金 5）
    initial_capital: float = 100_000.0,
) -> SingleStockBacktestResult:
    """
    回测逻辑：
    1. 计算 dk_signal 序列
    2. 遇到 'buy' → 次日开盘买入（T+1，规避一字涨停）
    3. 遇到 'sell' → 次日开盘卖出（T+1）
    4. 收集每笔交易的持仓收益（扣除成本）
    5. 汇总绩效
    """
```

### 3.4 新增脚本

#### `scripts/fetch_stock.py`（替换原 `fetch_only.py`，大幅精简）

```bash
# 拉取/更新单只股票日线
python scripts/fetch_stock.py --symbol 600930

# 批量拉取多只
python scripts/fetch_stock.py --symbols 600930 000001 300750

# 拉取并检查数据质量
python scripts/fetch_stock.py --symbol 600930 --check-quality
```

实现：直接调用 `DuckDBManager` 的增量拉取，去掉所有月度宇宙逻辑。

---

#### `scripts/run_signal.py` — 核心入口

```bash
# 查看当前信号
python scripts/run_signal.py --symbol 600930

# 查看最近 60 天信号历史
python scripts/run_signal.py --symbol 600930 --history 60

# 指定指标模式
python scripts/run_signal.py --symbol 600930 --mode macd_cross

# 多股批量扫描（找所有「刚变红」的股票）
python scripts/run_signal.py --watchlist 600930 000001 300750 --filter buy
```

**标准输出示例：**

```
华电新能 (600930)  |  最新交易日：2026-05-12
────────────────────────────────────────────
当前多空趋势：🔴 多头  |  已连续 3 天
最新收盘：6.45

近期信号记录：
  日期          信号    收盘     趋势色   连续天
  2026-04-28    BUY    6.08    🔴 多头    1
  2026-04-15    SELL   6.52    🟢 空头    1
  2026-03-20    BUY    5.87    🔴 多头    1

操作建议：持仓观望（多头趋势持续中）
```

---

#### `scripts/run_backtest_single.py` — 单股回测入口

```bash
# 回测华电新能，默认近 5 年
python scripts/run_backtest_single.py --symbol 600930

# 指定时间段
python scripts/run_backtest_single.py --symbol 600930 --start 2020-01-01 --end 2025-12-31

# 对比三种指标模式
python scripts/run_backtest_single.py --symbol 600930 --compare-modes

# 输出详细交易记录到 CSV
python scripts/run_backtest_single.py --symbol 600930 --export-trades
```

**标准输出示例：**

```
华电新能 (600930)  回测报告
回测区间：2021-01-01 ~ 2026-05-12  |  指标：macd_cross
──────────────────────────────────────────────────────
总收益率：+47.3%    买入持有：+32.1%    超额：+15.2%
年化收益：+8.1%     夏普比率：0.83      最大回撤：-18.4%

交易统计：
  总交易次数：23     平均持仓天数：28.6
  胜率：60.9%       平均盈利：+6.8%
  平均亏损：-3.2%   盈亏比：2.13
  最大连续盈利：4次  最大连续亏损：3次

──────────────────────────────────────────────────────
交易记录（最近 5 笔）：
  买入日        卖出日        买价    卖价    收益
  2026-02-18   2026-03-14   5.62   6.03   +7.3%
  2025-11-04   2025-12-19   5.18   5.51   +6.4%
  ...
```

---

### 3.5 通知推送增强（`src/notify/__init__.py`）

原有企业微信 Webhook 基础上，增加信号专用消息模板：

```python
def send_trend_signal(
    handler: WecomWebhookHandler,
    symbol: str,
    stock_name: str,
    signal: Signal,         # BUY / SELL
    close: float,
    dk_run_len: int,
    trade_date: str,
) -> bool:
    """
    推送格式：
    【多空趋势信号】华电新能 (600930)
    📈 买入信号  |  2026-05-12
    收盘价：6.45  |  趋势刚变红
    """
```

---

## 四、新配置文件结构（`config.yaml.example`）

```yaml
# ── 数据路径 ─────────────────────────────────────────────
paths:
  duckdb_path: data/market.duckdb
  output_dir: data/output
  # asof_trade_date: ""   # 留空=运行当日

# ── AkShare 行情拉取 ───────────────────────────────────────
akshare:
  adjust: qfq              # 前复权
  sleep_between_symbols_sec: 0.5
  max_fetch_retries: 3
  retry_delay_sec: 2.0
  request_timeout_sec: 10.0
  fetch_workers: 2

# ── DuckDB ─────────────────────────────────────────────────
database:
  table_daily: a_share_daily
  table_audit: data_fetch_audit

# ── 多空趋势指标 ──────────────────────────────────────────
trend_signal:
  mode: macd_cross         # macd_cross | ma_cross | boll_trend
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  ma_fast: 5
  ma_slow: 20
  ma_smooth: 3
  boll_window: 20

# ── 回测参数 ──────────────────────────────────────────────
backtest:
  cost_bps: 15.0           # 单边：印花税 10 + 佣金 5
  initial_capital: 100000  # 初始资金（元）
  execution: tplus1_open   # T+1 开盘执行

# ── 通知（可选）──────────────────────────────────────────
notify:
  wecom_webhook_url: ""    # 留空=不推送
  mention_all: false
```

---

## 五、改造后的目录树

```
quant-stock-advisor/
├── config.yaml.example          # 大幅精简后的配置模板
├── README.md                    # 全面改写
├── requirements.txt             # 删除 lightgbm, scikit-learn, xgboost 等
├── requirements-base.txt        # 保留 akshare, duckdb, pandas, numpy, pyyaml
│
├── src/
│   ├── __init__.py
│   ├── settings.py              # 保留（新增 trend_signal 节解析）
│   ├── logging_config.py        # 保留
│   ├── event_log.py             # 保留
│   ├── env_check.py             # 保留（精简依赖检查项）
│   │
│   ├── data_fetcher/            # 保留（全部保留，无删改）
│   │   ├── akshare_client.py
│   │   ├── akshare_resilience.py
│   │   ├── db_manager.py
│   │   ├── data_quality.py
│   │   ├── stock_name_cache.py
│   │   ├── index_benchmarks.py  # 精简
│   │   └── __init__.py
│   │
│   ├── market/
│   │   ├── tradability.py       # 保留
│   │   └── __init__.py
│   │
│   ├── indicators/              # ★ 新增
│   │   ├── __init__.py
│   │   ├── dktrend.py
│   │   └── utils.py
│   │
│   ├── signals/                 # ★ 新增
│   │   ├── __init__.py
│   │   ├── types.py
│   │   └── generator.py
│   │
│   ├── backtest/
│   │   ├── engine.py            # 精简（删除 walk_forward 多股逻辑）
│   │   ├── single_stock.py      # ★ 新增
│   │   ├── performance_panel.py # 保留
│   │   ├── transaction_costs.py # 保留
│   │   ├── risk_metrics.py      # 保留
│   │   └── __init__.py
│   │
│   └── notify/
│       └── __init__.py          # 保留并增加信号推送模板
│
├── scripts/
│   ├── fetch_stock.py           # ★ 新增（替换 fetch_only.py）
│   ├── run_signal.py            # ★ 新增（核心入口）
│   ├── run_backtest_single.py   # ★ 新增
│   ├── env_check.py             # 保留
│   └── akshare_network_doctor.py# 保留
│
├── tests/
│   ├── test_dktrend.py          # ★ 新增（指标计算单元测试）
│   ├── test_signals.py          # ★ 新增（信号生成测试）
│   └── test_single_stock_bt.py  # ★ 新增（回测结果合理性测试）
│
└── docs/
    ├── README.md                # 改写
    ├── indicator_formula.md     # ★ 新增：三种指标公式详解与对比
    └── backtest_guide.md        # ★ 新增：回测使用说明
```

---

## 六、实施步骤（有序）

### Phase 1：清场（1 天）

**目标**：删除所有无关代码，保证剩余代码仍可 `import` 不报错。

1. 删除整目录：`src/features/`、`src/models/`、`src/portfolio/`、`src/pipeline/`、`src/reporting/`、`src/research/`、`src/monitoring/`、`src/analysis/`、`src/cli/`
2. 删除脚本：所有 `scripts/run_monthly_*.py`、`run_oracle_*.py`、`run_factor_*.py` 等（见 2.4 节清单）
3. 删除文档：`docs/archive/`、`docs/reports/`、`docs/monthly_selection_*.md`、`config.yaml.backtest`、`configs/experiments/`、`configs/promoted/`
4. 精简 `src/backtest/engine.py`：移除 `walk_forward` 相关导入和函数（约 200 行），保留 `BacktestConfig` 和单日收益计算核心
5. 精简 `src/settings.py`：移除 `DEFAULT_CONFIG` 里的 `portfolio.*`、`signals.top_k` 等月度参数
6. 精简 `requirements.txt`：移除 `lightgbm`、`xgboost`、`scikit-learn`、`torch`、`scipy`（组合优化部分）、`polars`（非必需）

**验收**：`python -c "from src.data_fetcher.db_manager import DuckDBManager"` 无报错。

---

### Phase 2：指标核心（2 天）

**目标**：实现并验证三种多空趋势近似算法。

**Step 2.1**：实现 `src/indicators/utils.py`

- `ema(series, period)` —— 使用 `pandas.Series.ewm(span=period, adjust=False)`
- `highest(series, period)`、`lowest(series, period)` —— `rolling(period).max/min()`
- 单元测试：验证 EMA 值与 TradingView 参数一致（取已知数据对比）

**Step 2.2**：实现 `src/indicators/dktrend.py`

- 实现三种模式的 `compute_dktrend()`
- `dk_value`、`dk_color`、`dk_signal`、`dk_run_len` 四列均经过边界情况处理（NaN 初始段）
- 测试：用华电新能 (600930) 近 3 个月数据，与截图中 App 显示的柱子颜色对比验证

**Step 2.3**：视觉校验脚本（可选）

```python
# scripts/debug_indicator.py  （临时调试脚本，不纳入正式目录）
# 用 matplotlib 绘制 K 线 + 多空趋势柱，与东方财富截图对比
```

---

### Phase 3：信号系统（1 天）

**目标**：`src/signals/` 模块可正确生成信号序列并返回最新状态。

**Step 3.1**：实现 `src/signals/types.py`（`Signal`、`Position` 枚举）

**Step 3.2**：实现 `src/signals/generator.py`

- `generate_signals()` 输出 `list[SignalRecord]`，只在颜色变化时产生 BUY/SELL，其余为 HOLD
- 测试：验证「连续红不重复触发买入」「一次红→绿=一次卖出」的状态机行为

**Step 3.3**：实现 `scripts/run_signal.py`

- 解析 `--symbol`、`--symbols`、`--watchlist`、`--history`、`--mode`、`--filter`
- 调用 `DuckDBManager` 拉取最新数据
- 格式化输出（终端彩色 ANSI）

---

### Phase 4：单股回测（2 天）

**目标**：`run_backtest_single.py` 可输出完整回测报告。

**Step 4.1**：实现 `src/backtest/single_stock.py`

- T+1 开盘执行逻辑（`is_open_limit_up_unbuyable()` 过滤一字涨停）：若次日一字涨停则顺延至可成交的第一天开盘
- 扣除单边成本后计算每笔净收益
- 汇总 `SingleStockBacktestResult`（`n_trades`、`win_rate`、`avg_hold_days`、`total_return` 等）
- 复用 `PerformancePanel` 计算夏普/最大回撤/Calmar
- 增加「买入持有」基准对比

**Step 4.2**：实现 `scripts/run_backtest_single.py`

- `--compare-modes`：同一股票对比三种指标模式的回测结果（表格形式）
- `--export-trades`：将交易记录输出到 `data/output/{symbol}_trades.csv`

**Step 4.3**：测试

```python
# tests/test_single_stock_bt.py
# 用确定性的合成价格序列验证：
#   - 买入后持仓直到卖出，不跳单
#   - 成本正确扣除
#   - 一字涨停延迟执行
#   - 回测末尾未平仓则按最后收盘平仓
```

---

### Phase 5：数据拉取精简与收尾（1 天）

**Step 5.1**：实现 `scripts/fetch_stock.py`

- 基于 `DuckDBManager`，仅支持 `--symbol` / `--symbols` 参数
- 去掉月度宇宙全量拉取逻辑（原 `fetch_only.py` 有整个全市场扫描流程）
- 增加 `--check-quality` 标志：拉取后输出该股近 30 天的数据质量摘要

**Step 5.2**：重写 `config.yaml.example`（见四节）

**Step 5.3**：精简 `requirements.txt`，最终依赖清单：

```
akshare>=1.12
duckdb>=0.10
pandas>=2.0
numpy>=1.26
pyyaml>=6.0
requests>=2.31
```

可选（如需绘图调试）：`matplotlib>=3.8`

**Step 5.4**：重写 `README.md`

- 5 分钟快速上手教程
- 三步流程：`fetch_stock` → `run_signal` → `run_backtest_single`
- 指标公式说明与选择建议

**Step 5.5**：通知推送接入（可选）

- 在 `run_signal.py` 末尾：若 `config.yaml` 配置了 `notify.wecom_webhook_url`，且当日有 BUY/SELL 信号，自动推送

---

## 七、代码规模对比

| 维度 | 改造前 | 改造后 |
|---|---|---|
| Python 文件数（src） | ~80 | ~20 |
| 总代码行数（src） | ~25,000 | ~1,500 |
| 直接依赖包数 | ~12 | ~6 |
| 支持的核心功能 | 月度截面选股 | 个股多空趋势择时 |
| 新用户上手时间 | 数天（需理解整套量化体系） | 30 分钟 |

---

## 八、后续可扩展方向（本次不做）

1. **指标参数自动优化**：对某只股票用网格搜索找最优 `macd_fast/slow/signal` 参数组合（注意过拟合风险）
2. **多股批量扫描 + 自动预警**：每天收盘后自动扫描自选股列表，变色则推送
3. **Web Dashboard**：基于 Streamlit/Gradio 的图形界面，显示 K 线 + 多空趋势柱
4. **信号确认过滤器**：叠加成交量放量、BOLL 带宽等条件，提升信号质量
5. **止损规则**：在持仓期间如跌超 N% 则触发止损，不等待卖出信号

---

## 九、关键风险与注意事项

1. **指标逆向不完全**：本系统实现的是东方财富多空趋势的**近似复现**，三种模式与 App 原版在具体数值上可能存在差异。建议以回测结果而非视觉精确匹配作为验收标准。

2. **T+1 执行约束**：所有信号均在次日开盘执行。一字涨停时延迟买入，可能错过最大涨幅，这是合理的保守假设。

3. **数据调整问题**：使用前复权（qfq）日线，历史回测中涉及分红除权的时间段收益计算已自动处理；实盘须注意复权因子更新。

4. **单股集中风险**：系统仅做择时，不做分散。实际使用时建议对多只自选股分别运行，人工决定仓位大小。

5. **过拟合警告**：不要用同一支股票的历史数据优化指标参数，再用该股回测验证——这是典型的 in-sample 过拟合。参数应在宽泛的默认值（MACD 12/26/9 为业界通用）下使用，或用其他股票验证后再应用。

---

## 十、实施进度记录

### 2026-05-12

已完成：

1. Phase 1 清场基本完成：仓库当前仅保留单股趋势系统相关的 `src/data_fetcher/`、`src/indicators/`、`src/signals/`、`src/backtest/`、`src/market/`、`src/notify/` 等目录；旧的 `features/models/portfolio/pipeline/reporting/research/monitoring/analysis/cli` 目录已不存在。
2. Phase 2 指标核心完成：`src/indicators/dktrend.py` 已实现 `macd_cross`、`ma_cross`、`boll_trend` 三种模式，输出 `dk_value/dk_color/dk_signal/dk_run_len`；`tests/test_dktrend.py` 覆盖滚动工具和变色信号。
3. Phase 3 信号系统完成：`src/signals/` 已实现 `SignalRecord`、`generate_signals()`、`get_current_signal()`；`scripts/run_signal.py` 支持单股、多股、自选列表、历史输出、模式切换、信号过滤和企业微信推送。
4. Phase 4 单股回测完成：`src/backtest/single_stock.py` 已实现 T+1 开盘执行、买入遇一字涨停顺延、成本扣减、末尾持仓平仓和绩效汇总；`scripts/run_backtest_single.py` 支持三模式对比和交易记录导出。
5. Phase 5 数据拉取与配置文档基本完成：`scripts/fetch_stock.py` 支持单股/多股增量拉取；`config.yaml.example`、`README.md`、`docs/README.md`、`docs/indicator_formula.md`、`docs/backtest_guide.md` 已按单股趋势系统更新；依赖已精简到 `requirements-base.txt`。

本轮新增推进：

1. `src/settings.py`：`load_config()` 改为默认配置与用户配置递归合并，缺省配置文件时也返回完整默认骨架，避免部分配置文件导致脚本缺字段。
2. `scripts/fetch_stock.py`：`--check-quality` 改为按本次拉取标的输出最近 30 条日线摘要，包括日期区间、最新收盘价、最大自然日间隔、OHLCV 缺失数和 OHLC 异常数。
3. `src/event_log.py`、`src/notify/__init__.py`、`src/market/tradability.py`：清理残留的月度选股/IC 监控描述，改为单股趋势系统语境。
4. `tests/test_settings.py`：新增配置合并回归测试。

验收：

```bash
pytest
# 6 passed, 1 warning

python -m py_compile scripts/fetch_stock.py scripts/run_signal.py scripts/run_backtest_single.py src/settings.py src/event_log.py src/notify/__init__.py
# passed
```

后续建议：

1. 运行真实数据链路验证：`python scripts/fetch_stock.py --symbol 600930 --check-quality` → `python scripts/run_signal.py --symbol 600930 --history 60` → `python scripts/run_backtest_single.py --symbol 600930 --compare-modes`。
2. 如需进一步收尾，可继续精简 `src/backtest/engine.py`、`src/backtest/transaction_costs.py`、`src/data_fetcher/migrations.py` 中仍保留的组合/月度历史兼容逻辑，但当前单股主链路已可独立运行。

继续推进：

1. `src/market/tradability.py`：删除旧截面候选池过滤函数 `prefilter_stock_pool()`，该模块现在只保留单股回测需要的涨停价、一字涨停不可买、停牌近似判断。
2. `src/backtest/single_stock.py`：买入执行除一字涨停外，新增跳过停牌/无成交量日；卖出执行也会顺延到第一天可交易开盘。
3. `src/backtest/single_stock.py`：延迟买入若在实际成交前遇到卖出信号，会取消待执行买入，避免趋势已翻空后仍进场。
4. `tests/test_tradability.py`：新增 A 股涨停比例、一字涨停容差、无效价格、停牌近似测试。
5. `tests/test_single_stock_bt.py`：新增延迟买入取消、停牌日卖出顺延测试。

本轮验收：

```bash
pytest
# 12 passed, 1 warning

python -m py_compile src/market/tradability.py tests/test_tradability.py src/backtest/single_stock.py tests/test_single_stock_bt.py
# passed
```

2026-05-13 继续推进：

1. `src/data_fetcher/stock_name_cache.py`：移除旧的 `fetch_stock_names.py` 子进程刷新逻辑，改为纯本地 CSV 名称缓存读取；新增 `resolve_stock_name_cache_path()`、`load_stock_name_map()`、`resolve_stock_names()`。
2. `config.yaml.example` / `src/settings.py`：新增 `paths.stock_name_cache: data/stock_names.csv` 默认配置。
3. `scripts/run_signal.py`：输出和企业微信推送使用本地名称缓存中的股票名；缓存缺失或无匹配时自动回退 6 位代码。
4. `scripts/run_backtest_single.py`：回测报告标题使用本地名称缓存中的股票名，并传入 `SingleStockBacktestResult.stock_name`。
5. `README.md`：补充股票名称缓存 CSV 的使用说明。
6. `docs/backtest_guide.md`：补充停牌/无成交量卖出顺延，以及延迟买入遇卖出信号取消的回测假设。
7. `tests/test_stock_name_cache.py`：新增名称缓存列名兼容、名称回退、配置路径解析测试。

本轮验收：

```bash
pytest
# 15 passed, 1 warning

python -m py_compile src/data_fetcher/stock_name_cache.py scripts/run_signal.py scripts/run_backtest_single.py src/settings.py
# passed

python scripts/run_signal.py --help
python scripts/run_backtest_single.py --help
# passed
```

### 2026-05-13 状态汇总与剩余事项

当前状态：

1. 单股多空趋势主链路已打通：本地日线拉取、DK 趋势计算、信号查看、T+1 单股回测、三模式对比、交易记录导出、可选企业微信推送都已有入口。
2. 单股回测约束已补强：买入遇一字涨停会顺延；买卖都会跳过停牌/无成交量开盘；延迟买入若在成交前遇到卖出信号会取消。
3. 配置和名称显示已补强：配置会与默认值递归合并；CLI 支持本地股票名称缓存，缺失时回退 6 位代码。
4. 当前本地验收：`pytest` 为 `15 passed, 1 warning`；核心脚本 `--help` 和相关模块 `py_compile` 通过。

这个文档还剩的实际事项：

1. **真实数据链路验收（必做）**：需要在有网络和 AkShare 可用时跑通 `fetch_stock -> run_signal -> run_backtest_single`。建议命令：

   ```bash
   python scripts/fetch_stock.py --symbol 600930 --check-quality
   python scripts/run_signal.py --symbol 600930 --history 60
   python scripts/run_backtest_single.py --symbol 600930 --compare-modes
   ```

2. **旧兼容模块是否继续裁剪（可选）**：`src/backtest/engine.py`、`src/backtest/transaction_costs.py`、`src/data_fetcher/migrations.py` 仍保留一部分组合/月度历史兼容逻辑。当前单股主链路不依赖这些旧能力；若目标是彻底瘦身，可继续拆除或降级为内部兼容模块。
3. **历史结果目录是否归档（可选）**：`results/` 里仍有旧月度选股实验 CSV/JSON。代码主链路已不依赖它们；若要让仓库呈现为纯单股趋势系统，可以移到归档目录或从版本中移除。
4. **实盘输出打磨（可选）**：`run_signal.py` 目前是纯文本输出，可继续优化为更接近文档示例的中文表格、最近信号列表、操作建议文案。
5. **数据质量策略细化（可选）**：`fetch_stock --check-quality` 已输出近 30 条摘要；后续可增加“失败即退出”的阈值参数，避免低质量数据进入回测。
6. **可视化校验（可选）**：文档中的“视觉校验脚本”尚未实现；如需要和东方财富截图对比，可增加临时或正式绘图脚本。
7. **通知闭环验证（可选）**：代码已接入企业微信 Webhook，但还需要用真实 webhook 做一次 BUY/SELL 样例推送验证。

结论：Phase 2、Phase 3、Phase 4、Phase 5 的代码主功能已基本完成；文档里真正剩下的是真实数据链路验收，以及是否继续做仓库瘦身和体验打磨。

2026-05-13 本轮继续推进：

1. 真实数据链路改用本地已有数据验收：从 `~/hjx/lh/data/market.duckdb` 复制到当前项目 `data/market.duckdb`，从 `~/hjx/lh/data/cache/a_share_stock_names.csv` 复制到 `data/stock_names.csv`，未使用跨仓库链接。
2. 数据库验收：`a_share_daily` 共 10,532,711 行、5,197 个标的，日期覆盖 `2015-01-05` 至 `2026-05-08`；`600930` 最新日线为 `2026-05-08`。
3. `scripts/run_signal.py`：输出打磨为中文状态页，包含最新交易日、指标模式、当前多空趋势、连续天数、最新信号、操作建议；`--history` 改为展示最近 N 条日线内的 BUY/SELL 变色记录。
4. `scripts/run_signal.py`：增加 `--duckdb-path` 与 `--stock-name-cache` 覆盖参数，默认仍读取当前项目 `data/market.duckdb` 和 `data/stock_names.csv`。
5. `README.md`：补充复用已有本地 DuckDB 数据时的复制方式，避免依赖其他仓库路径。

本轮真实数据验收：

```bash
python scripts/run_signal.py --symbol 600930 --history 60
# 华电新能 (600930) | 最新交易日：2026-05-08 | 指标：macd_cross
# 当前多空趋势：多头 | 已连续 9 天 | 最新收盘：6.32 | 最新信号：HOLD
# 近期变色信号：2026-02-09 BUY、2026-03-30 SELL、2026-04-23 BUY

python scripts/run_backtest_single.py --symbol 600930 --compare-modes
# macd_cross total=-1.85%, buy_hold=-11.48%, sharpe=-0.02, max_dd=18.00%, trades=4, win_rate=25.00%
# ma_cross   total=-9.50%, buy_hold=-11.48%, sharpe=-0.54, max_dd=19.46%, trades=7, win_rate=28.57%
# boll_trend total=-9.79%, buy_hold=-11.48%, sharpe=-0.53, max_dd=18.00%, trades=10, win_rate=10.00%
```

本轮验收：

```bash
pytest
# 15 passed, 1 warning

python -m py_compile scripts/run_signal.py scripts/run_backtest_single.py
# passed
```

备注：直接运行 `fetch_stock.py` 时，当前沙箱网络无法访问 Sina / Eastmoney，报 `Operation not permitted`；本轮真实链路验收已使用复制到当前项目的本地 DuckDB 完成。`data/` 已在 `.gitignore` 中，复制的 2.4G 数据库不会进入版本控制。

2026-05-13 本轮继续推进（数据质量门禁）：

1. `scripts/fetch_stock.py`：`--check-quality` 的近端摘要抽成 `RecentQualitySummary`，统一统计最近 N 条日线的行数、日期区间、最新收盘、最大自然日间隔、OHLCV 缺失数和 OHLC 异常数。
2. `scripts/fetch_stock.py`：新增 `--fail-on-quality`，质量不达标时返回退出码 `2`；新增 `--quality-window`、`--quality-min-rows`、`--quality-max-gap-days`、`--quality-allow-nulls`、`--quality-allow-invalid-ohlc`，用于控制“失败即退出”的阈值。
3. `README.md`：补充数据拉取后的质量门禁示例。
4. `tests/test_fetch_stock_quality.py`：新增近端质量摘要与阈值判断回归测试。

2026-05-13 本轮继续推进（旧兼容 schema 收敛）：

1. `src/data_fetcher/db_manager.py`：历史 migrations 改为显式开关 `database.apply_legacy_migrations`，默认不再为新 DuckDB 创建旧月度研究表。
2. `config.yaml.example` / `src/settings.py`：新增 `database.apply_legacy_migrations: false` 默认值。
3. `src/event_log.py`：新增 `ensure_event_log_schema()`，事件日志首次写入/查询前自行确保 `run_events` 表存在，不再依赖旧 migrations v8。
4. `tests/test_db_manager_schema.py`：验证默认新库只创建日线与审计核心表，不创建 `schema_migrations` / `oos_tracking`。
5. `tests/test_event_log.py`：验证空 DuckDB 连接上可直接写入并查询趋势信号事件。

本轮验收：

```bash
pytest
# 19 passed, 1 warning

python -m py_compile scripts/run_signal.py scripts/run_backtest_single.py src/settings.py src/data_fetcher/db_manager.py src/event_log.py
# passed

python scripts/fetch_stock.py --help
python scripts/run_signal.py --help
# passed
```

2026-05-13 本轮继续推进（回测入口体验打磨）：

1. `scripts/run_backtest_single.py`：新增 `--duckdb-path` 与 `--stock-name-cache` 覆盖参数，与 `run_signal.py` 的本地数据/名称缓存覆盖能力保持一致。
2. `scripts/run_backtest_single.py`：单模式回测输出改为中文报告，包含总收益、买入持有、超额、年化收益、夏普、最大回撤、Calmar、交易统计和最近 5 笔交易。
3. `scripts/run_backtest_single.py`：`--compare-modes` 输出改为中文三模式对比表，并补充超额收益列。
4. `README.md`、`docs/backtest_guide.md`：补充回测入口的 DuckDB 与名称缓存路径覆盖示例。

本轮验收：

```bash
python -m py_compile scripts/run_backtest_single.py
# passed

python scripts/run_backtest_single.py --help
# passed

pytest
# 19 passed, 1 warning

python scripts/run_backtest_single.py --symbol 600930 --compare-modes
# 华电新能 (600930) 三模式回测对比
# macd_cross total=-1.85%, buy_hold=-11.48%, excess=+9.63%, sharpe=-0.02, max_dd=18.00%, trades=4, win_rate=25.00%
# ma_cross   total=-9.50%, buy_hold=-11.48%, excess=+1.99%, sharpe=-0.54, max_dd=19.46%, trades=7, win_rate=28.57%
# boll_trend total=-9.79%, buy_hold=-11.48%, excess=+1.69%, sharpe=-0.53, max_dd=18.00%, trades=10, win_rate=10.00%
```
