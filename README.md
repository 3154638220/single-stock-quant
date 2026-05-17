# single-stock-quant

This repository is a local A-share single-stock quantitative trading system. It stores daily OHLCV data in DuckDB, computes configurable trend-timing signals, prints buy/sell state changes, and runs T+1 open single-stock backtests.

The project goal is single-stock trading validation and execution support. It no longer targets replication of Eastmoney's proprietary long/short trend curve. Legacy names such as `eastmoney_dkbar` are kept only for backward compatibility with existing configs, outputs, and tests; they should be treated as DKBar-style trading signals, not official indicator replicas.

## Quick Start

```bash
cp config.yaml.example config.yaml
pip install -r requirements.txt
pip install -e .

python scripts/fetch_stock.py --symbol 600930 --check-quality
python scripts/run_signal.py --symbol 600930 --history 60
python scripts/run_backtest_single.py --symbol 600930 --compare-modes
python scripts/plot_dktrend.py --symbol 600930 --history 180
```

Optional stock display names can be supplied through `paths.stock_name_cache`
(`data/stock_names.csv` by default). The CSV may use columns such as
`symbol,name` or `证券代码,证券简称`; commands fall back to the 6-digit symbol
when the cache is absent.

If you already have a compatible DuckDB market database, copy it into this
project instead of pointing commands at another repository:

```bash
mkdir -p data
cp /path/to/market.duckdb data/market.duckdb
cp /path/to/a_share_stock_names.csv data/stock_names.csv
```

## Trend Modes

| mode | rule | use case |
| --- | --- | --- |
| `macd_cross` | `EMA(12)-EMA(26)` crosses its 9-day signal | default, lower turnover |
| `ma_cross` | smoothed `MA5-MA20` | simple trend confirmation |
| `boll_trend` | close above/below MA20 | simplest baseline |
| `eastmoney_dkbar` | DKBar-style low-frequency trend state | trading research only; not an official replica |

Signals are emitted only when the color changes:

- green to red: `buy`
- red to green: `sell`
- same color: `hold`

Backtests execute signals on the next trading day's open. If a buy execution day opens at limit-up, the buy is delayed until the first open that is not limit-up.

## Main Commands

Fetch one or more stocks:

```bash
python scripts/fetch_stock.py --symbols 600930 000001 300750
python scripts/fetch_stock.py --symbol 600930 --check-quality --fail-on-quality --quality-min-rows 20
```

`--fail-on-quality` turns the recent-data summary into a gate. It exits with
code `2` when the selected window has too few rows, a calendar gap above
`quality.max_calendar_gap_days` (or `--quality-max-gap-days`), OHLCV nulls, or
invalid OHLC rows. Use `--quality-allow-nulls` or
`--quality-allow-invalid-ohlc` only when you intentionally want a soft check.

New DuckDB files create only the daily data and fetch-audit tables by default.
`database.apply_legacy_migrations` is off by default; enabling it now only
records a compact schema version marker and keeps very old daily-bar files
compatible. Old research/OOS tables are no longer created.

Show the latest signal:

```bash
python scripts/run_signal.py --symbol 600930 --mode macd_cross
python scripts/run_signal.py --watchlist 600930 000001 300750 --filter buy
```

Run a backtest:

```bash
python scripts/run_backtest_single.py --symbol 600930 --start 2020-01-01 --end 2025-12-31
python scripts/run_backtest_single.py --symbol 600930 --export-trades
python scripts/run_backtest_single.py --symbol 600930 --duckdb-path /path/to/market.duckdb --stock-name-cache /path/to/stock_names.csv
```

Plot a local visual check:

```bash
pip install ".[viz]"
python scripts/plot_dktrend.py --symbol 600930 --history 180
python scripts/plot_dktrend.py --symbol 600930 --mode ma_cross --output data/output/600930_ma_cross.png
```

## Structure

```text
src/
  data_fetcher/     AkShare fetch, DuckDB storage, data quality checks
  indicators/       DK trend formulas
  signals/          signal records and current-state helpers
  backtest/         costs, performance metrics, single-stock backtest
  market/           A-share tradability helpers
  notify/           optional WeCom webhook notification
scripts/
  fetch_stock.py
  run_signal.py
  run_backtest_single.py
  plot_dktrend.py
```
