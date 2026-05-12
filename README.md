# single-stock-quant

This repository is a local A-share single-stock trend timing system. It stores daily OHLCV data in DuckDB, computes an Eastmoney-style long/short trend approximation, prints buy/sell state changes, and runs T+1 open single-stock backtests.

## Quick Start

```bash
cp config.yaml.example config.yaml
pip install -r requirements.txt
pip install -e .

python scripts/fetch_stock.py --symbol 600930 --check-quality
python scripts/run_signal.py --symbol 600930 --history 60
python scripts/run_backtest_single.py --symbol 600930 --compare-modes
```

## Trend Modes

| mode | rule | use case |
| --- | --- | --- |
| `macd_cross` | `EMA(12)-EMA(26)` crosses its 9-day signal | default, lower turnover |
| `ma_cross` | smoothed `MA5-MA20` | simple trend confirmation |
| `boll_trend` | close above/below MA20 | simplest baseline |

Signals are emitted only when the color changes:

- green to red: `buy`
- red to green: `sell`
- same color: `hold`

Backtests execute signals on the next trading day's open. If a buy execution day opens at limit-up, the buy is delayed until the first open that is not limit-up.

## Main Commands

Fetch one or more stocks:

```bash
python scripts/fetch_stock.py --symbols 600930 000001 300750
```

Show the latest signal:

```bash
python scripts/run_signal.py --symbol 600930 --mode macd_cross
python scripts/run_signal.py --watchlist 600930 000001 300750 --filter buy
```

Run a backtest:

```bash
python scripts/run_backtest_single.py --symbol 600930 --start 2020-01-01 --end 2025-12-31
python scripts/run_backtest_single.py --symbol 600930 --export-trades
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
```
