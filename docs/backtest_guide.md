# Single-Stock Backtest Guide

The single-stock backtest is a long/flat state machine:

1. Compute DK trend signals from daily OHLCV.
2. Execute buy signals on the next trading day's open.
3. If the buy execution day opens at limit-up, delay the buy until the first tradable open.
4. Execute sell signals on the next trading day's open, delaying through suspended/no-volume days.
5. Close any open position at the final close.
6. Apply `backtest.cost_bps` on both buy and sell sides.

If a delayed buy has not executed before a sell signal appears, the pending buy
is cancelled. This avoids opening a new position after the DK trend has already
turned short.

Example:

```bash
python scripts/run_backtest_single.py --symbol 600930 --compare-modes
python scripts/run_backtest_single.py --symbol 600930 --mode macd_cross --export-trades
python scripts/run_backtest_single.py --symbol 600930 --duckdb-path /path/to/market.duckdb
```

Use `--stock-name-cache /path/to/stock_names.csv` when the display-name CSV is
outside the configured project data directory.
