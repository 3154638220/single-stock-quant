# DK Trend Indicator

This project uses transparent trend rules for single-stock trading research and backtesting.
It does not try to replicate Eastmoney's proprietary long/short trend curve.

## `macd_cross`

```text
DIFF = EMA(close, 12) - EMA(close, 26)
DEA = EMA(DIFF, 9)
dk_value = DIFF - DEA
```

Red means `dk_value > 0`; green means `dk_value <= 0`.

## `ma_cross`

```text
dk_value = EMA(MA(close, 5) - MA(close, 20), 3)
```

## `boll_trend`

```text
dk_value = close - MA(close, 20)
```

## `eastmoney_dkbar`

`eastmoney_dkbar` is a legacy mode name retained for compatibility. In the
current project it means a DKBar-style trading signal family with separate
visual bar fields and trading `trend_state`; it is not a target for official
indicator replication.

For all modes:

```text
green -> red = buy
red -> green = sell
same color   = hold
```

## Visual Check

Use the local plotting script to compare the close-price path with red/green DK bars:

```bash
pip install ".[viz]"
python scripts/plot_dktrend.py --symbol 600930 --mode macd_cross --history 180
```

The PNG is written to `data/output/{symbol}_{mode}_dktrend.png` unless `--output` is supplied.
