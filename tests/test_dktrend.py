import pandas as pd

from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.indicators.utils import ema, highest, lowest


def test_utils_rolling_helpers():
    s = pd.Series([1, 2, 3, 2, 5], dtype=float)
    assert highest(s, 3).tolist()[-1] == 5
    assert lowest(s, 3).tolist()[-1] == 2
    assert pd.isna(ema(s, 3).iloc[1])
    assert ema(s, 3).notna().iloc[2]


def test_boll_trend_generates_color_and_signal():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12, 11, 10],
            "high": [10, 10, 10, 10, 11, 12, 11, 10],
            "low": [10, 10, 10, 10, 11, 12, 11, 10],
            "close": [10, 10, 10, 10, 11, 12, 9, 8],
        }
    )
    out = compute_dktrend(df, DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3))
    assert {"dk_value", "dk_color", "dk_signal", "dk_run_len"} <= set(out.columns)
    assert "buy" in set(out["dk_signal"])
    assert "sell" in set(out["dk_signal"])
    assert out.loc[out["dk_signal"] == "buy", "dk_run_len"].iloc[0] == 1


def test_eastmoney_dkbar_outputs_structured_bar_state():
    closes = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
        }
    )
    out = compute_dktrend(
        df,
        DKTrendParams(
            mode=TrendMode.EASTMONEY_DKBAR,
            lst_period=4,
            bar_period=2,
            bar_range_period=2,
            bar_range_mult=0.10,
            slope_lookback=1,
            slope_tolerance=99.0,
            state_confirm_days=1,
            hysteresis_pct=0.0,
        ),
    )

    expected = {
        "lst",
        "bar_high",
        "bar_low",
        "bar_mid",
        "bar_color",
        "bar_run_len",
        "trend_state",
        "trend_run_len",
    }
    assert expected <= set(out.columns)
    bars = out[["bar_high", "bar_low"]].dropna()
    assert (bars["bar_high"] >= bars["bar_low"]).all()
    assert {"red", "green"} <= set(out["trend_state"])
    assert (out["dk_color"] == out["trend_state"]).all()
    assert {"buy", "sell"} <= {x for x in out["dk_signal"] if x}


def test_eastmoney_dkbar_visual_color_is_separate_from_trend_state():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 10, 10, 10, 10],
            "high": [10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2],
            "low": [9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8],
            "close": [10.0, 10.2, 10.4, 10.1, 10.5, 10.3, 10.7, 10.6],
        }
    )
    out = compute_dktrend(
        df,
        DKTrendParams(
            mode=TrendMode.EASTMONEY_DKBAR,
            lst_period=3,
            bar_period=2,
            bar_range_period=2,
            state_confirm_days=1,
            bar_color_method="price_change",
            slope_lookback=1,
            slope_tolerance=99.0,
        ),
    )

    valid = out[out["bar_color"].isin(["red", "green"])].copy()
    assert {"red", "green"} <= set(valid["bar_color"])
    assert "trend_state" in out.columns
    assert "dk_signal" in out.columns


def test_eastmoney_dkbar_persistent_price_change_holds_red_pullbacks():
    closes = [10.0, 10.5, 11.0, 11.5, 12.0, 11.8, 11.7, 11.9]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
        }
    )

    base = DKTrendParams(
        mode=TrendMode.EASTMONEY_DKBAR,
        lst_period=5,
        lst_method="sma",
        bar_period=1,
        bar_method="sma",
        bar_range_period=2,
        bar_range_mult=0.0,
        state_confirm_days=1,
        slope_lookback=1,
        slope_tolerance=99.0,
        hysteresis_pct=0.0,
    )
    price_change = compute_dktrend(df, base)
    persistent = compute_dktrend(
        df,
        DKTrendParams(
            **{
                **base.__dict__,
                "bar_color_method": "persistent_price_change",
                "bar_color_hold_days": 2,
                "bar_color_min_red_run": 2,
            }
        ),
    )

    assert price_change.iloc[5]["bar_color"] == "green"
    assert persistent.iloc[5]["bar_color"] == "red"
    assert persistent.iloc[6]["bar_color"] == "red"
