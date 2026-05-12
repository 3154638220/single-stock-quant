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
