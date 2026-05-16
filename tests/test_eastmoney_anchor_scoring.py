import pandas as pd

from scripts.calibrate_eastmoney_dkbar import load_anchors, score_candidate
from src.indicators import DKTrendParams, TrendMode


def test_eastmoney_anchor_csv_is_loadable():
    anchors = load_anchors("data/anchors/eastmoney_dkbar_000783.csv", symbol="000783")

    assert len(anchors) == 7
    assert set(anchors.columns) >= {"symbol", "date", "lst", "bar_high", "bar_low", "bar_color"}
    assert set(anchors["bar_color"]) == {"red", "green"}
    assert anchors["date"].is_monotonic_increasing


def test_score_candidate_reports_anchor_errors_and_color_accuracy():
    closes = [10.0, 10.4, 10.8, 10.6, 11.0, 10.9, 11.3, 11.1]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
        }
    )
    anchors = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": pd.to_datetime(["2024-01-05", "2024-01-06"]),
            "lst": [10.7, 10.8],
            "bar_high": [10.95, 10.95],
            "bar_low": [10.85, 10.85],
            "bar_color": ["red", "green"],
        }
    )
    params = DKTrendParams(
        mode=TrendMode.EASTMONEY_DKBAR,
        lst_period=3,
        lst_method="sma",
        bar_period=2,
        bar_method="sma",
        bar_range_period=2,
        bar_range_mult=0.0,
        state_confirm_days=1,
        slope_lookback=1,
        slope_tolerance=99.0,
        bar_color_method="price_change",
    )

    metrics, detail = score_candidate(df, anchors, params, label="unit")

    assert metrics["label"] == "unit"
    assert metrics["matched_anchors"] == 2
    assert metrics["missing_anchors"] == 0
    assert 0.0 <= metrics["bar_color_accuracy"] <= 1.0
    assert metrics["trend_switches_per_year"] >= 0.0
    assert {"lst_error", "bar_high_error", "bar_low_error", "reason"} <= set(detail.columns)
