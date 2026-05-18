import math

import pandas as pd

from src.backtest.wfo import (
    _BT_PARAM_KEYS,
    DEFAULT_PARAM_GRID,
    _composite_score,
    _oos_trend_with_warmup,
    _params_with,
    _select_stable_params,
    _stability,
    json_safe,
    normalize_param_grid,
    run_walk_forward_optimization,
    trade_contribution_metrics,
)
from src.indicators import DKTrendParams, TrendMode


class TestParamSplit:
    def test_bt_param_keys_known_set(self):
        assert "stop_loss_pct" in _BT_PARAM_KEYS
        assert "atr_stop_multiplier" in _BT_PARAM_KEYS
        assert "trailing_stop_pct" in _BT_PARAM_KEYS
        assert "require_index_trend_bullish" in _BT_PARAM_KEYS
        # min_run_len belongs to DKTrendParams, NOT to bt_kwargs
        assert "min_run_len" not in _BT_PARAM_KEYS
        # macd params belong to DKTrendParams
        assert "macd_fast" not in _BT_PARAM_KEYS

    def test_params_with_returns_tuple(self):
        base = DKTrendParams(mode=TrendMode.MACD_CROSS)
        params, bt_kwargs = _params_with(base, {}, "macd_cross")
        assert isinstance(params, DKTrendParams)
        assert isinstance(bt_kwargs, dict)

    def test_stop_loss_pct_routed_to_bt_kwargs(self):
        base = DKTrendParams(mode=TrendMode.MACD_CROSS)
        _, bt = _params_with(base, {"stop_loss_pct": 0.08, "macd_fast": 10}, "macd_cross")
        assert bt["stop_loss_pct"] == 0.08
        assert "macd_fast" not in bt

    def test_min_run_len_stays_in_trend_params(self):
        base = DKTrendParams(mode=TrendMode.MACD_CROSS, min_run_len=1)
        params, bt = _params_with(base, {"min_run_len": 3}, "macd_cross")
        assert params.min_run_len == 3
        assert "min_run_len" not in bt


class TestDefaultParamGrid:
    def test_default_grid_has_expanded_keys(self):
        assert "min_run_len" in DEFAULT_PARAM_GRID
        assert "stop_loss_pct" in DEFAULT_PARAM_GRID
        assert DEFAULT_PARAM_GRID["min_run_len"] == [1, 2, 3]

    def test_normalize_empty_returns_default(self):
        grid = normalize_param_grid(None)
        assert "macd_fast" in grid
        assert "min_run_len" in grid

    def test_normalize_non_list_value_wraps(self):
        grid = normalize_param_grid({"macd_fast": 12, "macd_slow": [26]})
        assert grid["macd_fast"] == [12]
        assert grid["macd_slow"] == [26]

    def test_stability_accepts_categorical_params(self):
        result = _stability(
            [
                {"params": {"lst_method": "sma", "lst_period": 205}, "is_score": 1.0, "oos_sharpe": 0.2},
                {"params": {"lst_method": "sma", "lst_period": 250}, "is_score": 0.8, "oos_sharpe": 0.1},
            ]
        )
        assert result["lst_method"]["mode"] == "sma"
        assert math.isnan(result["lst_method"]["variance"])
        assert result["lst_period"]["variance"] > 0

    def test_oos_trend_uses_train_rows_as_warmup(self):
        closes = [10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 12.0, 13.0]
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
                "open": closes,
                "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes],
                "close": closes,
                "volume": [100.0] * len(closes),
            }
        )
        params = DKTrendParams(
            mode=TrendMode.EASTMONEY_DKBAR,
            lst_period=5,
            lst_method="sma",
            bar_period=1,
            bar_method="wma",
            bar_range_mult=0.0,
            slope_lookback=1,
            state_confirm_days=1,
            hysteresis_pct=0.0,
        )

        trend = _oos_trend_with_warmup(df.iloc[:5], df.iloc[5:], params, {})

        assert len(trend) == 3
        assert trend["lst"].notna().any()

    def test_json_safe_replaces_non_finite_values(self):
        result = json_safe({"a": float("nan"), "b": [float("inf"), 1.0]})

        assert result == {"a": None, "b": [None, 1.0]}

    def test_trade_contribution_metrics(self):
        trades = pd.DataFrame({"return": [-0.02, 0.10, 0.04]})
        result = trade_contribution_metrics(trades, total_return=0.20)

        assert result["largest_trade_return"] == 0.10
        assert result["largest_trade_contribution"] == 0.5

    def test_quality_first_reliability_mode_does_not_discount_low_trade_count(self):
        class Result:
            n_trades = 2
            max_drawdown = 0.10
            sharpe_ratio = 1.0
            calmar_ratio = 1.5
            total_return = 0.20

        standard = _composite_score(Result(), train_days=504, min_trades_per_year=1.0)
        quality_first = _composite_score(
            Result(),
            train_days=504,
            min_trades_per_year=1.0,
            reliability_mode="quality_first",
        )

        assert quality_first > standard


class TestStableParamSelection:
    def test_requires_minimum_fold_count(self):
        result = _select_stable_params(
            [{"params": {"macd_fast": 8}, "is_score": 1.0}],
            min_folds=5,
        )
        assert result["used"] is False
        assert result["params"] == {}

    def test_prefers_stable_region_over_single_high_peak(self):
        fold_results = [
            {"params": {"macd_fast": 8, "min_run_len": 1}, "is_score": 1.00},
            {"params": {"macd_fast": 8, "min_run_len": 1}, "is_score": 1.05},
            {"params": {"macd_fast": 8, "min_run_len": 2}, "is_score": 0.95},
            {"params": {"macd_fast": 8, "min_run_len": 2}, "is_score": 1.02},
            {"params": {"macd_fast": 14, "min_run_len": 3}, "is_score": 1.80},
            {"params": {"macd_fast": 14, "min_run_len": 3}, "is_score": -0.20},
        ]
        result = _select_stable_params(fold_results, min_folds=5, top_n=2)
        assert result["used"] is True
        assert result["params"]["macd_fast"] == 8


class TestWFOMetaLabel:
    def test_wfo_runs_with_meta_label_enabled(self):
        closes = ([10.0, 10.0, 10.5, 11.0, 12.0, 11.0, 10.0, 9.0, 8.5, 9.5] * 14)
        n = len(closes)
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=n),
                "open": closes,
                "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes],
                "close": closes,
                "volume": [100.0] * n,
            }
        )
        result = run_walk_forward_optimization(
            "600000",
            df,
            base_params=DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            param_grid={"boll_window": [3], "min_run_len": [1]},
            train_days=80,
            oos_days=20,
            mode=TrendMode.BOLL_TREND,
            enable_meta_label=True,
            meta_label_min_samples=1,
        )

        assert result["enable_meta_label"] is True
        assert result["n_folds"] >= 1
