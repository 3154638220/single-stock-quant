from src.backtest.wfo import (
    DEFAULT_PARAM_GRID,
    _BT_PARAM_KEYS,
    _params_with,
    normalize_param_grid,
)
from src.indicators import DKTrendParams, TrendMode


class TestParamSplit:
    def test_bt_param_keys_known_set(self):
        assert "stop_loss_pct" in _BT_PARAM_KEYS
        assert "atr_stop_multiplier" in _BT_PARAM_KEYS
        assert "trailing_stop_pct" in _BT_PARAM_KEYS
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
