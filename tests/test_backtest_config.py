from src.backtest.config import build_bt_kwargs
from src.backtest.transaction_costs import TransactionCostParams


def test_build_bt_kwargs_minimal_config():
    """With a minimal config, all default values are populated."""
    cfg = {}
    kw = build_bt_kwargs(cfg)
    assert kw["cost_bps"] == 15.0
    assert kw["cost_params"] is None
    assert kw["initial_capital"] == 100000
    assert kw["volume_confirm"] is False
    assert kw["volume_lookback"] == 20
    assert kw["volume_ratio_min"] == 1.0
    assert kw["consensus_n_agree"] is None
    assert kw["enable_index_filter"] is False
    assert kw["index_ohlcv"] is None
    assert kw["stop_loss_pct"] == 0.0
    assert kw["trailing_stop_pct"] == 0.0
    assert kw["atr_stop_multiplier"] == 0.0
    assert kw["atr_stop_period"] == 14
    assert kw["atr_trailing_mult"] == 0.0
    assert kw["atr_trailing_min_gain"] == 0.0
    assert kw["risk_per_trade_pct"] == 0.0
    assert kw["position_size_cap"] == 1.0
    assert kw["stop_reentry_enabled"] is False
    assert kw["stop_reentry_cooldown"] == 3
    assert kw["stop_reentry_min_run"] == 2
    assert kw["require_weekly_bullish"] is False
    assert kw["require_index_trend_bullish"] is False
    assert kw["weekly_ma_fast"] == 5
    assert kw["weekly_ma_slow"] == 13
    assert kw["volatility_high_vol_multiple"] == 1.5
    assert kw["volatility_high_vol_scale"] == 0.5


def test_build_bt_kwargs_with_transaction_cost():
    cfg = {
        "backtest": {
            "transaction_cost": {
                "commission_buy_bps": 2.5,
                "commission_sell_bps": 2.5,
                "slippage_bps_per_side": 2.0,
                "stamp_duty_sell_bps": 5.0,
            }
        }
    }
    kw = build_bt_kwargs(cfg)
    assert isinstance(kw["cost_params"], TransactionCostParams)
    assert kw["cost_params"].commission_buy_bps == 2.5


def test_build_bt_kwargs_includes_atr_trailing_exit_params():
    kw = build_bt_kwargs({"backtest": {"atr_trailing_mult": 2.5, "atr_trailing_min_gain": 0.08}})

    assert kw["atr_trailing_mult"] == 2.5
    assert kw["atr_trailing_min_gain"] == 0.08


def test_build_bt_kwargs_consensus_mode():
    cfg = {"trend_signal": {"mode": "consensus", "consensus_n_agree": 2}}
    kw = build_bt_kwargs(cfg)
    assert kw["consensus_n_agree"] == 2


def test_build_bt_kwargs_index_ohlcv_passthrough():
    """index_ohlcv is passed through as-is."""
    sentinel = object()
    kw = build_bt_kwargs({}, index_ohlcv=sentinel)
    assert kw["index_ohlcv"] is sentinel


def test_build_bt_kwargs_all_risk_params():
    cfg = {
        "risk": {
            "enable_index_filter": True,
            "benchmark_symbol": "000300",
            "extreme_lookback_days": 5,
            "extreme_drop_threshold": 0.03,
            "risk_off_factor": 0.5,
        }
    }
    kw = build_bt_kwargs(cfg)
    assert kw["enable_index_filter"] is True
    assert kw["benchmark_symbol"] == "000300"
    assert kw["extreme_lookback_days"] == 5
    assert kw["extreme_drop_threshold"] == 0.03
    assert kw["risk_off_factor"] == 0.5
