import numpy as np
import pandas as pd

from src.backtest.single_stock import run_single_stock_backtest
from src.backtest.transaction_costs import (
    TransactionCostParams,
    cost_params_dict_for_logging,
    net_simple_return_from_long_hold,
    transaction_cost_params_from_mapping,
    turnover_cost_drag,
)
from src.indicators import DKTrendParams, TrendMode


def _flat_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )


class TestTransactionCostParams:
    def test_buy_fraction_commission_plus_slippage(self):
        tc = TransactionCostParams(commission_buy_bps=3.0, slippage_bps_per_side=2.0)
        assert abs(tc.buy_fraction() - 5.0 * 1e-4) < 1e-10

    def test_sell_fraction_includes_stamp_duty(self):
        tc = TransactionCostParams(
            commission_sell_bps=3.0, slippage_bps_per_side=2.0, stamp_duty_sell_bps=5.0
        )
        assert abs(tc.sell_fraction() - 10.0 * 1e-4) < 1e-10

    def test_defaults_match_real_ashare(self):
        tc = TransactionCostParams()
        assert abs(tc.buy_fraction() - 4.5 * 1e-4) < 1e-10
        assert abs(tc.sell_fraction() - 9.5 * 1e-4) < 1e-10

    def test_from_mapping_round_trips(self):
        raw = {"commission_buy_bps": 3, "stamp_duty_sell_bps": 6}
        tc = transaction_cost_params_from_mapping(raw)
        assert tc.commission_buy_bps == 3.0
        assert tc.stamp_duty_sell_bps == 6.0
        assert tc.slippage_bps_per_side == 2.0  # unspecified keeps default

    def test_cost_params_dict_for_logging_keys(self):
        tc = TransactionCostParams()
        d = cost_params_dict_for_logging(tc)
        assert "buy_fraction" in d
        assert "sell_fraction" in d
        assert "stamp_duty_sell_bps" in d

    def test_net_simple_return_deducts_both_sides(self):
        tc = TransactionCostParams()
        net = net_simple_return_from_long_hold(0.10, tc)
        assert net < 0.10

    def test_turnover_cost_drag_bounds(self):
        tc = TransactionCostParams()
        drag = turnover_cost_drag(0.5, tc)
        assert 0.0 < drag < 1.0


class TestAsymmetricCostBacktest:
    def test_asymmetric_cost_gives_higher_return_than_costbps_15(self):
        """Real A-share costs (4.5+9.5=14bps) are cheaper than symmetric 15bps per side (30bps total)."""
        closes = [10, 10, 10, 10, 11, 12, 13, 14]
        df = _flat_df(closes)
        res_sym = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            cost_bps=15.0, initial_capital=10000,
        )
        res_asym = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            cost_params=TransactionCostParams(), initial_capital=10000,
        )
        # asymmetric 14bps total < symmetric 30bps total → higher net return
        assert res_asym.total_return > res_sym.total_return

    def test_cost_model_recorded_in_result(self):
        closes = [10, 10, 10, 10, 11, 12, 13, 14]
        df = _flat_df(closes)
        tc = TransactionCostParams(commission_buy_bps=3.0, stamp_duty_sell_bps=7.0)
        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            cost_params=tc, initial_capital=10000,
        )
        assert res.cost_model is not None
        assert "buy_fraction" in res.cost_model
        assert "sell_fraction" in res.cost_model

    def test_zero_cost_upper_bound(self):
        closes = [10, 10, 10, 10, 11, 12, 13, 14]
        df = _flat_df(closes)
        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            cost_bps=0.0, initial_capital=10000,
        )
        assert res.n_trades >= 1
        assert res.total_return > 0
