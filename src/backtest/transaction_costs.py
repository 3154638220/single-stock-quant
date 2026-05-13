"""Simple A-share transaction cost helpers for long-only timing backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class TransactionCostParams:
    """Transaction fees in basis points. One bps is 0.01%."""

    commission_buy_bps: float = 2.5
    commission_sell_bps: float = 2.5
    slippage_bps_per_side: float = 2.0
    stamp_duty_sell_bps: float = 5.0

    def buy_fraction(self) -> float:
        """Total buy-side friction as a return fraction."""
        return (self.commission_buy_bps + self.slippage_bps_per_side) * 1e-4

    def sell_fraction(self) -> float:
        """Total sell-side friction as a return fraction."""
        return (
            self.commission_sell_bps
            + self.slippage_bps_per_side
            + self.stamp_duty_sell_bps
        ) * 1e-4


def transaction_cost_params_from_mapping(m: Mapping[str, Any]) -> TransactionCostParams:
    d = dict(m) if isinstance(m, dict) else {}
    return TransactionCostParams(
        commission_buy_bps=float(d.get("commission_buy_bps", 2.5)),
        commission_sell_bps=float(d.get("commission_sell_bps", 2.5)),
        slippage_bps_per_side=float(d.get("slippage_bps_per_side", 2.0)),
        stamp_duty_sell_bps=float(d.get("stamp_duty_sell_bps", 5.0)),
    )


def net_simple_return_from_long_hold(
    gross_simple_return: float,
    costs: TransactionCostParams,
) -> float:
    """Net return for one buy-hold-sell cycle."""
    return (1.0 - costs.buy_fraction()) * (1.0 + float(gross_simple_return)) * (
        1.0 - costs.sell_fraction()
    ) - 1.0


def turnover_cost_drag(
    turnover_half_l1: float,
    costs: TransactionCostParams,
) -> float:
    """Approximate drag from a one-way turnover amount."""
    t = max(0.0, min(1.0, float(turnover_half_l1)))
    return t * (costs.buy_fraction() + costs.sell_fraction())


def cost_params_dict_for_logging(costs: TransactionCostParams) -> Dict[str, Any]:
    return {
        "buy_fraction": costs.buy_fraction(),
        "sell_fraction": costs.sell_fraction(),
        "commission_buy_bps": costs.commission_buy_bps,
        "commission_sell_bps": costs.commission_sell_bps,
        "slippage_bps_per_side": costs.slippage_bps_per_side,
        "stamp_duty_sell_bps": costs.stamp_duty_sell_bps,
    }
