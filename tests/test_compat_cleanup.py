import duckdb
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.transaction_costs import TransactionCostParams, cost_params_dict_for_logging
import src.data_fetcher as data_fetcher
from src.data_fetcher.migrations import apply_migrations


def test_optional_migrations_do_not_create_old_research_tables():
    conn = duckdb.connect(":memory:")
    try:
        applied = apply_migrations(conn)
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }

        assert applied == [1, 2, 3]
        assert {"a_share_daily", "data_fetch_audit", "schema_migrations"} <= tables
        assert "oos_tracking" not in tables
        assert "ic_monitor" not in tables
        assert "a_share_fundamental_raw" not in tables
    finally:
        conn.close()


def test_transaction_cost_logging_is_simple_cost_only():
    payload = cost_params_dict_for_logging(TransactionCostParams())

    assert "impact_model" not in payload
    assert "impact_k" not in payload
    assert payload["buy_fraction"] > 0
    assert payload["sell_fraction"] > payload["buy_fraction"]


def test_compat_engine_runs_simple_daily_weight_backtest():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    returns = pd.DataFrame({"600000": [0.0, 0.02, -0.01]}, index=idx)
    weights = pd.DataFrame({"600000": [1.0]}, index=[idx[0]])

    res = run_backtest(returns, weights, config=BacktestConfig())

    np.testing.assert_allclose(res.daily_returns.to_numpy(), [0.0, 0.02, -0.01])
    assert res.meta["compatibility_engine"] is True


def test_data_fetcher_public_api_excludes_old_research_clients():
    old_names = {
        "FundamentalClient",
        "FundFlowClient",
        "ShareholderClient",
        "register_fundamental_source",
        "register_fund_flow_source",
        "fetch_industry_mapping",
    }

    assert old_names.isdisjoint(set(data_fetcher.__all__))
    for name in old_names:
        assert not hasattr(data_fetcher, name)
