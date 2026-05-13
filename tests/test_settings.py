from pathlib import Path

from src.backtest.wfo import normalize_param_grid
from src.settings import load_config


def test_load_config_merges_defaults_for_partial_file(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
trend_signal:
  mode: ma_cross
backtest:
  cost_bps: 8
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["trend_signal"]["mode"] == "ma_cross"
    assert cfg["trend_signal"]["macd_fast"] == 12
    assert cfg["backtest"]["cost_bps"] == 8
    assert cfg["backtest"]["execution"] == "tplus1_open"
    assert cfg["paths"]["duckdb_path"] == "data/market.duckdb"
    assert cfg["wfo"]["param_grid"]["macd_fast"] == [8, 10, 12, 14]


def test_normalize_param_grid_accepts_configured_values():
    grid = normalize_param_grid({"macd_fast": [6, 12], "macd_signal": 9})

    assert grid == {"macd_fast": [6, 12], "macd_signal": [9]}
