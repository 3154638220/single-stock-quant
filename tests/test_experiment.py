"""Tests for experiment management utilities."""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.backtest.experiment import (
    config_hash,
    create_experiment_dir,
    evaluate_experiment,
    get_git_commit,
    update_experiment_index,
)


class TestExperimentUtils:
    def test_get_git_commit_returns_string(self):
        commit = get_git_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0

    def test_config_hash_deterministic(self):
        cfg = {"a": 1, "b": {"c": 2}}
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2
        assert len(h1) == 12

    def test_config_hash_different_for_different_configs(self):
        h1 = config_hash({"a": 1})
        h2 = config_hash({"a": 2})
        assert h1 != h2


class TestExperimentDir:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_experiment_dir(self):
        d = create_experiment_dir("test_exp", base_dir=self.tmp, notes="test notes")
        assert d.exists()
        assert (d / "notes.md").exists()

    def test_update_index_creates_csv(self):
        d = create_experiment_dir("idx_test", base_dir=self.tmp)
        idx = update_experiment_index("idx_test", exp_dir=d, index_path=Path(self.tmp) / "index.csv")
        assert idx.exists()
        df = pd.read_csv(idx)
        assert len(df) >= 1

    def test_update_index_replace_existing(self):
        d = create_experiment_dir("idx_test2", base_dir=self.tmp)
        update_experiment_index("idx_test2", exp_dir=d, index_path=Path(self.tmp) / "index.csv")
        update_experiment_index("idx_test2", exp_dir=d, index_path=Path(self.tmp) / "index.csv", notes="updated")
        df = pd.read_csv(Path(self.tmp) / "index.csv")
        matched = df[df["experiment_id"] == "idx_test2"]
        assert len(matched) == 1


class TestEvaluateExperiment:
    def test_pass_with_no_baseline(self):
        result = evaluate_experiment({"median_sharpe": 0.5})
        assert result["verdict"] == "pass"

    def test_flag_high_risk_when_mdd_worsens(self):
        result = evaluate_experiment(
            {"median_sharpe": 0.5, "median_max_drawdown": 0.45, "median_n_trades": 50},
            baseline={"median_sharpe": 0.3, "median_max_drawdown": 0.35, "median_n_trades": 50},
        )
        assert result["verdict"] == "high_risk_variant_only"

    def test_flag_insufficient_trades(self):
        result = evaluate_experiment(
            {"median_sharpe": 0.5, "median_max_drawdown": 0.30, "median_n_trades": 5},
            baseline={"median_sharpe": 0.3, "median_max_drawdown": 0.35, "median_n_trades": 50},
        )
        assert result["verdict"] == "insufficient_samples"
