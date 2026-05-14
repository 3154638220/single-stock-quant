"""Tests for experiment management utilities."""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.backtest.experiment import (
    compare_metric_summaries,
    config_hash,
    create_experiment_dir,
    evaluate_experiment,
    expected_experiment_artifacts,
    get_git_commit,
    load_experiment_metrics,
    render_experiment_comparison_html,
    update_experiment_index,
    write_delta_markdown,
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
        assert (d / "ARTIFACTS.md").exists()
        assert (d / "DELTA.md").exists()

    def test_expected_artifacts_include_phase15_outputs(self):
        artifacts = expected_experiment_artifacts()
        assert "portfolio_summary.csv" in artifacts
        assert "meta_label_calibration.csv" in artifacts
        assert "feature_importance.csv" in artifacts
        assert "stability_heatmap.html" in artifacts
        assert "DELTA.md" in artifacts

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


class TestExperimentComparison:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_metrics_from_standard_experiment_files(self):
        exp = create_experiment_dir("metrics", base_dir=self.tmp)
        pd.DataFrame(
            [
                {"symbol": "000001", "status": "ok", "sharpe_ratio": 0.2, "max_drawdown": 0.30},
                {"symbol": "000002", "status": "ok", "sharpe_ratio": 0.6, "max_drawdown": 0.20},
                {"symbol": "000003", "status": "no_data", "sharpe_ratio": 9.0, "max_drawdown": 0.90},
            ]
        ).to_csv(exp / "batch_summary.csv", index=False)
        pd.DataFrame(
            [{"annualized_return": 0.08, "sharpe_ratio": 0.75, "max_drawdown": 0.25}]
        ).to_csv(exp / "portfolio_summary.csv", index=False)

        metrics = load_experiment_metrics(exp)

        assert metrics["batch_n_rows"] == 2
        assert metrics["batch_median_sharpe_ratio"] == 0.4
        assert metrics["batch_median_max_drawdown"] == 0.25
        assert metrics["portfolio_sharpe_ratio"] == 0.75

    def test_compare_metric_summaries_marks_lower_drawdown_as_improved(self):
        rows = compare_metric_summaries(
            {"portfolio_sharpe_ratio": 0.5, "portfolio_max_drawdown": 0.35},
            {"portfolio_sharpe_ratio": 0.7, "portfolio_max_drawdown": 0.25},
        )
        by_metric = {row["metric"]: row for row in rows}

        assert by_metric["portfolio_sharpe_ratio"]["direction"] == "improved"
        assert by_metric["portfolio_max_drawdown"]["direction"] == "improved"

    def test_compare_metric_summaries_reports_ci_overlap(self):
        rows = compare_metric_summaries(
            {
                "portfolio_sharpe_ratio": 0.5,
                "portfolio_sharpe_ratio_ci_low": 0.3,
                "portfolio_sharpe_ratio_ci_high": 0.6,
            },
            {
                "portfolio_sharpe_ratio": 0.8,
                "portfolio_sharpe_ratio_ci_low": 0.7,
                "portfolio_sharpe_ratio_ci_high": 1.0,
            },
        )

        assert rows[0]["metric"] == "portfolio_sharpe_ratio"
        assert rows[0]["ci_overlap"] == "no"

    def test_render_html_and_write_delta(self):
        rows = compare_metric_summaries(
            {"portfolio_sharpe_ratio": 0.5},
            {"portfolio_sharpe_ratio": 0.8},
        )
        html = render_experiment_comparison_html(
            rows,
            baseline_dir=self.tmp / "base",
            current_dir=self.tmp / "current",
        )
        delta = write_delta_markdown(
            rows,
            baseline_dir=self.tmp / "base",
            current_dir=self.tmp / "current",
            output_path=self.tmp / "DELTA.md",
        )

        assert "Experiment Comparison" in html
        assert delta.exists()
        assert "portfolio_sharpe_ratio" in delta.read_text(encoding="utf-8")
