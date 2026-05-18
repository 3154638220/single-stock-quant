<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-18 5:21pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (20,610t read) | 782,963t work | 97% savings

### May 17, 2026
S61 用户请求Codex重构最新的plan文件（docs/plan-05-15(2).md），目标单一：提升单股量化收益 (May 17, 10:01 PM)
S62 重构最新plan文件 docs/plan-05-15(2).md，目标单一：提升单股量化收益 (May 17, 10:03 PM)
S63 用户询问"当前模型公式是什么"——Codex深入解读了项目核心模型实现，给出了完整的公式、算法和架构说明 (May 17, 10:04 PM)
S64 Complete plan-05-15(2).md — execute all five R01-R05 tasks for 000783 single-stock return improvement, including P0 refactoring of the scoring pipeline from anchor-matching to pure return-quality optimization (May 17, 10:07 PM)
690 10:20p 🔵 run_000783_return_plan.py crashes at YAML write: numpy int64 not serializable by PyYAML safe_dump
691 " 🔴 YAML serialization crash fixed: _native() function converts numpy scalars to Python native types before safe_dump
692 " 🔵 Re-launched plan script still computing after 30s — full workload of ~300 backtests in progress
693 10:21p 🟣 Plan-05-15(2) fully executed: all five R01-R05 outputs produced with best strategy identified
694 " ⚖️ Exit-layer optimization confirmed as primary return lever over entry signal tuning
695 10:22p ✅ Plan-05-15(2).md updated with completion status and execution results section 7.1
696 " 🟣 Plan-05-15(2) fully completed: P0 refactoring + R01-R05 execution delivered in single session
697 10:23p ✅ Code quality: ruff linting errors resolved, unused variable removed from nested WFO path
698 " ✅ Final quality gate: all 201 tests pass, all ruff linting passes, clean git diff shows targeted changes only
S65 Complete plan-05-15(2).md — execute all five R01-R05 tasks: refactor scoring from anchor-matching to pure return-quality, re-rank grid candidates, run fixed-parameter WFO on top-10, test exit/entry experiment overlays, and produce best strategy config (May 17, 10:24 PM)
699 10:25p 🔵 Best experiment trade-level details confirmed: 7 trades across 3 OOS folds, all exits via intrapos_dd_stop or signal
S66 多股票最佳单股策略配置导出：产出 5 只 WFO passing 股票的独立最佳参数配置及 246 笔逐笔交易明细 (May 17, 10:27 PM)
### May 18, 2026
700 4:26p 🔵 量化交易策略优化任务启动：多股票最佳单股策略配置分析
701 4:27p 🔵 项目结构探索：single-stock-quant 量化交易系统的关键配置与数据文件
702 4:28p 🔵 全面代码审查完成：single-stock-quant 项目架构与多股票策略配置任务的理解
703 4:29p 🔵 WFO 结果跨目录对比：5只股票在不同配置下的参数表现
704 4:30p 🟣 新增多股票最佳策略配置导出脚本 export_selected_single_stock_best.py
705 " 🟣 5只股票最佳单股策略配置产出完成：每只独立 MACD 参数 + 完整交易明细
706 4:31p 🟣 任务完成：多股票最佳单股策略配置导出脚本最终版完善
S67 Record multi-stock best single-stock config results into plan-05-15(2).md (May 18, 4:32 PM)
707 4:35p 🟣 Multi-stock best single-stock config export system created
708 " 🔵 000783 single-stock best strategy identified with 47.61% OOS total return
S68 Complete tasks documented in docs/plan-05-18.md (May 18, 4:37 PM)
S69 Complete tasks documented in docs/plan-05-18.md (May 18, 5:02 PM)
709 5:03p 🔵 Comprehensive diagnostic of single-stock-quant strategy reveals 15 algorithmic deficiencies across 6 layers
710 " 🔵 WFO code defaults already match plan's proposed 504/126 fold structure
711 " 🔵 analyze_trade_forensics.py already computes MFE/MAE/exit_efficiency — P1.2 work is incremental not greenfield
712 " 🔵 WFO composite_score reliability multiplier penalizes low-trade strategies with asymmetric negative amplification
713 " 🔵 score_symbol_eligibility.py weights trend_quality at 30% but signal_frequency only 15% despite WFO penalizing low-frequency signals
714 5:05p ⚖️ Implementation plan created with 4 phases: P0/P2/P3/P4/P5 code changes, experiment scripts, 601318 review, and validation
715 5:06p 🔄 WFO composite_score redesigned: relaxed trade-frequency constraints, added total_return weight, reduced reliability penalty floor to 0.30
716 " 🟣 WFO parameter grid expanded from 81 to 864 combinations with macd_slow, min_run_len, dk_fade_exit_n added
717 " 🟣 WFO functions now support configurable step_days for overlapping/non-overlapping fold windows
718 " ✅ WFO default score constraints updated: min_trades 4→2, max_trades 24→30, max_dd 40%→35%
719 5:07p 🟣 4 new meta-label features added: atr_normalized_entry_risk, market_vol_20, sector_rs_20, trend_age_ratio
720 " ✅ wfo_stable.yaml updated: explicit train/oos/step 504/126/126, expanded 7-param grid, relaxed score constraints
721 " ✅ 601318 removed from WFO passing watchlist; analyze_trade_forensics enhanced with P25/P50/P75 MFE utilization distribution
722 " 🟣 score_symbol_eligibility now computes per-stock volume CV and suggested volume_ratio_min threshold
723 5:08p 🟣 quality_v1 meta-label type added: triage-based training ignores neutral signals, focuses on high-quality vs poor
724 " 🟣 Numba JIT acceleration added for _persistent_price_change_color with optional import and pure Python fallback
725 " ✅ GBM meta-label model parameters tuned for small-sample safety: n_estimators 100→50, max_depth 2→3, minimum 20 samples required
726 5:10p 🟣 New run_plan_05_18.py script automates all P1/P2/P4 experiments across multiple symbols with standardized JSON output and auto-decision classification
727 " ✅ All 43 tests in wfo_params_split, meta_label, meta_label_gbm, and dktrend pass after plan modifications
728 5:11p ⚖️ Phase 1 of implementation plan completed: all P0/P2/P3/P4/P5 code and config changes applied and tests pass
729 " ✅ Experiments completed: 80 WFO runs across P1_exit, P1_dk_fade, P2_entry, P4_position groups for 000783 and 300750
730 " 🔵 601318 compare-modes reveals macd_cross full-sample total return +45.93% but catastrophic 39.41% max drawdown — confirms exclusion decision
731 5:12p 🔵 Experiment results: 0 promote, 29 retry, 35 reject — no configuration reaches 14% annualized/1.30 Calmar thresholds
732 " 🔴 Ruff lint fixes: 2 f-string-without-placeholders in analyze_trade_forensics.py corrected
733 5:13p 🟣 WFO parallelization via ProcessPoolExecutor added: _eval_wfo_combo worker function with n_jobs parameter in both WFO functions
734 5:14p ✅ wfo_stable.yaml configured with n_jobs: 4 for parallel scoring; run_wfo.py exposes --n-jobs CLI flag
735 " ✅ Plan document updated with Section 10: complete execution results, experiment findings, and 601318 re-evaluation conclusion
736 " 🔵 Critical finding: volatility-target position sizing and drawdown throttle catastrophically increase max drawdown on 000783
737 " 🔴 stock_trend_quality fallback changed from constant 0.0 to 20-day return mean/std ratio when no DK trend state provided
738 5:15p ✅ Full test suite passes: 200+ tests green, ruff linting clean, all plan phases complete
739 " ✅ All 4 implementation phases complete: plan-05-18.md fully executed with code changes, experiments, validation, and documentation
S70 执行 docs/plan-05-18.md 中的单只A股量化策略收益提升计划 — 包括WFO框架修复、出场层重设计、参数网格扩展、元标签升级、波动率仓位实验、性能优化和601318复评剔除 (May 18, 5:17 PM)
**Investigated**: The primary session thoroughly explored 19 source files across the single-stock-quant codebase to cross-reference the plan document's claims against actual code state. Key files examined included: src/backtest/wfo.py (WFO optimization engine and composite scoring), scripts/run_wfo.py (CLI entry point), configs/wfo_stable.yaml (WFO configuration), src/models/meta_label.py (29-feature logistic regression meta-label), src/models/meta_label_gbm.py (GBM classifier already built but unused), src/indicators/dktrend.py (8 trend modes with pure-Python color state machine), scripts/analyze_trade_forensics.py (MFE/MAE computation), scripts/score_symbol_eligibility.py (stock eligibility scoring), scripts/run_backtest_single.py (backtest runner), configs/research/selected_single_stock_best.yaml (5-stock best config), configs/research/000783_best_return.yaml (top performer), configs/watchlist_wfo_passing.txt (WFO-passing stock list), tests/test_wfo_params_split.py, tests/test_meta_label.py, tests/test_meta_label_gbm.py, and requirements-base.txt. The session also ran 80 WFO experiments across 4 groups and a full 8-mode comparison for 601318 re-evaluation.

**Learned**: Several critical discoveries were made during cross-referencing the plan against actual code. First, the plan mistakenly claimed WFO uses 756/252 day windows (3 folds) — the actual wfo.py defaults are 504/126 (producing ~6-8 folds). Second, DEFAULT_PARAM_GRID already includes macd_slow and min_run_len, but wfo_stable.yaml's grid only searched macd_fast and macd_signal, so the bottleneck was the YAML config, not the code constants. Third, the analyze_trade_forensics.py already computes MFE/MAE/exit_efficiency — the plan's P1.2 task needed only P25/P50/P75 distribution stats added. Fourth, and most critically, volatility-target position sizing and drawdown throttle mechanisms are actively harmful: enabling them on 000783 increased max drawdown from 12.2% to 70.5% (5.8x worse), suggesting a fundamental implementation flaw where positions are scaled UP during high-volatility periods rather than DOWN. Fifth, under 8-fold WFO evaluation, the strategy's true OOS performance ceiling is ~9-10% annualized, significantly below the 13.86% reported from the previous 3-fold evaluation — the extra folds revealed overfitting in the original parameter selection. Sixth, the WFO composite_score reliability multiplier has an asymmetric bias: negative scores are DIVIDED by reliability (amplifying penalties for sparse losing strategies) while positive scores are multiplied (discounting sparse winners).

**Completed**: All four implementation phases completed: (1) P0/P2/P3/P4/P5 code and config changes applied across 10 files — WFO composite_score redesigned to 5-factor model (Calmar 35%, Sharpe 25%, DD 15%, total_return 15%, trade_freq 10%) with relaxed trade-frequency constraints (2-30/year) and tightened drawdown limit (35%), FOCUSED_PARAM_GRID expanded from 4 to 7 parameters (864 combinations including macd_slow, min_run_len, dk_fade_exit_n), step_days parameter added to both WFO functions for configurable fold stride, 4 new meta-label features (trend_age_ratio, atr_normalized_entry_risk, market_vol_20, sector_rs_20), quality_v1 triage label type (trains only high-quality vs poor signals, skips neutral), GBM model tuned for small samples (n_estimators 100→50, max_depth 2→3, minimum 20 samples), Numba JIT acceleration for _persistent_price_change_color with pure-Python fallback, ProcessPoolExecutor parallelization with n_jobs parameter and --n-jobs CLI flag, per-stock volume CV profiling in score_symbol_eligibility.py, MFE utilization P25/P50/P75 in analyze_trade_forensics.py. (2) New experiment automation script run_plan_05_18.py created and executed — produced 80 WFO runs across P1_exit (16 exit mechanism variants), P1_dk_fade (4 dk_fade values), P2_entry (4 Donchian breakout variants), and P4_position (8 volatility×throttle combinations) for both 000783 and 300750, all at 504/126/126 with 8 OOS folds. Results: 0 promote, 29 retry, 35 reject. Best result was P2_entry_000783_breakout_15 at Calmar 1.20 but only 3 trades. (3) 601318 re-evaluated with --compare-modes across all 8 trend modes — macd_cross showed +45.93% but 39.41% drawdown, no mode passed both return and risk criteria; definitively excluded from watchlist_wfo_passing.txt (now 4 stocks). (4) Plan document updated with comprehensive Section 10 recording all results, technical debt table updated, ruff lint issues fixed in all modified files (4 issues resolved), full pytest suite passing (200+ tests), ruff clean on all plan-modified files (36 pre-existing issues remain in untouched notebooks/scripts/tests).

**Next Steps**: The implementation is complete and all experiments have been run. The plan document's Section 10.5 identifies the most valuable next research directions: (1) for 000783, investigate the breakout_15 configuration to increase trade count beyond 3 while preserving the Calmar 1.20 quality; (2) for 300750, prioritize drawdown analysis over return improvement, focusing on 2021-2022 holding paths and profit_lock timing; (3) for the multi-stock pool, review the 300059 WFO OOS vs full-sample discrepancy (66% vs 2.4%). The critical finding about vol_target/drawdown_throttle being actively harmful requires root-cause debugging of the position scaling logic before these mechanisms can be productively used.


Access 783k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>