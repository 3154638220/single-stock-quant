<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-18 6:44pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (20,739t read) | 994,181t work | 98% savings

### May 18, 2026
S67 Record multi-stock best single-stock config results into plan-05-15(2).md (May 18, 4:32 PM)
S68 Complete tasks documented in docs/plan-05-18.md (May 18, 4:37 PM)
S69 Complete tasks documented in docs/plan-05-18.md (May 18, 5:02 PM)
S70 执行 docs/plan-05-18.md 中的单只A股量化策略收益提升计划 — 包括WFO框架修复、出场层重设计、参数网格扩展、元标签升级、波动率仓位实验、性能优化和601318复评剔除 (May 18, 5:03 PM)
S71 Check completion status of plan-05-18.md — user wants to know if all planned tasks are done (May 18, 5:17 PM)
S72 Check completion status of plan-05-18.md — user asked whether all originally planned content has been completed (May 18, 5:21 PM)
S73 Execute the plan in docs/plan-05-18(1).md — implement per-symbol WFO configs for 000783 and 300750, add signal quality validation, and run tests (May 18, 5:22 PM)
763 5:43p ✅ run_wfo.py --grid-key flag added to switch between param_grid and fast_grid in WFO configs
764 " ✅ Sector ETF data (515030 new energy, 516160 new energy vehicle) successfully fetched to DuckDB sector_index table
S74 Executed plan-05-18(1).md: implemented all P0-P2 code features, ran 5 WFO experiments, validated signal quality rules, scored stock eligibility, and documented results back into the plan as section 11 (May 18, 5:44 PM)
767 5:45p 🔵 000783 fast_grid WFO results: OOS annualized 3.51%, max drawdown 43.29% — far below baseline
768 " 🔵 300750 WFO with sector exit still running after 60+ seconds — 2,592 param combinations with n_jobs=4
769 " 🔵 300750 WFO with sector exit still computing after ~2 minutes — 2,592 combinations expected to take ~40 minutes
770 5:46p 🔵 300750 WFO with sector exit still running after ~5 minutes — normal for 2,592 combos
771 5:49p 🔵 300750 WFO with sector exit still computing after ~7 minutes — within expected ~40 minute runtime
772 5:51p 🔵 300750 WFO with sector exit still running after ~10 minutes — within expected ~40 minute runtime
773 5:52p 🔵 300750 WFO with sector exit terminated after ~8.5 minutes without producing results
774 " ✅ fast_grid added to wfo_300750.yaml for rapid direction validation before full grid runs
775 5:53p 🔵 300750 WFO fast_grid with sector exit launched — 96 combos, expected ~8-10 minutes
776 " 🔵 300750 fast_grid WFO with sector exit completed: max drawdown unchanged at 31.32% vs baseline 31.00%, annualized dropped from 11.95% to 9.56%
777 5:54p 🔵 Sector exit mechanism failed to reduce drawdown for 300750 — max DD 31.32% unchanged from 31.00% baseline
778 " 🔵 Signal quality rule validation reveals ATR "low=good" rule direction is wrong — high ATR outperforms low ATR for forward returns
779 5:55p 🔵 Stock eligibility scoring completed: 5 green, 8 watch, 12 exclude from watchlist_25
780 " ⚖️ Sector exit mechanism needs redesign — dual-condition OR trigger causes premature exits without preventing crash drawdowns
781 " 🔵 600519 (贵州茅台) scores only 30 on eligibility — rolling Sharpe 0.18, liquidity 0.50, contradicting plan's top-priority candidate designation
782 " 🔵 600036 (招商银行) WFO with wfo_stable.yaml launched for P1-C pool expansion — still running after ~2 minutes
783 " 🔵 000783 fast_grid WFO underperformed baseline dramatically — OOS annualized 3.51% vs 10.09%, max DD 43.29% vs 8.40%
784 5:57p 🔵 600036 (招商银行) WFO completed: OOS annualized 4.60%, max DD 26.48% — fails all promotion thresholds despite green eligibility score of 70
785 6:01p 🔵 601166 (兴业银行) WFO still running after ~3 minutes — 864 parameter combos with n_jobs=4
786 " 🔵 601166 (兴业银行) WFO completed: OOS annualized 3.03%, max DD 18.65% — also fails all promotion thresholds despite green eligibility score of 70
787 " 🔵 P2-F GBM meta-label A/B experiment launched on 000783 with fast_grid, quality_v1 labels, scale mode
788 6:02p 🔵 GBM meta-label trained in 0/12 folds for 000783 — insufficient BUY signals for training with min_samples=15
789 " ⚖️ Session summary: P0-B (sector exit) and P1-C (pool expansion) hypotheses not supported by fast_grid WFO evidence
790 6:03p ⚖️ Plan-05-18(1).md execution results documented as section 11 — all 5 experiments failed promotion thresholds; code complete and verified
S75 执行 plan-05-18(2).md 第三轮收益提升计划：实现 P0-P3 所有代码修改、运行 fast_grid WFO 实验、验证信号质量与 eligibility 评分，并将结论写入计划文档第十一节 (May 18, 6:03 PM)
791 6:26p ⚖️ Third-round execution plan created for single-stock-quant WFO performance improvement
792 6:27p 🔵 Code validation confirms all 6 planned entry filter parameters are already registered in WFO parameter routing
793 " 🔵 Signal frequency scoring in eligibility script penalizes stocks with fewer than 4 annual signals — confirmed root cause of 600519 exclusion
794 " 🔵 WFO composite_score triple frequency penalty confirmed: trade_freq_score + reliability discount + min_trades_per_year hard constraint all penalize low-frequency strategies
795 " 🔵 Existing experiment data reveals extensive prior WFO runs across 14+ stocks with structured experiment directories
796 " ⚖️ P1-A requires new parameter require_index_trend_bullish that does not exist in current code — needs full implementation across three files
797 6:29p 🟣 P1-A implemented: CSI300 MACD histogram entry gate (require_index_trend_bullish) for market-state-aware 000783 trading
798 " 🟣 P2-A implemented: signal quality scoring overhaul — dead rules removed, 4 new predictive rules added
799 " 🟣 P2-B implemented: eligibility signal frequency scoring relaxed from 4-12 to 1-12 annual signals to include slow-trend mega-caps like 茅台
800 " 🟣 P3-A implemented: reliability_mode parameter added to WFO composite scoring to remove frequency penalty for low-trade stocks
801 " ✅ Cleanup: dk_fade_exit_n removed from EXTENDED_PARAM_GRID; P0 baseline config and experiment-id CLI flag created
802 6:31p ✅ P0 baseline WFO experiment launched for 000783 with fast_grid — first execution of the third-round plan
803 " 🔵 P0 baseline results: 000783 achieves 10.62% annualized return with 28.51% drawdown under clean baseline; 300750 collapses to -2.42% annualized with 57.07% drawdown without sector_exit
804 6:32p 🔴 Benchmark index data auto-fetch fixed: WFO runner now fetches CSI300 data when require_index_trend_bullish or require_positive_rs60 is in param_grid
805 " ✅ P0 filter experiment launched for 000783 combining all entry filters in fast_grid with quality_first reliability_mode
806 6:33p 🔵 P0 filter experiment for 000783 shows all-entry-filters combined degraded OOS from +135.88% total return to -2.06% — over-filtering counterproductive for signal-sparse stocks
807 " 🔵 300750 filter experiment confirms entry filters dramatically effective: -2.42% baseline → 12.65% annualized, MDD from 57.07% → 24.81%
808 6:34p 🔵 Signal quality validation confirms 3 old rules never fire in real data — run_len_ge_2, consensus_red_ge_2, market_10d_gt_0 all show n_true=0 across all 4 stocks
809 6:35p 🔵 P2-A fix validation FAILS: revised compute_signal_quality still negatively correlated with 20-day forward returns across all 4 stocks
810 " 🔵 P2-B eligibility re-run reveals trend_quality score is universally zero — _trend_quality_score function measuring "fraction of DK red days with price rising" fails for 24/25 stocks
811 " ⚖️ Session summary: All planned P0-P3 code changes implemented, tested, and partially validated — 300750 filters highly effective (+14.91pp annualized), quality_score fix fails validation, eligibility scoring has deeper structural issues
812 6:36p ✅ WFO configs updated with output_dir for proper experiment file placement in plan_05_19 directory
813 " ⚖️ Third-round plan execution complete: all P0-P3 code delivered, validated, and 6 WFO experiments confirm stock-specific filtering is mandatory — 300750 transforms to respectable while 000783 degrades
814 6:38p ⚖️ Session closed: third-round plan fully documented in plan-05-18(2).md with execution record, experiment results, and go/no-go decisions for all branches
S76 执行 plan-05-18(2).md 第三轮收益提升计划全部任务并写入执行记录 (May 18, 6:40 PM)
**Investigated**: 代码审查了 single_stock.py 入场过滤链、generator.py 质量评分函数、wfo.py 复合评分函数、score_symbol_eligibility.py 信号频率评分；运行了 4 个 fast_grid WFO 实验、signal_quality v2 验证、eligibility round3 复评

**Learned**: 入场过滤器有强烈股票特异性——300750 受益极大（+14.91pp 年化，MDD -56.5%），000783 被严重损害（年化 -10.86pp）；quality_score 修复后仍与收益负相关（Spearman ρ -0.038 至 -0.105），简单加法无法产生有效评分；trend_quality 评分对 24/25 只标的输出 0.00 存在深层计算缺陷

**Completed**: P1-A（指数 MACD 门控）、P2-A（质量评分重构）、P2-B（eligibility 放宽）、P3-A（reliability_mode）全部实现；p0_baseline.yaml 创建；wfo_000783/300750.yaml 网格补齐；benchmark 自动加载修复；--experiment-id CLI 新增；205 测试通过，ruff 通过；计划文档第十一节执行记录写入完成

**Next Steps**: 本轮全部任务已完成，plan 状态 marked completed。按决策树：000783 放弃过滤器路径，300750 作为下一轮唯一深化对象聚焦退出层优化


Access 994k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>