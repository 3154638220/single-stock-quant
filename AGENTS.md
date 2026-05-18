<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-18 6:03pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (16,137t read) | 1,077,953t work | 99% savings

### May 17, 2026
S64 Complete plan-05-15(2).md — execute all five R01-R05 tasks for 000783 single-stock return improvement, including P0 refactoring of the scoring pipeline from anchor-matching to pure return-quality optimization (May 17, 10:07 PM)
S65 Complete plan-05-15(2).md — execute all five R01-R05 tasks: refactor scoring from anchor-matching to pure return-quality, re-rank grid candidates, run fixed-parameter WFO on top-10, test exit/entry experiment overlays, and produce best strategy config (May 17, 10:24 PM)
S66 多股票最佳单股策略配置导出：产出 5 只 WFO passing 股票的独立最佳参数配置及 246 笔逐笔交易明细 (May 17, 10:27 PM)
### May 18, 2026
S67 Record multi-stock best single-stock config results into plan-05-15(2).md (May 18, 4:32 PM)
S68 Complete tasks documented in docs/plan-05-18.md (May 18, 4:37 PM)
S69 Complete tasks documented in docs/plan-05-18.md (May 18, 5:02 PM)
S70 执行 docs/plan-05-18.md 中的单只A股量化策略收益提升计划 — 包括WFO框架修复、出场层重设计、参数网格扩展、元标签升级、波动率仓位实验、性能优化和601318复评剔除 (May 18, 5:03 PM)
S71 Check completion status of plan-05-18.md — user wants to know if all planned tasks are done (May 18, 5:17 PM)
S72 Check completion status of plan-05-18.md — user asked whether all originally planned content has been completed (May 18, 5:21 PM)
740 5:35p 🔵 05-18 experiments completed with no tradable configurations meeting thresholds
741 " ⚖️ dk_fade_exit mechanism formally abandoned for eastmoney_dkbar mode
742 " ⚖️ Revised priority matrix defines 8 improvement directions with P0 focused on 000783 WFO constraints and 300750 sector exits
743 " 🔵 WFO composite score has hidden bias favoring high-trade-frequency parameter combinations
744 " 🔵 300750's 31% max drawdown traced to 2022 new energy sector collapse — no price-based exit can protect
746 5:37p 🟣 Sector exit mechanism implemented in single_stock.py backtest engine
747 " 🟣 Dynamic profit_lock based on entry quality (high-quality entries get wider stops) implemented in single_stock.py
748 " ✅ New backtest parameters wired through config.py and wfo.py for WFO grid search
749 5:38p 🟣 ETF daily data fetcher `fetch_etf_daily()` added to akshare_client.py via AkShare Eastmoney fund_etf_hist_em
750 " 🟣 Sector index auto-loading wired into run_wfo.py and run_backtest_single.py entry points
751 5:40p ✅ GBM meta-label conservative defaults aligned with plan P2-F recommendations
752 " ✅ dk_fade_exit_n removed from wfo_stable.yaml param grid as confirmed ineffective mechanism
753 " 🟣 Two per-symbol WFO configs created: wfo_000783.yaml and wfo_300750.yaml
754 " 🟣 fetch_stock.py extended with --watchlist flag, --db-table override, and ETF fetching support
755 5:41p 🟣 Signal quality rule validation script validate_signal_quality_rules.py created per plan P2-E
756 " ✅ Two new unit tests added for sector_exit and dynamic profit_lock HQ thresholds
757 5:42p 🟣 New per-symbol WFO configs for 000783 and 300750 with symbol-specific strategies
758 " 🟣 Signal quality rule validation script using Mann-Whitney U and Cliff's delta
759 " 🔴 Fixed profit_lock_trigger threshold too high in HQ profit lock test
760 " ✅ Added sector exit and HQ profit-lock backtest tests
761 " ✅ Fixed 5 ruff lint issues across backtest and fetch modules
762 " 🔵 ETF data for 515030/516160 missing from local DuckDB; network blocked in sandbox
763 5:43p ✅ run_wfo.py --grid-key flag added to switch between param_grid and fast_grid in WFO configs
764 " ✅ Sector ETF data (515030 new energy, 516160 new energy vehicle) successfully fetched to DuckDB sector_index table
765 " ✅ Sector index fallback logic: both WFO and single backtest entry points auto-detect sector_index DuckDB table
766 " ✅ All 50 unit tests pass; ruff lint clean across all modified files
S73 Execute the plan in docs/plan-05-18(1).md — implement per-symbol WFO configs for 000783 and 300750, add signal quality validation, and run tests (May 18, 5:44 PM)
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

Access 1078k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>