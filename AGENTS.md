<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-14 12:05pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (21,073t read) | 863,164t work | 98% savings

### May 13, 2026
S15 用户询问如何在 Claude Code 终端中重启，并确认使用的模型配置 (May 13, 6:47 PM)
S17 Process architecture audit: verified claude-mem plugin model routing and identified stale processes (May 13, 6:51 PM)
S18 单股票量化回测系统 Phase 2/4/5 多任务推进：质量评分缩放仓位、Donchian突破信号、WFO复合目标函数、增强质量评分引擎 (May 13, 6:58 PM)
S19 用户询问进度是否已记录到 plan.md，主会话读取并更新了该文件 (May 13, 7:19 PM)
S20 Executed docs/plan.md Phase 8 (Meta-label integration) and Phase 9 (MA120/RS60 trend filters) — the highest-ROI improvements for the single-stock-quant A-share backtest framework (May 13, 7:26 PM)
S21 Execute docs/plan.md — implement Phase 8 (Meta-label integration), Phase 9 (MA120/RS60 trend filters), and Phase 10 (Portfolio layer with meta-label scoring, CLI, experiments) (May 13, 10:20 PM)
### May 14, 2026
298 1:31a 🔴 Portfolio HTML report crash fixed — SingleStockBacktestResult now initializes all 19 required fields
299 " 🔴 FutureWarning spam in signal_ranker.py silenced — pct_change fill_method=None and reindex fill_value=False
300 " 🔵 Phase 10 portfolio OOS results: Meta-label + MA120 configuration underperforms baseline
301 1:32a 🔵 Second portfolio run confirms meta-label + MA120 underperformance; HTML report now generates successfully
S22 Execute docs/plan.md improvement plan — completed Phase 11 (meta-label features v2) and Phase 12 (WFO stability architecture upgrade) with full test verification (May 14, 1:33 AM)
302 11:28a 🔵 Plan document structure: 15-phase improvement roadmap for single-stock-quant
303 " 🔵 WFO module already has platform-based parameter selection with stability scoring
304 " 🔵 Meta-label model uses L2-regularized logistic regression with 10 features from technical indicators
305 " 🔵 15 modified files and 1 new untracked file in working tree indicate active development across the codebase
306 11:30a 🟣 Phase 11: Eight new meta-label features added to build_signal_features()
307 " 🟣 Test for Phase 11 features verifies all 8 new columns exist and validate bounds
308 11:31a 🔵 Phase 12 patch application failed: file state mismatch in wfo.py nested WFO section
309 11:32a 🟣 Phase 12: _select_stable_params() cross-fold stability selector added to wfo.py
310 " 🟣 Phase 12 integration complete: stability-weighted parameter selection wired into nested WFO
311 " 🔄 All Phase 11 and Phase 12 tests pass: 15 tests across test_meta_label, test_signal_features_stability, and test_wfo_params_split
312 11:33a ✅ Plan document updated to reflect Phase 11 and Phase 12 completion
313 " ✅ Full test suite passes: 161 tests across 24 test files, all green
314 11:34a ✅ Plan execution complete: Phases 11 and 12 implemented and verified with full test suite
315 " 🟣 stability_weighting parameter wired through full WFO stack with CLI flag
316 11:35a ✅ Full test suite confirms Phases 8-12 complete: 161 tests pass with zero failures
S23 执行 docs/plan.md 改进计划：实现阶段 13（周线多周期趋势过滤）和阶段 14（EWMA波动率动态仓位精细化），完成全部 5 步执行计划，168 个测试全绿 (May 14, 11:36 AM)
317 11:48a 🔵 Plan execution scope revealed across single-stock-quant codebase
318 11:49a 🔵 Stages 8-12 code completion confirmed; plan execution targets stages 13-14
319 " ⚖️ Execution plan scoped to stages 13 (weekly trend) and 14 (EWMA/ATR position sizing)
320 11:50a 🟣 Weekly trend aggregation module created for multi-timeframe signal filtering
321 " 🟣 Weekly trend filter and EWMA volatility position sizing integrated into backtest engine
322 11:51a 🟣 Stage 13/14 parameters wired through config, WFO, and CLI layers
323 11:52a ✅ Default config and settings updated with stage 13/14 parameters
324 " 🟣 Weekly trend unit tests created with bullish/bearish detection and lookahead bias protection
325 " 🟣 Integration test added for weekly bullish filter blocking daily BUY in weekly downtrend
326 11:53a 🟣 Position management and config tests extended for stage 14 EWMA/ATR features
327 " ✅ All targeted tests pass for stages 13 and 14; plan status updated
328 " ✅ plan.md updated with stage 13/14 completion status and verification commands
329 11:54a 🟣 EWMA volatility CLI arguments added to both single and WFO entry points
330 " ✅ Final plan step reached — full test suite execution begins
331 " ✅ Trailing whitespace lint fixed in plan.md
332 " 🟣 Stages 13-14 complete: all 168 tests pass with zero regressions
333 11:55a 🔴 Removed overly aggressive dropna from weekly_trend _prepare_daily to prevent data loss
334 11:56a 🟣 Plan execution complete: stages 8-14 implemented, 168 tests passing, zero regressions
335 " ✅ Plan execution fully completed: all 5 steps done, 168 tests green
336 11:57a 🔵 Portfolio E10 first real-data experiment produced poor OOS results
337 " ⚖️ Four-phase experiment plan defined to improve portfolio OOS metrics after E10 failure
338 " 🔵 New untracked files since last session: weekly_trend.py and two test files
S24 执行 docs/plan.md — 补齐阶段 15 实验闭环与报告完善工具 (May 14, 11:57 AM)
339 11:58a 🟣 Phase 15 implementation started: experiment comparison and report automation tools
340 11:59a 🟣 Phase 15 experiment comparison engine implemented in src/backtest/experiment.py
341 12:00p 🟣 Phase 15 CLI comparison script created: scripts/compare_experiments.py
342 " 🟣 Phase 15 tests added for experiment comparison engine
343 " 🟣 Phase 15 fully implemented: experiment comparison engine, CLI script, and tests all passing
344 12:01p ✅ Plan.md updated: Phase 15 marked complete, all 15 phases now implemented
345 " ✅ Full test suite passes: 168+ tests green including Phase 15 additions
346 " 🔄 Removed unused variable in evaluate_experiment() to pass ruff lint
347 12:02p ✅ Plan execution complete: all 15 phases implemented and tested
S25 执行 docs/plan.md Phase 15 — 补齐实验闭环与报告完善工具 (May 14, 12:03 PM)
**Investigated**: 确认了 plan.md 中阶段 8-14 代码层已完成，仅阶段 15 未落地。审查了现有 experiment.py、test_experiment.py 及多个 CLI 脚本以了解代码模式。发现 evaluate_experiment() 中存在一个既有的未使用变量 lint 错误。

**Learned**: 阶段 15 需要四个核心能力：多文件实验指标加载（含中位数聚合）、指标方向感知的对比引擎、HTML/Markdown 双格式报告输出、CLI 封装。create_experiment_dir() 需要预生成 ARTIFACTS.md 和 DELTA.md。

**Completed**: src/backtest/experiment.py 新增 ~260 行：EXPECTED_EXPERIMENT_ARTIFACTS 常量、load_experiment_metrics()、compare_metric_summaries()（含 CI 重叠检测）、write_delta_markdown()、render_experiment_comparison_html()，create_experiment_dir() 现在自动生成 ARTIFACTS.md 和 DELTA.md。scripts/compare_experiments.py 新建（80 行 CLI）。tests/test_experiment.py 新增 TestExperimentComparison 类（4 个测试）。docs/plan.md 更新阶段 15 状态为已完成。修复了 evaluate_experiment() 中的 F841 lint 错误。全量 pytest 173 通过，ruff 检查通过。

**Next Steps**: 代码层 15 个阶段全部完成。下一步应运行 E11/E12/E13/E14/E_FINAL 真实数据实验，验证新特征、稳定选参、周线过滤和 EWMA 仓位约束是否能改善 WFO OOS 指标，并使用 scripts/compare_experiments.py 生成 DELTA 对比报告。


Access 863k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>