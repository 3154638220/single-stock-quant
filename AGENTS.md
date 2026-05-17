<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-17 10:25pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (15,268t read) | 2,130,074t work | 99% savings

### May 16, 2026
S55 用户请求全面分析 single-stock-quant 量化交易项目存在的问题和需要改进的地方 (May 16, 9:53 PM)
### May 17, 2026
648 9:38p 🔵 DK 趋势指标复刻状态：核心问题是从视觉逆向转向收益优先的工程重构
649 " 🔵 长江证券 (000783) WFO 深度分析揭示策略收益集中度过高和低价股摩擦成本问题
650 " 🔵 000783 DK Bar 收益率网格校准输出 252 个参数组合，存在大量负收益候选
651 " 🔵 data/output 目录积累了跨 15+ 股票的 WFO 结果和多种格式输出，缺乏清理和组织
652 " 🔵 项目配置管理体系复杂：20+ YAML 文件覆盖策略变体、WFO 阶段和生产环境
654 9:41p 🔵 2026-05-16 最新 WFO 结果确认：仅 2/6 只标的 Sharpe 为正，泛化能力严重不足
655 " 🔵 生产环境配置 prod-v1 仅使用 4 只 WFO 通过的标的，Meta-label 因 600030 上劣化被禁用
656 " 🔵 calibrate_eastmoney_dkbar.py 的 objective_score 公式权重硬编码且缺乏校准依据
657 " 🔵 persistent_price_change 颜色状态机在纯 Python 循环中逐行计算，性能瓶颈明显
658 " 🔵 README 文档严重过时——仅描述 macd_cross/ma_cross/boll_trend 三种模式，未提及 eastmoney_dkbar
659 " 🔵 回测引擎积累了 7 层退出机制和 5 层仓位缩放，复杂度已接近生产系统但缺乏集成测试覆盖
660 " 🔵 test_dktrend.py 和 test_eastmoney_anchor_scoring.py 验证了 bar_color/trend_state 拆分和锚点加载的基础逻辑
661 9:43p 🔵 代码质量扫描：ruff 发现 43 个问题（30 个可自动修复），mypy 发现 13 个类型错误
662 " 🔵 WFO JSON 输出存在 NaN 序列化问题——heatmap 数据包含大量 NaN 值
663 " 🔵 000783 DK WFO 的参数选择揭示了 platform 机制的实际效果：两个 Fold 均未选峰值
S56 系统化更新项目目标表述和文档措辞：将项目从"东财指标复刻"重新定位为"单股量化交易系统" (May 17, 9:45 PM)
664 9:48p 🔵 新任务启动：系统化更新项目目标表述和文档措辞
665 " 🔵 全项目"复刻/东财"关键词搜索揭示目标表述散布在 30+ 个文件中
666 " 🔵 docs/indicator_formula.md 和 configs/README.md 同样过时——仅覆盖 3 种基本模式，未提及 EASTMONEY_DKBAR
667 9:49p ✅ README.md 已更新：修正"Eastmoney-style"误导性表述，明确项目目标为单股交易验证
668 " ✅ docs/indicator_formula.md 已更新：移除"approximates Eastmoney"的误导性声明，新增 eastmoney_dkbar 章节
669 9:50p ✅ docs/plan-05-15(2).md 已完全重写：从"DK Trend 复刻计划"转为"单股量化交易计划"
670 " ✅ src/indicators/dktrend.py 模块文档字符串和注释已更新以反映新目标
671 " ✅ calibrate_eastmoney_dkbar.py 和 plot_dktrend.py 的文档字符串和图表标题已更新
672 " ✅ DK Bar 配置文件（eastmoney_dkbar_test.yaml、wfo_eastmoney_dkbar.yaml）添加研究性质声明头注释
S57 完成项目目标表述和文档措辞的系统化更新：将 single-stock-quant 从"东财指标复刻"重新定位为"单股量化交易系统"，涵盖 10 个文件的修改 (May 17, 9:51 PM)
S58 session 完成了两阶段工作：(1) 全面分析 single-stock-quant 项目的问题和改进方向；(2) 系统化更新项目文档将定位从"东财指标复刻"改为"单股量化交易系统" (May 17, 9:52 PM)
S59 用户请求Codex重构最新的plan文件，目标单一：提升单股量化收益 (May 17, 9:53 PM)
S60 用户请求Codex重构最新的plan文件（docs/plan-05-15(2).md），目标单一：提升单股量化收益 (May 17, 10:00 PM)
S61 用户请求Codex重构最新的plan文件（docs/plan-05-15(2).md），目标单一：提升单股量化收益 (May 17, 10:01 PM)
673 10:02p 🔵 000783 DK Bar参数网格搜索结果：最优配置为sma205_wma5组合
674 " 🔵 000783 WFO两折OOS结果：跨折Sharpe波动极大，Bootstrap置信区间极宽
675 " 🔵 数据输出目录包含丰富的跨股票WFO和多模式回测结果
676 " 🔄 plan-05-15(2).md 全面重构为单一目标：提升000783单股量化收益
S62 重构最新plan文件 docs/plan-05-15(2).md，目标单一：提升单股量化收益 (May 17, 10:03 PM)
S63 用户询问"当前模型公式是什么"——Codex深入解读了项目核心模型实现，给出了完整的公式、算法和架构说明 (May 17, 10:04 PM)
677 10:04p 🔵 dktrend.py是核心指标模块，支持8种趋势模式，DK Bar参数已注明不追求官方复刻
678 10:05p 🔵 single_stock.py回测引擎支持10种退出机制和5层仓位管理
679 " 🔵 WFO复合目标函数以Calmar为主权重（40%），硬约束限制年交易4-24次且回撤≤40%
680 " 🔵 compute_signal_quality评分函数为BUY信号提供8维度0-100质量分
681 10:13p 🔵 Strategic pivot: single-stock quant project shifts from anchor-matching to pure return optimization
682 " 🔵 000783 return grid shows sma205/wma5/c3 as current WFO baseline with low drawdown, sma60 variants as higher-return alternatives
683 " 🔵 Codebase structure: single-stock-quant project has mature WFO framework, trade forensics, and experiment infrastructure
684 10:16p 🔵 WFO JSON output files contain widespread NaN values across heatmap and stability fields
685 " 🟣 R01 implemented: return_quality_score extracted, trade concentration metrics added, anchor diagnostics removed from primary scoring
686 10:17p 🟣 WFO now outputs per-fold trade concentration metrics and JSON-safe NaN/inf sanitization
687 10:18p 🟣 New orchestration script run_000783_return_plan.py implements all five R01-R05 tasks from plan-05-15(2)
688 10:19p 🟣 All 16 existing tests pass after P0/P1 code changes; plan execution script launched successfully
689 " 🔵 Plan execution script still running after 30s — 252 grid re-enrichment + 34 WFO runs expected to take minutes
690 10:20p 🔵 run_000783_return_plan.py crashes at YAML write: numpy int64 not serializable by PyYAML safe_dump
691 " 🔴 YAML serialization crash fixed: _native() function converts numpy scalars to Python native types before safe_dump
692 " 🔵 Re-launched plan script still computing after 30s — full workload of ~300 backtests in progress
693 10:21p 🟣 Plan-05-15(2) fully executed: all five R01-R05 outputs produced with best strategy identified
694 " ⚖️ Exit-layer optimization confirmed as primary return lever over entry signal tuning
695 10:22p ✅ Plan-05-15(2).md updated with completion status and execution results section 7.1
696 " 🟣 Plan-05-15(2) fully completed: P0 refactoring + R01-R05 execution delivered in single session
697 10:23p ✅ Code quality: ruff linting errors resolved, unused variable removed from nested WFO path
698 " ✅ Final quality gate: all 201 tests pass, all ruff linting passes, clean git diff shows targeted changes only
S64 Complete plan-05-15(2).md — execute all five R01-R05 tasks for 000783 single-stock return improvement, including P0 refactoring of the scoring pipeline from anchor-matching to pure return-quality optimization (May 17, 10:24 PM)
**Investigated**: The plan document docs/plan-05-15(2).md was read in full, establishing the strategic pivot from eastmoney DKBar anchor replication to pure return optimization for stock 000783. The existing grid search output (000783_dkbar_return_grid_full.csv, 252 rows) was examined, revealing that all 7 anchor diagnostic columns are near-identical across candidates (matched_anchors=7, bar_color_accuracy=0.714 constant), confirming they provide zero discrimination for parameter selection. The full codebase was surveyed: WFO framework (wfo.py with _select_platform, _select_stable_params, bootstrap_sharpe_ci, nested WFO), backtest engine (single_stock.py with exit mechanisms including intrapos_dd_limit, dk_fade_exit_n, profit_lock, time_stop), grid search script (calibrate_eastmoney_dkbar.py with blended scoring), and 22 config YAMLs. WFO JSON outputs from 20260516 were found to contain widespread NaN values in heatmap grids and turnover_mean fields, confirming the plan's P0 requirement to fix NaN/inf serialization.

**Learned**: The existing objective_score function in calibrate_eastmoney_dkbar.py blends return_quality with 0.35*anchor_fit, inflating scores for visually "anchor-like" parameter sets that don't necessarily trade well. The exit layer is confirmed as the primary bottleneck — the strategy captures directional moves but loses profit on exits, validated by the experiment results where intrapos_dd_limit=0.05 outperformed all baseline candidates. The sma205/wma5/c3 family produces only 5 trades over the full period with 11.7% drawdown but 30% total return, while sma60 variants produce 15+ trades with up to 54.9% return but 27%+ drawdown. Meta-label models (meta_label.py, meta_label_gbm.py) exist but are orphaned — p_win predictions never affect trade decisions. Pandas Series preserves numpy dtypes (np.int64) when accessed by key, causing PyYAML safe_dump to fail — requiring explicit _native() conversion. The WFO framework already supports nested WFO, stability-weighted param selection, Bootstrap CI, and platform selection (avoiding isolated parameter peaks), but currently only generates 2 OOS folds for 000783.

**Completed**: **P0 Refactoring (scoring pipeline unification):**
    - Extracted `return_quality_score()` from `objective_score()` in calibrate_eastmoney_dkbar.py — pure trading score with: sharpe + 1.50*ann + 0.50*excess + 0.25*ret + 0.20*calmar + trade_bonus - 1.25*max_dd - low_trade_penalty - concentration_penalty
    - Added `trade_contribution_metrics()` to both calibrate_eastmoney_dkbar.py (tuple return) and wfo.py (dict return) — computes largest single winning trade and its share of total return
    - Concentration penalty: max(0, contribution - 0.60) * 0.75 penalizes single-trade-dependent strategies
    - Trade bonus: min(max(trades, 0), 20) / 20.0 * 0.15 rewards independent trades (capped at 20)
    - Anchor diagnostics removed from primary score; only enter via "blended" or "anchor_fit" sort modes
    - Main loop now emits both `return_quality` and `objective_score` columns

    **P0 NaN/Inf fix:**
    - Added `json_safe()` to wfo.py — recursively converts NaN/Inf→None, np.generic→native, pd.Timestamp→isoformat
    - Wired into run_wfo.py: `json.dump(json_safe(result), ..., allow_nan=False)` for strict JSON compliance
    - Verified: `{"x": null, "y": [null, null, 1.0]}` output for NaN/Inf inputs

    **P0 Trade contribution in WFO:**
    - Regular WFO oos_panels now include per-fold: total_return, annualized_return, max_drawdown, calmar_ratio, n_trades, largest_trade_return, largest_trade_contribution
    - Nested WFO outer_fold_results now include oos_largest_trade_return and oos_largest_trade_contribution

    **R01-R05 Execution (new orchestration script):**
    - Created `scripts/run_000783_return_plan.py` (~340 lines) — single CLI script implementing all five tasks
    - R01: Re-enriched 252 grid candidates with new return_quality scoring → `data/output/000783_return_candidates.csv` (85KB)
    - R02: Fixed-parameter WFO on top-10 candidates (756 train / 252 OOS days, 3 folds) → `data/output/000783_top10_wfo_contribution.csv` (2.7KB), `data/output/000783_top10_wfo_trades.csv` (14KB)
    - R03/R04: 26 experiment variants (intrapos_dd [0.04-0.08], dk_fade [3-5], profit_lock [3 combos], state_confirm_days [1,2]) crossed with best sma60 and sma205 candidates → `data/output/000783_exit_entry_experiments.csv` (7.9KB)
    - R05: Multi-tier eligibility gate selects best → `configs/research/000783_best_return.yaml` (2.1KB)
    - Fixed numpy→YAML serialization crash by adding `_native()` converter function
    - Fixed dead `panel` variable in nested WFO path (ruff F841)
    - Fixed import ordering (ruff I001 via --fix)

    **Best result:**
    - Experiment: `R03_R04_03_intrapos_dd_0.05` (sma60/wma5/price_change/c2/h0 with intrapos_dd_limit=0.05)
    - OOS total return: 47.61% (↑ from 29.6% baseline)
    - OOS annualized: 13.86% (short of 16% target)
    - OOS max drawdown: 10.46% (within ≤15% target)
    - OOS Calmar: 1.33 (↑ from 1.28 baseline)
    - 7 trades, largest contribution: 48.39% (within ≤60% target)

    **Quality gates:**
    - 201/201 tests pass (zero failures)
    - ruff: "All checks passed!" across all 5 modified Python files
    - Git diff: 157 insertions, 30 deletions across 6 modified files + 2 new files

    **Plan document updated:**
    - Section 7 task table marked all R01-R05 as "已完成" with output paths
    - New section 7.1 records execution results and next-round guidance

**Next Steps**: The plan's section 7.1 conclusion recommends next round focus on sma60/wma5/c2 exit layer refinement and trade count improvement to reach the 16% annualized return target. The orchtration script (run_000783_return_plan.py) is fully reproducible — can be re-run with different parameters. No further work was requested or indicated in this session.


Access 2130k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>