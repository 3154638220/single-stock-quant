<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-17 10:03pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (9,604t read) | 1,657,537t work | 99% savings

### May 15, 2026
S51 Advance plan-05-15(2).md through E21 and E22 implementation, then pivot strategy from anchor-first to return-first calibration for Eastmoney DK Trend reverse-engineering on stock 000783 (May 15, 9:53 PM)
S52 诊断000783长江证券DK趋势背测问题：5笔交易总收益+20.61%但依赖单笔大赢家，分析入场信号、退出机制、参数稳定性 (May 15, 9:54 PM)
### May 16, 2026
S53 诊断000783长江证券DK趋势背测：5笔交易总收益+20.61%但依赖单笔大赢家，分析入场信号滞后、退出保护缺失、参数稳定性不足三大问题 (May 16, 8:44 PM)
S54 DK策略WFO修复、退出层增强、跨股验证及计划文档同步——从单股优化阶段转入标的适配阶段 (May 16, 8:46 PM)
626 9:42p ✅ 计划文档同步更新：intrapos_dd_limit=0.06 显著提升回测表现
627 " ⚖️ intrapos_dd_limit=0.06 成为回测配置的标准风控参数
629 9:43p ✅ 计划文档完成最终同步：E24/E25 里程碑更新，下一步进入 5 股小样本验证
630 " 🔵 Trade 3 具体修复：intrapos_dd_stop 将 -1.13% 扭转为 +6.71%
631 9:44p 🔵 跨股 WFO 验证启动：600030 运行中，600036 因 DuckDB 锁冲突失败
632 " ⚖️ 5股小样本跨股WFO验证启动，600030率先运行，600036需等待DuckDB锁释放
633 9:45p 🔵 600030（中信证券）DK WFO 失败：OOS Sharpe -1.15，参数漂移 0.913
634 " ⚖️ 600030 WFO 失败不阻断跨股验证，600036（招商银行）WFO 继续串行执行
635 " 🔵 600036（招商银行）DK WFO 同样失败：OOS Sharpe -0.05，参数漂移 0.816
636 9:46p 🔵 跨股验证第三只标的 000568（泸州老窖）WFO 启动
637 9:47p 🔵 000568（泸州老窖）DK WFO 也失败：OOS Sharpe -0.83，三只跨股标的连续未通过
638 " ⚖️ 跨股验证扩展至第四只标的 300750（宁德时代），测试创业板新能源股
639 9:48p 🔵 300750（宁德时代）WFO 出现两次运行结果：第二次 OOS Sharpe 达 0.47 且参数漂移为零
640 " ✅ 第五只跨股验证标的 600519（贵州茅台）WFO 启动
641 9:50p 🔵 600519（贵州茅台）DK WFO 也失败：OOS Sharpe -1.53，5只跨股标的全部未通过
642 9:51p 🔵 跨股WFO汇总分析：6只股票仅000783通过，300750两个fold参数完全相同
643 " ⚖️ 跨股WFO验证结论：DK策略不具备跨股泛化能力，不能进入25股扩展
644 " ✅ 计划文档全面更新完成：跨股WFO结果、小样本汇总表、E25方向调整全部写入
645 9:52p ⚖️ 本轮工作会话完成：7个文件修改，224行新增，DK策略从单股优化转向标的适配阶段
S55 用户请求全面分析 single-stock-quant 量化交易项目存在的问题和需要改进的地方 (May 16, 9:53 PM)
### May 17, 2026
646 9:37p 🔵 项目分析任务启动：single-stock-quant 仓库审查
647 " 🔵 single-stock-quant 项目结构梳理完成
648 9:38p 🔵 DK 趋势指标复刻状态：核心问题是从视觉逆向转向收益优先的工程重构
649 " 🔵 长江证券 (000783) WFO 深度分析揭示策略收益集中度过高和低价股摩擦成本问题
650 " 🔵 000783 DK Bar 收益率网格校准输出 252 个参数组合，存在大量负收益候选
651 " 🔵 data/output 目录积累了跨 15+ 股票的 WFO 结果和多种格式输出，缺乏清理和组织
652 " 🔵 项目配置管理体系复杂：20+ YAML 文件覆盖策略变体、WFO 阶段和生产环境
653 " 🔵 文档归档体系混乱：plan 文件散布在 docs/ 和 docs/archive/ 中，文件名含括号导致 shell 兼容问题
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
S60 用户请求Codex重构最新的plan文件（docs/plan-05-15(2).md），目标单一：提升单股量化收益 (May 17, 10:01 PM)
673 10:02p 🔵 000783 DK Bar参数网格搜索结果：最优配置为sma205_wma5组合
674 " 🔵 000783 WFO两折OOS结果：跨折Sharpe波动极大，Bootstrap置信区间极宽
675 " 🔵 数据输出目录包含丰富的跨股票WFO和多模式回测结果
676 " 🔄 plan-05-15(2).md 全面重构为单一目标：提升000783单股量化收益

Access 1658k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>