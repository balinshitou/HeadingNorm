# 运行前判定标准（写于分析脚本运行之前，2026-09-26）

问题：论文报告的外挂HN收益 = HN(α=0) − 原样(ψ=0)。α（规范系朝向约定）与 ψ（初始参考方向）都是任意的。
收益在两边都按 8 个等距角平均后是否仍然存在？

数据：results/hn_plugin_p0_20260912/ext_resnet.json（原作者 RoNIN ResNet；orig_ate_by_psi、hn_ate_by_alpha，文献协议）。
次要：本文 ResNet18-YawAug 4 种子（e1 ψ 扫描 + unified_eval ours_yaw_hn 的 ate_lit；无 α 扫描）。

主终点（RoNIN ResNet）：Δ_fair = mean_α HN − mean_ψ 原样，受试者等权 ATE。
主要测试集：RoNIN-unseen、RIDI、TLIO-test（3 项；RoNIN-seen 为原作者训练受试者，只作描述）。
区间：10,000 次受试者整群 bootstrap，种子 20260906；Bonferroni 3 项 → 98.33%。
判定：3 项中 ≥2 项区间排除零且为负、且无一项区间排除零为正 → "收益不是方向约定的偶然结果"成立；
全部含零 → 不成立（外挂收益在方向平均意义下未检出）；出现显著为正 → 反转。
描述：H0−U0、H0−Ū、H0−H̄（约定运气）、U0−Ū（原生方向运气）、序列级 H̄<Ū 比例、Umin 对照。
