"""生成 v3 补充材料 paper/merged/supplement_zh_v3.md（2026-09-15）。

来源：v2 补充材料 paper/merged/build/supplement_zh_v2.md（其中"现算"表格由 tools/build_supplement_v2_20260913.py 从结果文件生成）。
改动：
  1. 删去已被推翻或已被取代的实验章节：旧 S17（完整管线 vs 无前端卷积管线）、S19（学习曲线与 TLIO 事后检验）、
     S20（历史管线的速度残差）、S23（外挂到序列网络与重训结构）、S24（加速度计尺度自校准）。依据见 docs/04。
  2. 删去表 S18 的 25%/50% 两行（学习曲线子集）与统一评测全表中的两个"整段 HN"模型；删去运行前修订中尺度自校准一条。
  3. 以旧稿为数字出处的表，改写为结果文件出处（results/、verification/ 与 results/supplement_sources_20260909/）。
  4. 新增 S26：确认性检验的完整结果，由 results/confirm_20260913/verdict.json 直接生成。
  5. 章节、表、图重新连续编号；只改补充材料内部的交叉引用（设备名 S10/S21、公式标签不受影响）。
  6. 逐表核对：表注中列出的出处文件里，能否找到表中的每个数（按印出的小数位），报告写入 provenance/补充材料v3_出处核对_20260915.json。
用法：.venv/bin/python tools/build_supplement_v3_20260915.py
"""
from __future__ import annotations
import bisect, csv, glob, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'paper/merged/build/supplement_zh_v2.md'
OUT = ROOT / 'paper/merged/supplement_zh_v3.md'
REPORT = ROOT / 'provenance/补充材料v3_出处核对_20260915.json'
V3_TITLE = '惯性速度回归中的水平参考方向：主轴符号歧义的消除与已训练网络的测试时精确规范化'
DROP = {17, 19, 20, 23, 24}
SS = 'results/supplement_sources_20260909'

# 旧表号 → 新出处（替换表注中"出处：……。"一句）
SOURCES = {
    1: f'出处：`{SS}/accuracy_unrounded.csv`（由`results/sensors_v4/analysis.json`与`results/e3_backbone_frontend/analysis.json`汇总）。',
    2: f'出处：`{SS}/table_s2.csv`（由`results/e1_heading_repeatability/`、`results/e3_backbone_frontend/`与`results/e6_sign_ablation/`的逐序列记录计算）。',
    3: f'出处：`{SS}/table_s3a.csv`（由`results/e1b_yaw_aligned/`的逐序列记录计算）。',
    4: '出处：`results/sensors_v4/analysis.json`（`suffix_sensitivity`）。',
    5: f'出处：`{SS}/table_s4a.csv`（由`results/e4_error_timescale/`计算）。',
    6: f'出处：`{SS}/table_s4b.csv`（由`results/e4_error_timescale/`计算，与正文表9一致）。',
    7: f'出处：`{SS}/geometry_unrounded.csv`。',
    8: '出处：`results/sensors_v4/diagnostics.json`。',
    10: '出处：`results/sensors_v4/analysis.json`（`budget`）。',
    11: '出处：`results/e6_sign_ablation/analytic_frame_check.json`（与正文表8一致）。',
    13: f'出处：`{SS}/table_euclidean.csv`。',
    # 2026-09-16：表中的 Huber 阈值与 AdamW 权重衰减不在配置文件里，补上代码出处
    14: '出处：RoNIN论文[2]与`config/sensors_revision_v4.json`；Huber阈值0.27见`src/pkg_train.py`的`--delta`默认值，'
        'AdamW权重衰减1e-4见`src/sensors_train.py`。',
    15: '出处：`verification/priority_revision_20260909/loss_report.json`。',
    16: '出处：`verification/priority_revision_20260909/temporal_summary.json`与`verification/priority_revision_20260909/temporal_horizons.csv`。',
    17: f'出处：`{SS}/table_tails.csv`。',
    18: '出处：检查点`models/submission_frozen/ResNet18_v3_hn_gn_s0.pt`中的`gnorm_mu`与`gnorm_sd`。',
}
# 旧图号的出处（原写"出处：历史中文补充证据……"）
FIG_SRC = '图由`tools/make_rq_paper_assets.py`从`results/submission_frozen/eval_rq1_rq4_v3_registered_summary.json`生成。'

HEADER = f"""# 补充材料

**{V3_TITLE}**

Yanlin LI，Xichen CUI，Yu QUAN\\*

> 本补充材料与正文v3配套，编号以S开头，由`tools/build_supplement_v3_20260915.py`生成：在v2补充材料的基础上删去已被推翻实验的章节、
> 把以旧稿为出处的表格改写为结果文件出处，并重新编号；S26由确认性检验的结果文件直接生成。凡标注"现算"的表格直接从结果文件生成；
> 其余表格在表注中写明结果文件出处，构建脚本逐表核对表中每个数都能在所注出处中找到（`provenance/补充材料v3_出处核对_20260915.json`）。
> ATE与RTE一律采用RoNIN官方实现的逐坐标RMSE约定（换算见S11）。

## 目录

- S1–S13　受控前端比较的实现、诊断与复现（对应正文§3–§4.6）
- S14–S17　GlobalNorm、PreFilter与40次训练中的组件消融、前端移植与主干比较
- S18–S19　跨数据集统一评测的完整结果与显著性（对应正文§3.8、§4.7）
- S20–S22　手机数据预处理、运行前的协议修订、脚本与结果文件
- S23–S25　测试时精确规范化：增强训练 + 外挂HN、C8测试时平均、两方向帧平均（对应正文§4.8）
- S26　确认性检验的完整结果（对应正文§4.9）
- S27　HN与偏航增强在三种主干上的配对比较（对应正文§4.5）

---
"""

NAME = {'ours_yaw_hn': '增强网络外挂HN', 'ours_yaw_fa2': '增强网络两方向帧平均', 'ours_yaw': '增强网络原样', 'ours_hn': '训练时用HN',
        'ours_pca': '混合PCA', 'ours_gn': '只做GN', 'ext_resnet_hn': '官方ResNet外挂HN', 'ext_resnet': '官方ResNet原样',
        'ours_yaw_tta8': '增强网络C8平均'}
DSN = {'tlio_c': 'TLIO（train+val）', 'imunet_c': 'IMUNet手机（训练划分）'}
MET = {'ate_u': '统一口径ATE', 'rte_u': '统一口径RTE', 'ate_lit': '文献协议ATE'}


def f3(x): return f'{x:+.3f}'.replace('-', '−')
def m3(x): return f'{x:.3f}'
def pc(x): return f'{x:+.1f}%'.replace('-', '−')
def ci(c): return f'[{f3(c[0])}, {f3(c[1])}]'


def s26() -> str:
    v = json.loads((ROOT / 'results/confirm_20260913/verdict.json').read_text())
    n_t, n_i = v['manifest_n']['tlio_c'], v['manifest_n']['imunet_c']
    call = {'better': '成立', 'noninferior': '非劣成立', 'not_shown': '未显示非劣', 'worse': '显著更差'}
    # 标题写旧编号 31（v2 最大章节号为 30），由下面的重排逻辑统一映射为 S26；
    # 曾误写为 '## S26.'，被当成旧编号 26（运行前的协议修订）映射成 S21，与该节重号（2026-09-16 修正）
    lines = ['## S31. 确认性检验的完整结果', '',
             '判定标准在任何评测之前写定（`docs/53`），结果与判定见`docs/54`；判定脚本`tools/confirm_verdict_20260913.py`只运行一次。',
             f'数据是本文此前从未评测过的两批序列：TLIO的train与val划分（{n_t}条；该数据集不提供受试者信息，按序列计），',
             f'以及IMUNet作者手机数据的训练划分（{n_i}条、4名受试者；另有{len(v["failed"]["imunet_c"])}条因时间戳不递增按运行前规则剔除）。',
             '本文模型均不在这两批数据上训练。主要检验族共8项（4个假设×2个数据集），按Bonferroni校正取99.375%区间；',
             '手机数据要求受试者区间与序列区间同时满足。H2为非劣检验，界值为对照均值的+2%。', '',
             '**表S39. 主要检验族（统一口径ATE，m；处理 − 对照；现算自`results/confirm_20260913/verdict.json`）。**', '',
             '| 假设 | 比较 | 数据集 | 处理 | 对照 | 差值（相对） | 受试者区间 | 序列区间 | 改善（受试者/序列） | 判定 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in v['main']:
        lines.append(f"| {r['hypothesis']} | {NAME[r['treatment']]} − {NAME[r['control']]} | {DSN[r['dataset']]} | {m3(r['treatment_mean'])} | "
                     f"{m3(r['control_mean'])} | {f3(r['difference'])}（{pc(r['rel_pct'])}） | {ci(r['subject_ci'])} | {ci(r['seq_ci'])} | "
                     f"{r['improved_subjects']}/{r['n_subjects']}、{r['improved_sequences']}/{r['n_sequences']} | {call.get(r['call'], r['call'])} |")
    lines += ['', '**表S40. 描述性比较（95%区间，不参与判定；现算自`results/confirm_20260913/verdict.json`）。**', '',
              '| 比较 | 数据集 | 指标 | 处理 | 对照 | 差值（相对） | 受试者区间 | 序列区间 | 改善受试者 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in v['descriptive']:
        t, c = NAME.get(r['treatment'], r['treatment']), NAME.get(r['control'], r['control'])
        lines.append(f"| {t} − {c} | {DSN[r['dataset']]} | {MET.get(r['metric'], r['metric'])} | {m3(r['treatment_mean'])} | {m3(r['control_mean'])} | "
                     f"{f3(r['difference'])}（{pc(r['rel_pct'])}） | {ci(r['subject_ci'])} | {ci(r['seq_ci'])} | {r['improved_subjects']}/{r['n_subjects']} |")
    dev = v['imunet_by_device']
    models = sorted({k.split('|')[0] for k in dev}, key=lambda m: list(NAME).index(m) if m in NAME else 99)
    devices = sorted({k.split('|')[1] for k in dev})
    lines += ['', '**表S41. IMUNet手机数据（训练划分）按设备的序列平均统一口径ATE（m；现算自`results/confirm_20260913/verdict.json`）。**', '',
              '| 设备（条数） | ' + ' | '.join(NAME.get(m, m) for m in models) + ' |', '|---|' + '---:|' * len(models)]
    for d in devices:
        n = dev[f'{models[0]}|{d}']['n']
        lines.append(f'| {d}（{n}） | ' + ' | '.join(m3(dev[f"{m}|{d}"]["mean"]) for m in models) + ' |')
    return '\n'.join(lines) + '\n'



def s32_backbone_aug() -> str:
    """新增节（旧编号 32 → 重排后 S27）：三种主干上 HN 与偏航增强的配对差。

    2026-09-16 补：原 v3 只在正文 §4.1 报告 ResNet18 上的这组比较，Transformer 与 IMUNet 的
    对照仅以点估计出现在表 S1，而 §6 局限写作"只在 ResNet18 上做过"，与表 S1 矛盾。此节给出
    三种主干的完整配对差与区间。数字现算自 results/e3_backbone_frontend/analysis.json。
    """
    d = json.loads((ROOT / 'results/e3_backbone_frontend/analysis.json').read_text())
    g = {(c['backbone'], c['group'], c['metric']): c for c in d['contrasts']
         if c['control'] == 'GN + yaw augmentation'}
    gname = {'independent-subject group': '独立受试者组', 'RIDI': 'RIDI', 'training-subject group': '训练受试者组'}
    order_b = ['ResNet18', 'Transformer', 'IMUNet']
    order_g = ['independent-subject group', 'RIDI', 'training-subject group']

    def cell(c):
        lo, hi = c['conditional_subject_bootstrap95']
        txt = f'{f3(c["difference"])} [{f3(lo)}, {f3(hi)}]'
        return f'**{txt}**' if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else txt

    n_ate_excl = sum(1 for (b, gr, m), c in g.items() if m == 'ate'
                     and not (c['conditional_subject_bootstrap95'][0] < 0 < c['conditional_subject_bootstrap95'][1]))
    n_rte_excl = sum(1 for (b, gr, m), c in g.items() if m == 'rte'
                     and not (c['conditional_subject_bootstrap95'][0] < 0 < c['conditional_subject_bootstrap95'][1]))
    lines = ['## S32. HN与偏航增强在三种主干上的配对比较', '',
             '正文§4.1的四前端受控比较在ResNet18上完成。Transformer与IMUNet在同一协议下各训练了HN臂与偏航增强臂',
             '（`config/e3_backbone_frontend.json`，每臂4个种子），因此可以在三种主干上给出同一组配对差。', '',
             '**表S55. 三种主干上HN与偏航增强的配对差（受试者平均，官方协议，4个种子；现算自`results/e3_backbone_frontend/analysis.json`）。**',
             '差值为HN减偏航增强，负值表示HN更低；区间为10,000次受试者整群bootstrap的条件95%区间（与正文§3.7同一口径），粗体为区间排除零。',
             'ResNet18各行与正文表3、表4同源。这九项ATE比较未做多重比较校正，属描述性结果。', '',
             '| 主干 | 测试组 | 受试者数 | HN ATE | 偏航增强 ATE | ATE差 [区间] | 改善人数 | HN RTE | 偏航增强 RTE | RTE差 [区间] |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in order_b:
        for gr in order_g:
            a, r = g[(b, gr, 'ate')], g[(b, gr, 'rte')]
            lines.append(f'| {b} | {gname[gr]} | {a["n_subjects"]} | {m3(a["treatment_mean"])} | {m3(a["control_mean"])} | '
                         f'{cell(a)} | {a["improved_subjects"]}/{a["n_subjects"]} | {m3(r["treatment_mean"])} | '
                         f'{m3(r["control_mean"])} | {cell(r)} |')
    lines += ['',
              f'九项ATE比较中{n_ate_excl}项区间排除零，方向相反：IMUNet在独立受试者组上HN更低，Transformer在RIDI上偏航增强更低；',
              f'其余含零。九项RTE比较中{n_rte_excl}项排除零，全部出现在独立受试者组，且全部是偏航增强更低。',
              '结果文件的`primary_sign_consistent_across_backbones`字段记为`false`，即主要终点的符号不跨主干一致。']
    return '\n'.join(lines) + '\n'


def s22_table() -> str:
    files = [('统一评测', 'tools/unified_eval_20260913.py'), ('统一评测汇总', 'tools/unified_analyze_20260913.py'),
             ('统一评测显著性', 'tools/unified_significance_20260913.py'), ('复增益上限', 'tools/complex_gain_oracle_20260913.py'),
             ('手机数据缓存', 'tools/build_imunet_cache_20260913.py'), ('逐窗推理与外挂HN', 'tools/hn_plugin_p0_20260912.py'),
             ('增强训练 + 外挂HN 汇总', 'tools/yawhn_summary_20260913.py'), ('增强训练 + 外挂HN 判定', 'tools/yawhn_verdict_20260913.py'),
             ('C8 测试时平均：跨角重复性', 'tools/tta_repeat_20260913.py'), ('C8 测试时平均：判定', 'tools/tta_verdict_20260913.py'),
             ('两方向帧平均：跨角重复性', 'tools/fa2_repeat_20260913.py'), ('两方向帧平均：判定', 'tools/fa2_verdict_20260913.py'),
             ('确认性检验：缓存', 'tools/build_confirm_caches_20260913.py'), ('确认性检验：评测', 'tools/confirm_eval_20260913.py'),
             ('确认性检验：判定', 'tools/confirm_verdict_20260913.py'), ('表9 时间尺度区间', 'tools/e4_intervals_20260913.py'),
             ('图（v2 新增）', 'tools/make_v2_figures_20260913.py'), ('图（v3 新增）', 'tools/make_v3_figures_20260915.py'),
             ('图（原有）', 'tools/mst_figures.py'), ('轨迹叠加', 'tools/e0_trajectory_overlay.py'),
             ('统一评测结果', 'results/unified_eval_20260913/summary.json'), ('统一评测显著性结果', 'results/unified_eval_20260913/significance.json'),
             ('增强训练 + 外挂HN 判定结果', 'results/unified_eval_20260913/yawhn_verdict.json'),
             ('C8 判定结果', 'results/tta_repeat_20260913/verdict.json'), ('两方向帧平均判定结果', 'results/fa2_repeat_20260913/verdict.json'),
             ('确认性检验判定结果', 'results/confirm_20260913/verdict.json'), ('表9 时间尺度区间结果', 'results/e4_error_timescale/intervals.json'),
             ('窗口一致性与耗时', 'results/sensors_v4/diagnostics.json'), ('受控前端汇总', 'results/sensors_v4/analysis.json'),
             ('40 次训练汇总', 'results/submission_frozen/eval_rq1_rq4_v3_registered_summary.json'),
             ('40 次训练矩阵', 'config/rq_experiment_matrix_v3.json'), ('受试者划分', 'config/splits_subject_disjoint_v2.json'),
             ('本补充材料生成脚本', 'tools/build_supplement_v3_20260915.py')]
    rows = ['| 用途 | 文件 | SHA-256 前 16 位 |', '|---|---|---|']
    for use, p in files:
        h = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16] if (ROOT / p).exists() else '（缺失）'
        rows.append(f'| {use} | `{p}` | `{h}` |')
    return '\n'.join(rows)


def main():
    text = SRC.read_text(encoding='utf-8')
    # ---------- 按章节切分
    parts = re.split(r'(?m)^(?=## S\d+\. )', text)
    body = {int(re.match(r'## S(\d+)\.', p).group(1)): p for p in parts if p.startswith('## S')}
    for k in DROP:
        body.pop(k)

    def rep(sec, old, new, count=1):
        assert body[sec].count(old) == count, (sec, old[:50], body[sec].count(old))
        body[sec] = body[sec].replace(old, new)

    # ---------- 逐节改动（仍用旧编号书写，稍后统一重排）
    rep(6, '构建脚本`tools/build_priority_revision_20260909.py`只读取已保存结果，不运行模型；它重新计算主得分、旋转极差、配对区间与时间诊断，\n'
           '并把主要汇总与早期分析对照。未取整的数值另行导出，`audit.json`记录来源哈希与完整性检查。数值模板字段在构建时解析，\n'
           '避免在不同稿本之间人工抄写。',
        f'表S1–S3、S5–S7、S13与S17的未取整数值保存在`{SS}/`，由`tools/build_priority_revision_20260909.py`（2026-09-09）从上表的结果文件计算，'
        '该脚本只读取已保存结果、不运行模型。v3的构建脚本`tools/build_supplement_v3_20260915.py`逐表核对表中每个数都能在所注出处中找到。')
    rep(6, '| 硬件、耗时与窗口旋转诊断 | `results/sensors_v4/diagnostics.json` |',
        f'| 硬件、耗时与窗口旋转诊断 | `results/sensors_v4/diagnostics.json` |\n| 补充表格的未取整数值 | `{SS}/` |')
    rep(6, 'v2新增部分的文件见S27', '其余脚本与结果文件见S27')
    rep(4, '../priority_revision_20260909/figures/figure_S01_elapsed_error.png', '../../figures/v3/figure_S01_elapsed_error.png')
    rep(10, '../priority_revision_20260909/figures/figure_02_trajectories.png', '../../figures/v3/figure_02_trajectories.png')
    rep(14, '**表S18. 三个训练子集的GlobalNorm归档常量。** 角速度与加速度的单位分别为rad/s与m/s²。100%一行即式(S1)、(S2)。',
        '**表S18. 训练集的GlobalNorm归档常量。** 角速度与加速度的单位分别为rad/s与m/s²，即式(S1)、(S2)。')
    body[14] = '\n'.join(l for l in body[14].split('\n') if not re.match(r'\| (25|50)% \|', l))
    rep(15, 'PreFilter只用于历史完整管线（S16–S20）', 'PreFilter只用于历史完整管线（S16、S18）')
    rep(16, '\n学习曲线另有25%与50%数据各4次训练（S19），TLIO复用其中12个完整模型检查点，不另行训练。', '')
    rep(16, 'S17–S20的数值为', 'S18的数值为')
    for old in ('出处：历史中文补充证据图3。', '出处：历史中文补充证据（原图4）。', '出处：历史中文补充证据（原图5）。', '出处：历史中文补充证据（原图6）。'):
        rep(18, old, FIG_SRC)
    RQ = '现算自`results/submission_frozen/eval_rq1_rq4_v3_registered_summary.json`）。**'
    rep(18, '（ATE，m，现算）。**', '（ATE，m，' + RQ, count=3)
    rep(18, '在ATE尺度上的交互（现算）。**', '在ATE尺度上的交互（' + RQ)
    rep(22, '**表S31. 统一口径60 s RTE。**', '**表S31. 统一口径60 s RTE（现算自`results/unified_eval_20260913/significance.json`）。**')
    rep(22, '**表S32. 文献协议ATE。**', '**表S32. 文献协议ATE（现算自`results/unified_eval_20260913/significance.json`）。**')
    rep(25, '**表S44. 手机测试数据概况（现算自`data/eval/imunet_owndata/`）。**', '**表S44. 手机测试数据概况（由`data/eval/imunet_owndata/`的缓存npz现算时长与路长）。**')
    rep(21, '**表S29. 14个模型×5个测试集的统一评测', '**表S29. 12个模型×5个测试集的统一评测')
    body[21] = '\n'.join(l for l in body[21].split('\n') if '整段HN' not in l)
    rep(26, '2. **尺度自校准的调参数据（`docs/38`第6节）。** 原定的16条"验证序列"中有14条在`subject_disjoint_v2`划分下属于训练集，\n'
            '   改为冻结模型训练时实际使用的验证集（8名验证受试者，14条序列，2.27 h）。当时尚无任何尺度自校准结果。\n3. **手机数据缓存。**',
        '2. **手机数据缓存。**')
    # S27：脚本与结果文件表整体重写
    s27 = body[27]
    head, _, _ = s27.partition('**表S45.')
    body[27] = ('## S27. 脚本与结果文件\n\n本文数字所用的主要脚本与结果文件如下（现算SHA-256前16位）。所有脚本只用项目自带的Python环境（`.venv`）运行；'
                '每个产物由哪个实验、哪个脚本生成，见`provenance/ARTIFACT_INDEX_20260915.csv`。\n\n'
                '**表S45. 脚本与结果文件（现算SHA-256前16位）。**\n\n' + s22_table() + '\n\n')
    # 出处改写：旧表号 → 新出处
    all_text = ''.join(body[k] for k in sorted(body))
    for t, src in SOURCES.items():
        sec = next(k for k in body if f'**表S{t}.' in body[k])
        s = body[sec]
        i = s.index(f'**表S{t}.')
        j = s.index('\n|', i)
        cap = s[i:j]
        new_cap, n = re.subn(r'出处：[^。]*。', src, cap, count=1)
        if n == 0:
            new_cap = cap.rstrip() + src
        body[sec] = s[:i] + new_cap + s[j:]

    # ---------- 新增 S26（旧编号 31，排在最后）
    body[31] = s26()
    # ---------- 新增 S27（旧编号 32，2026-09-16 补：三主干上 HN 与偏航增强的配对比较）
    body[32] = s32_backbone_aug()
    body[18] = body[18].rstrip('\n') + '\n\nHN与偏航增强在三种主干上的配对差（含受试者bootstrap区间）见S32。\n'

    # ---------- 重排编号
    old_secs = sorted(body)
    sec_map = {o: n for n, o in enumerate(old_secs, 1)}
    assembled = ''.join(body[k].rstrip('\n') + '\n\n' for k in old_secs)
    tab_old = [int(x) for x in re.findall(r'\*\*表S(\d+)\.', assembled)]
    tab_map = {o: n for n, o in enumerate(dict.fromkeys(tab_old), 1)}
    fig_old = [int(x) for x in re.findall(r'\*\*图S(\d+)\.', assembled)]
    fig_map = {o: n for n, o in enumerate(dict.fromkeys(fig_old), 1)}

    def sub_tab(m):
        o = int(m.group(1))
        return f'表S{tab_map[o]}' if o in tab_map else f'表S{o}（已删）'

    def sub_fig(m):
        o = int(m.group(1))
        return f'图S{fig_map[o]}' if o in fig_map else f'图S{o}（已删）'

    out = re.sub(r'表S(\d+)', sub_tab, assembled)
    out = re.sub(r'图S(\d+)', sub_fig, out)
    out = re.sub(r'(?m)^## S(\d+)\.', lambda m: f'## S{sec_map[int(m.group(1))]}.', out)
    # 章节引用：前一个字符是中文或中文标点（排除"表""图"），设备名 S10/S21、公式 (S4)、\tag{S1}、figure_S01 不受影响
    bad_refs = []

    def sub_sec(m):
        o = int(m.group(1))
        if o in sec_map:
            return f'S{sec_map[o]}'
        bad_refs.append(o)
        return f'S{o}'
    out = re.sub(r'(?<![表图])(?<=[^\x00-\x7F])S(\d+)(?![\d\w])', sub_sec, out)
    final = HEADER + '\n' + out
    OUT.write_text(final, encoding='utf-8')

    # ---------- 逐表核对出处
    def values(path_list):
        vals = []

        def walk(o):
            if isinstance(o, dict):
                for x in o.values(): walk(x)
            elif isinstance(o, list):
                for x in o: walk(x)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                vals.append(float(o))
        for p in path_list:
            P = ROOT / p
            files = [P] if P.is_file() else [f for f in P.rglob('*') if f.suffix in ('.json', '.csv')] if P.is_dir() else []
            for f in files:
                try:
                    if f.suffix == '.json':
                        walk(json.loads(f.read_text()))
                    else:
                        for row in csv.reader(open(f, errors='ignore')):
                            for c in row:
                                try: vals.append(float(c))
                                except ValueError: pass
                except Exception:
                    pass
        return sorted(vals)

    def hit(tok, V):
        x = float(tok); dec = len(tok.split('e')[0].split('.')[1])
        tol = abs(x) * 0.5 * 10 ** (-dec) * 1.0001 if 'e' in tok else 0.5 * 10 ** (-dec) + 1e-12
        for y in (x, -x, x / 100, -x / 100):
            t2 = tol if abs(y) == abs(x) else tol / 100
            i = bisect.bisect_left(V, y - t2)
            if i < len(V) and V[i] <= y + t2:
                return True
        return False

    report = {}
    for m in re.finditer(r'\*\*表S(\d+)\.(.*?)\n(?=\|)', final, flags=re.S):
        tno, cap = int(m.group(1)), m.group(2)
        paths = [p.rstrip('/') for p in re.findall(r'`((?:results|verification|models|config|data)/[^`]+)`', cap)]
        tbl = final[m.end():final.find('\n\n', m.end())]
        nums = [t.replace('−', '-') for row in tbl.split('\n')[2:] for t in re.findall(r'[−+\-]?\d+\.\d+(?:e[−+\-]?\d+)?', row)]
        if not nums:
            continue
        paths = [p.replace('.pt', '.json') if p.endswith('.pt') else p for p in paths]
        if not paths:
            report[f'表S{tno}'] = dict(n=len(nums), sources=[], found=None, note='表注无结果文件路径')
            continue
        V = values(paths)
        miss = [n for n in nums if not hit(n, V)]
        report[f'表S{tno}'] = dict(n=len(nums), sources=paths, found=len(nums) - len(miss), missing=miss[:12])
    json.dump(dict(section_map=sec_map, table_map=tab_map, figure_map=fig_map, dangling_section_refs=bad_refs, tables=report),
              open(REPORT, 'w'), ensure_ascii=False, indent=1)
    print('章节对应（旧→新）：', sec_map)
    print('表对应（旧→新）：', tab_map)
    print('图对应（旧→新）：', fig_map)
    print('指向已删章节的引用：', bad_refs or '无')
    incomplete = {k: v for k, v in report.items() if v['found'] is None or v['found'] < v['n']}
    print('出处核对：共', len(report), '张表；未能完全核对的：', json.dumps(incomplete, ensure_ascii=False)[:1500] if incomplete else '无')
    print('[写出]', OUT.relative_to(ROOT), len(final), '字符')


if __name__ == '__main__':
    main()
