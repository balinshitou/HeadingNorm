"""图内用词与正文统一（2026-10-01，作者意见："图内用词与正文不一致，这需要进行修改"）。

只改图中文字（坐标轴、标题、图例、刻度标签、注释位置），数据与绘图逻辑不变。
做法：读入原作图脚本的源码，按下表逐条替换文字（每条断言出现次数），再在内存中执行；
原脚本文件不改动，因此补充材料表S30中列出的脚本哈希保持有效。输出写到 figures/v15/。
用词依据 paper/draft_zh_20260927/polish/20_英文术语表.md。

  图1  tools/make_v4_figures_20260919.py:problem_overview      "8 directions" → "8 headings"
  图2  tools/mst_figures.py:fig02                               centred → centered
  图4  tools/make_hn_pipeline_figures_20260929.py:fig4          "(Fig. 3)" → "(Figure 3)"
  图5  tools/make_v4_figures_20260919.py:repeatability          RoNIN-unseen、subject-mean ATE、initial reference heading、RoNIN ResNet (released)
  图6  tools/make_plugin_provenance_figure_20260925.py          去掉 discovery / never evaluated；confirmation sets
  图S1 tools/make_v2_figures_20260913.py:fig_unified_paired     数据集名同表1；HN − YawAug / GN only；RoNIN ResNet: Plug-in HN − Unmodified；CI
  图S2 tools/mst_figures.py:fig10                               HN w/o sign / HN (with third-moment sign)；canonical heading；注释不再压住柱标
  图S3 tools/make_supp_figures_S1_S3_20260926.py:plot           GN + YawAug / GN + HN w/o sign / GN + HN；Ground truth

2026-10-04 图1（作者意见：子图(b)比另两个小；删去图内总标题，由题注说明）：三栏共用同一坐标范围（真值与(b)(c)全部轨迹的包络），
  等比例坐标下三栏框因此等大；删去 fig.text 总标题。--fig1-only 只重画图1。
2026-10-04 图S2（作者同意）：删去图内标题（与题注、纵轴标签重复）。--figS2-only 只重画图S2。
2026-10-04 正文图（作者同意；只改作图代码的文字与字体，数据与绘图逻辑不变）：
  图2  删去图内标题 'The axis is a line; the sign is a choice'
  图3  'Eq. 4/5' → 'Equation (4)/(5)'；图下两行说明移入题注；改由本脚本出图（原为 figures/v13）
  图4  图下两行说明移入题注；画布 y 下限 0→6、图高同比例缩小，裁去空出的底部空白
  图5  栏标题加 (a)(b)(c)，第三栏标题改为 '(c) RoNIN-unseen'（与横轴标签不再重复）；第三栏横轴标签缩为
       'cross-angle range / ATE at ψ=0 (%)'（原标签宽于坐标框，tight 裁切截掉右端；"per-sequence" 由题注说明）
  图7  tools/make_fig8_shape_example_20261001.py 原只设字号、用 matplotlib 默认字体 DejaVu Sans；改为与其余各图相同的
       Helvetica Neue 字体列表；在 rc_context 内先恢复 matplotlib 默认参数再执行，其余样式与原脚本单独运行时相同；
       读缓存 results/fig8_shape_example_20261001/，不重跑模型，脚本内仍逐项断言数字与存档一致；输出改到 figures/v15

2026-10-04 一键复现（reproduce/make_figures.py）：环境变量 HN_FIG_OUT 指定输出目录（默认 figures/v15），复现时写到
  reproduce/output/figures，不覆盖存档的图；没有 RoNIN 评测缓存时，图2 读 tools/archive_small_inputs_20261004.py
  导出的同一个窗口（results/figure_inputs_20261004/fig02_window.npz），输出与读缓存时逐字节相同。
  图7 同理：没有 TLIO-confirm 评测缓存时，序列清单只含导出的那一条序列（fig07_tlio_confirm_sequence.npz，时间戳与真值），
  选序列规则照常执行；没有 RoNIN 官方源码时，以 pkg_common 代替 hn_plugin_p0（读缓存时只用到它的 .P）。
  图7 的 provenance.json 原写入 results/fig8_shape_example_20261001/（存档目录），改为写在图旁
  （<输出目录>/fig8_shape_example_provenance.json），作图不再改动存档。

用法：.venv/bin/python tools/make_figures_terms_20261001.py [--only=fig1,fig2,...,figS3 | --fig1-only | --figS2-only]
  不带参数时重画全部图。
"""
from __future__ import annotations
import contextlib, io, os, shutil, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
OUT = Path(os.environ['HN_FIG_OUT']).resolve() if os.environ.get('HN_FIG_OUT') else ROOT / 'figures/v15'
TO = f'OUT = Path({str(OUT)!r})'      # 各原脚本的输出目录行替换成这一行
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TOOLS))
ONLY = {k for a in sys.argv[1:] if a.startswith('--only=') for k in a[len('--only='):].split(',') if k}
ONLY |= {'fig1'} if '--fig1-only' in sys.argv else set()
ONLY |= {'figS2'} if '--figS2-only' in sys.argv else set()
KNOWN = {'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'figS1', 'figS2', 'figS3'}
assert ONLY <= KNOWN, ONLY - KNOWN


def want(k: str) -> bool:
    return not ONLY or k in ONLY


def load(fname: str, reps: list[tuple[str, str, int]], modname: str | None = None, run=True):
    """读入 tools/fname，按 reps 做断言替换后作为模块执行（__file__ 仍指向原文件，ROOT 不变）。"""
    path = TOOLS / fname
    src = path.read_text(encoding='utf-8')
    for old, new, n in reps:
        c = src.count(old)
        assert c == n, (fname, old, c, n)
        src = src.replace(old, new)
    mod = types.ModuleType(modname or path.stem)
    mod.__file__ = str(path)
    if modname:
        sys.modules[modname] = mod
    if run:
        exec(compile(src, str(path), 'exec'), mod.__dict__)
    return mod


# ---------------------------------------------------------------- mst_figures（图2、图S2；图5的图例表也在这里）
M = load('mst_figures.py', [
    ("ax.set_xlabel('centred $a_x$  (m s$^{-2}$)')", "ax.set_xlabel('centered $a_x$  (m s$^{-2}$)')", 1),
    ("ax.set_ylabel('centred $a_y$  (m s$^{-2}$)')", "ax.set_ylabel('centered $a_y$  (m s$^{-2}$)')", 1),
    ("linewidth=.5, zorder=2, label='without the sign rule')", "linewidth=.5, zorder=2, label='HN w/o sign')", 1),
    ("label='with the third-moment sign rule')", "label='HN (with third-moment sign)')", 1),
    ("ax.set_ylabel('windows whose frame does not\\nfollow the rotation (%)')",
     "ax.set_ylabel('windows whose canonical heading\\ndoes not follow the rotation (%)')", 1),
    ("ax.set_xticks(x); ax.set_xticklabels([f'{int(d)}' for d in deg])\n    ax.set_xlabel('yaw applied to the test input (deg)')",
     "ax.set_xticks(x); ax.set_xticklabels([f'{int(d)}' for d in deg])\n    ax.set_xlabel('yaw $\\\\psi$ applied to the input (deg)')", 1),
    ("xy=(5.2, .8),\n                xytext=(5.4, 34), fontsize=7.6, color=COLOUR['GN + HeadingNorm'],\n                ha='left', va='center',",
     "xy=(7.2, .8),\n                xytext=(7.0, 100), fontsize=7.6, color=COLOUR['GN + HeadingNorm'],\n                ha='center', va='center',", 1),
    ("ax.set_title('The axis is a line; the sign is a choice', pad=8)",
     "pass  # 图内标题已删（2026-10-04），由题注说明", 1),
    ("ax.set_title(f'Frame equivariance on {n} fixed windows', pad=8)",
     "pass  # 图内标题已删（2026-10-04），由题注说明", 1),
], modname='mst_figures')
M.LABEL.update({  # 与 tools/make_fig4_repeatability_20260929.py 相同的图例名（正文表3的写法）
    'GN only': 'GN only', 'GN + yaw augmentation': 'GN + YawAug',
    'GN + mixed-sensor PCA (adapted)': 'GN + PCA frame',
    'GN + HeadingNorm without sign rule': 'GN + HN w/o sign', 'GN + HeadingNorm': 'GN + HN',
})
if not any((ROOT / 'data/eval/benchmark').glob('ronin__*.npz')):     # 只有存档：图2 读导出的同一个窗口
    def _archived_window(index_from_middle=0):
        import numpy as np
        z = np.load(ROOT / 'results/figure_inputs_20261004/fig02_window.npz')
        w = z['window']
        return None, w - w.mean(axis=0), str(z['seq']), int(z['start'])
    M.real_window = _archived_window
    print('[图2] 无评测缓存，读 results/figure_inputs_20261004/fig02_window.npz')
for e in M.REGISTRY:
    if (e['id'] == 'fig02' and want('fig2')) or (e['id'] == 'fig10' and want('figS2')):
        M.build(e, OUT, ['png', 'pdf'])

# ---------------------------------------------------------------- 图1、图5
V4 = load('make_v4_figures_20260919.py', [
    ("OUT = ROOT / 'figures/v4'", TO, 1),
    ("f'8 directions: ATE", "f'8 headings: ATE", 1),
    ("f'8 directions coincide: ATE", "f'8 headings coincide: ATE", 1),
    ("'RoNIN independent-subject group'", "'RoNIN-unseen'", 1),
    ("ax.set_xlabel('yaw applied to the test input (deg)')", "ax.set_xlabel('initial reference heading $\\\\psi$ (deg)')", 1),
    ("axes[0].set_ylabel('subject-macro ATE (m)')", "axes[0].set_ylabel('subject-mean ATE (m)')", 1),
    ("label=('Original RoNIN ResNet' if fam == OFFICIAL", "label=('RoNIN ResNet (released)' if fam == OFFICIAL", 1),
    ("ax.set_xlabel('per-recording ATE range / ATE at $\\\\psi=0$ (%)')",
     "ax.set_xlabel('cross-angle range / ATE at $\\\\psi=0$ (%)')", 1),  # 10-04 缩短：原标签宽于坐标框，tight 裁切会截掉右端
    ("ax.set_title('What a single recording sees (independent group)', pad=6)",
     "ax.set_title('(c) RoNIN-unseen', pad=6)", 1),
    ("        ax.set_title(title, pad=6)\n    axes[0].set_ylabel(",
     "        ax.set_title(('(a) ' if ds == 'ronin' else '(b) ') + title, pad=6)\n    axes[0].set_ylabel(", 1),
    ("    fig.text(.5, .99, 'Horizontal coordinate rotation changes representation, not motion.',\n"
     "             ha='center', va='top', fontsize=10)\n"
     "    fig.subplots_adjust(top=.86, bottom=.2, wspace=.25)\n",
     "    xy = np.array([[float(r[3]), float(r[4])] for r in rows])\n"
     "    lo, hi = xy.min(0), xy.max(0); pad = .04 * (hi - lo)\n"
     "    for ax in axes:\n"
     "        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])\n"
     "    fig.subplots_adjust(top=.93, bottom=.2, wspace=.25)\n", 1),
])
if want('fig1'):
    V4.problem_overview()
med = V4.repeatability() if want('fig5') else {}

# ---------------------------------------------------------------- 图3、图4
# 该脚本载入时设置全局 rcParams，影响其后的图6、S1、S3；只要画其中任一图就载入，保证与全量运行相同
P = load('make_hn_pipeline_figures_20260929.py', [
    ("OUT = ROOT / 'figures' / 'v13'", TO, 1),
    ("there is no $\\\\psi$ (Fig. 3).", "there is no $\\\\psi$ (Figure 3).", 1),
    ("'principal axis (Eq. 4) + third-moment sign (Eq. 5)'",
     "'principal axis, Equation (4); third-moment sign, Equation (5)'", 1),
    ("    ax.text(50, -3.5, ", "    if False: ax.text(50, -3.5, ", 1),          # 图3 图下说明移入题注
    ("    ax.text(70, 3.0, 'Orange dashed", "    if False: ax.text(70, 3.0, 'Orange dashed", 1),  # 图4 同上
    ("    fig, ax = canvas(8.2, 4.6, 140, 74)",   # 图4 说明移走后裁去底部空白：y 下限 0→6，图高按同比例缩，内容比例不变
     "    fig, ax = canvas(8.2, 4.6 * 68 / 74, 140, 74); ax.set_ylim(6, 74)", 1),
]) if any(want(k) for k in ('fig3', 'fig4', 'fig6', 'figS1', 'figS3')) else None
if want('fig3'):
    P.fig3()
if want('fig4'):
    P.fig4()

# ---------------------------------------------------------------- 图6（脚本在载入时即作图）
if want('fig6'):
  with contextlib.redirect_stdout(io.StringIO()):
    load('make_plugin_provenance_figure_20260925.py', [
        ("OUT = ROOT / 'figures/v10'", TO, 1),
        ("'RoNIN ResNet (released by authors)'", "'RoNIN ResNet (released by original authors)'", 1),
        ("('ronin_unseen', 'RoNIN-unseen\\n(discovery)')", "('ronin_unseen', 'RoNIN-unseen')", 1),
        ("'TLIO-confirm\\n(318, never evaluated)'", "'TLIO-confirm\\n(318)'", 1),
        ("'Phone-confirm\\n(87, never evaluated)'", "'Phone-confirm\\n(87)'", 1),
        ("label='shaded: confirmation splits (never evaluated before)'", "label='shaded: confirmation sets'", 1),
    ])

# ---------------------------------------------------------------- 图S1
V2 = load('make_v2_figures_20260913.py', [
    ("OUT = ROOT / 'figures' / 'v2'", TO, 1),
    ("DATASETS = [('ronin_seen', 'RoNIN training subjects'), ('ronin_unseen', 'RoNIN independent subjects'),\n"
     "            ('ridi', 'RIDI'), ('tlio', 'TLIO'), ('imunet', 'IMUNet phones')]",
     "DATASETS = [('ronin_seen', 'RoNIN-seen'), ('ronin_unseen', 'RoNIN-unseen'),\n"
     "            ('ridi', 'RIDI'), ('tlio', 'TLIO-test'), ('imunet', 'Phone-test')]", 1),
    ("'HN −\\nyaw augmentation'", "'HN −\\nYawAug'", 1),
    ("'HN −\\nstandardisation only'", "'HN −\\nGN only'", 1),
    ("'Official ResNet:\\nbolt-on HN − as is'", "'RoNIN ResNet:\\nPlug-in HN − Unmodified'", 1),
    ("label='interval excludes 0'", "label='CI excludes 0'", 1),
    ("label='interval includes 0')]\n    fig.tight_layout(w_pad=1.2)", "label='CI includes 0')]\n    fig.tight_layout(w_pad=1.2)", 1),
])
if want('figS1'):
    V2.fig_unified_paired()

# ---------------------------------------------------------------- 图S3（只读 v11 的源数据 CSV）
if want('figS3'):
  for n in ('figS01_elapsed_error_source_data.csv', 'figS03_trajectories_source_data.csv'):
    shutil.copy2(ROOT / 'figures/v11' / n, OUT / n)
S3 = load('make_supp_figures_S1_S3_20260926.py', [
    ("OUT = ROOT / 'figures/v11'", TO, 1),
    ("ARMS = [('yaw', 'Yaw augmentation'), ('ns', 'HN without sign rule'), ('hn', 'HN')]",
     "ARMS = [('yaw', 'GN + YawAug'), ('ns', 'GN + HN w/o sign'), ('hn', 'GN + HN')]", 1),
    ("label='Reference')", "label='Ground truth')", 1),
])
if want('figS3'):
    S3.plot()

# ---------------------------------------------------------------- 图7（字体与其余各图统一）
if want('fig7'):
    import matplotlib.pyplot as plt
    with plt.rc_context():
        plt.rcdefaults()  # 与原脚本单独运行时相同的起点，不受前面各脚本全局 rcParams 影响
        sys.path[:0] = [str(ROOT / 'src')]
        try:
            import hn_plugin_p0_20260912  # noqa: F401
        except ModuleNotFoundError:      # 只有存档：无 RoNIN 官方源码；读缓存的轨迹时只用到 H.P
            import pkg_common
            sys.modules['hn_plugin_p0_20260912'] = types.SimpleNamespace(P=pkg_common)
            print('[图7] 无 RoNIN 官方源码，读缓存的轨迹')
        fig7_archive = not any((ROOT / 'data/eval/confirm_tlio').glob('*.npz'))
        F8 = load('make_fig8_shape_example_20261001.py', [
            ("OUT = ROOT / 'figures/v14'", TO, 1),
            ("plt.rcParams.update({'font.size': 8.5,",
             "plt.rcParams.update({'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],"
             " 'font.size': 8.5,", 1),  # 字族列表可逐字回退：Helvetica Neue 无 '→'，该字取 DejaVu Sans
            ("json.dump(info, open(CACHE / 'provenance.json', 'w')", "json.dump(info, open(OUT / f'{STEM}_provenance.json', 'w')", 1),
        ])
        if fig7_archive:                 # 只有存档：序列清单只含导出的那一条序列
            import numpy as np
            A = ROOT / 'results/figure_inputs_20261004/fig07_tlio_confirm_sequence.npz'
            F8.CE.files = lambda ds: [A]
            F8.CE.read = lambda path: {k: np.asarray(v) for k, v in np.load(path).items()}
            print('[图7] 无 TLIO-confirm 评测缓存，读 results/figure_inputs_20261004/fig07_tlio_confirm_sequence.npz')
        F8.main()
for k, v in med.items():
    print(f'  图5 中位极差占比 {k}: {v:.1f}%')
print(f'完成：{OUT}')
