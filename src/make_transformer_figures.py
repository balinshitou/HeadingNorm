"""为 Transformer 教学文档生成原理图/流程图（纯 matplotlib）。

输出目录：figures/architecture/。每个函数对应文档里的一张图。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os
import argparse

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "architecture")
os.makedirs(OUT, exist_ok=True)

# 配色
C_IN = "#E8F0FE"     # 输入/张量
C_OP = "#FFF3CD"     # 运算/变换
C_ATTN = "#D8E8D4"   # 注意力
C_NORM = "#F5E6FF"   # 归一化
C_OUT = "#FDE2D3"    # 输出
C_EDGE = "#333333"


def box(ax, x, y, w, h, text, fc=C_OP, fs=11, ec=C_EDGE, lw=1.4, zorder=3, style="round,pad=0.02,rounding_size=0.06"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=zorder + 1, linespacing=1.4)
    return (x, y, w, h)


def arrow(ax, p0, p1, lw=1.6, color=C_EDGE, style="-|>", connectionstyle=None, mutation_scale=14, zorder=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mutation_scale, lw=lw, color=color,
                         connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(a)


def down(ax, box_top, dy=0.35):
    x, y, w, h = box_top
    arrow(ax, (x + w / 2, y), (x + w / 2, y - dy))


def new_ax(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# ------------------------------------------------------------------ 图1 自注意力
def fig_self_attention():
    fig, ax = new_ax((9.5, 10), (0, 9.5), (0, 12.5))

    box(ax, 3.2, 11.2, 3.2, 0.9, "输入序列 X\n[n 个位置 × d 维]", fc=C_IN, fs=11)

    y0 = 9.7
    xs = [0.4, 3.55, 6.7]
    labels = ["Q = X·W_Q\n(查询 Query)", "K = X·W_K\n(键 Key)", "V = X·W_V\n(值 Value)"]
    for x, lab in zip(xs, labels):
        arrow(ax, (5, 11.2), (x + 1.35, y0 + 0.9))
        box(ax, x, y0, 2.7, 0.9, lab, fc=C_OP, fs=10.5)

    # Q,K -> QK^T
    arrow(ax, (0.4 + 1.35, y0), (2.9, 8.15), connectionstyle="arc3,rad=0.15")
    arrow(ax, (3.55 + 1.35, y0), (3.6, 8.15))
    box(ax, 2.0, 7.25, 3.2, 0.9, r"score $= Q K^\top$" + "\n[n × n]（相关性打分）", fc=C_ATTN, fs=10.5)

    down(ax, (2.0, 7.25, 3.2, 0.9))
    box(ax, 2.0, 5.75, 3.2, 0.75, "÷ √d_k　（数值稳定）", fc=C_ATTN, fs=10.5)

    down(ax, (2.0, 5.75, 3.2, 0.75))
    box(ax, 2.0, 4.35, 3.2, 0.9, "softmax（按行）\n→ 注意力权重 [n × n]", fc=C_ATTN, fs=10.5)

    # weights + V -> weighted sum
    arrow(ax, (3.6, 4.35), (3.6, 3.1))
    arrow(ax, (6.7 + 1.35, y0), (5.3, 3.55), connectionstyle="arc3,rad=-0.2")
    box(ax, 2.0, 2.2, 3.2, 0.9, "加权求和：权重 × V\n= Attention(Q,K,V)", fc=C_OUT, fs=10.5)

    down(ax, (2.0, 2.2, 3.2, 0.9))
    box(ax, 2.4, 0.7, 2.4, 0.9, "输出 [n × d]\n每个位置的新表示", fc=C_IN, fs=10.5)

    ax.text(7.3, 6.6, r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\dfrac{QK^{\top}}{\sqrt{d_k}}\right)V$",
            fontsize=12, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888"))
    ax.set_title("自注意力（Self-Attention）计算流程", fontsize=14, pad=10)
    save(fig, "fig01_self_attention.png")


# ------------------------------------------------------------------ 图2 多头注意力
def fig_multihead():
    fig, ax = new_ax((11.2, 6.8), (0, 12), (0, 7.6))
    box(ax, 4.3, 6.3, 3.2, 0.9, "输入 X  [n × d]", fc=C_IN, fs=11)

    n_head = 4
    xs = np.linspace(0.6, 8.9, n_head)
    for i, x in enumerate(xs):
        arrow(ax, (5.9, 6.3), (x + 0.85, 5.25))
        box(ax, x, 4.35, 1.7, 0.9, f"头 {i+1}\nAttention\n(d/h 维子空间)", fc=C_ATTN, fs=8.8)
        arrow(ax, (x + 0.85, 4.35), (5.9, 3.15), connectionstyle="arc3,rad=0.0")

    box(ax, 3.9, 2.35, 4.0, 0.8, "拼接（Concat）四个头的输出", fc=C_OP, fs=10.5)
    down(ax, (3.9, 2.35, 4.0, 0.8))
    box(ax, 4.1, 1.05, 3.6, 0.8, "线性融合 × W_O  →  输出 [n × d]", fc=C_OUT, fs=10.5)

    ax.text(11.7, 6.75, "h 个头并行，各自\n独立的 W_Q, W_K, W_V\n(HN-Transformer: h=8, d/h=32)", fontsize=9.5, ha="right", va="center", color="#555555")
    ax.set_title("多头注意力（Multi-Head Attention）：把一次注意力拆成 h 个并行子空间", fontsize=13, pad=10)
    save(fig, "fig02_multihead.png")


# ------------------------------------------------------------------ 图3 Pre-LN Transformer Encoder Block
def fig_encoder_block():
    fig, ax = new_ax((7.4, 11.5), (0, 8.5), (0, 14.5))

    box(ax, 3.1, 13.2, 2.3, 0.85, "x（输入）", fc=C_IN, fs=11)
    down(ax, (3.1, 13.2, 2.3, 0.85), dy=0.4)

    box(ax, 3.1, 11.9, 2.3, 0.8, "LayerNorm", fc=C_NORM, fs=10.5)
    down(ax, (3.1, 11.9, 2.3, 0.8), dy=0.4)

    box(ax, 2.6, 10.4, 3.3, 0.9, "多头自注意力\nMulti-Head Attention", fc=C_ATTN, fs=10.5)
    down(ax, (2.6, 10.4, 3.3, 0.9), dy=0.4)

    # residual add 1
    arrow(ax, (3.1 + 0.1, 13.2), (0.9, 13.2 - 0.05), connectionstyle="arc3,rad=0.0")
    ax.add_patch(FancyArrowPatch((1.0, 13.6), (1.0, 9.85), connectionstyle="arc3,rad=0", arrowstyle="-", lw=1.6, color=C_EDGE))
    arrow(ax, (1.0, 9.85), (2.6 + 0.05, 9.65), connectionstyle="arc3,rad=-0.25")
    box(ax, 2.9, 9.15, 2.7, 0.75, "+  （残差相加）", fc="white", fs=10.5, ec="#888888")
    ax.text(0.55, 11.5, "跳线\n(x 原样传递)", fontsize=8.6, ha="center", color="#555555", rotation=90)

    down(ax, (2.9, 9.15, 2.7, 0.75), dy=0.35)
    box(ax, 3.1, 7.85, 2.3, 0.75, "LayerNorm", fc=C_NORM, fs=10.5)
    down(ax, (3.1, 7.85, 2.3, 0.75), dy=0.35)
    box(ax, 2.5, 6.55, 3.5, 0.9, "前馈网络 FFN\nLinear→GELU→Linear\n(隐藏维 = 4d)", fc=C_OP, fs=10)
    down(ax, (2.5, 6.55, 3.5, 0.9), dy=0.35)

    ax.add_patch(FancyArrowPatch((7.6, 9.5), (7.6, 5.9), connectionstyle="arc3,rad=0", arrowstyle="-", lw=1.6, color=C_EDGE))
    arrow(ax, (2.9+2.7, 9.5), (7.55, 9.5), connectionstyle="arc3,rad=0", style="-")
    arrow(ax, (7.6, 5.9), (2.5+3.5-0.05, 5.65), connectionstyle="arc3,rad=0.25")
    box(ax, 2.9, 5.15, 2.7, 0.75, "+  （残差相加）", fc="white", fs=10.5, ec="#888888")
    ax.text(7.95, 7.7, "跳线\n(x' 原样传递)", fontsize=8.6, ha="center", color="#555555", rotation=90)

    down(ax, (2.9, 5.15, 2.7, 0.75), dy=0.35)
    box(ax, 3.1, 3.85, 2.3, 0.85, "block 输出", fc=C_IN, fs=11)

    ax.text(1.0, 2.9, "公式：\n" r"$x'=x+\mathrm{Attn}(\mathrm{LN}(x))$" "\n"
            r"$y=x'+\mathrm{FFN}(\mathrm{LN}(x'))$",
            fontsize=10.5, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA", ec="#888888"))
    ax.text(5.6, 2.9, "这是 Pre-LN：LayerNorm\n在子层运算之前、\n残差分支内部。\n本项目全部 Transformer\n(HN-Transformer等) 均用此结构\n(norm_first=True)。",
            fontsize=9.5, ha="left", va="top", color="#333333")

    ax.set_title("一个完整的 Pre-LN Transformer Encoder Block", fontsize=13, pad=10)
    save(fig, "fig03_encoder_block.png")


# ------------------------------------------------------------------ 图4 三种范式对比
def fig_paradigms():
    fig, ax = new_ax((11.5, 5.2), (0, 12), (0, 6))
    cols = [
        (0.4, "Encoder-only\n(如 BERT)", ["Token1", "Token2", "Token3"], "双向：\n每个位置都能\n看到全部输入", "分类 / 理解 /\n回归（本项目属此类）"),
        (4.4, "Decoder-only\n(如 GPT)", ["Token1", "Token2", "Token3"], "因果：\n只能看到\n自己左边", "文本生成"),
        (8.4, "Encoder-Decoder\n(原始 Transformer)", ["Enc→Dec", "交叉注意力", "逐词生成"], "编码器读全文\n解码器逐步生成", "翻译 / 摘要"),
    ]
    for x0, title, rows, note, task in cols:
        box(ax, x0, 5.0, 3.2, 0.7, title, fc=C_OUT, fs=10.5)
        for i, r in enumerate(rows):
            box(ax, x0 + 0.2, 3.5 - i * 0.85, 2.8, 0.7, r, fc=C_ATTN if i == 0 else C_OP, fs=9.5)
        ax.text(x0 + 1.6, 1.15, note, fontsize=9, ha="center", va="top", color="#444444")
        box(ax, x0, 0.15, 3.2, 0.6, task, fc=C_IN, fs=9.5)
    ax.set_title("三种 Transformer 范式：本项目的任务属于 Encoder-only", fontsize=13, pad=10)
    save(fig, "fig04_paradigms.png")


# ------------------------------------------------------------------ 图5 Patch Embedding 类比
def fig_patch_analogy():
    fig, ax = new_ax((10.5, 7.2), (0, 11), (0, 8.8))
    rows = [
        (6.7, "文本 (BERT)", "一句话", ["The", "cat", "sat", "...", "mat"], "每个词/子词"),
        (4.3, "图像 (ViT)", "一张图片", ["■", "■", "■", "...", "■"], "16×16 像素块"),
        (1.9, "IMU（本项目）", "1 秒信号 [6,200]", ["8点", "8点", "8点", "...", "8点"], "40 ms / 8 采样点"),
    ]
    for y, name, src, toks, unit in rows:
        box(ax, 0.2, y, 1.9, 0.9, name, fc=C_IN, fs=9.5)
        box(ax, 2.4, y, 1.9, 0.9, src, fc=C_OP, fs=9.5)
        arrow(ax, (4.3, y + 0.45), (4.85, y + 0.45))
        tx = 4.9
        for t in toks:
            box(ax, tx, y + 0.15, 0.72, 0.6, t, fc=C_ATTN, fs=8.5)
            tx += 0.85
        ax.text(tx + 0.25, y + 0.45, f"→ {unit}", fontsize=9, ha="left", va="center", color="#444444")
    ax.text(5.3, 8.4, "三个领域共享同一套 Transformer 计算，只是\u201c什么算一个词\u201d这一步的具体实现不同", fontsize=10.5, ha="center", color="#333333")
    box(ax, 4.9, 0.4, 5.6, 0.75, "序列前拼一个可学习 [CLS] token → 过 Transformer → 取 CLS 输出做分类/回归", fc=C_OUT, fs=9.3)
    save(fig, "fig05_patch_analogy.png")


# ------------------------------------------------------------------ 图6 位置编码可视化
def fig_posenc():
    # d、maxlen 直接取 A6 的真实用量：26 个 token（25 个 patch + 1 个 CLS），
    # d 用 32（A6 实际 d=256，这里降到 32 只是为了让热力图的条纹肉眼可辨）。
    d, maxlen = 32, 26
    pos = np.arange(maxlen)[:, None]
    i = np.arange(d)[None, :]
    div = np.power(10000.0, (2 * (i // 2)) / d)
    pe = np.zeros((maxlen, d))
    pe[:, 0::2] = np.sin(pos / div[:, 0::2])
    pe[:, 1::2] = np.cos(pos / div[:, 1::2])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu_r", origin="lower")
    axes[0].set_xlabel("序列位置 pos（0 = CLS，1~25 = 25 个 patch）")
    axes[0].set_ylabel("编码维度 i")
    axes[0].set_title("正弦位置编码矩阵 PE[pos, i]（颜色=数值）")
    axes[0].set_xticks(range(0, maxlen, 2))
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for dim in [0, 2, 8, 24]:
        axes[1].plot(pos[:, 0], pe[:, dim], marker="o", ms=3, label=f"维度 {dim}")
    axes[1].set_xlabel("序列位置 pos")
    axes[1].set_ylabel("编码值")
    axes[1].set_title("不同维度对应不同频率的正弦波")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("位置编码（Positional Encoding）：$PE_{(pos,2i)}=\\sin(pos/10000^{2i/d})$\n"
                 "（此处按HN-Transformer实际序列长度26绘制，d缩小到32便于观察条纹）", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig06_posenc.png")


# ------------------------------------------------------------------ 图7 A6 完整流水线
def fig_a6_pipeline():
    fig, ax = new_ax((9.6, 9.4), (0, 10.2), (5.2, 18.3))

    steps = [
        ("输入 IMU\n陀螺xyz + 加计xyz, 200 Hz", "[B, 6, 200]", C_IN, 0.85),
        ("① HeadingNorm\n绕重力轴转到水平加速度主轴，记下角 θ", "[B, 6, 200] + θ", C_NORM, 0.95),
        ("② GlobalNorm\n(x－μ)/σ，训练集常数，x/y共尺度", "[B, 6, 200]", C_NORM, 0.95),
        ("③ PreFilter\n原始+低通+带通+高通，6→24通道", "[B, 24, 200]", C_OP, 0.95),
        ("④ Patch Embed\nConv1d(24→256, kernel=8, stride=8)", "[B, 25, 256]", C_OP, 0.95),
        ("④' 拼 CLS + 正弦位置编码", "[B, 26, 256]", C_OP, 0.85),
        ("⑤ Transformer Encoder × 4\nPre-LN, 8头, FFN=1024, GELU, dropout 0.1", "[B, 26, 256]", C_ATTN, 1.05),
        ("⑥ 取 CLS → LN → Linear(256,256)\n→ GELU → Dropout(0.2) → Linear(256,2)", "[B, 2]（归一化坐标系）", C_OUT, 1.05),
        ("⑦ 反旋转 _unrotate(θ)\n转回世界坐标系", "[B, 2] m/s", C_OUT, 0.85),
        ("速度序列 20 Hz → 按 Δt 累加 → 轨迹", "轨迹 (x,y) 序列", C_IN, 0.85),
    ]
    y = 17.6
    w, x0 = 6.6, 1.3
    for text, shape, fc, h in steps:
        box(ax, x0, y - h, w, h, text, fc=fc, fs=9.6)
        ax.text(x0 + w + 0.25, y - h / 2, shape, fontsize=9, va="center", ha="left", color="#555555", style="italic")
        if y != 17.6:
            pass
        y_next = y - h
        y = y_next
        if text != steps[-1][0]:
            arrow(ax, (x0 + w / 2, y), (x0 + w / 2, y - 0.28))
            y -= 0.28

    ax.set_title("HN-Transformer：从原始IMU到轨迹的完整流水线（每步标注张量形状）", fontsize=13, pad=10)
    save(fig, "fig07_a6_pipeline.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", help="图输出目录；相对路径按项目根目录解释")
    args = parser.parse_args()
    if args.out_dir:
        candidate = os.path.abspath(
            args.out_dir if os.path.isabs(args.out_dir)
            else os.path.join(os.path.dirname(__file__), "..", args.out_dir))
        OUT = candidate
        os.makedirs(OUT, exist_ok=True)
    fig_self_attention()
    fig_multihead()
    fig_encoder_block()
    fig_paradigms()
    fig_patch_analogy()
    fig_posenc()
    fig_a6_pipeline()
