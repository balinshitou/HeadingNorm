"""候选架构库：CNN / TCN / LSTM / Transformer / 混合，以及模型前后的滤波模块。

输入 [B, 6, 200]（世界系 gyro3 + acce3，1 s @200Hz），输出 [B, 2]（水平速度 m/s）。

**模型前的滤波**（用户建议）：`PreFilter` 把原始 6 通道扩成多尺度——
低通（保留姿态与步态包络）、带通（步频段）、高通（高频抖动）。
这等于把"多尺度分解"显式喂给网络，而不指望它自己从原始信号里学出来。
滤波核固定（不可学），因此不增加过拟合风险。

**模型后的滤波**在推理侧做（见 bench.py 的后处理方法），此处不涉及。
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from imunet_baseline import IMUNet2024


# ---------------------------------------------------------------- 前置滤波

class GlobalNorm(nn.Module):
    """全局标准化：按**训练集统计**的常数做仿射，不是逐样本归一化。

    依据 PedestrianDiffusion——该文明确采用全局标准化（数据集最大 std 的 5σ 缩放），
    并指出**逐样本归一化会破坏物理尺度**（同一段运动被缩放后与真实速度失配）。
    本项目此前不做任何归一化：原始物理量直接入网，其中
    **加计 z 携带 9.810 m/s² 的重力常量**，比该通道自身标准差（2.618）还大 3.7 倍，
    白白占用动态范围。

    常数在此写死（由 train.load_split 的训练集实测，见 docs/13），
    不随批次变化，故：
      · 不破坏偏航等变性——水平 x/y 用**同一个**尺度，z 单独一个；
        若 x、y 尺度不同，旋转就不再与缩放交换。
      · 训练/推理完全一致，无统计漂移。
    """
    # 训练集实测：陀螺 xy 0.854、z 0.993；加计 xy 2.134、z 2.618（重力已扣）
    #
    # mu/sd 可选覆盖——用于 docs/22 §30.3.2 的 AirIO body 系消融实验：
    # 该实验输入不再是世界系 6 通道而是 body 系 gyro+acce+四元数共 10 通道，
    # 默认的 6 元素常量在形状上就对不上，需要传入按该实验训练集实测的常量。
    # 不传时行为与此前完全一致（向后兼容）。
    def __init__(self, enabled=True, mu=None, sd=None):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return
        # 顺序 [gx,gy,gz,ax,ay,az]；xy 共用尺度以保等变性
        if mu is None:
            mu = [0., 0., 0., 0., 0., 9.810]
        if sd is None:
            sd = [0.854, 0.854, 0.993, 2.134, 2.134, 2.618]
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))

    def forward(self, x):
        if not self.enabled:
            return x
        return (x - self.mu.view(1, -1, 1)) / self.sd.view(1, -1, 1)

class HeadingNorm(nn.Module):
    """水平姿态归一化：把每个窗口绕重力轴旋到统一的水平参考朝向。

    重力对齐（z 轴朝上）在数据加载时已完成，但**水平面内的朝向仍是任意的**——
    手持、裤兜、手提包、推车等携带方式，设备水平朝向各不相同
    （实测各库"加速度主轴与行走方向夹角"的标准差达 11°–29°）。

    参考朝向取窗口内水平加速度的**主轴**（协方差最大特征向量），
    它由行走产生的前后向摆动主导，与携带方式关系较弱。
    做法：把水平分量旋到主轴与 x 轴重合。

    ⚠ 关键：这个旋转角必须**输出给调用方**。本实现在网络输出端
    把预测速度旋回世界系，因此训练损失直接与世界系目标比较，不额外旋转目标。
    """
    def forward(self, x):
        # x: [B, 6, T]，通道 0-2 陀螺、3-5 加计（世界系，z 已对齐重力）
        a = x[:, 3:5, :]                                   # [B,2,T]
        ac = a - a.mean(dim=2, keepdim=True)
        C = torch.einsum('bit,bjt->bij', ac, ac) / ac.shape[2]
        # 2x2 对称阵主特征向量的闭式解，避免 eigh 在 MPS 上的开销与不稳定
        cxx, cyy, cxy = C[:, 0, 0], C[:, 1, 1], C[:, 0, 1]
        theta = 0.5 * torch.atan2(2 * cxy, cxx - cyy)      # 主轴方位角
        # ⚠ 主轴有 180° 符号歧义（±v 同为主轴），atan2 在 2θ 上取值故 θ 本身
        # 落在 (-π/2, π/2]，输入整体旋转 φ 时 θ 会按 mod π 折叠、不连续。
        # 用第三阶矩定符号：行走的前后向加速度不对称（蹬地强于回摆），
        # 沿主轴的三阶矩符号稳定，据此把 θ 唯一化到整个圆周。
        cth, sth = torch.cos(theta), torch.sin(theta)
        proj = ac[:, 0, :] * cth[:, None] + ac[:, 1, :] * sth[:, None]
        skew = (proj ** 3).mean(dim=1)
        theta = theta + torch.where(skew < 0, torch.pi, 0.0)
        c, s = torch.cos(-theta), torch.sin(-theta)        # 旋转 -theta 使主轴对齐 x
        out = x.clone()
        for off in (0, 3):
            xx, yy = x[:, off, :], x[:, off + 1, :]
            out[:, off, :] = c[:, None] * xx - s[:, None] * yy
            out[:, off + 1, :] = s[:, None] * xx + c[:, None] * yy
        return out, theta


class PreFilter(nn.Module):
    """把 6 通道扩成 6*(1+3)=24 通道：原始 + 低通 + 带通 + 高通。

    低通 ~0.5 Hz（姿态/朝向的慢分量）、带通 0.5–3 Hz（人的步频段）、
    其余为高通。核用高斯差分实现，固定不可学。
    """
    def __init__(self, rate=200.0, enabled=True, gnorm=False):
        super().__init__()
        self.enabled = enabled
        self.gn = GlobalNorm(enabled=gnorm)
        if not enabled:
            self.out_mult = 1
            return
        self.out_mult = 4
        def gauss(sigma):
            r = max(1, int(3 * sigma))
            x = torch.arange(-r, r + 1, dtype=torch.float32)
            k = torch.exp(-0.5 * (x / sigma) ** 2)
            return (k / k.sum()).view(1, 1, -1)
        self.register_buffer("k_lo", gauss(rate / (2 * math.pi * 0.5)))
        self.register_buffer("k_mid", gauss(rate / (2 * math.pi * 3.0)))

    def forward(self, x):
        x = self.gn(x)
        if not self.enabled:
            return x
        B, C, T = x.shape
        z = x.reshape(B * C, 1, T)
        lo = F.conv1d(z, self.k_lo, padding=self.k_lo.shape[-1] // 2)[..., :T]
        mid = F.conv1d(z, self.k_mid, padding=self.k_mid.shape[-1] // 2)[..., :T]
        lo = lo.reshape(B, C, T); mid = mid.reshape(B, C, T)
        return torch.cat([x, lo, mid - lo, x - mid], dim=1)


def _unrotate(v, theta):
    """把在归一化坐标系下预测的速度旋回原坐标系。

    若输出多于 2 维（后续为对数方差），只旋转前两维；
    方差维在归一化坐标系下定义，配套的损失也在该坐标系下计算，故不旋。
    """
    if theta is None:
        return v
    c, s = torch.cos(theta), torch.sin(theta)
    out = torch.stack([c * v[:, 0] - s * v[:, 1],
                       s * v[:, 0] + c * v[:, 1]], dim=1)
    return out if v.shape[1] == 2 else torch.cat([out, v[:, 2:]], dim=1)


# ---------------------------------------------------------------- 架构

class ConvStem(nn.Module):
    def __init__(self, cin, cout, k=7, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU(),
            nn.Conv1d(cout, cout, k, stride=stride, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU())

    def forward(self, x): return self.net(x)


class CNNNet(nn.Module):
    """朴素多层一维 CNN（对照组）。"""
    def __init__(self, cin=6, w=128, pre=True, hnorm=False, nout=2):
        super().__init__()
        self.nout = nout
        self.hn = HeadingNorm() if hnorm else None
        self.pre = PreFilter(enabled=pre)
        c = cin * self.pre.out_mult
        layers, ch = [], c
        for out, s in ((w, 2), (w, 2), (w * 2, 2), (w * 2, 2)):
            layers += [nn.Conv1d(ch, out, 5, stride=s, padding=2), nn.BatchNorm1d(out), nn.GELU()]
            ch = out
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(ch, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        v = self.head(self.body(self.pre(x)))
        return _unrotate(v, th)


class TCNBlock(nn.Module):
    def __init__(self, c, d, k=5):
        super().__init__()
        p = (k - 1) * d // 2
        self.c1 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        self.c2 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        self.n1, self.n2 = nn.BatchNorm1d(c), nn.BatchNorm1d(c)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(x + h)


class TCNNet(nn.Module):
    """空洞卷积时序网络：感受野随层数指数增长。"""
    def __init__(self, cin=6, w=128, n=5, pre=True, hnorm=False, nout=2):
        super().__init__()
        self.nout = nout
        self.hn = HeadingNorm() if hnorm else None
        self.pre = PreFilter(enabled=pre)
        self.stem = nn.Conv1d(cin * self.pre.out_mult, w, 5, stride=2, padding=2)
        self.blocks = nn.Sequential(*[TCNBlock(w, 2 ** i) for i in range(n)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(w, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        v = self.head(self.blocks(self.stem(self.pre(x))))
        return _unrotate(v, th)


class LSTMNet(nn.Module):
    """CNN 下采样 + 双向 LSTM。"""
    def __init__(self, cin=6, w=128, pre=True, hnorm=False, nout=2):
        super().__init__()
        self.nout = nout
        self.hn = HeadingNorm() if hnorm else None
        self.pre = PreFilter(enabled=pre)
        self.stem = ConvStem(cin * self.pre.out_mult, w)
        self.rnn = nn.LSTM(w, w, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * w, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        h = self.stem(self.pre(x)).transpose(1, 2)
        o, _ = self.rnn(h)
        return _unrotate(self.head(o.mean(1)), th)


class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        # ⚠ d 为奇数时 pe[:,0::2] 比 pe[:,1::2] 多一列，直接赋值会尺寸不匹配。
        # 按各自实际列数截断 div，使任意 d 都可用（width 非偶数倍时会触发）。
        pe[:, 0::2] = torch.sin(pos * div[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.shape[1]]


class TransformerNet(nn.Module):
    """卷积分块嵌入 + Transformer 编码器（含 CLS 汇聚）。"""
    def __init__(self, cin=6, d=128, nhead=4, nlayer=4, pre=True, patch=8,
                 hnorm=False, nout=2, rate=200.0):
        super().__init__()
        self.nout = nout
        self.hn = HeadingNorm() if hnorm else None
        # rate 必须显式传入固定滤波器组；论文权重均为 200 Hz，因此这一修复
        # 在既有配置下数值不变，同时避免其它采样率静默使用错误的截止频率。
        self.pre = PreFilter(rate=rate, enabled=pre)
        self.embed = nn.Conv1d(cin * self.pre.out_mult, d, patch, stride=patch)
        self.pos = PosEnc(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                         batch_first=True, activation="gelu",
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, nlayer)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        h = self.embed(self.pre(x)).transpose(1, 2)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1)
        h = self.enc(self.pos(h))
        return _unrotate(self.head(h[:, 0]), th)


class RoNINResidualBlock1D(nn.Module):
    """RoNIN 论文所用一维 ResNet-18 的基本残差块。

    本实现依据论文和官方公开架构参数独立编写，不复制其 GPL 源文件；
    结构一致性由参数量、各层输出长度和单批前向测试共同校验。
    """
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(cout)))

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class RoNINResNet18(nn.Module):
    """面向 200 点 IMU 窗口的 RoNIN 一维 ResNet-18 速度回归器。

    结构为 7 点/步长2输入层、最大池化、[2,2,2,2] 残差组、512→128 的 1×1 过渡层，
    以及论文给出的 512-512-2 全连接头。完整管线对照不含本文 HeadingNorm、GlobalNorm
    或多频带 PreFilter。
    """
    def __init__(self, cin=6, nout=2):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv1d(cin, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        channels = (64, 128, 256, 512)
        groups, current = [], 64
        for gi, channel in enumerate(channels):
            stride = 1 if gi == 0 else 2
            groups.append(nn.Sequential(
                RoNINResidualBlock1D(current, channel, stride=stride),
                RoNINResidualBlock1D(channel, channel)))
            current = channel
        self.residual_groups = nn.Sequential(*groups)
        self.transition = nn.Sequential(
            nn.Conv1d(512, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.head = nn.Sequential(
            nn.Linear(128 * 7, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, nout))
        self._initialize()

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight); nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0.0, 0.01); nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.transition(self.residual_groups(self.input_block(x)))
        if x.shape[-1] != 7:
            raise ValueError(f"RoNINResNet18 要求 200 点窗口，当前末层长度为 {x.shape[-1]}")
        return self.head(x.flatten(1))


class NormalizedBackbone(nn.Module):
    """把 HN-Transformer 的坐标/尺度前端接到六通道对照主干。

    该包装器只用于 RQ2 的受控主干比较：保持 RoNIN-ResNet18 和 IMUNet 已发表的六通道
    输入拓扑，同时使用与无 PreFilter Transformer 相同的 HeadingNorm 和 GlobalNorm。
    三个主干都关闭 PreFilter，因为其 24 通道输出会迫使 IMUNet 改造输入校正分支，导致
    “只比较主干”的解释不再成立。
    """

    def __init__(self, backbone, hnorm=False, gnorm=False,
                 gnorm_mu=None, gnorm_sd=None):
        super().__init__()
        self.hn = HeadingNorm() if hnorm else None
        self.gn = GlobalNorm(enabled=gnorm, mu=gnorm_mu, sd=gnorm_sd)
        self.backbone = backbone

    def forward(self, x):
        theta = None
        if self.hn is not None:
            x, theta = self.hn(x)
        return _unrotate(self.backbone(self.gn(x)), theta)


class ConvTransformerNet(nn.Module):
    """CNN 提局部特征 + Transformer 建长程依赖（混合，通常最稳）。"""
    def __init__(self, cin=6, w=128, d=128, nhead=4, nlayer=3, pre=True, hnorm=False, nout=2):
        super().__init__()
        self.nout = nout
        self.hn = HeadingNorm() if hnorm else None
        self.pre = PreFilter(enabled=pre)
        self.stem = ConvStem(cin * self.pre.out_mult, w)      # 200 -> 50
        self.proj = nn.Conv1d(w, d, 3, stride=2, padding=1)   # 50 -> 25
        self.pos = PosEnc(d)
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                         batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, nlayer)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        h = self.proj(self.stem(self.pre(x))).transpose(1, 2)
        h = self.enc(self.pos(h))
        return _unrotate(self.head(h.mean(1)), th)


class TwoStageAttnNet(nn.Module):
    """双阶段注意力（X-IONet 思路）：时间自注意力 + 传感器轴间自注意力。

    动机：6 个通道（陀螺 xyz、加计 xyz）之间存在强物理耦合——转弯时角速度与
    向心加速度对应、步态中竖直加计与水平摆动同步。普通 Transformer 把 6 个通道
    在嵌入层就混成一个向量，之后只在时间维做注意力，**轴间关系被固化在第一层**。
    X-IONet 的消融显示显式建模轴间关系是其最大单项贡献（去掉后 ATE +33%~108%）。

    做法：把窗口切成 N 段，每段每通道独立投影 → 特征张量 [B, C, N, D]
      · 时间注意力：对每个通道，在 N 段之间做自注意力
      · 轴间注意力：对每个时间段，在 C 个通道之间做自注意力
    两者交替 nlayer 层，最后池化回归。
    """
    def __init__(self, cin=6, d=64, nhead=4, nlayer=3, nseg=20, seglen=10,
                 pre=True, hnorm=False, nout=2, cgroup=4):
        super().__init__()
        self.nout = nout
        self.hn = HeadingNorm() if hnorm else None
        self.pre = PreFilter(enabled=pre)
        # 24 个通道里，PreFilter 产生的 4 个频带是同一物理量的分解，
        # 先用分组 1x1 卷积把每个物理轴的 4 个频带压成 cgroup 个，
        # 把轴间注意力的序列长度从 24 降到 6*cgroup/4——显著降计算量。
        C0 = cin * self.pre.out_mult
        self.squeeze = nn.Conv1d(C0, cin * cgroup, 1, groups=cin) if pre else None
        self.C = cin * cgroup if pre else C0
        self.nseg = nseg
        self.d = d
        # 段嵌入必须在构造时建好——惰性构建会让 state_dict 缺键、无法保存/加载。
        # 段长固定为 seglen；前向时按实际窗长自适应池化到 seglen，
        # 从而同一模型可用于不同窗长/采样率。
        self.seglen = seglen
        self.seg_embed = nn.Linear(seglen, d)
        self.pos_t = nn.Parameter(torch.zeros(1, 1, nseg, d))
        self.pos_c = nn.Parameter(torch.zeros(1, self.C, 1, d))
        mk = lambda: nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                                batch_first=True, activation="gelu",
                                                norm_first=True)
        self.t_layers = nn.ModuleList([mk() for _ in range(nlayer)])
        self.c_layers = nn.ModuleList([mk() for _ in range(nlayer)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nout))

    def _embed(self, x):
        if self.squeeze is not None:
            # PreFilter 输出按 [原始6, 低通6, 带通6, 高通6] 排列，
            # 需重排成 [轴0的4带, 轴1的4带, ...] 才能按轴分组
            B0, C0, T0 = x.shape
            x = x.reshape(B0, 4, C0 // 4, T0).transpose(1, 2).reshape(B0, C0, T0)
            x = self.squeeze(x)
        B, C, T = x.shape
        L = T // self.nseg
        x = x[:, :, :L * self.nseg].reshape(B * C * self.nseg, 1, L)
        if L != self.seglen:
            x = F.adaptive_avg_pool1d(x, self.seglen)
        x = x.reshape(B, C, self.nseg, self.seglen)
        return self.seg_embed(x)             # [B, C, nseg, d]

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        h = self._embed(self.pre(x))
        h = h + self.pos_t + self.pos_c
        B, C, N, D = h.shape
        for tl, cl in zip(self.t_layers, self.c_layers):
            h = tl(h.reshape(B * C, N, D)).reshape(B, C, N, D)          # 时间维
            h = cl(h.permute(0, 2, 1, 3).reshape(B * N, C, D)) \
                    .reshape(B, N, C, D).permute(0, 2, 1, 3)            # 轴间维
        return _unrotate(self.head(h.mean(dim=(1, 2))), th)


class SeqCtxNet(nn.Module):
    """跨窗时序上下文网络：逐窗编码 -> 因果 GRU 融合 -> 逐窗输出。

    动机（本项目实测得出，非照搬文献）：
    前几轮实验反复出现同一现象——**方向通道是瓶颈**（oracle 上界显示
    修正方向可降 42%–67% ATE，修正长度只有 3%–19%），而方向误差是
    **快抖动型**（1 s 自相关仅 0.44–0.67），所以事后平滑无效
    （dirsmooth1s −0%、dirsmooth5s +12%）。

    但"误差快抖"不等于"没有可用的慢变结构"：设备系与行走方向之间的
    **失准角**本身是慢变隐状态（携带方式几秒内不变），只是逐窗估计它
    的噪声大。事后对**输出**平滑会连真实转弯一起抹掉，而在**特征层**
    做跨窗融合可以只对失准角做时间平均、保留真实转向。这是本模块与
    已否决的输出平滑路线的本质区别。

    ⚠ 与 HeadingNorm 的交互：归一化后每个窗有各自的规范系 θ_k，
    不同窗的嵌入并不在同一坐标系里，直接送进时序层是在比较不可比的量。
    故把 (cos θ_k, sin θ_k) 一并输入，让时序层能自行换算。

    ⚠ 训练/推理的窗间步长必须一致：推理按 0.05 s 步长滑窗，训练也必须
    按 0.05 s 采样连续块，否则时序层学到的是完全不同时间尺度的上下文。
    """

    def __init__(self, cin=6, d=128, nhead=4, nlayer=4, pre=True, patch=8,
                 hnorm=True, nout=2, ctx_layers=2, use_ctx=True):
        super().__init__()
        self.nout = nout
        # use_ctx=False 时把上下文向量置零（GRU 仍存在、参数量不变），
        # 用作**采样方式完全相同、只关掉记忆**的对照组——
        # 否则"跨窗模型输了"会与"块采样牺牲了样本多样性"混淆。
        self.use_ctx = use_ctx
        self.d = d
        self.ctx_layers = ctx_layers
        self.hn = HeadingNorm() if hnorm else None
        self.pre = PreFilter(enabled=pre)
        self.embed = nn.Conv1d(cin * self.pre.out_mult, d, patch, stride=patch)
        self.pos = PosEnc(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                         batch_first=True, activation="gelu",
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, nlayer)
        self.ctx_in = nn.Linear(d + 2, d)
        self.gru = nn.GRU(d, d, ctx_layers, batch_first=True,
                          dropout=0.1 if ctx_layers > 1 else 0.0)
        self.mix = nn.Linear(2 * d, d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nout))

    def encode(self, x):
        """逐窗编码，返回 (特征 [N,d], 规范系角 θ [N] 或 None)。"""
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        h = self.embed(self.pre(x)).transpose(1, 2)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1)
        return self.enc(self.pos(h))[:, 0], th

    def forward(self, x, nblock=1, state=None, return_state=False):
        """x: [nblock*mwin, C, W]，**块内按时间顺序排列**（先块后窗）。

        state = (GRU 隐状态, 上一窗的 θ)，供推理时分块滚动。

        ⚠ 送进时序层的角度必须是**相邻窗之差** Δθ_k = θ_k − θ_{k−1}，
        不能是绝对角 θ_k。整条数据在水平面旋转 φ 时，所有 θ_k 同加 φ，
        绝对角会随之改变、破坏偏航等变性（姿态归一化的最大价值所在）；
        相邻差则完全不变。Δθ 用 (cos, sin) 表示，自动处理 ±π 卷绕。
        """
        h, th = self.encode(x)
        n = h.shape[0]
        m = n // max(nblock, 1)
        hs, ths = state if state is not None else (None, None)
        if th is not None:
            t = th.view(nblock, m)
            prev = ths if ths is not None else t[:, :1]
            dth = t - torch.cat([prev, t[:, :-1]], 1)
            ang = torch.stack([torch.cos(dth), torch.sin(dth)], -1).reshape(n, 2)
            ths = t[:, -1:].detach()
        else:
            ang = h.new_zeros(n, 2)
        g = self.ctx_in(torch.cat([h, ang], -1)).view(nblock, m, self.d)
        c, hs = self.gru(g, hs)
        if not self.use_ctx:
            c = torch.zeros_like(c)
        f = torch.cat([h.view(nblock, m, self.d), c], -1)
        o = self.head(torch.tanh(self.mix(f))).reshape(n, self.nout)
        o = _unrotate(o, th)
        return (o, (hs, ths)) if return_state else o

class HierEncDecNet(nn.Module):
    """层级编码器-解码器（X-IONet 组件二，其消融显示去掉后 ATE +7.6%~40.8%）。

    与本库现有 TransformerNet 的区别：后者是**单一尺度**——把 1 s 窗切成
    固定长度的 patch 后平铺进 Transformer，全程只有一种时间粒度。
    本模块逐层把**相邻时间步两两合并**，形成 L→L/2→L/4→… 的多尺度金字塔，
    再用解码器逐层上采样并与对应尺度的编码器特征做**交叉注意力**。

    动机与本项目实测的契合：既往探针显示方向误差是**快变**的
    （1 s 自相关 0.44–0.67），而步态本身有约 1 s 的周期结构——
    二者处于不同时间尺度。单尺度表示需要用同一组 patch 同时兼顾，
    多尺度金字塔则让不同层各管一段尺度。

    ⚠ 保持与本库其余模型相同的接口：输入 [B,6,T]，输出 [B,nout]，
    并沿用 HeadingNorm + PreFilter，使对比只反映主干结构的差异。
    """

    def __init__(self, cin=6, d=128, nhead=4, nlevel=3, nlayer=2, pre=True,
                 patch=8, hnorm=False, nout=2):
        super().__init__()
        self.nout = nout
        self.nlevel = nlevel
        self.hn = HeadingNorm() if hnorm else None
        self.pre = PreFilter(enabled=pre)
        self.embed = nn.Conv1d(cin * self.pre.out_mult, d, patch, stride=patch)
        self.pos = PosEnc(d)

        def block(n):
            layer = nn.TransformerEncoderLayer(
                d, nhead, 4 * d, dropout=0.1, batch_first=True,
                activation="gelu", norm_first=True)
            return nn.TransformerEncoder(layer, n)

        # 编码器：每级先自注意力，再把相邻两个时间步合并（线性降采样）
        self.enc = nn.ModuleList([block(nlayer) for _ in range(nlevel)])
        self.merge = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(nlevel - 1)])
        # 解码器：自底向上，每级与同尺度编码器特征做交叉注意力
        self.dec = nn.ModuleList([block(nlayer) for _ in range(nlevel - 1)])
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(d, nhead, dropout=0.1, batch_first=True)
            for _ in range(nlevel - 1)])
        self.dnorm = nn.ModuleList([nn.LayerNorm(d) for _ in range(nlevel - 1)])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        h = self.pos(self.embed(self.pre(x)).transpose(1, 2))   # [B,L,d]
        feats = []
        for i in range(self.nlevel):
            h = self.enc[i](h)
            feats.append(h)
            if i < self.nlevel - 1:
                if h.shape[1] % 2:                              # 奇数长度补齐
                    h = torch.cat([h, h[:, -1:]], 1)
                b, l, dd = h.shape
                h = self.merge[i](h.reshape(b, l // 2, 2 * dd))
        # 自最粗尺度向上解码
        for j in range(self.nlevel - 2, -1, -1):
            h = h.repeat_interleave(2, dim=1)[:, :feats[j].shape[1]]
            q = self.dec[j](h)
            a, _ = self.cross[j](q, feats[j], feats[j])
            h = self.dnorm[j](q + a)
        # ⚠ 曾在此处把 CLS 拼在**所有注意力层之后**再取 h[:,0]——
        # 那个位置就是 CLS 本身，从未参与任何计算，模型退化为输出常数向量
        # （实测：不同输入下输出模长恒为 0.0597，仅被 HeadingNorm 的角度旋转；
        #  val_loss 0.3969 ≈ 目标方差，即"预测训练集均值"）。
        # 改为对解码后的序列做**均值池化**——层级结构中序列长度逐层变化，
        # 把 CLS 贯穿到底会与合并/上采样冲突，均值池化最简洁且无此问题。
        return _unrotate(self.head(h.mean(dim=1)), th)

class STFTNet(nn.Module):
    """短时傅里叶变换前端 + Transformer（PedestrianDiffusion 的谱域表示）。

    该文消融显示谱域优于时域；本库现有模型均为时域 + 固定滤波器组
    （PreFilter 把 6 通道扩成 raw/低通/带通/高通 24 通道），
    等价于一组**手工挑选的固定频带**。STFT 则给出完整的时频分解，
    让网络自行决定用哪些频带。

    ⚠ 必须取**实部/虚部**而非幅值：STFT 对输入是线性的，故水平面旋转
    会同样地旋转 x/y 通道的谱系数，偏航等变性得以保持；取幅值会丢掉相位、
    破坏等变性（本项目的姿态归一化正是靠等变性起作用）。

    输出形状链（默认 n_fft=64, hop=16, 200 点窗）：
      [B,6,200] --HeadingNorm--> [B,6,200]
                --STFT--> [B,6,33,10] 复数
                --实虚拼接--> [B, 6*33*2=396, 10]
                --Linear--> [B,10,d] --Transformer(+CLS)--> [B,d] --head--> [B,nout]
    """

    def __init__(self, cin=6, d=128, nhead=4, nlayer=4, n_fft=64, hop=16,
                 pre=False, hnorm=False, nout=2):
        super().__init__()
        self.nout = nout
        self.n_fft = n_fft
        self.hop = hop
        self.hn = HeadingNorm() if hnorm else None
        self.register_buffer("win", torch.hann_window(n_fft))
        nfreq = n_fft // 2 + 1
        self.proj = nn.Linear(cin * nfreq * 2, d)
        self.pos = PosEnc(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                         batch_first=True, activation="gelu",
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, nlayer)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        b, c, t = x.shape
        z = torch.stft(x.reshape(b * c, t).float(), self.n_fft, self.hop,
                       window=self.win, return_complex=True, center=True,
                       pad_mode="reflect")                      # [B*C, F, T']
        f, tt = z.shape[-2], z.shape[-1]
        z = torch.stack([z.real, z.imag], -1)                   # [B*C,F,T',2]
        z = z.reshape(b, c, f, tt, 2).permute(0, 3, 1, 2, 4).reshape(b, tt, c * f * 2)
        h = self.pos(self.proj(z))
        h = torch.cat([self.cls.expand(b, -1, -1), h], 1)
        return _unrotate(self.head(self.enc(h)[:, 0]), th)

def build(name, pre=True, width=1.0, depth=1.0, hnorm=False, nout=2,
          gnorm=False, gnorm_mu=None, gnorm_sd=None,
          nseg=None, seglen=None, rate=200.0):
    """width/depth 用于放大模型：官方 RoNIN ResNet18-1d 约 4.6M 参数，
    本库默认配置约 0.85M，公平对比时需放大到同量级。"""
    w = int(round(128 * width))
    d = int(round(128 * width))
    nl = max(1, int(round(4 * depth)))
    nb = max(1, int(round(5 * depth)))
    # ⚠ 注意力要求 d 能被头数整除。width 非整数倍时（如 1.2 → d=154，头数 4）
    # 会在 MultiheadAttention 构造处报错。这里把 d 向上取整到头数的倍数，
    # 使任意 width 都可用——参数量微调时必须能取非整数倍。
    nh = max(2, d // 32)
    if d % nh:
        d += nh - d % nh

    def _fin(m):
        """统一把全局标准化开关注入各架构的 PreFilter（唯一的输入入口）。"""
        if gnorm and hasattr(m, "pre"):
            m.pre.gn = GlobalNorm(enabled=True, mu=gnorm_mu, sd=gnorm_sd)
        return m
    def _six_channel_backbone(factory):
        if pre:
            raise ValueError("six-channel backbone comparison requires pre=False")
        backbone = factory()
        if hnorm or gnorm:
            return NormalizedBackbone(
                backbone, hnorm=hnorm, gnorm=gnorm,
                gnorm_mu=gnorm_mu, gnorm_sd=gnorm_sd)
        return backbone

    return _fin({"cnn": lambda: CNNNet(w=w, pre=pre, hnorm=hnorm, nout=nout),
            "ronin_resnet18": lambda: _six_channel_backbone(
                lambda: RoNINResNet18(nout=nout)),
            "imunet2024": lambda: _six_channel_backbone(IMUNet2024),
            "tcn": lambda: TCNNet(w=w, n=nb, pre=pre, hnorm=hnorm, nout=nout),
            "lstm": lambda: LSTMNet(w=w, pre=pre, hnorm=hnorm, nout=nout),
            "transformer": lambda: TransformerNet(d=d, nlayer=nl, pre=pre,
                                                  nhead=nh, hnorm=hnorm,
                                                  nout=nout, rate=rate),
            "conv_transformer": lambda: ConvTransformerNet(
                w=w, d=d, nlayer=max(1, int(round(3 * depth))),
                nhead=nh, pre=pre, hnorm=hnorm, nout=nout),
            "stft": lambda: STFTNet(d=d, nlayer=nl, nhead=nh, pre=pre,
                                    hnorm=hnorm, nout=nout),
            "hier": lambda: HierEncDecNet(
                d=d, nlevel=3, nlayer=max(1, int(round(2 * depth))), pre=pre,
                nhead=nh, hnorm=hnorm, nout=nout),
            "seq_ctx": lambda: SeqCtxNet(d=d, nlayer=nl, pre=pre,
                                         nhead=nh, hnorm=hnorm,
                                         nout=nout),
            # nseg/seglen 默认取 X-IONet 口径 20×10；
            # 传入其它值是为了加载**改口径之前**训练的检查点（如 M_twostage 的 10×20），
            # 那些权重的 pos_t / seg_embed 形状与当前默认不符，直接 load 会报错。
            "two_stage": lambda: TwoStageAttnNet(
                d=max(32, int(round(96 * width))), nlayer=max(1, int(round(4 * depth))),
                nhead=max(2, int(round(96 * width)) // 32), pre=pre,
                hnorm=hnorm, nout=nout,
                **({} if nseg is None else {"nseg": nseg}),
                **({} if seglen is None else {"seglen": seglen}))}[name]())


if __name__ == "__main__":
    x = torch.randn(4, 6, 200)
    for n in ("cnn", "tcn", "lstm", "transformer", "conv_transformer"):
        for pre in (True, False):
            m = build(n, pre)
            p = sum(q.numel() for q in m.parameters()) / 1e6
            print(f"{n:<18} pre={str(pre):<5} 参数 {p:5.2f}M  输出 {tuple(m(x).shape)}")
