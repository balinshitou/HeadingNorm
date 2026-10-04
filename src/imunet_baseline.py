"""依据论文独立实现的 IMUNet PyTorch 拓扑，用作同数据、同预算卷积对照。

参考：B. Zeinali, H. Zanddizari, and M. J. Chang, IEEE TIM, 2024，
doi:10.1109/TIM.2024.3381717。张量尺寸曾对照作者仓库固定提交
``c57f14d0f4f5bbb8e29d836dabacc8718ca825e1``；因冻结审计时未发现仓库许可证，
本项目没有复制或再分发其源代码。
"""
from __future__ import annotations

import torch
from torch import nn


class SeparableResidual1D(nn.Module):
    """IMUNet 主干使用的深度可分离一维残差单元。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 project_shortcut: bool = False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, 3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.depth_bn = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.point_bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ELU()
        self.shortcut = (nn.Identity() if not project_shortcut and stride == 1 and
                         in_channels == out_channels else
                         nn.Sequential(nn.Conv1d(in_channels, out_channels, 1,
                                                 stride=stride, bias=False),
                                       nn.BatchNorm1d(out_channels)))

    def forward(self, values):
        residual = self.shortcut(values)
        values = self.activation(self.depth_bn(self.depthwise(values)))
        values = self.activation(self.point_bn(self.pointwise(values)))
        return self.activation(values + residual)


class AffineInputCorrection(nn.Module):
    """在 1,200 个编码值与原窗口值之间执行可学习的逐元素输入校正。"""

    def __init__(self, size: int = 1200):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, size))
        self.bias = nn.Parameter(torch.zeros(1, size))

    def forward(self, encoded, flattened_input):
        return encoded - self.scale * flattened_input + self.bias


class IMUNet2024(nn.Module):
    """六通道、200 采样点输入、二维速度输出的 IMUNet 回归器。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1))
        blocks = []
        current = 64
        # 两个 64 通道组之后接四个步长为 2 的扩宽组。
        for group_index, (target, stride) in enumerate(((64, 1), (64, 1), (128, 2),
                                                        (256, 2), (512, 2), (1024, 2))):
            blocks.extend((SeparableResidual1D(current, target, stride,
                                                project_shortcut=group_index > 0),
                           SeparableResidual1D(target, target, 1)))
            current = target
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Sequential(nn.Conv1d(1024, 400, 2), nn.BatchNorm1d(400))
        self.correction = AffineInputCorrection(1200)
        self.activation = nn.ELU()
        self.output = nn.Linear(1200, 2)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (6, 200):
            raise ValueError(f"IMUNet2024 requires [B,6,200], received {tuple(values.shape)}")
        raw = values.flatten(1)
        encoded = self.projection(self.blocks(self.stem(values))).flatten(1)
        if encoded.shape[1] != 1200:
            raise RuntimeError(f"unexpected IMUNet encoded width: {encoded.shape[1]}")
        return self.output(self.activation(self.correction(encoded, raw)))


if __name__ == "__main__":
    model = IMUNet2024()
    print(sum(p.numel() for p in model.parameters()), model(torch.zeros(2, 6, 200)).shape)
