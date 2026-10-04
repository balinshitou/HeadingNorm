"""E9：EqNIO 学习式规范参考系，按本文配方重训（docs/83）。

模型类与预处理逐字来自 EqNIO commit 97b7a60，经 tools/eqnio_models_20260912.py 导入，不作改动；
前面加本文 GlobalNorm（水平两轴零中心、共享尺度，与水平旋转可交换，不破坏 EqNIO 的等变性），使各臂输入标准化一致。
冻结代码（src/models.py、src/sensors_frontend.py）不改动，已有运行的哈希照旧可核。
"""
from __future__ import annotations
import sys
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import eqnio_models_20260912 as EM   # 把 EqNIO 源码放到 sys.path 最前（自带同名 model_resnet1d）
import models

ARCH = {'eqnio_so2': 'resnet18_eq_frame_2vec', 'eqnio_o2': 'resnet18_eq_frame_o2'}


class GNEqNIO(nn.Module):
    def __init__(self, frontend, mu, sd):
        super().__init__()
        arch = ARCH[frontend]
        self.gn = models.GlobalNorm(True, mu, sd)
        self.eq = EM.EqNIOWindow(EM.get_model(arch), arch)

    def forward(self, x):
        return self.eq(self.gn(x))


def build(frontend, mu, sd, arch='eqnio', pre=False, width=1.0):
    if frontend not in ARCH or arch != 'eqnio' or pre:
        raise ValueError((frontend, arch, pre))
    return GNEqNIO(frontend, mu, sd)
