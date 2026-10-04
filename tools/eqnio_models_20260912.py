"""EqNIO（ICLR 2025）模型的加载封装：只从使用者自行获取的 EqNIO 源码导入，本仓库不复制其代码。

来源：github.com/RoyinaJayanth/EqNIO，commit 97b7a60a5e5cc8f0132bba343c83de114211f827。
该仓库在此 commit 上没有 LICENSE 文件，因此本仓库不复制、不再分发其任何代码
（`docs/DATA_AND_LICENSE_AUDIT.md`）。2026-09-20 起，`get_model` 与两个预处理函数改为直接从上游
`RONIN/source/ronin_resnet.py` 导入；此前本文件曾逐字保留这三个函数，行为与现在完全一致
（等价性由 `verification/generated/eqnio_upstream_reference_20260920.npz` 逐元素核对，见 `tools/eqnio_upstream_check_20260920.py`）。

准备工作（使用者自行完成）：
  git clone https://github.com/RoyinaJayanth/EqNIO.git
  git -C EqNIO checkout 97b7a60a5e5cc8f0132bba343c83de114211f827
  # 权重按其 README 的 “RONIN + 50% data + SO(2)/O(2) Eq. Frame” 链接下载
路径由环境变量覆盖：EQNIO_SRC（默认 code-data/external_code/EqNIO/RONIN/source）、
EQNIO_WEIGHTS（默认 code-data/external_weights/eqnio_ronin）、PYLIB（einops 等依赖）。

`ronin_resnet.py` 顶部会导入若干只在其训练脚本里用到的第三方包（tensorboardX、numba、tqdm、quaternion）。
本模块为这些包装入最小占位模块，只为完成导入；被导入的三个函数不使用它们。
"""
from __future__ import annotations
import os, sys, types
from pathlib import Path

import torch
from torch import nn

import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src')); import hn_paths   # 外部资料路径表
EQ_SRC = Path(os.environ.get('EQNIO_SRC', hn_paths.EQNIO_SRC))
PYLIB = Path(os.environ.get('PYLIB', hn_paths.PYLIB))          # einops
EQ_WEIGHTS = Path(os.environ.get('EQNIO_WEIGHTS', hn_paths.EQNIO_WEIGHTS))

WEIGHTS = {'eqnio_so2': ('resnet18_eq_frame_2vec', EQ_WEIGHTS / 'Ronin_so2/Ronin_so2/checkpoint_111.pt'),
           'eqnio_o2': ('resnet18_eq_frame_o2', EQ_WEIGHTS / 'Ronin_o2/Ronin_o2/checkpoint_38.pt')}

_STUBS = {'tensorboardX': {'SummaryWriter': object},
          'numba': {'jit': lambda *a, **k: (lambda f: f), 'njit': lambda *a, **k: (lambda f: f)},
          'tqdm': {'tqdm': lambda x=None, *a, **k: x},
          'quaternion': {}}


def _upstream():
    """导入上游 ronin_resnet 并返回该模块（其自带同名 model_resnet1d，须置于 sys.path 最前）。"""
    if not (EQ_SRC / 'ronin_resnet.py').exists():
        raise RuntimeError(f'未找到 EqNIO 源码：{EQ_SRC}。请按本文件开头的说明获取该 commit，或设置 EQNIO_SRC。')
    for p in (str(PYLIB), str(EQ_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name, attrs in _STUBS.items():
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                m = types.ModuleType(name)
                for k, v in attrs.items():
                    setattr(m, k, v)
                sys.modules[name] = m
    import ronin_resnet
    return ronin_resnet


def get_model(arch):
    """上游 ronin_resnet.get_model：两个 arch 的超参数由上游定义，本仓库不复制。"""
    return _upstream().get_model(arch)


def preprocess_eq_frame(feat):
    """上游 ronin_resnet.preprocess_eq_frame（SO(2) 臂的向量/标量拆分）。"""
    return _upstream().preprocess_eq_frame(feat)


def preprocess_eq_o2_frame(feat):
    """上游 ronin_resnet.preprocess_eq_o2_frame（O(2) 臂的向量/标量拆分）。"""
    return _upstream().preprocess_eq_o2_frame(feat)


class EqNIOWindow(nn.Module):
    """把 EqNIO 包成与本项目窗口模型相同的接口：输入 [B,6,200]，输出世界系速度 [B,2]。"""
    def __init__(self, net, arch):
        super().__init__()
        self.net = net
        self.pre = preprocess_eq_o2_frame if 'o2' in arch else preprocess_eq_frame

    def forward(self, x):
        v, s, o = self.pre(x)
        return self.net(v.float(), s.float(), o.float())[1]


def load_eqnio(name, dev):
    arch, path = WEIGHTS[name]
    net = get_model(arch)
    obj = torch.load(path, map_location='cpu', weights_only=False)
    net.load_state_dict(obj['model_state_dict'], strict=True)
    return EqNIOWindow(net, arch).eval().to(dev), path
