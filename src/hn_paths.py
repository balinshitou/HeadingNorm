"""项目外部资料（第三方原始数据集、官方代码、官方权重）的唯一路径表。

脚本不再各自写死绝对路径，统一从这里取。每一项按下面的顺序找，用第一个存在的：
  1. 环境变量 HN_EXTERNAL 指向的目录；
  2. 项目内的 external/（把数据或权重放进来，脚本就会优先用它，不依赖本机其他位置）；
两处都没有时返回第 1 处的路径（便于报错信息指出该放到哪里）。

检查全部路径：.venv/bin/python tools/check_paths_20261001.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ENV = os.environ.get('HN_EXTERNAL')
EXTERNAL_ROOTS = ([Path(_ENV)] if _ENV else []) + [ROOT / 'external']


def ext(*candidates):
    """candidates：相对外部根目录的路径，可给多个等价写法（如发布包与本机目录名不同）。"""
    for root in EXTERNAL_ROOTS:
        for rel in candidates:
            if (root / rel).exists():
                return root / rel
    return EXTERNAL_ROOTS[0] / candidates[0]


# 第三方原始数据集
RONIN_RAW = ext('RoNIN_Dataset')
RIDI_RAW = ext('RIDI/data_publish_v2')
OXIOD_RAW = ext('OxIOD')
TLIO_RAW = ext('tlio_golden')
IMUNET_RAW = ext('IMUNet_dataset')
# 官方代码
RONIN_SRC = ext('ronin/source')
RONIN_LISTS = ext('ronin/lists')
EQNIO_SRC = ext('external_code/EqNIO/RONIN/source')
PYLIB = ext('external_code/pylib')                    # einops 等，EqNIO 用
PYLIB_IMUNET = ext('external_code/pylib_imunet')      # numpy-quaternion，IMUNet 数据解析用
IMUNET_RUNS = ext('external_code/IMUNet_runs/RONIN_torch')
# 官方权重
RONIN_WEIGHTS = ext('ronin_models', 'ronin_zero_shot_analysis/models')
EQNIO_WEIGHTS = ext('external_weights/eqnio_ronin')
# 论文二与早期工作用到的自采数据与旧代码
FWHP_DATA = ext('data/Four Ways to Hold the Phone')
MLTS_DATA = ext('data/Machine_Learning_Training_Set')
LEGACY_CODE = ext('code')
PDR_SRC = ext('主线_可辨识性/code')

# name -> (路径, 用途, 现行稿是否需要)
TABLE = {
    'RONIN_RAW': (RONIN_RAW, 'RoNIN 原始数据集（重建训练/评测缓存）', True),
    'RIDI_RAW': (RIDI_RAW, 'RIDI 原始数据集（重建评测缓存）', True),
    'OXIOD_RAW': (OXIOD_RAW, 'OxIOD 原始数据集（现行稿不用）', False),
    'TLIO_RAW': (TLIO_RAW, 'TLIO 原始数据集（重建测试/确认集缓存）', True),
    'IMUNET_RAW': (IMUNET_RAW, 'IMUNet 手机数据集（重建测试/确认集缓存）', True),
    'RONIN_SRC': (RONIN_SRC, 'RoNIN 官方模型代码（加载官方权重）', True),
    'RONIN_LISTS': (RONIN_LISTS, 'RoNIN 官方序列清单', True),
    'RONIN_WEIGHTS': (RONIN_WEIGHTS, 'RoNIN 官方 ResNet/LSTM/TCN 权重', True),
    'PYLIB_IMUNET': (PYLIB_IMUNET, 'numpy-quaternion（解析 IMUNet 数据）', True),
    'EQNIO_SRC': (EQNIO_SRC, 'EqNIO 官方代码（现行稿不用）', False),
    'PYLIB': (PYLIB, 'EqNIO 依赖（现行稿不用）', False),
    'EQNIO_WEIGHTS': (EQNIO_WEIGHTS, 'EqNIO 官方权重（现行稿不用）', False),
    'IMUNET_RUNS': (IMUNET_RUNS, 'IMUNet 官方代码与重训权重（现行稿不用）', False),
    'FWHP_DATA': (FWHP_DATA, '自采 Four Ways to Hold the Phone（论文二）', False),
    'MLTS_DATA': (MLTS_DATA, '自采 Machine_Learning_Training_Set（论文二）', False),
    'LEGACY_CODE': (LEGACY_CODE, '早期代码 phyphox_to_ronin 等（论文二）', False),
    'PDR_SRC': (PDR_SRC, '经典 PDR 基线代码（论文二）', False),
}
