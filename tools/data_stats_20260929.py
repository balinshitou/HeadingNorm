"""正文表1：各测试集与确认集的序列数、受试者（分组）数、时长与真值路长，由评测缓存npz直接计算。
训练/验证划分的时长取自 config/splits_subject_disjoint_v2.json。
用法：.venv/bin/python tools/data_stats_20260929.py
输出：results/data_stats_20260929/stats.json
"""
import collections, json, os, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / 'data/eval'


def stats(files, group):
    h = km = 0.0
    g = collections.Counter()
    for f in files:
        z = np.load(f, allow_pickle=True)
        h += (z['tm'][-1] - z['tm'][0]) / 3600
        km += float(np.linalg.norm(np.diff(z['gt'], axis=0), axis=1).sum()) / 1000
        g[group(f.name)] += 1
    return {'sequences': len(files), 'groups': len(g), 'hours': round(h, 3), 'gt_path_km': round(km, 2),
            'per_group': dict(sorted(g.items()))}


ronin = sorted((E / 'benchmark').glob('ronin__*.npz'))
subj = lambda n: n.split('__')[1].split('_')[0]
phone_subj = lambda n: re.search(r'Subje(?:ct|tc)_(\d+)', n).group(1)  # 原数据有一处拼写为 Subjetc
out = {
    'ronin_seen': stats([f for f in ronin if str(np.load(f)['split']) == 'seen'], subj),
    'ronin_unseen': stats([f for f in ronin if str(np.load(f)['split']) == 'unseen'], subj),
    'ridi': stats(sorted((E / 'benchmark').glob('ridi__*.npz')), subj),
    'tlio_test': stats(sorted((E / 'tlio_posthoc').glob('*.npz')), lambda n: '-'),
    'tlio_confirm': stats(sorted((E / 'confirm_tlio').glob('*.npz')), lambda n: '-'),
    'phone_test': stats(sorted((E / 'imunet_owndata').glob('*.npz')), phone_subj),
    'phone_confirm': stats(sorted((E / 'confirm_imunet').glob('*.npz')), phone_subj),
}
sp = json.loads((ROOT / 'config/splits_subject_disjoint_v2.json').read_text())
out['ronin_train_hours'] = sp['actual_training_hours']['1.00']
out['ronin_val_hours'] = sp['validation_selection']['actual_hours']
(ROOT / 'results/data_stats_20260929/stats.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
for k, v in out.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != 'per_group'} if isinstance(v, dict) else v)
