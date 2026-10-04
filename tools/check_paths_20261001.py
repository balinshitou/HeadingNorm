"""路径与权重体检：确认现行稿用到的每个文件都在、每个权重文件完好，并给出权重用途清单。只读，不改任何结果。

检查项
  1. 外部资料（src/hn_paths.py 的路径表）：原始数据集、官方代码、官方权重是否存在；
  2. 项目内权重：逐个核对 .pt 与配对的 .json、SHA-256（元数据里有记录的）；按用途分类；
  3. 数据缓存：data/eval、data/train 各目录的文件数；
  4. 现行稿正文与补充材料里写到的每个路径是否存在；
  5. 稿件目录 figures/ 与原件是否一致。
输出：provenance/MODEL_WEIGHTS_AUDIT_20261001.csv、provenance/path_check_20261001.json；有问题时退出码为 1。
用法：.venv/bin/python tools/check_paths_20261001.py [--no-hash]
"""
import csv
import glob
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import hn_paths  # noqa: E402

DRAFT = ROOT / 'paper/draft_zh_20260927'
MAIN = DRAFT / 'HeadingNorm_正文_中文初稿_20260927.md'
SUPP = DRAFT / 'HeadingNorm_补充材料_中文初稿_20260927.md'

OFFICIAL = {   # 官方权重文件：相对 hn_paths 中的根
    'RoNIN ResNet（外挂对象）': hn_paths.RONIN_WEIGHTS / 'ronin_resnet/checkpoint_gsn_latest.pt',
    'RoNIN LSTM': hn_paths.RONIN_WEIGHTS / 'ronin_lstm/checkpoints/ronin_lstm_checkpoint.pt',
    'RoNIN TCN': hn_paths.RONIN_WEIGHTS / 'ronin_tcn/checkpoints/ronin_tcn_checkpoint.pt',
}
CACHES = {'data/eval/benchmark': 186, 'data/eval/tlio_posthoc': 36, 'data/eval/imunet_owndata': 36,
          'data/eval/confirm_tlio': 318, 'data/eval/confirm_imunet': 87, 'data/train/ronin': 88}

# (目录, 组名正则) -> (类别, 稿中名称/说明)。组名 = 文件名去掉 _s<种子> 与扩展名。
A, B, C, D_, E, F = ('A 现行稿直接使用', 'B 只在出处清单中出现', 'C 已从稿中移除的实验',
                     'D 训练断点（只用于续训）', 'E 重复件', 'F 论文二（弱标签）')
RULES = [
    ('sensors_v4', r'.*\.last$', D_, '训练中途快照，评测从不读取'),
    ('sensors_v4', r'Sensors_v4_resnet_gn$', A, 'ResNet18 GN only（表3、表6）'),
    ('sensors_v4', r'Sensors_v4_resnet_yaw$', A, 'ResNet18-YawAug（表3、表6、表7、表8、图1、图4、图5；外挂对象）'),
    ('sensors_v4', r'Sensors_v4_resnet_mixed_pca$', A, 'ResNet18 PCA frame（表3、表6）'),
    ('sensors_v4', r'Sensors_v4_budget_', C, '预算检查（seed 0），相关叙述已从稿中删去'),
    ('e6_sign_ablation', r'.*', A, 'ResNet18 HN w/o sign（表3、表6）'),
    ('e3_backbone_frontend', r'E3_transformer_yaw$', A, 'Transformer-YawAug（表7、表8、补充材料表S16；外挂对象）'),
    ('e3_backbone_frontend', r'E3_imunet_yaw$', A, 'IMUNet-YawAug（表7、表8、补充材料表S16；外挂对象）'),
    ('submission_frozen', r'ResNet18_v3_hn_gn$', A, 'ResNet18 HN（表3、表6、图1、图5；补充材料S1的GN常数）'),
    ('submission_frozen', r'A6_v3_nopre$', A, 'Transformer HN（补充材料表S27）'),
    ('submission_frozen', r'IMUNet2024_v3_hn_gn$', A, 'IMUNet HN（补充材料表S27）'),
    ('submission_frozen', r'ResNet18_v2_matched$', A, '无GN无HN的ResNet18（补充材料表S26）'),
    ('submission_frozen', r'.*', B, '40次训练矩阵的其余配置；补充材料只列汇总文件的哈希，数字未进现行稿'),
    ('generated', r'.*', E, '与 models/submission_frozen/ 同名文件逐字节相同（训练脚本的写出目录）'),
    ('e9_eqnio_retrain', r'.*', C, 'EqNIO 同协议重训，整组已从稿中删去'),
    ('mlts_ft', r'.*', F, '弱标签微调'),
    ('p0_mm', r'.*', F, '三项对照'),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def classify(dirn, group):
    for d, pat, cls, note in RULES:
        if d == dirn and re.match(pat, group):
            return cls, note
    return '未分类', ''


def main():
    do_hash = '--no-hash' not in sys.argv
    problems, report = [], {}

    print('== 1. 外部资料 ==')
    ext = {}
    for name, (p, use, needed) in hn_paths.TABLE.items():
        ok = p.exists()
        ext[name] = dict(path=str(p), exists=ok, needed_by_current_paper=needed, use=use)
        print(f"  {'OK ' if ok else '缺 '} {'[现行稿]' if needed else '[其他]  '} {name:14s} {p}")
        if needed and not ok:
            problems.append(f'外部资料缺失：{name} {p}')
    for name, p in OFFICIAL.items():
        ok = p.exists()
        ext[name] = dict(path=str(p), exists=ok, needed_by_current_paper=True, bytes=p.stat().st_size if ok else 0)
        print(f"  {'OK ' if ok else '缺 '} [现行稿] {name:14s} {p}")
        if not ok:
            problems.append(f'官方权重缺失：{p}')
    report['external'] = ext

    print('== 2. 项目内权重 ==')
    rows, by_hash = [], defaultdict(list)
    for p in sorted((ROOT / 'models').rglob('*.pt')):
        last = p.name.endswith('.last.pt')
        stem = p.name[:-8] if last else p.name[:-3]
        group = re.sub(r'_s\d+$', '', stem) + ('.last' if last else '')
        cls, note = classify(p.parent.name, group)
        meta = p.parent / (stem + '.json')
        rec = dict(file=str(p.relative_to(ROOT)), group=group, seed=(re.search(r'_s(\d+)$', stem) or [None, ''])[1],
                   mb=round(p.stat().st_size / 1e6, 1), category=cls, note=note, metadata='有' if meta.exists() else '无',
                   sha256='', sha_check='')
        if do_hash:
            rec['sha256'] = sha(p)
            by_hash[rec['sha256']].append(rec['file'])
            want = json.loads(meta.read_text()).get('checkpoint_sha256') if meta.exists() and not last else None
            rec['sha_check'] = '无记录' if not want else ('一致' if want == rec['sha256'] else '不一致')
            if rec['sha_check'] == '不一致':
                problems.append(f"权重与元数据哈希不一致：{rec['file']}")
        if cls == '未分类':
            problems.append(f"权重未分类：{rec['file']}")
        rows.append(rec)
    with open(ROOT / 'provenance/MODEL_WEIGHTS_AUDIT_20261001.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = defaultdict(lambda: [0, 0.0])
    for r in rows:
        summ[r['category']][0] += 1
        summ[r['category']][1] += r['mb']
    for k in sorted(summ):
        print(f'  {k:24s} {summ[k][0]:4d} 个  {summ[k][1]:8.1f} MB')
    a_groups = sorted({(r['group'], r['note']) for r in rows if r['category'] == A})
    for g, note in a_groups:
        seeds = sorted(r['seed'] for r in rows if r['group'] == g and r['category'] == A)
        print(f"    {g:32s} 种子 {','.join(seeds)}  {note}")
        if seeds != ['0', '1', '2', '3']:
            problems.append(f'现行稿用到的权重种子不全：{g} {seeds}')
    report['weights'] = {k: dict(n=v[0], mb=round(v[1], 1)) for k, v in summ.items()}
    report['duplicate_sets'] = sum(1 for v in by_hash.values() if len(v) > 1)

    print('== 3. 数据缓存 ==')
    report['caches'] = {}
    for d, want in CACHES.items():
        n = len(list((ROOT / d).glob('*.npz')))
        report['caches'][d] = n
        print(f"  {'OK ' if n == want else '异常'} {d:28s} {n} 个 npz（应为 {want}）")
        if n != want:
            problems.append(f'缓存数量不符：{d} {n}≠{want}')

    print('== 4. 稿中写到的路径 ==')
    txt = MAIN.read_text(encoding='utf-8') + SUPP.read_text(encoding='utf-8')
    cited = sorted(set(re.findall(r'`((?:results|tools|src|config|data|models|figures|docs|provenance|verification|supp_data)/[^`\s]*)`', txt)))
    missing = []
    for c in cited:
        p = c.rstrip('/')
        base = DRAFT if p.startswith('supp_data') else ROOT
        pat = re.sub(r'\{[^}]*\}', '*', p)
        if not (glob.glob(str(base / pat)) or glob.glob(str(base / pat) + '*')):
            missing.append(c)
    imgs = re.findall(r'^!\[[^\]]*\]\(([^)]+)\)', txt, re.M)
    missing += [i for i in imgs if not (DRAFT / i).exists()]
    print(f'  稿中路径 {len(cited)} 个、图片 {len(imgs)} 张；找不到的：{missing or "无"}')
    problems += [f'稿中路径不存在：{m}' for m in missing]
    report['cited_paths'] = dict(n=len(cited), images=len(imgs), missing=missing)

    print('== 5. 稿件图片与原件 ==')
    r = subprocess.run([sys.executable, str(ROOT / 'tools/sync_draft_figures_20260930.py'), '--check'], capture_output=True, text=True)
    print('  ' + r.stdout.strip())
    if r.returncode:
        problems.append('稿件 figures/ 与原件不一致')

    report['problems'] = problems
    (ROOT / 'provenance/path_check_20261001.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n结论：' + ('全部通过' if not problems else f'{len(problems)} 个问题'))
    for p in problems:
        print('  - ' + p)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
