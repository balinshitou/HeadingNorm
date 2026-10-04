"""Export the three small inputs that the one-command reproduction needs from the evaluation caches (2026-10-04).

The evaluation caches (data/eval/, rebuilt from the public datasets) are not redistributed. Three items of the paper
read them directly; this script stores what they use, so that reproduce/run_all.py runs from the archive alone:

  results/data_stats_20261004/phone_test_devices.json   Supplementary Table S4: sequences, duration and ground-truth
                                                          path length of the phone test data per device
  results/figure_inputs_20261004/fig02_window.npz        Figure 2: the one 1 s window of horizontal acceleration
                                                          (the window tools/mst_figures.py:real_window selects)
  results/figure_inputs_20261004/fig07_tlio_confirm_sequence.npz
                                                          Figure 7: timestamps and ground truth of the one TLIO-confirm
                                                          sequence drawn (the predicted trajectories are archived in
                                                          results/fig8_shape_example_20261001/)

Usage: python tools/archive_small_inputs_20261004.py [--check]   (needs data/eval/benchmark, imunet_owndata, confirm_tlio)
--check only compares the archived files with what the caches give and writes nothing (exit status 1 if they differ).
"""
import glob
import json
import sys
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def phone_devices():
    dev = defaultdict(lambda: dict(sequences=0, minutes=0.0, km=0.0))
    files = sorted(glob.glob(str(ROOT / 'data/eval/imunet_owndata/*.npz')))
    assert files, 'phone test caches not found in data/eval/imunet_owndata (tools/build_imunet_cache_20260913.py)'
    for f in files:
        z = np.load(f)
        q = str(z['seq']); t = np.asarray(z['tm'], float); gt = np.asarray(z['gt'], float)
        d = re.search(r'_(S10|S21|Tango|Xiaomi)_', q).group(1)
        dev[d]['sequences'] += 1
        dev[d]['minutes'] += (t[-1] - t[0]) / 60
        dev[d]['km'] += float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()) / 1000
    order = ('S10', 'S21', 'Tango', 'Xiaomi')
    return dict(source='data/eval/imunet_owndata/*.npz', definition='duration = last minus first timestamp; path length = '
                'sum of ground-truth step lengths', devices={k: dev[k] for k in order})


def fig02_window():
    for path in sorted((ROOT / 'data/eval/benchmark').glob('ronin__*.npz')):     # identical to mst_figures.real_window
        with np.load(path, allow_pickle=False) as z:
            if str(z['split']) != 'unseen':
                continue
            feat = np.asarray(z['feat'], float)
            seq = str(z['seq'])
        j = len(feat) // 2
        return dict(window=feat[j:j + 200, 3:5], seq=np.array(seq), start=np.array(j), source=np.array(str(path.relative_to(ROOT))))
    raise SystemExit('RoNIN evaluation caches not found in data/eval/benchmark (tools/rebuild_caches.py)')


def fig07_sequence():
    seq = json.load(open(ROOT / 'results/fig8_shape_example_20261001/provenance.json'))['sequence']
    path = ROOT / f'data/eval/confirm_tlio/tlio_c__{seq}.npz'
    assert path.exists(), f'TLIO-confirm cache {path.relative_to(ROOT)} not found (tools/build_confirm_caches_20260913.py)'
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ('tm', 'te', 'gt', 'seq')}, path


CHECK = '--check' in sys.argv
DIFF = []


def save_npz(out, arrays):
    if out.exists():
        old = np.load(out)
        if set(old.files) == set(arrays) and all(np.array_equal(old[k], np.asarray(arrays[k])) for k in arrays):
            print(f'[unchanged] {out.relative_to(ROOT)}')
            return
    DIFF.append(out)
    if CHECK:
        print(f'[DIFFERS] {out.relative_to(ROOT)}')
        return
    np.savez(out, **arrays)
    print(f'[written] {out.relative_to(ROOT)}')


def main():
    out1 = ROOT / 'results/data_stats_20261004/phone_test_devices.json'
    out2 = ROOT / 'results/figure_inputs_20261004/fig02_window.npz'
    for p in (out1, out2):
        p.parent.mkdir(parents=True, exist_ok=True)
    new1 = json.dumps(phone_devices(), indent=1) + '\n'
    same = out1.exists() and out1.read_text() == new1
    print(f"[{'unchanged' if same else ('DIFFERS' if CHECK else 'written')}] {out1.relative_to(ROOT)}")
    if not same:
        DIFF.append(out1)
        if not CHECK:
            out1.write_text(new1)
    save_npz(out2, fig02_window())
    arrays, src = fig07_sequence()
    save_npz(ROOT / 'results/figure_inputs_20261004/fig07_tlio_confirm_sequence.npz', dict(arrays, source=np.array(str(src.relative_to(ROOT)))))
    sys.exit(1 if CHECK and DIFF else 0)


if __name__ == '__main__':
    main()
