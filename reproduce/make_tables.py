"""Regenerate every table of the main text and the Supplementary Materials, and the nine numerical tables distributed with
them as supp_data/*.csv, from the archived results, and compare each cell with the manuscript.

    python reproduce/make_tables.py                 # Tables 1-9, S1-S28 and supp_data/*.csv
    python reproduce/make_tables.py 3 S16           # selected tables

For each table the generated version is written to reproduce/output/tables/Table_<id>.md and .csv (the CSV files to
reproduce/output/tables/supp_data/). Row and column labels
are fixed in the code; every number is recomputed from results/ (per-sequence records wherever they exist). The
comparison with the manuscript (paper/*.md) is cell by cell: a cell agrees when the generated text is identical, or when
every number in it agrees with the manuscript at the precision shown there. Summary: reproduce/output/tables_summary.txt.
Exit status 1 if any cell disagrees or any table fails to build.
"""
import csv
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tables_lib as L                     # noqa: E402
import tables_main                         # noqa: E402,F401  (registers Tables 1-9)
import tables_supp                         # noqa: E402,F401  (registers Tables S1-S28)
import tables_suppdata                     # noqa: E402,F401  (registers the nine supp_data/*.csv files)
from tables_registry import TABLES        # noqa: E402

PAPER = {'main': L.ROOT / 'paper/HeadingNorm_Main_EN.md', 'supp': L.ROOT / 'paper/HeadingNorm_Supplement_EN.md'}
SUPP_DATA = L.ROOT / 'supp_data'
if not PAPER['main'].exists():               # authors' working tree: the manuscript lives in the draft folder
    D = L.ROOT / 'paper/draft_zh_20260927'
    PAPER = {'main': D / 'HeadingNorm_Main_EN_20261001.md', 'supp': D / 'HeadingNorm_Supplement_EN_20261001.md'}
    SUPP_DATA = D / 'supp_data_en'
OUT = HERE / 'output/tables'


def split_row(s):
    return [c.strip() for c in re.split(r'(?<!\\)\|', s)[1:-1]]


def manuscript_tables():
    out = {}
    for doc, path in PAPER.items():
        lines = path.read_text().splitlines()
        for i, s in enumerate(lines):
            m = re.match(r'\*\*Table (S?\d+)\.\*\*', s)
            if not m:
                continue
            j = i + 1
            while not lines[j].startswith('|'):
                j += 1
            rows = []
            while j < len(lines) and lines[j].startswith('|'):
                rows.append(split_row(lines[j]))
                j += 1
            out[m.group(1)] = (rows[0], rows[2:])
    for f in sorted(SUPP_DATA.glob('*.csv')):          # numerical tables that accompany the Supplementary Materials
        rows = list(csv.reader(open(f, encoding='utf-8-sig')))
        out[f'supp_data/{f.name}'] = (rows[0], rows[1:])
    return out


_SCI = re.compile(r'\$?([0-9.]+)\s*\\times\s*10\^\{([−-]?\d+)\}\$?')
_NUM = re.compile(r'\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?')


def numbers(cell):
    """(value, decimals, exponent) of every number in a cell; text without the numbers."""
    s = cell.replace('−', '-').replace('\\*', '*')
    s = _SCI.sub(lambda m: f'{m.group(1)}e{m.group(2)}', s)
    vals, text, last = [], [], 0
    for m in _NUM.finditer(s):
        t = m.group(0).replace(',', '')
        neg = m.start() > 0 and s[m.start() - 1] == '-'
        text.append(s[last:m.start() - (1 if neg else 0)]); last = m.end()
        mant, _, ex = t.partition('e')
        d = len(mant.split('.')[1]) if '.' in mant else 0
        vals.append(((-1 if neg else 1) * float(t), d, int(ex) if ex else None))
    text.append(s[last:])
    return vals, ''.join(text).replace('+', '')


def agree(gen, ms):
    if gen == ms:
        return True
    vg, tg = numbers(gen)
    vm, tm = numbers(ms)
    if tg != tm or len(vg) != len(vm):                     # text, signs and bold markers must be identical
        return False
    for (g, _, _), (m, d, ex) in zip(vg, vm):
        tol = 0.5 * 10 ** ((ex or 0) - d) * 1.0001 + 1e-12
        if abs(g - m) > tol:
            return False
    return True


def write(tid, header, rows):
    if tid.startswith('supp_data/'):
        (OUT / 'supp_data').mkdir(parents=True, exist_ok=True)
        with open(OUT / tid, 'w', newline='', encoding='utf-8-sig') as fh:
            csv.writer(fh).writerows([header] + rows)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    sep = ['---' if k == 0 else '---:' for k in range(len(header))]
    md = ['| ' + ' | '.join(r) + ' |' for r in [header, sep] + rows]
    (OUT / f'Table_{tid}.md').write_text(f'**Table {tid}.**\n\n' + '\n'.join(md) + '\n')
    with open(OUT / f'Table_{tid}.csv', 'w', newline='') as fh:
        csv.writer(fh).writerows([[c.replace('**', '') for c in r] for r in [header] + rows])


def main():
    want = sys.argv[1:] or list(TABLES)
    ms = manuscript_tables()
    lines, n_cells, n_bad, n_exact, failed = [], 0, 0, 0, []
    for tid in want:
        try:
            header, rows = TABLES[tid]()
        except Exception:                                  # noqa: BLE001
            failed.append(tid)
            lines.append(f'Table {tid}: FAILED\n' + traceback.format_exc())
            continue
        write(tid, header, rows)
        mh, mr = ms[tid]
        bad = []
        if len(rows) != len(mr) or len(header) != len(mh):
            bad.append(f'shape {len(rows)}x{len(header)} vs manuscript {len(mr)}x{len(mh)}')
        for j, (g, m) in enumerate(zip(header, mh)):
            if g != m:
                bad.append(f'header col {j + 1}: generated {g!r} | manuscript {m!r}')
        cells = 0
        for i, (gr, mrow) in enumerate(zip(rows, mr)):
            for j, (g, m) in enumerate(zip(gr, mrow)):
                cells += 1
                n_exact += g == m
                if not agree(g, m):
                    bad.append(f'row {i + 1} col {j + 1} ({gr[0][:40]}): generated {g!r} | manuscript {m!r}')
        n_cells += cells; n_bad += len(bad)
        lines.append(f'Table {tid}: {cells} cells, ' + ('all agree' if not bad else f'{len(bad)} DISAGREE'))
        lines += ['    ' + b for b in bad]
    lines.append(f'TOTAL: {len(want)} tables, {n_cells} cells compared ({n_exact} identical text, {n_cells - n_exact - n_bad} equal at the '
                 f'displayed precision), {n_bad} disagreements, {len(failed)} tables failed'
                 + (f' ({", ".join(failed)})' if failed else ''))
    text = '\n'.join(lines)
    print(text)
    (HERE / 'output').mkdir(exist_ok=True)
    (HERE / 'output/tables_summary.txt').write_text(text + '\n')
    sys.exit(1 if n_bad or failed else 0)


if __name__ == '__main__':
    main()
