"""Reproduce every table and figure of the paper from the archived results, with one command:

    python reproduce/run_all.py

1. reproduce/verify_manifest.py  every file of the repository present with the SHA-256 recorded in MANIFEST.tsv
2. reproduce/make_tables.py      Tables 1-9, S1-S31 and the nine supp_data/*.csv regenerated and compared cell by cell
                                 with the manuscript (paper/*.md, supp_data/)
3. reproduce/make_figures.py     Figures 1-7 and S1-S3 redrawn; source data compared byte by byte, images pixel by pixel

Generated tables:  reproduce/output/tables/      Generated figures:  reproduce/output/figures/
Combined report:   reproduce/output/REPORT.txt   Exit status 0 means that everything agrees with the manuscript.
No dataset is needed; the raw-data route is described in README.md ("Starting from the raw datasets").
"""
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduce/output'
STEPS = [('Repository integrity (MANIFEST.tsv)', 'reproduce/verify_manifest.py'),
         ('Tables', 'reproduce/make_tables.py'),
         ('Figures', 'reproduce/make_figures.py')]


def versions():
    out = [f'Python {sys.version.split()[0]} on {platform.platform()}']
    for m in ('numpy', 'scipy', 'torch', 'matplotlib'):
        try:
            out.append(f'{m} {__import__(m).__version__}')
        except ImportError:
            out.append(f'{m} not installed')
    return ', '.join(out)


def main():
    OUT.mkdir(exist_ok=True)
    report = [f'HeadingNorm reproduction report ({time.strftime("%Y-%m-%d %H:%M")})', versions(), '']
    status = 0
    for title, script in STEPS:
        if script.endswith('verify_manifest.py') and not (ROOT / 'MANIFEST.tsv').exists():
            report += [f'## {title}: skipped (no MANIFEST.tsv in this working tree)', '']
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
        status |= r.returncode
        body = [l for l in r.stdout.splitlines() if l.strip()]
        if script.endswith('make_tables.py'):
            body = [l for l in body if not l.endswith('all agree')]          # list only tables with a disagreement
        report += [f'## {title} ({script}): exit {r.returncode}, {time.time() - t0:.0f} s'] + body[-60:] + ['']
        if r.returncode and r.stderr.strip():
            report += ['stderr (last lines):'] + r.stderr.strip().splitlines()[-5:] + ['']
        print(f'{title}: {"ok" if r.returncode == 0 else "SEE REPORT"}', flush=True)
    report.append('RESULT: ' + ('every table and figure agrees with the manuscript' if status == 0 else 'see the disagreements above'))
    (OUT / 'REPORT.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))
    sys.exit(1 if status else 0)


if __name__ == '__main__':
    main()
