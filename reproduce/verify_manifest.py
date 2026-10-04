"""Check that every file listed in MANIFEST.tsv is present with the recorded size and SHA-256.

    python reproduce/verify_manifest.py
"""
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    rows = [l.split('\t') for l in (ROOT / 'MANIFEST.tsv').read_text().splitlines()[1:] if l.strip()]
    missing, bad = [], []
    for rel, size, digest in rows:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size != int(size) or sha(p) != digest:
            bad.append(rel)
    for x in missing:
        print('MISSING', x)
    for x in bad:
        print('CHANGED', x)
    print(f'files in manifest: {len(rows)}; missing: {len(missing)}; changed: {len(bad)}')
    sys.exit(1 if missing or bad else 0)


if __name__ == '__main__':
    main()
