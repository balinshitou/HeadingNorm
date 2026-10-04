"""E6, analytic part: does the frame angle itself track an input rotation?

Exact yaw equivariance of the pipeline reduces to one property of the frontend:
under an input rotation by psi the selected reference direction must satisfy
phi(R_psi X) = phi(X) + psi (mod 2*pi). This is a property of the frame
construction alone, so it can be checked without a trained network and holds or
fails independently of the weights.

With the third-moment sign rule phi is lifted to the full circle and the
identity holds. Without it phi is only defined modulo pi, so the identity holds
only while theta + psi stays inside one branch and fails once it crosses pi/2.
This script measures how often each case occurs on the same 72 fixed evaluation
windows used by the rotation diagnostics.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import models
import e6_frontend

DIAGNOSTICS = ROOT / 'results/sensors_v4/diagnostics.json'
OUT = ROOT / 'results/e6_sign_ablation'
ANGLES = [2 * np.pi * k / 8 for k in range(8)]
TOLERANCE = 1e-4  # radians; float32 frame angles agree far below this when exact


def rotate_horizontal(x, psi):
    c, s = float(np.cos(psi)), float(np.sin(psi))
    out = x.clone()
    for off in (0, 3):
        xx, yy = x[:, off, :], x[:, off + 1, :]
        out[:, off, :] = c * xx - s * yy
        out[:, off + 1, :] = s * xx + c * yy
    return out


def wrap(value, period):
    """Signed distance of `value` to the nearest multiple of `period`."""
    return np.abs((value + period / 2) % period - period / 2)


def main():
    sources = json.loads(DIAGNOSTICS.read_text())['rotation_window_sources']
    windows = []
    for entry in sources:
        with np.load(ROOT / entry['file'], allow_pickle=False) as z:
            feat = np.asarray(z['feat'], np.float32)
        start = int(entry['start'])
        windows.append(feat[start:start + 200].T)
    x = torch.from_numpy(np.stack(windows)).float()

    signed = models.HeadingNorm()
    unsigned = e6_frontend.HeadingNormNoSign()
    report = []
    with torch.no_grad():
        base_signed = signed(x)[1].numpy()
        base_unsigned = unsigned(x)[1].numpy()
        for psi in ANGLES:
            rotated = rotate_horizontal(x, psi)
            got_signed = signed(rotated)[1].numpy()
            got_unsigned = unsigned(rotated)[1].numpy()
            # exact equivariance requires the frame to advance by exactly psi
            err_signed = wrap(got_signed - base_signed - psi, 2 * np.pi)
            err_unsigned = wrap(got_unsigned - base_unsigned - psi, 2 * np.pi)
            # the unsigned frame is only ever defined modulo pi
            err_unsigned_mod_pi = wrap(got_unsigned - base_unsigned - psi, np.pi)
            report.append(dict(
                angle_rad=float(psi), angle_deg=round(float(np.degrees(psi)), 1),
                windows=len(windows),
                signed_max_error_rad=float(err_signed.max()),
                signed_inconsistent=int((err_signed > TOLERANCE).sum()),
                unsigned_max_error_rad=float(err_unsigned.max()),
                unsigned_inconsistent=int((err_unsigned > TOLERANCE).sum()),
                unsigned_max_error_mod_pi_rad=float(err_unsigned_mod_pi.max())))

    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        tolerance_rad=TOLERANCE, angles=len(ANGLES), windows=len(windows),
        window_source=str(DIAGNOSTICS.relative_to(ROOT)),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Frame-angle property only; no network weights are involved, so the '
             'result is independent of training.',
        rows=report)
    (OUT / 'analytic_frame_check.json').write_text(json.dumps(payload, indent=1))

    print(f'{len(windows)} fixed windows, tolerance {TOLERANCE} rad\n')
    print(f"{'yaw':>6s} {'with sign rule: inconsistent':>30s} {'without: inconsistent':>24s} "
          f"{'without, max err mod pi':>25s}")
    for row in report:
        print(f"{row['angle_deg']:5.0f}° {row['signed_inconsistent']:>12d}/{row['windows']:<17d} "
              f"{row['unsigned_inconsistent']:>10d}/{row['windows']:<12d} "
              f"{row['unsigned_max_error_mod_pi_rad']:25.2e}")
    total = sum(r['unsigned_inconsistent'] for r in report)
    n = sum(r['windows'] for r in report)
    print(f"\nwithout the sign rule: {total}/{n} = {100 * total / n:.1f}% of "
          f"(window, angle) pairs have a frame that does not track the rotation")
    print(f"with the sign rule:    {sum(r['signed_inconsistent'] for r in report)}/{n}")
    print(f"[written] {(OUT / 'analytic_frame_check.json').relative_to(ROOT)}")


if __name__ == '__main__':
    main()
