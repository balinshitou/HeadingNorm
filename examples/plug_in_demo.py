"""Plug HeadingNorm into a trained network and check yaw equivariance (runs on CPU in a few seconds, no dataset needed).

    python examples/plug_in_demo.py

A trained ResNet18-YawAug checkpoint from models/ predicts velocity for synthetic gravity-aligned windows. Each window is
rotated about the vertical axis by psi; an exactly yaw-equivariant regressor F satisfies F(R x) = R F(x). The script prints
the mean violation |F(R x) - R F(x)| for the unmodified network and for the same network with plug-in HN.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from models import HeadingNorm, _unrotate          # noqa: E402
from sensors_evaluate import load                  # noqa: E402

HN = HeadingNorm()


def plug_in(net, x):
    """x: [B, 6, 200] gravity-aligned gyroscope (channels 0-2) and accelerometer (3-5) at 200 Hz."""
    x_canonical, phi = HN(x)                       # rotate each window into its canonical heading (closed form, no parameters)
    return _unrotate(net(x_canonical), phi)        # predict, then rotate the velocity back by the same angle


def rotate(x, psi):
    """Rotate the horizontal components of gyroscope and accelerometer by psi (rad)."""
    c, s = np.cos(psi), np.sin(psi)
    out = x.clone()
    for k in (0, 3):
        out[:, k], out[:, k + 1] = c * x[:, k] - s * x[:, k + 1], s * x[:, k] + c * x[:, k + 1]
    return out


def synthetic_windows(n=64, seed=0):
    """Walking-like windows: an asymmetric fore-aft acceleration along a random heading, vertical bounce, small gyro."""
    rng = np.random.default_rng(seed)
    t = np.arange(200) / 200.0
    x = np.zeros((n, 6, 200), np.float32)
    for i in range(n):
        f, h = rng.uniform(1.6, 2.2), rng.uniform(0, 2 * np.pi)
        fa = 1.5 * np.sin(2 * np.pi * f * t) + 0.6 * np.sin(4 * np.pi * f * t + 0.8)       # asymmetric along the walking axis
        la = 0.4 * np.sin(np.pi * f * t + rng.uniform(0, 2 * np.pi))
        x[i, 3] = np.cos(h) * fa - np.sin(h) * la + 0.1 * rng.standard_normal(200)
        x[i, 4] = np.sin(h) * fa + np.cos(h) * la + 0.1 * rng.standard_normal(200)
        x[i, 5] = 9.81 + 1.2 * np.sin(4 * np.pi * f * t) + 0.1 * rng.standard_normal(200)
        x[i, :3] = 0.3 * rng.standard_normal((3, 200))
    return torch.from_numpy(x)


def main():
    torch.manual_seed(0)
    net = load('Sensors_v4_resnet_yaw_s0', torch.device('cpu'))[0]   # ResNet18 trained with GN and random yaw augmentation
    x = synthetic_windows()
    with torch.no_grad():
        base_u, base_h = net(x), plug_in(net, x)
        print(f'{"psi (deg)":>9s}  {"unmodified |F(Rx)-RF(x)|":>26s}  {"plug-in HN |F(Rx)-RF(x)|":>26s}   (m/s, mean over 64 windows)')
        for k in range(1, 8):
            psi = 2 * np.pi * k / 8
            xr = rotate(x, psi)
            p = torch.full((len(x),), psi)
            eu = (net(xr) - _unrotate(base_u, p)).norm(dim=1).mean()
            eh = (plug_in(net, xr) - _unrotate(base_h, p)).norm(dim=1).mean()
            print(f'{45 * k:9d}  {eu:26.3e}  {eh:26.3e}')
    print('Plug-in HN makes the trained network yaw-equivariant up to float32 rounding; the unmodified network is not.')


if __name__ == '__main__':
    main()
