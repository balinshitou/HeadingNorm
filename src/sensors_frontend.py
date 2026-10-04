"""Isolated Sensors revision frontends; existing frozen model code is unchanged.

The mixed-sensor PCA control follows EqNIO's PCA *construction*, not its full
network or training recipe. Differences: deterministic exact 2x2 eigensolver
on CPU instead of randomized pca_lowrank; native gyro3/acc3 channel order;
the common RoNIN ResNet18, GlobalNorm and 2D velocity objective. No moment-based
sign correction is used. These differences must accompany reported results.
Reference: RoyinaJayanth/EqNIO, commit 97b7a60a5e5cc8f0132bba343c83de114211f827,
TLIO-master/src/network/model_resnet_frame_pca.py.
"""
from __future__ import annotations

import torch
from torch import nn
import models


def rotate_batch(x, y, angles):
    """Rotate both horizontal vector pairs and the corresponding velocity labels."""
    c, s = angles.cos(), angles.sin()
    out = x.clone()
    for off in (0, 3):
        out[:, off] = c[:, None] * x[:, off] - s[:, None] * x[:, off + 1]
        out[:, off + 1] = s[:, None] * x[:, off] + c[:, None] * x[:, off + 1]
    target = torch.stack((c * y[:, 0] - s * y[:, 1],
                          s * y[:, 0] + c * y[:, 1]), dim=1)
    return out, target


class MixedPCAFrame(nn.Module):
    """Jointly center 2T horizontal gyro/acc samples, then extract two PCA axes."""

    def forward(self, x):
        samples = torch.cat((x[:, :2], x[:, 3:5]), dim=2)
        centered = samples - samples.mean(dim=2, keepdim=True)
        covariance = centered @ centered.transpose(1, 2) / samples.shape[2]
        # MPS has no native eigh; explicitly fix the solver device for all runs.
        _, axes = torch.linalg.eigh(covariance.to('cpu'))
        axes = axes.flip(-1).to(x.device)  # columns: decreasing eigenvalue order
        frame = axes.transpose(1, 2)
        out = x.clone()
        for off in (0, 3):
            out[:, off:off + 2] = frame @ x[:, off:off + 2]
        return out, axes


class MixedPCABackbone(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        # Construct backbone first, matching initialization order in models.build.
        backbone = models.RoNINResNet18()
        self.frame = MixedPCAFrame()
        self.gn = models.GlobalNorm(True, mu, sd)
        self.backbone = backbone

    def forward(self, x):
        canonical, axes = self.frame(x)
        velocity = self.backbone(self.gn(canonical))
        return (axes @ velocity.unsqueeze(-1)).squeeze(-1)


def build(frontend, mu, sd, arch='ronin_resnet18', pre=False, width=1.0):
    if frontend == 'mixed_pca':
        if arch != 'ronin_resnet18' or pre:
            raise ValueError('Mixed PCA is registered only for the six-channel ResNet18 control')
        return MixedPCABackbone(mu, sd)
    if frontend not in ('none', 'heading'):
        raise ValueError(frontend)
    return models.build(arch, pre=pre, width=width, hnorm=frontend == 'heading',
                        gnorm=mu is not None, gnorm_mu=mu, gnorm_sd=sd)
