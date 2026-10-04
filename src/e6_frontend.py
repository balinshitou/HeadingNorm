"""E6: HeadingNorm with the third-moment sign rule removed.

The paper attributes exact yaw equivariance to a two-part construction: the
closed-form principal axis, and the third central moment that resolves which of
the two opposite directions along that axis is taken. Only the second part is
specific to this work, and the mixed-sensor PCA control cannot isolate it
because it changes the covariance source at the same time. This module keeps
everything else identical and removes only the sign rule.

Prediction registered before the runs: without the sign rule the frame is
defined modulo pi, so under an input rotation psi the axis flips whenever
theta + psi crosses pi/2. The canonicalized input is then the horizontal
signals negated, which is a different input to the backbone. Exact invariance
must therefore fail, and it must fail as a function of the angle rather than
uniformly. Accuracy may be largely unaffected, since the backbone can learn to
tolerate the ambiguity.

Frozen code is not touched: src/models.py and src/sensors_frontend.py keep the
hashes recorded in the existing run metadata, so every earlier run still
verifies. Only the pieces that must differ are reimplemented here; GlobalNorm
and the backbone are imported unchanged.
"""
from __future__ import annotations

import torch
from torch import nn

import models


class HeadingNormNoSign(nn.Module):
    """HeadingNorm without the third-moment sign selection.

    Identical to models.HeadingNorm except that theta is left on the principal
    axis modulo pi instead of being lifted to the full circle.
    """

    def forward(self, x):
        a = x[:, 3:5, :]
        ac = a - a.mean(dim=2, keepdim=True)
        C = torch.einsum('bit,bjt->bij', ac, ac) / ac.shape[2]
        cxx, cyy, cxy = C[:, 0, 0], C[:, 1, 1], C[:, 0, 1]
        theta = 0.5 * torch.atan2(2 * cxy, cxx - cyy)
        # models.HeadingNorm applies here:
        #     theta = theta + torch.where(skew < 0, torch.pi, 0.0)
        # That single term is the object of this ablation and is omitted.
        c, s = torch.cos(-theta), torch.sin(-theta)
        out = x.clone()
        for off in (0, 3):
            xx, yy = x[:, off, :], x[:, off + 1, :]
            out[:, off, :] = c[:, None] * xx - s[:, None] * yy
            out[:, off + 1, :] = s[:, None] * xx + c[:, None] * yy
        return out, theta


class NoSignBackbone(nn.Module):
    """Same wiring as models.NormalizedBackbone, with the ablated frontend."""

    def __init__(self, mu, sd):
        super().__init__()
        # Construct the backbone first so parameter initialisation order matches
        # the reference arm exactly under a shared seed.
        backbone = models.RoNINResNet18()
        self.hn = HeadingNormNoSign()
        self.gn = models.GlobalNorm(enabled=True, mu=mu, sd=sd)
        self.backbone = backbone

    def forward(self, x):
        canonical, theta = self.hn(x)
        velocity = self.backbone(self.gn(canonical))
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack((c * velocity[:, 0] - s * velocity[:, 1],
                            s * velocity[:, 0] + c * velocity[:, 1]), dim=1)


def build(frontend, mu, sd, arch='ronin_resnet18', pre=False, width=1.0):
    if frontend != 'heading_nosign':
        raise ValueError(frontend)
    if arch != 'ronin_resnet18' or pre or width != 1.0:
        raise ValueError('The sign ablation is registered only for the six-channel ResNet18 arm')
    if mu is None:
        raise ValueError('The ablation arm uses the same GlobalNorm as every other arm')
    return NoSignBackbone(mu, sd)
