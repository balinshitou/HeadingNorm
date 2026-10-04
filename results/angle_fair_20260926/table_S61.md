| Network | Test set | Seqs / subjects | Unmodified, ψ=0 | Unmodified, mean over 8 ψ | Plug-in HN, α=0 | Plug-in HN, mean over 8 α | Mean-α HN − mean-ψ Unmodified [CI] | Seqs improved (mean vs. mean) | α=0 − mean-α HN [95% CI] | Per-seq. range over α, median (% of ATE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RoNIN ResNet (released) | RoNIN-seen | 32 / 31 | 3.687 | 3.621 | 3.527 | 3.550 | **−0.071 (−2.0%) [−0.119, −0.038]** (95.00%) | 26/32 | −0.022 (−0.6%) [−0.132, +0.084] | 0.617 m (22.8%) |
| RoNIN ResNet (released) | RoNIN-unseen | 32 / 15 | 5.309 | 5.387 | 5.041 | 4.911 | **−0.475 (−8.8%) [−1.044, −0.127]** (98.33%) | 30/32 | **+0.130 (+2.6%) [+0.015, +0.260]** | 0.741 m (20.9%) |
| RoNIN ResNet (released) | RIDI | 94 / 11 | 2.822 | 2.885 | 2.741 | 2.762 | **−0.123 (−4.3%) [−0.179, −0.072]** (98.33%) | 87/94 | −0.021 (−0.8%) [−0.100, +0.036] | 0.519 m (23.1%) |
| RoNIN ResNet (released) | TLIO-test | 36 / — | 4.146 | 4.115 | 3.949 | 3.989 | **−0.126 (−3.1%) [−0.211, −0.057]** (98.33%) | 30/36 | −0.040 (−1.0%) [−0.201, +0.103] | 0.909 m (29.2%) |
| ResNet18-YawAug (ours, 4 seeds) | RoNIN-seen | 32 / 31 | 5.009 | 4.944 | 4.796 | not run | α=0: **−0.147 (−3.0%) [−0.255, −0.046]** (95.00%) | 24/32 | — | — |
| ResNet18-YawAug (ours, 4 seeds) | RoNIN-unseen | 32 / 15 | 6.195 | 5.876 | 5.246 | not run | α=0: **−0.630 (−10.7%) [−0.979, −0.320]** (95.00%) | 28/32 | — | — |
| ResNet18-YawAug (ours, 4 seeds) | RIDI | 94 / 11 | 3.122 | 3.172 | 2.973 | not run | α=0: **−0.199 (−6.3%) [−0.263, −0.138]** (95.00%) | 83/94 | — | — |

verdict = PASS (3/3 primary sets pass, 0 reversed)
