## 表7 数据行
| ID | Provenance | Comparison (treatment − control) | Dataset | Treatment | Control | Difference (rel.) | Subject CI | Sequence CI | Improved | Verdict | Fixed convention (α = 0, ψ = 0): difference (rel.), verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| H1a′ | Trained by us | ResNet18-YawAug: Plug-in HN (8 α) − Unmodified (8 ψ) | TLIO-confirm (318) | 3.803 | 4.252 | −0.449 (−10.6%) | — | [−0.586, −0.330] | 294/318 seqs | **Pass** | −0.419 (−10.0%), Pass |
| H1a′ | Trained by us | ResNet18-YawAug: Plug-in HN (8 α) − Unmodified (8 ψ) | Phone-confirm (87, 4 subj.) | 10.608 | 11.205 | −0.597 (−5.3%) | [−0.726, −0.438] | [−0.768, −0.418] | 4/4 subj., 87/87 seqs | Pass by rule† | −0.478 (−4.3%), Pass |
| H1b′ | Trained by us | ResNet18-YawAug: FA-2 (4 α) − Unmodified (8 ψ) | TLIO-confirm (318) | 3.766 | 4.252 | −0.486 (−11.4%) | — | [−0.627, −0.361] | 296/318 seqs | **Pass** | −0.435 (−10.4%), Pass |
| H1b′ | Trained by us | ResNet18-YawAug: FA-2 (4 α) − Unmodified (8 ψ) | Phone-confirm (87, 4 subj.) | 10.554 | 11.205 | −0.651 (−5.8%) | [−0.812, −0.472] | [−0.834, −0.462] | 4/4 subj., 87/87 seqs | Pass by rule† | −0.648 (−5.8%), Pass |
| H2′ | Released | RoNIN ResNet: Plug-in HN (8 α) − Unmodified (8 ψ) | TLIO-confirm (318) | 3.150 | 3.325 | −0.175 (−5.3%) | — | [−0.252, −0.118] | 281/318 seqs | **Non-inferior** | −0.201 (−6.0%), Non-inferior |
| H2′ | Released | RoNIN ResNet: Plug-in HN (8 α) − Unmodified (8 ψ) | Phone-confirm (87, 4 subj.) | 10.042 | 10.224 | −0.182 (−1.8%) | [−0.266, −0.041] | [−0.297, −0.038] | 4/4 subj., 79/87 seqs | Non-inferior by rule† | −0.105 (−1.0%), Non-inferiority not shown |
| H3′ | Trained by us (training-time) | GN + HN − GN + MixPCA (8 ψ) | TLIO-confirm (318) | 4.047 | 4.912 | −0.866 (−17.6%) | — | [−1.086, −0.670] | 276/318 seqs | **Pass** | −0.935 (−18.8%), Pass |
| H3′ | Trained by us (training-time) | GN + HN − GN + MixPCA (8 ψ) | Phone-confirm (87, 4 subj.) | 10.534 | 12.480 | −1.946 (−15.6%) | [−2.478, −1.390] | [−2.455, −1.330] | 4/4 subj., 78/87 seqs | Pass by rule† | −2.183 (−17.2%), Pass |

## 表8 数据行
| Provenance | Frozen network | Unmodified ATE (8 ψ) | Plug-in HN ATE (8 α) | ATE difference (rel.) | ATE CI (level, family) | Improved seqs | Shape-error difference (rel.) | Shape-error CI (95%, descriptive) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Trained by us | ResNet18-YawAug | 4.252 | 3.803 | −0.449 (−10.6%) | [−0.586, −0.330] 99.375% (H1a′) | 294/318 | −0.526 (−21.6%) | [−0.637, −0.421] |
| Trained by us | Transformer-YawAug | 3.695 | 3.327 | −0.368 (−10.0%) | [−0.458, −0.284] 97.5% (cross-architecture) | 299/318 | −0.358 (−16.9%) | [−0.433, −0.287] |
| Trained by us | IMUNet-YawAug | 4.542 | 3.942 | −0.600 (−13.2%) | [−0.729, −0.481] 97.5% (cross-architecture) | 306/318 | −0.751 (−27.9%) | [−0.899, −0.611] |
| Released | RoNIN ResNet | 3.325 | 3.150 | −0.175 (−5.3%) | [−0.252, −0.118] 99.375% (H2′) | 281/318 | −0.184 (−10.7%) | [−0.227, −0.144] |

## 表S62 数据行
| Frozen network, test-time method | Dataset | Comparison | Treatment | Control | Difference (rel.) | Subject CI | Sequence CI | Improved seqs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | Fixed: T(α=0) − U(ψ=0) | 3.774 | 4.193 | −0.419 (−10.0%) | — | [−0.563, −0.281] | 200/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | T(α=0) − Ū | 3.774 | 4.252 | −0.478 (−11.2%) | — | [−0.582, −0.377] | 250/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū (primary) | 3.803 | 4.252 | −0.449 (−10.6%) | — | [−0.543, −0.361] | 294/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | Worst α − Ū | 3.861 | 4.252 | −0.391 (−9.2%) | — | [−0.487, −0.302] | 223/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | U(ψ=0) − Ū | 4.193 | 4.252 | −0.058 (−1.4%) | — | [−0.144, +0.023] | 182/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | T(α=0) − T̄ | 3.774 | 3.803 | −0.029 (−0.8%) | — | [−0.060, +0.003] | 171/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | TTA-8 − T̄ | 3.746 | 3.803 | −0.057 (−1.5%) | — | [−0.066, −0.048] | 284/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, RTE | 3.127 | 3.386 | −0.258 (−7.6%) | — | [−0.312, −0.207] | 294/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, literature ATE | 4.037 | 4.459 | −0.422 (−9.5%) | — | [−0.513, −0.336] | 300/318 |
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, shape error | 1.909 | 2.436 | −0.526 (−21.6%) | — | [−0.637, −0.421] | 271/318 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | Fixed: T(α=0) − U(ψ=0) | 10.712 | 11.191 | −0.478 (−4.3%) | [−0.765, −0.191] | [−0.650, −0.236] | 57/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − Ū | 10.712 | 11.205 | −0.493 (−4.4%) | [−0.686, −0.239] | [−0.632, −0.291] | 70/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū (primary) | 10.608 | 11.205 | −0.597 (−5.3%) | [−0.695, −0.484] | [−0.715, −0.461] | 87/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | Worst α − Ū | 10.712 | 11.205 | −0.493 (−4.4%) | [−0.686, −0.239] | [−0.632, −0.291] | 70/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | U(ψ=0) − Ū | 11.191 | 11.205 | −0.015 (−0.1%) | [−0.128, +0.128] | [−0.199, +0.162] | 49/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − T̄ | 10.712 | 10.608 | 0.104 (+1.0%) | [−0.001, +0.245] | [−0.031, +0.277] | 40/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | TTA-8 − T̄ | 10.520 | 10.608 | −0.089 (−0.8%) | [−0.115, −0.066] | [−0.107, −0.067] | 82/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, RTE | 9.232 | 9.473 | −0.241 (−2.5%) | [−0.308, −0.192] | [−0.282, −0.193] | 87/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, literature ATE | 10.424 | 11.040 | −0.617 (−5.6%) | [−0.695, −0.513] | [−0.736, −0.478] | 86/87 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, shape error | 5.032 | 5.913 | −0.880 (−14.9%) | [−1.049, −0.702] | [−1.148, −0.664] | 87/87 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | Fixed: T(α=0) − U(ψ=0) | 3.759 | 4.193 | −0.435 (−10.4%) | — | [−0.575, −0.302] | 202/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | T(α=0) − Ū | 3.759 | 4.252 | −0.493 (−11.6%) | — | [−0.596, −0.395] | 263/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | T̄ − Ū (primary) | 3.766 | 4.252 | −0.486 (−11.4%) | — | [−0.585, −0.393] | 296/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | Worst α − Ū | 3.774 | 4.252 | −0.478 (−11.2%) | — | [−0.581, −0.381] | 254/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | U(ψ=0) − Ū | 4.193 | 4.252 | −0.058 (−1.4%) | — | [−0.144, +0.023] | 182/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | T(α=0) − T̄ | 3.759 | 3.766 | −0.007 (−0.2%) | — | [−0.025, +0.010] | 166/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | TTA-8 − T̄ | 3.746 | 3.766 | −0.020 (−0.5%) | — | [−0.023, −0.017] | 270/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | T̄ − Ū, RTE | 3.096 | 3.386 | −0.290 (−8.6%) | — | [−0.348, −0.236] | 298/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | T̄ − Ū, literature ATE | 3.984 | 4.459 | −0.475 (−10.7%) | — | [−0.570, −0.384] | 300/318 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | T̄ − Ū, shape error | 1.874 | 2.436 | −0.561 (−23.0%) | — | [−0.677, −0.451] | 271/318 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | Fixed: T(α=0) − U(ψ=0) | 10.543 | 11.191 | −0.648 (−5.8%) | [−0.876, −0.425] | [−0.846, −0.413] | 59/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | T(α=0) − Ū | 10.543 | 11.205 | −0.662 (−5.9%) | [−0.892, −0.469] | [−0.794, −0.499] | 77/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | T̄ − Ū (primary) | 10.554 | 11.205 | −0.651 (−5.8%) | [−0.774, −0.519] | [−0.776, −0.506] | 87/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | Worst α − Ū | 10.555 | 11.205 | −0.650 (−5.8%) | [−0.832, −0.429] | [−0.769, −0.458] | 73/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | U(ψ=0) − Ū | 11.191 | 11.205 | −0.015 (−0.1%) | [−0.128, +0.128] | [−0.199, +0.162] | 49/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | T(α=0) − T̄ | 10.543 | 10.554 | −0.011 (−0.1%) | [−0.124, +0.093] | [−0.086, +0.067] | 35/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | TTA-8 − T̄ | 10.520 | 10.554 | −0.034 (−0.3%) | [−0.040, −0.026] | [−0.043, −0.023] | 77/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | T̄ − Ū, RTE | 9.188 | 9.473 | −0.285 (−3.0%) | [−0.343, −0.250] | [−0.332, −0.232] | 87/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | T̄ − Ū, literature ATE | 10.363 | 11.040 | −0.678 (−6.1%) | [−0.777, −0.565] | [−0.808, −0.529] | 86/87 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | T̄ − Ū, shape error | 4.981 | 5.913 | −0.931 (−15.8%) | [−1.118, −0.734] | [−1.213, −0.706] | 87/87 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | Fixed: T(α=0) − U(ψ=0) | 3.323 | 3.706 | −0.383 (−10.3%) | — | [−0.465, −0.307] | 242/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | T(α=0) − Ū | 3.323 | 3.695 | −0.373 (−10.1%) | — | [−0.449, −0.300] | 253/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū (primary) | 3.327 | 3.695 | −0.368 (−10.0%) | — | [−0.447, −0.294] | 299/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | Worst α − Ū | 3.368 | 3.695 | −0.327 (−8.9%) | — | [−0.404, −0.257] | 248/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | U(ψ=0) − Ū | 3.706 | 3.695 | 0.010 (+0.3%) | — | [−0.024, +0.044] | 137/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | T(α=0) − T̄ | 3.323 | 3.327 | −0.005 (−0.1%) | — | [−0.025, +0.016] | 162/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | TTA-8 − T̄ | 3.278 | 3.327 | −0.050 (−1.5%) | — | [−0.058, −0.042] | 294/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, RTE | 2.688 | 2.889 | −0.201 (−6.9%) | — | [−0.246, −0.159] | 298/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, literature ATE | 3.768 | 4.100 | −0.331 (−8.1%) | — | [−0.404, −0.264] | 284/318 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, shape error | 1.760 | 2.118 | −0.358 (−16.9%) | — | [−0.433, −0.287] | 275/318 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | Fixed: T(α=0) − U(ψ=0) | 9.246 | 9.537 | −0.290 (−3.0%) | [−0.643, −0.047] | [−0.511, −0.108] | 51/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − Ū | 9.246 | 9.636 | −0.389 (−4.0%) | [−0.528, −0.250] | [−0.522, −0.251] | 66/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū (primary) | 9.221 | 9.636 | −0.415 (−4.3%) | [−0.536, −0.314] | [−0.519, −0.321] | 85/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | Worst α − Ū | 9.271 | 9.636 | −0.364 (−3.8%) | [−0.484, −0.258] | [−0.492, −0.206] | 66/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | U(ψ=0) − Ū | 9.537 | 9.636 | −0.099 (−1.0%) | [−0.396, +0.169] | [−0.254, +0.111] | 51/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − T̄ | 9.246 | 9.221 | 0.026 (+0.3%) | [−0.042, +0.063] | [−0.048, +0.110] | 40/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | TTA-8 − T̄ | 9.143 | 9.221 | −0.078 (−0.8%) | [−0.094, −0.061] | [−0.100, −0.059] | 86/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, RTE | 8.234 | 8.406 | −0.173 (−2.1%) | [−0.248, −0.127] | [−0.200, −0.135] | 87/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, literature ATE | 9.042 | 9.458 | −0.416 (−4.4%) | [−0.516, −0.315] | [−0.513, −0.326] | 85/87 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, shape error | 4.859 | 5.307 | −0.449 (−8.5%) | [−0.595, −0.302] | [−0.546, −0.344] | 87/87 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | Fixed: T(α=0) − U(ψ=0) | 3.890 | 4.588 | −0.698 (−15.2%) | — | [−0.860, −0.547] | 228/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | T(α=0) − Ū | 3.890 | 4.542 | −0.652 (−14.4%) | — | [−0.780, −0.530] | 270/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū (primary) | 3.942 | 4.542 | −0.600 (−13.2%) | — | [−0.711, −0.494] | 306/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | Worst α − Ū | 4.047 | 4.542 | −0.495 (−10.9%) | — | [−0.601, −0.394] | 252/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | U(ψ=0) − Ū | 4.588 | 4.542 | 0.045 (+1.0%) | — | [−0.028, +0.121] | 151/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | T(α=0) − T̄ | 3.890 | 3.942 | −0.052 (−1.3%) | — | [−0.085, −0.022] | 159/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | TTA-8 − T̄ | 3.884 | 3.942 | −0.058 (−1.5%) | — | [−0.070, −0.047] | 279/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, RTE | 3.235 | 3.542 | −0.307 (−8.7%) | — | [−0.364, −0.253] | 297/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, literature ATE | 4.177 | 4.759 | −0.581 (−12.2%) | — | [−0.688, −0.480] | 303/318 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | T̄ − Ū, shape error | 1.944 | 2.695 | −0.751 (−27.9%) | — | [−0.899, −0.611] | 271/318 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | Fixed: T(α=0) − U(ψ=0) | 10.832 | 11.471 | −0.639 (−5.6%) | [−1.053, −0.245] | [−0.920, −0.370] | 61/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − Ū | 10.832 | 11.489 | −0.657 (−5.7%) | [−1.055, −0.389] | [−0.887, −0.444] | 71/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū (primary) | 10.826 | 11.489 | −0.663 (−5.8%) | [−1.017, −0.454] | [−0.820, −0.517] | 86/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | Worst α − Ū | 10.955 | 11.489 | −0.534 (−4.6%) | [−0.673, −0.395] | [−0.720, −0.376] | 69/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | U(ψ=0) − Ū | 11.471 | 11.489 | −0.018 (−0.2%) | [−0.193, +0.150] | [−0.213, +0.180] | 38/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − T̄ | 10.832 | 10.826 | 0.006 (+0.1%) | [−0.054, +0.066] | [−0.105, +0.121] | 40/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | TTA-8 − T̄ | 10.747 | 10.826 | −0.079 (−0.7%) | [−0.105, −0.060] | [−0.095, −0.061] | 78/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, RTE | 9.467 | 9.721 | −0.253 (−2.6%) | [−0.309, −0.193] | [−0.295, −0.219] | 87/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, literature ATE | 10.568 | 11.252 | −0.685 (−6.1%) | [−1.021, −0.485] | [−0.844, −0.537] | 84/87 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, shape error | 4.967 | 5.956 | −0.989 (−16.6%) | [−1.546, −0.628] | [−1.305, −0.745] | 86/87 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | Fixed: T(α=0) − U(ψ=0) | 3.152 | 3.354 | −0.201 (−6.0%) | — | [−0.283, −0.120] | 199/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | T(α=0) − Ū | 3.152 | 3.325 | −0.173 (−5.2%) | — | [−0.236, −0.115] | 218/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | T̄ − Ū (primary) | 3.150 | 3.325 | −0.175 (−5.3%) | — | [−0.229, −0.133] | 281/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | Worst α − Ū | 3.176 | 3.325 | −0.150 (−4.5%) | — | [−0.222, −0.085] | 201/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | U(ψ=0) − Ū | 3.354 | 3.325 | 0.028 (+0.8%) | — | [−0.053, +0.106] | 150/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | T(α=0) − T̄ | 3.152 | 3.150 | 0.002 (+0.1%) | — | [−0.047, +0.053] | 175/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | TTA-8 − T̄ | 3.109 | 3.150 | −0.041 (−1.3%) | — | [−0.054, −0.029] | 272/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | T̄ − Ū, RTE | 2.507 | 2.616 | −0.109 (−4.2%) | — | [−0.136, −0.084] | 286/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | T̄ − Ū, literature ATE | 3.684 | 3.869 | −0.185 (−4.8%) | — | [−0.234, −0.143] | 271/318 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | T̄ − Ū, shape error | 1.539 | 1.722 | −0.184 (−10.7%) | — | [−0.227, −0.144] | 275/318 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | Fixed: T(α=0) − U(ψ=0) | 10.089 | 10.193 | −0.105 (−1.0%) | [−0.549, +0.340] | [−0.982, +0.372] | 38/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − Ū | 10.089 | 10.224 | −0.135 (−1.3%) | [−0.505, +0.133] | [−0.608, +0.107] | 43/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū (primary) | 10.042 | 10.224 | −0.182 (−1.8%) | [−0.264, −0.097] | [−0.254, −0.078] | 79/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | Worst α − Ū | 10.635 | 10.224 | 0.411 (+4.0%) | [−0.406, +2.005] | [−0.241, +2.120] | 55/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | U(ψ=0) − Ū | 10.193 | 10.224 | −0.031 (−0.3%) | [−0.252, +0.196] | [−0.412, +0.452] | 48/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | T(α=0) − T̄ | 10.089 | 10.042 | 0.046 (+0.5%) | [−0.408, +0.369] | [−0.500, +0.305] | 32/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | TTA-8 − T̄ | 9.764 | 10.042 | −0.279 (−2.8%) | [−0.493, −0.065] | [−0.644, −0.073] | 75/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, RTE | 8.906 | 8.995 | −0.089 (−1.0%) | [−0.150, −0.052] | [−0.134, −0.022] | 79/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, literature ATE | 9.783 | 10.007 | −0.224 (−2.2%) | [−0.329, −0.118] | [−0.315, −0.123] | 80/87 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | T̄ − Ū, shape error | 4.686 | 4.974 | −0.288 (−5.8%) | [−0.385, −0.191] | [−0.406, −0.211] | 81/87 |

## 各角度均值（序列均值，m；本文网络先对4种子平均）
| Frozen network, method | Dataset | Unmodified by ψ = 0°, 45°, …, 315° | Test-time method by α |
|---|---|---|---|
| ResNet18-YawAug, Plug-in HN | TLIO-confirm (318) | 4.193 / 4.256 / 4.236 / 4.266 / 4.277 / 4.284 / 4.257 / 4.243 | 3.774 / 3.752 / 3.749 / 3.800 / 3.814 / 3.861 / 3.853 / 3.817 |
| ResNet18-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | 11.153 / 11.090 / 11.249 / 11.285 / 11.264 / 11.199 / 11.011 / 11.111 | 10.712 / 10.618 / 10.475 / 10.514 / 10.465 / 10.519 / 10.690 / 10.707 |
| ResNet18-YawAug, FA-2 | TLIO-confirm (318) | 4.193 / 4.256 / 4.236 / 4.266 / 4.277 / 4.284 / 4.257 / 4.243 | 3.759 / 3.774 / 3.766 / 3.765 |
| ResNet18-YawAug, FA-2 | Phone-confirm (87, 4 subj.) | 11.153 / 11.090 / 11.249 / 11.285 / 11.264 / 11.199 / 11.011 / 11.111 | 10.528 / 10.511 / 10.539 / 10.560 |
| Transformer-YawAug, Plug-in HN | TLIO-confirm (318) | 3.706 / 3.708 / 3.704 / 3.684 / 3.651 / 3.678 / 3.687 / 3.746 | 3.323 / 3.368 / 3.350 / 3.363 / 3.354 / 3.300 / 3.266 / 3.296 |
| Transformer-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | 9.523 / 9.589 / 9.666 / 9.705 / 9.744 / 9.518 / 9.548 / 9.454 | 9.215 / 9.249 / 9.196 / 9.124 / 9.200 / 9.201 / 9.140 / 9.132 |
| IMUNet-YawAug, Plug-in HN | TLIO-confirm (318) | 4.588 / 4.627 / 4.562 / 4.552 / 4.536 / 4.477 / 4.463 / 4.533 | 3.890 / 3.837 / 3.854 / 3.916 / 3.967 / 4.014 / 4.047 / 4.014 |
| IMUNet-YawAug, Plug-in HN | Phone-confirm (87, 4 subj.) | 11.448 / 11.418 / 11.337 / 11.466 / 11.599 / 11.386 / 11.495 / 11.533 | 10.807 / 10.595 / 10.606 / 10.867 / 10.910 / 10.880 / 10.884 / 10.859 |
| RoNIN ResNet (released), Plug-in HN | TLIO-confirm (318) | 3.354 / 3.325 / 3.281 / 3.283 / 3.226 / 3.336 / 3.420 / 3.377 | 3.152 / 3.155 / 3.174 / 3.127 / 3.176 / 3.160 / 3.162 / 3.095 |
| RoNIN ResNet (released), Plug-in HN | Phone-confirm (87, 4 subj.) | 10.245 / 10.528 / 9.938 / 10.143 / 10.270 / 10.579 / 10.486 / 9.978 | 10.071 / 10.948 / 10.630 / 9.845 / 9.680 / 10.009 / 9.837 / 9.815 |
