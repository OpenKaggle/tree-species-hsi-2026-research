# Campaign log

All times are UTC unless another timezone is stated. Test labels, private
datasets, and hand-authored test predictions are not used.

## Competition state

- Competition: Hyperspectral Tree Species Identification Challenge 2026
- Competition ID / slug: `157321` / `tree-species-hsi-2026`
- Rules accepted: 2026-09-08 UTC, through the official Kaggle join dialog
- Team: Jiayi Du (`jahyee`), team leader
- Official final deadline: 2026-09-25 04:00 UTC / 2026-09-25 12:00 Beijing
- New entrant deadline: not separately configured by the Kaggle API
- Team merger deadline: not separately configured by the Kaggle API
- Submission allowance: 5 per day; stricter organizer team limit of 5 is used
- Evaluation: overall accuracy, average accuracy as tie-breaker

## Experiments

| Name | Change | Validation | OA | AA | Runtime | Seed | Model SHA-256 | Decision |
|---|---|---|---:|---:|---:|---:|---|---|
| `sam_v1` | Spectral-angle class centroids; raw L2 spectra | 2-fold per-class connected-region OOF | 0.36169 | 0.37566 | 46.1 s | 20260909 | `b38630a8116ce21da412ed323817551d05ab5113851410302ee5971f431f7e82` | Submitted as the format/cross-region baseline |
| `lightgbm_v1` | LightGBM, 152 spectral/shape features; model family is the only submission-level change | Same 2-fold connected-region OOF | 0.62085 | 0.57243 | 732 s training; 1782.9 s test inference | 20260909 | `d6540b24115d8c800df0eeedfb11b2c57bd0776109bc106bd3cd944b3a7c1f4a` | Passed all gates and improved the public score by 41.64% |
| `lightgbm_full_small_control` | Full v1 features, 5000 samples/class, 160 trees | Same 2-fold connected-region OOF | 0.60286 | 0.57514 | 286.6 s | 20260909 | `da56f48ca2b748b42e016ee9c7f362d283f3073727052161f6b9f4c170dfb0bf` | Control only; no submission |
| `lightgbm_robust_v2` | Remove raw intensity and summaries; full-spectrum SNV plus first differences, same 5000/160 control budget | Same 2-fold connected-region OOF | 0.59076 | 0.57473 | 259.7 s | 20260909 | `30ee05b73574542df9fef6024899f8204bcdfc85a44b79241ac37eafe10c5ac4` | Rejected: lower OA/AA and larger fold ranges than the matched full-feature control |
| `lightgbm_spatial_r1_v3` | Replace each spectrum with its exact 3x3 spatial mean; otherwise matched to the 5000/160 full-feature control | Same 2-fold connected-region OOF | 0.68814 | 0.62649 | 195.8 s training; 909.5 s test inference | 20260909 | `adcaea3ff353e0dfa163204a6a806bf4c1ada25a09381e0b41f1e99a81679058` | Submitted; public score +11.60% over v1 and official rank improved 34 to 31 |
| `lightgbm_spatial_r1_cap10k_v4` | Increase per-class cap 5000 to 10000; radius 1, 160 trees, features, folds, and seed fixed | Same 2-fold connected-region OOF | 0.69317 | 0.61906 | 158.6 s | 20260909 | `4864fe4b7c9707fb23e4b394a20f4879397bc5e936380cc28648647d5ef80922` | Rejected and frozen: OA +0.00503 but AA -0.00743, failing the preregistered both-must-improve gate; no submission |
| `phase2_pca_local_texture` | Stage2-only fold-fit PCA24 + exact 3x3 mean/std + first spectral differences; symmetric low-shot LightGBM | 4-fold 128x128 spatial-tile OOF; 170 labels, 10/class | 0.22941 | 0.22941 | 398.7 s validation; 395.1 s inference | 20260909 | `6f0f4cfade6bc17aeaf705d6935b2fe0069bc6262caabaac327be100e3457c95` | Passed all frozen offline gates and earned exactly one Phase 2 submission; public 0.06404 triggered the frozen seal rule |

LightGBM fold OA values were 0.6434 and 0.5978; fold AA range was 0.00719.
The deterministic unlabeled-domain audit found mean per-band shifts of 0.0307
and 0.1474 train-scene standard deviations for test scenes 1 and 2. This did
not support spending a submission on simple brightness calibration or SNV.
The matched 5000-sample/160-tree ablation also rejected robust-only features:
OA fell by 0.01210 and AA by 0.00040, while the OA/AA fold ranges worsened from
0.05578/0.01321 to 0.06816/0.02914. No Kaggle slot was spent on this candidate.
The 3x3 spatial-mean experiment was the next method family and passed cleanly:
fold OA values were 0.71010 and 0.66573 (range 0.04438), improving both primary
OA and tie-break AA without using neighboring labels.
The sole cap-size follow-up held every other setting fixed. Its fold OA/AA
ranges were 0.04122/0.01652, but because AA fell from 0.62649 to 0.61906 it
failed the fixed gate. This direction is frozen without generating test
predictions or spending a Kaggle submission.

## Submissions

| Kaggle ref | File | Submitted | Code revision | File SHA-256 | Status | Public score | Interpretation |
|---:|---|---|---|---|---|---:|---|
| 56108130 | `sam_v1.csv` | 2026-09-08 22:36:44.777 | `d95b8e3905ddfef8a653dc01e4290ab4389add6c` | `6d4385be35d38f97c3cd8f0adc79f11b5cf4b53bb7d6ca659d5a85454bbbf363` | COMPLETE | 0.06571 | Valid format and end-to-end path; poor cross-region transfer, so do not repeat this model family unchanged |
| 56108653 | `lightgbm_v1.csv` | 2026-09-08 23:25:44.213 | `e64c6f4` | `90dd37d7359f8bfb5f0f9a91bfcb979bf52b3dcae4395999f4e06cd80b257ab4` | COMPLETE | 0.09307 | 41.64% relative improvement; official leaderboard rank 34 at 2026-09-08 23:28 UTC |
| 56120914 | `lightgbm_spatial_r1_v3.zip` | 2026-09-09 11:04:27.807 | `6c84550` | CSV: `6c14f7eefb48e7ec117e2dbe5fd3205242828f8a5eac33dc1fb7e1fd8f0f48c6`; ZIP: `c99417a976a6c5ff50c3a4497b21b8fc43ebc49ed8de5ce972d040a477a4f3d3` | COMPLETE | 0.10387 | 11.60% over v1; official leaderboard rank 31 at 2026-09-09 11:07 UTC |
| 56503530 | `phase2_pca_local_texture_2026-09-24.zip` | 2026-09-23 20:48:30.123 | workspace state, receipt `phase2_submission_receipt_2026-09-24.json` | CSV: `bf38812b1537dcc17de2dbd9c3b1fb6ec275b36e719e0edbfc2b5065766c7c23`; ZIP: `c4882e49e778182a43b5adff4ce9e39f22e2494d16954d9dd5dffd142a55ee88` | COMPLETE | 0.06404 | Below the frozen 0.15 Phase 2 floor; line sealed with no follow-up submission |

The official leaderboard snapshot contains two 1.00000 rows named
`perfect_submission.csv` and `test_ignore`. They are treated as organizer/test
artifacts and are neither used as evidence about labels nor imitated. The best
non-artifact score in this snapshot is 0.26119.

## Submission gates

1. Exact official ID order and row count are checked by streaming the full CSV.
2. Every label must be a canonical integer in 1–17.
3. Only organizer-provided labeled train/validation pixels train the classifier.
4. Unlabeled test pixels are used only for aggregate domain diagnostics.
5. A candidate must beat the baseline OOF result and avoid fold instability and
   single-class collapse before it earns a Kaggle submission slot.

## Phase 2 terminal decision — 2026-09-24 Beijing

The released Stage 2 payload contained only 170 labels, exactly ten per class.
The actual-data addendum fixed 128x128 spatial groups and a symmetric low-shot
LightGBM before metrics were observed. The PCA/local-texture candidate improved
both OOF OA and AA by 0.05882 over its same-data reference, kept the worst-fold
declines within 0.02, and predicted all 17 classes, so it lawfully earned the
single Phase 2 slot.

Kaggle ref `56503530` completed at public score `0.06404`. This is below the
pre-registered `0.15` stop floor and also below the Phase 1 spatial model's
`0.10387`. The mismatch is decisive evidence that the sparse Stage 2 spatial
folds do not estimate transfer to the two test scenes. No second submission,
class remapping, pseudo-label loop, or leaderboard-directed tuning is allowed
under the frozen protocol. The line is `SEALED_NO_FOLLOWUP_SUBMISSION`.
