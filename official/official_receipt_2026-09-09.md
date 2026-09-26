# Official competition receipt — 2026-09-09

Captured from Kaggle's official UI and authenticated API on 2026-09-08 UTC /
2026-09-09 Asia/Shanghai.

## Entry and identity

- Competition ID: `157321`
- Slug: `tree-species-hsi-2026`
- Account/team leader identity was verified in the private source receipt and
  is omitted from this public archive copy.
- Entry status after acceptance: `user_has_entered=true`
- UI receipt: `Rules accepted. Good luck!`
- Submission UI became available immediately after acceptance.
- No optional marketing or data-sharing choice was presented or accepted.

## Deadlines and limits

- Final submission deadline: **2026-09-25 04:00:00 UTC** =
  **2026-09-25 12:00:00 Asia/Shanghai**.
- Phase 1: 2026-07-25 through 2026-09-21.
- Phase 2 data release/start: 2026-09-21.
- Phase 2 competition period: 2026-09-21 through 2026-09-25.
- Code review deadline: 2026-09-27.
- Winners announced: 2026-09-28.
- Authenticated API fields `new_entrant_deadline` and `merger_deadline` are
  `null`; Kaggle has configured no separate Entry or Team Merger cutoff.
- Maximum submissions: 5 per day.
- File submissions are allowed; this is not a notebook-only competition.
- Submissions are currently enabled.

## Task and metric

- Pixel-level classification of 17 urban tree species.
- One labeled train/validation scene and two unlabeled test scenes.
- 98 bands, approximately 399–924 nm.
- Phase 1 tests cross-region generalization. Phase 2 may use different species
  categories and determines the final result by organizer reproduction.
- Primary metric: Overall Accuracy (OA).
- Tie-break metric: Average Accuracy (AA).

## Submission schema receipt

Official ranged reads of `sample_submission.csv` verified:

- Total bytes: `407070393`.
- Header: `id,label`.
- First row: `scene1_00000000,1`.
- Scene 1 contains `13,989,728` IDs, ending at `scene1_13989727`.
- Scene 2 contains `7,435,029` IDs, ending at `scene2_07435028`.
- Total prediction rows: `21,424,757`.
- IDs use zero-based, eight-digit, C-order flattened pixel indices.
- Only the `label` values may be replaced; all IDs and column names must remain
  unchanged. Labels must be integers 1–17.

## Data license and restrictive rules

- Kaggle Data page metadata: CC BY-NC-SA 4.0.
- Competition-specific rule: the SZUTreeHSI data are owned by Professor Sen
  Jia's Shenzhen University research team and may be used only for algorithm
  development, training, validation, and evaluation in this competition.
- Without prior written permission, do not publish, redistribute, transfer,
  commercially use, or upload the data elsewhere; do not use it for unrelated
  research or publication.
- No manual/fabricated test labels, leaked annotations, unreleased ground truth,
  undeclared private datasets, multiple accounts, evaluation attacks, or
  private sharing outside the registered team.
- Public external resources/pretrained models must be legally accessible and
  disclosed in final documentation.
- Top 6 teams must send training/inference source, weights, environment,
  instructions, reproducible inference script, and a short method description
  to the organizer by the code-review deadline.

## Awards and rule conflict

- Competition rules list: first 2,500 RMB + certificate; second 1,500 RMB +
  certificate; third 1,000 RMB + certificate; three excellence certificates.
- Kaggle's summary/API show reward `Kudos` and no Kaggle points/medals. These
  statements are compatible: organizer cash/certificates, but no Kaggle ranking
  points or medals.
- Competition-specific rules say maximum team size is 5; Kaggle's team UI and
  API currently show 10. Follow the stricter **5-person** rule unless the host
  posts a clarification.

## Official sources

- Overview: https://www.kaggle.com/competitions/tree-species-hsi-2026
- Data: https://www.kaggle.com/competitions/tree-species-hsi-2026/data
- Rules: https://www.kaggle.com/competitions/tree-species-hsi-2026/rules
- Team: https://www.kaggle.com/competitions/tree-species-hsi-2026/team
- Submissions: https://www.kaggle.com/competitions/tree-species-hsi-2026/submissions
