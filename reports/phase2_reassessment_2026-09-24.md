# Phase 2 reassessment — 2026-09-24

## Decision

The line is actionable, but it is a **170-shot cross-scene adaptation problem**, not a
continuation of the 690,829-label Phase 1 experiment. The official contract and hashes
pass. The next authorized action is one fixed spatial cross-validation experiment; a
submission is still forbidden until every offline GO condition passes.

## What the second review changed

- The old Phase 1 model parameter `min_child_samples=40` is incompatible with a
  four-fold split of 170 points. Both reference and candidate therefore receive the
  same pre-registered low-shot LightGBM settings in
  `phase2_actual_data_addendum.json`; this does not alter the candidate or lower a gate.
- Fixed 128×128 spatial tiles yield 126 occupied groups. The four deterministic folds
  contain 44, 43, 41, and 42 labels, and all 17 classes occur in every fold. This makes
  spatial validation possible without random-pixel leakage.
- Phase 1 weights and its 690,829 validation labels stay outside the first experiment.
  Their apparent abundance is misleading: Phase 1 OOF was 0.68814 while the public
  score was only 0.10387, direct evidence of severe distribution shift.
- Deep learning, pseudo-labeling, source-label mixing, and leaderboard-guided mapping
  are rejected for the first pass. With ten labels per class they add degrees of freedom
  faster than evidence.

## First experiment

Reference: 3×3 local-mean spectrum with the shared low-shot LightGBM.

Candidate: fold-fit 24-component PCA of the center spectrum, plus 3×3 local mean,
local standard deviation, and first spectral differences, with the identical classifier.

Every fold fits PCA using its training partition only. Selection remains frozen:

1. candidate OA improvement at least 0.03;
2. candidate AA improvement at least 0.03;
3. worst-fold OA and AA decline no more than 0.02;
4. all schema, spatial-group, class-coverage, runtime, and class-collapse checks pass.

Failure of any item means no full test inference and no submission. Passing all items
permits exactly one full inference and one Phase 2 submission.

## Edge cases explicitly covered

- HDF5 windows are read in canonical row/column/band order and tested against MATLAB
  axis order.
- Edge pixels use the available clipped neighborhood; none are silently dropped.
- Every 128×128 tile belongs to one fold only.
- Each training and validation split contains every class.
- PCA is re-fit inside every fold.
- Candidate OOF predictions must cover all 17 classes.
- A paired class-stratified bootstrap is reported as an uncertainty diagnostic, but it
  cannot relax the frozen GO rule.
