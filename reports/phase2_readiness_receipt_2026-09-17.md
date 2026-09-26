# Tree HSI Phase 2 readiness receipt — 2026-09-17

## Live official fact

Authenticated Kaggle CLI inventory was read on 2026-09-17 Asia/Shanghai. It
contains the same seven files, byte sizes, and creation timestamps as the
frozen 2026-07-25 Phase 1 manifest. No Phase 2 file has been released.

- Baseline manifest signature:
  `7d209482f0811b5480027c01f03d9a3f50436697d83b9555ebb5245ad016e4af`
- Current manifest signature:
  `7d209482f0811b5480027c01f03d9a3f50436697d83b9555ebb5245ad016e4af`
- Added files: none
- Removed files: none
- Changed files: none

The authenticated submissions endpoint also confirms that the latest and best
entry is still `56120914` (`lightgbm_spatial_r1_v3.zip`), status `COMPLETE`,
public `0.10387`; no newer submission exists.

## Decision

`WAIT_PHASE2_NOT_RELEASED` / `NO_TRAIN_NO_INFERENCE_NO_SUBMISSION`.

The Phase 1 public score of `0.10387` disproved the old region-holdout proxy as
a useful cross-scene selector. The old `0.68814` OOF score is not a Phase 2
threshold and cannot justify another Phase 1 LightGBM/window tweak.

## Prepared fail-closed path

`src/phase2_readiness.py` now:

1. detects an official manifest delta before touching training;
2. requires exact local file sizes and full SHA-256 receipts after a delta;
3. discovers band count, the contiguous union and per-file subsets of label
   IDs, cube layouts, scene blocks, and submission
   ID ordering instead of assuming 98 bands / 17 classes / two scenes;
4. rejects missing files, non-contiguous labels, split scene blocks, ID gaps,
   unmanifested files, or ambiguous cube/label layouts;
5. blocks all old Phase 1 artifacts after a manifest change, even when the raw
   number of bands/classes happens to remain unchanged, unless an explicit
   semantic mapping exists.

The preregistered first comparison is not an old-score comparison. Both the
frozen 3x3 spatial-mean LightGBM reference and the PCA/local-texture candidate
must be retrained from scratch on identical Phase 2 spatial-block folds. The
candidate must improve both OA and AA by at least `+0.03`, with no more than a
`0.02` decline in the worst block, before full test inference is allowed.

## Verification

- Ten local tests pass, including five new Phase 2 gate tests.
- Synthetic changed-schema coverage verifies dynamic discovery of four bands,
  three classes, a new scene, exact ID order, complete hashes, and hard blocking
  of old artifacts.
- A malformed sample submission with an ID gap is rejected.
- The end-to-end event script was smoke-tested against the authenticated live
  Kaggle endpoint. It safely ignores the old CLI's pre-header upgrade warning
  and produced the same frozen `WAIT_PHASE2_NOT_RELEASED` receipt.
- No Kaggle submission, training run, test inference, or payload download was
  performed in this re-baseline.
