# Hyperspectral Tree Species Identification Challenge 2026

> **Archived public source and evidence snapshot.** This repository preserves
> the Phase 1 baseline plus the later uncommitted Phase 2 source, protocol,
> tests, and lightweight receipts from the local research workspace. Restricted
> competition data, data-derived model payloads, generated submissions,
> virtual environments, caches, and absolute-path symlinks are not published.

Low-cost, reproducible pipeline for the Kaggle competition
[`tree-species-hsi-2026`](https://www.kaggle.com/competitions/tree-species-hsi-2026).

The task is **pixel-level classification**, not object detection: predict one of
17 tree-species IDs for every pixel ID in two test scenes. The data contain 98
hyperspectral bands (approximately 399–924 nm). Overall Accuracy is the primary
metric and Average Accuracy is the tie-breaker.

> This description and the legacy commands below are frozen **Phase 1** facts.
> They must not be assumed to describe Phase 2.

## Phase 2 re-baseline — updated 2026-09-24

The official inventory changed on 2026-09-21 and added
`Stage2/Stage2/data.mat` plus `Stage2/Stage2/train_label.mat`; the sample
submission was also replaced. The complete changed payload was downloaded on
2026-09-24 through authenticated Kaggle URLs. It must be exposed through the
isolated `raw_phase2/` data root and pass the full-hash contract gate before any
training, inference, or submission.

On or after the 2026-09-21 release event, run the one-shot inventory gate:

```bash
scripts/phase2_event_gate.sh
```

If the inventory changes, first obtain the complete official payload. Then run
the contract gate with full hashes before any training:

```bash
../.venv/bin/python -m src.phase2_readiness \
  --baseline-manifest official/file_manifest.csv \
  --current-manifest official/file_manifest_latest.csv \
  --raw-dir raw_phase2 \
  --report reports/phase2_readiness_latest.json \
  --full-hash
```

Only `READY_FOR_PHASE2_VALIDATION` authorizes fold construction and training.
It still does not authorize a submission. The frozen protocol, first candidate,
offline advancement thresholds, and public-score stop rules are in
[`phase2_protocol.json`](phase2_protocol.json). Any Phase 2 manifest change
blocks reuse of Phase 1 model artifacts unless the organizer supplies an
explicit semantic class mapping.

### Phase 2 terminal result

The full contract passed and the pre-registered candidate passed all offline
gates, so it earned the one permitted submission. Kaggle submission `56503530`
completed with public score `0.06404`, below the frozen `0.15` stop floor. The
line is therefore sealed with no follow-up submission. See
`reports/phase2_submission_receipt_2026-09-24.json` for the hashes and live
readback evidence.

## Guardrails

- Competition data stay under `raw/` and are ignored by Git.
- Test labels are never inferred manually or sourced externally.
- Public pretrained models or external data require a license check and a
  disclosure entry before use.
- Model selection uses connected-region holdouts, never random pixel splits.
- A submission must pass the schema, row-count/ID, label-range, leakage, and
  rule checks in `src/pipeline.py` before upload.

## Reproduce the first baseline

From this directory, using the workspace virtual environment:

```bash
../.venv/bin/python -m src.pipeline audit --raw-dir raw --report reports/data_audit.json
../.venv/bin/python -m src.pipeline train --raw-dir raw --model artifacts/sam_centroids.npz --report reports/oof_sam.json
../.venv/bin/python -m src.pipeline predict --raw-dir raw --model artifacts/sam_centroids.npz --output submissions/sam_v1.csv
../.venv/bin/python -m src.pipeline validate --raw-dir raw --submission submissions/sam_v1.csv
```

The first model compares per-pixel spectra with class centroids using several
normalizations and selects the best variant on a two-fold connected-region
holdout. It is deliberately simple, CPU-only, and auditable. Tree/CNN and
spatial-spectral experiments should be added only after this baseline is valid.

Official facts and acceptance receipts are recorded in
[`official/official_receipt_2026-09-09.md`](official/official_receipt_2026-09-09.md).

See [`PUBLIC_ARCHIVE.md`](PUBLIC_ARCHIVE.md), [`PROVENANCE.yml`](PROVENANCE.yml),
and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the public-release boundary.

## Cite this repository

Please cite the repository snapshot and the immutable commit or tag you used.

```bibtex
@software{openkaggle_tree_species_hsi_2026,
  author = {Jah-yee},
  title = {Hyperspectral Tree Species Identification Challenge 2026 research},
  year = {2026},
  url = {https://github.com/OpenKaggle/tree-species-hsi-2026-research},
  version = {snapshot-2026-09}
}
```
