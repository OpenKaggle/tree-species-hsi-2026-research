# Reproducibility boundary

This public snapshot preserves the source and evidence needed to inspect the
Phase 1 and Phase 2 research process. It is not a turnkey data bundle.

## Inputs

Obtain competition inputs only through a currently authorized Kaggle account
and follow the competition-specific terms. Do not copy the local `raw/`,
`raw_phase2/`, `downloads/`, `artifacts/`, or `submissions/` layout into a
public checkout. The public repository intentionally contains none of those
payloads.

`official/file_manifest.csv` and `official/file_manifest_latest.csv` record
the historical inventory boundary. Hashes document identity; they do not grant
redistribution rights.

## Entry points

- `README.md` records the historical commands and gates.
- `src/pipeline.py` contains the Phase 1 audit, train, predict, and validation
  path.
- `src/phase2_readiness.py` enforces the Phase 2 manifest and hash contract.
- `src/phase2_experiment.py` and `src/phase2_predict.py` preserve the later
  experiment and inference path.
- `tests/` contains dependency-light checks of the retained logic.

Paths in the historical commands assume the original workspace-level virtual
environment. Recreate an isolated environment from `requirements.txt` rather
than relying on those relative paths verbatim.

## Evidence limits

The Phase 2 report records one completed gated submission and a public score.
It is leaderboard evidence, not a private-score, organizer-reproduction, or
generalization claim. The original large submission file and model artifacts
are excluded; lightweight hash-bound receipts remain.

The source overlay came from an uncommitted working tree. `PROVENANCE.yml`
records the base commit and the SHA-256 of the tracked patch, while
`SOURCE_MANIFEST.sha256` binds the exact public file set.
