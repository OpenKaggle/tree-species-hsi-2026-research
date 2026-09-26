"""Pre-registered low-shot Phase 2 experiment for Tree Species HSI 2026.

The module implements the frozen Phase 2 protocol plus its actual-data
addendum. It extracts only organizer-labeled pixels, keeps 128x128 spatial
tiles wholly inside one fold, fits every transform inside its training fold,
and evaluates exactly one scientific candidate against one same-data
reference. Full-scene inference is forbidden unless the recorded GO gate
passes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

from src.pipeline import HsiCube, N_BANDS, N_CLASSES, SEED, load_label, metrics, sha256_file


TILE_SIZE = 128
N_FOLDS = 4
PCA_COMPONENTS = 24
EPS = 1e-6


@dataclass(frozen=True)
class SparseFeatures:
    center: np.ndarray
    local_mean: np.ndarray
    local_std: np.ndarray
    labels: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    cube_layout: dict[str, object]
    hdf5_chunks: tuple[int, ...] | None


def _snv(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + EPS
    )


def extract_sparse_features(
    cube_path: Path,
    label_path: Path,
    radius: int = 1,
    cache_tile_size: int = TILE_SIZE,
) -> SparseFeatures:
    """Extract exact local statistics while decompressing each occupied tile once."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    labels_map = load_label(label_path)
    rows, cols = np.nonzero(labels_map)
    labels = labels_map[rows, cols]
    if len(labels) != N_CLASSES * 10:
        raise ValueError(f"Expected 170 Stage 2 labels, found {len(labels)}")
    counts = np.bincount(labels, minlength=N_CLASSES + 1)[1:]
    if not np.array_equal(counts, np.full(N_CLASSES, 10)):
        raise ValueError(f"Expected ten labels per class, found {counts.tolist()}")

    center = np.empty((len(labels), N_BANDS), dtype=np.float32)
    local_mean = np.empty_like(center)
    local_std = np.empty_like(center)
    tile_rows = rows // cache_tile_size
    tile_cols = cols // cache_tile_size
    occupied = sorted(set(zip(tile_rows.tolist(), tile_cols.tolist())))

    with HsiCube(cube_path, expected_shape=labels_map.shape) as cube:
        for tile_row, tile_col in occupied:
            indices = np.flatnonzero((tile_rows == tile_row) & (tile_cols == tile_col))
            core_row_start = tile_row * cache_tile_size
            core_col_start = tile_col * cache_tile_size
            row_start = max(0, core_row_start - radius)
            row_stop = min(cube.layout.rows, core_row_start + cache_tile_size + radius)
            col_start = max(0, core_col_start - radius)
            col_stop = min(cube.layout.cols, core_col_start + cache_tile_size + radius)
            block = cube.read_window(row_start, row_stop, col_start, col_stop)
            for index in indices:
                row, col = int(rows[index]), int(cols[index])
                block_row, block_col = row - row_start, col - col_start
                patch = block[
                    max(0, block_row - radius) : min(block.shape[0], block_row + radius + 1),
                    max(0, block_col - radius) : min(block.shape[1], block_col + radius + 1),
                ]
                center[index] = block[block_row, block_col]
                local_mean[index] = patch.mean(axis=(0, 1))
                local_std[index] = patch.std(axis=(0, 1))
        layout = dict(cube.layout.__dict__)
        chunks = tuple(int(x) for x in cube.dataset.chunks) if cube.dataset.chunks else None

    return SparseFeatures(
        center=center,
        local_mean=local_mean,
        local_std=local_std,
        labels=labels.astype(np.uint8, copy=False),
        rows=rows.astype(np.int32, copy=False),
        cols=cols.astype(np.int32, copy=False),
        cube_layout=layout,
        hdf5_chunks=chunks,
    )


def spatial_folds(
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    tile_size: int = TILE_SIZE,
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic folds and integer tile groups with strict invariants."""
    n_tile_cols = int(cols.max() // tile_size) + 1
    groups = (rows // tile_size) * n_tile_cols + cols // tile_size
    folds = np.full(len(labels), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels), dtype=np.uint8), labels, groups)
    ):
        train_classes = set(labels[train_indices].tolist())
        validation_classes = set(labels[validation_indices].tolist())
        expected = set(range(1, N_CLASSES + 1))
        if train_classes != expected or validation_classes != expected:
            raise ValueError(
                f"Fold {fold} class coverage failed: train={sorted(train_classes)}, "
                f"validation={sorted(validation_classes)}"
            )
        folds[validation_indices] = fold
    if np.any(folds < 0):
        raise RuntimeError("Not every label received a validation fold")
    for group in np.unique(groups):
        if len(np.unique(folds[groups == group])) != 1:
            raise RuntimeError(f"Spatial tile {int(group)} crosses validation folds")
    return folds, groups.astype(np.int32, copy=False)


def new_model() -> lgb.LGBMClassifier:
    """The symmetric low-shot classifier frozen in the actual-data addendum."""
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=160,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=2,
        max_bin=63,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def reference_features(data: SparseFeatures, train_indices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    del train_indices
    return np.ascontiguousarray(data.local_mean[indices] / 10_000.0, dtype=np.float32)


def candidate_feature_pair(
    data: SparseFeatures, train_indices: np.ndarray, validation_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit PCA on the fold's training spectra and transform both partitions."""
    components = min(PCA_COMPONENTS, len(train_indices) - 1, N_BANDS)
    pca = PCA(n_components=components, svd_solver="full", random_state=SEED)
    pca.fit(_snv(data.center[train_indices]))

    metadata = {
        "pca_components": int(components),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "feature_count": int(components + N_BANDS + N_BANDS + N_BANDS - 1),
    }
    return (
        candidate_features_from_arrays(
            data.center[train_indices],
            data.local_mean[train_indices],
            data.local_std[train_indices],
            pca.mean_,
            pca.components_,
        ),
        candidate_features_from_arrays(
            data.center[validation_indices],
            data.local_mean[validation_indices],
            data.local_std[validation_indices],
            pca.mean_,
            pca.components_,
        ),
        metadata,
    )


def candidate_features_from_arrays(
    center: np.ndarray,
    local_mean: np.ndarray,
    local_std: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Apply the frozen candidate transform to an arbitrary pixel batch."""
    center_snv = _snv(center)
    mean_snv = _snv(local_mean)
    pca_scores = (center_snv - pca_mean) @ pca_components.T
    relative_std = np.log1p(local_std / (np.abs(local_mean) + 1.0))
    differences = np.diff(mean_snv, axis=1)
    return np.ascontiguousarray(
        np.column_stack((pca_scores, mean_snv, relative_std, differences)),
        dtype=np.float32,
    )


def _prediction_counts(predictions: np.ndarray) -> list[int]:
    return np.bincount(predictions, minlength=N_CLASSES + 1)[1:].astype(int).tolist()


def evaluate_reference(data: SparseFeatures, folds: np.ndarray) -> dict[str, object]:
    oof = np.zeros(len(data.labels), dtype=np.uint8)
    fold_reports: list[dict[str, object]] = []
    for fold in range(N_FOLDS):
        train_indices = np.flatnonzero(folds != fold)
        validation_indices = np.flatnonzero(folds == fold)
        model = new_model()
        model.fit(
            reference_features(data, train_indices, train_indices),
            data.labels[train_indices] - 1,
        )
        predictions = model.predict(
            reference_features(data, train_indices, validation_indices)
        ).astype(np.uint8) + 1
        oof[validation_indices] = predictions
        fold_reports.append(
            {
                "fold": fold,
                "samples": int(len(validation_indices)),
                **metrics(data.labels[validation_indices], predictions),
                "prediction_counts": _prediction_counts(predictions),
            }
        )
    overall = metrics(data.labels, oof)
    return {
        "name": "phase2_retrained_spatial_mean_lightgbm",
        "features": N_BANDS,
        "overall": overall,
        "folds": fold_reports,
        "worst_fold_oa": float(min(report["oa"] for report in fold_reports)),
        "worst_fold_aa": float(min(report["aa"] for report in fold_reports)),
        "prediction_counts": _prediction_counts(oof),
        "oof_predictions": oof,
    }


def evaluate_candidate(data: SparseFeatures, folds: np.ndarray) -> dict[str, object]:
    oof = np.zeros(len(data.labels), dtype=np.uint8)
    fold_reports: list[dict[str, object]] = []
    feature_count = 0
    for fold in range(N_FOLDS):
        train_indices = np.flatnonzero(folds != fold)
        validation_indices = np.flatnonzero(folds == fold)
        x_train, x_validation, transform_report = candidate_feature_pair(
            data, train_indices, validation_indices
        )
        feature_count = int(transform_report["feature_count"])
        model = new_model()
        model.fit(x_train, data.labels[train_indices] - 1)
        predictions = model.predict(x_validation).astype(np.uint8) + 1
        oof[validation_indices] = predictions
        fold_reports.append(
            {
                "fold": fold,
                "samples": int(len(validation_indices)),
                **transform_report,
                **metrics(data.labels[validation_indices], predictions),
                "prediction_counts": _prediction_counts(predictions),
            }
        )
    overall = metrics(data.labels, oof)
    return {
        "name": "pca_local_texture_lightgbm",
        "features": feature_count,
        "overall": overall,
        "folds": fold_reports,
        "worst_fold_oa": float(min(report["oa"] for report in fold_reports)),
        "worst_fold_aa": float(min(report["aa"] for report in fold_reports)),
        "prediction_counts": _prediction_counts(oof),
        "oof_predictions": oof,
    }


def paired_stratified_bootstrap(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    iterations: int = 4000,
) -> dict[str, object]:
    """Quantify score-delta fragility without changing the frozen GO rule."""
    rng = np.random.default_rng(SEED)
    by_class = [np.flatnonzero(labels == class_id) for class_id in range(1, N_CLASSES + 1)]
    oa_deltas = np.empty(iterations, dtype=np.float32)
    aa_deltas = np.empty(iterations, dtype=np.float32)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_class]
        )
        reference_metrics = metrics(labels[sampled], reference[sampled])
        candidate_metrics = metrics(labels[sampled], candidate[sampled])
        oa_deltas[iteration] = candidate_metrics["oa"] - reference_metrics["oa"]
        aa_deltas[iteration] = candidate_metrics["aa"] - reference_metrics["aa"]
    return {
        "iterations": iterations,
        "oa_delta_percentiles_2_5_50_97_5": np.percentile(
            oa_deltas, [2.5, 50.0, 97.5]
        ).astype(float).tolist(),
        "aa_delta_percentiles_2_5_50_97_5": np.percentile(
            aa_deltas, [2.5, 50.0, 97.5]
        ).astype(float).tolist(),
        "probability_candidate_oa_delta_positive": float((oa_deltas > 0).mean()),
        "probability_candidate_aa_delta_positive": float((aa_deltas > 0).mean()),
    }


def advancement_gate(reference: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    reference_overall = reference["overall"]
    candidate_overall = candidate["overall"]
    checks = {
        "oa_delta_at_least_0_03": bool(
            candidate_overall["oa"] - reference_overall["oa"] >= 0.03
        ),
        "aa_delta_at_least_0_03": bool(
            candidate_overall["aa"] - reference_overall["aa"] >= 0.03
        ),
        "worst_fold_oa_decline_at_most_0_02": bool(
            candidate["worst_fold_oa"] >= reference["worst_fold_oa"] - 0.02
        ),
        "worst_fold_aa_decline_at_most_0_02": bool(
            candidate["worst_fold_aa"] >= reference["worst_fold_aa"] - 0.02
        ),
        "no_oof_class_collapse": bool(all(count > 0 for count in candidate["prediction_counts"])),
    }
    return {
        "checks": checks,
        "offline_go": bool(all(checks.values())),
        "decision": "MAY_RUN_ONE_FULL_INFERENCE" if all(checks.values()) else "STOP_NO_INFERENCE_NO_SUBMISSION",
    }


def strip_arrays(report: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in report.items() if key != "oof_predictions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="raw_phase2")
    parser.add_argument("--report", default="reports/phase2_first_experiment.json")
    args = parser.parse_args()

    started = time.time()
    raw_dir = Path(args.raw_dir)
    data = extract_sparse_features(
        raw_dir / "Stage2" / "Stage2" / "data.mat",
        raw_dir / "Stage2" / "Stage2" / "train_label.mat",
    )
    folds, groups = spatial_folds(data.rows, data.cols, data.labels)
    reference = evaluate_reference(data, folds)
    candidate = evaluate_candidate(data, folds)
    bootstrap = paired_stratified_bootstrap(
        data.labels, reference["oof_predictions"], candidate["oof_predictions"]
    )
    gate = advancement_gate(reference, candidate)
    fold_class_counts = [
        np.bincount(data.labels[folds == fold], minlength=N_CLASSES + 1)[1:].astype(int).tolist()
        for fold in range(N_FOLDS)
    ]
    report = {
        "protocol": {
            "parent": "phase2_protocol.json",
            "actual_data_addendum": "phase2_actual_data_addendum.json",
            "seed": SEED,
        },
        "data": {
            "cube": str(raw_dir / "Stage2" / "Stage2" / "data.mat"),
            "labels": str(raw_dir / "Stage2" / "Stage2" / "train_label.mat"),
            "cube_sha256": sha256_file(raw_dir / "Stage2" / "Stage2" / "data.mat"),
            "label_sha256": sha256_file(raw_dir / "Stage2" / "Stage2" / "train_label.mat"),
            "cube_layout": data.cube_layout,
            "hdf5_chunks": list(data.hdf5_chunks) if data.hdf5_chunks else None,
            "samples": int(len(data.labels)),
            "class_counts": np.bincount(data.labels, minlength=N_CLASSES + 1)[1:].astype(int).tolist(),
        },
        "validation": {
            "splitter": "StratifiedGroupKFold",
            "folds": N_FOLDS,
            "tile_size": TILE_SIZE,
            "tile_groups": int(len(np.unique(groups))),
            "fold_sizes": np.bincount(folds, minlength=N_FOLDS).astype(int).tolist(),
            "fold_class_counts": fold_class_counts,
            "tile_group_integrity": True,
            "class_coverage_integrity": True,
        },
        "reference": strip_arrays(reference),
        "candidate": strip_arrays(candidate),
        "paired_stratified_bootstrap": bootstrap,
        "gate": gate,
        "runtime_seconds": time.time() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
