"""LightGBM spectral experiment with the same leakage-resistant folds.

This is intentionally separate from the cheap centroid baseline. It only earns
a Kaggle submission slot if its connected-region OOF score clears the recorded
centroid baseline and both folds remain stable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.pipeline import N_CLASSES, SEED, extract_training, metrics, sha256_file


def features(x: np.ndarray, mode: str = "full") -> np.ndarray:
    x = x.astype(np.float32, copy=False) / 10_000.0
    if mode == "robust":
        snv = (x - x.mean(axis=1, keepdims=True)) / (
            x.std(axis=1, keepdims=True) + 1e-6
        )
        return np.ascontiguousarray(np.column_stack((snv, np.diff(snv, axis=1))))
    if mode != "full":
        raise ValueError(f"Unknown feature mode: {mode}")
    selected = np.linspace(0, x.shape[1] - 1, 25).round().astype(int)
    sampled = x[:, selected]
    sampled_snv = (sampled - sampled.mean(axis=1, keepdims=True)) / (
        sampled.std(axis=1, keepdims=True) + 1e-6
    )
    sampled_diff = np.diff(sampled_snv, axis=1)
    summaries = np.column_stack(
        (
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            x[:, 60:80].mean(axis=1) - x[:, 20:40].mean(axis=1),
        )
    )
    return np.ascontiguousarray(np.column_stack((x, sampled_snv, sampled_diff, summaries)))


def balanced_sample(y: np.ndarray, eligible: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    selected: list[np.ndarray] = []
    for class_id in range(1, N_CLASSES + 1):
        indices = np.flatnonzero(eligible & (y == class_id))
        if len(indices) > per_class:
            indices = rng.choice(indices, size=per_class, replace=False)
        selected.append(indices)
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def new_model(n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--model", default="artifacts/lightgbm_v1.txt")
    parser.add_argument("--report", default="reports/oof_lightgbm_v1.json")
    parser.add_argument("--per-class", type=int, default=10_000)
    parser.add_argument("--estimators", type=int, default=220)
    parser.add_argument("--feature-mode", choices=("full", "robust"), default="full")
    parser.add_argument("--spatial-radius", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    raw_dir = Path(args.raw_dir)
    x_raw, y, folds, metadata = extract_training(raw_dir, args.spatial_radius)
    source = np.asarray(metadata.pop("source"))
    x = features(x_raw, args.feature_mode)
    del x_raw
    rng = np.random.default_rng(SEED)
    oof = np.zeros(len(y), dtype=np.uint8)
    fold_metrics = []
    for fold in (0, 1):
        training = (folds != fold) | (source == 0)
        validation = folds == fold
        fit_indices = balanced_sample(y, training, args.per_class, rng)
        model = new_model(args.estimators)
        model.fit(x[fit_indices], y[fit_indices] - 1)
        oof[validation] = model.predict(x[validation]).astype(np.uint8) + 1
        fold_metrics.append(metrics(y[validation], oof[validation]))

    evaluated = folds >= 0
    overall = metrics(y[evaluated], oof[evaluated])
    final_indices = balanced_sample(y, np.ones(len(y), dtype=bool), args.per_class, rng)
    final_model = new_model(args.estimators)
    final_model.fit(x[final_indices], y[final_indices] - 1)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(model_path))
    report = {
        "seed": SEED,
        "features": int(x.shape[1]),
        "feature_mode": args.feature_mode,
        "spatial_radius": args.spatial_radius,
        "per_class_cap": args.per_class,
        "estimators": args.estimators,
        "validation": "two-fold, per-class connected-region holdout",
        "overall": overall,
        "folds": fold_metrics,
        "fold_oa_range": float(max(m["oa"] for m in fold_metrics) - min(m["oa"] for m in fold_metrics)),
        "fold_aa_range": float(max(m["aa"] for m in fold_metrics) - min(m["aa"] for m in fold_metrics)),
        "runtime_seconds": time.time() - started,
        "data": metadata,
        "model_sha256": sha256_file(model_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
