"""Diagnose unlabeled train/test spectral shift without using test labels.

The diagnostic uses deterministic spatial grids from each official cube. It
reports only aggregate spectra, nearest-centroid class proportions, and
confidence margins. Test pixels never influence the class centroids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pipeline import (
    HsiCube,
    N_BANDS,
    SEED,
    extract_training,
    fit_centroids,
    predict_centroids,
    read_scene_info,
    transform_spectra,
)


VARIANTS = ("raw_l2", "snv_l2", "diff_l2", "snv_no80_l2")


def grid_sample(path: Path, shape: tuple[int, int], target_pixels: int) -> np.ndarray:
    """Sample a regular pixel grid with bounded memory and deterministic order."""
    rows, cols = shape
    stride = max(1, int(np.sqrt((rows * cols) / target_pixels)))
    sampled: list[np.ndarray] = []
    with HsiCube(path, expected_shape=shape) as cube:
        for row in range(stride // 2, rows, stride):
            block = cube.read_rows(row, row + 1)[0]
            sampled.append(block[stride // 2 :: stride])
    return np.concatenate(sampled, axis=0).astype(np.float32, copy=False)


def aggregate(values: np.ndarray) -> dict[str, object]:
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "pixels": int(len(values)),
        "band_mean": values.mean(axis=0).astype(float).tolist(),
        "band_std": values.std(axis=0).astype(float).tolist(),
        "band_quantiles": quantiles.astype(float).tolist(),
    }


def prediction_summary(
    values: np.ndarray, centroids: np.ndarray, variant: str
) -> dict[str, object]:
    transformed = transform_spectra(values, variant)
    scores = transformed @ centroids.T
    prediction = scores.argmax(axis=1).astype(np.uint8) + 1
    partitioned = np.partition(scores, -2, axis=1)
    margins = partitioned[:, -1] - partitioned[:, -2]
    return {
        "class_counts": np.bincount(prediction, minlength=18)[1:].astype(int).tolist(),
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--report", default="reports/domain_diagnostic.json")
    parser.add_argument("--target-pixels", type=int, default=60_000)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    labeled_x, labeled_y, _, _ = extract_training(raw_dir)
    centroids = {
        variant: fit_centroids(labeled_x, labeled_y, variant) for variant in VARIANTS
    }
    scenes = read_scene_info(raw_dir / "scene_info.csv")
    sampled = {
        "train_scene": grid_sample(raw_dir / "data_hsi.mat", (4040, 2444), args.target_pixels)
    }
    for scene in scenes:
        name = str(scene["scene"])
        sampled[name] = grid_sample(
            raw_dir / f"test_{name}.mat",
            (int(scene["height"]), int(scene["width"])),
            args.target_pixels,
        )

    report: dict[str, object] = {
        "seed": SEED,
        "sampling": "deterministic regular spatial grid; no test labels",
        "target_pixels_per_scene": args.target_pixels,
        "scenes": {},
    }
    train_mean = sampled["train_scene"].mean(axis=0)
    train_std = sampled["train_scene"].std(axis=0)
    for name, values in sampled.items():
        scene_report = aggregate(values)
        scene_report["mean_shift_vs_train_in_train_sd"] = float(
            np.mean(np.abs(values.mean(axis=0) - train_mean) / np.maximum(train_std, 1e-6))
        )
        scene_report["predictions"] = {
            variant: prediction_summary(values, centroids[variant], variant)
            for variant in VARIANTS
        }
        report["scenes"][name] = scene_report

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        name: {
            "pixels": values["pixels"],
            "mean_shift_vs_train_in_train_sd": values["mean_shift_vs_train_in_train_sd"],
            "predictions": values["predictions"],
        }
        for name, values in report["scenes"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
