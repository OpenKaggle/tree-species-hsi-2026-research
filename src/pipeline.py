"""Auditable CPU baseline for the Tree Species HSI 2026 competition.

The implementation avoids loading a full hyperspectral scene into memory. It
uses row chunks, connected-region holdouts, and spectral-angle centroids. The
generated submission order is derived from the organizer's `scene_info.csv` and
verified against ranged reads of the official sample submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
from scipy import ndimage
from scipy.io import loadmat


N_CLASSES = 17
N_BANDS = 98
SEED = 20260909
EPS = 1e-8


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def numeric_hdf5_datasets(group: h5py.Group, prefix: str = "") -> list[str]:
    names: list[str] = []
    for key, value in group.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number):
            names.append(name)
        elif isinstance(value, h5py.Group):
            names.extend(numeric_hdf5_datasets(value, name))
    return names


@dataclass(frozen=True)
class CubeLayout:
    dataset_name: str
    raw_shape: tuple[int, int, int]
    row_axis: int
    col_axis: int
    band_axis: int
    rows: int
    cols: int
    bands: int


class HsiCube(AbstractContextManager["HsiCube"]):
    """HDF5-backed cube exposed canonically as (row, column, band)."""

    def __init__(self, path: Path, expected_shape: tuple[int, int] | None = None):
        self.path = path
        self.handle = h5py.File(path, "r")
        candidates = numeric_hdf5_datasets(self.handle)
        if not candidates:
            self.handle.close()
            raise ValueError(f"No numeric HDF5 dataset found in {path}")
        three_d = [name for name in candidates if self.handle[name].ndim == 3]
        if len(three_d) != 1:
            self.handle.close()
            raise ValueError(f"Expected one 3-D dataset in {path}; found {three_d}")
        name = three_d[0]
        ds = self.handle[name]
        shape = tuple(int(x) for x in ds.shape)
        band_axes = [i for i, size in enumerate(shape) if size == N_BANDS]
        if len(band_axes) != 1:
            self.handle.close()
            raise ValueError(f"Cannot identify the {N_BANDS}-band axis in {path}: {shape}")
        band_axis = band_axes[0]
        spatial_axes = [i for i in range(3) if i != band_axis]
        if expected_shape is None:
            row_axis, col_axis = spatial_axes
        else:
            rows, cols = expected_shape
            matches = [
                (r, c)
                for r in spatial_axes
                for c in spatial_axes
                if r != c and shape[r] == rows and shape[c] == cols
            ]
            if len(matches) != 1:
                self.handle.close()
                raise ValueError(
                    f"Cannot map spatial axes {shape} to expected {expected_shape} in {path}"
                )
            row_axis, col_axis = matches[0]
        self.dataset = ds
        self.layout = CubeLayout(
            dataset_name=name,
            raw_shape=shape,
            row_axis=row_axis,
            col_axis=col_axis,
            band_axis=band_axis,
            rows=shape[row_axis],
            cols=shape[col_axis],
            bands=shape[band_axis],
        )

    def read_rows(self, start: int, stop: int) -> np.ndarray:
        slices: list[slice] = [slice(None), slice(None), slice(None)]
        slices[self.layout.row_axis] = slice(start, stop)
        raw = np.asarray(self.dataset[tuple(slices)])
        canonical = np.moveaxis(
            raw,
            (self.layout.row_axis, self.layout.col_axis, self.layout.band_axis),
            (0, 1, 2),
        )
        return canonical.astype(np.float32, copy=False)

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> np.ndarray:
        """Read a canonical ``(row, column, band)`` spatial window.

        This is deliberately separate from :meth:`read_rows`: sparse Phase 2
        labels occupy only a small number of HDF5 chunks, so reading full scene
        rows would multiply I/O by the scene width.
        """
        if not (
            0 <= row_start < row_stop <= self.layout.rows
            and 0 <= col_start < col_stop <= self.layout.cols
        ):
            raise ValueError(
                "Invalid window "
                f"({row_start}:{row_stop}, {col_start}:{col_stop}) for "
                f"{self.layout.rows}x{self.layout.cols} cube"
            )
        slices: list[slice] = [slice(None), slice(None), slice(None)]
        slices[self.layout.row_axis] = slice(row_start, row_stop)
        slices[self.layout.col_axis] = slice(col_start, col_stop)
        raw = np.asarray(self.dataset[tuple(slices)])
        canonical = np.moveaxis(
            raw,
            (self.layout.row_axis, self.layout.col_axis, self.layout.band_axis),
            (0, 1, 2),
        )
        return canonical.astype(np.float32, copy=False)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.handle.close()


def read_spatial_mean_rows(
    cube: HsiCube, start: int, stop: int, radius: int
) -> np.ndarray:
    """Read rows with an exact square spatial mean and chunk-boundary halos."""
    if radius < 0:
        raise ValueError("Spatial radius must be non-negative")
    if radius == 0:
        return cube.read_rows(start, stop)
    read_start = max(0, start - radius)
    read_stop = min(cube.layout.rows, stop + radius)
    block = cube.read_rows(read_start, read_stop)
    smoothed = ndimage.uniform_filter(
        block, size=(2 * radius + 1, 2 * radius + 1, 1), mode="nearest"
    )
    offset = start - read_start
    return smoothed[offset : offset + stop - start]


def read_local_stats_rows(
    cube: HsiCube, start: int, stop: int, radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read center spectra plus exact clipped-window local mean and std.

    Row halos make the result invariant to the caller's chunk boundaries.
    At true image boundaries the divisor is reduced instead of padding or
    duplicating pixels, matching the sparse training extractor.
    """
    if radius < 0:
        raise ValueError("Spatial radius must be non-negative")
    if radius == 0:
        center = cube.read_rows(start, stop)
        return center, center.copy(), np.zeros_like(center)
    read_start = max(0, start - radius)
    read_stop = min(cube.layout.rows, stop + radius)
    block = cube.read_window(read_start, read_stop, 0, cube.layout.cols)
    width = 2 * radius + 1
    area = float(width * width)
    counts = ndimage.uniform_filter(
        np.ones(block.shape[:2], dtype=np.float32),
        size=(width, width),
        mode="constant",
        cval=0.0,
    ) * area
    sums = ndimage.uniform_filter(
        block, size=(width, width, 1), mode="constant", cval=0.0
    ) * area
    square_sums = ndimage.uniform_filter(
        block * block, size=(width, width, 1), mode="constant", cval=0.0
    ) * area
    mean = sums / counts[..., None]
    variance = np.maximum(square_sums / counts[..., None] - mean * mean, 0.0)
    std = np.sqrt(variance, dtype=np.float32)
    offset = start - read_start
    selected = slice(offset, offset + stop - start)
    return block[selected], mean[selected], std[selected]


def load_label(path: Path) -> np.ndarray:
    payload = loadmat(path)
    arrays = [
        value
        for key, value in payload.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2
    ]
    if len(arrays) != 1:
        raise ValueError(f"Expected one 2-D label array in {path}; found {len(arrays)}")
    label = arrays[0]
    if label.min() < 0 or label.max() > N_CLASSES:
        raise ValueError(f"Unexpected label range in {path}: {label.min()}..{label.max()}")
    return label.astype(np.uint8, copy=False)


def read_scene_info(path: Path) -> list[dict[str, int | str | bool]]:
    scenes: list[dict[str, int | str | bool]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            height, width, count = int(row["height"]), int(row["width"]), int(row["pixel_count"])
            if height * width != count:
                raise ValueError(f"Scene dimensions do not match pixel_count: {row}")
            scenes.append(
                {
                    "scene": row["scene"],
                    "height": height,
                    "width": width,
                    "pixel_count": count,
                    "transpose": row["transpose"].lower() == "true",
                    "flatten_order": row["flatten_order"],
                }
            )
    if [scene["scene"] for scene in scenes] != ["scene1", "scene2"]:
        raise ValueError(f"Unexpected scene order: {scenes}")
    return scenes


def iter_labeled(
    cube: HsiCube,
    labels: np.ndarray,
    chunk_rows: int = 64,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if labels.shape != (cube.layout.rows, cube.layout.cols):
        raise ValueError(f"Label/cube mismatch: {labels.shape} vs {(cube.layout.rows, cube.layout.cols)}")
    for start in range(0, cube.layout.rows, chunk_rows):
        stop = min(start + chunk_rows, cube.layout.rows)
        y = labels[start:stop]
        local_rows, cols = np.nonzero(y)
        if not len(local_rows):
            continue
        block = cube.read_rows(start, stop)
        yield (
            block[local_rows, cols],
            y[local_rows, cols],
            local_rows.astype(np.int32) + start,
            cols.astype(np.int32),
        )


def connected_region_folds(labels: np.ndarray, n_folds: int = 2) -> np.ndarray:
    """Assign every same-class connected component wholly to one balanced fold."""
    folds = np.full(labels.shape, -1, dtype=np.int8)
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_id in range(1, N_CLASSES + 1):
        components, count = ndimage.label(labels == class_id, structure=structure)
        if count < n_folds:
            raise ValueError(f"Class {class_id} has only {count} connected regions")
        sizes = np.bincount(components.ravel())[1:]
        fold_loads = np.zeros(n_folds, dtype=np.int64)
        for component_index in np.argsort(sizes)[::-1]:
            fold = int(np.argmin(fold_loads))
            component_id = int(component_index + 1)
            folds[components == component_id] = fold
            fold_loads[fold] += int(sizes[component_index])
    return folds


def transform_spectra(values: np.ndarray, variant: str) -> np.ndarray:
    x = values.astype(np.float32, copy=False)
    if variant == "raw_l2":
        transformed = x
    elif variant == "snv_l2":
        transformed = (x - x.mean(axis=1, keepdims=True)) / (
            x.std(axis=1, keepdims=True) + EPS
        )
    elif variant == "diff_l2":
        snv = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + EPS)
        transformed = np.diff(snv, axis=1)
    elif variant == "snv_no80_l2":
        kept = np.concatenate((x[:, :80], x[:, 81:]), axis=1)
        transformed = (kept - kept.mean(axis=1, keepdims=True)) / (
            kept.std(axis=1, keepdims=True) + EPS
        )
    else:
        raise ValueError(f"Unknown spectral variant: {variant}")
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return transformed / np.maximum(norms, EPS)


def fit_centroids(x: np.ndarray, y: np.ndarray, variant: str) -> np.ndarray:
    z = transform_spectra(x, variant)
    centroids = []
    for class_id in range(1, N_CLASSES + 1):
        if not np.any(y == class_id):
            raise ValueError(f"Training fold has no samples for class {class_id}")
        center = z[y == class_id].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), EPS)
        centroids.append(center)
    return np.asarray(centroids, dtype=np.float32)


def predict_centroids(x: np.ndarray, centroids: np.ndarray, variant: str) -> np.ndarray:
    scores = transform_spectra(x, variant) @ centroids.T
    return scores.argmax(axis=1).astype(np.uint8) + 1


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(confusion, (y_true.astype(int) - 1, y_pred.astype(int) - 1), 1)
    per_class = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(N_CLASSES, dtype=float),
        where=confusion.sum(axis=1) > 0,
    )
    return {
        "oa": float(np.trace(confusion) / confusion.sum()),
        "aa": float(per_class.mean()),
        "per_class_accuracy": [float(x) for x in per_class],
        "confusion": confusion.tolist(),
    }


def extract_training(
    raw_dir: Path, spatial_radius: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train = load_label(raw_dir / "train_label.mat")
    val = load_label(raw_dir / "val_label.mat")
    if train.shape != val.shape or np.any((train > 0) & (val > 0)):
        raise ValueError("Training and validation label maps must be same-shaped and disjoint")
    folds_map = connected_region_folds(val)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    with HsiCube(raw_dir / "data_hsi.mat", expected_shape=train.shape) as cube:
        if spatial_radius == 0:
            for spectra, labels, rows, cols in iter_labeled(cube, val):
                x_parts.append(spectra)
                y_parts.append(labels)
                fold_parts.append(folds_map[rows, cols])
                source_parts.append(np.ones(len(labels), dtype=np.uint8))
            for spectra, labels, rows, cols in iter_labeled(cube, train):
                x_parts.append(spectra)
                y_parts.append(labels)
                fold_parts.append(np.full(len(labels), -1, dtype=np.int8))
                source_parts.append(np.zeros(len(labels), dtype=np.uint8))
        else:
            combined = np.where(val > 0, val, train)
            for start in range(0, cube.layout.rows, 64):
                stop = min(start + 64, cube.layout.rows)
                labels = combined[start:stop]
                local_rows, cols = np.nonzero(labels)
                if not len(local_rows):
                    continue
                block = read_spatial_mean_rows(cube, start, stop, spatial_radius)
                rows = local_rows + start
                x_parts.append(block[local_rows, cols])
                y_parts.append(labels[local_rows, cols])
                fold_parts.append(folds_map[rows, cols])
                source_parts.append((val[rows, cols] > 0).astype(np.uint8))
        layout = cube.layout.__dict__
    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(fold_parts),
        {
            "source": np.concatenate(source_parts),
            "layout": layout,
            "train_nonzero": int((train > 0).sum()),
            "val_nonzero": int((val > 0).sum()),
            "spatial_radius": spatial_radius,
        },
    )


def audit_command(args: argparse.Namespace) -> None:
    raw_dir, output = Path(args.raw_dir), Path(args.report)
    expected = {
        "scene_info.csv": 123,
        "train_label.mat": 10484,
        "val_label.mat": 95708,
        "data_hsi.mat": 1497412123,
        "test_scene1.mat": 1637435207,
        "test_scene2.mat": 879370957,
    }
    files: dict[str, object] = {}
    for name, expected_size in expected.items():
        path = raw_dir / name
        info: dict[str, object] = {"exists": path.exists(), "expected_size": expected_size}
        if path.exists():
            info.update({"size": path.stat().st_size, "sha256": sha256_file(path)})
            info["size_matches"] = path.stat().st_size == expected_size
        files[name] = info
    train = load_label(raw_dir / "train_label.mat")
    val = load_label(raw_dir / "val_label.mat")
    scenes = read_scene_info(raw_dir / "scene_info.csv")
    report: dict[str, object] = {
        "files": files,
        "scenes": scenes,
        "sample_submission_expected_rows": sum(int(scene["pixel_count"]) for scene in scenes),
        "labels": {
            "shape": list(train.shape),
            "overlap": int(((train > 0) & (val > 0)).sum()),
            "train_counts": np.bincount(train.ravel(), minlength=N_CLASSES + 1).tolist(),
            "val_counts": np.bincount(val.ravel(), minlength=N_CLASSES + 1).tolist(),
        },
    }
    if (raw_dir / "data_hsi.mat").exists() and (raw_dir / "data_hsi.mat").stat().st_size == expected["data_hsi.mat"]:
        with HsiCube(raw_dir / "data_hsi.mat", expected_shape=train.shape) as cube:
            report["train_cube_layout"] = cube.layout.__dict__
            row_mean_parts = []
            # Read each chunk once. Row-mean variation exposes systematic
            # horizontal striping without multiplying I/O by the band count.
            for start in range(0, cube.layout.rows, 128):
                block = cube.read_rows(start, min(start + 128, cube.layout.rows))
                row_mean_parts.append(block.mean(axis=1))
            row_means = np.concatenate(row_mean_parts, axis=0)
            stripe_stats = np.std(np.diff(row_means, axis=0), axis=0).astype(float).tolist()
            report["row_stripe_diff_std_by_band"] = stripe_stats
            report["largest_stripe_bands"] = np.argsort(stripe_stats)[-10:][::-1].astype(int).tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def train_command(args: argparse.Namespace) -> None:
    started = time.time()
    raw_dir, model_path, report_path = Path(args.raw_dir), Path(args.model), Path(args.report)
    x, y, folds, metadata = extract_training(raw_dir)
    source = np.asarray(metadata.pop("source"))
    variants = ["raw_l2", "snv_l2", "diff_l2", "snv_no80_l2"]
    results: dict[str, object] = {}
    for variant in variants:
        oof = np.zeros(len(y), dtype=np.uint8)
        evaluated = folds >= 0
        for fold in sorted(set(folds[evaluated].tolist())):
            training = (folds != fold) | (source == 0)
            validation = folds == fold
            centers = fit_centroids(x[training], y[training], variant)
            oof[validation] = predict_centroids(x[validation], centers, variant)
        results[variant] = metrics(y[evaluated], oof[evaluated])
    best = max(variants, key=lambda name: (results[name]["aa"], results[name]["oa"]))
    centroids = fit_centroids(x, y, best)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        centroids=centroids,
        variant=np.asarray(best),
        seed=np.asarray(SEED),
        n_bands=np.asarray(N_BANDS),
        n_classes=np.asarray(N_CLASSES),
    )
    report = {
        "seed": SEED,
        "validation": "two-fold, per-class connected-region holdout; train seeds included in both training folds",
        "selection_metric": "AA then OA",
        "selected_variant": best,
        "samples": int(len(y)),
        "runtime_seconds": time.time() - started,
        "data": metadata,
        "variants": results,
        "model_sha256": sha256_file(model_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def predict_command(args: argparse.Namespace) -> None:
    raw_dir, model_path, output = Path(args.raw_dir), Path(args.model), Path(args.output)
    model = np.load(model_path, allow_pickle=False)
    centroids = model["centroids"]
    variant = str(model["variant"].item())
    scenes = read_scene_info(raw_dir / "scene_info.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,label\n")
        for scene in scenes:
            scene_name = str(scene["scene"])
            expected_shape = (int(scene["height"]), int(scene["width"]))
            path = raw_dir / f"test_{scene_name}.mat"
            with HsiCube(path, expected_shape=expected_shape) as cube:
                flat_index = 0
                for start in range(0, cube.layout.rows, args.chunk_rows):
                    stop = min(start + args.chunk_rows, cube.layout.rows)
                    block = cube.read_rows(start, stop)
                    predictions = predict_centroids(block.reshape(-1, N_BANDS), centroids, variant)
                    lines = "".join(
                        f"{scene_name}_{index:08d},{int(label)}\n"
                        for index, label in enumerate(predictions, start=flat_index)
                    )
                    handle.write(lines)
                    flat_index += len(predictions)
                if flat_index != int(scene["pixel_count"]):
                    raise RuntimeError(f"Generated {flat_index} rows for {scene_name}, expected {scene['pixel_count']}")
                total += flat_index
    print(json.dumps({"output": str(output), "rows": total, "sha256": sha256_file(output)}, indent=2))


def validate_command(args: argparse.Namespace) -> None:
    raw_dir, submission = Path(args.raw_dir), Path(args.submission)
    scenes = read_scene_info(raw_dir / "scene_info.csv")
    expected_total = sum(int(scene["pixel_count"]) for scene in scenes)
    expected_scene = 0
    expected_index = 0
    row_count = 0
    with submission.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["id", "label"]:
            raise ValueError(f"Invalid header: {header}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"Row {row_number} has {len(row)} columns")
            while expected_scene < len(scenes) and expected_index == int(scenes[expected_scene]["pixel_count"]):
                expected_scene += 1
                expected_index = 0
            if expected_scene >= len(scenes):
                raise ValueError(f"Unexpected extra row {row_number}")
            expected_id = f"{scenes[expected_scene]['scene']}_{expected_index:08d}"
            if row[0] != expected_id:
                raise ValueError(f"ID mismatch at row {row_number}: {row[0]} != {expected_id}")
            try:
                label = int(row[1])
            except ValueError as exc:
                raise ValueError(f"Non-integer label at row {row_number}: {row[1]}") from exc
            if not 1 <= label <= N_CLASSES or row[1] != str(label):
                raise ValueError(f"Invalid label at row {row_number}: {row[1]}")
            expected_index += 1
            row_count += 1
    if row_count != expected_total:
        raise ValueError(f"Row count {row_count} != {expected_total}")
    print(json.dumps({"valid": True, "rows": row_count, "sha256": sha256_file(submission)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--raw-dir", default="raw")
    audit.add_argument("--report", default="reports/data_audit.json")
    audit.set_defaults(func=audit_command)

    train = subparsers.add_parser("train")
    train.add_argument("--raw-dir", default="raw")
    train.add_argument("--model", default="artifacts/sam_centroids.npz")
    train.add_argument("--report", default="reports/oof_sam.json")
    train.set_defaults(func=train_command)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--raw-dir", default="raw")
    predict.add_argument("--model", default="artifacts/sam_centroids.npz")
    predict.add_argument("--output", default="submissions/sam_v1.csv")
    predict.add_argument("--chunk-rows", type=int, default=64)
    predict.set_defaults(func=predict_command)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--raw-dir", default="raw")
    validate.add_argument("--submission", required=True)
    validate.set_defaults(func=validate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
