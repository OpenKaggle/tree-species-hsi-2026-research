"""Fail-closed Phase 2 data-contract gate for Tree Species HSI 2026.

This module never trains a model and never creates or uploads a submission.  It
answers a narrower question: has the official file inventory changed, and if
so is the locally mounted payload complete, frozen, and structurally safe to
hand to a Phase 2 training pipeline?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
from scipy.io import loadmat, whosmat


BASELINE_CLASS_IDS = tuple(range(1, 18))
BASELINE_BANDS = 98
ID_PATTERN = re.compile(r"^(?P<scene>.+)_(?P<index>\d+)$")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    size_bytes: int
    creation_date_utc: str


def read_manifest(path: Path) -> dict[str, ManifestEntry]:
    entries: dict[str, ManifestEntry] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        # Kaggle CLI 2.2.0 may print an upgrade warning to stdout before its
        # CSV header. Ignore only preamble lines; never ignore malformed rows
        # after the recognized header.
        lines = handle.readlines()
        header_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip()
                in {"name,size_bytes,creation_date_utc", "name,size,creationDate"}
            ),
            None,
        )
        if header_index is None:
            raise ValueError(f"No recognized manifest header in {path}")
        reader = csv.DictReader(lines[header_index:])
        normalized = {"name", "size_bytes", "creation_date_utc"}
        kaggle_cli = {"name", "size", "creationDate"}
        fields = set(reader.fieldnames or ())
        if fields != normalized and fields != kaggle_cli:
            raise ValueError(f"Unexpected manifest columns in {path}: {reader.fieldnames}")
        size_field = "size_bytes" if fields == normalized else "size"
        date_field = "creation_date_utc" if fields == normalized else "creationDate"
        for row in reader:
            name = row["name"].strip()
            if not name or name in entries:
                raise ValueError(f"Empty or duplicate manifest name: {name!r}")
            size = int(row[size_field])
            if size < 0:
                raise ValueError(f"Negative size for {name}: {size}")
            entries[name] = ManifestEntry(name, size, row[date_field].strip())
    if not entries:
        raise ValueError(f"Manifest is empty: {path}")
    return entries


def manifest_delta(
    baseline: dict[str, ManifestEntry], current: dict[str, ManifestEntry]
) -> dict[str, object]:
    baseline_names, current_names = set(baseline), set(current)
    changed = sorted(
        name
        for name in baseline_names & current_names
        if (
            baseline[name].size_bytes,
            baseline[name].creation_date_utc,
        )
        != (current[name].size_bytes, current[name].creation_date_utc)
    )
    return {
        "added": sorted(current_names - baseline_names),
        "removed": sorted(baseline_names - current_names),
        "changed": changed,
    }


def delta_is_empty(delta: dict[str, object]) -> bool:
    return not any(delta[key] for key in ("added", "removed", "changed"))


def _numeric_hdf5_shapes(path: Path) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, value: h5py.Dataset | h5py.Group) -> None:
            if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number):
                found.append(
                    {"dataset": name, "shape": [int(x) for x in value.shape], "dtype": str(value.dtype)}
                )

        handle.visititems(visit)
    return found


def numeric_mat_shapes(path: Path) -> list[dict[str, object]]:
    if h5py.is_hdf5(path):
        return _numeric_hdf5_shapes(path)
    return [
        {"dataset": name, "shape": [int(x) for x in shape], "dtype": kind}
        for name, shape, kind in whosmat(path)
        if kind not in {"cell", "struct", "object", "char"}
    ]


def read_small_label_contract(
    path: Path,
    dataset: str,
    file_name: str | None = None,
    max_bytes: int = 128 * 1024 * 1024,
) -> dict[str, object]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Refusing to load label candidate over {max_bytes} bytes: {path.name}")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle[dataset])
    else:
        values = np.asarray(loadmat(path, variable_names=[dataset])[dataset])
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Label candidate is not a numeric 2-D array: {path.name}:{dataset}")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"Label candidate contains non-finite or non-integral values: {path.name}")
    class_ids = sorted(int(x) for x in np.unique(values) if int(x) != 0)
    if class_ids and class_ids != list(range(1, max(class_ids) + 1)):
        raise ValueError(f"Non-contiguous class IDs in {path.name}: {class_ids}")
    return {
        "file": file_name or path.name,
        "dataset": dataset,
        "shape": [int(x) for x in values.shape],
        "class_ids": class_ids,
        "labeled_pixels": int(np.count_nonzero(values)),
        "sha256": sha256_file(path),
    }


def inspect_sample_submission(path: Path) -> dict[str, object]:
    row_count = 0
    scene_order: list[str] = []
    scene_counts: dict[str, int] = {}
    expected_next: dict[str, int] = {}
    placeholder_labels: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["id", "label"]:
            raise ValueError(f"Submission header must be id,label; got {header}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"Submission row {line_number} has {len(row)} columns")
            match = ID_PATTERN.fullmatch(row[0])
            if match is None:
                raise ValueError(f"Submission ID does not end in _<integer>: {row[0]!r}")
            scene, index = match.group("scene"), int(match.group("index"))
            if scene not in scene_counts:
                scene_order.append(scene)
                scene_counts[scene] = 0
                expected_next[scene] = 0
            elif scene != scene_order[-1]:
                raise ValueError(f"Scene {scene!r} is split into non-contiguous blocks")
            if index != expected_next[scene]:
                raise ValueError(
                    f"Non-contiguous index for {scene!r}: {index} != {expected_next[scene]}"
                )
            try:
                label = int(row[1])
            except ValueError as exc:
                raise ValueError(f"Non-integer placeholder label at row {line_number}") from exc
            if str(label) != row[1]:
                raise ValueError(f"Non-canonical placeholder label at row {line_number}: {row[1]!r}")
            placeholder_labels.add(label)
            expected_next[scene] += 1
            scene_counts[scene] += 1
            row_count += 1
    if row_count == 0:
        raise ValueError("Sample submission has no prediction rows")
    return {
        "file": path.name,
        "header": ["id", "label"],
        "rows": row_count,
        "scene_order": scene_order,
        "scene_counts": scene_counts,
        "placeholder_labels": sorted(placeholder_labels),
        "id_contract": "contiguous scene blocks; zero-based integer suffix",
        "sha256": sha256_file(path),
    }


def discover_contract(raw_dir: Path) -> dict[str, object]:
    mat_inventory: list[dict[str, object]] = []
    label_candidates: list[dict[str, object]] = []
    cube_candidates: list[dict[str, object]] = []
    # Organizer files may be nested (the 2026 Phase 2 release uses
    # Stage2/Stage2/*.mat). Preserve relative paths so same-basename Phase 1
    # and Phase 2 files remain distinguishable in the audit receipt.
    for path in sorted(raw_dir.rglob("*.mat")):
        relative_name = path.relative_to(raw_dir).as_posix()
        shapes = numeric_mat_shapes(path)
        mat_inventory.append({"file": relative_name, "datasets": shapes})
        for item in shapes:
            shape = item["shape"]
            if len(shape) == 2:
                label_candidates.append(
                    read_small_label_contract(path, str(item["dataset"]), relative_name)
                )
            elif len(shape) == 3:
                cube_candidates.append({"file": relative_name, **item})

    labeled = [item for item in label_candidates if item["class_ids"]]
    if not labeled:
        raise ValueError("No non-empty 2-D integer label arrays found")
    # A legitimate validation split may omit a rare class, so require the
    # union—not every individual file—to be a contiguous canonical label set.
    class_ids = sorted({class_id for item in labeled for class_id in item["class_ids"]})
    if class_ids != list(range(1, max(class_ids) + 1)):
        raise ValueError(f"Union of label files is non-contiguous: {class_ids}")

    label_shapes = {tuple(item["shape"]) for item in labeled}
    inferred_bands: set[int] = set()
    for cube in cube_candidates:
        shape = tuple(int(x) for x in cube["shape"])
        for label_shape in label_shapes:
            for band_axis in range(3):
                spatial = tuple(shape[i] for i in range(3) if i != band_axis)
                if spatial in {label_shape, label_shape[::-1]}:
                    inferred_bands.add(shape[band_axis])
    if len(inferred_bands) != 1:
        raise ValueError(f"Cannot uniquely infer band count from labels/cubes: {sorted(inferred_bands)}")

    samples = sorted(raw_dir.rglob("*sample*submission*.csv"))
    if len(samples) != 1:
        raise ValueError(f"Expected exactly one sample submission; found {[p.name for p in samples]}")
    sample = inspect_sample_submission(samples[0])
    return {
        "mat_inventory": mat_inventory,
        "labels": labeled,
        "class_ids": class_ids,
        "n_classes": len(class_ids),
        "n_bands": next(iter(inferred_bands)),
        "cubes": cube_candidates,
        "submission": sample,
    }


def validate_local_payload(
    raw_dir: Path,
    current: dict[str, ManifestEntry],
    full_hash: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    files: list[dict[str, object]] = []
    errors: list[str] = []
    for name in sorted(current):
        entry = current[name]
        path = raw_dir / name
        record: dict[str, object] = {
            "name": name,
            "expected_size": entry.size_bytes,
            "exists": path.is_file(),
        }
        if path.is_file():
            record["size"] = path.stat().st_size
            record["size_matches"] = path.stat().st_size == entry.size_bytes
            if not record["size_matches"]:
                errors.append(f"size mismatch: {name}")
            if full_hash:
                record["sha256"] = sha256_file(path)
        else:
            errors.append(f"missing: {name}")
        files.append(record)
    extras = sorted(
        path.relative_to(raw_dir).as_posix()
        for path in raw_dir.rglob("*")
        if path.is_file() and path.relative_to(raw_dir).as_posix() not in current
    )
    if extras:
        errors.append(f"unmanifested files present: {extras}")
    return files, errors


def run_gate(
    baseline_manifest: Path,
    current_manifest: Path,
    raw_dir: Path,
    full_hash: bool = False,
) -> dict[str, object]:
    baseline = read_manifest(baseline_manifest)
    current = read_manifest(current_manifest)
    delta = manifest_delta(baseline, current)
    report: dict[str, object] = {
        "competition": "tree-species-hsi-2026",
        "purpose": "Phase 2 data-contract readiness only; no training, inference, or submission",
        "baseline_manifest": str(baseline_manifest),
        "current_manifest": str(current_manifest),
        "baseline_manifest_signature": canonical_sha256(
            {name: entry.__dict__ for name, entry in sorted(baseline.items())}
        ),
        "current_manifest_signature": canonical_sha256(
            {name: entry.__dict__ for name, entry in sorted(current.items())}
        ),
        "manifest_delta": delta,
        "full_local_hashes_required_for_ready": True,
        "full_local_hashes_requested": full_hash,
        "legal_data_boundary": {
            "allowed": [
                "organizer-provided competition files",
                "public pretrained resources only after license and disclosure audit",
            ],
            "forbidden": [
                "test labels or inferred manual labels",
                "private or undeclared external datasets",
                "using leaderboard feedback to select validation thresholds",
            ],
        },
    }
    if delta_is_empty(delta):
        report.update(
            {
                "status": "WAIT_PHASE2_NOT_RELEASED",
                "decision": "NO_TRAIN_NO_INFERENCE_NO_SUBMISSION",
                "reason": "Official inventory is byte/date-identical to the frozen Phase 1 manifest.",
                "old_artifact_reuse": "BLOCKED_UNTIL_PHASE2_CONTRACT_EXISTS",
            }
        )
        return report

    files, payload_errors = validate_local_payload(raw_dir, current, full_hash)
    report["local_payload"] = files
    if payload_errors:
        report.update(
            {
                "status": "BLOCKED_PHASE2_PAYLOAD_INCOMPLETE",
                "decision": "NO_TRAIN_NO_INFERENCE_NO_SUBMISSION",
                "errors": payload_errors,
                "old_artifact_reuse": "BLOCKED_MANIFEST_CHANGED",
            }
        )
        return report
    if not full_hash:
        report.update(
            {
                "status": "BLOCKED_FULL_HASHES_REQUIRED",
                "decision": "RERUN_CONTRACT_GATE_WITH_FULL_HASH_NO_TRAINING",
                "old_artifact_reuse": "BLOCKED_MANIFEST_CHANGED",
            }
        )
        return report

    try:
        contract = discover_contract(raw_dir)
    except (OSError, KeyError, ValueError) as exc:
        report.update(
            {
                "status": "BLOCKED_PHASE2_CONTRACT_INVALID",
                "decision": "NO_TRAIN_NO_INFERENCE_NO_SUBMISSION",
                "errors": [str(exc)],
                "old_artifact_reuse": "BLOCKED_MANIFEST_CHANGED",
            }
        )
        return report

    report["contract"] = contract
    report["contract_signature"] = canonical_sha256(contract)
    label_or_band_changed = (
        tuple(contract["class_ids"]) != BASELINE_CLASS_IDS
        or int(contract["n_bands"]) != BASELINE_BANDS
    )
    report["old_artifact_reuse"] = (
        "BLOCKED_CLASS_OR_BAND_CONTRACT_CHANGED"
        if label_or_band_changed
        else "BLOCKED_PHASE2_MANIFEST_CHANGED_AND_NO_SEMANTIC_CLASS_MAPPING_PROVIDED"
    )
    report.update(
        {
            "status": "READY_FOR_PHASE2_VALIDATION",
            "decision": "MAY_BUILD_PHASE2_FOLDS_AND_RETRAIN_FROM_SCRATCH; NO_SUBMISSION_YET",
        }
    )
    return report


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Hash every official file. Mandatory before READY_FOR_PHASE2_VALIDATION.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gate(
        args.baseline_manifest,
        args.current_manifest,
        args.raw_dir,
        full_hash=args.full_hash,
    )
    write_report(report, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
