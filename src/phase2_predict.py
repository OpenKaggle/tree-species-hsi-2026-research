"""Train and stream the one candidate authorized by the Phase 2 offline gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA

from src.phase2_experiment import (
    PCA_COMPONENTS,
    _snv,
    candidate_features_from_arrays,
    extract_sparse_features,
    new_model,
)
from src.pipeline import (
    HsiCube,
    N_CLASSES,
    read_local_stats_rows,
    read_scene_info,
    sha256_file,
)


def require_offline_go(report_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("gate", {}).get("offline_go") is not True:
        raise RuntimeError(f"Offline gate is not GO in {report_path}")
    return report


def train_final(args: argparse.Namespace) -> None:
    started = time.time()
    raw_dir = Path(args.raw_dir)
    gate_path = Path(args.gate_report)
    gate = require_offline_go(gate_path)
    cube_path = raw_dir / "Stage2" / "Stage2" / "data.mat"
    label_path = raw_dir / "Stage2" / "Stage2" / "train_label.mat"
    if gate["data"]["cube_sha256"] != sha256_file(cube_path):
        raise RuntimeError("Stage 2 cube hash changed after the offline GO decision")
    if gate["data"]["label_sha256"] != sha256_file(label_path):
        raise RuntimeError("Stage 2 label hash changed after the offline GO decision")

    data = extract_sparse_features(cube_path, label_path)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full", random_state=20260909)
    pca.fit(_snv(data.center))
    x = candidate_features_from_arrays(
        data.center, data.local_mean, data.local_std, pca.mean_, pca.components_
    )
    model = new_model()
    model.fit(x, data.labels - 1)

    model_path = Path(args.model)
    transform_path = Path(args.transform)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    np.savez_compressed(
        transform_path,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    training_predictions = model.predict(x).astype(np.uint8) + 1
    report = {
        "decision_source": str(gate_path),
        "decision_source_sha256": sha256_file(gate_path),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "transform": str(transform_path),
        "transform_sha256": sha256_file(transform_path),
        "samples": int(len(data.labels)),
        "features": int(x.shape[1]),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "training_accuracy_diagnostic_only": float((training_predictions == data.labels).mean()),
        "runtime_seconds": time.time() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def predict(args: argparse.Namespace) -> None:
    started = time.time()
    require_offline_go(Path(args.gate_report))
    raw_dir = Path(args.raw_dir)
    model_path = Path(args.model)
    transform_path = Path(args.transform)
    output_path = Path(args.output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    booster = lgb.Booster(model_file=str(model_path))
    transform = np.load(transform_path, allow_pickle=False)
    pca_mean = transform["pca_mean"]
    pca_components = transform["pca_components"]
    if booster.num_feature() != pca_components.shape[0] + 98 + 98 + 97:
        raise ValueError("Model and PCA feature contracts disagree")
    scenes = read_scene_info(raw_dir / "scene_info.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts_by_scene: dict[str, list[int]] = {}
    total = 0
    with partial_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,label\n")
        for scene in scenes:
            scene_name = str(scene["scene"])
            expected_shape = (int(scene["height"]), int(scene["width"]))
            counts = np.zeros(N_CLASSES + 1, dtype=np.int64)
            flat_index = 0
            with HsiCube(raw_dir / f"test_{scene_name}.mat", expected_shape) as cube:
                for start in range(0, cube.layout.rows, args.chunk_rows):
                    stop = min(start + args.chunk_rows, cube.layout.rows)
                    center, local_mean, local_std = read_local_stats_rows(cube, start, stop, 1)
                    x = candidate_features_from_arrays(
                        center.reshape(-1, center.shape[-1]),
                        local_mean.reshape(-1, local_mean.shape[-1]),
                        local_std.reshape(-1, local_std.shape[-1]),
                        pca_mean,
                        pca_components,
                    )
                    probabilities = booster.predict(x, num_threads=-1)
                    predictions = probabilities.argmax(axis=1).astype(np.uint8) + 1
                    counts += np.bincount(predictions, minlength=N_CLASSES + 1)
                    handle.write(
                        "".join(
                            f"{scene_name}_{index:08d},{int(label)}\n"
                            for index, label in enumerate(predictions, start=flat_index)
                        )
                    )
                    flat_index += len(predictions)
                    print(
                        json.dumps(
                            {
                                "scene": scene_name,
                                "rows_complete": stop,
                                "rows_total": cube.layout.rows,
                                "predictions": flat_index,
                            }
                        ),
                        flush=True,
                    )
                if flat_index != int(scene["pixel_count"]):
                    raise RuntimeError(
                        f"Generated {flat_index} rows for {scene_name}; "
                        f"expected {scene['pixel_count']}"
                    )
            class_counts_by_scene[scene_name] = counts[1:].astype(int).tolist()
            total += flat_index
    os.replace(partial_path, output_path)
    report = {
        "output": str(output_path),
        "rows": total,
        "class_counts_by_scene": class_counts_by_scene,
        "model_sha256": sha256_file(model_path),
        "transform_sha256": sha256_file(transform_path),
        "submission_sha256": sha256_file(output_path),
        "runtime_seconds": time.time() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--raw-dir", default="raw_phase2")
    common.add_argument(
        "--gate-report", default="reports/phase2_first_experiment_2026-09-24.json"
    )
    common.add_argument("--model", default="artifacts/phase2_pca_local_texture_lgbm.txt")
    common.add_argument("--transform", default="artifacts/phase2_pca_transform.npz")

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--report", default="reports/phase2_final_train.json")
    train.set_defaults(func=train_final)

    prediction = subparsers.add_parser("predict", parents=[common])
    prediction.add_argument("--output", default="submissions/phase2_pca_local_texture.csv")
    prediction.add_argument("--report", default="reports/phase2_predict.json")
    prediction.add_argument("--chunk-rows", type=int, default=64)
    prediction.set_defaults(func=predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
