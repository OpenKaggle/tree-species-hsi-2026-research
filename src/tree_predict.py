"""Stream LightGBM inference over the two official test scenes."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.pipeline import HsiCube, N_BANDS, read_scene_info, read_spatial_mean_rows, sha256_file
from src.tree_experiment import features


FEATURE_COUNTS = {"full": 152, "robust": 195}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--model", default="artifacts/lightgbm_v1.txt")
    parser.add_argument("--output", default="submissions/lightgbm_v1.csv")
    parser.add_argument("--report", default="reports/predict_lightgbm_v1.json")
    parser.add_argument("--chunk-rows", type=int, default=16)
    parser.add_argument("--feature-mode", choices=("auto", "full", "robust"), default="auto")
    parser.add_argument("--spatial-radius", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    raw_dir = Path(args.raw_dir)
    model_path = Path(args.model)
    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    scenes = read_scene_info(raw_dir / "scene_info.csv")
    booster = lgb.Booster(model_file=str(model_path))
    if args.feature_mode == "auto":
        matches = [name for name, count in FEATURE_COUNTS.items() if count == booster.num_feature()]
        if len(matches) != 1:
            raise ValueError(f"Cannot infer feature mode from {booster.num_feature()} features")
        feature_mode = matches[0]
    else:
        feature_mode = args.feature_mode
        if FEATURE_COUNTS[feature_mode] != booster.num_feature():
            raise ValueError(
                f"Model has {booster.num_feature()} features, but {feature_mode} "
                f"produces {FEATURE_COUNTS[feature_mode]}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    scene_counts: dict[str, list[int]] = {}
    with partial.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,label\n")
        for scene in scenes:
            scene_name = str(scene["scene"])
            expected_shape = (int(scene["height"]), int(scene["width"]))
            counts = np.zeros(18, dtype=np.int64)
            with HsiCube(raw_dir / f"test_{scene_name}.mat", expected_shape) as cube:
                flat_index = 0
                for start in range(0, cube.layout.rows, args.chunk_rows):
                    stop = min(start + args.chunk_rows, cube.layout.rows)
                    block = read_spatial_mean_rows(
                        cube, start, stop, args.spatial_radius
                    ).reshape(-1, N_BANDS)
                    probabilities = booster.predict(features(block, feature_mode))
                    predictions = probabilities.argmax(axis=1).astype(np.uint8) + 1
                    counts += np.bincount(predictions, minlength=18)
                    handle.write(
                        "".join(
                            f"{scene_name}_{index:08d},{int(label)}\n"
                            for index, label in enumerate(predictions, start=flat_index)
                        )
                    )
                    flat_index += len(predictions)
                if flat_index != int(scene["pixel_count"]):
                    raise RuntimeError(
                        f"Generated {flat_index} rows for {scene_name}; "
                        f"expected {scene['pixel_count']}"
                    )
                total += flat_index
                scene_counts[scene_name] = counts[1:].astype(int).tolist()
    os.replace(partial, output)

    report = {
        "output": str(output),
        "rows": total,
        "feature_mode": feature_mode,
        "spatial_radius": args.spatial_radius,
        "class_counts_by_scene": scene_counts,
        "runtime_seconds": time.time() - started,
        "model_sha256": sha256_file(model_path),
        "submission_sha256": sha256_file(output),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
