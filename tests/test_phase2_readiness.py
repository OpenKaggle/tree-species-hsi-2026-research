import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from src.phase2_readiness import discover_contract, read_manifest, run_gate


def write_manifest(path: Path, rows: list[tuple[str, int, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "size_bytes", "creation_date_utc"])
        writer.writerows(rows)


class Phase2ReadinessTests(unittest.TestCase):
    def test_accepts_kaggle_cli_csv_column_names(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "cli.csv"
            manifest.write_text(
                "Warning: CLI update available, please upgrade\n"
                "name,size,creationDate\nfile.mat,42,2026-09-21 00:00:00\n",
                encoding="utf-8",
            )
            parsed = read_manifest(manifest)
            self.assertEqual(parsed["file.mat"].size_bytes, 42)

    def test_identical_inventory_waits_without_touching_raw_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            write_manifest(manifest, [("phase1.bin", 123, "2026-07-25")])
            report = run_gate(manifest, manifest, root / "does-not-exist")
            self.assertEqual(report["status"], "WAIT_PHASE2_NOT_RELEASED")
            self.assertEqual(report["decision"], "NO_TRAIN_NO_INFERENCE_NO_SUBMISSION")

    def test_changed_inventory_missing_payload_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, current = root / "baseline.csv", root / "current.csv"
            write_manifest(baseline, [("phase1.bin", 1, "old")])
            write_manifest(current, [("phase2.bin", 7, "new")])
            raw = root / "raw"
            raw.mkdir()
            report = run_gate(baseline, current, raw)
            self.assertEqual(report["status"], "BLOCKED_PHASE2_PAYLOAD_INCOMPLETE")
            self.assertIn("missing: phase2.bin", report["errors"])

    def test_dynamic_contract_discovers_new_bands_classes_scenes_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            labels = np.asarray([[0, 1, 2], [3, 1, 2]], dtype=np.uint8)
            savemat(raw / "phase2_label.mat", {"labels": labels})
            with h5py.File(raw / "phase2_cube.mat", "w") as handle:
                handle.create_dataset("cube", data=np.zeros((4, 3, 2), dtype=np.uint16))
            (raw / "sample_submission.csv").write_text(
                "id,label\n"
                + "".join(f"newscene_{index:08d},1\n" for index in range(6)),
                encoding="utf-8",
            )
            rows = [
                (path.name, path.stat().st_size, "2026-09-21")
                for path in sorted(raw.iterdir())
            ]
            baseline, current = root / "baseline.csv", root / "current.csv"
            write_manifest(baseline, [("phase1.bin", 1, "2026-07-25")])
            write_manifest(current, rows)
            report = run_gate(baseline, current, raw, full_hash=True)
            self.assertEqual(report["status"], "READY_FOR_PHASE2_VALIDATION")
            self.assertEqual(report["contract"]["n_bands"], 4)
            self.assertEqual(report["contract"]["class_ids"], [1, 2, 3])
            self.assertEqual(report["contract"]["submission"]["scene_order"], ["newscene"])
            self.assertEqual(report["contract"]["submission"]["rows"], 6)
            self.assertEqual(report["old_artifact_reuse"], "BLOCKED_CLASS_OR_BAND_CONTRACT_CHANGED")
            self.assertTrue(all("sha256" in item for item in report["local_payload"]))

    def test_non_contiguous_submission_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            labels = np.asarray([[1, 2], [1, 2]], dtype=np.uint8)
            savemat(raw / "label.mat", {"labels": labels})
            with h5py.File(raw / "cube.mat", "w") as handle:
                handle.create_dataset("cube", data=np.zeros((5, 2, 2), dtype=np.uint16))
            (raw / "sample_submission.csv").write_text(
                "id,label\nscene_00000000,1\nscene_00000002,1\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Non-contiguous index"):
                discover_contract(raw)

    def test_nested_phase2_files_are_discovered_and_audited_by_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            nested = raw / "Stage2" / "Stage2"
            nested.mkdir(parents=True)
            labels = np.asarray([[0, 1, 2], [3, 1, 2]], dtype=np.uint8)
            savemat(nested / "train_label.mat", {"labels": labels})
            with h5py.File(nested / "data.mat", "w") as handle:
                handle.create_dataset("cube", data=np.zeros((4, 3, 2), dtype=np.uint16))
            (raw / "sample_submission.csv").write_text(
                "id,label\n" + "".join(f"scene_{index:08d},1\n" for index in range(6)),
                encoding="utf-8",
            )
            rows = [
                (path.relative_to(raw).as_posix(), path.stat().st_size, "2026-09-21")
                for path in sorted(raw.rglob("*"))
                if path.is_file()
            ]
            baseline, current = root / "baseline.csv", root / "current.csv"
            write_manifest(baseline, [("phase1.bin", 1, "2026-07-25")])
            write_manifest(current, rows)
            report = run_gate(baseline, current, raw, full_hash=True)
            self.assertEqual(report["status"], "READY_FOR_PHASE2_VALIDATION")
            self.assertEqual(report["contract"]["n_bands"], 4)
            self.assertEqual(
                {item["file"] for item in report["contract"]["mat_inventory"]},
                {"Stage2/Stage2/data.mat", "Stage2/Stage2/train_label.mat"},
            )
            self.assertEqual(
                {item["name"] for item in report["local_payload"]},
                {
                    "Stage2/Stage2/data.mat",
                    "Stage2/Stage2/train_label.mat",
                    "sample_submission.csv",
                },
            )


if __name__ == "__main__":
    unittest.main()
