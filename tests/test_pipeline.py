import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.pipeline import (
    HsiCube,
    connected_region_folds,
    read_scene_info,
    read_local_stats_rows,
    read_spatial_mean_rows,
    transform_spectra,
)
from src.tree_experiment import features


class PipelineTests(unittest.TestCase):
    def test_hsi_cube_maps_matlab_axis_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.mat"
            raw = np.arange(98 * 4 * 3, dtype=np.uint16).reshape(98, 4, 3)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("data", data=raw)
            with HsiCube(path, expected_shape=(3, 4)) as cube:
                got = cube.read_rows(0, 3)
            self.assertEqual(got.shape, (3, 4, 98))
            self.assertTrue(np.array_equal(got, np.moveaxis(raw, (2, 1, 0), (0, 1, 2))))

    def test_hsi_cube_reads_canonical_spatial_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.mat"
            canonical = np.arange(5 * 6 * 98, dtype=np.uint16).reshape(5, 6, 98)
            raw = np.moveaxis(canonical, (0, 1, 2), (2, 1, 0))
            with h5py.File(path, "w") as handle:
                handle.create_dataset("data", data=raw)
            with HsiCube(path, expected_shape=(5, 6)) as cube:
                got = cube.read_window(1, 4, 2, 6)
            np.testing.assert_array_equal(got, canonical[1:4, 2:6])

    def test_connected_components_do_not_cross_folds(self):
        # Give every class two isolated pixels in a wide canvas.
        labels = np.zeros((4, 40), dtype=np.uint8)
        for class_id in range(1, 18):
            labels[0, 2 * class_id] = class_id
            labels[3, 2 * class_id] = class_id
        folds = connected_region_folds(labels, n_folds=2)
        for class_id in range(1, 18):
            self.assertEqual(set(folds[labels == class_id].tolist()), {0, 1})

    def test_spectral_transforms_are_unit_length(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(5, 98)).astype(np.float32)
        for variant in ("raw_l2", "snv_l2", "diff_l2", "snv_no80_l2"):
            got = transform_spectra(x, variant)
            self.assertTrue(np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5))

    def test_robust_tree_features_are_affine_invariant(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(100, 10_000, size=(5, 98)).astype(np.float32)
        expected = features(x, "robust")
        actual = features(x * 1.7 + 230.0, "robust")
        self.assertEqual(expected.shape, (5, 195))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_spatial_mean_is_exact_across_requested_row_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.mat"
            canonical = np.arange(5 * 6 * 98, dtype=np.uint16).reshape(5, 6, 98)
            raw = np.moveaxis(canonical, (0, 1, 2), (2, 1, 0))
            with h5py.File(path, "w") as handle:
                handle.create_dataset("data", data=raw)
            with HsiCube(path, expected_shape=(5, 6)) as cube:
                whole = read_spatial_mean_rows(cube, 0, 5, 1)
                chunked = np.concatenate(
                    (
                        read_spatial_mean_rows(cube, 0, 2, 1),
                        read_spatial_mean_rows(cube, 2, 4, 1),
                        read_spatial_mean_rows(cube, 4, 5, 1),
                    ),
                    axis=0,
                )
            np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-5)

    def test_local_stats_match_clipped_windows_across_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.mat"
            rng = np.random.default_rng(9)
            canonical = rng.integers(0, 1000, size=(5, 6, 98), dtype=np.uint16)
            raw = np.moveaxis(canonical, (0, 1, 2), (2, 1, 0))
            with h5py.File(path, "w") as handle:
                handle.create_dataset("data", data=raw)
            with HsiCube(path, expected_shape=(5, 6)) as cube:
                center, mean, std = read_local_stats_rows(cube, 1, 4, 1)
            np.testing.assert_array_equal(center, canonical[1:4].astype(np.float32))
            for local_row, row in enumerate(range(1, 4)):
                for col in range(6):
                    patch = canonical[
                        max(0, row - 1) : min(5, row + 2),
                        max(0, col - 1) : min(6, col + 2),
                    ].astype(np.float32)
                    np.testing.assert_allclose(mean[local_row, col], patch.mean((0, 1)), atol=1e-3)
                    np.testing.assert_allclose(std[local_row, col], patch.std((0, 1)), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
