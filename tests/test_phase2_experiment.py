import unittest

import numpy as np

from src.phase2_experiment import N_FOLDS, TILE_SIZE, advancement_gate, spatial_folds


class Phase2ExperimentTests(unittest.TestCase):
    def test_spatial_tiles_never_cross_folds_and_every_fold_has_every_class(self):
        rows = []
        cols = []
        labels = []
        # Eight disjoint tiles per class make strict four-fold class coverage feasible.
        for class_id in range(1, 18):
            for replica in range(8):
                tile = (class_id - 1) * 8 + replica
                rows.append((tile // 24) * TILE_SIZE + 3)
                cols.append((tile % 24) * TILE_SIZE + 5)
                labels.append(class_id)
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.uint8)
        folds, groups = spatial_folds(rows, cols, labels)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups == group])), 1)
        for fold in range(N_FOLDS):
            self.assertEqual(set(labels[folds == fold].tolist()), set(range(1, 18)))

    def test_advancement_gate_requires_every_condition(self):
        reference = {
            "overall": {"oa": 0.30, "aa": 0.30},
            "worst_fold_oa": 0.25,
            "worst_fold_aa": 0.25,
        }
        candidate = {
            "overall": {"oa": 0.34, "aa": 0.34},
            "worst_fold_oa": 0.24,
            "worst_fold_aa": 0.24,
            "prediction_counts": [1] * 17,
        }
        self.assertTrue(advancement_gate(reference, candidate)["offline_go"])
        candidate["prediction_counts"][-1] = 0
        self.assertFalse(advancement_gate(reference, candidate)["offline_go"])


if __name__ == "__main__":
    unittest.main()
