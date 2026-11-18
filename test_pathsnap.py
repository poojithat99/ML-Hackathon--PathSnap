import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from experiments import arc_metrics
from pathsnap import CandidateConfig, DecoderConfig, Trip, candidate_search, decode, load_trip


class PathSnapTests(unittest.TestCase):
    def test_radius_has_nearest_fallback_and_unique_eligible_candidates(self):
        starts = np.array([[0., 0.], [0., 20.]])
        ends = starts + [100., 0.]
        candidates, distances, eligible = candidate_search(
            np.array([[0., 100.], [0., 20.]]), starts, ends, np.array([100., 100.]),
            CandidateConfig(2, 1, "sampled"))
        np.testing.assert_array_equal(candidates[:, 0], [1, 1])
        np.testing.assert_array_equal(eligible, [[True, False], [True, False]])
        np.testing.assert_allclose(distances[:, 0], [80, 0])

    def test_single_arc_sampled_search_masks_duplicates(self):
        starts = np.array([[0., 0.]])
        candidates, _, eligible = candidate_search(
            np.array([[0., 0.]]), starts, starts + [100., 0.], np.array([100.]),
            CandidateConfig(8, index="sampled"))
        self.assertEqual(candidates[eligible].tolist(), [0])

    def test_features_without_reference_and_large_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "00000000"
            folder.mkdir()
            prefix = folder / folder.name
            prefix.with_suffix(".nodes").write_text("0 0\n0.001 0\n")
            prefix.with_suffix(".arcs").write_text("0 1\n")
            prefix.with_suffix(".track").write_text("0 0 0\n0.0005 0 100\n")
            trip = load_trip(folder, CandidateConfig(), "combined")
            self.assertEqual(trip.features.shape, (2, 1, 10))
            np.testing.assert_allclose(trip.features[:, :, [1, 3, 4, 5, 6, 7]], 0)

    def test_gap_backtrace_and_masked_state(self):
        trip = Trip(np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 20.]]),
                    np.zeros((4, 2)), np.array([[0, 1], [2, 3]]),
                    np.tile([0, 1], (3, 1)), np.empty((3, 2, 0)))
        values = np.array([[.9, .1], [.9, .1], [0, .9]])
        np.testing.assert_array_equal(decode(trip, values, DecoderConfig()), [0, 0, 1])
        np.testing.assert_array_equal(
            decode(trip, values, DecoderConfig(jump_penalty=6, motion_weight=1.5)), [0, 0, 1])

    def test_arc_metrics_match_sklearn(self):
        reference = [2, 2, 2, 4, 4, 4, 4]
        predicted = [2, 5, 2, 5, 4, 4, 4]
        metrics = arc_metrics(reference, predicted)
        for average in ("macro", "micro", "weighted"):
            self.assertAlmostEqual(metrics[f"{average}_f1"], f1_score(
                reference, predicted, average=average))
        for metric, calculate in (("precision", precision_score), ("recall", recall_score)):
            self.assertAlmostEqual(metrics[f"macro_{metric}"], calculate(
                reference, predicted, average="macro", zero_division=0))


if __name__ == "__main__":
    unittest.main()
