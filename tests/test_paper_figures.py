"""Regression checks for the matched-policy figure using released measurements.

Run with: python -m unittest discover -s tests
"""
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper_final.common.figure_style import matched_figures


class MatchedPolicyFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_released_checkpoint_statistics_and_inputs(self):
        archive = Path(__file__).resolve().parents[1] / "data/module2/arrays.npz"
        with np.load(archive, allow_pickle=False) as data:
            native = data["native_reductions_pct"]
            weights = data["first_repetition_primary_weights"]
            cases = data["case_ids"]
            seeds = data["seeds"]
            totals = data["costs_sec"][..., 0].sum(axis=2)
        np.testing.assert_allclose(native, 100 * (1 - totals / totals[:, :1]))
        original_native, original_weights = native.copy(), weights.copy()
        figures = matched_figures(native, weights, cases, 0, seeds, "Test")
        self.assertEqual([name for name, _ in figures],
                         ["matched_policy_savings", "matched_policy_weights"])
        ax = figures[0][1].axes[0]
        self.assertEqual([text.get_text() for text in ax.texts],
                         ["33.4 ± 0.5", "35.0 ± 0.8", "40.7 ± 1.3", "40.2 ± 1.5"])
        for j, points in enumerate(ax.collections[::2], start=1):
            np.testing.assert_allclose(points.get_offsets()[:, 0], native[:, j])
            segments = ax.containers[j - 1].lines[2][0].get_segments()
            np.testing.assert_allclose(np.asarray(segments)[0, :, 0],
                                       native[:, j].mean() + np.array([-1, 1]) * native[:, j].std(ddof=1))
        np.testing.assert_array_equal(native, original_native)
        np.testing.assert_array_equal(weights, original_weights)
        # The compact plot's typography must not leak into the heatmap.
        self.assertEqual(figures[1][1].axes[0].title.get_fontsize(), 8)

    def test_missing_singleton_and_out_of_range_checkpoints(self):
        native = np.array([[0, np.nan, -70, 120, 300],
                           [0, np.nan, np.nan, 180, 300]], dtype=float)
        weights = np.ones((2, 5, 3, 4))
        figures = matched_figures(native, weights, np.arange(3), 0, [1, 2], "Test")
        fig = figures[0][1]
        ax = fig.axes[0]
        self.assertEqual([text.get_text() for text in ax.texts],
                         ["undefined", "-70.0", "150.0 ± 42.4", "300.0 ± 0.0"])
        self.assertLess(ax.get_xlim()[0], -70)
        self.assertGreater(ax.get_xlim()[1], 300)
        # Numerical annotations must fit within the plotting area on new data.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in ax.texts:
            self.assertLessEqual(text.get_window_extent(renderer).x1, ax.bbox.x1)

    def test_all_undefined_comparators_render(self):
        native = np.full((1, 5), np.nan)
        figures = matched_figures(native, np.ones((1, 5, 2, 3)), [1, 2], 0, [1], "Test")
        fig = figures[0][1]
        fig.canvas.draw()
        self.assertEqual([text.get_text() for text in fig.axes[0].texts], ["undefined"] * 4)


if __name__ == "__main__":
    unittest.main()
