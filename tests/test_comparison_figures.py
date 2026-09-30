"""Check source integrity and scientific bounds for recorded-result figures."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from mlhdr_ptycho.comparison_figures import (
    _bounds, _exposure_series, load_recorded_results,
    plot_method_heatmaps, plot_noise_comparison,
)


DOCS = Path(__file__).resolve().parents[1] / "docs"


class ComparisonFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load_recorded_results(DOCS)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name)
        for filename in ("summary_metrics.csv", "experiment_manifest.json"):
            (self.source / filename).write_bytes((DOCS / filename).read_bytes())

    def tearDown(self):
        plt.close("all")
        self.temp.cleanup()

    def edit_rows(self, edit):
        path = self.source / "summary_metrics.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        fields = list(rows[0])
        edit(rows)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_recorded_values_and_order_are_preserved(self):
        self.assertEqual(len(self.results.rows), 27)
        self.assertEqual(self.results.cases[:2], ("single_0.5ms", "single_1ms"))
        self.assertEqual(self.results.cases[-2:], ("paper_ml_hdr", "saturation_mask_extension"))
        self.assertEqual(self.results.total_exposure_ms, 666.5)
        self.assertEqual(self.results.noise_e("low_noise"), 5)
        self.assertEqual(self.results.value("low_noise", "saturation_mask_extension", "baseline_ssim"),
                         .8164364670251155)
        self.assertLess(self.results.value("read_noise_1adu", "single_50ms", "baseline_ssim"), 0)
        self.assertGreater(self.results.value("read_noise_1adu", "single_0.5ms", "high_q_diffraction_nrmse"), 800)

    def test_duplicate_or_missing_methods_are_rejected(self):
        self.edit_rows(lambda rows: rows.append(dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "Duplicate summary row"):
            load_recorded_results(self.source)
        self.edit_rows(lambda rows: rows.pop())
        self.edit_rows(lambda rows: rows.pop(0))
        with self.assertRaisesRegex(ValueError, "Missing summary rows"):
            load_recorded_results(self.source)

    def test_missing_sd_and_nonfinite_numbers_are_rejected(self):
        self.edit_rows(lambda rows: rows[0].update(baseline_ssim_std=""))
        with self.assertRaisesRegex(ValueError, "Missing or invalid number"):
            load_recorded_results(self.source)
        self.edit_rows(lambda rows: rows[0].update(baseline_ssim_std="nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite number"):
            load_recorded_results(self.source)

    def test_manifest_counts_and_unknown_profiles_are_rejected(self):
        self.edit_rows(lambda rows: rows[0].update(runs="2"))
        with self.assertRaisesRegex(ValueError, "Run counts disagree"):
            load_recorded_results(self.source)
        self.edit_rows(lambda rows: rows[0].update(runs="3", profile="unrecorded"))
        with self.assertRaisesRegex(ValueError, "Unexpected profile"):
            load_recorded_results(self.source)

    def test_common_limits_keep_negative_ssim_error_bar_endpoints(self):
        lo, hi = _bounds(self.results, "baseline_ssim")
        for profile, case in self.results.rows:
            mean = self.results.value(profile, case, "baseline_ssim")
            sd = self.results.value(profile, case, "baseline_ssim", "std")
            self.assertLess(lo, mean - sd)
            self.assertGreater(hi, mean + sd)
        self.assertLess(lo, -.01556)

    def test_saturation_plot_uses_only_measured_exposures_and_percent_sd(self):
        fig, ax = plt.subplots()
        _exposure_series(ax, self.results, "read_noise_1adu", "saturation_fraction",
                         include_fusion=False, factor=100)
        self.assertEqual(len(ax.containers), 7)
        last = ax.containers[-1]
        self.assertAlmostEqual(float(last.lines[0].get_ydata()[0]), .011650102749433107 * 100)
        endpoints = last.lines[2][0].get_segments()[0][:, 1]
        sd = self.results.value("read_noise_1adu", "single_500ms", "saturation_fraction", "std")
        self.assertAlmostEqual(float(endpoints[1] - endpoints[0]), 2 * sd * 100)
        self.assertEqual(len(ax.patches), 0)  # No synthetic fused saturation bands.

    def test_heatmap_cells_map_to_the_recorded_method_and_profile(self):
        captured = []
        with patch("mlhdr_ptycho.comparison_figures._save", side_effect=lambda fig, *_: captured.append(fig)):
            plot_method_heatmaps(self.results, self.source)
        image = captured[0].axes[1].images[0]
        self.assertAlmostEqual(float(image.get_array()[4, 2]), -.009363559898075852)
        self.assertAlmostEqual(float(image.get_array()[8, 0]), .8164364670251155)

    def test_noise_comparison_keeps_same_four_methods(self):
        captured = []
        with patch("mlhdr_ptycho.comparison_figures._save", side_effect=lambda fig, *_: captured.append(fig)):
            plot_noise_comparison(self.results, self.source)
        ax = captured[0].axes[0]
        cases = ("single_1ms", "single_500ms", "paper_ml_hdr", "saturation_mask_extension")
        self.assertEqual(len(ax.containers), len(cases))
        for container, case in zip(ax.containers, cases):
            expected = [self.results.value(p, case, "baseline_psnr_db") for p in self.results.profiles]
            np.testing.assert_array_equal(container.lines[0].get_ydata(), expected)


if __name__ == "__main__":
    unittest.main()
