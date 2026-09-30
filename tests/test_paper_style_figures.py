"""Paper-style figures: loader validation and plotted values vs source data.

Uses a tiny synthetic fixture that follows DATA_CONTRACT.md; it never needs outputs/.
"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from mlhdr_ptycho import paper_style_figures as psf


ROOT = Path(__file__).resolve().parents[1]
DX = 0.5
METHODS = psf.METHODS
BITS = (2, 8, 16)
SNRS = (6, 30, 54)
TIMES = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
BASE = {"single": 10.0, "lrfc": 20.0, "ml_eq14_15": 25.0, "ml_masked": 22.0}


def _metric(method, x, seed):
    psnr = BASE[method] + 0.8 * x + 0.3 * seed
    return psnr, min(0.99, 0.2 + 0.03 * x + 0.01 * seed + 0.01 * METHODS.index(method)), 1.0 / psnr


def _geometry():
    """Groups 8 and 9, elements 1-3, both orientations, line width 1 px."""
    geometry = []
    for group, x0 in ((8, 12.0), (9, 30.0)):
        for element in (1, 2, 3):
            y0 = 10.0 + 8 * (element - 1)
            for orientation, xs in (("horizontal", x0), ("vertical", x0 + 7)):
                edges = [[y0 + 2 * k, y0 + 2 * k + 1] for k in range(3)]
                geometry.append({
                    "group": group, "element": element, "orientation": orientation,
                    "line_width_um": DX, "line_width_px": 1.0, "bbox_px": [y0, y0 + 5, xs, xs + 5],
                    "profile_axis": "y" if orientation == "horizontal" else "x",
                    "bar_edges_px": edges if orientation == "horizontal" else [[xs + 2 * k, xs + 2 * k + 1] for k in range(3)],
                    "bar_centers_px": [e[0] + 0.5 for e in edges], "bar_extent_px": [xs, xs + 5],
                })
    return geometry


def _runs_npz(path, labels, rng, usaf, limits=None):
    n, roi = len(labels), (5, 55, 5, 55)
    truth = np.ones((60, 60), np.complex64)
    truth[20:30, 20:30] = 0.2
    k = 5
    arrays = {
        "truth": truth, "truth_roi": truth[5:55, 5:55], "roi": np.array(roi, np.int64), "dx_um": np.float64(DX),
        "positions": np.zeros((4, 2), np.int64), "probe_true": np.ones((16, 16), np.complex64),
        "run_method": np.array([lab.split("@")[0] for lab in labels]),
        "run_bits": np.array([int(lab.split("@")[1]) for lab in labels], np.int64),
        "run_label": np.array(labels),
        "aligned_roi": (np.abs(truth[5:55, 5:55])[None] * rng.uniform(0.8, 1.1, (n, 50, 50))).astype(np.complex64),
        "psnr_db": rng.uniform(10, 30, n), "ssim": rng.uniform(0.2, 0.9, n), "nrmse": rng.uniform(0.05, 0.3, n),
        "history_iteration": np.arange(k, dtype=np.int64) * 5,
        "history_wallclock_s": np.cumsum(np.full((n, k), 0.5), axis=1) - 0.5,
        "history_object_nrmse": np.array([np.r_[0.4, np.full(k - 1, 0.4 / (i + 2))] for i in range(n)]),
        "frc_frequency_nyquist": np.linspace(0, 1, 30),
        "frc": np.clip(1 - np.linspace(0, 1, 30)[None] * rng.uniform(1.0, 2.5, (n, 1)), -0.2, 1),
        "frc_threshold": np.full(30, 0.3), "frc_cutoff_nyquist": np.linspace(0.3, 0.8, n),
    }
    arrays["frc_resolution_um"] = DX / arrays["frc_cutoff_nyquist"]
    if usaf:
        geometry = _geometry()
        edges = np.array([e for g in geometry if g["group"] == 9 and g["orientation"] == "horizontal"
                          for e in g["bar_edges_px"]], float)
        start, length = edges[0, 0] - 2.0, 26
        arrays.update({
            "usaf_geometry_json": np.array(json.dumps(geometry)), "resolved_json": np.array("{}"),
            "usaf_limit_label": np.array([limits.get(lab) or "" for lab in labels]),
            "usaf_limit_width_um": np.array([DX if limits.get(lab) else np.nan for lab in labels]),
            "profile_position_um": np.arange(length) * DX, "profile_truth": np.ones(length),
            "profile": rng.uniform(0.1, 1.0, (n, length)),
            "profile_ideal_position_um": np.linspace(0, length * DX, 200), "profile_ideal": np.ones(200),
            "profile_bar_edges_um": (edges - start) * DX, "profile_line_x_px": np.float64(32.5),
            "profile_columns_px": np.array([32, 33], np.int64),
        })
    np.savez_compressed(path, **arrays)


def make_fixture(root, seeds=(0, 1, 2), limits8=None, summary_limits8=None):
    """Write a complete, contract-conforming source directory into ``root``."""
    root = Path(root)
    rng = np.random.default_rng(7)
    fields = ["experiment", "object", "method", "bits", "seed", "snr_db", "read_noise_e", "exposure_s",
              "psnr_db", "ssim", "nrmse", "diffraction_nrmse", "frc_cutoff_nyquist", "frc_resolution_um", "recon_seconds", "status", "error"]
    for experiment, xs in (("bit_sweep", BITS), ("noise_sweep", SNRS)):
        with (root / f"{experiment}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for seed in seeds:
                for x in xs:
                    for method in METHODS:
                        bits = x if experiment == "bit_sweep" else (16 if method == "single" else 8)
                        snr = 114.0 if experiment == "bit_sweep" else float(x)
                        psnr, ssim, nrmse = _metric(method, x, seed)
                        failed = experiment == "bit_sweep" and seed == seeds[-1] and x == 2 and method == "lrfc" and len(seeds) > 1
                        writer.writerow({
                            "experiment": experiment, "object": "cameraman", "method": method, "bits": bits,
                            "seed": seed, "snr_db": snr, "read_noise_e": 5.0, "exposure_s": 0.01 if method == "single" else "",
                            "psnr_db": "" if failed else psnr, "ssim": "" if failed else ssim,
                            "nrmse": "" if failed else nrmse, "diffraction_nrmse": "" if failed else 0.1 + 0.01 * seed,
                            "frc_cutoff_nyquist": 0.5, "frc_resolution_um": 1.0,
                            "recon_seconds": 1.0, "status": "failed" if failed else "ok", "error": "boom" if failed else ""})
    labels8 = ["single@8", "single@16", "lrfc@8", "ml_eq14_15@8", "ml_masked@8"]
    labels16 = ["single@16", "lrfc@16", "ml_eq14_15@16", "ml_masked@16"]
    limits8 = limits8 or {"single@8": None, "single@16": "8-3", "lrfc@8": "9-1", "ml_eq14_15@8": "8-2", "ml_masked@8": "9-1"}
    summary_limits8 = summary_limits8 or limits8
    limits16 = {"single@16": "8-3", "lrfc@16": "9-2", "ml_eq14_15@16": None, "ml_masked@16": "9-3"}
    _runs_npz(root / "cameraman_8bit.npz", ["single@8", "lrfc@8", "ml_eq14_15@8", "ml_masked@8"], rng, False)
    _runs_npz(root / "usaf_8bit.npz", labels8, rng, True, limits8)
    _runs_npz(root / "usaf_16bit.npz", labels16, rng, True, limits16)

    def block(labels, limits):
        return {"truth": {"limit": {"label": "9-3", "line_width_um": DX}, "finest_any": {"label": "9-3", "line_width_um": DX}},
                "runs": [{"method": lab.split("@")[0], "bits": int(lab.split("@")[1]), "label": lab, "status": "ok",
                          "psnr_db": 20.0, "ssim": 0.8, "nrmse": 0.1, "frc_cutoff_nyquist": 0.5, "frc_resolution_um": 1.0,
                          "frc_reached_nyquist": False,
                          "usaf_limit": {"label": limits.get(lab), "line_width_um": DX if limits.get(lab) else None},
                          "usaf_finest_any": {"label": limits.get(lab), "line_width_um": None}} for lab in labels]}
    (root / "resolution_summary.json").write_text(json.dumps({
        "dx_um": DX, "usaf_contrast_threshold": 0.2, "cameraman_8bit": {"runs": []},
        "usaf_8bit": block(labels8, summary_limits8), "usaf_16bit": block(labels16, limits16)}), encoding="utf-8")
    n = 16
    raw = np.zeros((7, n, n), np.uint8)
    for i, t in enumerate(TIMES):
        raw[i, 8, 8] = min(255, int(2 + 500 * t * 100))
    np.savez_compressed(
        root / "diffraction_example.npz", position_index=np.int64(3), position_px=np.array([1, 2], np.int64),
        bits=np.int64(8), max_count=np.int64(255), exposure_times_s=np.array(TIMES), raw_counts=raw,
        dark_mean_counts=np.zeros((7, n, n)), dark_var_counts=np.zeros((7, n, n)),
        saturation_fraction_frame=np.array([0, 0, 0, 0, 0.001, 0.002, 0.004]),
        saturation_fraction_stack=np.zeros(7), rate_clean=np.ones((n, n)),
        rate_single=np.full((n, n), 10, np.float32), rate_lrfc=np.full((n, n), 20, np.float32),
        rate_ml_eq14_15=np.linspace(-5, 30, n * n).reshape(n, n).astype(np.float32),
        rate_ml_masked=np.full((n, n), 40, np.float32), single_exposure_s=np.float64(0.01),
        count_rate_scale=np.float64(1.0), electrons_per_count=np.float64(1.0), q_inv_um=np.arange(n) - n / 2.0)
    meta = {
        "profile": "fixture", "created": "2026-01-01T00:00:00", "command": "synthetic", "seeds": list(seeds),
        "profile_settings": {"iterations": 20, "bits": list(BITS), "snr_db": list(SNRS)}, "experiments": ["A"],
        "failures": [], "runtime_s": {"total_wallclock": 1.0},
        "simulation_config": {"dx_um": DX, "exposure_times_s": list(TIMES), "full_well_e": 2.5e6, "read_noise_e": 5.0},
        "mpie_config": {"iterations": 20},
        "noise_sweep": {"definition": "snr_db = 20*log10(full_well_e / read_noise_e); OUR assumption",
                        "bits": {"single": 16, "lrfc": 8, "ml_eq14_15": 8, "ml_masked": 8},
                        "read_noise_e": {str(s): 2.5e6 / 10 ** (s / 20) for s in SNRS}},
        "objects": {"cameraman": {"auto_exposure_s": 0.01}},
        "usaf": {"resolved_criterion": "3 distinct minima; contrast >= 0.2"},
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "DATA_CONTRACT.md").write_text("synthetic fixture\n", encoding="utf-8")
    return root


class PaperStyleFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        make_fixture(self.source)
        self.out = Path(self.temp.name) / "figures"
        self.out.mkdir()
        self.data = psf.load_paper_style_data(self.source)
        self.style = plt.rc_context(psf.STYLE)
        self.style.__enter__()

    def tearDown(self):
        self.style.__exit__(None, None, None)
        plt.close("all")
        self.temp.cleanup()

    def csv_rows(self, name):
        with (self.source / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    # ------------------------------------------------------------------ loader
    def test_sweep_statistics_match_csv_and_skip_failed_runs(self):
        rows = self.csv_rows("bit_sweep.csv")
        for method in METHODS:
            s = self.data.sweep("bit_sweep", method, "psnr_db")
            self.assertEqual(list(s.x), list(BITS))
            for x, mean, sd, n in zip(s.x, s.mean, s.sd, s.n):
                vals = [float(r["psnr_db"]) for r in rows if r["method"] == method and int(r["bits"]) == x
                        and r["status"] == "ok"]
                self.assertEqual(n, len(vals))
                self.assertAlmostEqual(mean, np.mean(vals))
                self.assertAlmostEqual(sd, np.std(vals, ddof=1))
        self.assertEqual(self.data.sweep("bit_sweep", "lrfc", "psnr_db").n[0], 2)  # one failed seed excluded

    def test_missing_file_is_reported_by_name(self):
        (self.source / "usaf_16bit.npz").unlink()
        with self.assertRaisesRegex(psf.DataContractError, "usaf_16bit.npz"):
            psf.load_paper_style_data(self.source)

    def test_unknown_method_and_missing_column_raise(self):
        text = (self.source / "noise_sweep.csv").read_text(encoding="utf-8")
        (self.source / "noise_sweep.csv").write_text(text.replace(",ml_masked,", ",ml_other,", 1), encoding="utf-8")
        with self.assertRaisesRegex(psf.DataContractError, "unknown method"):
            psf.load_paper_style_data(self.source)
        lines = text.splitlines()
        header = lines[0].split(",")
        idx = header.index("ssim")
        cut = [",".join(v for j, v in enumerate(line.split(",")) if j != idx) for line in lines]
        (self.source / "noise_sweep.csv").write_text("\n".join(cut) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(psf.DataContractError, "missing column"):
            psf.load_paper_style_data(self.source)

    def test_npz_and_summary_limits_must_agree(self):
        other = Path(self.temp.name) / "other"
        other.mkdir()
        make_fixture(other, summary_limits8={"single@8": None, "single@16": "8-3", "lrfc@8": "9-2",
                                             "ml_eq14_15@8": "8-2", "ml_masked@8": "9-1"})
        with self.assertRaisesRegex(psf.DataContractError, "lrfc@8"):
            psf.load_paper_style_data(other)

    # ----------------------------------------------------------------- figures
    def test_noise_figure_plots_csv_means_with_error_bars(self):
        fig = psf.plot_fig3_noise(self.data, self.out, dpi=50)
        axes = [ax for ax in fig.axes if ax.get_xlabel() == "Noise magnitude (dB)"]
        self.assertEqual(len(axes), 3)
        rows = self.csv_rows("noise_sweep.csv")
        for ax, (metric, _) in zip(axes, psf.METRICS):
            series = {line.get_gid(): line for line in ax.get_lines() if (line.get_gid() or "").startswith("ours:")}
            self.assertEqual(len(series), 4)
            self.assertEqual(len(ax.containers), 4)  # 3 seeds -> error bars for every method
            for method in METHODS:
                line = series[f"ours:noise_sweep:{metric}:{method}"]
                expected = [np.mean([float(r[metric]) for r in rows if r["method"] == method
                                     and float(r["snr_db"]) == x]) for x in SNRS]
                np.testing.assert_allclose(line.get_xdata(), SNRS)
                np.testing.assert_allclose(line.get_ydata(), expected)

    def test_single_seed_has_no_error_bars(self):
        single = Path(self.temp.name) / "single"
        single.mkdir()
        data = psf.load_paper_style_data(make_fixture(single, seeds=(0,)))
        fig = psf.plot_fig3_noise(data, self.out, dpi=50)
        self.assertEqual(sum(len(ax.containers) for ax in fig.axes), 0)

    def test_bit_depth_reference_line_and_letter_arrows(self):
        fig = psf.plot_fig2_bit_depth(self.data, self.out, dpi=50)
        for metric, _ in psf.METRICS:
            ax = next(a for a in fig.axes if any(l.get_gid() == f"ours:bit_sweep:{metric}:single" for l in a.get_lines()))
            ref = next(l for l in ax.get_lines() if l.get_gid() == f"reference:bit_sweep:{metric}:single@16")
            self.assertAlmostEqual(ref.get_ydata()[0], self.data.value("bit_sweep", "single", metric, 16))
            arrows = {t.get_gid(): t for t in ax.texts if (t.get_gid() or "").startswith("arrow:")}
            self.assertEqual(len(arrows), 4)
            for method in METHODS:
                ann = arrows[f"arrow:{metric}:{method}"]
                self.assertEqual(ann.get_text(), psf.LETTER[method])
                self.assertEqual(ann.xy, (8.0, self.data.value("bit_sweep", method, metric, 8)))
        mosaic = [im.get_gid() for ax in fig.axes for im in ax.images]
        self.assertEqual(sorted(g for g in mosaic if g), sorted(["mosaic:truth"] + [f"mosaic:{m}" for m in METHODS]))

    def test_resolved_element_box_comes_from_summary(self):
        fig = psf.plot_fig6_usaf_8bit(self.data, self.out, dpi=50)
        boxes = [p for ax in fig.axes for p in ax.patches if isinstance(p, Rectangle)
                 and (p.get_gid() or "").startswith("limit-box:usaf_8bit:lrfc@8:")]
        self.assertEqual(len(boxes), 2)  # full field of view + magnified
        label = self.data.usaf_limit("usaf_8bit", "lrfc@8")["label"]
        self.assertEqual(label, "9-1")
        y0, y1, x0, x1 = psf.element_bbox(self.data.usaf8.geometry, 9, 1)
        for box in boxes:
            self.assertTrue(box.get_gid().endswith(":9-1"))
            self.assertAlmostEqual(box.get_x(), x0 - 0.8)
            self.assertAlmostEqual(box.get_y(), y0 - 0.8)
            self.assertAlmostEqual(box.get_width(), x1 - x0 + 1.6)
            self.assertAlmostEqual(box.get_height(), y1 - y0 + 1.6)
        none = [t for ax in fig.axes for t in ax.texts if t.get_gid() == "limit-none:usaf_8bit:single@8"]
        self.assertEqual(len(none), 2)

    def test_usaf_16bit_profile_and_trace(self):
        fig = psf.plot_fig7_usaf_16bit(self.data, self.out, dpi=50)
        lines = {l.get_gid(): l for ax in fig.axes for l in ax.get_lines() if l.get_gid()}
        runs = self.data.usaf16
        for label in runs.labels:
            np.testing.assert_allclose(lines[f"profile:{label}"].get_ydata(), runs.run("profile", label))
        x, y0, y1 = psf.profile_line(runs)
        self.assertAlmostEqual(y0, 8.0)
        np.testing.assert_allclose(lines["trace:lrfc@16"].get_xdata(), [x, x])
        np.testing.assert_allclose(lines["frc:lrfc@16"].get_ydata(), runs.run("frc", "lrfc@16"))

    def test_profile_geometry_inconsistency_raises(self):
        runs = self.data.usaf16
        runs.arrays["profile_bar_edges_um"] = runs.arrays["profile_bar_edges_um"] + np.linspace(0, 1, 9)[:, None]
        with self.assertRaisesRegex(psf.DataContractError, "inconsistent"):
            psf.profile_line(runs)

    def test_fused_diffraction_uses_longest_exposure_counts(self):
        fig = psf.plot_fig5_diffraction(self.data, self.out, dpi=50)
        images = {im.get_gid(): im for ax in fig.axes for im in ax.images}
        self.assertEqual(len(images), 10)
        rate = np.asarray(self.data.diffraction["rate_ml_eq14_15"], float)
        np.testing.assert_allclose(images["diffraction:ml_eq14_15"].get_array(), np.log2(1 + np.clip(rate, 0, None) * 0.5))
        raw = np.asarray(self.data.diffraction["raw_counts"][6], float)
        np.testing.assert_allclose(images["diffraction:raw_500ms"].get_array(), np.log2(1 + raw))

    def test_convergence_index(self):
        noisy = np.log10(np.r_[0.5, 0.2, 0.12, 0.1, 0.101, 0.099, 0.1, 0.102, 0.098])
        self.assertEqual(psf._convergence_index(noisy), 3)
        self.assertIsNone(psf._convergence_index(np.log10([0.4, 0.45, 0.39, 0.41, 0.4])))

    # ------------------------------------------------------------ end to end
    def test_generate_figures_writes_every_figure_and_manifest(self):
        manifest = psf.generate_figures(self.source, self.out, dpi=50)
        self.assertEqual(set(manifest["figures"]), set(psf.FIGURES))
        for name in psf.FIGURES:
            for ext in ("png", "svg"):
                path = self.out / f"{name}.{ext}"
                self.assertEqual(manifest["figures"][name][ext]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        on_disk = json.loads((self.out / "figures_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["source_meta"]["seeds"], [0, 1, 2])
        self.assertEqual(set(on_disk["source_files"]), set(psf.REQUIRED_FILES) | {"DATA_CONTRACT.md"})
        csv_hash = hashlib.sha256((self.source / "bit_sweep.csv").read_bytes()).hexdigest()
        self.assertEqual(on_disk["source_files"]["bit_sweep.csv"]["sha256"], csv_hash)
        self.assertIn("paper_digitized", on_disk)

    def test_missing_optional_diffraction_metric_still_renders_every_figure(self):
        # The loader accepts the required reconstruction metrics without this
        # optional diagnostic; the figure footer must not invent its value.
        for filename in ("bit_sweep.csv", "noise_sweep.csv"):
            path = self.source / filename
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = [f for f in reader.fieldnames if f != "diffraction_nrmse"]
                rows = [{k: v for k, v in row.items() if k in fields} for row in reader]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(rows)
        data = psf.load_paper_style_data(self.source)
        self.assertNotIn("ml_diff_nrmse_8", psf.findings(data))
        self.assertNotIn("diffraction NRMSE", psf._fig2_findings(data))
        manifest = psf.generate_figures(self.source, self.out, dpi=50)
        self.assertEqual(set(manifest["figures"]), set(psf.FIGURES))
        for name in psf.FIGURES:
            svg = (self.out / f"{name}.svg").read_text(encoding="utf-8")
            self.assertNotRegex(svg.lower(), r"\bnan\b")

    def test_optional_diffraction_metric_is_preserved_in_findings(self):
        self.assertAlmostEqual(psf.findings(self.data)["ml_diff_nrmse_8"], 0.11)
        self.assertIn("8-bit diffraction NRMSE 0.11", psf._fig2_findings(self.data))

    def test_paper_digitized_json_is_consistent(self):
        paper = psf.load_paper_digitized(ROOT / "docs/paper_style/paper_digitized.json")
        self.assertEqual(len(paper["series"]), 18)
        for s in paper["series"]:
            self.assertIn(s["figure"], ("Fig. 2", "Fig. 3"))
            self.assertEqual(s["x"], list(range(2, 21, 2)) if s["figure"] == "Fig. 2" else list(range(6, 55, 6)))
            self.assertTrue(all(isinstance(a, bool) for a in s["approximate"]))
        ml = psf.paper_series(paper, "Fig. 2", "ml_eq14_15", "psnr_db")
        self.assertEqual(ml["values"][ml["x"].index(8)], 40.7)

    def test_summary_rows_pair_paper_and_reproduction(self):
        rows = psf.summary_rows(self.data)
        row = next(r for r in rows if r["section"] == "bit_sweep_8bit" and r["quantity"] == "psnr_db"
                   and r["method"] == "ml_eq14_15")
        self.assertEqual(row["paper"], "40.7")
        self.assertAlmostEqual(float(row["reproduction"]), self.data.value("bit_sweep", "ml_eq14_15", "psnr_db", 8), places=2)
        masked = next(r for r in rows if r["section"] == "usaf_8bit" and r["quantity"] == "smallest_resolved_element"
                      and r["method"] == "ml_masked")
        self.assertTrue(masked["reproduction"].startswith("9-1"))


if __name__ == "__main__":
    unittest.main()
