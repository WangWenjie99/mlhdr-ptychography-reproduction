"""Paper-style figures: loader validation and plotted values vs source data.

Uses a tiny synthetic fixture that follows DATA_CONTRACT.md; it never needs outputs/.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from mlhdr_ptycho import paper_style_figures as psf


DX = 0.5
METHODS = psf.METHODS
BITS = (2, 8, 16)
SNRS = (6, 30, 54)
TIMES = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
BASE = {"single": 10.0, "lrfc": 20.0, "ml_eq14_15": 25.0, "ml_masked": 22.0}


def _metric(method, x, seed):
    psnr = BASE[method] + 0.8 * x + 0.3 * seed
    return psnr, min(0.99, 0.2 + 0.03 * x + 0.01 * seed + 0.01 * METHODS.index(method)), 1.0 / psnr


def _runs_npz(path, labels, rng):
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
    np.savez_compressed(path, **arrays)


def make_fixture(root, seeds=(0, 1, 2)):
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
    _runs_npz(root / "cameraman_8bit.npz", ["single@8", "lrfc@8", "ml_eq14_15@8", "ml_masked@8"], rng)
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
        (self.source / "cameraman_8bit.npz").unlink()
        with self.assertRaisesRegex(psf.DataContractError, "cameraman_8bit.npz"):
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

    # ------------------------------------------------------------ end to end
    def test_generate_figures_writes_every_figure_and_manifest(self):
        self.assertEqual(psf.FIGURES, ("fig2_bit_depth", "fig3_noise"))
        self.assertEqual(psf.REQUIRED_FILES, ("meta.json", "bit_sweep.csv", "noise_sweep.csv", "cameraman_8bit.npz"))
        manifest = psf.generate_figures(self.source, self.out, dpi=50)
        self.assertEqual(set(manifest["figures"]), set(psf.FIGURES))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         sorted(["figures_manifest.json"] + [f"{n}.{e}" for n in psf.FIGURES for e in ("png", "svg")]))
        for name in psf.FIGURES:
            for ext in ("png", "svg"):
                path = self.out / f"{name}.{ext}"
                self.assertEqual(manifest["figures"][name][ext]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        on_disk = json.loads((self.out / "figures_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["source_meta"]["seeds"], [0, 1, 2])
        self.assertEqual(set(on_disk["source_files"]), set(psf.REQUIRED_FILES) | {"DATA_CONTRACT.md"})
        csv_hash = hashlib.sha256((self.source / "bit_sweep.csv").read_bytes()).hexdigest()
        self.assertEqual(on_disk["source_files"]["bit_sweep.csv"]["sha256"], csv_hash)
        self.assertFalse([key for key in on_disk if "paper" in key.lower()])
        self.assertNotRegex(json.dumps(on_disk).lower(), "digiti")

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

    def test_noise_lsb_uses_adc_step_of_masked_noise_runs(self):
        meta_path = self.source / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertNotIn("sigma_lsb_top", psf.findings(self.data))  # needs the dark-frame count
        meta["simulation_config"]["dark_frames"] = 20
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        f = psf.findings(psf.load_paper_style_data(self.source))
        # ml_masked runs at 8 bit in the noise sweep: one count = full well / 255 electrons.
        self.assertAlmostEqual(f["sigma_lsb_top"], meta["noise_sweep"]["read_noise_e"]["54"] / (2.5e6 / 255))
        self.assertAlmostEqual(f["p_zero_dark_var"],
                               (0.5 * (1 + math.erf(0.5 / f["sigma_lsb_top"] / math.sqrt(2)))) ** 20)

    def test_summary_table_holds_only_reproduction_values(self):
        path = self.out / "summary.csv"
        rows = psf.write_summary_csv(self.data, path)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(tuple(reader.fieldnames), psf.SUMMARY_FIELDS)
            written = list(reader)
        self.assertFalse([field for field in psf.SUMMARY_FIELDS if "paper" in field])
        self.assertEqual(len(written), len(rows))
        for row in written:
            self.assertNotRegex(" ".join(row.values()).lower(), "paper|digiti")
        row = next(r for r in written if r["section"] == "bit_sweep_8bit" and r["quantity"] == "psnr_db"
                   and r["method"] == "ml_eq14_15")
        self.assertAlmostEqual(float(row["value"]), self.data.value("bit_sweep", "ml_eq14_15", "psnr_db", 8), places=2)
        gap = next(r for r in written if r["section"] == "hdr8_vs_single16" and r["method"] == "lrfc"
                   and r["quantity"] == "psnr_db_8bit_minus_single_16bit")
        self.assertAlmostEqual(float(gap["value"]), self.data.value("bit_sweep", "lrfc", "psnr_db", 8)
                               - self.data.value("bit_sweep", "single", "psnr_db", 16), places=2)
        self.assertEqual({r["section"] for r in written},
                         {"bit_sweep_8bit", "bit_sweep_16bit_reference", "hdr8_vs_single16"})


if __name__ == "__main__":
    unittest.main()
