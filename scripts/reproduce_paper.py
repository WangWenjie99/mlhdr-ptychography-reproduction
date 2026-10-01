#!/usr/bin/env python3
"""Run the fixed-baseline, multi-exposure paper reproduction and write a report."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlhdr_mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlhdr_ptycho.data import load_diff_npy, normalize_stack
from mlhdr_ptycho.paper_reproduction import (
    PaperCamera, amplitude_comparison, diffraction_comparison, paper_eq14_15,
    saturation_mask_extension, simulate_paper_camera, single_rate,
)
from mlhdr_ptycho.ptylab_reconstruction import (
    PtyLabConfig, coverage_roi, reconstruction_metrics, run_mpie,
    save_reconstruction_outputs,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_data(archive, crop, frame_shape, input_stack):
    # This is the user's trusted local baseline, which stores config as a pickle.
    with np.load(archive, allow_pickle=True) as data:
        config = PtyLabConfig(**data["config"].ravel()[0])
        obj, probe = data["object"].copy(), data["probe"].copy()
        encoder, error = data["encoder"].copy(), data["error"].copy()
    if len(encoder) != crop ** 2 or probe.shape != frame_shape:
        raise ValueError("Baseline scan or detector size does not match input crop")
    n = obj.shape[0]
    positions = np.rint(encoder / config.object_pixel_m).astype(int) + n // 2 - probe.shape[0] // 2
    coverage = np.zeros(obj.shape, np.uint16)
    for y, x in positions:
        coverage[y:y + probe.shape[0], x:x + probe.shape[1]] += 1
    roi = coverage_roi(coverage)
    if roi is None or not np.all(coverage[roi] > 0):
        raise ValueError("Expected a contiguous covered ROI for this baseline")
    return config, obj, probe, error, coverage, roi


def contact_sheet(path, panels, reference, title, difference=False):
    cols = 3 if len(panels) > 4 else len(panels)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False,
                             layout="constrained")
    lo, hi = np.percentile(np.abs(reference), [1, 99])
    # All panels share the baseline's fixed limits. Difference panels share 20%
    # of its amplitude range; values outside limits remain in numerical metrics.
    if difference:
        lo, hi = 0, 0.2 * float(np.ptp(np.abs(reference)))
    for ax, (name, field) in zip(axes.ravel(), panels):
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        if field is None:
            ax.text(.5, .5, "FAILED / NO SIGNAL", ha="center", transform=ax.transAxes)
            continue
        shown = np.abs(field - np.abs(reference)) if difference else field
        im = ax.imshow(shown, cmap="magma" if difference else "gray", vmin=lo, vmax=hi,
                       interpolation="nearest")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    if any(field is not None for _, field in panels):
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65,
                     label="Absolute amplitude difference" if difference else "Amplitude (one fitted scale)")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["profile"], row["case"]), []).append(row)
    result = []
    keys = ("baseline_psnr_db", "baseline_ssim", "baseline_nrmse", "diffraction_nrmse",
            "high_q_diffraction_nrmse", "final_error_per_frame", "saturation_fraction")
    for (profile, case), items in groups.items():
        row = {"profile": profile, "case": case, "runs": len(items),
               "successful_runs": sum(x["status"] == "ok" for x in items)}
        for key in keys:
            vals = [float(x[key]) for x in items if x.get(key) is not None
                    and np.isfinite(x[key]) and x["status"] == "ok"]
            row[key + "_mean"] = float(np.mean(vals)) if vals else None
            row[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
        result.append(row)
    return result


def metric_plot(path, summary, profile, exposure_ms):
    data = {r["case"]: r for r in summary if r["profile"] == profile}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, metric, label in zip(axes,
            ["baseline_psnr_db", "baseline_ssim", "diffraction_nrmse"],
            ["PSNR vs saved baseline (dB)", "SSIM vs saved baseline", "Diffraction NRMSE vs clean input"]):
        means = [data[f"single_{t:g}ms"].get(metric + "_mean") for t in exposure_ms]
        stds = [data[f"single_{t:g}ms"].get(metric + "_std") for t in exposure_ms]
        ax.errorbar(exposure_ms, means, yerr=stds, marker="o", color="#4464ad", label="Single exposure")
        for case, name, color, style in [
            ("paper_ml_hdr", "Published Eq.14-15", "#cc4444", "-"),
            ("saturation_mask_extension", "Saturation mask extension", "#16846c", "--"),
        ]:
            mean = data[case].get(metric + "_mean")
            std = data[case].get(metric + "_std")
            if mean is not None:
                ax.axhline(mean, color=color, linestyle=style, label=name)
                ax.axhspan(mean - std, mean + std, color=color, alpha=.12)
        ax.set_xscale("log")
        ax.set_xlabel("Single exposure time (ms)")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{profile}: mean +/- sample standard deviation; all reconstructions 80 iterations")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diffraction_plot(path, clean, rates, measurement):
    # Fixed center scan position, selected before seeing any reconstruction.
    index = len(clean) // 2
    panels = [("Clean input", clean[index])]
    panels += [(label, a[index] / measurement.count_rate_scale) for label, a in rates]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout="constrained")
    for ax, (label, a) in zip(axes.ravel(), panels):
        im = ax.imshow(np.log10(np.maximum(a, 1e-8)), vmin=-8, vmax=0, cmap="viridis")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="log10 relative intensity (common scale)")
    fig.suptitle("Center scan diffraction; known camera gain undone; no per-panel normalization")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(output, manifest, summary, baseline_check):
    conf = manifest["experiment"]
    main_profile = manifest["selected_profiles"][0]
    items = [r for r in summary if r["profile"] == main_profile]
    by_case = {r["case"]: r for r in items}
    singles = [r for r in items if r["case"].startswith("single_") and r["baseline_ssim_mean"] is not None]
    best = max(singles, key=lambda r: r["baseline_ssim_mean"]) if singles else None
    strict = by_case["paper_ml_hdr"]
    extension = by_case["saturation_mask_extension"]
    def number(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}"
    lines = [
        "# ML-HDR paper reproduction and fixed-baseline comparison", "",
        "All reconstructions use the user-selected baseline settings: a 21×21 scan, 32×32 diffraction frames, "
        "an 8 px scan step, a 31 px initial probe diameter, circ_smooth / ones initialization, "
        "80 CPU mPIE iterations, and reconstruction seed 0.",
        "", "## Conclusions and main results", "",
        "The following table averages multiple camera-noise seeds for the main profile `" + main_profile + "`. "
        "PSNR/SSIM measure agreement with the saved baseline reconstruction; "
        "they do not measure accuracy or resolution against the true object.",
        "", "| Method | PSNR / dB | SSIM | Amplitude NRMSE | Diffraction intensity NRMSE | Successful runs |",
        "|---|---:|---:|---:|---:|---:|",
        "| Saved raw-data baseline (self-reference) | ∞ | 1 | 0 | 0 | — |",
    ]
    for r in items:
        lines.append(f"| {r['case']} | {number(r['baseline_psnr_db_mean'],2)} | {number(r['baseline_ssim_mean'])} "
                     f"| {number(r['baseline_nrmse_mean'])} | {number(r['diffraction_nrmse_mean'])} "
                     f"| {r['successful_runs']}/{r['runs']} |")
    if best:
        lines += ["", f"The best single exposure by mean SSIM is **{best['case']}**. "
                  f"The strict paper formula has an SSIM of **{number(strict['baseline_ssim_mean'])}**, "
                  f"compared with **{number(best['baseline_ssim_mean'])}** for that single exposure.",
                  "The strict formula " + ("outperforms the best single exposure on this metric."
                               if strict['baseline_ssim_mean'] > best['baseline_ssim_mean']
                               else "does not outperform the best single exposure on this metric."),
                  f"The separate saturation-mask extension has an SSIM of **{number(extension['baseline_ssim_mean'])}**. "
                  "Its gains cannot be attributed to the paper's original formula."]
    lines += ["", f"![Strict paper formula and all single exposures]({main_profile}/comparison_amplitude.png)",
              "", f"![Additional saturation-handling control]({main_profile}/extension_comparison.png)",
              "", f"![Metrics and variation across noise seeds]({main_profile}/metrics_vs_exposure.png)",
              "", "## Paper specifications, additional assumptions, and differences", ""]
    for key, value in conf["provenance"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "Paper: [Liu et al., IEEE TIM 2024](https://doi.org/10.1109/TIM.2024.3363788). "
              "Equations (14)–(15) on page 3, the simulation settings on page 5, "
              "and the exposure times on page 7 were checked against the paper.",
              "", "The main profile inherits the previous project's read noise of 5 e− and dark current of 80 e−/s. "
              "The other two profiles set read noise to the electron equivalents of 0.25 and 1 ADC count "
              "to examine behavior when dark-field variance is recorded after quantization. "
              "These values are disclosed sensitivity-analysis assumptions.",
              "", "Photon-flux mapping: one global factor sets the total detected photon rate of the brightest "
              "frame to 10⁹/s while preserving relative energies across scan positions. Quantum efficiency is assumed "
              "to be 1. The paper does not provide enough information to determine this mapping uniquely, "
              "so its Figures 2/3 cannot be reproduced point by point.",
              "", "## Implementation and saturation diagnostics", "",
              "Each exposure independently simulates Poisson(signal rate × exposure time), "
              "Poisson(dark current × exposure time), and N(0, read-noise standard deviation²), "
              "followed by full-well clipping and ADC rounding. Twenty additional dark frames per exposure "
              "provide the mean and unbiased sample variance. All single-exposure and fusion methods share "
              "the same measurements.",
              "", "Strict implementation: r̄ = mean((Zᵢ−B̄ᵢ)/tᵢ), wᵢ = tᵢ²/(tᵢr̄+Var(Bᵢ)), "
              "and fusion = sum(wᵢ(Zᵢ−B̄ᵢ)/tᵢ)/sum(wᵢ), with nonnegative estimates and numerical "
              "division-by-zero protection. Saturated values are retained; no reference image is added, "
              "and the reconstruction engine and iteration count are unchanged.",
              "", "When dark-field variance approaches zero and the denominator is valid, the original formula "
              "reduces to sum(Zᵢ−B̄ᵢ)/sum(tᵢ). Saturated readings still enter the average as ordinary observations, "
              "underestimating bright regions. Independent formula tests and diagnostics of the saved diffraction "
              "inputs verify this behavior.",
              "", "The additional extension uses only unsaturated observations to estimate the initial rate "
              "and calculate weights, while retaining zero-valued observations. It reports failure if all exposures "
              "of any pixel are saturated. This control examines saturation bias and is separate from the paper's "
              "original algorithm.",
              "", "## Comparison methods and fairness", "",
              "- Use a common scan-coverage ROI. Full-size outputs still contain an unupdated border, "
              "which is excluded from evaluation.",
              "- All figures share the baseline's grayscale limits. Each image fits only one positive amplitude "
              "scale to compensate for object/probe scale ambiguity, without shifting, filtering, histogram matching, "
              "or independent contrast stretching.",
              "- PSNR uses the fixed baseline's amplitude range; SSIM uses a 7×7 window. The baseline is a "
              "reconstruction rather than the true object and is not treated as ground truth for resolution gains.",
              "- Diffraction NRMSE compares against the known input before camera simulation and undoes the known "
              "gain to restore a common scale, without fitting a separate intensity scale for each method.",
              "- Per-frame fitting error is only a convergence diagnostic for each method; it does not establish "
              "image-quality advantages across different inputs.",
              "- All methods use the same initialization settings, without initializing from the baseline object "
              "or probe. Probe power is estimated from each method's own measurements using the baseline procedure.",
              "- Noise seeds are preset to " + str(manifest["selected_seeds"]) + ". Figures always show the first "
              "seed, and tables summarize all seeds without selecting the best run.",
              "- Total multi-exposure integration time is 666.5 ms per position, excluding dark frames and readout "
              "overhead; each single exposure uses its corresponding time. Acquisition time and photon budgets "
              "are unequal, so any gains cannot be interpreted as improved acquisition efficiency.",
              "- All exposure results and failure statuses are retained. LRFC-HDR, a 16-bit control, and FRC "
              "against the true object are not included.",
              "", "## Baseline recalculation check", "",
              f"Recalculation in the current environment using the saved settings gives amplitude "
              f"NRMSE={number(baseline_check['baseline_nrmse'],6)} and "
              f"SSIM={number(baseline_check['baseline_ssim'],6)} against the historical baseline. "
              "The historical baseline remains the common reference, and the discrepancy is retained and reported.",
              "", "## Outputs and rerunning", "",
              "- `manifest.json`: complete settings, sources and assumptions, software versions, and code/data SHA-256 hashes.",
              "- `comparison_metrics.csv`: complete metrics for each profile, seed, and method.",
              "- `summary_metrics.csv`: means and sample standard deviations across seeds.",
              "- `camera_diagnostics.csv`: saturation and zero fractions, all-zero frames, dark-field means/variances, and related diagnostics.",
              "- `baseline_reference/`: copies of the original outputs for the user-selected baseline; "
              "`baseline_recomputed/`: recalculation in the current environment.",
              "- `<profile>/seed_<seed>/measurement.npz`: all digital measurements, dark frames, and exposure times.",
              "- `<profile>/seed_<seed>/<method>/`: reconstruction arrays, fusion/single-exposure inputs, and metadata. "
              "The first seed also includes complete PNG and PtyLab HDF5 outputs.",
              "", "Run from the project directory using a new output path; the script refuses to overwrite "
              "an existing directory:", "", "```bash",
              "python scripts/reproduce_paper.py --output outputs/paper_reproduction_repeat",
              "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    # A simple self-contained index uses relative local assets, no external JS.
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
        [r['case'], number(r['baseline_psnr_db_mean'],2), number(r['baseline_ssim_mean']),
         number(r['baseline_nrmse_mean'])]) + "</tr>" for r in items)
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>ML-HDR reproduction comparison</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#243044}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#176e8c}}.note{{background:#eef4f8;padding:18px}}</style>
<h1>ML-HDR paper reproduction</h1><p>Fixed baseline: 21×21 scan · 80 mPIE iterations · 8 px step · 31 px initial probe diameter</p>
<p class="note">The red curves and paper_ml_hdr implement the paper's Equations (14)–(15). The green dashed curves and saturation_mask_extension use additional saturation handling, whose results are separate from the original algorithm. PSNR/SSIM measure agreement with the saved baseline rather than true-object resolution.</p>
<p><a href="REPORT.md">Full experiment report and limitations</a> · <a href="manifest.json">Parameters and sources</a> · <a href="summary_metrics.csv">Metrics table</a></p>
<table><tr><th>Main-profile method</th><th>PSNR / dB</th><th>SSIM</th><th>Amplitude NRMSE</th></tr>{table_rows}</table>"""
    for profile in manifest["selected_profiles"]:
        body += f'<h2>{html.escape(profile)}</h2>'
        for name in ["comparison_amplitude.png", "extension_comparison.png", "metrics_vs_exposure.png", "diffraction_comparison.png"]:
            body += f'<p><a href="{profile}/{name}"><img src="{profile}/{name}" loading="lazy"></a></p>'
    (output / "index.html").write_text(body + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/paper_baseline_comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_reproduction_21x21_i80")
    parser.add_argument("--profiles", help="Comma-separated profile subset (default: all)")
    parser.add_argument("--seeds", help="Comma-separated camera seed subset (default: configured seeds)")
    args = parser.parse_args()
    conf = json.loads(args.config.read_text())
    profiles = args.profiles.split(",") if args.profiles else list(conf["noise_profiles"])
    seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else conf["measurement_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("Seeds must be a nonempty list of unique nonnegative integers")
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in conf["noise_profiles"] for p in profiles):
        parser.error("Unknown or duplicated noise profile")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output}")
    logging.getLogger().setLevel(logging.ERROR)
    archive, input_path = ROOT / conf["baseline_archive"], ROOT / conf["input"]
    diff = load_diff_npy(input_path, conf["scan_crop"])
    clean = normalize_stack(diff.stack)
    config, ref_obj, ref_probe, ref_error, coverage, roi = reference_data(
        archive, conf["scan_crop"], diff.frame_shape, diff.stack)
    if config.iterations != 80:
        raise ValueError("Expected the user's selected 80-iteration baseline")
    reference = ref_obj[roi]
    output.mkdir(parents=True)
    reference_dir = output / "baseline_reference"
    reference_dir.mkdir()
    for source in archive.parent.glob("best_*"):
        if source.is_file():
            shutil.copy2(source, reference_dir / source.name)
    write_json(reference_dir / "config.json", asdict(config))
    np.savez_compressed(reference_dir / "reference_roi.npz", object=reference, coverage=coverage[roi])
    manifest = {
        "experiment": conf, "selected_profiles": profiles, "selected_seeds": seeds,
        "ptylab_config": asdict(config), "input_shape": [*diff.scan_shape, *diff.frame_shape],
        "roi": [[r.start, r.stop] for r in roi], "object_shape": list(ref_obj.shape),
        "baseline_sha256": sha256(archive), "input_sha256": sha256(input_path),
        "python": sys.version, "executable": sys.executable,
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "matplotlib", "PtyLab", "h5py", "scikit-image"]},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                          [*sorted((ROOT / "mlhdr_ptycho").glob("*.py")), Path(__file__), args.config.resolve()]},
        "comparison_reference": "Saved baseline reconstruction, not ground truth",
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    rec, data, _ = run_mpie(diff.stack, diff.scan_shape, config)
    baseline_check, _ = amplitude_comparison(np.squeeze(rec.object)[roi], reference)
    save_reconstruction_outputs(output / "baseline_recomputed", "raw", rec, data, config)
    write_json(output / "baseline_recomputed/comparison_to_saved.json", baseline_check)
    print("Baseline recheck:", baseline_check, flush=True)
    rows, diagnostics, previews = [], [], {}
    exposure_ms = conf["exposure_times_ms"]
    times = np.asarray(exposure_ms, float) / 1000
    for profile in profiles:
        camera = PaperCamera(**conf["camera"], **conf["noise_profiles"][profile])
        profile_dir = output / profile
        profile_dir.mkdir()
        for seed in seeds:
            seed_dir = profile_dir / f"seed_{seed}"
            seed_dir.mkdir()
            m = simulate_paper_camera(clean, times, camera, seed)
            np.savez_compressed(seed_dir / "measurement.npz", z=m.z, dark_frames=m.dark_frames,
                                dark_mean=m.dark_mean, dark_var=m.dark_var,
                                exposure_times_s=times, rate_scale=m.rate_scale)
            write_json(seed_dir / "camera.json", asdict(camera))
            for i, t in enumerate(exposure_ms):
                z = m.z[i]
                diagnostics.append({"profile": profile, "seed": seed, "exposure_ms": t,
                    "electrons_per_count": camera.electrons_per_count,
                    "saturation_fraction": float(np.mean(z == camera.max_count)),
                    "zero_fraction": float(np.mean(z == 0)),
                    "all_zero_frames": int(np.sum(z.sum(axis=(-2,-1)) == 0)),
                    "dark_mean_average": float(m.dark_mean[i].mean()),
                    "dark_variance_average": float(m.dark_var[i].mean()),
                    "dark_variance_nonzero_fraction": float(np.mean(m.dark_var[i] > 0)),
                    "max_expected_signal_e": float(clean.max() * m.rate_scale * times[i]),
                })
            rates = [(f"single_{t:g}ms", single_rate(m, i)) for i, t in enumerate(exposure_ms)]
            rates += [("paper_ml_hdr", paper_eq14_15(m)),
                      ("saturation_mask_extension", saturation_mask_extension(m))]
            if seed == seeds[0]:
                diffraction_plot(profile_dir / "diffraction_comparison.png", clean, rates, m)
                previews[profile] = {"baseline": np.abs(reference)}
            for case, rate in rates:
                start = time.monotonic()
                case_dir = seed_dir / case
                case_dir.mkdir()
                np.savez_compressed(case_dir / "input_rate.npz", count_rate=rate)
                row = {"profile": profile, "seed": seed, "case": case, "status": "ok",
                       **diffraction_comparison(rate, m, clean)}
                if case.startswith("single_"):
                    i = [f"single_{t:g}ms" for t in exposure_ms].index(case)
                    row["saturation_fraction"] = float(np.mean(m.z[i] == camera.max_count))
                aligned = None
                if not np.isfinite(rate).all() or not np.any(rate > 0):
                    row["status"] = "failed_no_finite_signal"
                else:
                    rec, data, _ = run_mpie(rate, diff.scan_shape, config)
                    obj, probe, error = np.squeeze(rec.object), np.squeeze(rec.probe), np.asarray(rec.error)
                    np.savez_compressed(case_dir / "arrays.npz", object=obj, probe=probe,
                                        error=error, encoder=data.encoder, coverage=coverage)
                    if not all(np.isfinite(a).all() for a in [obj, probe, error]):
                        row["status"] = "failed_nonfinite"
                    else:
                        metrics, aligned = amplitude_comparison(obj[roi], reference)
                        row.update(metrics)
                        row.update(reconstruction_metrics(rec, data))
                        if seed == seeds[0]:
                            save_reconstruction_outputs(case_dir, case, rec, data, config)
                row["seconds"] = round(time.monotonic() - start, 3)
                rows.append(row)
                write_json(case_dir / "metadata.json", {"metrics": row, "camera": asdict(camera),
                    "measurement_seed": seed, "ptylab_config": asdict(config),
                    "method": "published_eq14_15" if case == "paper_ml_hdr" else case})
                if seed == seeds[0]:
                    previews[profile][case] = aligned
                print(f"{profile} seed={seed} {case}: {row['status']} "
                      f"PSNR={row.get('baseline_psnr_db')} SSIM={row.get('baseline_ssim')}", flush=True)
                write_csv(output / "comparison_metrics.csv", rows)
            write_csv(output / "camera_diagnostics.csv", diagnostics)
    summary = summarize(rows)
    write_csv(output / "summary_metrics.csv", summary)
    for profile in profiles:
        prev = previews[profile]
        panels = [("Saved raw baseline", prev["baseline"])]
        panels += [(f"Single {t:g} ms", prev[f"single_{t:g}ms"]) for t in exposure_ms]
        panels += [("Published Eq.14-15", prev["paper_ml_hdr"])]
        contact_sheet(output / profile / "comparison_amplitude.png", panels, reference,
                      "Saved baseline / single exposures / published ML-HDR; same grayscale")
        contact_sheet(output / profile / "difference_to_baseline.png", panels, reference,
                      "Absolute amplitude difference from saved baseline", difference=True)
        best = max((r for r in summary if r["profile"] == profile and r["case"].startswith("single_")
                    and r["baseline_ssim_mean"] is not None), key=lambda r: r["baseline_ssim_mean"])
        control = [("Saved raw baseline", prev["baseline"]),
                   (f"Best single by mean SSIM\n{best['case']}", prev[best["case"]]),
                   ("Published Eq.14-15", prev["paper_ml_hdr"]),
                   ("Additional saturation mask\nNOT published algorithm", prev["saturation_mask_extension"])]
        contact_sheet(output / profile / "extension_comparison.png", control, reference,
                      "Saturation diagnostic control (first fixed noise seed)")
        metric_plot(output / profile / "metrics_vs_exposure.png", summary, profile, exposure_ms)
    manifest["status"] = "complete"
    manifest["successful_reconstructions"] = sum(r["status"] == "ok" for r in rows)
    manifest["total_reconstructions"] = len(rows)
    write_json(output / "manifest.json", manifest)
    report(output, manifest, summary, baseline_check)
    print(f"Complete: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
