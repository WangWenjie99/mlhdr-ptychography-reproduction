"""Paper-style figures from the recorded metrics, without running PtyLab.

This module never estimates numerical data from PNGs. All plotted means and
sample standard deviations are read verbatim from the source CSV.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


QUALITY = (
    ("baseline_psnr_db", "PSNR relative to baseline (dB) ↑", "viridis"),
    ("baseline_ssim", "SSIM relative to baseline ↑", "viridis"),
    ("baseline_nrmse", "Amplitude NRMSE relative to baseline ↓", "viridis_r"),
)
DIAGNOSTICS = (
    ("diffraction_nrmse", "Full diffraction NRMSE ↓"),
    ("high_q_diffraction_nrmse", "High-q diffraction NRMSE ↓"),
    ("saturation_fraction", "Saturated pixels in a single exposure (%)"),
)
FUSIONS = ("paper_ml_hdr", "saturation_mask_extension")
NAMES = {
    "paper_ml_hdr": "ML-HDR, Eq. (14)–(15)",
    "saturation_mask_extension": "Saturation-mask extension*",
}
FUSION_STYLES = {
    "paper_ml_hdr": {"color": "#C13C3C", "linestyle": "--", "marker": "s"},
    "saturation_mask_extension": {"color": "#12816D", "linestyle": "-.", "marker": "^"},
}
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "savefig.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "mlhdr-comparison-figures-v1",
}


@dataclass(frozen=True)
class RecordedResults:
    source: Path
    csv_path: Path
    manifest_path: Path
    manifest: dict
    profiles: tuple[str, ...]
    exposures: tuple[float, ...]
    cases: tuple[str, ...]
    rows: dict[tuple[str, str], dict]

    def value(self, profile, case, metric, statistic="mean"):
        return self.rows[profile, case][f"{metric}_{statistic}"]

    @property
    def total_exposure_ms(self):
        return sum(self.exposures)

    def noise_e(self, profile):
        conf = self.manifest["experiment"]
        camera = dict(conf["camera"], **conf["noise_profiles"][profile])
        return float(camera["read_noise_e"])

    def noise_label(self, profile):
        return f"Read-noise σ = {self.noise_e(profile):,.2f} e−".replace("5.00 e", "5 e")

    def sample_label(self):
        successes = sorted({r["successful_runs"] for r in self.rows.values()})
        counts = str(successes[0]) if len(successes) == 1 else "–".join(map(str, (min(successes), max(successes))))
        return f"Mean ± sample SD; n = {counts} successful runs per method/condition"


def case_label(case):
    return NAMES.get(case, "Single " + case.removeprefix("single_").replace("ms", " ms"))


def _finite_number(raw, location):
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid number: {location}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite number: {location}")
    return value


def load_recorded_results(source):
    """Validate a complete metrics table against its recorded experiment.

    Missing/duplicated methods, missing SDs, and unmatched profiles raise. Failed
    repeats remain visible through successful-run counts in the manifest.
    """
    source = Path(source).resolve()
    csv_path = source / "summary_metrics.csv"
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = source / "experiment_manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError("Source must contain summary_metrics.csv and manifest.json or experiment_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        experiment = manifest["experiment"]
        profiles = tuple(manifest["selected_profiles"])
        exposures = tuple(sorted(float(t) for t in experiment["exposure_times_ms"]))
        seeds = tuple(manifest["selected_seeds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest is missing experiment, selected profiles/seeds, or exposure times") from exc
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("Manifest profiles must be nonempty and unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Manifest seeds must be nonempty and unique")
    if not exposures or len(set(exposures)) != len(exposures) or any(not np.isfinite(t) or t <= 0 for t in exposures):
        raise ValueError("Exposure times must be finite, positive and unique")
    if not {1.0, 500.0}.issubset(exposures):
        raise ValueError("Fixed-exposure noise comparison requires the recorded 1 ms and 500 ms methods")
    shape = manifest.get("input_shape")
    if not isinstance(shape, list) or len(shape) != 4 or any(not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("Manifest input_shape must contain four positive integer dimensions")
    for profile in profiles:
        try:
            camera = dict(experiment["camera"], **experiment["noise_profiles"][profile])
            noise = _finite_number(camera["read_noise_e"], f"{profile}/read_noise_e")
        except KeyError as exc:
            raise ValueError(f"Missing camera settings for {profile}") from exc
        if noise < 0:
            raise ValueError(f"Negative read-noise standard deviation for {profile}")
    # Physical noise order, never lexical filename order.
    profiles = tuple(sorted(profiles, key=lambda p: float(dict(experiment["camera"], **experiment["noise_profiles"][p])["read_noise_e"])))
    cases = tuple(f"single_{t:g}ms" for t in exposures) + FUSIONS
    rows = {}
    required = [metric for metric, *_ in QUALITY] + [metric for metric, _ in DIAGNOSTICS[:2]]
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            profile, case = row.get("profile"), row.get("case")
            key = (profile, case)
            if key in rows:
                raise ValueError(f"Duplicate summary row: {profile}/{case}")
            if profile not in profiles or case not in cases:
                raise ValueError(f"Unexpected profile or method: {profile}/{case}")
            for count in ("runs", "successful_runs"):
                value = _finite_number(row.get(count), f"{profile}/{case}/{count}")
                if not value.is_integer():
                    raise ValueError(f"Run count must be an integer: {profile}/{case}/{count}")
                row[count] = int(value)
            if row["runs"] != len(seeds) or not 0 < row["successful_runs"] <= row["runs"]:
                raise ValueError(f"Run counts disagree with manifest, or no successful result: {profile}/{case}")
            for metric in required + (["saturation_fraction"] if case.startswith("single_") else []):
                for statistic in ("mean", "std"):
                    field = f"{metric}_{statistic}"
                    value = _finite_number(row.get(field), f"{profile}/{case}/{field}")
                    if statistic == "std" and value < 0:
                        raise ValueError(f"Negative standard deviation: {profile}/{case}/{field}")
                    if statistic == "mean" and ("nrmse" in metric or metric == "saturation_fraction") and value < 0:
                        raise ValueError(f"Negative error/fraction: {profile}/{case}/{field}")
                    if metric == "saturation_fraction" and statistic == "mean" and value > 1:
                        raise ValueError(f"Saturation fraction exceeds one: {profile}/{case}")
                    row[field] = value
            rows[key] = row
    missing = [(p, c) for p in profiles for c in cases if (p, c) not in rows]
    if missing:
        raise ValueError("Missing summary rows: " + ", ".join(f"{p}/{c}" for p, c in missing))
    return RecordedResults(source, csv_path, manifest_path, manifest, profiles, exposures, cases, rows)


def _colors(results):
    shades = plt.get_cmap("Blues")(np.linspace(.50, .92, len(results.exposures)))
    colors = {case: color for case, color in zip(results.cases, shades)}
    colors.update({case: style["color"] for case, style in FUSION_STYLES.items()})
    return colors


def _bounds(results, metric, cases=None, factor=1):
    intervals = [(results.value(p, c, metric) * factor, results.value(p, c, metric, "std") * factor)
                 for p in results.profiles for c in (cases or results.cases)]
    low = min(m - s for m, s in intervals)
    high = max(m + s for m, s in intervals)
    margin = max(high - low, abs(high) * .1, .01) * .08
    return low - margin, high + margin


def _save(fig, output, name, dpi):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=dpi, metadata={"Creator": "ML-HDR recorded-results comparison"})
    plt.close(fig)


def _footer(fig, results, *, diffraction=False):
    reference = "Diffraction reference: known pre-camera input." if diffraction else "Reference: saved raw-data reconstruction, not object ground truth."
    fig.text(.055, .037, results.sample_label() + ". " + reference, fontsize=9)
    fig.text(.055, .016,
             f"* Extension is not the published method. HDR total exposure: {results.total_exposure_ms:g} ms/position; acquisition budgets are unequal.",
             fontsize=9)


def _exposure_axis(ax, results):
    ax.set_xscale("log")
    ax.set_xticks(results.exposures, [f"{x:g}" for x in results.exposures])
    ax.minorticks_off()
    ax.set_xlabel("Single-exposure time (ms; log scale)")
    ax.grid(alpha=.2, linewidth=.6)


def _exposure_series(ax, results, profile, metric, *, include_fusion=True, factor=1):
    singles = results.cases[:-2]
    means = [results.value(profile, case, metric) * factor for case in singles]
    stds = [results.value(profile, case, metric, "std") * factor for case in singles]
    colors = _colors(results)
    ax.plot(results.exposures, means, color="#6C87AA", linewidth=1.35, zorder=2)
    for t, m, s, case in zip(results.exposures, means, stds, singles):
        ax.errorbar(t, m, yerr=s, color=colors[case], marker="o", markersize=4.7,
                    capsize=3, elinewidth=1, zorder=3)
    if include_fusion:
        for case in FUSIONS:
            style = FUSION_STYLES[case]
            mean = results.value(profile, case, metric) * factor
            std = results.value(profile, case, metric, "std") * factor
            ax.axhspan(mean - std, mean + std, color=style["color"], alpha=.12, linewidth=0)
            ax.axhline(mean, color=style["color"], linestyle=style["linestyle"], linewidth=1.8)
    _exposure_axis(ax, results)


def _legend(fig):
    handles = [Line2D([], [], color="#4F73B4", marker="o", markersize=5, label="Single exposure")]
    handles += [Line2D([], [], color=FUSION_STYLES[c]["color"], linestyle=FUSION_STYLES[c]["linestyle"],
                       linewidth=1.8, label=NAMES[c]) for c in FUSIONS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .955), ncol=3, frameon=False, fontsize=10)


def plot_exposure_comparison(results, output, dpi=300):
    n = len(results.profiles)
    fig, axes = plt.subplots(n, 3, figsize=(15.8, 3.2 * n + 1.25), squeeze=False)
    fig.subplots_adjust(left=.07, right=.98, bottom=.13, top=.865, hspace=.55, wspace=.30)
    for col, (metric, label, _) in enumerate(QUALITY):
        bounds = _bounds(results, metric)
        for row, profile in enumerate(results.profiles):
            ax = axes[row, col]
            _exposure_series(ax, results, profile, metric)
            ax.set_ylim(bounds)
            ax.set_ylabel(label)
            ax.set_title(f"({chr(97 + row * 3 + col)}) {results.noise_label(profile)}", loc="left")
    fig.suptitle("Reconstruction quality across exposure and read noise", y=.985, fontsize=16)
    _legend(fig)
    _footer(fig, results)
    _save(fig, output, "exposure_comparison", dpi)


def plot_method_heatmaps(results, output, dpi=300):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.6))
    fig.subplots_adjust(left=.195, right=.98, bottom=.23, top=.84, wspace=.34)
    for i, (ax, (metric, label, cmap)) in enumerate(zip(axes, QUALITY)):
        values = np.array([[results.value(p, c, metric) for p in results.profiles] for c in results.cases])
        im = ax.imshow(values, cmap=cmap, aspect="auto", interpolation="nearest", vmin=values.min(), vmax=values.max())
        ax.set_title(f"({chr(97 + i)}) {label}", loc="left", pad=14, fontsize=11)
        ax.set_xticks(range(len(results.profiles)), [f"{results.noise_e(p):,.2f}".replace("5.00", "5") for p in results.profiles])
        ax.set_xlabel("Read-noise σ (e−)")
        ax.set_yticks(range(len(results.cases)), [case_label(c) for c in results.cases] if i == 0 else [])
        ax.tick_params(length=0, pad=8)
        ax.axhline(len(results.exposures) - .5, color="white", linewidth=2.3)
        for y, case in enumerate(results.cases):
            for x, profile in enumerate(results.profiles):
                val, sd = values[y, x], results.value(profile, case, metric, "std")
                rgba = im.cmap(im.norm(val))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                # Four decimals retain the small, nonzero recorded SDs.
                sd_text = f"{sd:.1e}" if 0 < sd < .00005 else f"{sd:.4f}"
                label_text = f"{val:.2f}\n±{sd:.2f}" if metric.endswith("db") else f"{val:.4f}\n±{sd_text}"
                ax.text(x, y, label_text, ha="center", va="center", color="black" if luminance > .56 else "white", fontsize=9.2)
        bar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.035, pad=.18)
        bar.set_label("Brighter = better; color encodes the mean", fontsize=9)
    fig.suptitle("Every recorded method and noise condition", y=.97, fontsize=16)
    fig.text(.5, .915, "Cell = mean ± sample SD; shared color scale across noise settings within each metric", ha="center", fontsize=11)
    _footer(fig, results)
    _save(fig, output, "method_heatmaps", dpi)


def plot_noise_comparison(results, output, dpi=300):
    cases = ("single_1ms", "single_500ms", *FUSIONS)
    colors = _colors(results)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8))
    fig.subplots_adjust(left=.07, right=.98, bottom=.27, top=.74, wspace=.30)
    x = np.arange(len(results.profiles))
    for ax, (metric, label, _) in zip(axes, QUALITY):
        for i, case in enumerate(cases):
            means = [results.value(p, case, metric) for p in results.profiles]
            stds = [results.value(p, case, metric, "std") for p in results.profiles]
            marker = FUSION_STYLES[case]["marker"] if case in FUSIONS else ("o" if i == 0 else "D")
            linestyle = FUSION_STYLES[case]["linestyle"] if case in FUSIONS else "-"
            # Small offsets keep all error bars visible; categories are labeled.
            ax.errorbar(x + (i - 1.5) * .035, means, yerr=stds, color=colors[case], marker=marker,
                        linestyle=linestyle, linewidth=1.5, markersize=5, capsize=3, label=case_label(case))
        ax.set_xticks(x, [f"{results.noise_e(p):,.2f}".replace("5.00", "5") for p in results.profiles])
        ax.set_xlabel("Read-noise σ (e−; equally spaced settings)")
        ax.set_ylabel(label)
        ax.set_ylim(_bounds(results, metric))
        ax.grid(alpha=.2, linewidth=.6)
    fig.suptitle("Read-noise sensitivity with the same methods at every setting", y=.98, fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92), ncol=4, frameon=False, fontsize=10)
    fig.text(.5, .81, "Fixed 1 ms and 500 ms controls; no selection of a different best exposure at each noise level", ha="center", fontsize=10)
    _footer(fig, results)
    _save(fig, output, "noise_comparison", dpi)


def plot_diffraction_diagnostics(results, output, dpi=300):
    n = len(results.profiles)
    fig, axes = plt.subplots(n, 3, figsize=(15.8, 3.2 * n + 1.55), squeeze=False)
    fig.subplots_adjust(left=.07, right=.98, bottom=.145, top=.865, hspace=.55, wspace=.30)
    for col, (metric, label) in enumerate(DIAGNOSTICS):
        cases = results.cases[:-2] if col == 2 else results.cases
        factor = 100 if col == 2 else 1
        values = [(results.value(p, c, metric) * factor, results.value(p, c, metric, "std") * factor)
                  for p in results.profiles for c in cases]
        lo, hi = min(m-s for m,s in values), max(m+s for m,s in values)
        for row, profile in enumerate(results.profiles):
            ax = axes[row, col]
            _exposure_series(ax, results, profile, metric, include_fusion=col != 2, factor=factor)
            if col < 2:
                if lo > 0:
                    ax.set_yscale("log")
                    ax.set_ylim(lo / 1.6, hi * 1.6)
                else:
                    # Preserve zero/negative mean-SD endpoints instead of dropping them.
                    ax.set_yscale("symlog", linthresh=max(hi * 1e-4, 1e-10))
                    ax.set_ylim(*_bounds(results, metric))
                ax.set_ylabel(label + (" (log scale)" if lo > 0 else " (symlog)"))
            else:
                ax.set_ylim(*_bounds(results, metric, cases, factor=100))
                ax.set_ylabel(label)
            ax.set_title(f"({chr(97 + row * 3 + col)}) {results.noise_label(profile)}", loc="left")
    fig.suptitle("Diffraction error and measurement saturation", y=.985, fontsize=16)
    _legend(fig)
    size = results.manifest["input_shape"][-2:]
    radius = min(size) / 4
    fig.text(.055, .063,
             f"High-q: detector radius r ≥ {radius:g} pixels (min(frame shape)/4). Saturation is a measurement property; no HDR saturation value is inferred.", fontsize=9)
    _footer(fig, results, diffraction=True)
    _save(fig, output, "diffraction_diagnostics", dpi)


def generate_figures(source, output, dpi=300):
    """Render the four figure pairs and an auditable provenance sidecar."""
    if dpi < 72:
        raise ValueError("DPI must be at least 72")
    results = load_recorded_results(source)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    functions = [plot_exposure_comparison, plot_method_heatmaps, plot_noise_comparison, plot_diffraction_diagnostics]
    with plt.rc_context(STYLE):
        for function in functions:
            function(results, output, dpi)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def reference(path):
        import os
        return {"path": Path(os.path.relpath(path, output)).as_posix(), "sha256": digest(path)}
    sources = {"summary_metrics": reference(results.csv_path), "experiment_manifest": reference(results.manifest_path)}
    root = Path(__file__).resolve().parents[1]
    experiment_path = root / "experiments/paper_baseline_comparison.json"
    if experiment_path.is_file() and json.loads(experiment_path.read_text(encoding="utf-8")) == results.manifest["experiment"]:
        sources["matching_experiment_config"] = reference(experiment_path)
    names = ["exposure_comparison", "method_heatmaps", "noise_comparison", "diffraction_diagnostics"]
    provenance = {
        "sources": sources,
        "source_experiment": results.manifest["experiment"],
        "code_sha256": {"mlhdr_ptycho/comparison_figures.py": digest(Path(__file__)),
                        "scripts/plot_paper_comparisons.py": digest(root / "scripts/plot_paper_comparisons.py")},
        "method_order": list(results.cases), "profile_order": list(results.profiles),
        "noise_standard_deviation_e": {p: results.noise_e(p) for p in results.profiles},
        "fixed_noise_comparison_methods": ["single_1ms", "single_500ms", *FUSIONS],
        "statistics": "CSV means and sample standard deviations; error bars/bands are not confidence intervals",
        "successful_runs": {f"{p}/{c}": {k: results.rows[p, c][k] for k in ("runs", "successful_runs")} for p in results.profiles for c in results.cases},
        "reference": "Saved raw-data reconstruction, not object ground truth; diffraction errors use known pre-camera input",
        "normalization": "Quality metrics are plotted as recorded; saturation fractions are converted to percent. No recomputation, smoothing, reconstruction, or PNG digitization",
        "saturation": "Single-exposure measurement property; no fused-method saturation values plotted",
        "high_q_mask": "radius >= min(detector_frame_shape)/4, as in diffraction_comparison",
        "total_hdr_exposure_ms": results.total_exposure_ms,
        "acquisition_budget": "Unequal between HDR and single exposures; excludes dark frames and readout",
        "formats": ["png", "svg"], "png_dpi": dpi,
        "figures": {name: {ext: reference(output / f"{name}.{ext}") for ext in ("png", "svg")} for name in names},
    }
    (output / "figures_manifest.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return provenance
