"""Figures imitating Liu et al. (IEEE TIM 73:4502711, 2024), Figs. 2 and 3.

The only inputs are the files written by ``scripts/run_paper_style_simulation.py``
(see ``DATA_CONTRACT.md`` in its output directory). The figures compare this
project's methods with each other; no number is ever estimated from a rendered
PNG, and every plotted value is read from those files.

Method encoding (identical in every figure):
    single      black solid line, circles
    lrfc        blue dash-dot, diamonds            (linear-response baseline following Eq. 16-17;
                                                   longest unsaturated exposure per pixel is our choice)
    ml_eq14_15  red dotted, squares                (ML-HDR Eq. 14-15, as published)
    ml_masked   green dash-dot-dot, triangles      (saturation mask, OUR extension)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

METHODS = ("single", "lrfc", "ml_eq14_15", "ml_masked")
BLUE, RED, GREEN = "#1F4FD1", "#C13C3C", "#12816D"
REF_GRAY = "#6E6E6E"
METHOD_STYLE = {
    "single": {"color": "black", "linestyle": "-", "marker": "o", "markersize": 8.5, "zorder": 3},
    "lrfc": {"color": BLUE, "linestyle": "-.", "marker": "D", "markersize": 8.5, "zorder": 4},
    "ml_eq14_15": {"color": RED, "linestyle": ":", "marker": "s", "markersize": 8.0, "zorder": 5},
    "ml_masked": {"color": GREEN, "linestyle": (0, (6.5, 2, 1.5, 2, 1.5, 2)), "marker": "^",
                  "markersize": 7.0, "zorder": 6},
}
LABEL = {
    "single": "Single exposure",
    "lrfc": "LRFC-style HDR (linear-response baseline)",
    "ml_eq14_15": "ML-HDR, Eq. 14–15 (published)",
    "ml_masked": "ML-HDR + saturation mask (extension)",
}
LETTER = {"single": "A", "lrfc": "B", "ml_eq14_15": "C", "ml_masked": "D"}

COMMON_FOOTER = ("Adapted simulation: our detector, object sampling and propagation distance differ from the paper; "
                 "parameters are recorded in meta.json. Our LRFC-style linear-response baseline selects the longest "
                 "unsaturated exposure per pixel; this selection rule is our implementation choice.")
FIGURES = ("fig2_bit_depth", "fig3_noise")
FIGURE_DESCRIPTIONS = {
    "fig2_bit_depth": "Paper Fig. 2 analogue: PSNR/SSIM/object-amplitude NRMSE vs ADC bit depth + 8-bit reconstruction mosaic",
    "fig3_noise": "Paper Fig. 3 analogue: PSNR/SSIM/object-amplitude NRMSE vs noise magnitude (dB)",
}

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11,
    "axes.labelsize": 12.5, "axes.labelweight": "bold", "axes.titlesize": 12,
    "axes.linewidth": 1.2, "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 5, "ytick.major.size": 5, "xtick.minor.size": 2.6, "ytick.minor.size": 2.6,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1, "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.frameon": False, "legend.fontsize": 10, "legend.handlelength": 3.0,
    "legend.borderaxespad": 0.35, "legend.labelspacing": 0.3,
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "mlhdr-paper-style-figures-v1",
    "image.interpolation": "nearest",
}

REQUIRED_FILES = ("meta.json", "bit_sweep.csv", "noise_sweep.csv", "cameraman_8bit.npz")
CSV_COLUMNS = ("experiment", "object", "method", "bits", "seed", "snr_db", "read_noise_e",
               "psnr_db", "ssim", "nrmse", "status")
CSV_FLOATS = ("snr_db", "read_noise_e", "psnr_db", "ssim", "nrmse", "frc_cutoff_nyquist", "diffraction_nrmse",
              "frc_resolution_um", "recon_seconds")
RUN_KEYS = ("truth", "truth_roi", "roi", "dx_um", "run_method", "run_bits", "run_label",
            "aligned_roi", "psnr_db", "ssim", "nrmse", "history_iteration",
            "history_wallclock_s", "history_object_nrmse", "frc_frequency_nyquist", "frc",
            "frc_threshold", "frc_cutoff_nyquist", "frc_resolution_um")
PER_RUN = ("aligned_roi", "psnr_db", "ssim", "nrmse", "history_wallclock_s",
           "history_object_nrmse", "frc", "frc_cutoff_nyquist", "frc_resolution_um")
REQUIRED_RUNS = ("single@8", "lrfc@8", "ml_eq14_15@8", "ml_masked@8")  # cameraman_8bit.npz


class DataContractError(ValueError):
    """The source directory does not satisfy DATA_CONTRACT.md."""


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Series:
    x: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class RunSet:
    """One NPZ with per-run arrays stacked on axis 0 (DATA_CONTRACT)."""
    path: Path
    arrays: dict
    labels: tuple

    def index(self, label):
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise DataContractError(f"{self.path.name}: run {label!r} not present (runs: {self.labels})") from exc

    def run(self, key, label):
        return self.arrays[key][self.index(label)]


@dataclass(frozen=True)
class PaperStyleData:
    source: Path
    files: dict
    meta: dict
    bit_rows: tuple
    noise_rows: tuple
    cameraman: RunSet

    def rows(self, experiment):
        if experiment == "bit_sweep":
            return self.bit_rows
        if experiment == "noise_sweep":
            return self.noise_rows
        raise ValueError(f"Unknown experiment {experiment!r}")

    def sweep(self, experiment, method, metric):
        """Mean, sample SD and n per x over successful rows (bits or snr_db)."""
        key = "bits" if experiment == "bit_sweep" else "snr_db"
        groups = {}
        for row in self.rows(experiment):
            # Diagnostic columns are optional in the input contract. Required
            # reconstruction metrics are still checked in _read_sweep.
            value = row.get(metric, math.nan)
            if row["method"] == method and row["status"] == "ok" and np.isfinite(value):
                groups.setdefault(row[key], []).append(value)
        xs = sorted(groups)
        vals = [np.asarray(groups[x], float) for x in xs]
        return Series(np.asarray(xs, float),
                      np.array([v.mean() for v in vals]),
                      np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in vals]),
                      np.array([len(v) for v in vals], int))

    def value(self, experiment, method, metric, x):
        s = self.sweep(experiment, method, metric)
        hit = np.flatnonzero(np.isclose(s.x, x))
        return float(s.mean[hit[0]]) if hit.size else math.nan

    def seed_counts(self, experiment):
        s = [self.sweep(experiment, m, "psnr_db").n for m in METHODS]
        s = np.concatenate([a for a in s if a.size]) if any(a.size for a in s) else np.array([0])
        return int(s.min()), int(s.max())


def _float(raw, where):
    if raw is None or str(raw).strip().lower() in ("", "nan", "none", "null"):
        return math.nan
    try:
        return float(raw)
    except ValueError as exc:
        raise DataContractError(f"{where}: not a number: {raw!r}") from exc


def _int(raw, where):
    value = _float(raw, where)
    if not np.isfinite(value) or not float(value).is_integer():
        raise DataContractError(f"{where}: expected an integer, got {raw!r}")
    return int(value)


def _read_sweep(path, experiment):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DataContractError(f"{path.name}: missing column(s) {missing}")
        rows = []
        for i, raw in enumerate(reader, start=2):
            where = f"{path.name} line {i}"
            if raw["experiment"] != experiment:
                raise DataContractError(f"{where}: experiment {raw['experiment']!r} != {experiment!r}")
            if raw["method"] not in METHODS:
                raise DataContractError(f"{where}: unknown method {raw['method']!r} (expected {METHODS})")
            if raw["status"] not in ("ok", "failed", "diverged"):
                raise DataContractError(f"{where}: unknown status {raw['status']!r}")
            row = dict(raw)
            row["bits"] = _int(raw["bits"], where + " bits")
            row["seed"] = _int(raw["seed"], where + " seed")
            for col in CSV_FLOATS:
                if col in raw:
                    row[col] = _float(raw[col], f"{where} {col}")
            row["snr_db"] = round(row["snr_db"], 6)
            if row["status"] == "ok" and not all(np.isfinite(row[m]) for m in ("psnr_db", "ssim", "nrmse")):
                raise DataContractError(f"{where}: status ok but a metric is missing")
            rows.append(row)
    if not rows:
        raise DataContractError(f"{path.name}: no rows")
    key = "bits" if experiment == "bit_sweep" else "snr_db"
    seen = set()
    for row in rows:
        k = (row["method"], row[key], row["seed"])
        if k in seen:
            raise DataContractError(f"{path.name}: duplicate row {k}")
        seen.add(k)
    absent = [m for m in METHODS if not any(r["method"] == m and r["status"] == "ok" for r in rows)]
    if absent:
        raise DataContractError(f"{path.name}: no successful rows for method(s) {absent}")
    return tuple(rows)


def _load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError) as exc:
        raise DataContractError(f"{path.name}: cannot read NPZ ({exc})") from exc


def _runset(path):
    arrays = _load_npz(path)
    missing = [k for k in RUN_KEYS if k not in arrays]
    if missing:
        raise DataContractError(f"{path.name}: missing key(s) {missing}")
    labels = tuple(str(v) for v in arrays["run_label"])
    if len(set(labels)) != len(labels):
        raise DataContractError(f"{path.name}: duplicate run labels {labels}")
    expected = tuple(f"{m}@{int(b)}" for m, b in zip(arrays["run_method"], arrays["run_bits"]))
    if expected != labels:
        raise DataContractError(f"{path.name}: run_label {labels} disagrees with run_method/run_bits")
    for key in PER_RUN:
        if key in arrays and arrays[key].shape[:1] != (len(labels),):
            raise DataContractError(f"{path.name}: {key} has shape {arrays[key].shape}, expected {len(labels)} runs on axis 0")
    k = arrays["history_iteration"].shape[0]
    for key in ("history_wallclock_s", "history_object_nrmse"):
        if arrays[key].shape[1] != k:
            raise DataContractError(f"{path.name}: {key} does not match history_iteration")
    f = arrays["frc_frequency_nyquist"].shape[0]
    if arrays["frc"].shape[1] != f or arrays["frc_threshold"].shape[0] != f:
        raise DataContractError(f"{path.name}: FRC arrays do not match frc_frequency_nyquist")
    y0, y1, x0, x1 = (int(v) for v in arrays["roi"])
    if arrays["aligned_roi"].shape[1:] != (y1 - y0, x1 - x0) or arrays["truth_roi"].shape != (y1 - y0, x1 - x0):
        raise DataContractError(f"{path.name}: aligned_roi/truth_roi do not match roi {arrays['roi']}")
    return RunSet(path, arrays, labels)


def load_paper_style_data(source):
    """Load and validate one ``outputs/paper_style/<profile>`` directory."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise DataContractError(f"Source directory does not exist: {source}")
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise DataContractError(f"{source}: missing contract file(s) {missing} "
                                "(is the simulation still running?)")
    files = {name: source / name for name in REQUIRED_FILES}
    if (source / "DATA_CONTRACT.md").is_file():
        files["DATA_CONTRACT.md"] = source / "DATA_CONTRACT.md"
    try:
        meta = json.loads(files["meta.json"].read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DataContractError(f"Invalid JSON in {source}: {exc}") from exc
    for key in ("profile", "seeds", "simulation_config"):
        if key not in meta:
            raise DataContractError(f"meta.json: missing {key!r}")
    for key in ("dx_um", "exposure_times_s", "full_well_e", "read_noise_e"):
        if key not in meta["simulation_config"]:
            raise DataContractError(f"meta.json: simulation_config lacks {key!r}")
    bit_rows = _read_sweep(files["bit_sweep.csv"], "bit_sweep")
    noise_rows = _read_sweep(files["noise_sweep.csv"], "noise_sweep")
    cameraman = _runset(files["cameraman_8bit.npz"])
    absent = [lab for lab in REQUIRED_RUNS if lab not in cameraman.labels]
    if absent:
        raise DataContractError(f"{cameraman.path.name}: required run(s) {absent} missing")
    return PaperStyleData(source, files, meta, bit_rows, noise_rows, cameraman)


# --------------------------------------------------------------------------
# Generic drawing helpers
# --------------------------------------------------------------------------
def _darker(color, factor=0.6):
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _wrap_measured(paragraphs, width_in, fontsize):
    """Greedy word wrap using rendered text widths (mathtext-aware)."""
    probe = plt.figure(figsize=(width_in + 1, 1))
    renderer = probe.canvas.get_renderer()
    cache = {}

    def width(s):
        if s not in cache:
            t = probe.text(0, 0, s, fontsize=fontsize)
            cache[s] = t.get_window_extent(renderer).width / probe.dpi
            t.remove()
        return cache[s]

    space = width("a a") - width("aa")
    limit = 0.97 * width_in  # slack for kerning across word boundaries
    lines = []
    for p in paragraphs:
        current, used = "", 0.0
        for word in p.split():
            w = width(word)
            if current and used + space + w > limit:
                lines.append(current)
                current, used = word, w
            else:
                current, used = (f"{current} {word}", used + space + w) if current else (word, w)
        lines.append(current)
    plt.close(probe)
    return lines


def _new_figure(width, content_height, paragraphs, fontsize=8.6):
    """Figure with a wrapped footer (common note + figure notes) below the content."""
    with plt.rc_context({"font.family": "serif", "font.serif": STYLE["font.serif"],
                         "mathtext.fontset": STYLE["mathtext.fontset"]}):
        lines = _wrap_measured((COMMON_FOOTER, *paragraphs), width - 0.4, fontsize)
    line_h = fontsize * 1.32 / 72
    footer = 0.1 + len(lines) * line_h + 0.08
    height = content_height + footer
    fig = plt.figure(figsize=(width, height))
    for i, line in enumerate(reversed(lines)):
        fig.text(0.15 / width, (0.1 + i * line_h) / height, line, fontsize=fontsize,
                 color="#333333", ha="left", va="bottom")
    fig.add_artist(Line2D([0.15 / width, 1 - 0.15 / width], [(footer - 0.03) / height] * 2,
                          color="#BBBBBB", linewidth=0.6))
    return fig, footer


def _axes_in(fig, left, bottom, width, height):
    """Axes placed in inches from the lower-left figure corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def _paper_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()


def _bold_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _image_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _tag(ax, x, y, text, color, *, ha="left", va="top", size=11):
    """Bold text on a small translucent white rounded box (legible on any image background)."""
    box = {"boxstyle": "round,pad=0.18,rounding_size=0.25", "facecolor": "white", "alpha": 0.82,
           "edgecolor": "none"}
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=size, fontweight="bold",
            bbox=box, zorder=30, clip_on=False)


def _panel_letter(ax, letter):
    ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=14,
            fontweight="bold", color="black", zorder=30)


def _save(fig, output, name, dpi):
    fig.savefig(output / f"{name}.png", dpi=dpi,
                metadata={"Software": None, "Title": name, "Source": "mlhdr_ptycho/paper_style_figures.py"})
    fig.savefig(output / f"{name}.svg",
                metadata={"Date": None, "Creator": "mlhdr_ptycho/paper_style_figures.py", "Title": name})
    plt.close(fig)
    return fig


def _to_in(ax, xy):
    return ax.transData.transform(np.asarray(xy, float).reshape(-1, 2)) / ax.figure.dpi


def _axes_box_in(ax):
    fig = ax.figure
    W, H = fig.get_size_inches()
    p = ax.get_position()
    return p.x0 * W, p.y0 * H, p.x1 * W, p.y1 * H


def _curve_points_in(ax, x, y, per_segment=24):
    pts = _to_in(ax, np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 2:
        return pts
    t = np.linspace(0, 1, per_segment, endpoint=False)
    seg = pts[:-1, None, :] + (pts[1:, None, :] - pts[:-1, None, :]) * t[None, :, None]
    return np.vstack([seg.reshape(-1, 2), pts[-1:]])


def _rect_dist(points, c, hw, hh):
    if len(points) == 0:
        return np.array([np.inf])
    dx = np.maximum(np.abs(points[:, 0] - c[0]) - hw, 0)
    dy = np.maximum(np.abs(points[:, 1] - c[1]) - hh, 0)
    return np.hypot(dx, dy)


def _rects_overlap(c, hw, hh, box, margin=0.02):
    x0, y0, x1, y1 = box
    return not (c[0] + hw < x0 - margin or c[0] - hw > x1 + margin
                or c[1] + hh < y0 - margin or c[1] - hh > y1 + margin)


def _text_size_in(fig, text, **kw):
    t = fig.text(0, 0, text, **kw)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def _legend_box_in(ax, legend):
    bb = legend.get_window_extent(ax.figure.canvas.get_renderer())
    d = ax.figure.dpi
    return bb.x0 / d, bb.y0 / d, bb.x1 / d, bb.y1 / d


ANGLES = (270, 90, 300, 240, 60, 120, 0, 180, 330, 210, 30, 150)
PREFERRED_ANGLES = (270, 90)
ARROWS = (0.28, 0.55, 0.85)
SOFT_WEIGHT = 0.25


def _best_combination(clear, pair, bonus, cap):
    """argmax over option combinations of min(cap, clearances, pair distances) + bonus."""
    n, k = clear.shape
    if k ** n <= 2_000_000:
        base = np.full((k,) * n, cap)
        extra = np.zeros((k,) * n)
        soft = np.zeros((k,) * n)
        terms = n + len(pair)
        for i in range(n):
            shape = [1] * n
            shape[i] = k
            base = np.minimum(base, clear[i].reshape(shape))
            extra = extra + bonus[i].reshape(shape)
            soft = soft + np.clip(clear[i], -1, cap).reshape(shape) / terms
        for (i, j), m in pair.items():
            shape = [1] * n
            shape[i], shape[j] = k, k
            base = np.minimum(base, m.reshape(shape))
            soft = soft + np.clip(m, -1, cap).reshape(shape) / terms
        total = base + extra + SOFT_WEIGHT * soft
        return tuple(int(v) for v in np.unravel_index(int(np.argmax(total)), total.shape))

    def score(choice):
        vals = [clear[i, c] for i, c in enumerate(choice)] + [m[choice[i], choice[j]] for (i, j), m in pair.items()]
        soft = sum(min(max(v, -1), cap) for v in vals) / len(vals)
        return min([cap] + vals) + sum(bonus[i, c] for i, c in enumerate(choice)) + SOFT_WEIGHT * soft

    best, best_s = None, -np.inf
    rng = np.random.default_rng(0)
    starts = [tuple(int(np.argmax(clear[i] + bonus[i])) for i in range(n))]
    starts += [tuple(int(v) for v in rng.integers(0, k, n)) for _ in range(8)]
    for start in starts:
        choice = list(start)
        for _ in range(20):
            changed = False
            for i in range(n):
                cands = [score(tuple(choice[:i] + [c] + choice[i + 1:])) for c in range(k)]
                c = int(np.argmax(cands))
                if cands[c] > score(tuple(choice)) + 1e-12:
                    choice[i], changed = c, True
            if not changed:
                break
        s = score(tuple(choice))
        if s > best_s:
            best, best_s = tuple(choice), s
    return best


def _place_callouts(ax, targets, obstacles, avoid=(), cap=0.30, angles=ANGLES, text_kw=None):
    """Place text callouts with arrows so labels avoid curves, legends and each other.

    ``targets``: list of dicts {xy (data), text, color}. ``obstacles``: (P, 2) points in
    inches. Each label may sit in one of ``angles`` x ``ARROWS`` (arrow length, inches)
    positions; the joint layout maximises the smallest clearance between labels,
    curves, arrow shafts and markers (capped), preferring short arrows and
    ``PREFERRED_ANGLES``.
    """
    if not targets:
        return []
    fig = ax.figure
    text_kw = dict(text_kw or {"fontsize": 12, "fontweight": "bold"})
    box = _axes_box_in(ax)
    T = _to_in(ax, [t["xy"] for t in targets])
    obstacles = np.asarray(obstacles, float).reshape(-1, 2)
    options = [(a, L) for L in ARROWS for a in angles]
    n, k = len(targets), len(options)
    centres = np.zeros((n, k, 2))
    half = np.zeros((n, 2))
    clear = np.zeros((n, k))
    bonus = np.zeros((n, k))
    shafts = np.zeros((n, k, 8, 2))
    for i, t in enumerate(targets):
        w, h = _text_size_in(fig, t["text"], **text_kw)
        hw, hh = w / 2 + 0.03, h / 2 + 0.015
        half[i] = hw, hh
        # Markers coinciding with this target are where the arrow must end anyway.
        others = T[[j for j in range(n) if j != i and np.hypot(*(T[j] - T[i])) > 0.15]].reshape(-1, 2)
        for j, (ang, length) in enumerate(options):
            a = math.radians(ang)
            u = np.array([math.cos(a), math.sin(a)])
            reach = length + min(hw / max(abs(u[0]), 1e-9), hh / max(abs(u[1]), 1e-9))
            c = T[i] + reach * u
            centres[i, j] = c
            shaft = c + (T[i] - c) * np.linspace(0.3, 0.9, 8)[:, None]
            shafts[i, j] = shaft
            bonus[i, j] = (0.03 if ang in PREFERRED_ANGLES else 0.0) - 0.03 * ARROWS.index(length)
            ok = (box[0] + 0.03 < c[0] - hw and c[0] + hw < box[2] - 0.03
                  and box[1] + 0.03 < c[1] - hh and c[1] + hh < box[3] - 0.03)
            ok = ok and not any(_rects_overlap(c, hw, hh, b) for b in avoid)
            if not ok:
                clear[i, j] = -np.inf
                continue
            d = _rect_dist(obstacles, c, hw, hh).min()
            if len(others):
                d = min(d, _rect_dist(others, c, hw, hh).min() - 0.06,
                        np.min(np.hypot(*(shaft[:, None, :] - others[None, :, :]).transpose(2, 0, 1))) - 0.06)
            clear[i, j] = min(d, cap)
    pair = {}
    for i, j in itertools.combinations(range(n), 2):
        m = np.zeros((k, k))
        for a in range(k):
            if not np.isfinite(clear[i, a]):
                continue
            for b in range(k):
                if not np.isfinite(clear[j, b]):
                    continue
                ci, cj = centres[i, a], centres[j, b]
                gap_x = abs(ci[0] - cj[0]) - half[i, 0] - half[j, 0]
                gap_y = abs(ci[1] - cj[1]) - half[i, 1] - half[j, 1]
                d = math.hypot(max(gap_x, 0), max(gap_y, 0)) if (gap_x > 0 or gap_y > 0) else -1.0
                m[a, b] = min(d, _rect_dist(shafts[j, b], ci, *half[i]).min(),
                              _rect_dist(shafts[i, a], cj, *half[j]).min())
        pair[i, j] = m
    choice = _best_combination(clear, pair, bonus, cap)
    placed = []
    inv = ax.transAxes.inverted()
    for i, t in enumerate(targets):
        c = centres[i, choice[i]]
        pos = inv.transform(c * fig.dpi)
        ann = ax.annotate(
            t["text"], xy=t["xy"], xytext=pos, textcoords="axes fraction", ha="center", va="center",
            color=t["color"], zorder=40, annotation_clip=False,
            arrowprops={"arrowstyle": "-|>", "color": t["color"], "lw": 1.4, "shrinkA": 1.5,
                        "shrinkB": 4.5, "mutation_scale": 11},
            **text_kw)
        ann.set_gid(t.get("gid"))
        placed.append(ann)
    return placed


def _best_text(ax, text, candidates, obstacles, avoid=(), **text_kw):
    """Put ``text`` at the candidate (axes fraction, ha, va) with the largest clearance."""
    fig = ax.figure
    best, best_d = None, -np.inf
    r = fig.canvas.get_renderer()
    box = _axes_box_in(ax)
    for x, y, ha, va in candidates:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **text_kw)
        bb = t.get_window_extent(r)
        t.remove()
        c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2]) / fig.dpi
        hw, hh = bb.width / fig.dpi / 2, bb.height / fig.dpi / 2
        inside = box[0] < c[0] - hw and c[0] + hw < box[2] and box[1] < c[1] - hh and c[1] + hh < box[3]
        if not inside or any(_rects_overlap(c, hw, hh, b) for b in avoid):
            continue
        d = _rect_dist(np.asarray(obstacles, float).reshape(-1, 2), c, hw, hh).min()
        if d > best_d + 1e-9:
            best, best_d = (x, y, ha, va), d
    if best is None:
        best = candidates[0]
    x, y, ha, va = best
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **text_kw)


def _style_kw(method):
    st = METHOD_STYLE[method]
    return {"color": st["color"], "linestyle": st["linestyle"], "marker": st["marker"],
            "markersize": st["markersize"], "linewidth": 1.9,
            "markerfacecolor": st["color"], "markeredgecolor": _darker(st["color"]),
            "markeredgewidth": 0.9, "zorder": st["zorder"]}


def _plot_sweep(ax, data, experiment, metric, method, label):
    s = data.sweep(experiment, method, metric)
    if s.x.size == 0:
        return s, None
    kw = _style_kw(method)
    if np.any(s.n > 1):
        container = ax.errorbar(s.x, s.mean, yerr=s.sd, capsize=3.2, elinewidth=1.0,
                                capthick=1.0, label=label, **kw)
        line = container.lines[0]
    else:
        line, = ax.plot(s.x, s.mean, label=label, **kw)
    line.set_gid(f"ours:{experiment}:{metric}:{method}")
    return s, line


def _limits_under_legend(ax, legend, lo, hi):
    """y-limits that keep [lo, hi] (padded below by 7 % of the span) 0.05 axes heights under the legend."""
    lb = _legend_box_in(ax, legend)
    box = _axes_box_in(ax)
    frac = (lb[1] - box[1]) / (box[3] - box[1]) - 0.05
    frac = min(max(frac, 0.4), 0.95)
    span = (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
    lo2 = lo - 0.07 * span
    ax.set_ylim(lo2, lo2 + (hi - lo2) / frac)


def _seed_note(data, experiment):
    lo, hi = data.seed_counts(experiment)
    n = str(lo) if lo == hi else f"{lo}–{hi}"
    if hi > 1:
        return f"n = {n} seeds per point: mean ± sample SD (error bars)."
    return "n = 1 seed per point (no error bars)."


def _snapshot_note(data, what):
    seeds = data.meta.get("seeds") or [0]
    return f"{what} are single runs (seed {seeds[0]})."


def findings(data):
    """Numbers quoted in footers and docs, all computed from the source files."""
    out = {}
    ml = data.sweep("bit_sweep", "ml_eq14_15", "nrmse")
    out["ml_nrmse_range"] = (float(ml.mean.min()), float(ml.mean.max()))
    diffraction_nrmse = data.value("bit_sweep", "ml_eq14_15", "diffraction_nrmse", 8)
    if np.isfinite(diffraction_nrmse):
        out["ml_diff_nrmse_8"] = diffraction_nrmse
    out["nrmse8"] = {m: data.value("bit_sweep", m, "nrmse", 8) for m in METHODS}
    out["nrmse16"] = {m: data.value("bit_sweep", m, "nrmse", 16) for m in METHODS}
    out["hdr8_matches_single16"] = min(out["nrmse8"]["lrfc"], out["nrmse8"]["ml_masked"]) <= out["nrmse16"]["single"]
    snr = data.sweep("noise_sweep", "lrfc", "psnr_db").x
    top = float(snr.max()) if snr.size else math.nan
    out["snr_top"] = top
    out["psnr_top"] = {m: data.value("noise_sweep", m, "psnr_db", top) for m in METHODS}
    noise = data.meta.get("noise_sweep", {}).get("read_noise_e") or {}
    sigma_e = noise.get(f"{top:g}") if noise and np.isfinite(top) else None
    cfg = data.meta["simulation_config"]
    n_dark = cfg.get("dark_frames")
    masked_bits = {row["bits"] for row in data.noise_rows if row["method"] == "ml_masked"}
    if sigma_e and n_dark and len(masked_bits) == 1:
        # One ADC count (LSB) of the ML + mask noise-sweep runs, in electrons.
        e_per_count = cfg["full_well_e"] / (2 ** masked_bits.pop() - 1)
        sigma = float(sigma_e) / e_per_count
        out["sigma_lsb_top"] = sigma
        # P(all dark reads = 0) when the ADC clips electrons at 0: Phi(0.5 / sigma) ** n_dark
        out["p_zero_dark_var"] = (0.5 * (1 + math.erf(0.5 / sigma / math.sqrt(2)))) ** int(n_dark)
    return out


METRICS = (("psnr_db", "PSNR (dB)"), ("ssim", "SSIM"), ("nrmse", "Object-amplitude NRMSE"))


# --------------------------------------------------------------------------
# Fig. 2 — bit depth
# --------------------------------------------------------------------------
def _sweep_panel(fig, ax, data, experiment, metric, labels, *, reference=None, arrows=False,
                 xticks=None, xlim=None, xlabel=""):
    lines = {}
    if arrows:
        ax.axvline(8, color="#D2D2D2", linewidth=7, zorder=0, solid_capstyle="butt")
    for method in METHODS:
        s, line = _plot_sweep(ax, data, experiment, metric, method, labels[method])
        if line is not None:
            lines[method] = (s, line)
    _paper_axes(ax)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.xaxis.set_minor_locator(MultipleLocator((xticks[1] - xticks[0]) / 2))
    ax.set_xlabel(xlabel)
    legend = ax.legend(loc="upper right", fontsize=10, handlelength=3.4)
    ys = [s.mean + s.sd for s, _ in lines.values()] + [s.mean - s.sd for s, _ in lines.values()]
    ys = np.concatenate(ys)
    lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
    if reference is not None and np.isfinite(reference):
        lo, hi = min(lo, reference), max(hi, reference)
    _limits_under_legend(ax, legend, lo, hi)
    obstacles = [_curve_points_in(ax, s.x, s.mean) for s, _ in lines.values()]
    avoid = [_legend_box_in(ax, legend)]
    ref_line = None
    if reference is not None and np.isfinite(reference):
        ref_line = ax.axhline(reference, color=REF_GRAY, linestyle=(0, (5, 3)), linewidth=1.2, zorder=1)
        ref_line.set_gid(f"reference:{experiment}:{metric}:single@16")
        x0, x1 = ax.get_xlim()
        ref_pts = _curve_points_in(ax, [x0, x1], [reference, reference], per_segment=80)
        ylo, yhi = ax.get_ylim()
        off = 0.012 * 1
        yfrac = (reference - ylo) / (yhi - ylo)
        cands = [(0.03, yfrac + off, "left", "bottom"), (0.03, yfrac - off, "left", "top"),
                 (0.97, yfrac + off, "right", "bottom"), (0.97, yfrac - off, "right", "top"),
                 (0.5, yfrac + off, "center", "bottom"), (0.5, yfrac - off, "center", "top")]
        ref_text = _best_text(ax, "single exposure @ 16 bit", cands, np.vstack(obstacles), avoid,
                              fontsize=9.5, color=REF_GRAY, style="italic", zorder=8)
        ref_text.set_gid(f"reference-label:{metric}")
        bb = ref_text.get_window_extent(fig.canvas.get_renderer())
        avoid.append((bb.x0 / fig.dpi, bb.y0 / fig.dpi, bb.x1 / fig.dpi, bb.y1 / fig.dpi))
        obstacles.append(ref_pts)
    if arrows:
        targets = []
        for method, (s, _) in lines.items():
            hit = np.flatnonzero(np.isclose(s.x, 8))
            if hit.size:
                targets.append({"xy": (8.0, float(s.mean[hit[0]])), "text": LETTER[method],
                                "color": METHOD_STYLE[method]["color"], "gid": f"arrow:{metric}:{method}"})
        markers = np.vstack([_to_in(ax, np.column_stack([s.x, s.mean])) for s, _ in lines.values()])
        _place_callouts(ax, targets, np.vstack(obstacles + [markers]), avoid=avoid,
                        text_kw={"fontsize": 13, "fontweight": "bold"})
    return lines, legend


def _mosaic(fig, data, cell, footer_note):
    """Panel (d): rounded dashed box with Truth + A-D tiles (shared gray range)."""
    W, H = fig.get_size_inches()
    x0, y0 = cell.x0 * W, cell.y0 * H
    w, h = cell.width * W, cell.height * H
    pad, title_h = 0.14, 0.3
    tile = min((w - 2 * pad) / 3, (h - 2 * pad - title_h) / 2)
    mw, mh = 3 * tile, 2 * tile
    bx0 = x0 + (w - (mw + 2 * pad)) / 2
    by0 = y0 + (h - (mh + 2 * pad + title_h)) / 2
    bw, bh = mw + 2 * pad, mh + 2 * pad + title_h
    fig.add_artist(FancyBboxPatch((bx0 / W, by0 / H), bw / W, bh / H,
                                  boxstyle=f"round,pad=0,rounding_size={0.18 / W}",
                                  transform=fig.transFigure, facecolor="#F1F1F1", edgecolor="#8C8C8C",
                                  linestyle=(0, (4, 3)), linewidth=1.1, zorder=-5))
    fig.text((bx0 + 0.12) / W, (by0 + bh - 0.1) / H, "(d)", fontsize=14, fontweight="bold", va="top")
    fig.text((bx0 + bw / 2) / W, (by0 + bh - 0.1) / H, "Reconstruction results (8 bit)",
             fontsize=13, fontweight="bold", ha="center", va="top")
    rs = data.cameraman
    truth = np.abs(rs.arrays["truth_roi"])
    lo, hi = float(truth.min()), float(truth.max())
    vmin, vmax = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    tiles = [("truth", truth, "Truth", "", "black")]
    for method in METHODS:
        lab = f"{method}@8"
        amp = np.abs(rs.run("aligned_roi", lab))
        name = {"single": "Single", "lrfc": "LRFC-style HDR", "ml_eq14_15": "ML Eq. 14–15",
                "ml_masked": "ML + mask"}[method]
        tiles.append((method, amp, name, LETTER[method], METHOD_STYLE[method]["color"]))
    ims = []
    for k, (key, img, name, letter, color) in enumerate(tiles):
        r, c = divmod(k, 3)
        ax = _axes_in(fig, bx0 + pad + c * tile, by0 + pad + (1 - r) * tile, tile, tile)
        im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        im.set_gid(f"mosaic:{key}")
        ims.append(im)
        _image_axes(ax)
        _tag(ax, 0.035, 0.965, name, color)
        if letter:
            _tag(ax, 0.96, 0.04, letter, color, ha="right", va="bottom", size=13)
    # Sixth tile: shared gray scale.
    info_x, info_y = bx0 + pad + 2 * tile, by0 + pad
    cax = _axes_in(fig, info_x + 0.18 * tile, info_y + 0.52 * tile, 0.64 * tile, 0.09 * tile)
    bar = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    bar.ax.tick_params(labelsize=9, direction="in", length=3)
    bar.ax.minorticks_off()
    bar.set_ticks([round(vmin + 0.05 * (hi - lo), 2), round((vmin + vmax) / 2, 2), round(vmax - 0.05 * (hi - lo), 2)])
    fig.text((info_x + tile / 2) / W, (info_y + 0.76 * tile) / H, "|O|, shared gray scale",
             ha="center", va="center", fontsize=9.5)
    fig.text((info_x + tile / 2) / W, (info_y + 0.22 * tile) / H, footer_note,
             ha="center", va="center", fontsize=8.8, color="#444444", linespacing=1.25)
    return (vmin, vmax)


def _fig2_findings(data):
    f = findings(data)
    n8, n16 = f["nrmse8"], f["nrmse16"]
    diffraction_note = (f"its fused diffraction rate deviates from the noiseless rate (8-bit diffraction NRMSE "
                        f"{f['ml_diff_nrmse_8']:.2f}) and " if "ml_diff_nrmse_8" in f else "")
    text = (f"Findings: the published ML-HDR Eq. 14–15 keeps saturated samples, so {diffraction_note}its "
            f"object NRMSE stays {f['ml_nrmse_range'][0]:.2f}–{f['ml_nrmse_range'][1]:.2f} at every bit depth. At 8 bit, "
            f"LRFC-style HDR and ML + mask reach NRMSE {n8['lrfc']:.3f} / {n8['ml_masked']:.3f} vs {n16['single']:.3f} for the "
            "single exposure at 16 bit, so ")
    return text + ("at least one 8-bit HDR result matches the 16-bit single exposure." if f["hdr8_matches_single16"]
                   else "neither 8-bit HDR result reaches the 16-bit single exposure.")


def plot_fig2_bit_depth(data, output, dpi=300):
    bits_single = data.sweep("bit_sweep", "single", "psnr_db").x
    notes = [
        "Cameraman object; read noise 5 e$^-$; single = one exposure at the auto-exposure time; HDR methods fuse all "
        "7 exposures (0.5–500 ms). Metrics vs ground truth on the scanned ROI after a global complex-scalar and "
        "sub-pixel-shift alignment; object-amplitude NRMSE = ‖|O|−|O$_{true}$|‖/‖|O$_{true}$|‖. "
        "The paper's RMS error definition is unspecified.",
        _seed_note(data, "bit_sweep") + " Letters A–D mark the 8-bit results shown in (d). Where markers coincide, "
        "the smaller green triangles are drawn over the blue diamonds. Gray dashed line = our single exposure at 16 bit. "
        + _snapshot_note(data, "The (d) images"),
        _fig2_findings(data),
    ]
    W = 12.4
    fig, footer = _new_figure(W, 10.6, notes)
    H = fig.get_figheight()
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.985, bottom=(footer + 0.62) / H, top=1 - 0.12 / H,
                          hspace=0.26, wspace=0.2)
    xticks = list(range(2, 21, 2))
    for k, (metric, ylabel) in enumerate(METRICS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ref = data.value("bit_sweep", "single", metric, 16) if np.any(np.isclose(bits_single, 16)) else None
        _sweep_panel(fig, ax, data, "bit_sweep", metric, LABEL, reference=ref, arrows=True,
                     xticks=xticks, xlim=(0.8, 21.2), xlabel="Dynamic range (bits)")
        ax.set_ylabel(ylabel)
        _panel_letter(ax, "abc"[k])
        _bold_ticks(ax)
    cell = gs[1, 1].get_position(fig)
    auto = data.meta.get("objects", {}).get("cameraman", {}).get("auto_exposure_s")
    note = ("A: single exposure" + (f" ({auto * 1e3:g} ms)" if auto else "") + "\nB–D: 7 exposures fused"
            "\nC: Eq. 14–15 as published\nD: + saturation mask (ours)")
    _mosaic(fig, data, cell, note)
    return _save(fig, output, "fig2_bit_depth", dpi)


# --------------------------------------------------------------------------
# Fig. 3 — noise
# --------------------------------------------------------------------------
def _noise_labels(data):
    bits = data.meta.get("noise_sweep", {}).get("bits", {"single": 16, "lrfc": 8, "ml_eq14_15": 8, "ml_masked": 8})
    return {
        "single": f"Single exposure ({bits['single']} bit)",
        "lrfc": f"LRFC-style HDR ({bits['lrfc']} bit)",
        "ml_eq14_15": f"ML-HDR, Eq. 14–15 (published, {bits['ml_eq14_15']} bit)",
        "ml_masked": f"ML-HDR + mask (extension, {bits['ml_masked']} bit)",
    }


def _sci(v, digits=2):
    exp = int(math.floor(math.log10(abs(v)))) if v else 0
    mant = v / 10 ** exp
    return f"{mant:.{digits}g}×10$^{{{exp}}}$" if exp not in (0, 1, 2) else f"{v:.{digits + 1}g}"


def _snr_definition(data):
    """Footer text for the noise axis, from meta.json (engine definition)."""
    cfg = data.meta["simulation_config"]
    noise = data.meta.get("noise_sweep", {})
    definition = noise.get("definition", "")
    if "20*log10(full_well_e / read_noise_e)" not in definition.replace("  ", " "):
        raise DataContractError(f"meta.json: unexpected noise definition {definition!r}")
    text = (f"Noise magnitude is OUR definition (the paper does not define it): SNR$_{{dB}}$ = "
            f"20·log$_{{10}}$(full well / σ$_{{read}}$), full well = {_sci(cfg['full_well_e'])} e$^-$")
    table = noise.get("read_noise_e") or {}
    if table:
        pts = sorted((float(k), float(v)) for k, v in table.items())
        (d0, s0), (d1, s1) = pts[0], pts[-1]
        text += f"; σ$_{{read}}$ = {_sci(s0)} e$^-$ at {d0:g} dB … {_sci(s1)} e$^-$ at {d1:g} dB"
    return text + "."


def _fig3_findings(data):
    """Footer note on ML + mask vs LRFC-style HDR at the highest SNR ('' when it does not apply)."""
    f = findings(data)
    top, p = f["snr_top"], f["psnr_top"]
    if not (p["ml_masked"] < p["lrfc"] and "p_zero_dark_var" in f):
        return ""
    return (f"ML + mask falls below LRFC-style HDR at high SNR ({p['ml_masked']:.1f} vs {p['lrfc']:.1f} dB PSNR at "
            f"{top:g} dB): Eq. 14 weights use the per-pixel dark variance σ$_B^2$ from the dark frames; at σ$_{{read}}$ ≈ "
            f"{f['sigma_lsb_top']:.2f} LSB about {100 * f['p_zero_dark_var']:.0f} % of pixels get σ$_B^2$ = 0 exactly "
            "(all dark reads 0, since the ADC clips at 0), giving their noisy short exposures near-infinite weight — a "
            "property of the estimator as written, not a change we made.")


def plot_fig3_noise(data, output, dpi=300):
    notes = [
        _snr_definition(data),
        "Cameraman object; single exposure at 16 bit, HDR methods at 8 bit, as in the paper. Metrics vs ground truth "
        "(error is object-amplitude NRMSE; the paper's RMS definition is unspecified). " + _seed_note(data, "noise_sweep"),
    ]
    note = _fig3_findings(data)
    if note:
        notes.append(note)
    W = 16.5
    fig, footer = _new_figure(W, 5.5, notes)
    H = fig.get_figheight()
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.99, bottom=(footer + 0.6) / H, top=1 - 0.12 / H, wspace=0.2)
    labels = _noise_labels(data)
    for k, (metric, ylabel) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, k])
        _sweep_panel(fig, ax, data, "noise_sweep", metric, labels, xticks=list(range(6, 55, 6)),
                     xlim=(3, 57), xlabel="Noise magnitude (dB)")
        ax.set_ylabel(ylabel)
        _panel_letter(ax, "abc"[k])
        _bold_ticks(ax)
    return _save(fig, output, "fig3_noise", dpi)


# --------------------------------------------------------------------------
# Summary table (key numbers of this reproduction)
# --------------------------------------------------------------------------
SUMMARY_FIELDS = ("section", "quantity", "method", "bits", "value", "unit", "note")


def summary_rows(data):
    """Key numbers of this reproduction (one value per row), all read from the source files."""
    rows = []

    def add(section, quantity, method, bits, value, unit, note=""):
        if isinstance(value, float):
            value = "" if not np.isfinite(value) else f"{value:.4g}"
        rows.append({"section": section, "quantity": quantity, "method": method, "bits": bits,
                     "value": "" if value is None else str(value), "unit": unit, "note": note})

    def seeds(method, bits):
        hit = data.sweep("bit_sweep", method, "psnr_db")
        n = hit.n[np.isclose(hit.x, bits)]
        return f"mean over seeds (n = {int(n[0])}), vs ground truth" if n.size else ""

    units = {"psnr_db": "dB", "ssim": "", "nrmse": ""}
    for metric in ("psnr_db", "ssim", "nrmse"):
        for method in METHODS:
            add("bit_sweep_8bit", metric, method, 8, data.value("bit_sweep", method, metric, 8), units[metric],
                seeds(method, 8))
        add("bit_sweep_16bit_reference", metric, "single", 16, data.value("bit_sweep", "single", metric, 16),
            units[metric], seeds("single", 16))
    single16 = {m: data.value("bit_sweep", "single", m, 16) for m in ("psnr_db", "ssim")}
    for method in ("lrfc", "ml_eq14_15", "ml_masked"):
        psnr = data.value("bit_sweep", method, "psnr_db", 8) - single16["psnr_db"]
        add("hdr8_vs_single16", "psnr_db_8bit_minus_single_16bit", method, "8 vs 16", f"{psnr:+.2f}", "dB",
            "bit-sweep means; >= 0 means the 8-bit result matches or exceeds the 16-bit single exposure")
        ssim = data.value("bit_sweep", method, "ssim", 8) - single16["ssim"]
        add("hdr8_vs_single16", "ssim_8bit_minus_single_16bit", method, "8 vs 16", f"{ssim:+.3f}", "",
            "bit-sweep means; >= 0 means the 8-bit result matches or exceeds the 16-bit single exposure")
    return rows


def write_summary_csv(data, path):
    rows = summary_rows(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def copy_small_data(data, destination):
    """Copy the CSV and JSON files of the source (no NPZ) into ``destination``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, path in data.files.items():
        if path.suffix.lower() in (".csv", ".json"):
            shutil.copyfile(path, destination / name)
            copied.append(destination / name)
    return copied


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _display(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


PLOTS = {
    "fig2_bit_depth": plot_fig2_bit_depth,
    "fig3_noise": plot_fig3_noise,
}


def generate_figures(source, output, dpi=300):
    """Render all paper-style figures (PNG + SVG) and ``figures_manifest.json``."""
    if dpi < 50:
        raise ValueError("DPI must be at least 50")
    data = load_paper_style_data(source)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name in FIGURES:
            PLOTS[name](data, output, dpi)
    meta = data.meta
    manifest = {
        "generator": "mlhdr_ptycho/paper_style_figures.py",
        "source_dir": _display(data.source),
        "source_files": {name: {"path": _display(p), "sha256": _sha256(p)} for name, p in sorted(data.files.items())},
        "source_meta": {
            "profile": meta.get("profile"), "created": meta.get("created"), "command": meta.get("command"),
            "profile_settings": meta.get("profile_settings"), "seeds": meta.get("seeds"),
            "experiments": meta.get("experiments"), "failures": meta.get("failures"),
            "runtime_s": (meta.get("runtime_s") or {}).get("total_wallclock"),
            "noise_definition": (meta.get("noise_sweep") or {}).get("definition"),
            "simulation_config": {k: meta["simulation_config"].get(k) for k in (
                "wavelength_m", "distance_m", "detector_pixels", "dx_um", "exposure_times_s",
                "photon_flux_per_s", "full_well_e", "read_noise_e", "dark_current_e_per_s", "dark_frames")},
            "mpie_iterations": (meta.get("mpie_config") or {}).get("iterations"),
        },
        "code_sha256": {"mlhdr_ptycho/paper_style_figures.py": _sha256(Path(__file__))},
        "conventions": {
            "methods": {m: {"label": LABEL[m], "color": METHOD_STYLE[m]["color"], "marker": METHOD_STYLE[m]["marker"]}
                        for m in METHODS},
            "statistics": "Sweep points = mean over successful seeds; error bars = sample SD (only when n > 1)",
            "numbers": "All plotted values are read from the source CSV/NPZ/JSON files; "
                       "nothing is estimated from rendered images",
        },
        "formats": ["png", "svg"], "png_dpi": dpi,
        "figures": {name: {"description": FIGURE_DESCRIPTIONS[name],
                           **{ext: {"file": f"{name}.{ext}", "sha256": _sha256(output / f"{name}.{ext}")}
                              for ext in ("png", "svg")}} for name in FIGURES},
    }
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "plot_paper_style_figures.py"
    if script.is_file():
        manifest["code_sha256"]["scripts/plot_paper_style_figures.py"] = _sha256(script)
    (output / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
