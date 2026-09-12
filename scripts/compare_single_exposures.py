#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

Path("/private/tmp/mlhdr_mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mlhdr_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlhdr_ptycho.data import load_diff_npy, normalize_stack
from mlhdr_ptycho.ml_hdr import (
    CameraModel,
    ml_hdr_fusion,
    simulate_multiexposure,
    single_exposure_rate,
)
from mlhdr_ptycho.ptylab_reconstruction import (
    PtyLabConfig,
    coverage_roi,
    object_coverage,
    reconstruction_metrics,
    run_mpie,
    save_reconstruction_outputs,
)


def _parse_exposure_ms(value: str) -> np.ndarray:
    return np.asarray([float(v) for v in value.split(",")], dtype=np.float32) * 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare single-exposure ptychography against ML-HDR fusion."
    )
    parser.add_argument("--input", default="diff.npy")
    parser.add_argument("--output", default="outputs/single_exposure_compare")
    parser.add_argument("--scan-crop", type=int, default=21)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--scan-step-px", type=float, default=8.0)
    parser.add_argument("--probe-diameter-px", type=float, default=31.0)
    parser.add_argument("--initial-probe", choices=["circ", "circ_smooth", "mean_ifft"], default="circ_smooth")
    parser.add_argument("--position-order", choices=["random", "sequential", "NA"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exposure-ms", default="0.5,1,5,10,50,100,500")
    parser.add_argument("--bit-depth", type=int, default=8)
    parser.add_argument("--full-well", type=float, default=2.5e6)
    parser.add_argument("--read-noise", type=float, default=5.0)
    parser.add_argument("--dark-current", type=float, default=80.0)
    parser.add_argument("--saturation-fraction", type=float, default=1.25)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    scan_crop = None if args.scan_crop == 0 else args.scan_crop
    diff = load_diff_npy(args.input, scan_crop=scan_crop)
    clean_stack = normalize_stack(diff.stack)

    camera = CameraModel(
        bit_depth=args.bit_depth,
        full_well=args.full_well,
        read_noise=args.read_noise,
        dark_current=args.dark_current,
        saturation_fraction_at_longest=args.saturation_fraction,
    )
    exposure_times_s = _parse_exposure_ms(args.exposure_ms)
    measurement = simulate_multiexposure(
        clean_stack, exposure_times_s=exposure_times_s, camera=camera, seed=args.seed
    )

    config = PtyLabConfig(
        scan_step_px=args.scan_step_px,
        entrance_pupil_diameter_px=args.probe_diameter_px,
        iterations=args.iterations,
        position_order=args.position_order,
        initial_probe=args.initial_probe,
        seed=args.seed,
        quiet=True,
    )

    cases: list[tuple[str, str, np.ndarray, bool]] = []
    cases.append(("raw_clean", "raw clean", clean_stack, True))
    for i, exposure in enumerate(exposure_times_s):
        label = f"single_{i}_{exposure * 1e3:g}ms"
        title = f"single {exposure * 1e3:g} ms"
        single = single_exposure_rate(measurement, i)
        has_signal = bool(np.max(single) > 1e-12)
        cases.append((label, title, _safe_normalize(single), has_signal))
    cases.append(("mlhdr", "ML-HDR", _safe_normalize(ml_hdr_fusion(measurement)), True))

    rows = []
    preview_fields = []
    for label, title, ptychogram, has_signal in cases:
        if not has_signal:
            rows.append(
                {
                    "case": label,
                    "title": title,
                    "status": "skipped_no_signal",
                    "saturation_fraction": 0.0,
                    "final_error_sum": "",
                    "final_error_per_frame": "",
                    "object_amp_min": "",
                    "object_amp_max": "",
                    "object_amp_std": "",
                    "object_roi_amp_min": "",
                    "object_roi_amp_max": "",
                    "object_roi_amp_std": "",
                    "coverage_max": "",
                    "probe_amp_min": "",
                    "probe_amp_max": "",
                    "num_frames": diff.stack.shape[0],
                    "object_pixels": "",
                }
            )
            preview_fields.append((f"{title} (no signal)", np.zeros(diff.frame_shape)))
            print(f"{label}: skipped, no signal after dark subtraction")
            continue

        reconstruction, data, _ = run_mpie(ptychogram, diff.scan_shape, config)
        save_reconstruction_outputs(output / label, label, reconstruction, data, config)
        metrics = reconstruction_metrics(reconstruction, data)
        status = "ok"
        if not np.isfinite(metrics["final_error_per_frame"]):
            status = "failed_nonfinite"
        sat_fraction = ""
        if label.startswith("single_"):
            idx = int(label.split("_")[1])
            sat_fraction = float(
                np.mean(measurement.z[idx] == (1 << args.bit_depth) - 1)
            )
        row = {
            "case": label,
            "title": title,
            "status": status,
            "saturation_fraction": sat_fraction,
            **metrics,
        }
        rows.append(row)

        obj = np.squeeze(reconstruction.object)
        roi = coverage_roi(object_coverage(reconstruction))
        if roi is not None:
            obj = obj[roi]
        preview_fields.append((title, np.nan_to_num(np.abs(obj), nan=0.0)))
        print(
            f"{label}: error/frame={metrics['final_error_per_frame']:.6g}, "
            f"roi_std={metrics['object_roi_amp_std']:.6g}, sat={sat_fraction}"
        )

    csv_path = output / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _save_contact_sheet(output / "object_roi_amplitude_comparison.png", preview_fields)
    _save_metric_plot(output / "error_per_frame_comparison.png", rows)
    print(f"Wrote metrics to {csv_path}")


def _save_contact_sheet(path: Path, fields: list[tuple[str, np.ndarray]]) -> None:
    cols = 3
    rows = int(np.ceil(len(fields) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, field) in zip(axes, fields):
        lo, hi = np.percentile(field, [1, 99])
        ax.imshow(field, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(fields):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_metric_plot(path: Path, rows: list[dict]) -> None:
    labels = []
    values = []
    colors = []
    for row in rows:
        value = row.get("final_error_per_frame", "")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        if not np.isfinite(value):
            continue
        labels.append(row["title"])
        values.append(value)
        colors.append("#d95f02" if row["case"] == "mlhdr" else "#7570b3")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("PtyLab error / frame")
    ax.set_title("Single exposure vs ML-HDR")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _safe_normalize(stack: np.ndarray) -> np.ndarray:
    max_value = float(np.max(stack))
    if max_value <= 1e-12:
        return np.asarray(stack, dtype=np.float32)
    return normalize_stack(stack)


if __name__ == "__main__":
    main()
