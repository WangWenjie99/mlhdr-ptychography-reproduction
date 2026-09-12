#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlhdr_ptycho.data import load_diff_npy
from mlhdr_ptycho.ptylab_reconstruction import (
    PtyLabConfig,
    reconstruction_metrics,
    run_mpie,
    save_reconstruction_outputs,
)


def _parse_float_list(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PtyLab reconstruction parameters.")
    parser.add_argument("--input", default="diff.npy")
    parser.add_argument("--output", default="outputs/param_sweep")
    parser.add_argument("--scan-crop", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps", default="2,3,4,5,6,7,8")
    parser.add_argument("--probe-diameters", default="16,20,24,28,31")
    parser.add_argument("--initial-probes", default="circ_smooth,mean_ifft")
    parser.add_argument("--position-order", choices=["random", "sequential", "NA"], default="random")
    parser.add_argument("--position-correction", action="store_true")
    parser.add_argument("--position-correction-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    scan_crop = None if args.scan_crop == 0 else args.scan_crop
    diff = load_diff_npy(args.input, scan_crop=scan_crop)

    rows = []
    best = None
    best_payload = None
    for initial_probe in _parse_str_list(args.initial_probes):
        for step in _parse_float_list(args.steps):
            for diameter in _parse_float_list(args.probe_diameters):
                config = PtyLabConfig(
                    scan_step_px=step,
                    entrance_pupil_diameter_px=diameter,
                    iterations=args.iterations,
                    position_order=args.position_order,
                    initial_probe=initial_probe,
                    position_correction=args.position_correction,
                    position_correction_radius=args.position_correction_radius,
                    seed=args.seed,
                    quiet=True,
                )
                reconstruction, data, _ = run_mpie(diff.stack, diff.scan_shape, config)
                metrics = reconstruction_metrics(reconstruction, data)
                row = {
                    "initial_probe": initial_probe,
                    "scan_step_px": step,
                    "probe_diameter_px": diameter,
                    **metrics,
                }
                rows.append(row)
                score = metrics["final_error_per_frame"]
                print(
                    f"score={score:.6g} obj_std={metrics['object_amp_std']:.5g} "
                    f"step={step:g} dia={diameter:g} init={initial_probe}"
                )
                if best is None or score < best:
                    best = score
                    best_payload = (reconstruction, data, config)

    csv_path = output / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rows_sorted = sorted(rows, key=lambda row: row["final_error_per_frame"])
    print("\nBest configurations:")
    for row in rows_sorted[:10]:
        print(row)

    if args.save_best and best_payload is not None:
        reconstruction, data, config = best_payload
        save_reconstruction_outputs(output / "best", "best", reconstruction, data, config)
        print(f"Saved best reconstruction to {output / 'best'}")

    print(f"Wrote sweep table to {csv_path}")


if __name__ == "__main__":
    main()
