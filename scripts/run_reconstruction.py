#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
    reconstruction_metrics,
    run_mpie,
    save_reconstruction_outputs,
)


def _parse_exposure_ms(value: str) -> np.ndarray:
    return np.asarray([float(v) for v in value.split(",")], dtype=np.float32) * 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML-HDR/PtyLab reconstruction.")
    parser.add_argument("--input", default="diff.npy")
    parser.add_argument("--output", default="outputs/reconstruction")
    parser.add_argument("--mode", choices=["raw", "mlhdr", "single"], default="raw")
    parser.add_argument("--scan-crop", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--scan-step-px", type=float, default=1.0)
    parser.add_argument("--dxp-um", type=float, default=1.0)
    parser.add_argument("--detector-pixel-um", type=float, default=5.5)
    parser.add_argument("--probe-diameter-px", type=float, default=24.0)
    parser.add_argument("--wavelength-nm", type=float, default=632.8)
    parser.add_argument("--position-order", choices=["random", "sequential", "NA"], default="random")
    parser.add_argument("--initial-probe", choices=["circ", "circ_smooth", "mean_ifft"], default="circ_smooth")
    parser.add_argument("--position-correction", action="store_true")
    parser.add_argument("--position-correction-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exposure-ms", default="0.5,1,5,10,50,100,500")
    parser.add_argument("--bit-depth", type=int, default=8)
    parser.add_argument("--single-exposure-index", type=int, default=3)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    scan_crop = None if args.scan_crop == 0 else args.scan_crop
    diff = load_diff_npy(args.input, scan_crop=scan_crop)
    clean_stack = normalize_stack(diff.stack)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    measurement_meta = {}
    if args.mode == "raw":
        ptychogram = clean_stack
    else:
        exposure_times = _parse_exposure_ms(args.exposure_ms)
        camera = CameraModel(bit_depth=args.bit_depth)
        measurement = simulate_multiexposure(
            clean_stack,
            exposure_times_s=exposure_times,
            camera=camera,
            seed=args.seed,
        )
        if args.mode == "mlhdr":
            ptychogram = normalize_stack(ml_hdr_fusion(measurement))
        else:
            ptychogram = normalize_stack(
                single_exposure_rate(measurement, args.single_exposure_index)
            )
        np.savez_compressed(
            output / f"{args.mode}_measurement_summary.npz",
            exposure_times_s=measurement.exposure_times_s,
            dark_mean=measurement.dark_mean,
            dark_var=measurement.dark_var,
        )
        measurement_meta = {
            "exposure_times_s": measurement.exposure_times_s.tolist(),
            "camera": camera.__dict__,
            "single_exposure_index": args.single_exposure_index,
        }

    config = PtyLabConfig(
        wavelength_m=args.wavelength_nm * 1e-9,
        detector_pixel_m=args.detector_pixel_um * 1e-6,
        object_pixel_m=args.dxp_um * 1e-6,
        scan_step_px=args.scan_step_px,
        entrance_pupil_diameter_px=args.probe_diameter_px,
        iterations=args.iterations,
        position_order=args.position_order,
        initial_probe=args.initial_probe,
        position_correction=args.position_correction,
        position_correction_radius=args.position_correction_radius,
        seed=args.seed,
    )

    reconstruction, data, _ = run_mpie(ptychogram, diff.scan_shape, config)
    save_reconstruction_outputs(output, args.mode, reconstruction, data, config)
    metrics = reconstruction_metrics(reconstruction, data)

    metadata = {
        "input": args.input,
        "mode": args.mode,
        "scan_shape": diff.scan_shape,
        "frame_shape": diff.frame_shape,
        "ptylab_config": config.__dict__,
        "measurement": measurement_meta,
        "metrics": metrics,
    }
    (output / f"{args.mode}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
