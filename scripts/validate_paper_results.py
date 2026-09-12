#!/usr/bin/env python3
"""Audit saved settings, measurements, fusion inputs and reported comparisons."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mlhdr_ptycho.data import load_diff_npy, normalize_stack
from mlhdr_ptycho.paper_reproduction import (
    PaperCamera, amplitude_comparison, paper_eq14_15, saturation_mask_extension,
    simulate_paper_camera, single_rate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    conf = manifest["experiment"]
    checks = []

    def check(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    check(manifest["status"] == "complete", "Experiment completed")
    original = ROOT / conf["baseline_archive"]
    check(digest(original) == manifest["baseline_sha256"], "Original baseline archive unchanged")
    check(digest(output / "baseline_reference" / original.name) == manifest["baseline_sha256"],
          "Baseline reference is an exact archive copy")
    check(digest(ROOT / conf["input"]) == manifest["input_sha256"], "Input diffraction data unchanged")
    for path, checksum in manifest["source_sha256"].items():
        check(digest(ROOT / path) == checksum, f"Source matches experiment: {path}")
    reference = np.load(output / "baseline_reference/reference_roi.npz")["object"]
    roi = tuple(slice(a, b) for a, b in manifest["roi"])
    rows = list(csv.DictReader((output / "comparison_metrics.csv").open()))
    check(len(rows) == len(manifest["selected_profiles"]) * len(manifest["selected_seeds"]) * 9,
          "Every profile, seed and method has a metric row")
    check(len({(r['profile'],r['seed'],r['case']) for r in rows}) == len(rows), "No duplicate metric rows")
    clean = normalize_stack(load_diff_npy(ROOT / conf["input"], conf["scan_crop"]).stack)
    exposure_ms = conf["exposure_times_ms"]
    times = np.asarray(exposure_ms) / 1000
    for profile in manifest["selected_profiles"]:
        camera = PaperCamera(**conf["camera"], **conf["noise_profiles"][profile])
        for seed in manifest["selected_seeds"]:
            seed_dir = output / profile / f"seed_{seed}"
            measurement = simulate_paper_camera(clean, times, camera, seed)
            with np.load(seed_dir / "measurement.npz") as saved:
                check(np.array_equal(measurement.z, saved["z"]), f"Reproducible ADC measurements: {profile}/{seed}")
                check(np.array_equal(measurement.dark_frames, saved["dark_frames"]),
                      f"Reproducible dark frames: {profile}/{seed}")
            rates = {f"single_{t:g}ms": single_rate(measurement,i) for i,t in enumerate(exposure_ms)}
            rates["paper_ml_hdr"] = paper_eq14_15(measurement)
            rates["saturation_mask_extension"] = saturation_mask_extension(measurement)
            for case, expected_rate in rates.items():
                case_dir = seed_dir / case
                metadata = json.loads((case_dir / "metadata.json").read_text())
                check(metadata["ptylab_config"] == manifest["ptylab_config"],
                      f"Identical baseline solver configuration: {profile}/{seed}/{case}")
                check(metadata["camera"] == asdict(camera), f"Camera settings recorded: {profile}/{seed}/{case}")
                with np.load(case_dir / "input_rate.npz") as saved:
                    check(np.array_equal(expected_rate, saved["count_rate"]),
                          f"Stored input matches its labeled method: {profile}/{seed}/{case}")
                if metadata["metrics"]["status"] != "ok":
                    continue
                with np.load(case_dir / "arrays.npz") as saved:
                    check(saved["error"].size == 80, f"Exactly 80 iterations: {profile}/{seed}/{case}")
                    check(all(np.isfinite(saved[k]).all() for k in ["object","probe","error"]),
                          f"Finite reconstruction: {profile}/{seed}/{case}")
                    recomputed, _ = amplitude_comparison(saved["object"][roi], reference)
                row = next(r for r in rows if r["profile"] == profile and int(r["seed"]) == seed and r["case"] == case)
                # Reconstruction arrays are complex64. Computing magnitudes
                # from different memory layouts can differ by one float32 ULP.
                check(all(np.isclose(float(row[k]), recomputed[k], rtol=1e-6, atol=1e-7)
                          for k in ["baseline_psnr_db","baseline_ssim","baseline_nrmse"]),
                      f"Reported metrics match saved arrays: {profile}/{seed}/{case}")
        for name in ["comparison_amplitude.png", "difference_to_baseline.png", "extension_comparison.png",
                     "metrics_vs_exposure.png", "diffraction_comparison.png"]:
            check((output/profile/name).is_file(), f"Comparison figure exists: {profile}/{name}")
    distribution = importlib.metadata.distribution("PtyLab")
    import PtyLab
    package = Path(PtyLab.__file__).parent
    core = ["Engines/mPIE.py", "Engines/BaseEngine.py", "Operators/Operators.py",
            "Params/Params.py", "Reconstruction/Reconstruction.py", "utils/initializationFunctions.py"]
    validation = {
        "status": "passed", "checks_passed": len(checks), "checks": checks,
        "ptylab_direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
        "ptylab_core_sha256": {p: digest(package/p) for p in core},
        "validator_sha256": digest(Path(__file__)),
        "scope": "Exact camera draws and method inputs; saved metrics verified within complex64 precision; fixed solver settings. Solver output need not be bitwise repeatable.",
        "metric_comparison_tolerance": {"rtol": 1e-6, "atol": 1e-7,
            "reason": "Magnitude calculations originate in stored complex64 arrays"},
    }
    (output/"validation.json").write_text(json.dumps(validation,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"PASS: {len(checks)} checks; {len(rows)} reconstructions audited")


if __name__ == "__main__":
    main()
