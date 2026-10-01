#!/usr/bin/env python3
"""Paper-style simulation and analysis for Liu et al., IEEE TIM 73:4502711 (2024).

DATA CONTRACT
=============

Produced by ``scripts/run_paper_style_simulation.py`` into
``outputs/paper_style/<profile>/`` (or ``--output``). The plotting code
consumes ONLY these files.

**All simulation parameters are OUR choices.** Liu et al. do not report the
simulation's exposure times, read noise, dark current, wavelength or pixel
sizes, and our geometry, object sampling, probe and scan step differ from the
paper's simulation. We reuse the experimental wavelength (632.8 nm), distance
(13.9 mm) and the seven experimental exposure times (0.5-500 ms), and pick a
virtual detector pitch so the object pixel equals the experiment's
~0.571 um. Every value is recorded in ``meta.json``.

Conventions
-----------
* Method keys (exact): ``single`` = one exposure, the "auto-exposure" (longest
  of the 7 exposures whose NOISELESS peak pixel stays below full well; fixed
  per object, independent of bit depth / read noise); ``lrfc`` = conventional
  LRFC-HDR (paper Eq. 16-17: per pixel the longest unsaturated exposure,
  dark-subtracted and divided by t; shortest if saturated everywhere);
  ``ml_eq14_15`` = published ML-HDR Eq. 14-15 exactly, saturated pixels
  INCLUDED (no pixel rejection, as stated in the paper); ``ml_masked`` =
  Eq. 14-15 weights with saturated observations excluded (OUR extension, NOT
  the published algorithm). The bit depth is always a separate field
  (``bits`` in CSVs, ``run_bits`` in NPZs).
* NPZ per-run arrays are stacked on axis 0 (length R) in the order of
  ``run_method`` (str), ``run_bits`` (int64) and ``run_label`` (str,
  ``"<method>@<bits>"``).
* Objects are complex transmissions; ``truth`` is amplitude-only (phase 0).
  ``aligned_roi`` = c * (reconstruction Fourier-shifted by ``align_shift_px``
  and with a linear phase ramp removed), cropped to ``roi``; c is the complex
  scalar minimising ||c O - O_true||. All metrics use ``abs(aligned_roi)`` vs
  ``abs(truth_roi)`` (same scale, directly comparable, same display range).
* ``roi`` = int64 [y0, y1, x0, x1], half-open, square, object-pixel indices
  (well-scanned region = union of probe footprints minus a margin).
  ``dx_um`` = object pixel size in um.
* Metrics (vs ground truth, on the ROI): ``psnr_db`` (data_range =
  ptp(|truth|)), ``ssim`` (skimage, same data_range), ``nrmse`` =
  ||a - a_true|| / ||a_true||.
* FRC is computed between the aligned reconstruction amplitude and the
  GROUND-TRUTH amplitude (not between two independent reconstructions),
  Hann-apodised, 1-px rings, frequency normalised to Nyquist (1.0 =
  1/(2 dx)); van Heel half-bit threshold; cutoff = first crossing below the
  threshold at f >= 0.05 (linear interpolation; 1.0 if never crossed);
  ``resolution_um = dx_um / cutoff`` (half-period). A crossing at the first
  ring with f >= 0.05 (5/86.5 = 0.058 for the 173-px ROI; interpolation may
  place it between rings 4 and 5) gives cutoff <= 0.058, i.e.
  resolution_um >= 9.88: treat such values as "no resolution" (a floor, not
  a measurement).
* Diffraction error (mPIE): sum_j sum_q (|Psi_j| - sqrt(I_j))^2 / sum I,
  evaluated with the pre-update exit waves of each iteration.
* NaN (CSV: empty or ``nan``; JSON: null) = not available (iteration 0 of a
  history, failed/diverged run).

bit_sweep.csv  (Exp. A, paper Fig. 2: cameraman, bits 2..20)
------------------------------------------------------------
One row per (bits, method, seed); read noise = base (5 e-); its ``snr_db``
column is only the reference value 20 log10(2.5e6 / 5) = 114 dB.
noise_sweep.csv  (Exp. B, paper Fig. 3: cameraman)
--------------------------------------------------
One row per (snr_db, method, seed). "Noise magnitude" is OUR assumption:
``snr_db = 20 log10(full_well_e / read_noise_e)``; ``single`` at 16 bit,
``lrfc``/``ml_eq14_15``/``ml_masked`` at 8 bit.

Columns of both CSVs:
  experiment (bit_sweep|noise_sweep), object, method, bits, seed,
  snr_db, read_noise_e, exposure_s (single only, else empty),
  psnr_db, ssim, nrmse, frc_cutoff_nyquist, frc_resolution_um,
  frc_reached_nyquist (0/1), final_diffraction_error,
  diffraction_nrmse, high_q_diffraction_nrmse, diffraction_log10_rmse
  (fused rate vs noiseless rate on the whole stack; high-q = radius >= 16 px),
  saturated_fraction_<t>ms for t in 0p5,1,5,10,50,100,500 (fraction of
  pixels at the ADC maximum per exposure, whole stack),
  all_saturated_pixels (pixels saturated in all 7 exposures), amplitude_scale
  (|c|), align_shift_y_px, align_shift_x_px, iterations, recon_seconds (mPIE
  wall-clock, metric evaluation excluded), task_seconds (incl. simulation,
  fusion, metrics), diverged (0/1), status (ok|failed|diverged), error.

cameraman_8bit.npz  (Exp. A mosaic, first seed; runs single@8, lrfc@8, ml_eq14_15@8, ml_masked@8)
Keys:
  truth            complex64 (H, W)     ground-truth object
  truth_roi        complex64 (h, h)     truth[roi]
  roi              int64 (4,)
  dx_um            float64 ()
  positions        int64 (J, 2)         top-left (row, col) of each 64x64 window
  probe_true       complex64 (64, 64)
  run_method, run_label   str (R,);  run_bits int64 (R,)
  object           complex64 (R, H, W)  raw mPIE object (unaligned)
  probe            complex64 (R, 64, 64)
  aligned_roi      complex64 (R, h, h)  see conventions
  align_shift_px   float64 (R, 2)       (dy, dx) applied to the reconstruction
  align_scalar     complex128 (R,)
  psnr_db, ssim, nrmse                 float64 (R,)
  history_iteration          int64 (K,)   iterations at which history is sampled (0, k, 2k, ..., last)
  history_wallclock_s        float64 (R, K) cumulative mPIE time
  history_object_nrmse       float64 (R, K) object-amplitude NRMSE vs truth (aligned)
  history_object_psnr_db     float64 (R, K)
  history_diffraction_error  float64 (R, K) (NaN at iteration 0)
  diffraction_error          float64 (R, iterations) every iteration
  frc_frequency_nyquist      float64 (F,)
  frc                        float64 (R, F)
  frc_threshold              float64 (F,)  half-bit curve (same ROI for all runs)
  frc_cutoff_nyquist, frc_resolution_um  float64 (R,)

meta.json
  profile and its settings, seeds, data/reconstruction seed rules, the full
  SimulationConfig and MPIEConfig, per-object facts (object shape, ROI,
  number of positions, auto-exposure, photon rate scale, noiseless peak),
  method and noise definitions, assumptions (explicitly: these simulation
  parameters are OUR choices), package versions, platform, workers, runtime,
  failures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import datetime as _dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mlhdr_ptycho.mpie import MPIEConfig, initial_probe, reconstruct
from mlhdr_ptycho.paper_reproduction import diffraction_comparison
from mlhdr_ptycho.resolution import align_to_truth, amplitude_metrics, fourier_ring_correlation
from mlhdr_ptycho.simulation import METHODS, SimulationConfig, build_scene, fuse, saturation_fractions

PROFILES = {
    "quick": {"iterations": 20, "history_every": 5, "bits": [2, 8, 16, 20],
              "snr_db": [6, 30, 54]},
    "full": {"iterations": 250, "history_every": 5, "bits": list(range(2, 21, 2)),
             "snr_db": list(range(6, 55, 6))},
}
NOISE_SWEEP_BITS = {"single": 16, "lrfc": 8, "ml_eq14_15": 8, "ml_masked": 8}
RECON_SEED_OFFSET = 10_000
EXPERIMENTS = ("A", "B")
CSV_FIELDS = [
    "experiment", "object", "method", "bits", "seed", "snr_db", "read_noise_e", "exposure_s",
    "psnr_db", "ssim", "nrmse", "frc_cutoff_nyquist", "frc_resolution_um",
    "frc_reached_nyquist", "final_diffraction_error", "diffraction_nrmse",
    "high_q_diffraction_nrmse", "diffraction_log10_rmse",
]


def _exposure_tag(t):
    ms = t * 1e3
    return (f"{ms:g}".replace(".", "p")) + "ms"


def _sat_fields(times):
    return [f"saturated_fraction_{_exposure_tag(t)}" for t in times]


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------
_SCENES = {}


def get_scene(kind):
    if kind not in _SCENES:
        _SCENES[kind] = build_scene(kind, SimulationConfig())
    return _SCENES[kind]


def opt_out_of_power_throttling():
    """Ask Windows not to EcoQoS-throttle THIS process (per-process QoS hint).

    Windows 11 throttles background console processes after a few seconds
    (measured here: ~6.5x slower). This only affects the calling process and
    ends with it; no system setting is changed. No-op on other platforms.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        import ctypes.wintypes as wt

        class _State(ctypes.Structure):
            _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        kernel32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        kernel32.SetProcessInformation.restype = wt.BOOL
        state = _State(1, 1, 0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED, off
        return bool(kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False


def worker_init(opt_out):
    if opt_out:
        opt_out_of_power_throttling()


def read_noise_for_snr(snr_db, full_well_e):
    return full_well_e / 10 ** (snr_db / 20.0)


def run_task(task):
    start = time.perf_counter()
    try:
        out = _run_task(task)
    except Exception as exc:  # keep the sweep alive; record the failure
        out = {"row": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, "arrays": None}
    out["row"]["task_seconds"] = time.perf_counter() - start
    out["key"] = task["key"]
    return out


def _run_task(task):
    scene = get_scene(task["object"])
    cfg = scene.config
    y0, y1, x0, x1 = scene.roi
    truth_roi = scene.truth[y0:y1, x0:x1]
    measurement = scene.simulate(task["bits"], task["read_noise_e"], seed=task["seed"])
    rate, info = fuse(measurement, task["method"], scene.auto_exposure_index())
    intensities = rate.astype(np.float64) / measurement.count_rate_scale
    dmetrics = diffraction_comparison(rate, measurement, scene.clean)
    sat = saturation_fractions(measurement)
    del measurement, rate

    def monitor(obj, _probe):
        if not np.isfinite(obj).all():
            return {"object_nrmse": float("nan"), "object_psnr_db": float("nan")}
        al = align_to_truth(obj, scene.truth, scene.roi)
        m = amplitude_metrics(al.aligned_roi, truth_roi)
        return {"object_nrmse": m["nrmse"], "object_psnr_db": m["psnr_db"]}

    n = cfg.detector_pixels
    probe0 = initial_probe(n, cfg.probe_diameter_px,
                           float(intensities.sum(axis=(-2, -1)).max()))
    mcfg = MPIEConfig(iterations=task["iterations"], history_every=task["history_every"],
                      seed=RECON_SEED_OFFSET + task["seed"])
    result = reconstruct(intensities, scene.positions, scene.truth.shape, probe0, mcfg,
                         monitor=monitor)
    row = {
        "exposure_s": info.get("exposure_s", ""),
        "final_diffraction_error": float(result.diffraction_error[-1]) if result.diffraction_error.size else float("nan"),
        **{k: float(v) for k, v in dmetrics.items()},
        **dict(zip(_sat_fields(cfg.exposure_times_s), map(float, sat))),
        "all_saturated_pixels": info["all_saturated_pixels"],
        "iterations": len(result.diffraction_error),
        "recon_seconds": result.seconds,
        "diverged": int(result.diverged),
    }
    finite = np.isfinite(result.object).all() and np.isfinite(result.probe).all()
    if result.diverged or not finite:
        row.update(status="diverged", error="non-finite reconstruction")
        return {"row": row, "arrays": None}
    al = align_to_truth(result.object, scene.truth, scene.roi)
    metrics = amplitude_metrics(al.aligned_roi, truth_roi)
    frc = fourier_ring_correlation(al.amplitude, np.abs(truth_roi), scene.dx_um)
    row.update(metrics)
    row.update(frc_cutoff_nyquist=frc["cutoff_nyquist"], frc_resolution_um=frc["resolution_um"],
               frc_reached_nyquist=int(frc["reached_nyquist"]), amplitude_scale=abs(al.scalar),
               align_shift_y_px=al.shift_px[0], align_shift_x_px=al.shift_px[1],
               status="ok", error="")
    arrays = None
    if task["save"]:
        arrays = {
            "object": result.object.astype(np.complex64),
            "probe": result.probe.astype(np.complex64),
            "aligned_roi": al.aligned_roi.astype(np.complex64),
            "align_shift_px": np.asarray(al.shift_px, float),
            "align_scalar": complex(al.scalar),
            "history": {k: np.asarray(v, float) for k, v in result.history.items()},
            "diffraction_error": result.diffraction_error,
            "frc": frc,
            "metrics": metrics,
        }
    return {"row": row, "arrays": arrays}


# --------------------------------------------------------------------------
# Main side
# --------------------------------------------------------------------------
def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(_clean_json(value), indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k, "")) for k in fields})


def build_tasks(profile, seeds, experiments, base_read_noise):
    """Unique reconstruction tasks and, per experiment, the keys it needs."""
    tasks, plan = {}, {e: [] for e in experiments}

    def add(experiment, obj, method, bits, read_noise, seed, save, extra):
        key = (obj, method, int(bits), float(read_noise), int(seed))
        if key not in tasks:
            tasks[key] = dict(key=key, object=obj, method=method, bits=int(bits),
                              read_noise_e=float(read_noise), seed=int(seed), save=False,
                              iterations=profile["iterations"],
                              history_every=profile["history_every"])
        tasks[key]["save"] |= bool(save)
        plan[experiment].append((key, extra))

    first = seeds[0]
    fw = SimulationConfig().full_well_e
    if "A" in experiments:
        for seed in seeds:
            for bits in profile["bits"]:
                for method in METHODS:
                    add("A", "cameraman", method, bits, base_read_noise, seed,
                        bits == 8 and seed == first, {"experiment": "bit_sweep"})
    if "B" in experiments:
        for seed in seeds:
            for snr in profile["snr_db"]:
                sigma = read_noise_for_snr(snr, fw)
                for method in METHODS:
                    add("B", "cameraman", method, NOISE_SWEEP_BITS[method], sigma, seed, False,
                        {"experiment": "noise_sweep", "snr_db": float(snr)})
    return tasks, plan


def full_row(task, result_row, extra, full_well):
    row = {k: float("nan") for k in CSV_FIELDS}
    row.update(experiment=extra.get("experiment", ""), object=task["object"],
               method=task["method"], bits=task["bits"], seed=task["seed"],
               read_noise_e=task["read_noise_e"],
               snr_db=extra.get("snr_db", 20 * math.log10(full_well / task["read_noise_e"])))
    row.update(result_row)
    return row


def stack_runs(scene, runs, results):
    """Arrays of cameraman_8bit.npz, per-run arrays stacked in the order of ``runs``."""
    y0, y1, x0, x1 = scene.roi
    ok = [results[k]["arrays"] for k in runs]
    shape_obj = scene.truth.shape
    n = scene.probe.shape[0]
    h = y1 - y0
    r = len(runs)
    labels = [f"{k[1]}@{k[2]}" for k in runs]
    out = {
        "truth": scene.truth.astype(np.complex64),
        "truth_roi": scene.truth[y0:y1, x0:x1].astype(np.complex64),
        "roi": np.asarray(scene.roi, np.int64),
        "dx_um": np.float64(scene.dx_um),
        "positions": scene.positions.astype(np.int64),
        "probe_true": scene.probe.astype(np.complex64),
        "run_method": np.array([k[1] for k in runs]),
        "run_bits": np.array([k[2] for k in runs], np.int64),
        "run_label": np.array(labels),
        "object": np.full((r, *shape_obj), np.nan, np.complex64),
        "probe": np.full((r, n, n), np.nan, np.complex64),
        "aligned_roi": np.full((r, h, h), np.nan, np.complex64),
        "align_shift_px": np.full((r, 2), np.nan),
        "align_scalar": np.full(r, np.nan, np.complex128),
        "psnr_db": np.full(r, np.nan), "ssim": np.full(r, np.nan), "nrmse": np.full(r, np.nan),
        "frc_cutoff_nyquist": np.full(r, np.nan), "frc_resolution_um": np.full(r, np.nan),
    }
    hist_iter = next((a["history"]["iteration"] for a in ok if a is not None), np.array([0.0]))
    k_len = len(hist_iter)
    its = max((len(a["diffraction_error"]) for a in ok if a is not None), default=0)
    out["history_iteration"] = hist_iter.astype(np.int64)
    for name in ("wallclock_s", "object_nrmse", "object_psnr_db", "diffraction_error"):
        out[f"history_{name}"] = np.full((r, k_len), np.nan)
    out["diffraction_error"] = np.full((r, its), np.nan)
    frc_len = h // 2 + 1
    out["frc_frequency_nyquist"] = np.arange(frc_len) / (h / 2.0)
    out["frc"] = np.full((r, frc_len), np.nan)
    out["frc_threshold"] = np.full(frc_len, np.nan)
    for i, a in enumerate(ok):
        if a is None:
            continue
        out["object"][i] = a["object"]
        out["probe"][i] = a["probe"]
        out["aligned_roi"][i] = a["aligned_roi"]
        out["align_shift_px"][i] = a["align_shift_px"]
        out["align_scalar"][i] = a["align_scalar"]
        for m in ("psnr_db", "ssim", "nrmse"):
            out[m][i] = a["metrics"][m]
        out["frc_cutoff_nyquist"][i] = a["frc"]["cutoff_nyquist"]
        out["frc_resolution_um"][i] = a["frc"]["resolution_um"]
        out["frc"][i] = a["frc"]["frc"]
        out["frc_threshold"] = a["frc"]["threshold"]
        for name in ("wallclock_s", "object_nrmse", "object_psnr_db", "diffraction_error"):
            values = a["history"].get(name, np.array([]))
            out[f"history_{name}"][i, :min(k_len, len(values))] = values[:k_len]
        de = a["diffraction_error"]
        out["diffraction_error"][i, :len(de)] = de
    return out


def package_versions():
    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "scikit-image", "matplotlib", "h5py", "tqdm"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def object_facts(kind):
    s = get_scene(kind)
    auto = s.auto_exposure_index()
    return {
        "object_shape": list(s.truth.shape), "roi": list(s.roi), "roi_side_px": s.roi[1] - s.roi[0],
        "positions": int(len(s.positions)),
        "auto_exposure_index": auto, "auto_exposure_s": s.config.exposure_times_s[auto],
        "photons_per_s_per_relative_unit": s.rate_scale,
        "noiseless_peak_pixel_e_per_s": float(s.clean.max() * s.rate_scale),
        "noiseless_peak_fraction_of_frame": float(s.clean.max() / s.clean.sum(axis=(1, 2)).max()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="seeds for sweeps A/B (the first seed's 8-bit runs are saved to cameraman_8bit.npz)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default min(8, cpu_count)); ~0.3 GB each")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--iterations", type=int, default=None, help="override profile iterations")
    parser.add_argument("--allow-power-throttling", action="store_true",
                        help="do not opt this run's processes out of Windows EcoQoS throttling")
    args = parser.parse_args(argv)

    profile = dict(PROFILES[args.profile])
    if args.iterations:
        profile["iterations"] = args.iterations
    output = (args.output or ROOT / "outputs" / "paper_style" / args.profile).resolve()
    output.mkdir(parents=True, exist_ok=True)
    workers = args.workers or min(8, os.cpu_count() or 1)
    opt_out = not args.allow_power_throttling
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if opt_out:
        opt_out_of_power_throttling()

    cfg = SimulationConfig()
    experiments = [e for e in EXPERIMENTS if e in args.experiments]
    seeds = list(dict.fromkeys(args.seeds))
    t_start = time.perf_counter()
    tasks, plan = build_tasks(profile, seeds, experiments, cfg.read_noise_e)
    print(f"[paper_style] profile={args.profile} experiments={experiments} seeds={seeds} "
          f"tasks={len(tasks)} workers={workers} iterations={profile['iterations']} -> {output}",
          flush=True)

    results = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                                 initargs=(opt_out,)) as pool:
            futures = {pool.submit(run_task, t): k for k, t in tasks.items()}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results[res["key"]] = res
                k, row = res["key"], res["row"]
                print(f"  [{i}/{len(tasks)}] {k[0]} {k[1]}@{k[2]} sigma={k[3]:.3g}e seed={k[4]}: "
                      f"{row.get('status')} nrmse={row.get('nrmse', float('nan')):.4f} "
                      f"ssim={row.get('ssim', float('nan')):.3f} "
                      f"({row.get('task_seconds', 0):.1f} s) {row.get('error', '')}", flush=True)

    fw = cfg.full_well_e
    sat_cols = _sat_fields(cfg.exposure_times_s)
    fields = CSV_FIELDS + sat_cols + [
        "all_saturated_pixels", "amplitude_scale", "align_shift_y_px", "align_shift_x_px",
        "iterations", "recon_seconds", "task_seconds", "diverged", "status", "error"]
    exp_seconds = {}
    failures = []
    for res in results.values():
        if res["row"].get("status") != "ok":
            failures.append({"task": list(res["key"]), "status": res["row"].get("status"),
                             "error": res["row"].get("error")})

    for exp, name in (("A", "bit_sweep.csv"), ("B", "noise_sweep.csv")):
        if exp not in plan:
            continue
        rows = [full_row(tasks[k], results[k]["row"], extra, fw) for k, extra in plan[exp]]
        write_csv(output / name, rows, fields)
        exp_seconds[exp] = sum(results[k]["row"].get("task_seconds", 0) for k, _ in plan[exp])

    if "A" in plan:
        runs = [("cameraman", m, 8, float(cfg.read_noise_e), seeds[0]) for m in METHODS]
        if all(k in results for k in runs):
            arrays = stack_runs(get_scene("cameraman"), runs, results)
            np.savez_compressed(output / "cameraman_8bit.npz", **arrays)

    total = time.perf_counter() - t_start
    mcfg = MPIEConfig(iterations=profile["iterations"], history_every=profile["history_every"])
    meta = {
        "title": "Paper-style ML-HDR ptychography simulation (Liu et al., IEEE TIM 73:4502711, 2024)",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "profile": args.profile, "profile_settings": profile, "experiments": experiments,
        "seeds": seeds,
        "seed_rules": {
            "data": "simulate_paper_camera(seed=s): same electron-level noise draws for every "
                    "bit depth and method (common random numbers); read noise scales the same "
                    "standard-normal draws",
            "reconstruction": f"mPIE position order rng seed = {RECON_SEED_OFFSET} + s, shared by all methods",
            "scan": f"SimulationConfig.scan_seed = {cfg.scan_seed}",
            "saved_arrays": "cameraman_8bit.npz uses seeds[0]",
        },
        "simulation_config": cfg.to_dict(),
        "mpie_config": {**mcfg.to_dict(), "seed": f"{RECON_SEED_OFFSET} + s"},
        "objects": {"cameraman": object_facts("cameraman")},
        "methods": {
            "single": "single exposure at the auto-exposure time (longest of the 7 exposures whose "
                      "noiseless peak pixel, signal + dark, is below full well); rate=(Z-Bbar)/t, clipped >= 0",
            "lrfc": "conventional LRFC-HDR (Eq.16-17): per pixel the longest unsaturated exposure "
                    "(Z < Zmax), (Z-Bbar)/t; shortest exposure if saturated in all; clipped >= 0",
            "ml_eq14_15": "published ML-HDR Eq.14-15, saturated pixels included (no rejection); "
                          "denominator floored at 1e-8 count^2 (only matters for 0/0)",
            "ml_masked": "Eq.14-15 weights with saturated observations excluded; all-saturated "
                         "pixels use the shortest exposure; OUR extension, not the published algorithm",
        },
        "noise_sweep": {
            "definition": "snr_db = 20*log10(full_well_e / read_noise_e); OUR assumption for the "
                          "paper's 'noise magnitude' (6-54 dB)",
            "bits": NOISE_SWEEP_BITS,
            "read_noise_e": {str(s): read_noise_for_snr(s, fw) for s in profile["snr_db"]},
        },
        "bit_sweep": {"bits": profile["bits"], "read_noise_e": cfg.read_noise_e},
        "assumptions": [
            "These simulation parameters are OUR choices: Liu et al. do not report the "
            "simulation's exposure times, read noise, dark current, wavelength or pixel sizes, "
            "and our geometry, object sampling, probe and scan step differ from the paper's "
            "simulation.",
            "Exposure times = the paper's 7 experimental exposures (0.5-500 ms).",
            "Geometry: lambda=632.8 nm, z=13.9 mm, 64x64 virtual detector pixels of 240.64 um "
            "=> object pixel lambda z/(N pitch) = 0.5711 um (experiment ~0.571 um).",
            "Photon flux 1e9 photons/s = TOTAL rate of the brightest diffraction frame "
            "(simulate_paper_camera), quantum efficiency 1.",
            "Camera: full well 2.5e6 e-, dark current 80 e-/s, 20 dark frames, round-to-nearest ADC, "
            "base read noise 5 e- (repo 'low_noise' profile).",
            "Fraunhofer (FFT) propagation, integer-pixel scan positions (10% jitter rounded).",
            "mPIE is our pure-numpy implementation (PtyLab not installed); alpha_probe=1.0 "
            "instead of PtyLab's 0.1, which stagnated in our tests; probe centre-of-mass "
            "stabilisation (object shifted identically) prevents the object/probe translation "
            "drift (up to >8 px without it).",
            "Metrics and FRC are against the known ground truth.",
        ],
        "versions": package_versions(),
        "platform": {"system": platform.platform(), "processor": platform.processor(),
                     "cpu_count": os.cpu_count()},
        "workers": workers,
        "power_throttling_opt_out": opt_out,
        "runtime_s": {"total_wallclock": total, "task_seconds_sum_per_experiment": exp_seconds},
        "failures": failures,
    }
    write_json(output / "meta.json", meta)
    header = (f"<!-- Generated by scripts/run_paper_style_simulation.py (profile={args.profile}, "
              f"{meta['created']}). -->\n\n")
    (output / "DATA_CONTRACT.md").write_text(header + "```text\n" + __doc__.strip() + "\n```\n",
                                             encoding="utf-8")
    print(f"[paper_style] done in {total:.1f} s; failures={len(failures)}; outputs in {output}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
