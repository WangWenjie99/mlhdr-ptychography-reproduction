"""Auditable camera simulation and comparisons for Liu et al., TIM (2024).

The published equations are evaluated by ``ml_hdr_fusion`` without saturation
rejection. The explicitly named extension below is NOT the published method.
Noise magnitudes and the interpretation of total photon flux are assumptions;
see the generated experiment manifest and report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity

from .ml_hdr import CameraModel, MultiExposureMeasurement, ml_hdr_fusion


@dataclass(frozen=True)
class PaperCamera:
    photon_flux_per_s: float = 1e9
    full_well_e: float = 2.5e6
    bit_depth: int = 8
    read_noise_e: float = 5.0
    dark_current_e_per_s: float = 80.0
    dark_frames: int = 20
    # Opt-in for the simulated bit-depth sweep (paper Fig. 2 goes to 20 bit).
    # Default False keeps the recorded-camera limit of 16 bit. The ceiling of
    # 24 bit keeps every count exactly representable in float32 (ml_hdr_fusion).
    allow_extended_bit_depth: bool = False

    def __post_init__(self):
        for name in ("photon_flux_per_s", "full_well_e"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("read_noise_e", "dark_current_e_per_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        max_bits = 24 if self.allow_extended_bit_depth else 16
        if not isinstance(self.bit_depth, int) or not 1 <= self.bit_depth <= max_bits:
            raise ValueError(f"bit_depth must be an integer from 1 to {max_bits}")
        if not isinstance(self.dark_frames, int) or self.dark_frames < 2:
            raise ValueError("At least two dark frames are needed for variance")

    @property
    def max_count(self):
        return (1 << self.bit_depth) - 1

    @property
    def electrons_per_count(self):
        return self.full_well_e / self.max_count


@dataclass(frozen=True)
class PaperMeasurement:
    z: np.ndarray
    dark_frames: np.ndarray
    exposure_times_s: np.ndarray
    camera: PaperCamera
    rate_scale: float

    @property
    def dark_mean(self):
        return self.dark_frames.astype(np.float64).mean(axis=1)

    @property
    def dark_var(self):
        return self.dark_frames.astype(np.float64).var(axis=1, ddof=1)

    @property
    def count_rate_scale(self):
        return self.rate_scale / self.camera.electrons_per_count

    def paper_input(self):
        camera = CameraModel(
            bit_depth=self.camera.bit_depth,
            full_well=self.camera.full_well_e,
            read_noise=self.camera.read_noise_e,
            dark_current=self.camera.dark_current_e_per_s,
            dark_frames=self.camera.dark_frames,
        )
        return MultiExposureMeasurement(
            z=self.z, dark_mean=self.dark_mean, dark_var=self.dark_var,
            exposure_times_s=self.exposure_times_s, camera=camera,
        )


def quantize_electrons(electrons, camera: PaperCamera):
    """Linear gain, full-well clipping and round-to-nearest integer ADC."""
    value = np.rint(np.clip(electrons, 0, camera.full_well_e)
                    / camera.electrons_per_count)
    if camera.bit_depth <= 8:
        dtype = np.uint8
    elif camera.bit_depth <= 16:
        dtype = np.uint16
    else:  # only reachable with allow_extended_bit_depth=True
        dtype = np.uint32
    return value.astype(dtype)


def simulate_paper_camera(relative_intensity, exposure_times_s,
                          camera: PaperCamera, seed: int = 0):
    """Poisson photon + Poisson dark + zero-mean Gaussian read noise.

    One global scale makes the brightest frame's TOTAL photon rate 1e9/s
    (by default); all relative frame energies are preserved. Quantum efficiency
    is assumed to be one. This mapping is not specified by the paper.
    Separate random streams prevent dark-frame count from changing signal data.
    """
    intensity = np.asarray(relative_intensity, dtype=np.float64)
    times = np.asarray(exposure_times_s, dtype=np.float64)
    if (intensity.ndim != 3 or not intensity.size
            or not np.isfinite(intensity).all() or np.any(intensity < 0)):
        raise ValueError("Intensity must be finite, nonnegative and (frames,y,x)")
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or np.any(times <= 0)):
        raise ValueError("Exposure times must be a finite positive 1D array")
    brightest = float(intensity.sum(axis=(-2, -1)).max())
    if brightest <= 0:
        raise ValueError("Input intensity is all zero")
    rate_scale = camera.photon_flux_per_s / brightest
    rate = intensity * rate_scale
    seeds = np.random.SeedSequence(seed).spawn(2 * len(times))
    frames, dark = [], []
    for i, t in enumerate(times):
        rng = np.random.default_rng(seeds[2 * i])
        dark_rng = np.random.default_rng(seeds[2 * i + 1])
        signal_e = rng.poisson(rate * t).astype(np.float64)
        signal_e += rng.poisson(camera.dark_current_e_per_s * t, size=rate.shape)
        signal_e += rng.normal(0, camera.read_noise_e, size=rate.shape)
        frames.append(quantize_electrons(signal_e, camera))
        shape = (camera.dark_frames, *intensity.shape[-2:])
        dark_e = dark_rng.poisson(camera.dark_current_e_per_s * t, size=shape).astype(float)
        dark_e += dark_rng.normal(0, camera.read_noise_e, size=shape)
        dark.append(quantize_electrons(dark_e, camera))
    return PaperMeasurement(np.asarray(frames), np.asarray(dark), times,
                            camera, rate_scale)


def paper_eq14_15(measurement: PaperMeasurement):
    """Published weighting rule, including saturated pixels (no hidden mask)."""
    return ml_hdr_fusion(measurement.paper_input())


def single_rate(measurement: PaperMeasurement, index: int):
    if not 0 <= index < len(measurement.exposure_times_s):
        raise ValueError("Exposure index out of range")
    corrected = measurement.z[index].astype(float) - measurement.dark_mean[index]
    return np.maximum(corrected / measurement.exposure_times_s[index], 0).astype(np.float32)


def saturation_mask_extension(measurement: PaperMeasurement,
                              all_saturated: str = "raise"):
    """Additional control: Eq.14/15 structure with saturated observations masked.

    The preliminary mean also uses only valid exposures. Zero-valued pixels are
    retained. By default an all-saturated pixel has no recoverable value and
    raises, rather than silently substituting a clipped estimate.
    ``all_saturated="shortest"`` is an explicit opt-in (used by the simulation
    sweeps, where huge read noise can saturate a pixel in every exposure): such
    pixels keep only their shortest exposure, as in ``lrfc_hdr_fusion``.
    Not Liu et al.'s algorithm.
    """
    if all_saturated not in ("raise", "shortest"):
        raise ValueError("all_saturated must be 'raise' or 'shortest'")
    z = measurement.z.astype(np.float64)
    valid = z < measurement.camera.max_count
    hopeless = ~valid.any(axis=0)
    if np.any(hopeless):
        if all_saturated == "raise":
            raise ValueError("All exposures saturated at some pixels; shorten exposure")
        valid[int(np.argmin(measurement.exposure_times_s))] |= hopeless
    t = measurement.exposure_times_s[:, None, None, None]
    corrected = z - measurement.dark_mean[:, None]
    preliminary = np.sum(np.where(valid, corrected / t, 0), axis=0) / valid.sum(axis=0)
    denominator = t * np.maximum(preliminary, 0) + measurement.dark_var[:, None]
    w = np.where(valid, t * t / np.maximum(denominator, 1e-8), 0)
    fused = np.sum(w * corrected / t, axis=0) / np.maximum(w.sum(axis=0), 1e-30)
    return np.maximum(fused, 0).astype(np.float32)


def lrfc_hdr_fusion(measurement: PaperMeasurement):
    """Conventional linear-response (LRFC) HDR baseline, paper Eq. 16-17.

    Refs. Leong-Hoi et al. 2016 / Liu et al. 2021 as summarised by Liu et al.
    2024: the camera response is calibrated as linear, overexposed pixels are
    rejected and replaced by unsaturated pixels from shorter exposures
    rescaled by exposure time. Implemented per pixel as

        rate = (Z_i - Bbar_i) / t_i,  i = longest exposure with Z_i < Z_max,

    and, if a pixel is saturated in every exposure, i = the shortest exposure
    (its clipped value is the best available lower bound). No averaging across
    exposures is performed. Negative dark-corrected rates are clipped to zero,
    as for the other fusions in this module. Returns count/s, float32.
    """
    z = measurement.z
    t = np.asarray(measurement.exposure_times_s, dtype=np.float64)
    order = np.argsort(t, kind="stable")          # shortest ... longest
    valid = z[order] < measurement.camera.max_count
    last_valid = valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
    chosen = np.where(valid.any(axis=0), last_valid, 0)  # 0 = shortest
    index = order[chosen]
    counts = np.take_along_axis(z, index[None], axis=0)[0].astype(np.float64)
    dark = measurement.dark_mean                  # (exposures, y, x)
    dark_chosen = np.take_along_axis(
        np.broadcast_to(dark[:, None], z.shape), index[None], axis=0)[0]
    rate = (counts - dark_chosen) / t[index]
    return np.maximum(rate, 0).astype(np.float32)


def amplitude_comparison(field, reference, mask=None):
    """Baseline-relative metrics, removing only one positive amplitude scale.

    No translation, smoothing, histogram matching or per-image contrast stretch.
    The baseline is a reconstruction, not object ground truth. The fixed data
    range for PSNR/SSIM is max(reference amplitude)-min(reference amplitude).
    """
    a, b = np.abs(field).astype(float), np.abs(reference).astype(float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Fields must have identical 2D shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cannot compare nonfinite fields")
    mask = np.ones(a.shape, bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != a.shape or not mask.any():
        raise ValueError("Invalid comparison mask")
    power = float(np.sum(a[mask] ** 2))
    if power <= 0:
        raise ValueError("Cannot compare an all-zero amplitude")
    scale = float(np.sum(a[mask] * b[mask]) / power)
    aligned = a * scale
    diff = aligned[mask] - b[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    data_range = float(b.max() - b.min())
    if data_range <= 0:
        raise ValueError("Reference amplitude has no contrast")
    _, ssim_map = structural_similarity(b, aligned, data_range=data_range, full=True)
    ssim_mask = mask.copy()
    ssim_mask[:3] = ssim_mask[-3:] = False
    ssim_mask[:, :3] = ssim_mask[:, -3:] = False
    if not ssim_mask.any():
        raise ValueError("Comparison region is too small for SSIM")
    return {
        "amplitude_scale": scale,
        "baseline_nrmse": float(np.linalg.norm(diff) / np.linalg.norm(b[mask])),
        "baseline_psnr_db": None if rmse == 0 else float(20 * np.log10(data_range / rmse)),
        "baseline_ssim": float(ssim_map[ssim_mask].mean()),
    }, aligned


def diffraction_comparison(rate, measurement: PaperMeasurement, clean):
    """Error against the known pre-camera diffraction stack, on a common scale."""
    estimate = np.asarray(rate, dtype=np.float64) / measurement.count_rate_scale
    truth = np.asarray(clean, dtype=np.float64)
    error = estimate - truth
    yy, xx = np.indices(truth.shape[-2:])
    radius = np.sqrt((yy - truth.shape[-2] // 2) ** 2
                     + (xx - truth.shape[-1] // 2) ** 2)
    high = radius >= min(truth.shape[-2:]) / 4
    floor = float(truth.max()) * 1e-8
    return {
        "diffraction_nrmse": float(np.linalg.norm(error.ravel()) / np.linalg.norm(truth.ravel())),
        "high_q_diffraction_nrmse": float(np.linalg.norm(error[:, high]) / np.linalg.norm(truth[:, high])),
        "diffraction_log10_rmse": float(np.sqrt(np.mean((np.log10(estimate + floor)
                                                       - np.log10(truth + floor)) ** 2))),
    }
