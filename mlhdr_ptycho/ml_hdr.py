from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Simple mixed-noise camera model for ML-HDR simulation.

    Units are electrons unless explicitly noted.
    """

    bit_depth: int = 8
    full_well: float = 2.5e6
    read_noise: float = 5.0
    dark_current: float = 80.0
    saturation_fraction_at_longest: float = 1.25
    dark_frames: int = 20


@dataclass(frozen=True)
class MultiExposureMeasurement:
    z: np.ndarray
    dark_mean: np.ndarray
    dark_var: np.ndarray
    exposure_times_s: np.ndarray
    camera: CameraModel


def _counts_from_electrons(electrons: np.ndarray, camera: CameraModel) -> np.ndarray:
    max_count = (1 << camera.bit_depth) - 1
    clipped = np.clip(electrons, 0, camera.full_well)
    counts = np.rint(clipped / camera.full_well * max_count)
    dtype = np.uint16 if camera.bit_depth > 8 else np.uint8
    return counts.astype(dtype)


def simulate_multiexposure(
    irradiance: np.ndarray,
    exposure_times_s: np.ndarray,
    camera: CameraModel = CameraModel(),
    seed: int = 0,
) -> MultiExposureMeasurement:
    """Simulate noisy multi-exposure diffraction frames and dark frames.

    `irradiance` is a nonnegative stack with shape `(frames, y, x)`. It is
    interpreted as relative photon rate. The global maximum is scaled so the
    longest exposure slightly saturates the detector, creating an HDR problem.
    """

    irradiance = np.asarray(irradiance, dtype=np.float32)
    if irradiance.ndim != 3:
        raise ValueError(f"Expected irradiance shape (frames,y,x), got {irradiance.shape}")
    if np.any(irradiance < 0):
        raise ValueError("Irradiance must be nonnegative")

    exposure_times_s = np.asarray(exposure_times_s, dtype=np.float32)
    if exposure_times_s.ndim != 1 or exposure_times_s.size == 0:
        raise ValueError("exposure_times_s must be a nonempty 1D array")

    rng = np.random.default_rng(seed)
    max_irradiance = max(float(np.max(irradiance)), 1e-12)
    longest = float(np.max(exposure_times_s))
    rate_scale = (
        camera.saturation_fraction_at_longest
        * camera.full_well
        / (max_irradiance * longest)
    )
    photon_rate = irradiance * rate_scale

    z_frames = []
    dark_means = []
    dark_vars = []
    det_shape = irradiance.shape[-2:]

    for exposure in exposure_times_s:
        signal_e = photon_rate * exposure
        noisy_signal = rng.poisson(signal_e).astype(np.float32)
        dark_e = rng.poisson(camera.dark_current * exposure, size=signal_e.shape)
        read_e = rng.normal(0.0, camera.read_noise, size=signal_e.shape)
        z_frames.append(_counts_from_electrons(noisy_signal + dark_e + read_e, camera))

        dark_stack = []
        for _ in range(camera.dark_frames):
            dark_signal = rng.poisson(camera.dark_current * exposure, size=det_shape)
            dark_read = rng.normal(0.0, camera.read_noise, size=det_shape)
            dark_stack.append(_counts_from_electrons(dark_signal + dark_read, camera))
        dark_stack = np.asarray(dark_stack, dtype=np.float32)
        dark_means.append(dark_stack.mean(axis=0))
        dark_vars.append(dark_stack.var(axis=0, ddof=1))

    return MultiExposureMeasurement(
        z=np.asarray(z_frames),
        dark_mean=np.asarray(dark_means, dtype=np.float32),
        dark_var=np.asarray(dark_vars, dtype=np.float32),
        exposure_times_s=exposure_times_s,
        camera=camera,
    )


def ml_hdr_fusion(measurement: MultiExposureMeasurement, eps: float = 1e-8) -> np.ndarray:
    """Fuse multi-exposure frames using the paper's ML-HDR weighting rule.

    This implements Eq. (14)-(15) with gain absorbed into the digital counts.
    The result is a count-rate proportional HDR irradiance stack.

    0/0 guard: Eq. (14) is undefined when the preliminary rate is 0 AND the
    dark variance is 0 (e.g. low-bit cameras whose dark frames all quantise to
    0). The denominator is floored at ``eps`` (count^2), so such pixels get
    weights proportional to t_i^2 and a finite result. For the paper's seven
    exposures (0.5-500 ms) and 20 dark frames, a nonzero denominator is
    >= ~5e-6 count^2 >> eps (up to float32 rounding), so the floor does not
    alter a well-defined weight.
    """

    z = measurement.z.astype(np.float32)
    dark_mean = measurement.dark_mean.astype(np.float32)
    dark_var = measurement.dark_var.astype(np.float32)
    t = measurement.exposure_times_s.astype(np.float32)

    if z.ndim != 4:
        raise ValueError(f"Expected z shape (exposures,frames,y,x), got {z.shape}")
    if dark_mean.shape != z.shape[:1] + z.shape[-2:]:
        raise ValueError(
            f"dark_mean shape {dark_mean.shape} is incompatible with z shape {z.shape}"
        )

    t4 = t[:, None, None, None]
    corrected = z - dark_mean[:, None, :, :]
    preliminary_rate = np.mean(corrected / np.maximum(t4, eps), axis=0)
    preliminary_rate = np.clip(preliminary_rate, 0, None)

    denominator = t4 * preliminary_rate[None, :, :, :] + dark_var[:, None, :, :]
    weights = (t4**2) / np.maximum(denominator, eps)
    fused = np.sum(weights * corrected / np.maximum(t4, eps), axis=0) / np.maximum(
        np.sum(weights, axis=0), eps
    )
    return np.clip(fused, 0, None).astype(np.float32)


def single_exposure_rate(
    measurement: MultiExposureMeasurement, exposure_index: int = 0
) -> np.ndarray:
    """Return one dark-subtracted exposure as a count-rate stack."""

    idx = int(exposure_index)
    z = measurement.z[idx].astype(np.float32)
    dark = measurement.dark_mean[idx].astype(np.float32)
    exposure = float(measurement.exposure_times_s[idx])
    return np.clip((z - dark[None, :, :]) / max(exposure, 1e-12), 0, None).astype(
        np.float32
    )

