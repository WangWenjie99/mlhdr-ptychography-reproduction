"""Self-contained, paper-style ptychography simulation (Liu et al., TIM 2024).

This module builds the *scene* for the paper-style comparison figures: a
Fraunhofer (FFT) ptychography geometry, a pinhole-like focused probe, a
jittered raster scan, an amplitude-dominated test object (cameraman) and the
multi-exposure camera data produced by ``paper_reproduction.simulate_paper_camera``.

IMPORTANT - these numbers are OUR choices. Liu et al. do not report the
simulation's exposure times, read noise, dark current, wavelength or pixel
sizes, and our geometry, object sampling, probe and scan step differ from the
paper's simulation. We reuse the experimental wavelength, distance and the
seven experimental exposure times, and choose a virtual detector pitch so that
the object pixel matches the experiment's ~0.571 um transmission sampling.

Conventions
-----------
* Diffraction intensities are ``|fftshift(fft2(P * O_patch, norm="ortho"))|^2``
  with the zero frequency at pixel ``(N//2, N//2)``; the probe is centred at
  the same pixel. These "relative intensity" units are what the camera model
  rescales (brightest frame's TOTAL photon rate = ``photon_flux_per_s``).
* Positions are integer top-left ``(row, col)`` offsets of the N x N probe
  window in the object array, in raster order (row-major over the scan grid).
* Fused rates from ``fuse`` are in count/s (ADU/s). Divide by
  ``PaperMeasurement.count_rate_scale`` to return to relative intensity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.special import erfc

from .paper_reproduction import (
    PaperCamera, PaperMeasurement, lrfc_hdr_fusion, paper_eq14_15,
    saturation_mask_extension, simulate_paper_camera, single_rate,
)

METHODS = ("single", "lrfc", "ml_eq14_15", "ml_masked")
# Seven experimental exposure times of Liu et al. (s), used for every sweep.
EXPERIMENT_EXPOSURES_S = (0.5e-3, 1e-3, 5e-3, 10e-3, 50e-3, 100e-3, 500e-3)


@dataclass(frozen=True)
class SimulationConfig:
    """Every physical/numerical choice of the paper-style simulation."""

    wavelength_m: float = 632.8e-9
    distance_m: float = 13.9e-3
    detector_pixels: int = 64
    # Virtual (e.g. binned) detector pitch chosen so dx = lambda z / (N pitch)
    # reproduces the experiment's ~0.571 um object pixel.
    detector_pixel_m: float = 240.64e-6
    probe_diameter_px: float = 32.0      # 50 % amplitude diameter
    probe_edge_sigma_px: float = 1.0     # Gaussian softening of the aperture edge
    probe_curvature_rad: float = 1.0     # quadratic phase at the aperture rim
    scan_side: int = 20                  # scan_side x scan_side raster
    scan_step_px: float = 8.0            # 75 % linear overlap for 32 px probe
    scan_jitter_fraction: float = 0.1    # uniform +-10 % of the step, rounded
    scan_seed: int = 2024
    object_pad_px: int = 4
    roi_margin_px: int = 6
    exposure_times_s: tuple = EXPERIMENT_EXPOSURES_S
    photon_flux_per_s: float = 1e9
    full_well_e: float = 2.5e6
    read_noise_e: float = 5.0            # repo "low_noise" profile
    dark_current_e_per_s: float = 80.0
    dark_frames: int = 20
    cameraman_min_amplitude: float = 0.2

    @property
    def dx_m(self) -> float:
        return self.wavelength_m * self.distance_m / (
            self.detector_pixels * self.detector_pixel_m)

    @property
    def dx_um(self) -> float:
        return self.dx_m * 1e6

    @property
    def nominal_overlap(self) -> float:
        return 1.0 - self.scan_step_px / self.probe_diameter_px

    def camera(self, bits: int = 8, read_noise_e: float | None = None) -> PaperCamera:
        return PaperCamera(
            photon_flux_per_s=self.photon_flux_per_s,
            full_well_e=self.full_well_e,
            bit_depth=int(bits),
            read_noise_e=self.read_noise_e if read_noise_e is None else float(read_noise_e),
            dark_current_e_per_s=self.dark_current_e_per_s,
            dark_frames=self.dark_frames,
            allow_extended_bit_depth=True,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["exposure_times_s"] = list(self.exposure_times_s)
        d["dx_m"] = self.dx_m
        d["dx_um"] = self.dx_um
        d["nominal_overlap"] = self.nominal_overlap
        return d


# --------------------------------------------------------------------------
# Probe, scan and forward model
# --------------------------------------------------------------------------
def make_probe(n: int = 64, diameter_px: float = 32.0, edge_sigma_px: float = 1.0,
               curvature_rad: float = 1.0) -> np.ndarray:
    """Pinhole-like focused beam: soft-edged disk with mild quadratic phase.

    Amplitude = disk of radius ``diameter_px/2`` convolved (radially) with a
    Gaussian edge of sigma ``edge_sigma_px`` (value 0.5 at the rim, max 1).
    Phase = ``curvature_rad * (r / R)^2``. Centred at pixel (n//2, n//2).
    """
    yy, xx = np.indices((n, n), dtype=np.float64) - n // 2
    r = np.hypot(yy, xx)
    radius = diameter_px / 2.0
    amp = 0.5 * erfc((r - radius) / (math.sqrt(2.0) * max(edge_sigma_px, 1e-6)))
    phase = curvature_rad * (r / radius) ** 2
    return (amp * np.exp(1j * phase)).astype(np.complex128)


def scan_positions(side: int, step_px: float, jitter_fraction: float, seed: int,
                   pad_px: int = 4, n: int = 64):
    """Jittered raster scan: integer top-left offsets and the object size.

    Nominal grid ``k * step``; each coordinate gets an independent uniform
    offset in ``[-jitter_fraction, +jitter_fraction] * step`` and is rounded
    to the nearest pixel (integer-pixel positions, as in PtyLab's CPM mode).
    Returns ``(positions (side*side, 2) int64, object_size int)``; the scan is
    centred in a square object array.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(side) * float(step_px)
    gy, gx = np.meshgrid(grid, grid, indexing="ij")
    jitter = rng.uniform(-jitter_fraction, jitter_fraction, size=(side * side, 2)) * step_px
    pos = np.rint(np.stack([gy.ravel(), gx.ravel()], axis=1) + jitter).astype(np.int64)
    pos -= pos.min(axis=0)
    size = int(pos.max() + n + 2 * pad_px)
    size += size % 2  # even object size
    offset = (size - (pos.max(axis=0) + n)) // 2
    return pos + offset[None, :], size


def extract_patches(obj: np.ndarray, positions: np.ndarray, n: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(obj, (n, n))
    return windows[positions[:, 0], positions[:, 1]]


def forward_intensity(obj: np.ndarray, probe: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Noiseless far-field intensities (J, N, N), fftshifted, ortho FFT."""
    n = probe.shape[0]
    exit_waves = probe[None] * extract_patches(obj, positions, n)
    far = np.fft.fftshift(np.fft.fft2(exit_waves, norm="ortho"), axes=(-2, -1))
    return (far.real ** 2 + far.imag ** 2).astype(np.float64)


def illumination_roi(probe: np.ndarray, positions: np.ndarray, object_size: int,
                     margin_px: int = 6, footprint_level: float = 0.5):
    """Square ROI = bounding box of the union of probe footprints minus a margin.

    A footprint is ``|P| >= footprint_level * max|P|``. The box is shrunk by
    ``margin_px`` (and further if needed) until every ROI pixel lies inside
    the footprint union, then cropped to a centred square.
    Returns ``(y0, y1, x0, x1)`` half-open integer bounds.
    """
    n = probe.shape[0]
    foot = np.abs(probe) >= footprint_level * np.abs(probe).max()
    union = np.zeros((object_size, object_size), bool)
    for y, x in positions:
        union[y:y + n, x:x + n] |= foot
    rows, cols = np.nonzero(union.any(axis=1))[0], np.nonzero(union.any(axis=0))[0]
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    margin = int(margin_px)
    while True:
        a, b, c, d = y0 + margin, y1 - margin, x0 + margin, x1 - margin
        side = min(b - a, d - c)
        if side <= 8:
            raise ValueError("Scan too small for a well-scanned ROI")
        a += ((b - a) - side) // 2
        c += ((d - c) - side) // 2
        if union[a:a + side, c:c + side].all():
            return int(a), int(a + side), int(c), int(c + side)
        margin += 1


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------
def cameraman_object(size: int, min_amplitude: float = 0.2) -> np.ndarray:
    """skimage cameraman resized to ``size`` and mapped to [min_amplitude, 1]."""
    from skimage import data
    from skimage.transform import resize

    img = resize(data.camera().astype(np.float64), (size, size),
                 anti_aliasing=True, preserve_range=True)
    img = (img - img.min()) / max(img.max() - img.min(), 1e-12)
    return (min_amplitude + (1.0 - min_amplitude) * img).astype(np.complex128)


# --------------------------------------------------------------------------
# Scene, camera and fusion
# --------------------------------------------------------------------------
@dataclass
class Scene:
    kind: str
    truth: np.ndarray            # complex128 (H, W) object transmission
    probe: np.ndarray            # complex128 (N, N)
    positions: np.ndarray        # int64 (J, 2) top-left (row, col)
    clean: np.ndarray            # float64 (J, N, N) fftshifted relative intensity
    roi: tuple                   # (y0, y1, x0, x1) half-open, square
    dx_um: float
    config: SimulationConfig

    @property
    def rate_scale(self) -> float:
        """Photon/s per relative-intensity unit (see simulate_paper_camera)."""
        return self.config.photon_flux_per_s / float(self.clean.sum(axis=(-2, -1)).max())

    def auto_exposure_index(self) -> int:
        """Single-exposure baseline ("auto-exposure").

        The longest of the configured exposures whose NOISELESS peak pixel
        (signal + mean dark electrons) stays below the full well. It depends
        only on the object, flux, dark current and full well, hence is fixed
        across bit depths and read-noise levels. Falls back to the shortest.
        """
        cfg = self.config
        times = np.asarray(cfg.exposure_times_s, dtype=np.float64)
        peak = float(self.clean.max()) * self.rate_scale
        electrons = (peak + cfg.dark_current_e_per_s) * times
        ok = np.nonzero(electrons < cfg.full_well_e)[0]
        if not ok.size:
            return int(np.argmin(times))
        return int(ok[np.argmax(times[ok])])

    def simulate(self, bits: int = 8, read_noise_e: float | None = None,
                 seed: int = 0) -> PaperMeasurement:
        """Multi-exposure data via ``simulate_paper_camera`` (same seed ->
        identical electron-level noise at every bit depth / read-noise level)."""
        camera = self.config.camera(bits, read_noise_e)
        return simulate_paper_camera(self.clean, self.config.exposure_times_s, camera, seed)


def build_scene(kind: str, config: SimulationConfig = SimulationConfig()) -> Scene:
    """``kind`` is 'cameraman' (the only object of the paper-style sweeps)."""
    if kind != "cameraman":
        raise ValueError(f"Unknown object kind {kind!r}")
    n = config.detector_pixels
    probe = make_probe(n, config.probe_diameter_px, config.probe_edge_sigma_px,
                       config.probe_curvature_rad)
    positions, size = scan_positions(config.scan_side, config.scan_step_px,
                                     config.scan_jitter_fraction, config.scan_seed,
                                     config.object_pad_px, n)
    roi = illumination_roi(probe, positions, size, config.roi_margin_px)
    truth = cameraman_object(size, config.cameraman_min_amplitude)
    clean = forward_intensity(truth, probe, positions)
    return Scene(kind, truth, probe, positions, clean, roi, config.dx_um, config)


def fuse(measurement: PaperMeasurement, method: str, single_index: int, chunk: int = 50):
    """Return ``(rate count/s float32 (J, N, N), info dict)`` for one method key.

    single     -> ``single_rate`` at ``single_index`` (auto-exposure)
    lrfc       -> ``lrfc_hdr_fusion`` (conventional LRFC-HDR, paper Eq. 16-17)
    ml_eq14_15 -> ``paper_eq14_15`` (published Eq. 14-15, no pixel rejection)
    ml_masked  -> ``saturation_mask_extension(all_saturated="shortest")``
                  (our extension, NOT the published algorithm)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    max_count = measurement.camera.max_count
    all_saturated = int(np.count_nonzero((measurement.z >= max_count).all(axis=0)))
    info = {"all_saturated_pixels": all_saturated}
    if method == "single":
        info["exposure_s"] = float(measurement.exposure_times_s[single_index])
        return single_rate(measurement, single_index).astype(np.float32), info
    function = {
        "lrfc": lrfc_hdr_fusion,
        "ml_eq14_15": paper_eq14_15,
        "ml_masked": lambda m: saturation_mask_extension(m, all_saturated="shortest"),
    }[method]
    # Every fusion is pixel-wise, so processing positions in chunks gives
    # identical results with a much smaller memory peak.
    frames = measurement.z.shape[1]
    out = np.empty(measurement.z.shape[1:], np.float32)
    for start in range(0, frames, chunk):
        part = replace(measurement, z=measurement.z[:, start:start + chunk])
        out[start:start + chunk] = function(part)
    return out, info


def saturation_fractions(measurement: PaperMeasurement) -> np.ndarray:
    """Fraction of pixels at the ADC maximum, per exposure (whole stack)."""
    return (measurement.z >= measurement.camera.max_count).mean(axis=(1, 2, 3))


__all__ = [
    "METHODS", "EXPERIMENT_EXPOSURES_S", "SimulationConfig", "Scene", "build_scene",
    "cameraman_object", "make_probe", "scan_positions", "forward_intensity",
    "illumination_roi", "fuse", "saturation_fractions", "extract_patches",
]
