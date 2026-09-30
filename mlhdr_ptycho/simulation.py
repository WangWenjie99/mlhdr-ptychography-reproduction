"""Self-contained, paper-style ptychography simulation (Liu et al., TIM 2024).

This module builds the *scene* for the paper-style comparison figures: a
Fraunhofer (FFT) ptychography geometry, a pinhole-like focused probe, a
jittered raster scan, two amplitude-dominated test objects (cameraman and a
procedurally rendered USAF-1951 target, groups 7-9) and the multi-exposure
camera data produced by ``paper_reproduction.simulate_paper_camera``.

IMPORTANT - these numbers are OUR choices. Liu et al. do not report the
simulation's exposure times, read noise, wavelength, detector sampling or
probe; we reuse the experimental wavelength, distance and the seven
experimental exposure times, and choose a virtual detector pitch so that the
object pixel matches the experiment's ~0.571 um transmission sampling.

Conventions
-----------
* Diffraction intensities are ``|fftshift(fft2(P * O_patch, norm="ortho"))|^2``
  with the zero frequency at pixel ``(N//2, N//2)``; the probe is centred at
  the same pixel. These "relative intensity" units are what the camera model
  rescales (brightest frame's TOTAL photon rate = ``photon_flux_per_s``).
* Positions are integer top-left ``(row, col)`` offsets of the N x N probe
  window in the object array, in raster order (row-major over the scan grid).
* Continuous pixel coordinates put the centre of pixel ``(i, j)`` at
  ``(i, j)``; pixel ``i`` spans ``[i - 0.5, i + 0.5)``. USAF geometry uses them.
* Fused rates from ``fuse`` are in count/s (ADU/s). Divide by
  ``PaperMeasurement.count_rate_scale`` to return to relative intensity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
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


def usaf_line_width_um(group: int, element: int) -> float:
    """USAF-1951 line width, 1 / (2 * 2**(G + (E-1)/6)) mm, returned in um."""
    return 1000.0 / (2.0 * 2.0 ** (group + (element - 1) / 6.0))


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
    usaf_bar_amplitude: float = 0.15
    usaf_supersample: int = 16
    usaf_groups: tuple = (7, 8, 9)

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
        d["usaf_groups"] = list(self.usaf_groups)
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


def _axis_coverage(a: float, b: float, size: int, supersample: int) -> np.ndarray:
    """Fraction of the ``supersample`` sub-samples of each pixel inside [a, b)."""
    fine = (np.arange(size * supersample) + 0.5) / supersample - 0.5
    inside = (fine >= a) & (fine < b)
    return inside.reshape(size, supersample).mean(axis=1)


def _usaf_layout(dx_um: float, groups=(7, 8, 9)):
    """Element triplets in continuous px coordinates relative to (0, 0).

    Each element block = horizontal-bar triplet (5w x 5w, left) + gap 1.5w +
    vertical-bar triplet (5w x 5w, right); blocks in a column are separated by
    1.5 w of the lower element. Layout (like the real target, groups nest
    toward the centre): largest group - element 1 top right, elements 2-6 in a
    left column; second group - one column (E1-E6) under the first group's
    element 1; third group - one column (E1-E6) nested beside it.
    """
    if len(groups) != 3:
        raise ValueError("Layout expects exactly three consecutive groups")
    w = {(g, e): usaf_line_width_um(g, e) / dx_um for g in groups for e in range(1, 7)}
    blocks = {}

    def column(g, elements, x, y):
        for e in elements:
            blocks[(g, e)] = (y, x)
            y += 5 * w[(g, e)] + (1.5 * w[(g, e + 1)] if e < elements[-1] else 0)
        return y

    g0, g1, g2 = groups
    column(g0, range(2, 7), 0.0, 0.0)
    col_width0 = 11.5 * w[(g0, 2)]
    x_e1 = col_width0 + w[(g0, 1)]
    blocks[(g0, 1)] = (0.0, x_e1)
    y_inner = 5 * w[(g0, 1)] + w[(g0, 1)]
    bottom1 = column(g1, range(1, 7), x_e1, y_inner)
    height1 = bottom1 - y_inner
    height2 = sum(5 * w[(g2, e)] for e in range(1, 7)) + sum(1.5 * w[(g2, e)] for e in range(2, 7))
    x2 = x_e1 + 11.5 * w[(g1, 1)] + 2 * w[(g1, 1)]
    column(g2, range(1, 7), x2, y_inner + (height1 - height2) / 2)

    entries = []
    for (g, e), (y, x) in sorted(blocks.items()):
        ww = w[(g, e)]
        for orientation, bx in (("horizontal", x), ("vertical", x + 6.5 * ww)):
            if orientation == "horizontal":   # bars along x, stacked in y
                bars = [(y + 2 * k * ww, y + (2 * k + 1) * ww, bx, bx + 5 * ww) for k in range(3)]
            else:                             # bars along y, side by side in x
                bars = [(y, y + 5 * ww, bx + 2 * k * ww, bx + (2 * k + 1) * ww) for k in range(3)]
            entries.append(dict(group=g, element=e, orientation=orientation,
                                line_width_um=usaf_line_width_um(g, e),
                                line_width_px=ww, bars=bars,
                                bbox_px=[y, y + 5 * ww, bx, bx + 5 * ww]))
    return entries


def usaf1951_object(size: int, dx_um: float, groups=(7, 8, 9), bar_amplitude: float = 0.15,
                    supersample: int = 16, center_px=None):
    """Procedural USAF-1951 transmission target (amplitude only, phase 0).

    Background amplitude 1.0, bars ``bar_amplitude``. Bars have length 5w and
    width w, three per triplet separated by w. Rendering is exact
    ``supersample`` x supersampling followed by area (block-mean)
    downsampling, done separably because all bars are axis-aligned.

    Returns ``(object complex128 (size, size), geometry list)``. Each geometry
    entry: group, element, orientation ('horizontal' = bars along x, profile
    along y; 'vertical' = bars along y, profile along x), line_width_um,
    line_width_px, bbox_px [y0, y1, x0, x1], profile_axis, bar_edges_px
    (3 [start, end] along the profile axis), bar_centers_px, bar_extent_px
    ([start, end] along the bar length) -- all in continuous pixel coordinates
    of the object array.
    """
    if supersample < 8:
        raise ValueError("Use at least 8x supersampling")
    entries = _usaf_layout(dx_um, groups)
    ys = [v for en in entries for v in en["bbox_px"][:2]]
    xs = [v for en in entries for v in en["bbox_px"][2:]]
    cy, cx = ((size - 1) / 2.0,) * 2 if center_px is None else center_px
    oy = cy - (min(ys) + max(ys)) / 2.0
    ox = cx - (min(xs) + max(xs)) / 2.0
    coverage = np.zeros((size, size))
    geometry = []
    for en in entries:
        bars = [(a + oy, b + oy, c + ox, d + ox) for a, b, c, d in en["bars"]]
        for a, b, c, d in bars:
            if a < 0 or c < 0 or b > size - 1 or d > size - 1:
                raise ValueError("USAF target does not fit in the object array")
            coverage += np.outer(_axis_coverage(a, b, size, supersample),
                                 _axis_coverage(c, d, size, supersample))
        horizontal = en["orientation"] == "horizontal"
        edges = [[a, b] for a, b, _, _ in bars] if horizontal else [[c, d] for _, _, c, d in bars]
        extent = [bars[0][2], bars[0][3]] if horizontal else [bars[0][0], bars[0][1]]
        y0, y1, x0, x1 = en["bbox_px"]
        geometry.append(dict(
            group=en["group"], element=en["element"], orientation=en["orientation"],
            line_width_um=en["line_width_um"], line_width_px=en["line_width_px"],
            bbox_px=[y0 + oy, y1 + oy, x0 + ox, x1 + ox],
            profile_axis="y" if horizontal else "x",
            bar_edges_px=edges, bar_centers_px=[(a + b) / 2 for a, b in edges],
            bar_extent_px=extent,
        ))
    if coverage.max() > 1 + 1e-9:
        raise AssertionError("USAF bars overlap")
    amplitude = 1.0 - (1.0 - bar_amplitude) * coverage
    return amplitude.astype(np.complex128), geometry


def geometry_to_json(geometry) -> str:
    return json.dumps(geometry, separators=(",", ":"))


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
    usaf_geometry: list | None = field(default=None)

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
    """``kind`` is 'cameraman' or 'usaf'."""
    n = config.detector_pixels
    probe = make_probe(n, config.probe_diameter_px, config.probe_edge_sigma_px,
                       config.probe_curvature_rad)
    positions, size = scan_positions(config.scan_side, config.scan_step_px,
                                     config.scan_jitter_fraction, config.scan_seed,
                                     config.object_pad_px, n)
    roi = illumination_roi(probe, positions, size, config.roi_margin_px)
    geometry = None
    if kind == "cameraman":
        truth = cameraman_object(size, config.cameraman_min_amplitude)
    elif kind == "usaf":
        y0, y1, x0, x1 = roi
        truth, geometry = usaf1951_object(size, config.dx_um, config.usaf_groups,
                                          config.usaf_bar_amplitude, config.usaf_supersample,
                                          center_px=((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2))
        for g in geometry:
            a, b, c, d = g["bbox_px"]
            if a < y0 or c < x0 or b > y1 - 1 or d > x1 - 1:
                raise ValueError("USAF target exceeds the well-scanned ROI; enlarge the scan")
    else:
        raise ValueError(f"Unknown object kind {kind!r}")
    clean = forward_intensity(truth, probe, positions)
    return Scene(kind, truth, probe, positions, clean, roi, config.dx_um, config, geometry)


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
    "cameraman_object", "usaf1951_object", "usaf_line_width_um", "make_probe",
    "scan_positions", "forward_intensity", "illumination_roi", "fuse",
    "saturation_fractions", "geometry_to_json", "extract_patches",
]
