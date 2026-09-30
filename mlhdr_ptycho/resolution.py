"""Ground-truth alignment, image metrics, FRC and USAF-1951 analysis.

All metrics here compare a reconstruction with the KNOWN SIMULATED GROUND
TRUTH (not with a second independent reconstruction). In particular the FRC
is "reconstruction vs truth"; the van Heel half-bit threshold is applied to
it as a conventional, documented cut-off criterion.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

# Michelson contrast a USAF element must reach (worst gap vs worst bar).
USAF_CONTRAST_THRESHOLD = 0.2
# Tolerance (px) by which a bar minimum may lie outside the ideal bar edges.
USAF_BAR_TOLERANCE_PX = 0.25


def _crop(image, roi):
    y0, y1, x0, x1 = roi
    return image[y0:y1, x0:x1]


@dataclass
class Alignment:
    aligned_roi: np.ndarray    # complex128 (h, w): c * ramp-corrected shifted object
    shift_px: tuple            # (dy, dx) applied to the reconstruction
    scalar: complex            # fitted complex scalar c
    phase_ramp_rad_per_px: tuple  # (ky, kx) removed linear phase ramp

    @property
    def amplitude(self):
        return np.abs(self.aligned_roi)


def _hann2d(shape):
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def _coarse_shift(reference, moving, max_shift):
    """Integer shift s (dy, dx) maximising sum ref(r) * moving(r - s), |s| <= max."""
    win = _hann2d(reference.shape)
    a = (reference - reference.mean()) * win
    b = (moving - moving.mean()) * win
    xc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    h, w = xc.shape
    dy = np.fft.fftfreq(h, 1 / h).astype(int)
    dx = np.fft.fftfreq(w, 1 / w).astype(int)
    allowed = (np.abs(dy)[:, None] <= max_shift) & (np.abs(dx)[None, :] <= max_shift)
    idx = np.unravel_index(np.argmax(np.where(allowed, xc, -np.inf)), xc.shape)
    return np.array([dy[idx[0]], dx[idx[1]]], dtype=float)


def align_to_truth(obj, truth, roi, upsample_factor: int = 20, max_shift_px: int = 8,
                   remove_phase_ramp: bool = True, refine_passes: int = 3) -> Alignment:
    """Register a reconstruction to the ground truth inside ``roi``.

    1. Translation: integer peak of the windowed amplitude cross-correlation
       restricted to |shift| <= max_shift_px, refined by
       ``skimage.registration.phase_cross_correlation`` (upsample_factor >= 10,
       plain cross-correlation of mean-subtracted Hann-windowed amplitudes;
       up to ``refine_passes`` passes on the re-shifted object; a refinement
       step > 1 px is discarded). The total shift is applied to the full
       complex object by Fourier shift (ptychography's object/probe
       translation ambiguity).
    2. Optional removal of a linear phase ramp (the object/probe phase-ramp
       ambiguity), estimated from amplitude-weighted neighbour phase
       differences of ``O * conj(O_true)``.
    3. Complex scalar ``c = <O, O_true> / <O, O>`` minimising
       ``||c O - O_true||`` (global phase and scale).
    """
    obj = np.asarray(obj, dtype=np.complex128)
    truth = np.asarray(truth, dtype=np.complex128)
    ref = np.abs(_crop(truth, roi))
    shift = _coarse_shift(ref, np.abs(_crop(obj, roi)), max_shift_px)
    win = _hann2d(ref.shape)
    ref_w = (ref - ref.mean()) * win
    upsample = max(10, int(upsample_factor))
    for _ in range(refine_passes):
        # Windowed, mean-subtracted amplitudes; the small bias of a fixed window
        # toward zero shift scales with the residual, so re-registering the
        # re-shifted object converges onto the upsampled grid point.
        current = _fourier_shift(obj, shift) if np.any(shift) else obj
        mov = np.abs(_crop(current, roi))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # degenerate images warn about RMS error
                fine, _, _ = phase_cross_correlation(
                    ref_w, (mov - mov.mean()) * win, upsample_factor=upsample,
                    normalization=None)
        except Exception:  # degenerate (e.g. constant) images: keep current shift
            break
        if np.any(np.abs(fine) > 1.0) or not np.all(np.isfinite(fine)):
            break
        shift = shift + fine
        if np.all(np.abs(fine) < 0.5 / upsample):
            break
    shifted = _crop(_fourier_shift(obj, shift) if np.any(shift) else obj, roi)
    ramp = (0.0, 0.0)
    t_roi = _crop(truth, roi)
    if remove_phase_ramp:
        g = shifted * np.conj(t_roi)
        ky = float(np.angle(np.sum(g[1:, :] * np.conj(g[:-1, :]))))
        kx = float(np.angle(np.sum(g[:, 1:] * np.conj(g[:, :-1]))))
        yy, xx = np.indices(shifted.shape)
        shifted = shifted * np.exp(-1j * (ky * yy + kx * xx))
        ramp = (ky, kx)
    denom = np.vdot(shifted, shifted)
    c = complex(np.vdot(shifted, t_roi) / denom) if abs(denom) > 0 else 0j
    return Alignment(c * shifted, (float(shift[0]), float(shift[1])), c, ramp)


def _fourier_shift(image, shift):
    return np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(image), shift))


def amplitude_metrics(aligned_roi, truth_roi) -> dict:
    """PSNR/SSIM/NRMSE of |aligned| vs |truth| on the ROI.

    data_range = max(|truth|) - min(|truth|) (fixed by the truth, as in
    ``paper_reproduction.amplitude_comparison``); SSIM is skimage's default
    7x7 mean, which already excludes the 3-px border; NRMSE =
    ||a - a_true|| / ||a_true||.
    """
    a = np.abs(np.asarray(aligned_roi)).astype(np.float64)
    t = np.abs(np.asarray(truth_roi)).astype(np.float64)
    data_range = float(np.ptp(t))
    rmse = float(np.sqrt(np.mean((a - t) ** 2)))
    return {
        "psnr_db": float("inf") if rmse == 0 else float(20 * np.log10(data_range / rmse)),
        "ssim": float(structural_similarity(t, a, data_range=data_range)),
        "nrmse": float(np.linalg.norm(a - t) / np.linalg.norm(t)),
    }


def half_bit_threshold(n_pixels):
    """van Heel & Schatz (2005) half-bit threshold for ring pixel counts."""
    root = np.sqrt(np.maximum(np.asarray(n_pixels, dtype=np.float64), 1.0))
    return (0.2071 + 1.9102 / root) / (1.2071 + 0.9102 / root)


def fourier_ring_correlation(image, reference, dx_um: float, min_frequency: float = 0.05) -> dict:
    """FRC between ``image`` and ``reference`` (here: the GROUND TRUTH).

    Both (square, same shape) images are mean-subtracted and apodised with a
    2D Hann window. Rings are 1 px wide (ring k covers radius k +- 0.5 in
    frequency pixels); frequency is normalised to Nyquist (1.0 = 1/(2 dx)).
    The cutoff is the first crossing of FRC below the half-bit threshold at
    f >= ``min_frequency`` (linearly interpolated); if none, 1.0 (Nyquist)
    and ``reached_nyquist`` = True. ``resolution_um = dx_um / f_cut``
    (half-period resolution; f_cut = 1 -> dx).
    """
    a = np.asarray(image)
    b = np.asarray(reference)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("FRC expects two equal square 2D images")
    n = a.shape[0]
    win = _hann2d(a.shape)
    fa = np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))
    fb = np.fft.fftshift(np.fft.fft2((b - b.mean()) * win))
    yy, xx = np.indices(a.shape) - n // 2
    ring = np.rint(np.hypot(yy, xx)).astype(int).ravel()
    n_rings = n // 2 + 1
    keep = ring < n_rings
    ring = ring[keep]
    num = np.bincount(ring, (fa * np.conj(fb)).real.ravel()[keep], n_rings)
    pa = np.bincount(ring, (np.abs(fa) ** 2).ravel()[keep], n_rings)
    pb = np.bincount(ring, (np.abs(fb) ** 2).ravel()[keep], n_rings)
    count = np.bincount(ring, minlength=n_rings)
    with np.errstate(invalid="ignore", divide="ignore"):
        frc = num / np.sqrt(pa * pb)
    frc = np.nan_to_num(frc, nan=0.0)
    freq = np.arange(n_rings) / (n / 2.0)
    threshold = half_bit_threshold(count)
    cutoff, reached = 1.0, True
    for k in range(1, n_rings):
        if freq[k] < min_frequency:
            continue
        if frc[k] < threshold[k]:
            d0 = frc[k - 1] - threshold[k - 1]
            d1 = frc[k] - threshold[k]
            if d0 > 0:   # interpolate the crossing between rings k-1 and k
                cutoff = float(freq[k - 1] + d0 / (d0 - d1) * (freq[k] - freq[k - 1]))
            else:        # already below at the first ring considered
                cutoff = float(freq[k])
            reached = False
            break
    return {
        "frequency_nyquist": freq, "frc": frc, "threshold": threshold, "ring_pixels": count,
        "cutoff_nyquist": cutoff, "reached_nyquist": reached,
        "resolution_um": float(dx_um / cutoff) if cutoff > 0 else float("inf"),
    }


# --------------------------------------------------------------------------
# USAF-1951
# --------------------------------------------------------------------------
def _element_profile(amplitude, entry):
    """1D profile across the three bars at integer pixel positions.

    Averaged along the central 60 % of the bar length; spans the triplet plus
    one line width (>= 2 px) on each side. Returns (coords_px, values).
    """
    w = entry["line_width_px"]
    start, end = entry["bar_extent_px"]
    length = end - start
    lo, hi = start + 0.2 * length, end - 0.2 * length
    across = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1)
    if across.size == 0:
        across = np.array([int(round((start + end) / 2))])
    edges = entry["bar_edges_px"]
    margin = max(w, 2.0)
    coords = np.arange(int(np.floor(edges[0][0] - margin)), int(np.ceil(edges[2][1] + margin)) + 1)
    if entry["profile_axis"] == "y":
        values = amplitude[np.ix_(coords, across)].mean(axis=1)
    else:
        values = amplitude[np.ix_(across, coords)].mean(axis=0)
    return coords.astype(float), values


def element_resolved(amplitude, entry, contrast_threshold=USAF_CONTRAST_THRESHOLD,
                     tolerance_px=USAF_BAR_TOLERANCE_PX) -> dict:
    """Resolution test for one (group, element, orientation) triplet.

    Resolved iff (i) each of the three bar regions (ideal edges widened by
    ``tolerance_px``) contains a local minimum of the pixel-sampled profile
    (three distinct minima, since widened regions stay disjoint for
    w > 2 * tolerance), and (ii) the worst-case Michelson contrast
    ``(min gap peak - max bar minimum) / (min gap peak + max bar minimum)``
    is >= ``contrast_threshold`` (0.2), where a gap peak is the profile
    maximum strictly between two consecutive bar minima.
    """
    coords, values = _element_profile(amplitude, entry)
    interior = np.arange(1, len(values) - 1)
    is_min = (values[interior] <= values[interior - 1]) & (values[interior] <= values[interior + 1]) \
        & ((values[interior] < values[interior - 1]) | (values[interior] < values[interior + 1]))
    minima = interior[is_min]
    chosen = []
    for a, b in entry["bar_edges_px"]:
        inside = [i for i in minima if a - tolerance_px <= coords[i] <= b + tolerance_px]
        if not inside:
            return {"resolved": False, "contrast": 0.0, "reason": "missing bar minimum"}
        chosen.append(min(inside, key=lambda i: values[i]))
    if len(set(chosen)) < 3:
        return {"resolved": False, "contrast": 0.0, "reason": "minima not distinct"}
    peaks = [values[chosen[k] + 1:chosen[k + 1]].max() if chosen[k + 1] - chosen[k] > 1
             else -np.inf for k in range(2)]
    gap, bar = float(min(peaks)), float(max(values[i] for i in chosen))
    if not np.isfinite(gap) or gap + bar <= 0:
        return {"resolved": False, "contrast": 0.0, "reason": "no gap maximum"}
    contrast = (gap - bar) / (gap + bar)
    return {"resolved": bool(contrast >= contrast_threshold), "contrast": float(contrast),
            "reason": "ok" if contrast >= contrast_threshold else "low contrast"}


def usaf_resolved_elements(amplitude, geometry, roi_offset=(0, 0),
                           contrast_threshold=USAF_CONTRAST_THRESHOLD) -> dict:
    """Evaluate every triplet; report the resolution limit.

    ``amplitude`` may be an ROI crop; ``roi_offset`` = (y0, x0) of that crop in
    object coordinates (geometry is in object coordinates).
    Elements are ordered coarse -> fine (G7E1 ... G9E6). ``limit`` is the
    finest element such that it AND every coarser element are resolved in BOTH
    orientations; ``finest_any`` ignores the monotonic requirement.
    """
    oy, ox = roi_offset
    rows = []
    for entry in geometry:
        e = dict(entry)
        shift = oy if e["profile_axis"] == "y" else ox
        other = ox if e["profile_axis"] == "y" else oy
        e["bar_edges_px"] = [[a - shift, b - shift] for a, b in e["bar_edges_px"]]
        e["bar_extent_px"] = [e["bar_extent_px"][0] - other, e["bar_extent_px"][1] - other]
        res = element_resolved(amplitude, e, contrast_threshold)
        rows.append({"group": entry["group"], "element": entry["element"],
                     "orientation": entry["orientation"],
                     "line_width_um": entry["line_width_um"], **res})
    keys = sorted({(r["group"], r["element"]) for r in rows})
    both = {k: all(r["resolved"] for r in rows if (r["group"], r["element"]) == k) for k in keys}
    limit = None
    for k in keys:
        if not both[k]:
            break
        limit = k
    finest_any = [k for k in keys if both[k]]

    def describe(k):
        if k is None:
            return {"label": None, "line_width_um": None}
        return {"label": f"{k[0]}-{k[1]}",
                "line_width_um": next(r["line_width_um"] for r in rows
                                      if (r["group"], r["element"]) == k)}

    return {"elements": rows, "limit": describe(limit),
            "finest_any": describe(finest_any[-1] if finest_any else None),
            "contrast_threshold": contrast_threshold}


def usaf_line_profile(amplitude, geometry, dx_um: float, roi_offset=(0, 0), group: int = 9,
                      elements=(1, 2, 3), bar_amplitude: float = 0.15,
                      ideal_step_px: float = 0.02) -> dict:
    """Vertical line crossing the horizontal-bar triplets of ``group`` elements.

    The line's x is the centre of the common x-range of those triplets; the
    profile averages the pixel columns inside the central 60 % of that common
    range, at integer rows from 2 w above the first triplet to 2 w below the
    last. Positions are in um from the profile start. ``ideal`` is the square
    wave implied by the truth geometry (1.0 background, ``bar_amplitude``
    bars), sampled every ``ideal_step_px``.
    """
    oy, ox = roi_offset
    ents = [g for g in geometry if g["group"] == group and g["element"] in elements
            and g["orientation"] == "horizontal"]
    ents.sort(key=lambda g: g["element"])
    x_lo = max(g["bar_extent_px"][0] for g in ents) - ox
    x_hi = min(g["bar_extent_px"][1] for g in ents) - ox
    if x_hi <= x_lo:
        raise ValueError("Triplets have no common x-range")
    centre = (x_lo + x_hi) / 2
    half = 0.3 * (x_hi - x_lo)
    cols = np.arange(int(np.ceil(centre - half)), int(np.floor(centre + half)) + 1)
    if cols.size == 0:
        cols = np.array([int(round(centre))])
    w0 = ents[0]["line_width_px"]
    y_start = ents[0]["bar_edges_px"][0][0] - oy - 2 * w0
    y_end = ents[-1]["bar_edges_px"][2][1] - oy + 2 * w0
    rows = np.arange(int(np.ceil(y_start)), int(np.floor(y_end)) + 1)
    values = np.asarray(amplitude)[np.ix_(rows, cols)].mean(axis=1)
    fine = np.arange(y_start, y_end, ideal_step_px)
    ideal = np.ones_like(fine)
    bars = []
    for g in ents:
        for a, b in g["bar_edges_px"]:
            ideal[(fine >= a - oy) & (fine < b - oy)] = bar_amplitude
            bars.append([(a - oy - y_start) * dx_um, (b - oy - y_start) * dx_um])
    return {
        "position_um": (rows - y_start) * dx_um, "profile": values,
        "ideal_position_um": (fine - y_start) * dx_um, "ideal": ideal,
        "line_x_px": float(centre + ox), "columns_px": (cols + ox).tolist(),
        "y_start_px": float(y_start + oy), "bar_edges_um": bars,
        "elements": [f"{g['group']}-{g['element']}" for g in ents],
    }
