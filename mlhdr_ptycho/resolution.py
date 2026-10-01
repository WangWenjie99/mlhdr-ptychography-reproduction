"""Ground-truth alignment, image metrics and FRC.

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
