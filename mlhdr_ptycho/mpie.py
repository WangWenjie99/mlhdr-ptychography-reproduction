"""Pure-numpy mPIE (Maiden, Johnson & Li, Optica 4, 736 (2017)).

Sequential (per-position) rPIE object/probe updates with momentum, for the
Fraunhofer CPM geometry of ``simulation.py``:

    psi   = P * O_j,  Psi = F[psi] (orthonormal FFT)
    Psi'  = sqrt(I_j) * Psi / |Psi|                       (modulus constraint)
    dpsi  = F^-1[Psi'] - psi
    O_j  += beta_O * conj(P) dpsi / ((1-alpha_O)|P|^2 + alpha_O max|P|^2)
    P    += beta_P * conj(O_j) dpsi / ((1-alpha_P)|O_j|^2 + alpha_P max|O_j|^2)

(the probe update uses the object patch *before* its update). Every
``momentum_interval`` position updates, PtyLab-style momentum is applied to
object and probe:  m <- (buffer - X) + friction * m;  X <- X - feedback * m;
buffer <- X. PtyLab triggers momentum stochastically with probability 0.05
per update; we use the deterministic equivalent interval of 20 updates.

Defaults follow PtyLab's mPIE engine (alpha_O 0.1, beta_O = beta_P = 0.25,
friction 0.7, feedback 0.3) EXCEPT ``alpha_probe = 1.0`` (ePIE-type probe
normalisation by max|O_j|^2). With PtyLab's alpha_P = 0.1 the joint
object/probe iteration stagnated in our tests: noiseless USAF data reached
object NRMSE 0.047 after 250 iterations (0.0013 with alpha_P = 1.0), and the
8-bit ML-masked cameraman reconstruction failed (NRMSE 0.29 vs 0.053).

Positions are visited in a random order drawn each iteration from
``np.random.default_rng(config.seed)``; use the same seed for every method to
compare fairly. With ``probe_power_correction`` the probe is rescaled after
each iteration so that sum|P|^2 equals the brightest frame's energy (PtyLab's
probePowerCorrectionSwitch). With ``com_stabilization`` (default on here; cf.
PtyLab's comStabilizationSwitch) the probe's |P|^2 centre of mass is moved
back to the window centre (N//2) after each iteration by an integer shift,
and the object (plus momentum buffers) is shifted identically, which leaves
every exit wave P(r) O(r + R_j) - hence the data fit - unchanged. Without it
the object/probe translation ambiguity let reconstructions drift by up to
>8 px in 250 iterations. Computation is complex128 by default
(``precision="double"``); complex64 is available but can hit slow subnormal
arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy import fft as sfft
from scipy.special import erfc


@dataclass(frozen=True)
class MPIEConfig:
    iterations: int = 250
    alpha_object: float = 0.1
    alpha_probe: float = 1.0     # PtyLab: 0.1 (stagnates here, see docstring)
    beta_object: float = 0.25
    beta_probe: float = 0.25
    friction: float = 0.7
    feedback: float = 0.3
    momentum_interval: int = 20
    probe_update: bool = True
    probe_power_correction: bool = True
    probe_update_start: int = 1  # first iteration with probe updates
    com_stabilization: bool = True  # re-centre probe |P|^2 centre of mass (PtyLab comStabilizationSwitch)
    seed: int = 0
    history_every: int = 5       # monitor cadence (iterations); 0 disables
    precision: str = "double"    # "double" (complex128) or "single" (complex64)

    def to_dict(self):
        return asdict(self)


@dataclass
class MPIEResult:
    object: np.ndarray                 # complex (H, W), precision per config
    probe: np.ndarray                  # complex (N, N)
    diffraction_error: np.ndarray      # float64 (iterations_done,)
    history: dict                      # lists keyed by 'iteration', 'wallclock_s', ...
    seconds: float                     # reconstruction wall-clock, monitor excluded
    diverged: bool


def initial_probe(n: int, diameter_px: float, power: float, edge_sigma_px: float = 1.5):
    """Flat-phase soft circular aperture with sum|P|^2 == power (complex128).

    Values below 1e-12 of the peak are set to exactly zero (avoids subnormal
    floats in single precision).
    """
    yy, xx = np.indices((n, n), dtype=np.float64) - n // 2
    r = np.hypot(yy, xx)
    amp = 0.5 * erfc((r - diameter_px / 2.0) / (math.sqrt(2.0) * edge_sigma_px))
    amp[amp < 1e-12 * amp.max()] = 0.0
    amp *= math.sqrt(power / float(np.sum(amp ** 2)))
    return amp.astype(np.complex128)


def _shift_content(a, dy: int, dx: int, fill):
    """out(u) = a(u - (dy, dx)) for integer shifts; vacated pixels = ``fill``."""
    out = np.full_like(a, fill)
    h, w = a.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    src_y = slice(max(-dy, 0), h - max(dy, 0))
    dst_y = slice(max(dy, 0), h - max(-dy, 0))
    src_x = slice(max(-dx, 0), w - max(dx, 0))
    dst_x = slice(max(dx, 0), w - max(-dx, 0))
    out[dst_y, dst_x] = a[src_y, src_x]
    return out


def probe_com_offset(probe):
    """Integer offset (dy, dx) of the |P|^2 centre of mass from (N//2, N//2)."""
    p2 = np.abs(probe) ** 2
    total = float(p2.sum())
    if total <= 0 or not np.isfinite(total):
        return 0, 0
    n0, n1 = probe.shape
    cy = float(p2.sum(axis=1) @ np.arange(n0)) / total
    cx = float(p2.sum(axis=0) @ np.arange(n1)) / total
    return int(round(cy - n0 // 2)), int(round(cx - n1 // 2))


def reconstruct(intensities, positions, object_shape, probe0, config: MPIEConfig = MPIEConfig(),
                monitor=None, object0=None) -> MPIEResult:
    """Run mPIE.

    intensities : (J, N, N) nonnegative, fftshifted (DC at N//2) - e.g. fused
                  rate / count_rate_scale. Any global scale works.
    positions   : (J, 2) integer top-left (row, col) offsets.
    probe0      : (N, N) initial probe (see ``initial_probe``).
    monitor     : optional callable(object, probe) -> dict of floats, called
                  at iteration 0, every ``history_every`` iterations and at the
                  end; its time is excluded from ``seconds``/'wallclock_s'.

    Diffraction error (per iteration, from the pre-update exit waves):
    sum_j sum_q (|Psi_j| - sqrt(I_j))^2 / sum_j sum_q I_j.
    """
    if config.precision not in ("double", "single"):
        raise ValueError("precision must be 'double' or 'single'")
    cdt = np.complex128 if config.precision == "double" else np.complex64
    rdt = np.float64 if config.precision == "double" else np.float32
    intensities = np.asarray(intensities)
    j_count, n, _ = intensities.shape
    positions = np.asarray(positions, dtype=np.int64)
    if positions.shape != (j_count, 2):
        raise ValueError("positions must be (J, 2)")
    if not np.isfinite(intensities).all():
        raise ValueError("Intensities must be finite")
    amp64 = np.sqrt(np.maximum(intensities.astype(np.float64), 0))
    total = float(np.sum(amp64 ** 2))
    max_frame_power = float(np.max(np.sum(amp64 ** 2, axis=(-2, -1))))
    amp_data = np.ascontiguousarray(np.fft.ifftshift(amp64, axes=(-2, -1)).astype(rdt))

    obj = (np.ones(object_shape, cdt) if object0 is None else np.array(object0, dtype=cdt))
    probe = np.array(probe0, dtype=cdt)
    obj_buffer, probe_buffer = obj.copy(), probe.copy()
    obj_momentum = np.zeros_like(obj)
    probe_momentum = np.zeros_like(probe)

    rng = np.random.default_rng(config.seed)
    a_o, a_p = rdt(config.alpha_object), rdt(config.alpha_probe)
    b_o, b_p = rdt(config.beta_object), rdt(config.beta_probe)
    friction, feedback = rdt(config.friction), rdt(config.feedback)
    fft2, ifft2 = sfft.fft2, sfft.ifft2
    tiny = rdt(1e-30)

    history = {"iteration": [], "wallclock_s": [], "diffraction_error": []}
    errors = []
    elapsed = 0.0
    counter = 0
    diverged = False

    def record(iteration, err):
        history["iteration"].append(int(iteration))
        history["wallclock_s"].append(float(elapsed))
        history["diffraction_error"].append(float(err))
        if monitor is not None:
            for key, value in monitor(obj, probe).items():
                history.setdefault(key, []).append(float(value))

    if config.history_every:
        record(0, float("nan"))

    for it in range(1, config.iterations + 1):
        start = time.perf_counter()
        err_sum = 0.0
        update_probe = config.probe_update and it >= config.probe_update_start
        for j in rng.permutation(j_count):
            y, x = positions[j]
            view = obj[y:y + n, x:x + n]
            patch = view.copy()
            psi = probe * patch
            far = fft2(psi, norm="ortho")
            model = np.abs(far)
            measured = amp_data[j]
            diff = model - measured
            err_sum += float(np.vdot(diff, diff))
            far *= measured / (model + tiny)
            dpsi = ifft2(far, norm="ortho", overwrite_x=True)
            dpsi -= psi
            p2 = probe.real * probe.real + probe.imag * probe.imag
            view += (b_o * np.conj(probe) * dpsi) / ((1 - a_o) * p2 + a_o * p2.max())
            if update_probe:
                o2 = patch.real * patch.real + patch.imag * patch.imag
                probe += (b_p * np.conj(patch) * dpsi) / ((1 - a_p) * o2 + a_p * o2.max())
            counter += 1
            if config.momentum_interval and counter % config.momentum_interval == 0:
                obj_momentum = (obj_buffer - obj) + friction * obj_momentum
                obj -= feedback * obj_momentum
                obj_buffer = obj.copy()
                if update_probe:
                    probe_momentum = (probe_buffer - probe) + friction * probe_momentum
                    probe -= feedback * probe_momentum
                    probe_buffer = probe.copy()
                else:
                    probe_buffer = probe.copy()
        if config.com_stabilization and update_probe:
            dy, dx = probe_com_offset(probe)
            if dy or dx:   # P'(r) = P(r + d), O'(u) = O(u + d): exit waves unchanged
                probe = np.roll(probe, (-dy, -dx), axis=(0, 1))
                probe_buffer = np.roll(probe_buffer, (-dy, -dx), axis=(0, 1))
                probe_momentum = np.roll(probe_momentum, (-dy, -dx), axis=(0, 1))
                obj = _shift_content(obj, -dy, -dx, 1.0)
                obj_buffer = _shift_content(obj_buffer, -dy, -dx, 1.0)
                obj_momentum = _shift_content(obj_momentum, -dy, -dx, 0.0)
        if config.probe_power_correction and update_probe:
            power = float(np.sum(np.abs(probe.astype(np.complex128)) ** 2))
            if power > 0:
                scale = rdt(math.sqrt(max_frame_power / power))
                probe *= scale
                probe_buffer *= scale
                probe_momentum *= scale
        elapsed += time.perf_counter() - start
        err = err_sum / max(total, 1e-300)
        errors.append(err)
        if not (np.isfinite(err) and np.isfinite(obj).all() and np.isfinite(probe).all()):
            diverged = True
            break
        if config.history_every and (it % config.history_every == 0 or it == config.iterations):
            record(it, err)

    return MPIEResult(obj, probe, np.asarray(errors), history, elapsed, diverged)
