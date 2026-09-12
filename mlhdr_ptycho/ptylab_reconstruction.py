from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

_mpl_dir = Path(os.environ.get("MLHDR_MPLCONFIGDIR", "/private/tmp/mlhdr_mplconfig"))
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PtyLab.utils.utils import ifft2c

from PtyLab.Engines.mPIE import mPIE
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import DummyMonitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction

from .data import make_encoder_grid, normalize_stack


@dataclass(frozen=True)
class PtyLabConfig:
    wavelength_m: float = 632.8e-9
    detector_pixel_m: float = 5.5e-6
    object_pixel_m: float = 1.0e-6
    scan_step_px: float = 1.0
    entrance_pupil_diameter_px: float = 24.0
    iterations: int = 20
    position_order: str = "random"
    probe_power_correction: bool = True
    propagator_type: str = "Fraunhofer"
    seed: int = 0
    normalize_input: bool = True
    initial_probe: str = "circ_smooth"
    initial_object: str = "ones"
    position_correction: bool = False
    position_correction_radius: int = 1
    quiet: bool = True


class SilentMonitor(DummyMonitor):
    """Patch PtyLab's DummyMonitor with no-op methods used by current engines."""

    def update_encoder(self, *args, **kwargs):
        pass

    def update_positions(self, *args, **kwargs):
        pass

    def update_overlap(self, *args, **kwargs):
        pass

    def updateBeamWidth(self, *args, **kwargs):
        pass

    def update_focusing_metric(self, *args, **kwargs):
        pass


def build_experimental_data(
    ptychogram: np.ndarray,
    scan_shape: tuple[int, int],
    config: PtyLabConfig,
) -> ExperimentalData:
    """Populate a PtyLab `ExperimentalData` object for CPM reconstruction."""

    stack = np.asarray(ptychogram, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected ptychogram shape (frames,y,x), got {stack.shape}")
    if stack.shape[0] != scan_shape[0] * scan_shape[1]:
        raise ValueError(
            f"scan_shape {scan_shape} does not match {stack.shape[0]} frames"
        )
    if stack.shape[-1] != stack.shape[-2]:
        raise ValueError("PtyLab CPM reconstruction expects square detector frames")
    if np.any(stack < 0):
        raise ValueError("Ptychogram intensities must be nonnegative")

    if config.normalize_input:
        stack = normalize_stack(stack)

    data = ExperimentalData(operationMode="CPM")
    data.ptychogram = np.ascontiguousarray(stack)
    data.wavelength = config.wavelength_m
    data.dxd = config.detector_pixel_m
    data.zo = (
        config.object_pixel_m
        * stack.shape[-1]
        * config.detector_pixel_m
        / config.wavelength_m
    )
    data.encoder = make_encoder_grid(
        scan_shape, step_m=config.scan_step_px * config.object_pixel_m
    )
    data.entrancePupilDiameter = (
        config.entrance_pupil_diameter_px * config.object_pixel_m
    )
    data.spectralDensity = None
    data.theta = None
    data.emptyBeam = None
    data._setData()
    return data


def run_mpie(
    ptychogram: np.ndarray,
    scan_shape: tuple[int, int],
    config: PtyLabConfig,
) -> tuple[Reconstruction, ExperimentalData, Params]:
    """Run a PtyLab mPIE reconstruction."""

    np.random.seed(config.seed)
    data = build_experimental_data(ptychogram, scan_shape, config)
    params = Params()
    params.gpuSwitch = False
    params.positionOrder = config.position_order
    params.probePowerCorrectionSwitch = config.probe_power_correction
    params.propagatorType = config.propagator_type
    params.positionCorrectionSwitch = config.position_correction
    params.positionCorrectionSwitch_radius = config.position_correction_radius

    reconstruction = Reconstruction(data, params)
    reconstruction.initialProbe = (
        "circ_smooth" if config.initial_probe == "mean_ifft" else config.initial_probe
    )
    reconstruction.initialObject = config.initial_object
    reconstruction.initializeObjectProbe()
    if config.initial_probe == "mean_ifft":
        reconstruction.initialGuessProbe = _mean_ifft_probe(data, reconstruction)
        reconstruction.probe = reconstruction.initialGuessProbe.copy()

    with _maybe_quiet_tqdm(config.quiet):
        engine = mPIE(reconstruction, data, params, SilentMonitor())
        engine.numIterations = config.iterations
        engine.reconstruct()
    return reconstruction, data, params


def reconstruction_metrics(reconstruction: Reconstruction, data: ExperimentalData) -> dict:
    """Return compact no-reference diagnostics for a reconstruction."""

    error = np.asarray(reconstruction.error, dtype=np.float64)
    obj_amp = np.abs(np.squeeze(reconstruction.object))
    coverage = object_coverage(reconstruction)
    roi = coverage_roi(coverage)
    obj_amp_roi = obj_amp[roi] if roi is not None else obj_amp
    probe_amp = np.abs(np.squeeze(reconstruction.probe))
    final_error = float(error[-1]) if error.size else float("nan")
    return {
        "final_error_sum": final_error,
        "final_error_per_frame": final_error / max(int(data.numFrames), 1),
        "object_amp_min": float(obj_amp.min()),
        "object_amp_max": float(obj_amp.max()),
        "object_amp_std": float(obj_amp.std()),
        "object_roi_amp_min": float(obj_amp_roi.min()),
        "object_roi_amp_max": float(obj_amp_roi.max()),
        "object_roi_amp_std": float(obj_amp_roi.std()),
        "coverage_max": int(coverage.max()),
        "probe_amp_min": float(probe_amp.min()),
        "probe_amp_max": float(probe_amp.max()),
        "num_frames": int(data.numFrames),
        "object_pixels": int(reconstruction.No),
    }


def save_reconstruction_outputs(
    output_dir: str | Path,
    prefix: str,
    reconstruction: Reconstruction,
    data: ExperimentalData,
    config: PtyLabConfig,
) -> None:
    """Save reconstruction arrays, error curve, and quick-look figures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    obj = np.squeeze(reconstruction.object)
    probe = np.squeeze(reconstruction.probe)
    error = np.asarray(reconstruction.error)
    coverage = object_coverage(reconstruction)
    roi = coverage_roi(coverage)

    np.savez_compressed(
        output / f"{prefix}_reconstruction.npz",
        object=obj,
        probe=probe,
        error=error,
        encoder=data.encoder,
        coverage=coverage,
        config=np.array([asdict(config)], dtype=object),
    )
    reconstruction.saveResults(str(output / f"{prefix}_ptylab.h5"), squeeze=False)

    _save_complex_field(output / f"{prefix}_object.png", obj, "Object")
    if roi is not None:
        _save_complex_field(output / f"{prefix}_object_roi.png", obj[roi], "Object ROI")
    _save_complex_field(output / f"{prefix}_probe.png", probe, "Probe")
    _save_coverage(output / f"{prefix}_coverage.png", coverage)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(error, marker="o", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PtyLab error")
    ax.set_title("mPIE convergence")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_error.png", dpi=160)
    plt.close(fig)


def object_coverage(reconstruction: Reconstruction) -> np.ndarray:
    """Count how many scan patches update each object pixel."""

    coverage = np.zeros((reconstruction.No, reconstruction.No), dtype=np.uint16)
    for row, col in reconstruction.positions:
        coverage[row : row + reconstruction.Np, col : col + reconstruction.Np] += 1
    return coverage


def coverage_roi(coverage: np.ndarray) -> tuple[slice, slice] | None:
    """Bounding box of object pixels touched by at least one scan patch."""

    rows, cols = np.nonzero(coverage > 0)
    if rows.size == 0:
        return None
    return (
        slice(int(rows.min()), int(rows.max()) + 1),
        slice(int(cols.min()), int(cols.max()) + 1),
    )


def _mean_ifft_probe(
    data: ExperimentalData, reconstruction: Reconstruction, eps: float = 1e-12
) -> np.ndarray:
    """Initialize a probe from the average measured diffraction amplitude."""

    amplitude = np.sqrt(np.mean(data.ptychogram, axis=0).astype(np.float32) + eps)
    probe2d = ifft2c(amplitude).astype(np.complex64)
    power = np.sqrt(np.sum(np.abs(probe2d) ** 2))
    if power > eps:
        probe2d = probe2d / power * data.maxProbePower
    return probe2d.reshape(reconstruction.shape_P).astype(np.complex64)


class _QuietIterator:
    def __init__(self, iterable):
        self._iterable = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterable)

    def write(self, *args, **kwargs):
        pass


class _maybe_quiet_tqdm:
    def __init__(self, quiet: bool):
        self.quiet = quiet
        self._module = None
        self._old_tqdm = None
        self._old_trange = None

    def __enter__(self):
        if not self.quiet:
            return
        import importlib

        self._module = importlib.import_module("PtyLab.Engines.mPIE")
        self._old_tqdm = self._module.tqdm.tqdm
        self._old_trange = self._module.tqdm.trange
        self._module.tqdm.tqdm = lambda iterable, *args, **kwargs: _QuietIterator(
            iterable
        )
        self._module.tqdm.trange = lambda *args, **kwargs: _QuietIterator(range(*args))

    def __exit__(self, exc_type, exc, tb):
        if not self.quiet or self._module is None:
            return
        self._module.tqdm.tqdm = self._old_tqdm
        self._module.tqdm.trange = self._old_trange


def _save_complex_field(path: Path, field: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    amp = np.abs(field)
    phase = np.angle(field)
    im0 = axes[0].imshow(amp, cmap="gray")
    axes[0].set_title(f"{title} amplitude")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(phase, cmap="twilight")
    axes[1].set_title(f"{title} phase")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_coverage(path: Path, coverage: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(coverage, cmap="magma")
    ax.set_title("Patch coverage")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
