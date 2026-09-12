from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DiffStack:
    """A flattened ptychogram plus the original scan shape."""

    stack: np.ndarray
    scan_shape: tuple[int, int]
    frame_shape: tuple[int, int]


def load_diff_npy(path: str | Path, scan_crop: int | None = None) -> DiffStack:
    """Load a 4D `(scan_y, scan_x, det_y, det_x)` diffraction stack.

    If `scan_crop` is provided, a centered square crop is used. This is useful
    for fast smoke tests before running all scan positions.
    """

    array = np.load(path)
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D diffraction array, got shape {array.shape}")

    scan_y, scan_x, det_y, det_x = array.shape
    if det_y != det_x:
        raise ValueError(f"PtyLab expects square detector frames, got {det_y}x{det_x}")

    if scan_crop is not None:
        if scan_crop <= 0:
            raise ValueError("scan_crop must be positive when provided")
        if scan_crop > min(scan_y, scan_x):
            raise ValueError(
                f"scan_crop={scan_crop} is larger than scan shape {scan_y}x{scan_x}"
            )
        start_y = scan_y // 2 - scan_crop // 2
        start_x = scan_x // 2 - scan_crop // 2
        array = array[start_y : start_y + scan_crop, start_x : start_x + scan_crop]

    array = np.asarray(array, dtype=np.float32)
    return DiffStack(
        stack=array.reshape(-1, det_y, det_x),
        scan_shape=tuple(array.shape[:2]),
        frame_shape=tuple(array.shape[-2:]),
    )


def make_encoder_grid(
    scan_shape: tuple[int, int], step_m: float, *, centered: bool = True
) -> np.ndarray:
    """Create PtyLab CPM encoder positions in meters, row-major order."""

    scan_y, scan_x = scan_shape
    rows, cols = np.mgrid[:scan_y, :scan_x]
    rows = rows.astype(np.float64)
    cols = cols.astype(np.float64)
    if centered:
        rows -= (scan_y - 1) / 2
        cols -= (scan_x - 1) / 2

    return np.column_stack([rows.ravel(), cols.ravel()]) * step_m


def normalize_stack(stack: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale a ptychogram stack by its global maximum."""

    stack = np.asarray(stack, dtype=np.float32)
    max_value = float(np.max(stack))
    if max_value <= eps:
        raise ValueError("Cannot normalize an all-zero diffraction stack")
    return stack / max_value

