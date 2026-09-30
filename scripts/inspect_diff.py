#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / "mlhdr_mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect diff.npy ptychogram data.")
    parser.add_argument("--input", default="diff.npy")
    parser.add_argument("--output", default="outputs/inspect")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    diff = np.load(args.input)
    print(f"shape: {diff.shape}")
    print(f"dtype: {diff.dtype}")
    print(f"min/max: {diff.min():.6g} / {diff.max():.6g}")
    print(f"mean/std: {diff.mean():.6g} / {diff.std():.6g}")
    print(f"percentiles: {np.percentile(diff, [0, 1, 5, 50, 95, 99, 100])}")

    frame_sums = diff.sum(axis=(-1, -2))
    print(
        "frame sum min/max/mean/std: "
        f"{frame_sums.min():.6g} / {frame_sums.max():.6g} / "
        f"{frame_sums.mean():.6g} / {frame_sums.std():.6g}"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(frame_sums, cmap="magma")
    ax.set_title("Frame-sum map")
    ax.set_xlabel("scan x")
    ax.set_ylabel("scan y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "frame_sum_map.png", dpi=160)
    plt.close(fig)

    scan_y, scan_x = diff.shape[:2]
    coords = [
        (0, 0),
        (scan_y // 2, scan_x // 2),
        (scan_y - 1, scan_x - 1),
        np.unravel_index(np.argmax(frame_sums), frame_sums.shape),
    ]
    fig, axes = plt.subplots(1, len(coords), figsize=(3 * len(coords), 3))
    for ax, (row, col) in zip(axes, coords):
        ax.imshow(np.log10(diff[row, col] + 1e-8), cmap="viridis")
        ax.set_title(f"({row},{col})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "example_diffraction_frames.png", dpi=160)
    plt.close(fig)

    print(f"wrote inspection figures to {output}")


if __name__ == "__main__":
    main()
