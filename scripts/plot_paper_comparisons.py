#!/usr/bin/env python3
"""Create publication-style comparison figures from recorded metrics only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlhdr_mplconfig"))

from mlhdr_ptycho.comparison_figures import generate_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs",
                        help="Directory with summary_metrics.csv and experiment_manifest.json or manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/paper_comparisons")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300); SVG remains vector")
    args = parser.parse_args()
    try:
        provenance = generate_figures(args.source, args.output, args.dpi)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot generate comparison figures: {exc}\n")
    for name in provenance["figures"]:
        print(f"Created {args.output / name}.png / .svg")
    print(f"Provenance: {args.output / 'figures_manifest.json'}")


if __name__ == "__main__":
    main()
