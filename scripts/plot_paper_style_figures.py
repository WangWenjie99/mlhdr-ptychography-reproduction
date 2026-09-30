#!/usr/bin/env python3
"""Render figures imitating Liu et al. 2024 (Figs. 2, 3, 5, 6, 7) from paper-style simulation data.

Reads only the files of ``scripts/run_paper_style_simulation.py`` (DATA_CONTRACT.md)
and ``docs/paper_style/paper_digitized.json``; nothing is estimated from images.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlhdr_ptycho.paper_style_figures import (  # noqa: E402
    DataContractError, copy_small_data, generate_figures, load_paper_digitized,
    load_paper_style_data, write_summary_csv,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/paper_style/full",
                        help="Directory written by run_paper_style_simulation.py (default: outputs/paper_style/full)")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/paper_style",
                        help="Figure directory (PNG + SVG + figures_manifest.json)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default 300); SVG stays vector")
    parser.add_argument("--paper-json", type=Path, default=None,
                        help="Digitised paper values (default: docs/paper_style/paper_digitized.json)")
    parser.add_argument("--summary-csv", type=Path, default=None,
                        help="Also write the paper-vs-reproduction key-number table to this CSV")
    parser.add_argument("--copy-data-to", type=Path, default=None,
                        help="Also copy the source CSV and JSON files (no NPZ) into this directory")
    args = parser.parse_args(argv)
    try:
        manifest = generate_figures(args.source, args.output, args.dpi, args.paper_json)
        if args.summary_csv or args.copy_data_to:
            data = load_paper_style_data(args.source)
            if args.summary_csv:
                rows = write_summary_csv(data, args.summary_csv, load_paper_digitized(args.paper_json))
                print(f"Summary: {args.summary_csv} ({len(rows)} rows)")
            if args.copy_data_to:
                for path in copy_small_data(data, args.copy_data_to):
                    print(f"Copied {path}")
    except (DataContractError, ValueError, OSError) as exc:
        parser.exit(2, f"Cannot generate paper-style figures: {exc}\n")
    for name in manifest["figures"]:
        print(f"Created {args.output / name}.png / .svg")
    print(f"Provenance: {args.output / 'figures_manifest.json'}")


if __name__ == "__main__":
    main()
