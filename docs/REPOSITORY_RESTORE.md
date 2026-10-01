# Restore and continue on another device

The saved paper-style simulation data are tracked in Git together with the code. After cloning or pulling the repository, you can view results, regenerate all six comparison figure sets, read reconstruction arrays for further analysis, or run simulations in a new directory. No separate transfer of these simulation files from the original device is required.

## 1. Get the repository and install the environment

Run `git clone <repository-url>` using the remote repository's clone URL, or `git pull` in an existing clone. Enter the repository root containing `README.md`, `mlhdr_ptycho/`, and `scripts/`; run all commands below from that directory.

The saved simulation used Python 3.12.3. Create a virtual environment with Python 3.12:

```bash
python -m venv .venv
```

On Windows with the Python Launcher installed, you can also use `py -3.12 -m venv .venv`. Activate the environment for your shell:

| Shell | Activation command |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

Install the simulation and plotting dependencies:

```bash
python -m pip install -r requirements-simulation.txt
```

This file pins NumPy 1.26.4, SciPy 1.14.1, Matplotlib 3.8.4, and scikit-image 0.25.2. The simulations use this project's NumPy mPIE and do not require PtyLab. The saved `meta.json` files retain the original run's software versions, machine information, seeds, and scientific parameters. Machine information is a provenance record and does not need to be replaced with paths on your new device.

## 2. Verify the saved data

```bash
python scripts/verify_repository_data.py
```

The verifier reads [REPOSITORY_DATA.json](REPOSITORY_DATA.json) and checks the sizes and SHA-256 hashes of all 42 listed files. The manifest covers saved simulation data, plotting inputs, and published figures; the reference paper PDF is excluded. Verification does not launch reconstruction or repeat the 250-iteration runs. Verify these files before continuing analysis. If a check fails, first check whether the repository was fully pulled and whether any saved files were modified.

Both `outputs/paper_style/full/` and `outputs/paper_style/quick/` contain these five files:

| File | Contents available for further work |
|---|---|
| `DATA_CONTRACT.md` | All fields, array shapes, units, methods, and evaluation definitions |
| `meta.json` | Simulation and reconstruction settings, seeds, assumptions, versions, timings, and failure records (the recorded runs also contained experiments that have since been removed, which `meta.json` still lists) |
| `bit_sweep.csv` | Metrics and statuses for all saved bit-depth sweep runs |
| `noise_sweep.csv` | Metrics and statuses for all saved noise sweep runs |
| `cameraman_8bit.npz` | Representative reconstructions for four methods: complex objects and probes, ground truth, aligned ROI, convergence, and FRC |

The full profile uses 250 iterations and camera seeds 0/1/2. Its bit-depth and noise CSV files contain 120 and 108 rows, respectively, with an empty failure list. The Fig. 2 (d) images and the representative reconstruction arrays use the first seed. All sweep runs are retained as metric tables, but complete object arrays are not saved for every sweep run. The quick profile is a small 20-iteration configuration for checking the workflow; scientific conclusions use full.

`docs/paper_style/data/` contains convenient copies of the CSV/JSON files. Complete figure regeneration also requires `cameraman_8bit.npz`; use `outputs/paper_style/full/` directly. The reference paper is linked by [DOI: 10.1109/TIM.2024.3363788](https://doi.org/10.1109/TIM.2024.3363788); the PDF is not distributed with the repository and is not needed to regenerate the saved figures.

## 3. Regenerate existing figures directly

The four statistical figure sets for the historical fixed-baseline experiment require only `docs/summary_metrics.csv` and `docs/experiment_manifest.json`:

```bash
python scripts/plot_paper_comparisons.py --source docs --output outputs/restored_paper_comparisons
```

The two paper-style simulation figure sets read only the complete full data:

```bash
python scripts/plot_paper_style_figures.py --source outputs/paper_style/full --output outputs/paper_style/figures_restored --summary-csv outputs/paper_style/figures_restored/summary.csv
```

Both commands produce PNG, SVG, and a figure-provenance manifest. New output directories preserve the published figures and saved source data. To view the existing figures, read [results.md](results.md) and [paper_style/summary.md](paper_style/summary.md).

## 4. Continue simulations in a new directory

```bash
python scripts/run_paper_style_simulation.py --profile quick --output outputs/paper_style/quick_new
python scripts/run_paper_style_simulation.py --profile full --seeds 0 1 2 --output outputs/paper_style/full_new
```

The second command runs the complete experiment. Control the process count with an option such as `--workers 2`. The recorded eight-process full run took approximately 15 minutes; runtime varies by device. Set the plotting command's `--source` to the new output directory to plot new results. The simulation script writes into the selected directory, so use a new directory for each run to avoid overwriting the repository's saved full/quick data.

The four representative reconstructions saved in each `cameraman_8bit.npz` include complete complex objects and probes, ground truth, and analysis arrays. They can directly support zoomed views, phase analysis, convergence, and FRC. For other bit-depth/noise sweep runs whose complete arrays were not saved, rerun the corresponding simulations using the recorded settings and seeds. The input objects and camera model are generated by repository code.

## 5. Limits of restoring the historical fixed-baseline experiment

The historical fixed-baseline experiment used external raw diffraction data, separately from the known-ground-truth simulations. Its `diff.npy`, `outputs/sweep_21x21_steps8_14_i80/best/best_reconstruction.npz`, and complete binary outputs for 81 reconstructions were already missing from the working directory. The repository retains summary metrics, images, settings, audits, and repeatability records, but has no backup from which to restore those raw arrays. Summary metrics alone cannot reconstruct them on another device.

If the historical inputs are later recovered from the original device or a backup, place them at the relative paths in `experiments/paper_baseline_comparison.json`. The original `diff.npy` used shape `(61, 61, 32, 32)`, float64 values, and axes `(scan_y, scan_x, det_y, det_x)`. Historical input and baseline SHA-256 hashes are retained in `docs/experiment_manifest.json` for checking their identity.

The historical PtyLab reconstruction pipeline requires a separate Python 3.11 environment with `requirements.txt`. That file retains the original experiment's versions and PtyLab Git revision; installation requires Git and network access. Keep the simulation/plotting and historical environments separate. PtyLab installation is not a prerequisite for restoring the saved simulation data. Historical source hashes in experiment manifests and audit records retain their values at generation time rather than being rewritten after later portability changes.

The complete historical workflow and evaluation limitations remain documented in the [README](../README.md#prepare-data-and-baseline) and [results report](results.md).
