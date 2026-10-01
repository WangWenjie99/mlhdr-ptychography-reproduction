# Key values for the paper-style comparisons

This page summarizes the saved simulation results in `outputs/paper_style/full/`. The comparison figures were regenerated and their explanations expanded using existing data, without rerunning the simulations. These data are accounted for separately from the [fixed-baseline experiment](../results.md#paper-style-comparisons-from-saved-results): this page uses known simulation ground truth, whereas the fixed-baseline experiment uses a saved raw-data reconstruction as its reference.

Complete values are in [summary.csv](summary.csv), and per-run sweep records in [bit_sweep.csv](data/bit_sweep.csv) and [noise_sweep.csv](data/noise_sweep.csv). Figure and source hashes are recorded in [figures_manifest.json](../figures/paper_style/figures_manifest.json). The tables map methods using the source CSV's `method` field and display saved metrics; no values are read from PNG files. Table values are rounded from the data: PSNR to 0.01 dB, SSIM to four decimals, and NRMSE to four significant figures.

## Cameraman: 8-bit results and the 16-bit single-exposure reference

The sweep uses seeds 0, 1, and 2, with three successful runs per condition. The table gives means; figure error bars show sample standard deviations. PSNR, SSIM, and amplitude NRMSE are evaluated on the simulation ground-truth ROI after compensating for a global complex factor and subpixel translation. Single exposure is fixed at 10 ms, while HDR uses seven exposures with total integration of 666.5 ms. The acquisition budgets differ.

| Method (CSV label) | Bit depth | PSNR ↑ (dB) | SSIM ↑ | NRMSE ↓ | Statistic / reference |
|---|---:|---:|---:|---:|---|
| Single exposure `single` | 8 | 20.56 | 0.4747 | 0.1228 | Three-run mean / simulation ground truth |
| LRFC-HDR `lrfc` | 8 | 27.88 | 0.8214 | 0.05293 | Three-run mean / simulation ground truth |
| Paper Eqs. (14)–(15) `ml_eq14_15` | 8 | 16.65 | 0.5355 | 0.1928 | Three-run mean / simulation ground truth |
| Saturation-masking extension `ml_masked` | 8 | 27.82 | 0.8222 | 0.05330 | Three-run mean / simulation ground truth |
| Single-exposure reference `single` | 16 | 32.31 | 0.9152 | 0.03177 | Three-run mean / simulation ground truth |

Relative to the 16-bit single exposure, the 8-bit original paper equations have a PSNR difference of **−15.66 dB**; LRFC and the extension give **−4.43 and −4.49 dB**, respectively (SSIM differences −0.380, −0.094, and −0.093). None of the 8-bit methods reaches the 16-bit single-exposure reference in this simulation.

The paper-equation implementation includes nonnegativity and division-by-zero safeguards and retains saturated observations. Its mean 8-bit diffraction NRMSE is approximately 0.8201, compared with approximately 0.00339 and 0.00375 for LRFC and the extension. LRFC follows the paper's linear-response relationship; choosing the longest unsaturated exposure per pixel is this project's implementation choice. Full-well saturation loses bright-region observations, and increasing ADC bit depth alone cannot recover those counts. Saturation masking is an additional project control, not part of the paper's original algorithm.

## Noise sweep: the 54 dB endpoint

The project defines `SNR_dB = 20·log₁₀(full_well / σ_read)`. At 54 dB, read noise is approximately 4988 e⁻; larger dB values mean less noise. The paper does not define its noise axis in dB; this conversion is the project's choice. The table again gives three-seed means referenced to simulation ground truth.

| Method (CSV label) | Bit depth | PSNR ↑ (dB) | SSIM ↑ | NRMSE ↓ | Statistic / reference |
|---|---:|---:|---:|---:|---|
| Single exposure `single` | 16 | 10.05 | 0.0284 | 0.4121 | Three-run mean / simulation ground truth |
| LRFC-HDR `lrfc` | 8 | 25.74 | 0.5870 | 0.06766 | Three-run mean / simulation ground truth |
| Paper Eqs. (14)–(15) `ml_eq14_15` | 8 | 15.19 | 0.2055 | 0.2279 | Three-run mean / simulation ground truth |
| Saturation-masking extension `ml_masked` | 8 | 20.37 | 0.3366 | 0.1256 | Three-run mean / simulation ground truth |

Overall PSNR improves as noise decreases. The original equations have lower PSNR and higher NRMSE than LRFC at every noise setting. The extension is also below LRFC at high dB values, and its SSIM does not continue improving between 48 and 54 dB. Masking saturated observations does not guarantee the best result under every condition.

## Parameter scope

This project's own geometry choices (a 226×226 object, a 64×64 probe window and detector with a 32 px probe diameter, and 8 px steps with 10% random perturbation) differ from the paper's simulation; it uses 20×20 scan positions and 250 iterations. It adopts the paper's experimental wavelength of 632.8 nm and distance of 13.9 mm. The paper does not fully specify simulation exposure timing, numerical read noise, and several implementation details. Camera read noise, dark current, noise-dB conversion, and simulation exposure scheduling are explicit project choices; the seven exposures follow the paper's transmission experiment. All project parameters and software versions are in [meta.json](data/meta.json); because the recorded run also contained experiments that have since been removed from this project, that file still lists their parameters as well.

The project's amplitude error is object-amplitude NRMSE against the simulation ground truth (the paper does not define its RMS calculation). The figures and tables compare this project's four methods with each other only; they contain no values taken from the paper.

All 120 saved bit-depth rows and 108 noise rows have status `ok`, and the full metadata's failure list is empty. Source files, copied CSV/JSON data, plotting code, and figure hashes are checked through the provenance manifests; plotting reads only the saved results. See the [results documentation](../results.md#paper-style-simulation-comparisons) for each figure's interpretation and limitations.
