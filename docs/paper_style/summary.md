# Key values for the paper-style comparisons

This page summarizes the saved simulation results in `outputs/paper_style/full/`. The comparison figures were regenerated and their explanations expanded using existing data, without rerunning the simulations. These data are accounted for separately from the [fixed-baseline experiment](../results.md#paper-style-comparisons-from-saved-results): this page uses known simulation ground truth, whereas the fixed-baseline experiment uses a saved raw-data reconstruction as its reference.

Complete values are in [summary.csv](summary.csv), per-run sweep records in [bit_sweep.csv](data/bit_sweep.csv) and [noise_sweep.csv](data/noise_sweep.csv), and single-run USAF results in [resolution_summary.json](data/resolution_summary.json). Figure and source hashes are recorded in [figures_manifest.json](../figures/paper_style/figures_manifest.json). The tables map methods using the source CSV's `method` field and display saved metrics; no values are read from PNG files.

## Cameraman: 8-bit results and the 16-bit single-exposure reference

The sweep uses seeds 0, 1, and 2, with three successful runs per condition. The table gives means; figure error bars show sample standard deviations. PSNR, SSIM, and amplitude NRMSE are evaluated on the simulation ground-truth ROI after compensating for a global complex factor and subpixel translation. Single exposure is fixed at 10 ms, while HDR uses seven exposures with total integration of 666.5 ms. The acquisition budgets differ.

| Method (CSV label) | Bit depth | PSNR ↑ (dB) | SSIM ↑ | NRMSE ↓ | Statistic / reference |
|---|---:|---:|---:|---:|---|
| Single exposure `single` | 8 | 20.56 | 0.4747 | 0.12280 | Three-run mean / simulation ground truth |
| LRFC-HDR `lrfc` | 8 | 27.88 | 0.8214 | 0.05293 | Three-run mean / simulation ground truth |
| Paper Eqs. (14)–(15) `ml_eq14_15` | 8 | 16.65 | 0.5355 | 0.19280 | Three-run mean / simulation ground truth |
| Saturation-masking extension `ml_masked` | 8 | 27.82 | 0.8222 | 0.05330 | Three-run mean / simulation ground truth |
| Single-exposure reference `single` | 16 | 32.31 | 0.9152 | 0.03177 | Three-run mean / simulation ground truth |

The original 8-bit paper equations have a PSNR difference of **−15.66 dB** relative to the 16-bit single exposure; LRFC and the extension give **−4.43 and −4.49 dB**, respectively. This project therefore does not reproduce the paper's comparison in which 8-bit ML-HDR matches or exceeds 16-bit single exposure. The digitized Fig. 2 curves give a difference of approximately **+3.5 dB** for the original equations. Digitization precision is approximately ±0.5 dB and is unsuitable for point-by-point error evaluation.

The paper-equation implementation includes nonnegativity and division-by-zero safeguards and retains saturated observations. Its mean 8-bit diffraction NRMSE is approximately 0.8201, compared with approximately 0.00339 and 0.00375 for LRFC and the extension. LRFC follows the paper's linear-response relationship; choosing the longest unsaturated exposure per pixel is this project's implementation choice. Full-well saturation loses bright-region observations, and increasing ADC bit depth alone cannot recover those counts. Saturation masking is an additional project control, not part of the paper's original algorithm.

## Noise sweep: the 54 dB endpoint

The project defines `SNR_dB = 20·log₁₀(full_well / σ_read)`. At 54 dB, read noise is approximately 4988 e⁻; larger dB values mean less noise. The paper does not specify this conversion, so endpoint values cannot be directly compared at nominally equal paper-axis values. The table again gives three-seed means referenced to simulation ground truth.

| Method (CSV label) | Bit depth | PSNR ↑ (dB) | SSIM ↑ | NRMSE ↓ | Statistic / reference |
|---|---:|---:|---:|---:|---|
| Single exposure `single` | 16 | 10.05 | 0.0284 | 0.41214 | Three-run mean / simulation ground truth |
| LRFC-HDR `lrfc` | 8 | 25.74 | 0.5870 | 0.06766 | Three-run mean / simulation ground truth |
| Paper Eqs. (14)–(15) `ml_eq14_15` | 8 | 15.19 | 0.2055 | 0.22791 | Three-run mean / simulation ground truth |
| Saturation-masking extension `ml_masked` | 8 | 20.37 | 0.3366 | 0.12562 | Three-run mean / simulation ground truth |

Overall PSNR improves as noise decreases, but the original equations do not show the advantage over LRFC reported in Fig. 3. The extension is also below LRFC at high dB values, and its SSIM does not continue improving between 48 and 54 dB. Masking saturated observations does not guarantee the best result under every condition.

## USAF: single reconstructions with seed 0

The following quality metrics, images, profiles, and FRC values all come from seed 0, rather than three-seed means. The smallest element must be resolvable in both orientations along with every coarser element. FRC compares reconstructed amplitude with ground truth; these are distinct evaluation criteria.

| Method (CSV label) | Bit depth | PSNR ↑ (dB) | SSIM ↑ | NRMSE ↓ | Statistic / reference |
|---|---:|---:|---:|---:|---|
| Single exposure `single` | 8 | 10.50 | 0.3578 | 0.29000 | Seed 0 / simulation ground truth |
| LRFC-HDR `lrfc` | 8 | 17.37 | 0.7879 | 0.13160 | Seed 0 / simulation ground truth |
| Paper Eqs. (14)–(15) `ml_eq14_15` | 8 | 5.26 | 0.2871 | 0.53030 | Seed 0 / simulation ground truth |
| Saturation-masking extension `ml_masked` | 8 | 16.84 | 0.7854 | 0.13980 | Seed 0 / simulation ground truth |
| Single exposure `single` | 16 | 20.66 | 0.8892 | 0.09011 | Seed 0 / simulation ground truth |
| LRFC-HDR `lrfc` | 16 | 50.78 | 0.9988 | 0.002810 | Seed 0 / simulation ground truth |
| Paper Eqs. (14)–(15) `ml_eq14_15` | 16 | 10.86 | 0.5085 | 0.27830 | Seed 0 / simulation ground truth |
| Saturation-masking extension `ml_masked` | 16 | 51.47 | 0.9988 | 0.002593 | Seed 0 / simulation ground truth |

| Method | Bit depth | Smallest continuously resolvable element | Line width (μm) | Ground-truth FRC half-period (μm) |
|---|---:|---|---:|---:|
| Single exposure | 8 | G7 E1 | 3.906 | 2.025 |
| LRFC-HDR | 8 | G8 E6 | 1.096 | 1.125 |
| Original paper equations | 8 | G7 E1 | 3.906 | 2.420 |
| Saturation-masking extension | 8 | G8 E6 | 1.096 | 1.034 |
| Single exposure | 16 | G9 E1 | 0.977 | 0.799 |
| LRFC-HDR | 16 | G9 E4 | 0.691 | 0.571 (Nyquist) |
| Original paper equations | 16 | G8 E2 | 1.740 | 1.388 |
| Saturation-masking extension | 16 | G9 E4 | 0.691 | 0.571 (Nyquist) |

The pixel-sampled USAF ground truth itself is continuously resolvable only through G9 E4. The 16-bit LRFC and extension results reach this simulation limit. The 0.571 μm value is the limit set by the current pixel size and ground-truth-referenced FRC, not a measured instrument resolution. Figs. 6 and 7 of the paper show real experiments. The paper does not fully describe its FRC inputs; this project's ground-truth-referenced FRC is not directly equivalent to that experimental evaluation.

Under the project's plateau rule, 8-bit LRFC and the extension first meet the convergence criterion at 0.572 s and 0.447 s, both at iteration 5. Their complete 250-iteration runs take approximately 27.6 s and 24.1 s. Single exposure and the original equations do not meet the required error-reduction threshold, so no plateau time is reported. Plateau time is different from total computation time and cannot be directly compared with the paper's 225 s / 360 s.

## Diffraction display and parameter scope

For scan position 190 (seed 0) in Fig. 5, saturated-pixel fractions across the seven exposures are **0%, 0%, 0%, 0%, 0.1221%, 0.2197%, and 0.3418%**, corresponding to 0.5, 1, 5, 10, 50, 100, and 500 ms. These percentages equal the frame-level fractions in [meta.json](data/meta.json) multiplied by 100. HDR panels show fused rates multiplied by 500 ms as equivalent counts, which can exceed the original 8-bit ADC limit of 255. All ten panels share a `log₂(1 + counts)` color scale.

The paper specifies a 1024×1024 object, 256×256 probe and detector, 20×20 scan positions, approximately 40 px steps with 10% random perturbation, 50 mm propagation distance, and 250 iterations. This project uses a 226×226 object, 64×64 probe window and detector (32 px probe diameter), and 8 px steps with 10% perturbation. It adopts the paper's experimental wavelength of 632.8 nm and distance of 13.9 mm and retains 20×20 positions and 250 iterations. The paper does not fully specify simulation exposure timing, numerical read noise, and several implementation details. Camera read noise, dark current, noise-dB conversion, and simulation exposure scheduling are explicit project choices; the seven exposures follow the paper's transmission experiment. All project parameters and software versions are in [meta.json](data/meta.json).

The project's amplitude error is NRMSE. The paper does not explicitly define its RMS calculation, so the two cannot be treated as numerically equivalent. Geometry, sampling, and camera assumptions differ between the paper and project; overlaid curves compare trends and method rankings.

All 120 saved bit-depth rows and 108 noise rows have status `ok`, and the full metadata's failure list is empty. Source files, copied CSV/JSON data, plotting code, and figure hashes are checked through the provenance manifests; plotting reads only the saved results. Digitized paper points and approximate flags are in [paper_digitized.json](paper_digitized.json). See the [results documentation](../results.md#paper-style-simulation-comparisons) for each figure's interpretation and limitations.
