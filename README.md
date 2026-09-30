# ML-HDR Ptychography Reproduction

基于 Python 与 PtyLab 的 ML-HDR 叠层衍射成像方法复现和对照实验。参考 Liu 等人在 IEEE TIM 2024 发表的论文，在固定重建参数下比较原始数据基线、不同曝光时间的单曝光重建，以及论文式（14）—（15）的融合结果。

**当前结论：在本项目的数据与已记录的相机假设下，论文原式未表现出稳定优于最佳单曝光的效果。加入饱和像素屏蔽的额外对照明显改善了结果，但该扩展不属于论文原算法。** 本仓库同时保留两种方法及其区别，不将扩展的收益归于论文原式。

[实验结果与限制](docs/results.md) · [换设备恢复与继续工作](docs/REPOSITORY_RESTORE.md) · [参数及来源](experiments/paper_baseline_comparison.json) · [完整指标表](docs/summary_metrics.csv) · [结果审计记录](docs/validation_record.json)

新增四组论文式结果对比图：全曝光曲线、全方法热力图、固定方法跨噪声比较，以及衍射误差与饱和诊断。图表直接读取已保存的 27 行汇总指标，覆盖全部 81 次重建，不需要重新运行算法。它们参考论文图 2、图 3 的多方法比较方式；数据仍来自本项目的固定基线实验，没有新增 LRFC-HDR、位深扫描或真实分辨率结果。

![全部方法在三个噪声场景下的定量对比](docs/figures/paper_comparisons/method_heatmaps.png)

![基线、最佳单曝光、论文原式与附加饱和处理](docs/figures/low_noise_extension.png)

另有一组**论文风格仿真对比图**，按论文图 2、3、5、6、7 的版式绘制位深扫描、噪声扫描、多曝光衍射图和 USAF 分辨率结果，并把论文图中数字化的曲线与本项目结果叠加比较。数据来自已保存的已知真值仿真（含 LRFC-HDR、8/16 bit 对照和相对真值的 FRC）。论文报告了物体与探测器尺寸、扫描和迭代设置；本项目保留 400 个位置和 250 次迭代，另选较小的图像、探测器和扫描步长，并明确记录未由论文确定的相机与曝光假设。详见[论文风格仿真对比图](docs/results.md#论文风格仿真对比图)与[关键数值对照](docs/paper_style/summary.md)。

## 仓库内容

本仓库包含代码、说明、汇总指标、主要对比图、参考论文 PDF，以及当前保存的论文风格仿真 `outputs/paper_style/full/` 和 `quick/` 全部数据。两个仿真目录各有 9 个文件，包括真值、代表重建复数组、收敛与 FRC、USAF 几何与剖面、衍射示例及数据契约；换设备拉取后可直接重绘和继续分析。

历史固定基线实验的原始 `diff.npy`、基线存档和 81 次重建完整输出在本次工作目录中已经缺失，因此仓库只保留该实验的既有汇总指标、图像与审计记录。它们与本次完整保存的仿真数据分别统计；不能仅凭历史汇总文件重跑原始数据实验。

| 路径 | 用途 |
|---|---|
| `mlhdr_ptycho/data.py` | 衍射数据读取、中心裁剪、扫描坐标与归一化 |
| `mlhdr_ptycho/ml_hdr.py` | 初始相机模拟与论文融合公式 |
| `mlhdr_ptycho/paper_reproduction.py` | 明确记录物理量的相机模拟、融合与比较指标 |
| `mlhdr_ptycho/ptylab_reconstruction.py` | PtyLab mPIE 重建封装与结果导出 |
| `scripts/reproduce_paper.py` | 固定基线的完整对照实验入口 |
| `scripts/plot_paper_comparisons.py` | 仅使用已保存汇总指标生成四组 PNG / SVG 对比图 |
| `mlhdr_ptycho/comparison_figures.py` | 对比图数据校验、统一绘图与来源记录 |
| `mlhdr_ptycho/simulation.py` | 论文风格仿真：cameraman / USAF 物体、相机模型、四种融合方法 |
| `mlhdr_ptycho/mpie.py` | 纯 NumPy mPIE 重建（论文风格仿真使用） |
| `mlhdr_ptycho/resolution.py` | 对齐、PSNR / SSIM / NRMSE、相对真值的 FRC 与 USAF 可分辨判定 |
| `scripts/run_paper_style_simulation.py` | 论文风格仿真（位深、噪声、衍射、8/16 bit USAF）；输出格式见生成的 `DATA_CONTRACT.md` |
| `mlhdr_ptycho/paper_style_figures.py` | 论文风格图：数据校验、六组图、来源清单与关键数值表 |
| `scripts/plot_paper_style_figures.py` | 从仿真输出生成论文风格 PNG / SVG、`summary.csv` 并复制小型数据文件 |
| `docs/paper_style/` | 论文图数字化数值（`paper_digitized.json`）、关键数值对照与所用仿真数据（CSV / JSON） |
| `outputs/paper_style/full/` | 250 次迭代、3 种子的完整已保存仿真成果；六组论文风格图的直接来源 |
| `outputs/paper_style/quick/` | 20 次迭代的已保存冒烟成果；与 full 使用相同数据契约 |
| `requirements-simulation.txt` | 本次 Python 3.12 仿真与绘图环境，不包含 PtyLab |
| `docs/REPOSITORY_RESTORE.md` | 新设备安装、数据完整性检查、重绘与继续实验步骤 |
| `scripts/verify_repository_data.py` | 根据仓库数据清单验证已保存文件的大小、SHA-256 和数据契约 |
| `scripts/validate_paper_results.py` | 对完整本地实验产物进行审计 |
| `scripts/run_reconstruction.py` | 单次 raw / single / mlhdr 实验入口 |
| `scripts/inspect_diff.py` | 输入数据统计与衍射图预览 |
| `scripts/sweep_reconstruction_params.py` | 重建参数扫描 |
| `scripts/compare_single_exposures.py` | 早期单曝光对照脚本 |
| `experiments/` | 实验设置及论文给定值、补充假设的来源 |
| `tests/` | 公式、相机统计和指标验证 |
| `docs/` | 已完成实验的说明、指标及主要图像 |

## 已完成实验

所有比较使用同一基线配置：

| 参数 | 设置 |
|---|---|
| 扫描范围 | 中心 `21×21`，441 个扫描位置 |
| 探测器帧 | `32×32` 像素 |
| 扫描步长 | 8 个物面像素 |
| 初始探针 | `circ_smooth`，直径 31 个物面像素 |
| 初始物体 | `ones`，按 PtyLab 定义包含微小随机扰动 |
| 算法与迭代次数 | CPU mPIE，80 次 |
| 波长 / 物面像素 / 探测器像素 | 632.8 nm / 1 μm / 5.5 μm |
| 传播与更新顺序 | Fraunhofer，随机遍历，重建种子 0 |
| 校正 | 探针功率校正开启；位置校正关闭 |

七档曝光为 **0.5、1、5、10、50、100、500 ms**。三个噪声场景分别使用三个固定相机种子，共完成 **81 次重建**，另有基线复算。

低噪声场景的三次平均结果：

| 方法 | 相对基线 PSNR ↑ | 相对基线 SSIM ↑ |
|---|---:|---:|
| 最佳单曝光：1 ms | 19.15 dB | 0.4135 |
| `paper_ml_hdr`：论文式（14）—（15） | 14.44 dB | 0.0589 |
| `saturation_mask_extension`：额外饱和处理 | 28.52 dB | 0.8164 |

这些指标衡量与保存的原始数据重建基线的一致性；基线不是物体真值，不能据此声称复现了论文的分辨率提升。多曝光总积分时间为 666.5 ms/位置，本实验也不是等采集时间或等光子预算的比较。

## 直接生成论文式对比图

已有汇总指标足够生成定量图，无需 `diff.npy`、重建数组或运行 PtyLab。安装 Python、NumPy 和 Matplotlib 后，在仓库根目录运行：

```bash
python scripts/plot_paper_comparisons.py --source docs --output docs/figures/paper_comparisons
```

输出包括四组 PNG 与可编辑的矢量 SVG：`exposure_comparison`、`method_heatmaps`、`noise_comparison`、`diffraction_diagnostics`，以及记录输入来源与绘图规则的 `figures_manifest.json`。详见[图表解读](docs/results.md#论文式综合对比已有结果重绘)。

统计图显示三次相机噪声种子实验的均值与样本标准差；重建配置及重建种子保持一致。跨噪声图固定比较 1 ms、500 ms、论文原式和额外饱和屏蔽，避免把不同场景中重新选择的最佳曝光当成同一方法。现有图像快照展示相机种子 0，未由 PNG 反推数组、误差图或强度剖面。

## 论文风格仿真与对比图

这一组实验不依赖 `diff.npy` 或 PtyLab，只需 NumPy、SciPy、scikit-image 和 Matplotlib。已保存的 full 和 quick 数据随仓库提供，安装 `requirements-simulation.txt` 后可直接验证并重绘，无需先重新仿真：

```bash
python scripts/verify_repository_data.py
python scripts/plot_paper_style_figures.py --source outputs/paper_style/full --output outputs/paper_style/figures_restored --summary-csv outputs/paper_style/figures_restored/summary.csv
```

若继续开展新实验，使用新输出目录以保留已保存成果（`quick` 为 20 次迭代的冒烟配置；`full` 为 250 次迭代、位深 2–20 bit、噪声 6–54 dB 的完整配置，已记录的 8 个进程运行约 15 分钟）：

```bash
python scripts/run_paper_style_simulation.py --profile quick --output outputs/paper_style/quick_new
python scripts/run_paper_style_simulation.py --profile full --seeds 0 1 2 --output outputs/paper_style/full_new
```

输出写入 `outputs/paper_style/<profile>/`，其中的 `DATA_CONTRACT.md` 说明每个 CSV / NPZ / JSON 字段。随后生成论文风格图、关键数值表并把小型数据文件复制进 `docs/`：

```bash
python scripts/plot_paper_style_figures.py --source outputs/paper_style/full --output docs/figures/paper_style --summary-csv docs/paper_style/summary.csv --copy-data-to docs/paper_style/data
```

输出为六组 PNG（300 dpi）与 SVG：`fig2_bit_depth`、`fig3_noise`、`fig5_diffraction`、`fig6_usaf_8bit`、`fig7_usaf_16bit`、`paper_vs_reproduction`，以及记录源数据与代码 SHA-256 的 `figures_manifest.json`。绘图只读取仿真输出和 [`paper_digitized.json`](docs/paper_style/paper_digitized.json)（从论文图 2、图 3 人工读取的数值，精度约 ±0.5 dB / ±0.01，近似点单独标记），不从 PNG 反推任何数值。四种方法的颜色固定：黑色为单曝光，蓝色为 LRFC-HDR，红色为论文式（14）—（15）原式，绿色为本项目的饱和屏蔽扩展（**不属于论文算法**）。

## 环境安装

本次仿真与绘图使用 Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1、Matplotlib 3.8.4 和 scikit-image 0.25.2，版本记录在 [meta.json](outputs/paper_style/full/meta.json)。在新设备上用 Python 3.12 创建环境并安装：

```bash
python -m venv .venv
```

激活虚拟环境：Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`，Windows cmd 使用 `.venv\Scripts\activate.bat`，Linux / macOS 使用 `source .venv/bin/activate`。然后安装依赖：

```bash
python -m pip install -r requirements-simulation.txt
```

后续命令均使用该环境中的 `python`。完整步骤见[换设备恢复说明](docs/REPOSITORY_RESTORE.md)。

历史固定基线实验使用 Python 3.11、NumPy 1.26.4、SciPy 1.11.4 和 PtyLab 0.2.1。需要运行历史 PtyLab 链路时，另建 Python 3.11 环境；PtyLab 固定到实验使用的 Git 提交 `2a7cdefe536f3976b5c6596fcd4e72b1f513f304`，其他主要依赖记录在 `requirements.txt`。

在仓库根目录运行：

```bash
python3.11 -m venv .venv-ptylab
source .venv-ptylab/bin/activate
python -m pip install -r requirements.txt
```

Windows 可用 `py -3.11 -m venv .venv-ptylab` 创建环境，然后按所用终端激活。安装 PtyLab 的 Git 依赖还需要 Git 与网络。两个依赖文件分别记录对应实验的版本；新环境的完整安装尚未在其他操作系统上验证。

## 准备数据与基线

历史固定基线完整实验需要以下文件；它们在本次工作目录中已经缺失，因而未包含在仓库中：

1. `diff.npy`：四维非负衍射强度，轴顺序为 `(scan_y, scan_x, det_y, det_x)`。已记录实验的数据形状为 `(61, 61, 32, 32)`，原始类型为 float64，数值范围 `[0, 1]`。
2. 配置中指定的原始基线存档：`outputs/sweep_21x21_steps8_14_i80/best/best_reconstruction.npz`。历史存档来自原项目，需要从原设备或备份找回，才能重新运行与历史汇总对应的实验。

使用自己的数据时，可以先按相同设置建立新基线：

```bash
python scripts/inspect_diff.py --input diff.npy
python scripts/run_reconstruction.py --mode raw --scan-crop 21 --iterations 80 --scan-step-px 8 --probe-diameter-px 31 --initial-probe circ_smooth --seed 0 --output outputs/my_baseline
```

随后复制 `experiments/paper_baseline_comparison.json` 为 `experiments/my_comparison.json`，将其中的 `baseline_archive` 改为 `outputs/my_baseline/raw_reconstruction.npz`；如有需要，同步设置输入路径和相机假设。重新生成的基线并不等于本仓库历史实验的原始存档。

## 运行完整比较

已准备好历史数据与基线时：

```bash
python scripts/reproduce_paper.py --output outputs/paper_reproduction_repeat
```

使用自己的配置时：

```bash
python scripts/reproduce_paper.py --config experiments/my_comparison.json --output outputs/my_comparison
```

脚本拒绝覆盖已有输出目录。可使用 `--profiles low_noise --seeds 0` 先运行一个场景。每种方法使用同一份相机模拟测量，重建设置从选定的基线存档读取；没有使用基线物体或探针作为重建初始化。

输出包含 `index.html`、`REPORT.md`、逐次与汇总 CSV、相机测量与暗场、重建复数组、收敛曲线和对比图。第一个固定相机种子额外保存完整 PNG 与 PtyLab HDF5 文件。打开本地 `index.html` 查看报告；GitHub 上直接阅读 [结果说明](docs/results.md)。

## 验证

不需要原始数据即可运行科学单元测试：

```bash
python -B -m unittest discover -s tests -v
```

对已生成的**完整本地输出**运行审计：

```bash
python -B scripts/validate_paper_results.py outputs/my_comparison
```

历史固定基线实验通过 9 项科学测试和 532 项结果审计。仓库保留历史审计记录，但当前缺少重新执行该次完整审计所需的原始二进制产物。本次论文风格仿真数据完整性使用 `python scripts/verify_repository_data.py` 单独验证。

当前测试集共 47 项：上述 9 项科学测试、8 项对比图测试、13 项论文风格仿真测试和 17 项论文风格绘图测试。绘图测试使用临时生成的小型合成数据（含多种子与失败运行），不依赖 `outputs/`。

相机测量和融合输入可精确重现；当前安装的重建器即使初始化及 NumPy 随机序列一致，数值输出仍可能不逐位一致。详见 [重复性记录](docs/solver_repeatability.json)。

## 方法边界

- 论文给出泊松光子散粒噪声、泊松暗电流和高斯读出噪声、`2.5×10⁶` 电子满阱及每档 20 张暗场，但没有给出本实验可直接照搬的读出噪声、暗电流数值或噪声 dB 换算。
- 本项目采用的具体数值、`10⁹ photons/s` 到当前数据的映射、量子效率和噪声敏感性场景均有显式记录。这是将论文方法应用到本地数据的受控对照，不是原文全部图表的逐点复制。
- `paper_ml_hdr` 保留论文原式对饱和计数的处理；`saturation_mask_extension` 另行排除饱和观测，明确属于附加对照。
- 汇总振幅图使用相同色阶，仅补偿一个整体振幅比例；单独导出的振幅与相位图各自带有色条。
- 固定基线实验尚未完成 LRFC-HDR、16-bit 对照和真实物体的 FRC 分辨率验证。新增的论文风格仿真以已知真值补充了 LRFC-HDR、8/16 bit 与 FRC 的**仿真**对照；本项目采用真值参考 FRC，与论文实验评价不能直接等同，真实数据上的分辨率验证仍未完成。

## 参考文献与许可

Li Liu, Wenjie Li, Ming Gong, Lei Zhong, Honggang Gu, Shiyuan Liu. **Resolution-Enhanced Lensless Ptychographic Microscope Based on Maximum-Likelihood High-Dynamic-Range Image Fusion.** IEEE Transactions on Instrumentation and Measurement, vol. 73, 2024. [DOI: 10.1109/TIM.2024.3363788](https://doi.org/10.1109/TIM.2024.3363788).

历史重建依赖 [PtyLab/PtyLab.py](https://github.com/PtyLab/PtyLab.py)。仓库携带本次阅读使用的参考论文 PDF，并通过依赖文件引用 PtyLab；论文及第三方软件的权利与许可归原权利人，不能据此视为本项目授权。当前仓库尚未指定本项目代码的开源许可证。
