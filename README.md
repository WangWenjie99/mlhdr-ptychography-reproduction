# ML-HDR Ptychography Reproduction

基于 Python 与 PtyLab 的 ML-HDR 叠层衍射成像方法复现和对照实验。参考 Liu 等人在 IEEE TIM 2024 发表的论文，在固定重建参数下比较原始数据基线、不同曝光时间的单曝光重建，以及论文式（14）—（15）的融合结果。

**当前结论：在本项目的数据与已记录的相机假设下，论文原式未表现出稳定优于最佳单曝光的效果。加入饱和像素屏蔽的额外对照明显改善了结果，但该扩展不属于论文原算法。** 本仓库同时保留两种方法及其区别，不将扩展的收益归于论文原式。

[实验结果与限制](docs/results.md) · [参数及来源](experiments/paper_baseline_comparison.json) · [完整指标表](docs/summary_metrics.csv) · [结果审计记录](docs/validation_record.json)

![基线、最佳单曝光、论文原式与附加饱和处理](docs/figures/low_noise_extension.png)

## 仓库内容

本仓库发布代码、说明、汇总指标和主要对比图。原始 `diff.npy`、重建数组、全部 `outputs/`、相机模拟原始帧和本地论文 PDF **未上传**。

| 路径 | 用途 |
|---|---|
| `mlhdr_ptycho/data.py` | 衍射数据读取、中心裁剪、扫描坐标与归一化 |
| `mlhdr_ptycho/ml_hdr.py` | 初始相机模拟与论文融合公式 |
| `mlhdr_ptycho/paper_reproduction.py` | 明确记录物理量的相机模拟、融合与比较指标 |
| `mlhdr_ptycho/ptylab_reconstruction.py` | PtyLab mPIE 重建封装与结果导出 |
| `scripts/reproduce_paper.py` | 固定基线的完整对照实验入口 |
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

## 环境安装

已验证环境为 Python 3.11、NumPy 1.26.4、SciPy 1.11.4 和 PtyLab 0.2.1。PtyLab 固定到实验使用的 Git 提交 `2a7cdefe536f3976b5c6596fcd4e72b1f513f304`，其他主要依赖记录在 `requirements.txt`。

在仓库根目录运行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。后续命令均使用已安装依赖的环境中的 `python`。依赖文件记录本次实验使用的版本；新环境的完整安装尚未在其他操作系统上验证。

## 准备数据与基线

完整实验需要以下本地文件；它们未包含在公开仓库中：

1. `diff.npy`：四维非负衍射强度，轴顺序为 `(scan_y, scan_x, det_y, det_x)`。已记录实验的数据形状为 `(61, 61, 32, 32)`，原始类型为 float64，数值范围 `[0, 1]`。
2. 配置中指定的原始基线存档：`outputs/sweep_21x21_steps8_14_i80/best/best_reconstruction.npz`。历史存档来自本地项目；若需要获得与发布图像对应的原始数据与存档，请联系仓库维护者。

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

已发布实验通过 9 项科学测试和 532 项结果审计。公开仓库只包含审计记录，不包含重新执行该次完整审计所需的所有二进制产物。

相机测量和融合输入可精确重现；当前安装的重建器即使初始化及 NumPy 随机序列一致，数值输出仍可能不逐位一致。详见 [重复性记录](docs/solver_repeatability.json)。

## 方法边界

- 论文给出泊松光子散粒噪声、泊松暗电流和高斯读出噪声、`2.5×10⁶` 电子满阱及每档 20 张暗场，但没有给出本实验可直接照搬的读出噪声、暗电流数值或噪声 dB 换算。
- 本项目采用的具体数值、`10⁹ photons/s` 到当前数据的映射、量子效率和噪声敏感性场景均有显式记录。这是将论文方法应用到本地数据的受控对照，不是原文全部图表的逐点复制。
- `paper_ml_hdr` 保留论文原式对饱和计数的处理；`saturation_mask_extension` 另行排除饱和观测，明确属于附加对照。
- 汇总振幅图使用相同色阶，仅补偿一个整体振幅比例；单独导出的振幅与相位图各自带有色条。
- 尚未完成 LRFC-HDR、16-bit 对照和真实物体的 FRC 分辨率验证。

## 参考文献与许可

Li Liu, Wenjie Li, Ming Gong, Lei Zhong, Honggang Gu, Shiyuan Liu. **Resolution-Enhanced Lensless Ptychographic Microscope Based on Maximum-Likelihood High-Dynamic-Range Image Fusion.** IEEE Transactions on Instrumentation and Measurement, vol. 73, 2024. [DOI: 10.1109/TIM.2024.3363788](https://doi.org/10.1109/TIM.2024.3363788).

重建依赖 [PtyLab/PtyLab.py](https://github.com/PtyLab/PtyLab.py)。本项目没有复制论文 PDF 或 PtyLab 源码；第三方材料遵循各自许可。当前仓库尚未指定本项目代码的开源许可证。
