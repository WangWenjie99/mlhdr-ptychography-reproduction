# 换设备恢复与继续工作

本次提交把当前保存的论文风格仿真数据随代码一起纳入 Git。正常克隆或拉取仓库后，可以直接查看结果、重绘全部 10 组对比图、读取重建数组继续分析，或在新目录开展仿真；不需要从原设备另外拷贝本次仿真文件。

## 1. 获取仓库与安装环境

使用远程仓库的克隆地址运行 `git clone <仓库地址>`，或在已有副本中运行 `git pull`。随后进入含 `README.md`、`mlhdr_ptycho/` 和 `scripts/` 的仓库根目录；所有下述命令在此目录执行。

本次仿真实际使用 Python 3.12.3。用 Python 3.12 创建虚拟环境：

```bash
python -m venv .venv
```

Windows 安装了 Python Launcher 时也可使用 `py -3.12 -m venv .venv`。按所用终端激活环境：

| 终端 | 激活命令 |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

然后安装本次仿真与绘图依赖：

```bash
python -m pip install -r requirements-simulation.txt
```

该文件锁定 NumPy 1.26.4、SciPy 1.14.1、Matplotlib 3.8.4 和 scikit-image 0.25.2。本次仿真使用本项目的 NumPy mPIE，不需要 PtyLab。保存数据中的 `meta.json` 保留原运行的软件版本、机器信息、种子及科学参数；那些机器信息是来源记录，无需改成新设备路径。

## 2. 验证保存数据

```bash
python scripts/verify_repository_data.py
```

验证脚本读取 [REPOSITORY_DATA.json](REPOSITORY_DATA.json)，检查纳入清单文件的大小及 SHA-256，并读取 full / quick 仿真数据验证契约。清单覆盖已保存仿真数据、绘图输入、发布图和论文 PDF。无需启动重建或重新计算 250 次迭代。验证通过后再基于这些文件开展后续分析；若有差异，应先检查是否完整拉取以及是否改动了保存数据。

两个目录 `outputs/paper_style/full/` 和 `outputs/paper_style/quick/` 各包含以下 9 个文件：

| 文件 | 可用于继续工作的内容 |
|---|---|
| `DATA_CONTRACT.md` | 全部字段、数组形状、单位、方法和评价定义 |
| `meta.json` | 仿真与重建配置、种子、假设、版本、运行时间和失败记录 |
| `bit_sweep.csv` | 位深扫描所有已保存运行的指标与状态 |
| `noise_sweep.csv` | 噪声扫描所有已保存运行的指标与状态 |
| `cameraman_8bit.npz` | 4 种方法的代表重建、复数物体和探针、真值、对齐 ROI、收敛与 FRC |
| `usaf_8bit.npz` | 5 个代表重建（含 16 bit 单曝光参考），以及 USAF 几何、可分辨判定与剖面 |
| `usaf_16bit.npz` | 4 种方法的代表重建及 USAF 分析数组 |
| `diffraction_example.npz` | 同一位置的 7 档原始曝光、暗场统计、真值和各融合结果 |
| `resolution_summary.json` | 代表重建的指标、FRC 截止及 USAF 元素汇总 |

full 使用 250 次迭代、相机种子 0 / 1 / 2，位深和噪声 CSV 分别有 120 / 108 行，失败记录为空。图像、剖面和代表重建数组采用第一个种子；位深和噪声扫描的全部运行保存为指标表，并非每个扫描运行都保存了完整物体数组。quick 是 20 次迭代的小型配置，用于检查流程，科学结论使用 full。

`docs/paper_style/data/` 是 CSV / JSON 的便捷副本；完整重绘还需要上述 NPZ，直接使用 `outputs/paper_style/full/` 即可。`docs/paper_style/paper_digitized.json` 保存论文图 2 / 3 数字化曲线与近似点标记。仓库根目录还包含阅读使用的论文 PDF。

## 3. 直接重绘现有结果

历史固定基线的 4 组统计图只需 `docs/summary_metrics.csv` 与 `docs/experiment_manifest.json`：

```bash
python scripts/plot_paper_comparisons.py --source docs --output outputs/restored_paper_comparisons
```

本次仿真的 6 组论文风格图读取完整 full 数据及论文数字化曲线：

```bash
python scripts/plot_paper_style_figures.py --source outputs/paper_style/full --output outputs/paper_style/figures_restored --summary-csv outputs/paper_style/figures_restored/summary.csv
```

两条命令均输出 PNG、SVG 和图表来源清单。新的输出目录让已发布图与已保存源数据继续保留。若只需查看已有图，可直接阅读 [results.md](results.md) 和 [paper_style/summary.md](paper_style/summary.md)。

## 4. 在新目录继续仿真

```bash
python scripts/run_paper_style_simulation.py --profile quick --output outputs/paper_style/quick_new
python scripts/run_paper_style_simulation.py --profile full --seeds 0 1 2 --output outputs/paper_style/full_new
```

第二条命令为完整实验。可通过 `--workers 2` 等设置控制进程数；记录的 8 进程 full 运行耗时约 15 分钟，运行速度随设备变化。将绘图命令的 `--source` 改成新输出目录即可绘制新结果。仿真脚本会写入指定目录，因此每次使用新目录，避免覆盖仓库保存的 full / quick 数据。

已有 13 个代表重建提供完整复数物体与探针、真值和分析数组，可直接用于局部放大、相位分析、收敛、FRC 及 USAF 剖面等后续工作。对位深 / 噪声扫描里未保存完整数组的其他运行，需要根据保存配置与种子再运行相应仿真；输入物体和相机模型均由仓库代码生成。

## 5. 历史固定基线的恢复边界

历史固定基线实验用的是外部原始衍射数据，与本次已知真值仿真不同。其 `diff.npy`、`outputs/sweep_21x21_steps8_14_i80/best/best_reconstruction.npz` 和 81 次重建的完整二进制输出在本次工作目录中已经缺失。仓库保留了该实验的汇总指标、图像、设置、审计及重复性记录，但没有可恢复这批原始数组的备份；换设备也不能仅凭汇总指标重建它们。

如之后从原设备或备份找回历史输入，应按 `experiments/paper_baseline_comparison.json` 的相对路径放置文件。原始 `diff.npy` 形状为 `(61, 61, 32, 32)`、float64，按 `(scan_y, scan_x, det_y, det_x)` 排列。历史输入与基线的 SHA-256 保存在 `docs/experiment_manifest.json`，可核对是否确为当时的数据。

历史 PtyLab 重建链路需另建 Python 3.11 环境并安装 `requirements.txt`；该文件保留原实验的版本和 PtyLab Git 提交，安装需要 Git 与网络。本次仿真 / 绘图环境和历史环境分别使用，不将 PtyLab 安装作为本次数据恢复的前提。历史 `manifest` 和审计记录中的源码哈希保留生成时的值，不会因后续可移植性修复而改写。

完整历史实验步骤及评价限制仍见 [README](../README.md#准备数据与基线) 和 [结果报告](results.md)。
