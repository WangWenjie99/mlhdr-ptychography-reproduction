#!/usr/bin/env python3
"""Run the fixed-baseline, multi-exposure paper reproduction and write a report."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mlhdr_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlhdr_ptycho.data import load_diff_npy, normalize_stack
from mlhdr_ptycho.paper_reproduction import (
    PaperCamera, amplitude_comparison, diffraction_comparison, paper_eq14_15,
    saturation_mask_extension, simulate_paper_camera, single_rate,
)
from mlhdr_ptycho.ptylab_reconstruction import (
    PtyLabConfig, coverage_roi, reconstruction_metrics, run_mpie,
    save_reconstruction_outputs,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_data(archive, crop, frame_shape, input_stack):
    # This is the user's trusted local baseline, which stores config as a pickle.
    with np.load(archive, allow_pickle=True) as data:
        config = PtyLabConfig(**data["config"].ravel()[0])
        obj, probe = data["object"].copy(), data["probe"].copy()
        encoder, error = data["encoder"].copy(), data["error"].copy()
    if len(encoder) != crop ** 2 or probe.shape != frame_shape:
        raise ValueError("Baseline scan or detector size does not match input crop")
    n = obj.shape[0]
    positions = np.rint(encoder / config.object_pixel_m).astype(int) + n // 2 - probe.shape[0] // 2
    coverage = np.zeros(obj.shape, np.uint16)
    for y, x in positions:
        coverage[y:y + probe.shape[0], x:x + probe.shape[1]] += 1
    roi = coverage_roi(coverage)
    if roi is None or not np.all(coverage[roi] > 0):
        raise ValueError("Expected a contiguous covered ROI for this baseline")
    return config, obj, probe, error, coverage, roi


def contact_sheet(path, panels, reference, title, difference=False):
    cols = 3 if len(panels) > 4 else len(panels)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False,
                             layout="constrained")
    lo, hi = np.percentile(np.abs(reference), [1, 99])
    # All panels share the baseline's fixed limits. Difference panels share 20%
    # of its amplitude range; values outside limits remain in numerical metrics.
    if difference:
        lo, hi = 0, 0.2 * float(np.ptp(np.abs(reference)))
    for ax, (name, field) in zip(axes.ravel(), panels):
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        if field is None:
            ax.text(.5, .5, "FAILED / NO SIGNAL", ha="center", transform=ax.transAxes)
            continue
        shown = np.abs(field - np.abs(reference)) if difference else field
        im = ax.imshow(shown, cmap="magma" if difference else "gray", vmin=lo, vmax=hi,
                       interpolation="nearest")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    if any(field is not None for _, field in panels):
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65,
                     label="Absolute amplitude difference" if difference else "Amplitude (one fitted scale)")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["profile"], row["case"]), []).append(row)
    result = []
    keys = ("baseline_psnr_db", "baseline_ssim", "baseline_nrmse", "diffraction_nrmse",
            "high_q_diffraction_nrmse", "final_error_per_frame", "saturation_fraction")
    for (profile, case), items in groups.items():
        row = {"profile": profile, "case": case, "runs": len(items),
               "successful_runs": sum(x["status"] == "ok" for x in items)}
        for key in keys:
            vals = [float(x[key]) for x in items if x.get(key) is not None
                    and np.isfinite(x[key]) and x["status"] == "ok"]
            row[key + "_mean"] = float(np.mean(vals)) if vals else None
            row[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
        result.append(row)
    return result


def metric_plot(path, summary, profile, exposure_ms):
    data = {r["case"]: r for r in summary if r["profile"] == profile}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, metric, label in zip(axes,
            ["baseline_psnr_db", "baseline_ssim", "diffraction_nrmse"],
            ["PSNR vs saved baseline (dB)", "SSIM vs saved baseline", "Diffraction NRMSE vs clean input"]):
        means = [data[f"single_{t:g}ms"].get(metric + "_mean") for t in exposure_ms]
        stds = [data[f"single_{t:g}ms"].get(metric + "_std") for t in exposure_ms]
        ax.errorbar(exposure_ms, means, yerr=stds, marker="o", color="#4464ad", label="Single exposure")
        for case, name, color, style in [
            ("paper_ml_hdr", "Published Eq.14-15", "#cc4444", "-"),
            ("saturation_mask_extension", "Saturation mask extension", "#16846c", "--"),
        ]:
            mean = data[case].get(metric + "_mean")
            std = data[case].get(metric + "_std")
            if mean is not None:
                ax.axhline(mean, color=color, linestyle=style, label=name)
                ax.axhspan(mean - std, mean + std, color=color, alpha=.12)
        ax.set_xscale("log")
        ax.set_xlabel("Single exposure time (ms)")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{profile}: mean +/- sample standard deviation; all reconstructions 80 iterations")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diffraction_plot(path, clean, rates, measurement):
    # Fixed center scan position, selected before seeing any reconstruction.
    index = len(clean) // 2
    panels = [("Clean input", clean[index])]
    panels += [(label, a[index] / measurement.count_rate_scale) for label, a in rates]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout="constrained")
    for ax, (label, a) in zip(axes.ravel(), panels):
        im = ax.imshow(np.log10(np.maximum(a, 1e-8)), vmin=-8, vmax=0, cmap="viridis")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="log10 relative intensity (common scale)")
    fig.suptitle("Center scan diffraction; known camera gain undone; no per-panel normalization")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(output, manifest, summary, baseline_check):
    conf = manifest["experiment"]
    main_profile = manifest["selected_profiles"][0]
    items = [r for r in summary if r["profile"] == main_profile]
    by_case = {r["case"]: r for r in items}
    singles = [r for r in items if r["case"].startswith("single_") and r["baseline_ssim_mean"] is not None]
    best = max(singles, key=lambda r: r["baseline_ssim_mean"]) if singles else None
    strict = by_case["paper_ml_hdr"]
    extension = by_case["saturation_mask_extension"]
    def number(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}"
    lines = [
        "# ML-HDR 论文方法复现与固定基线对照", "",
        "所有重建均使用用户给定基线的设置：21×21 扫描、32×32 衍射帧、步长 8 px、"
        "初始探针直径 31 px、circ_smooth / ones 初始化、80 次 CPU mPIE 迭代、随机种子 0。",
        "", "## 结论与主结果", "",
        "下表为主场景 `" + main_profile + "` 的多个相机噪声种子平均值。"
        "PSNR/SSIM 衡量与已保存基线重建的一致性，不能解释为真实物体精度或分辨率。",
        "", "| 方法 | PSNR / dB | SSIM | 振幅 NRMSE | 衍射强度 NRMSE | 成功次数 |",
        "|---|---:|---:|---:|---:|---:|",
        "| 已保存原始数据基线（自身参考） | ∞ | 1 | 0 | 0 | — |",
    ]
    for r in items:
        lines.append(f"| {r['case']} | {number(r['baseline_psnr_db_mean'],2)} | {number(r['baseline_ssim_mean'])} "
                     f"| {number(r['baseline_nrmse_mean'])} | {number(r['diffraction_nrmse_mean'])} "
                     f"| {r['successful_runs']}/{r['runs']} |")
    if best:
        lines += ["", f"单曝光中按平均 SSIM 最好的是 **{best['case']}**。"
                  f"严格论文公式的 SSIM 为 **{number(strict['baseline_ssim_mean'])}**，"
                  f"该单曝光为 **{number(best['baseline_ssim_mean'])}**。",
                  "严格公式" + ("在该指标上优于最佳单曝光。" if strict['baseline_ssim_mean'] > best['baseline_ssim_mean']
                               else "在该指标上没有优于最佳单曝光，不能声称已经复现论文报告的全面优势。"),
                  f"单独的饱和屏蔽扩展 SSIM 为 **{number(extension['baseline_ssim_mean'])}**。"
                  "它的收益不能归到论文原式名下。"]
    lines += ["", f"![严格论文公式与全部单曝光]({main_profile}/comparison_amplitude.png)",
              "", f"![附加饱和处理对照]({main_profile}/extension_comparison.png)",
              "", f"![指标与随机种子波动]({main_profile}/metrics_vs_exposure.png)",
              "", "## 论文要求、补充假设与差异", ""]
    for key, value in conf["provenance"].items():
        lines.append(f"- `{key}`：{value}")
    lines += ["", "论文：[Liu et al., IEEE TIM 2024](https://doi.org/10.1109/TIM.2024.3363788)。"
              "已核对本地 PDF 第 3 页式（14）—（15）、第 5 页仿真设置、第 7 页曝光时间。",
              "", "主场景继承旧项目的读出噪声 5 e−、暗电流 80 e−/s。"
              "另外两个场景将读出噪声设为 0.25 和 1 ADC count 对应的电子数，"
              "检查暗场方差能够被量化记录时的行为；这些值是公开的敏感性分析假设。",
              "", "光通量映射：用一个全局系数使最亮一帧的总探测光子率为 10⁹/s，"
              "保持扫描位置之间的相对能量。假定量子效率为 1。"
              "本文没有足够信息唯一确定这一映射，因此无法逐点复现原文图 2/3。",
              "", "## 实现与饱和诊断", "",
              "每档独立模拟 Poisson(信号率×曝光时间)、Poisson(暗电流×曝光时间)、"
              "N(0, 读出噪声标准差²)，再进行满阱裁剪和 ADC 四舍五入。"
              "每档另采 20 张暗场，计算均值与无偏样本方差。所有单曝光和融合方法共享同一份测量。",
              "", "严格实现：r̄ = mean((Zᵢ−B̄ᵢ)/tᵢ)，wᵢ = tᵢ²/(tᵢr̄+Var(Bᵢ))，"
              "融合为 sum(wᵢ(Zᵢ−B̄ᵢ)/tᵢ)/sum(wᵢ)。采用非负估计与数值除零保护。"
              "没有剔除饱和值、加入参考图、调换重建器或增加迭代次数。",
              "", "暗场方差趋近零且分母有效时，原式退化为 sum(Zᵢ−B̄ᵢ)/sum(tᵢ)。"
              "饱和读数仍以普通观测参与平均，亮区会被低估。此现象已由独立公式测试和保存的衍射输入诊断验证。",
              "", "附加扩展只使用未饱和观测估计初始率并计算权重；保留零值观测。"
              "如果一个像素所有曝光都饱和则报告失败。它用于验证饱和偏差的影响，明确不属于论文原算法。",
              "", "## 比较方法与公平性", "",
              "- 使用共同的扫描覆盖 ROI；全幅输出仍含未更新边框，评价不包含该边框。",
              "- 所有图共享基线的灰度范围。每幅只拟合一个正振幅比例，补偿物体/探针尺度不唯一性；不进行平移、滤波、直方图匹配或单独拉伸对比度。",
              "- PSNR 使用固定基线振幅极差；SSIM 使用 7×7 窗口。基线是重建结果而非真实物体，没有把它当成真值报告分辨率提升。",
              "- 衍射 NRMSE 对比已知的相机模拟前输入，使用已知增益还原共同尺度；不单独拟合每种方法的强度比例。",
              "- 每帧拟合误差只作为各方法自身的收敛诊断，不用于跨输入证明图像质量优势。",
              "- 所有方法均从相同初始化设置开始，未使用基线物体或基线探针作为初始化。探针功率归一化按基线程序从各自测量估计。",
              "- 噪声种子预设为 " + str(manifest["selected_seeds"]) + "；图像展示固定使用第一个种子，表格汇总全部种子，没有选取最好的一次。",
              "- 多曝光总积分时间为 666.5 ms/位置（不含暗场及读出开销）；单曝光为对应的曝光时间。该比较并非等采集时间/等光子预算，优势不能解释为采集效率提升。",
              "- 保留所有曝光结果和失败状态。没有新增 LRFC-HDR、16-bit 对照或真实物体 FRC，因此不声称复现了论文全部实验。",
              "", "## 基线复算核验", "",
              f"当前环境按保存配置复算，与历史基线相比：振幅 NRMSE={number(baseline_check['baseline_nrmse'],6)}，"
              f"SSIM={number(baseline_check['baseline_ssim'],6)}。历史基线仍是统一参考；差异没有被替换或隐藏。",
              "", "## 输出与重跑", "",
              "- `manifest.json`：完整设置、来源与假设、软件版本和代码/数据 SHA-256。",
              "- `comparison_metrics.csv`：每个场景、种子和方法的完整指标。",
              "- `summary_metrics.csv`：跨种子平均值和样本标准差。",
              "- `camera_diagnostics.csv`：每档饱和率、零值率、全零帧、暗场均值/方差等。",
              "- `baseline_reference/`：用户选定基线的原始输出副本；`baseline_recomputed/`：当前环境复算。",
              "- `<场景>/seed_<种子>/measurement.npz`：全部数字测量、暗场、曝光时间。",
              "- `<场景>/seed_<种子>/<方法>/`：重建数组、融合/单曝光输入及元数据。第一个种子还包含完整 PNG 和 PtyLab HDF5。",
              "", "从项目目录运行（使用新的输出路径，脚本拒绝覆盖已有目录）：", "", "```bash",
              "/opt/anaconda3/bin/python scripts/reproduce_paper.py --output outputs/paper_reproduction_repeat",
              "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    # A simple self-contained index uses relative local assets, no external JS.
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
        [r['case'], number(r['baseline_psnr_db_mean'],2), number(r['baseline_ssim_mean']),
         number(r['baseline_nrmse_mean'])]) + "</tr>" for r in items)
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ML-HDR 复现比较</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#243044}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#176e8c}}.note{{background:#eef4f8;padding:18px}}</style>
<h1>ML-HDR 论文方法复现</h1><p>固定基线：21×21 扫描 · 80 次 mPIE · 8 px 步长 · 31 px 初始探针</p>
<p class="note">红色曲线与 paper_ml_hdr 为论文式（14）—（15）的实现；绿色虚线与 saturation_mask_extension 为额外饱和处理，不能视为原论文结果。PSNR/SSIM 是与保存基线的一致性，不是真实物体分辨率。</p>
<p><a href="REPORT.md">完整实验报告与限制</a> · <a href="manifest.json">参数与来源</a> · <a href="summary_metrics.csv">指标表</a></p>
<table><tr><th>主场景方法</th><th>PSNR / dB</th><th>SSIM</th><th>振幅 NRMSE</th></tr>{table_rows}</table>"""
    for profile in manifest["selected_profiles"]:
        body += f'<h2>{html.escape(profile)}</h2>'
        for name in ["comparison_amplitude.png", "extension_comparison.png", "metrics_vs_exposure.png", "diffraction_comparison.png"]:
            body += f'<p><a href="{profile}/{name}"><img src="{profile}/{name}" loading="lazy"></a></p>'
    (output / "index.html").write_text(body + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/paper_baseline_comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_reproduction_21x21_i80")
    parser.add_argument("--profiles", help="Comma-separated profile subset (default: all)")
    parser.add_argument("--seeds", help="Comma-separated camera seed subset (default: configured seeds)")
    args = parser.parse_args()
    conf = json.loads(args.config.read_text())
    profiles = args.profiles.split(",") if args.profiles else list(conf["noise_profiles"])
    seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else conf["measurement_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("Seeds must be a nonempty list of unique nonnegative integers")
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in conf["noise_profiles"] for p in profiles):
        parser.error("Unknown or duplicated noise profile")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output}")
    logging.getLogger().setLevel(logging.ERROR)
    archive, input_path = ROOT / conf["baseline_archive"], ROOT / conf["input"]
    diff = load_diff_npy(input_path, conf["scan_crop"])
    clean = normalize_stack(diff.stack)
    config, ref_obj, ref_probe, ref_error, coverage, roi = reference_data(
        archive, conf["scan_crop"], diff.frame_shape, diff.stack)
    if config.iterations != 80:
        raise ValueError("Expected the user's selected 80-iteration baseline")
    reference = ref_obj[roi]
    output.mkdir(parents=True)
    reference_dir = output / "baseline_reference"
    reference_dir.mkdir()
    for source in archive.parent.glob("best_*"):
        if source.is_file():
            shutil.copy2(source, reference_dir / source.name)
    write_json(reference_dir / "config.json", asdict(config))
    np.savez_compressed(reference_dir / "reference_roi.npz", object=reference, coverage=coverage[roi])
    manifest = {
        "experiment": conf, "selected_profiles": profiles, "selected_seeds": seeds,
        "ptylab_config": asdict(config), "input_shape": [*diff.scan_shape, *diff.frame_shape],
        "roi": [[r.start, r.stop] for r in roi], "object_shape": list(ref_obj.shape),
        "baseline_sha256": sha256(archive), "input_sha256": sha256(input_path),
        "python": sys.version, "executable": sys.executable,
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "matplotlib", "PtyLab", "h5py", "scikit-image"]},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                          [*sorted((ROOT / "mlhdr_ptycho").glob("*.py")), Path(__file__), args.config.resolve()]},
        "comparison_reference": "Saved baseline reconstruction, not ground truth",
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    rec, data, _ = run_mpie(diff.stack, diff.scan_shape, config)
    baseline_check, _ = amplitude_comparison(np.squeeze(rec.object)[roi], reference)
    save_reconstruction_outputs(output / "baseline_recomputed", "raw", rec, data, config)
    write_json(output / "baseline_recomputed/comparison_to_saved.json", baseline_check)
    print("Baseline recheck:", baseline_check, flush=True)
    rows, diagnostics, previews = [], [], {}
    exposure_ms = conf["exposure_times_ms"]
    times = np.asarray(exposure_ms, float) / 1000
    for profile in profiles:
        camera = PaperCamera(**conf["camera"], **conf["noise_profiles"][profile])
        profile_dir = output / profile
        profile_dir.mkdir()
        for seed in seeds:
            seed_dir = profile_dir / f"seed_{seed}"
            seed_dir.mkdir()
            m = simulate_paper_camera(clean, times, camera, seed)
            np.savez_compressed(seed_dir / "measurement.npz", z=m.z, dark_frames=m.dark_frames,
                                dark_mean=m.dark_mean, dark_var=m.dark_var,
                                exposure_times_s=times, rate_scale=m.rate_scale)
            write_json(seed_dir / "camera.json", asdict(camera))
            for i, t in enumerate(exposure_ms):
                z = m.z[i]
                diagnostics.append({"profile": profile, "seed": seed, "exposure_ms": t,
                    "electrons_per_count": camera.electrons_per_count,
                    "saturation_fraction": float(np.mean(z == camera.max_count)),
                    "zero_fraction": float(np.mean(z == 0)),
                    "all_zero_frames": int(np.sum(z.sum(axis=(-2,-1)) == 0)),
                    "dark_mean_average": float(m.dark_mean[i].mean()),
                    "dark_variance_average": float(m.dark_var[i].mean()),
                    "dark_variance_nonzero_fraction": float(np.mean(m.dark_var[i] > 0)),
                    "max_expected_signal_e": float(clean.max() * m.rate_scale * times[i]),
                })
            rates = [(f"single_{t:g}ms", single_rate(m, i)) for i, t in enumerate(exposure_ms)]
            rates += [("paper_ml_hdr", paper_eq14_15(m)),
                      ("saturation_mask_extension", saturation_mask_extension(m))]
            if seed == seeds[0]:
                diffraction_plot(profile_dir / "diffraction_comparison.png", clean, rates, m)
                previews[profile] = {"baseline": np.abs(reference)}
            for case, rate in rates:
                start = time.monotonic()
                case_dir = seed_dir / case
                case_dir.mkdir()
                np.savez_compressed(case_dir / "input_rate.npz", count_rate=rate)
                row = {"profile": profile, "seed": seed, "case": case, "status": "ok",
                       **diffraction_comparison(rate, m, clean)}
                if case.startswith("single_"):
                    i = [f"single_{t:g}ms" for t in exposure_ms].index(case)
                    row["saturation_fraction"] = float(np.mean(m.z[i] == camera.max_count))
                aligned = None
                if not np.isfinite(rate).all() or not np.any(rate > 0):
                    row["status"] = "failed_no_finite_signal"
                else:
                    rec, data, _ = run_mpie(rate, diff.scan_shape, config)
                    obj, probe, error = np.squeeze(rec.object), np.squeeze(rec.probe), np.asarray(rec.error)
                    np.savez_compressed(case_dir / "arrays.npz", object=obj, probe=probe,
                                        error=error, encoder=data.encoder, coverage=coverage)
                    if not all(np.isfinite(a).all() for a in [obj, probe, error]):
                        row["status"] = "failed_nonfinite"
                    else:
                        metrics, aligned = amplitude_comparison(obj[roi], reference)
                        row.update(metrics)
                        row.update(reconstruction_metrics(rec, data))
                        if seed == seeds[0]:
                            save_reconstruction_outputs(case_dir, case, rec, data, config)
                row["seconds"] = round(time.monotonic() - start, 3)
                rows.append(row)
                write_json(case_dir / "metadata.json", {"metrics": row, "camera": asdict(camera),
                    "measurement_seed": seed, "ptylab_config": asdict(config),
                    "method": "published_eq14_15" if case == "paper_ml_hdr" else case})
                if seed == seeds[0]:
                    previews[profile][case] = aligned
                print(f"{profile} seed={seed} {case}: {row['status']} "
                      f"PSNR={row.get('baseline_psnr_db')} SSIM={row.get('baseline_ssim')}", flush=True)
                write_csv(output / "comparison_metrics.csv", rows)
            write_csv(output / "camera_diagnostics.csv", diagnostics)
    summary = summarize(rows)
    write_csv(output / "summary_metrics.csv", summary)
    for profile in profiles:
        prev = previews[profile]
        panels = [("Saved raw baseline", prev["baseline"])]
        panels += [(f"Single {t:g} ms", prev[f"single_{t:g}ms"]) for t in exposure_ms]
        panels += [("Published Eq.14-15", prev["paper_ml_hdr"])]
        contact_sheet(output / profile / "comparison_amplitude.png", panels, reference,
                      "Saved baseline / single exposures / published ML-HDR; same grayscale")
        contact_sheet(output / profile / "difference_to_baseline.png", panels, reference,
                      "Absolute amplitude difference from saved baseline", difference=True)
        best = max((r for r in summary if r["profile"] == profile and r["case"].startswith("single_")
                    and r["baseline_ssim_mean"] is not None), key=lambda r: r["baseline_ssim_mean"])
        control = [("Saved raw baseline", prev["baseline"]),
                   (f"Best single by mean SSIM\n{best['case']}", prev[best["case"]]),
                   ("Published Eq.14-15", prev["paper_ml_hdr"]),
                   ("Additional saturation mask\nNOT published algorithm", prev["saturation_mask_extension"])]
        contact_sheet(output / profile / "extension_comparison.png", control, reference,
                      "Saturation diagnostic control (first fixed noise seed)")
        metric_plot(output / profile / "metrics_vs_exposure.png", summary, profile, exposure_ms)
    manifest["status"] = "complete"
    manifest["successful_reconstructions"] = sum(r["status"] == "ok" for r in rows)
    manifest["total_reconstructions"] = len(rows)
    write_json(output / "manifest.json", manifest)
    report(output, manifest, summary, baseline_check)
    print(f"Complete: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
