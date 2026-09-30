"""Checks for the paper-style simulation, fusions, mPIE and resolution analysis."""
import unittest
from dataclasses import replace

import numpy as np
from scipy import ndimage

from mlhdr_ptycho.ml_hdr import ml_hdr_fusion
from mlhdr_ptycho.mpie import (
    MPIEConfig, _shift_content, initial_probe, probe_com_offset, reconstruct,
)
from mlhdr_ptycho.paper_reproduction import (
    PaperCamera, PaperMeasurement, lrfc_hdr_fusion, paper_eq14_15, quantize_electrons,
    saturation_mask_extension, simulate_paper_camera,
)
from mlhdr_ptycho.resolution import (
    align_to_truth, amplitude_metrics, fourier_ring_correlation, usaf_line_profile,
    usaf_resolved_elements,
)
from mlhdr_ptycho.simulation import (
    SimulationConfig, build_scene, cameraman_object, forward_intensity, fuse, make_probe,
    scan_positions, usaf1951_object, usaf_line_width_um,
)


def _measurement(z, dark, times, camera=None):
    camera = camera or PaperCamera(dark_frames=2)
    return PaperMeasurement(np.asarray(z), np.asarray(dark), np.asarray(times, float), camera, 1.0)


class FusionTests(unittest.TestCase):
    def test_lrfc_uses_longest_unsaturated_exposure(self):
        # One frame, 3 pixels; exposures deliberately unsorted: t = [0.1, 0.001, 0.01].
        z = np.array([[[[255, 255, 40]]], [[[3, 255, 1]]], [[[30, 255, 9]]]], np.uint8)
        dark = np.zeros((3, 2, 1, 3), np.uint8)
        dark[0, :, 0, 2] = 2      # dark mean 2 counts for t=0.1 at pixel 2
        m = _measurement(z, dark, [0.1, 0.001, 0.01])
        rate = lrfc_hdr_fusion(m)[0, 0]
        self.assertAlmostEqual(float(rate[0]), 30 / 0.01, places=3)       # 0.1 s saturated -> 10 ms
        self.assertAlmostEqual(float(rate[1]), 255 / 0.001, places=1)     # all saturated -> shortest
        self.assertAlmostEqual(float(rate[2]), (40 - 2) / 0.1, places=3)  # longest valid, dark-corrected

    def test_ml_guard_is_finite_with_all_zero_darks(self):
        z = np.zeros((3, 2, 4, 4), np.uint8)
        z[2, 0, 0, 0] = 5
        z[1, 1, 1, 1] = 255   # saturated observation
        m = _measurement(z, np.zeros((3, 20, 4, 4), np.uint8), [0.001, 0.01, 0.1],
                         PaperCamera(dark_frames=20))
        for fused in (paper_eq14_15(m), saturation_mask_extension(m)):
            self.assertTrue(np.isfinite(fused).all())
            self.assertEqual(float(fused[1, 0, 0]), 0.0)   # 0/0 pixel stays zero
        # Zero dark variance -> weights proportional to t -> total counts / total time.
        self.assertAlmostEqual(float(paper_eq14_15(m)[0, 0, 0]), 5 / 0.111, places=3)

    def test_ml_literal_matches_independent_eq14_15_and_is_chunk_invariant(self):
        rng = np.random.default_rng(3)
        times = np.array([0.001, 0.005, 0.02])
        z = rng.integers(0, 256, size=(3, 7, 5, 5)).astype(np.uint8)
        dark = rng.integers(0, 4, size=(3, 20, 5, 5)).astype(np.uint8)
        m = _measurement(z, dark, times, PaperCamera(dark_frames=20))
        # Independent float64 evaluation of Eq. 14-15 (saturated pixels kept).
        bbar = dark.astype(float).mean(axis=1)[:, None]
        var = dark.astype(float).var(axis=1, ddof=1)[:, None]
        t = times[:, None, None, None]
        corrected = z - bbar
        rbar = np.maximum((corrected / t).mean(axis=0), 0)
        w = t ** 2 / (t * rbar + var)
        expected = np.maximum((w * corrected / t).sum(0) / w.sum(0), 0)
        np.testing.assert_allclose(paper_eq14_15(m), expected, rtol=1e-5, atol=1e-2)
        np.testing.assert_array_equal(paper_eq14_15(m), ml_hdr_fusion(m.paper_input()))
        whole = {"lrfc": lrfc_hdr_fusion, "ml_eq14_15": paper_eq14_15,
                 "ml_masked": lambda x: saturation_mask_extension(x, all_saturated="shortest")}
        for method, function in whole.items():
            np.testing.assert_array_equal(fuse(m, method, 0, chunk=2)[0], function(m))

    def test_masked_extension_shortest_option(self):
        z = np.full((2, 1, 1, 2), 255, np.uint8)
        z[:, 0, 0, 1] = [10, 100]
        m = _measurement(z, np.zeros((2, 2, 1, 2), np.uint8), [0.001, 0.01])
        with self.assertRaisesRegex(ValueError, "All exposures saturated"):
            saturation_mask_extension(m)
        fused = saturation_mask_extension(m, all_saturated="shortest")[0, 0]
        self.assertAlmostEqual(float(fused[0]), 255 / 0.001, places=1)
        self.assertAlmostEqual(float(fused[1]), 1e4, places=1)


class CameraBitsTests(unittest.TestCase):
    def test_extended_bit_depth_quantisation(self):
        with self.assertRaises(ValueError):
            PaperCamera(bit_depth=20)                       # default limit unchanged
        with self.assertRaises(ValueError):
            PaperCamera(bit_depth=25, allow_extended_bit_depth=True)
        cam = PaperCamera(bit_depth=20, full_well_e=2.5e6, allow_extended_bit_depth=True)
        k = cam.electrons_per_count
        self.assertEqual(cam.max_count, 2 ** 20 - 1)
        q = quantize_electrons(np.array([-5, 0.49 * k, 0.51 * k, 1234.4 * k, 3e6]), cam)
        self.assertEqual(q.dtype, np.uint32)
        np.testing.assert_array_equal(q, [0, 0, 1, 1234, 2 ** 20 - 1])
        self.assertEqual(quantize_electrons(np.array([1.0]), PaperCamera(bit_depth=12)).dtype,
                         np.uint16)
        for bits in (2, 8, 14, 20):
            c = SimulationConfig().camera(bits)
            x = np.linspace(0, c.full_well_e, 1001)
            counts = quantize_electrons(x, c).astype(float)
            self.assertLessEqual(np.abs(counts * c.electrons_per_count - x).max(),
                                 0.5 * c.electrons_per_count + 1e-6)
            self.assertEqual(int(counts.max()), 2 ** bits - 1)

    def test_same_seed_gives_same_electrons_across_bit_depths(self):
        intensity = np.random.default_rng(0).random((3, 8, 8))
        a = simulate_paper_camera(intensity, [0.01], SimulationConfig().camera(20), seed=5)
        b = simulate_paper_camera(intensity, [0.01], SimulationConfig().camera(8), seed=5)
        ka, kb = a.camera.electrons_per_count, b.camera.electrons_per_count
        np.testing.assert_allclose(np.rint(a.z * ka / kb), b.z, atol=1)


class GeometryTests(unittest.TestCase):
    def test_usaf_line_width_formula(self):
        self.assertAlmostEqual(usaf_line_width_um(7, 1), 1000 / 256)
        self.assertAlmostEqual(usaf_line_width_um(9, 1), 1000 / 1024)
        self.assertAlmostEqual(usaf_line_width_um(9, 6), 1000 / (2 * 2 ** (9 + 5 / 6)))
        self.assertAlmostEqual(usaf_line_width_um(8, 4) / usaf_line_width_um(8, 5), 2 ** (1 / 6))

    def test_usaf_rendering_geometry(self):
        dx = SimulationConfig().dx_um
        obj, geom = usaf1951_object(226, dx)
        self.assertEqual(len(geom), 3 * 6 * 2)
        self.assertTrue(np.allclose(obj.imag, 0))
        amp = obj.real
        self.assertAlmostEqual(float(amp.max()), 1.0)
        self.assertGreaterEqual(float(amp.min()), 0.15 - 1e-12)
        # Area-downsampled rendering preserves total bar area (15 w^2 per triplet).
        bar_area = sum(15 * g["line_width_px"] ** 2 for g in geom)
        self.assertAlmostEqual(float(np.sum(1 - amp) / 0.85), bar_area, delta=bar_area * 0.01)
        for g in geom:
            w = g["line_width_px"]
            self.assertAlmostEqual(w, g["line_width_um"] / dx)
            edges = np.array(g["bar_edges_px"])
            np.testing.assert_allclose(edges[:, 1] - edges[:, 0], w)
            np.testing.assert_allclose(np.diff(edges[:, 0]), 2 * w)
            np.testing.assert_allclose(np.diff(g["bar_extent_px"]), 5 * w)
        coarse = next(g for g in geom if (g["group"], g["element"], g["orientation"]) == (7, 1, "vertical"))
        a, b = coarse["bar_edges_px"][1]
        y0, y1 = coarse["bar_extent_px"]
        # pixels (centre j, extent j +- 0.5) lying entirely inside the bar
        inner = amp[int(np.ceil(y0 + 0.5)):int(np.floor(y1 - 0.5)) + 1,
                    int(np.ceil(a + 0.5)):int(np.floor(b - 0.5)) + 1]
        self.assertGreater(inner.size, 20)
        np.testing.assert_allclose(inner, 0.15, atol=1e-12)
        finest = [g["line_width_um"] for g in geom if g["group"] == 9]
        self.assertLess(min(finest), dx)            # straddles the pixel (Nyquist) limit
        self.assertGreater(max(finest), 1.5 * dx)

    def test_scene_truth_analysis(self):
        scene = build_scene("usaf")
        y0, y1, x0, x1 = scene.roi
        self.assertEqual(y1 - y0, x1 - x0)
        self.assertAlmostEqual(scene.dx_um, 0.5711, places=3)
        amp = np.abs(scene.truth[y0:y1, x0:x1])
        res = usaf_resolved_elements(amp, scene.usaf_geometry, (y0, x0))
        self.assertIsNotNone(res["limit"]["label"])
        self.assertTrue(res["limit"]["label"].startswith("9-"))
        blurred = ndimage.gaussian_filter(amp, 2.0)
        res_blur = usaf_resolved_elements(blurred, scene.usaf_geometry, (y0, x0))
        order = lambda lab: tuple(map(int, lab.split("-"))) if lab else (0, 0)
        self.assertLess(order(res_blur["limit"]["label"]), order(res["limit"]["label"]))
        prof = usaf_line_profile(amp, scene.usaf_geometry, scene.dx_um, (y0, x0))
        self.assertEqual(prof["elements"], ["9-1", "9-2", "9-3"])
        self.assertEqual(len(prof["bar_edges_um"]), 9)
        self.assertLess(prof["profile"].min(), 0.4)
        self.assertAlmostEqual(prof["profile"].max(), 1.0, places=6)
        # Auto-exposure is the longest exposure below full well and fixed per object.
        t = scene.auto_exposure_index()
        cfg = scene.config
        peak = scene.clean.max() * scene.rate_scale
        self.assertLess((peak + cfg.dark_current_e_per_s) * cfg.exposure_times_s[t], cfg.full_well_e)
        if t + 1 < len(cfg.exposure_times_s):
            self.assertGreaterEqual((peak + cfg.dark_current_e_per_s) * cfg.exposure_times_s[t + 1],
                                    cfg.full_well_e)


class AnalysisTests(unittest.TestCase):
    def test_frc_identical_images_is_one(self):
        img = np.abs(cameraman_object(96))
        frc = fourier_ring_correlation(img, img, 0.5)
        np.testing.assert_allclose(frc["frc"][1:], 1.0, atol=1e-12)
        self.assertTrue(frc["reached_nyquist"])
        self.assertAlmostEqual(frc["resolution_um"], 0.5)
        noisy = img + np.random.default_rng(0).normal(0, 0.3, img.shape)
        self.assertLess(fourier_ring_correlation(noisy, img, 0.5)["cutoff_nyquist"], 1.0)

    def test_alignment_removes_phase_scale_shift_and_ramp(self):
        truth = cameraman_object(128)
        roi = (24, 104, 24, 104)
        yy, xx = np.indices(truth.shape)
        shift = (3.0, -2.4)
        ky, kx = 2 * np.pi * 3 / 128, -2 * np.pi * 2 / 128   # periodic ramp
        ramped = truth * np.exp(1j * (ky * yy + kx * xx))       # object/probe ramp ambiguity
        moved = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(ramped), shift))
        recon = 2.3 * np.exp(0.7j) * moved
        al = align_to_truth(recon, truth, roi, upsample_factor=20)
        np.testing.assert_allclose(al.shift_px, (-3.0, 2.4), atol=1e-6)
        t_roi = truth[24:104, 24:104]
        np.testing.assert_allclose(al.phase_ramp_rad_per_px, (ky, kx), atol=1e-9)
        self.assertLess(np.linalg.norm(al.aligned_roi - t_roi) / np.linalg.norm(t_roi), 1e-9)
        m = amplitude_metrics(al.aligned_roi, t_roi)
        self.assertLess(m["nrmse"], 1e-9)
        self.assertAlmostEqual(m["ssim"], 1.0, places=9)
        # Integer-only and zero shifts, without a ramp.
        for s in [(0.0, 0.0), (-4.0, 1.0), (1.35, -0.7)]:
            m2 = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(truth), s)) * (0.4 - 0.9j)
            al2 = align_to_truth(m2, truth, roi)
            np.testing.assert_allclose(al2.shift_px, (-s[0], -s[1]), atol=1e-9)
            self.assertLess(amplitude_metrics(al2.aligned_roi, t_roi)["nrmse"], 1e-9)


class MPIETests(unittest.TestCase):
    def test_tiny_noiseless_reconstruction_converges(self):
        n = 32
        probe = make_probe(n, 16.0, 1.0, 0.5)
        positions, size = scan_positions(7, 4.0, 0.1, seed=1, pad_px=2, n=n)
        truth = cameraman_object(size)
        intensity = forward_intensity(truth, probe, positions)
        p0 = initial_probe(n, 16.0, float(intensity.sum(axis=(1, 2)).max()))
        cfg = MPIEConfig(iterations=80, history_every=20, seed=0)
        res = reconstruct(intensity, positions, truth.shape, p0, cfg)
        self.assertFalse(res.diverged)
        self.assertLess(res.diffraction_error[-1], res.diffraction_error[0] * 0.05)
        roi = (14, size - 14, 14, size - 14)
        al = align_to_truth(res.object, truth, roi)
        self.assertLess(amplitude_metrics(al.aligned_roi, truth[14:-14, 14:-14])["nrmse"], 0.05)
        self.assertEqual(res.history["iteration"], [0, 20, 40, 60, 80])
        # Same seed -> identical result (fair comparison across methods).
        again = reconstruct(intensity, positions, truth.shape, p0, replace(cfg, history_every=0))
        np.testing.assert_array_equal(again.object, res.object)

    def test_com_stabilisation_shift_preserves_exit_waves(self):
        n = 32
        probe = make_probe(n, 12.0, 1.0, 0.5)
        self.assertEqual(probe_com_offset(probe), (0, 0))
        drifted = np.roll(probe, (3, -2), axis=(0, 1))
        dy, dx = probe_com_offset(drifted)
        self.assertEqual((dy, dx), (3, -2))
        obj = cameraman_object(80)
        # Re-centring: P'(r) = P(r + d), O'(u) = O(u + d) leaves |F[P' O'_j]| unchanged.
        p2 = np.roll(drifted, (-dy, -dx), axis=(0, 1))
        o2 = _shift_content(obj, -dy, -dx, 1.0)
        np.testing.assert_allclose(p2, probe)
        pos = np.array([[20, 20], [25, 30]])
        a = forward_intensity(obj, drifted, pos)
        b = forward_intensity(o2, p2, pos)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * a.max())


if __name__ == "__main__":
    unittest.main()
