"""Scientific checks for the published formula, camera statistics and metrics."""
import unittest
from dataclasses import replace

import numpy as np

from mlhdr_ptycho.paper_reproduction import (
    PaperCamera, PaperMeasurement, amplitude_comparison, diffraction_comparison,
    paper_eq14_15, quantize_electrons, saturation_mask_extension, simulate_paper_camera,
)


class PaperReproductionTests(unittest.TestCase):
    def test_adc_bounds_and_rounding(self):
        camera = PaperCamera(full_well_e=255)
        np.testing.assert_array_equal(
            quantize_electrons(np.array([-10, 0, .49, .51, 254.6, 900]), camera),
            [0, 0, 0, 1, 255, 255],
        )

    def test_eq14_15_matches_independent_scalar_calculation(self):
        camera = PaperCamera(dark_frames=2)
        # Dark means [1,3], unbiased variances [2,2]. Rates [10,8].
        # Preliminary rate=9; weights=[1/(9+2),4/(18+2)].
        m = PaperMeasurement(np.array([[[[11]]], [[[19]]]], np.uint8),
                             np.array([[[[0]], [[2]]], [[[2]], [[4]]]], np.uint8),
                             np.array([1., 2.]), camera, 1.)
        expected = ((1/11)*10 + (4/20)*8) / (1/11 + 4/20)
        self.assertAlmostEqual(float(paper_eq14_15(m).item()), expected, places=5)

    def test_published_rule_keeps_saturation_and_extension_is_separate(self):
        m = PaperMeasurement(np.array([[[[100]]], [[[255]]]], np.uint8),
                             np.zeros((2,20,1,1), np.uint8), np.array([.001,.01]),
                             PaperCamera(), 1.)
        # Published zero-dark-variance limit is total counts / total time.
        self.assertAlmostEqual(float(paper_eq14_15(m).item()), 355/.011, delta=.02)
        self.assertAlmostEqual(float(saturation_mask_extension(m).item()), 1e5, delta=.1)
        saturated = replace(m, z=np.full(m.z.shape, 255, np.uint8))
        with self.assertRaisesRegex(ValueError, "All exposures saturated"):
            saturation_mask_extension(saturated)

    def test_camera_is_repeatable_and_preserves_global_frame_ratios(self):
        intensity = np.stack([np.ones((8,8)), .25*np.ones((8,8))])
        cam = PaperCamera(photon_flux_per_s=1e5, full_well_e=65535, bit_depth=16,
                          dark_current_e_per_s=0, read_noise_e=0)
        a = simulate_paper_camera(intensity, [.1, .2], cam, seed=7)
        b = simulate_paper_camera(intensity, [.1, .2], cam, seed=7)
        np.testing.assert_array_equal(a.z, b.z)
        np.testing.assert_array_equal(a.dark_frames, b.dark_frames)
        self.assertAlmostEqual(a.rate_scale, 1e5/64)
        self.assertAlmostEqual(a.z[1,0].mean()/a.z[1,1].mean(), 4, delta=.15)
        # Changing dark-frame sample count must not change diffraction draws.
        c = simulate_paper_camera(intensity, [.1,.2], replace(cam,dark_frames=25), seed=7)
        np.testing.assert_array_equal(a.z,c.z)

    def test_camera_mean_variance_match_poisson_plus_gaussian(self):
        camera = PaperCamera(photon_flux_per_s=1e7, full_well_e=65535, bit_depth=16,
                             read_noise_e=10, dark_current_e_per_s=1e5)
        m = simulate_paper_camera(np.ones((30,32,32)), [.02], camera, 123)
        signal_mean = 1e7/1024*.02
        expected_mean = signal_mean + 1e5*.02
        expected_var = expected_mean + 10**2
        self.assertAlmostEqual(float(m.z.mean()), expected_mean, delta=2)
        self.assertAlmostEqual(float(m.z.var()), expected_var, delta=expected_var*.05)
        self.assertAlmostEqual(float(m.dark_frames.mean()), 2000, delta=2)
        self.assertAlmostEqual(float(m.dark_frames.var()), 2100, delta=2100*.05)

    def test_default_dark_quantization_is_recorded_not_fabricated(self):
        m = simulate_paper_camera(np.ones((2,8,8)), [.0005,.5], PaperCamera(), 0)
        self.assertEqual(np.count_nonzero(m.dark_var), 0)
        self.assertEqual(np.count_nonzero(m.dark_mean), 0)

    def test_metrics_remove_only_amplitude_scale(self):
        reference = np.linspace(.2,1,256).reshape(16,16).astype(complex)
        metrics, aligned = amplitude_comparison(reference*3*np.exp(.7j), reference)
        self.assertLess(metrics['baseline_nrmse'], 1e-14)
        self.assertAlmostEqual(metrics['baseline_ssim'],1)
        np.testing.assert_allclose(aligned, np.abs(reference))
        shifted, _ = amplitude_comparison(np.roll(reference,3,axis=0), reference)
        self.assertGreater(shifted['baseline_nrmse'], .1)

    def test_common_intensity_scale_does_not_hide_bias(self):
        clean=np.ones((2,8,8))
        m=simulate_paper_camera(clean,[.001],PaperCamera(),0)
        correct=diffraction_comparison(clean*m.count_rate_scale,m,clean)
        biased=diffraction_comparison(clean*m.count_rate_scale*.5,m,clean)
        self.assertAlmostEqual(correct['diffraction_nrmse'],0)
        self.assertAlmostEqual(biased['diffraction_nrmse'],.5)

    def test_reject_invalid_physical_inputs(self):
        for kwargs in [{'bit_depth':17},{'dark_frames':1},{'read_noise_e':-1},
                       {'photon_flux_per_s':float('nan')}]:
            with self.assertRaises(ValueError):
                PaperCamera(**kwargs)
        for times in [[0],[-1],[float('nan')]]:
            with self.assertRaises(ValueError):
                simulate_paper_camera(np.ones((1,2,2)),times,PaperCamera())


if __name__ == '__main__':
    unittest.main()
