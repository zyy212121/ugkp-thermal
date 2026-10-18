"""Exercise the real offline generator without creating a production table."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mie_generator', ROOT / 'tool/radiation/mieTables/make_alumina_mie_table.py')
mie = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mie)


class MiePhysicalDataTests(unittest.TestCase):
    def test_nonfinite_phase_does_not_become_isotropic(self):
        with self.assertRaisesRegex(ArithmeticError, 'phase.*index=1'):
            mie.normalize_phi(np.array([1., np.nan, 1.]), np.array([-1., 0., 1.]))

    def test_zero_normalization_is_reported(self):
        with self.assertRaisesRegex(ArithmeticError, 'normalization.*0'):
            mie.normalize_phi(np.zeros(3), np.array([-1., 0., 1.]))

    def test_nonfinite_planck_input_is_reported(self):
        with self.assertRaisesRegex(ArithmeticError, 'Planck.*T=nan'):
            mie.compute_node(1e-5, float('nan'), np.linspace(-1, 1, 9), np.array([1e-6, 2e-6]))

    def test_nonfinite_mie_coefficients_do_not_become_zero(self):
        with self.assertRaisesRegex(ArithmeticError, 'Mie.*refractive'):
            mie.mie_coefficients(1., complex(float('nan'), 0.))

    def test_invalid_size_cannot_take_zero_scattering_limit(self):
        for diameter in (-float('inf'), float('inf'), float('nan'), -1e-6):
            with self.subTest(diameter=diameter):
                with self.assertRaisesRegex(ArithmeticError, 'Mie.*size'):
                    mie.compute_node(diameter, 1000., np.linspace(-1, 1, 9), np.geomspace(1e-6, 8e-6, 12))

    def test_zero_size_retains_no_scattering_limit(self):
        result = mie.compute_node(0., 1000., np.linspace(-1, 1, 9), np.geomspace(1e-6, 8e-6, 12))
        self.assertEqual(result[:4], (0., 0., 0., 0.))
        np.testing.assert_array_equal(result[4], np.ones(9))

    def test_invalid_consumed_planck_inputs_are_errors(self):
        for value in (0., -1., float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ArithmeticError):
                    mie.planck_weight_lambda(value, 1000.)
                with self.assertRaises(ArithmeticError):
                    mie.planck_weight_lambda(1e-6, value)

    def test_normal_physics_remains_finite(self):
        result = mie.compute_node(1e-6, 1000., np.linspace(-1, 1, 201), np.geomspace(1e-6, 8e-6, 12))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in result))
        self.assertAlmostEqual(result[5], 2.)

    def test_planck_underflow_is_legitimate_zero(self):
        self.assertEqual(mie.planck_weight_lambda(1e-10, 300.), 0.)

if __name__ == '__main__':
    unittest.main()
