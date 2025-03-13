import sys
import os
import logging
import unittest
import numpy as np
import pandas as pd

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.models.lens_model import fit_lens_model  # type: ignore
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have configured your PYTHONPATH correctly.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestLensModel(unittest.TestCase):
    """
    Unit tests for the lens modeling module.
    """

    def setUp(self):
        """
        Set up test environment. Creates dummy data.
        """
        self.image_positions = [(1.0, 1.0), (-1.0, -1.0), (0.5, -0.5), (-0.5, 0.5)]
        self.flux_ratios = [1.0, 0.5, 0.3, 0.2]
        self.kinematic_data = {'velocity_dispersion': 200, 'error': 10}
        self.time_delay = 10.0

        self.data = {
            'image_positions': self.image_positions,
            'flux_ratios': self.flux_ratios,
            'kinematic_data': self.kinematic_data
        }

    def test_fit_lens_model(self):
        """
        Tests the fit_lens_model function.
        """
        try:
            lens_model_results = fit_lens_model(self.data, self.time_delay)
            self.assertIsInstance(lens_model_results, dict)
            self.assertIn('best_fit_params', lens_model_results)
            self.assertIn('param_uncertainties', lens_model_results)
            self.assertIn('samples', lens_model_results)
            self.assertIsInstance(lens_model_results['best_fit_params'], np.ndarray)
            self.assertIsInstance(lens_model_results['param_uncertainties'], np.ndarray)
            self.assertIsInstance(lens_model_results['samples'], np.ndarray)
        except Exception as e:
            self.fail(f"fit_lens_model failed: {e}")

    def test_fit_lens_model_invalid_data(self):
        """
        Tests that fit_lens_model raises ValueError when the input data is invalid.
        """
        invalid_data = {
            'image_positions': self.image_positions,
            'flux_ratios': self.flux_ratios
        }
        with self.assertRaises(ValueError):
            fit_lens_model(invalid_data, self.time_delay)

    def test_fit_lens_model_no_time_delay(self):
        """
        Tests that fit_lens_model runs without error even if time delay is None.
        """
        try:
            lens_model_results = fit_lens_model(self.data, None)
            self.assertIsInstance(lens_model_results, dict)
        except Exception as e:
            self.fail(f"fit_lens_model failed with None time_delay: {e}")


if __name__ == '__main__':
    unittest.main()