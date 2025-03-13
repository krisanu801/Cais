import sys
import os
import logging
import unittest
import pandas as pd
import numpy as np

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.data.data_loader import load_data, load_csv, load_fits, preprocess_data  # type: ignore
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have configured your PYTHONPATH correctly.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestDataLoader(unittest.TestCase):
    """
    Unit tests for the data loading module.
    """

    def setUp(self):
        """
        Set up test environment.  Creates dummy data files.
        """
        self.data_dir = 'test_data'
        os.makedirs(self.data_dir, exist_ok=True)

        # Create a dummy CSV file
        self.dummy_csv_path = os.path.join(self.data_dir, 'dummy_data.csv')
        self.dummy_csv_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        self.dummy_csv_data.to_csv(self.dummy_csv_path, index=False)

        # Create a dummy FITS file (minimal example)
        self.dummy_fits_path = os.path.join(self.data_dir, 'dummy_data.fits')
        try:
            from astropy.io import fits  # type: ignore
            hdu = fits.PrimaryHDU(np.array([[1, 2], [3, 4]]))
            hdu.writeto(self.dummy_fits_path, overwrite=True)
        except ImportError:
            print("astropy not installed, skipping FITS test.")
            self.dummy_fits_path = None # Set to None to skip the test

    def tearDown(self):
        """
        Tear down test environment.  Removes dummy data files.
        """
        if os.path.exists(self.dummy_csv_path):
            os.remove(self.dummy_csv_path)
        if self.dummy_fits_path and os.path.exists(self.dummy_fits_path):
            os.remove(self.dummy_fits_path)
        if os.path.exists(self.data_dir):
            os.rmdir(self.data_dir) # Only remove if empty

    def test_load_csv(self):
        """
        Tests the load_csv function.
        """
        try:
            data = load_csv(self.dummy_csv_path)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 3)
            self.assertEqual(data['col1'].tolist(), [1, 2, 3])
        except Exception as e:
            self.fail(f"load_csv failed: {e}")

    def test_load_csv_file_not_found(self):
        """
        Tests that load_csv raises FileNotFoundError when the file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            load_csv('nonexistent_file.csv')

    def test_load_fits(self):
        """
        Tests the load_fits function.
        """
        if not self.dummy_fits_path:
            self.skipTest("astropy not installed, skipping FITS test.")

        try:
            data = load_fits(self.dummy_fits_path)
            self.assertIsInstance(data, np.ndarray)
            self.assertEqual(data.shape, (2, 2))
            self.assertEqual(data[0, 0], 1)
        except Exception as e:
            self.fail(f"load_fits failed: {e}")

    def test_load_fits_file_not_found(self):
        """
        Tests that load_fits raises FileNotFoundError when the file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            load_fits('nonexistent_file.fits')

    def test_preprocess_data_dataframe(self):
        """
        Tests the preprocess_data function with a pandas DataFrame.
        """
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        preprocessed_data = preprocess_data(data)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        # Add more specific checks based on the preprocessing steps

    def test_preprocess_data_ndarray(self):
        """
        Tests the preprocess_data function with a numpy array.
        """
        data = np.array([[1, 2], [3, 4]])
        preprocessed_data = preprocess_data(data)
        self.assertIsInstance(preprocessed_data, np.ndarray)
        # Add more specific checks based on the preprocessing steps

    def test_load_data_csv(self):
        """
        Tests the load_data function with a CSV file.
        """
        try:
            data = load_data(self.dummy_csv_path)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 3)
            self.assertEqual(data['col1'].tolist(), [1, 2, 3])
        except Exception as e:
            self.fail(f"load_data (CSV) failed: {e}")

    def test_load_data_fits(self):
        """
        Tests the load_data function with a FITS file.
        """
        if not self.dummy_fits_path:
            self.skipTest("astropy not installed, skipping FITS test.")

        try:
            data = load_data(self.dummy_fits_path)
            self.assertIsInstance(data, np.ndarray)
            self.assertEqual(data.shape, (2, 2))
            self.assertEqual(data[0, 0], 1)
        except Exception as e:
            self.fail(f"load_data (FITS) failed: {e}")

    def test_load_data_unsupported_format(self):
        """
        Tests that load_data raises ValueError when the file format is not supported.
        """
        with self.assertRaises(ValueError):
            load_data('unsupported_file.txt')

    def test_load_data_file_not_found(self):
        """
        Tests that load_data raises FileNotFoundError when the file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent_file.csv')


if __name__ == '__main__':
    unittest.main()