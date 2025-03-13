import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import unittest
import torch
from torch.utils.data import DataLoader
from src.data.data_loader import MakeMoonsDataset, SimplifiedMNISTDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestDataLoader(unittest.TestCase):
    """
    Unit tests for the data loading and preprocessing functionalities.
    """

    def test_make_moons_dataset(self):
        """
        Tests the MakeMoonsDataset.
        """
        try:
            dataset = MakeMoonsDataset(n_samples=100, noise=0.1, random_state=42)
            self.assertEqual(len(dataset), 100)
            data, label = dataset[0]
            self.assertEqual(data.shape, torch.Size([2]))
            self.assertTrue(isinstance(label, torch.Tensor))
            self.assertEqual(dataset.num_classes, 2)
            self.assertEqual(dataset.num_features, 2)
        except Exception as e:
            logging.error(f"Error in test_make_moons_dataset: {e}")
            self.fail(f"MakeMoonsDataset test failed: {e}")

    def test_simplified_mnist_dataset(self):
        """
        Tests the SimplifiedMNISTDataset.
        """
        try:
            dataset = SimplifiedMNISTDataset(num_classes=2, pca_dim=32, random_state=42)
            self.assertGreater(len(dataset), 0)  # MNIST has > 0 samples
            data, label = dataset[0]
            self.assertEqual(data.shape, torch.Size([32]))
            self.assertTrue(isinstance(label, torch.Tensor))
            self.assertEqual(dataset.num_classes, 2)
            self.assertEqual(dataset.pca_dim, 32)
        except Exception as e:
            logging.error(f"Error in test_simplified_mnist_dataset: {e}")
            self.fail(f"SimplifiedMNISTDataset test failed: {e}")

    def test_data_loaders(self):
        """
        Tests the data loaders with both datasets.
        """
        try:
            # MakeMoons
            moons_dataset = MakeMoonsDataset(n_samples=100, noise=0.1, random_state=42)
            moons_loader = DataLoader(moons_dataset, batch_size=32, shuffle=True)
            for batch in moons_loader:
                data, labels = batch
                self.assertEqual(data.shape[1], 2)  # Check feature dimension
                self.assertEqual(labels.shape[0], data.shape[0]) # Check batch size
                break

            # SimplifiedMNIST
            mnist_dataset = SimplifiedMNISTDataset(num_classes=2, pca_dim=32, random_state=42)
            mnist_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
            for batch in mnist_loader:
                data, labels = batch
                self.assertEqual(data.shape[1], 32)  # Check feature dimension
                self.assertEqual(labels.shape[0], data.shape[0]) # Check batch size
                break

        except Exception as e:
            logging.error(f"Error in test_data_loaders: {e}")
            self.fail(f"DataLoader test failed: {e}")


# Example Usage:
if __name__ == '__main__':
    try:
        unittest.main()
    except Exception as e:
        print(f"Error running tests: {e}")