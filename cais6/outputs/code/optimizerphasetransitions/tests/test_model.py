import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import unittest
import torch
import torch.nn as nn
from src.models.model import MLP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestMLP(unittest.TestCase):
    """
    Unit tests for the MLP model.
    """

    def test_mlp_creation(self):
        """
        Tests the creation of the MLP model with different configurations.
        """
        try:
            # Test case 1: MakeMoons-like model
            input_dim = 2
            hidden_dims = [16, 8]
            output_dim = 2
            model = MLP(input_dim, hidden_dims, output_dim)
            self.assertIsInstance(model, nn.Module)

            # Test case 2: MNIST-like model
            input_dim = 32
            hidden_dims = [64, 32, 16]
            output_dim = 2
            model = MLP(input_dim, hidden_dims, output_dim)
            self.assertIsInstance(model, nn.Module)

        except Exception as e:
            logging.error(f"Error in test_mlp_creation: {e}")
            self.fail(f"MLP creation test failed: {e}")

    def test_mlp_forward_pass(self):
        """
        Tests the forward pass of the MLP model.
        """
        try:
            # Test case 1: MakeMoons-like model
            input_dim = 2
            hidden_dims = [16, 8]
            output_dim = 2
            model = MLP(input_dim, hidden_dims, output_dim)

            # Create a dummy input
            dummy_input = torch.randn(1, input_dim)

            # Perform the forward pass
            output = model(dummy_input)

            # Check the output shape
            self.assertEqual(output.shape, torch.Size([1, output_dim]))

            # Test case 2: MNIST-like model
            input_dim = 32
            hidden_dims = [64, 32, 16]
            output_dim = 2
            model = MLP(input_dim, hidden_dims, output_dim)

            # Create a dummy input
            dummy_input = torch.randn(1, input_dim)

            # Perform the forward pass
            output = model(dummy_input)

            # Check the output shape
            self.assertEqual(output.shape, torch.Size([1, output_dim]))

        except Exception as e:
            logging.error(f"Error in test_mlp_forward_pass: {e}")
            self.fail(f"MLP forward pass test failed: {e}")

    def test_mlp_with_different_batch_sizes(self):
        """
        Tests the MLP model with different batch sizes.
        """
        try:
            input_dim = 2
            hidden_dims = [16, 8]
            output_dim = 2
            model = MLP(input_dim, hidden_dims, output_dim)

            # Test with batch size 16
            batch_size = 16
            dummy_input = torch.randn(batch_size, input_dim)
            output = model(dummy_input)
            self.assertEqual(output.shape, torch.Size([batch_size, output_dim]))

            # Test with batch size 32
            batch_size = 32
            dummy_input = torch.randn(batch_size, input_dim)
            output = model(dummy_input)
            self.assertEqual(output.shape, torch.Size([batch_size, output_dim]))

        except Exception as e:
            logging.error(f"Error in test_mlp_with_different_batch_sizes: {e}")
            self.fail(f"MLP batch size test failed: {e}")


# Example Usage:
if __name__ == '__main__':
    try:
        unittest.main()
    except Exception as e:
        print(f"Error running tests: {e}")