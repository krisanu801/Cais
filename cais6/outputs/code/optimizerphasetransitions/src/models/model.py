import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import torch
import torch.nn as nn
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) classifier.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): The input dimension.
            hidden_dims (List[int]): A list of hidden layer dimensions.
            output_dim (int): The output dimension (number of classes).
        """
        super(MLP, self).__init__()
        try:
            self.layers = nn.ModuleList()
            dims = [input_dim] + hidden_dims
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dims[-1], output_dim))  # Output layer
            # No softmax here, as it's included in the CrossEntropyLoss
        except Exception as e:
            logging.error(f"Error initializing MLP: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        try:
            for layer in self.layers:
                x = layer(x)
            return x
        except Exception as e:
            logging.error(f"Error during forward pass: {e}")
            raise


# Example Usage:
if __name__ == '__main__':
    try:
        # Example 1: MakeMoons-like model
        input_dim = 2
        hidden_dims = [16, 8]
        output_dim = 2
        model = MLP(input_dim, hidden_dims, output_dim)
        print(f"MLP model architecture:\n{model}")

        # Example 2: MNIST-like model
        input_dim = 32  # PCA-reduced MNIST
        hidden_dims = [64, 32, 16]
        output_dim = 2
        model = MLP(input_dim, hidden_dims, output_dim)
        print(f"MLP model architecture:\n{model}")

        # Example forward pass
        dummy_input = torch.randn(1, input_dim)  # Batch size 1
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in example usage: {e}")