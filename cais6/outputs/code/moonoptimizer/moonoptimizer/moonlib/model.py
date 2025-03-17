import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """
    A simple neural network with one hidden layer.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        Initializes the simple neural network.

        Args:
            input_size: The number of input features.
            hidden_size: The number of neurons in the hidden layer.
            num_classes: The number of output classes.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out