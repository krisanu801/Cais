import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_loss_landscape(model: nn.Module,
                        data_loader: DataLoader,
                        criterion: Callable,
                        device: torch.device,
                        w_min: float = -1.0,
                        w_max: float = 1.0,
                        num_points: int = 50,
                        filename: Optional[str] = None) -> None:
    """
    Visualizes the loss landscape around the model's current weights.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): The data loader for evaluation.
        criterion (Callable): The loss function.
        device (torch.device): The device to perform computations on.
        w_min (float): Minimum weight perturbation.
        w_max (float): Maximum weight perturbation.
        num_points (int): Number of points to sample along each axis.
        filename (Optional[str]): If provided, saves the plot to this file.
    """
    try:
        # Store original weights
        original_weights = {name: param.data.clone() for name, param in model.named_parameters()}

        # Define weight range
        w_range = np.linspace(w_min, w_max, num_points)

        # Initialize loss grid
        loss_grid = np.zeros((num_points, num_points))

        # Get a batch of data
        try:
            inputs, labels = next(iter(data_loader))
            inputs, labels = inputs.to(device), labels.to(device)
        except StopIteration:
            logging.error("DataLoader is empty.")
            return

        # Calculate loss for each weight combination
        for i, w1 in enumerate(w_range):
            for j, w2 in enumerate(w_range):
                # Perturb weights
                for name, param in model.named_parameters():
                    if 'weight' in name:  # Perturb only weight parameters
                        param.data = original_weights[name] + w1 * torch.randn_like(param.data) + w2 * torch.randn_like(param.data)

                # Calculate loss
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_grid[i, j] = loss.item()

        # Restore original weights
        for name, param in model.named_parameters():
            param.data = original_weights[name]

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.imshow(loss_grid, extent=[w_min, w_max, w_min, w_max], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Loss')
        plt.xlabel('Weight Perturbation 1')
        plt.ylabel('Weight Perturbation 2')
        plt.title('Loss Landscape')

        if filename:
            plt.savefig(filename)
            logging.info(f"Loss landscape plot saved to {filename}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        logging.error(f"Error plotting loss landscape: {e}")


# Example Usage:
if __name__ == '__main__':
    try:
        # Dummy model, data loader, and loss function
        model = nn.Linear(10, 2)
        data = torch.randn(64, 10)
        labels = torch.randint(0, 2, (64,))
        dataset = [(data[i], labels[i]) for i in range(64)]
        data_loader = DataLoader(dataset, batch_size=32)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')

        # Plot loss landscape
        plot_loss_landscape(model, data_loader, criterion, device, filename="loss_landscape.png")

    except Exception as e:
        print(f"Error in example usage: {e}")