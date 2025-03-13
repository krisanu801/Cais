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
from typing import Callable
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def approximate_hessian(model: nn.Module,
                        data_loader: DataLoader,
                        criterion: Callable,
                        device: torch.device,
                        h: float = 1e-3) -> torch.Tensor:
    """
    Approximates the Hessian matrix of the loss function with respect to the model parameters using finite differences.

    Args:
        model (nn.Module): The model for which to compute the Hessian.
        data_loader (DataLoader): DataLoader for a representative dataset.
        criterion (Callable): The loss function.
        device (torch.device): The device to perform computations on.
        h (float): Step size for finite differences.

    Returns:
        torch.Tensor: The approximated Hessian matrix.
    """
    try:
        # Get the number of parameters
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Initialize the Hessian matrix
        hessian = torch.zeros((num_parameters, num_parameters), device=device)

        # Get the current parameters
        original_params = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

        # Compute the loss at the original parameters
        model.eval()
        with torch.no_grad():
            inputs, labels = next(iter(data_loader))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            original_loss = loss.item()

        # Iterate over all pairs of parameters
        for i in range(num_parameters):
            for j in range(i, num_parameters):  # Exploit symmetry

                # Perturb parameters
                def perturb_parameters(delta_i: float, delta_j: float) -> None:
                    with torch.no_grad():
                        current_index = 0
                        for param in model.parameters():
                            if param.requires_grad:
                                param_size = param.numel()
                                param_data = param.data.view(-1)

                                if current_index <= i < current_index + param_size:
                                    param_data[i - current_index] += delta_i

                                if current_index <= j < current_index + param_size:
                                    param_data[j - current_index] += delta_j

                                current_index += param_size

                # Compute J(w* + h*e_i + h*e_j)
                perturb_parameters(h, h)
                model.eval()
                with torch.no_grad():
                    inputs, labels = next(iter(data_loader))
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_plus_i_plus_j = loss.item()

                # Compute J(w* + h*e_i)
                perturb_parameters(h, -h) # Reset j, keep i
                model.eval()
                with torch.no_grad():
                    inputs, labels = next(iter(data_loader))
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_plus_i = loss.item()

                # Compute J(w* + h*e_j)
                perturb_parameters(-h, h) # Reset i, keep j
                model.eval()
                with torch.no_grad():
                    inputs, labels = next(iter(data_loader))
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_plus_j = loss.item()

                # Compute the Hessian element
                hessian[i, j] = (loss_plus_i_plus_j - loss_plus_i - loss_plus_j + original_loss) / (h * h)
                hessian[j, i] = hessian[i, j]  # Exploit symmetry

                # Restore original parameters (undo perturbation)
                current_index = 0
                for param in model.parameters():
                    if param.requires_grad:
                        param.data.copy_(original_params[current_index:current_index + param.numel()].view(param.size()))
                        current_index += param.numel()

        return hessian

    except Exception as e:
        logging.error(f"Error approximating Hessian: {e}")
        raise


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

        # Approximate Hessian
        hessian = approximate_hessian(model, data_loader, criterion, device)

        # Print Hessian shape
        print(f"Hessian shape: {hessian.shape}")

        # Compute eigenvalues (for analysis)
        eigenvalues = np.linalg.eigvals(hessian.cpu().numpy())
        print(f"Eigenvalues: {eigenvalues}")

    except Exception as e:
        print(f"Error in example usage: {e}")