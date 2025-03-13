import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Tuple, List, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def learning_rate_finder(model: nn.Module,
                         optimizer: Optimizer,
                         criterion: Callable,
                         data_loader: DataLoader,
                         device: torch.device,
                         num_iterations: int = 100,
                         start_lr: float = 1e-7,
                         end_lr: float = 10,
                         beta: float = 0.98) -> Tuple[List[float], List[float]]:
    """
    Performs a learning rate range test to find an optimal learning rate.

    Args:
        model (nn.Module): The model to train.
        optimizer (Optimizer): The optimizer to use.
        criterion (Callable): The loss function.
        data_loader (DataLoader): The data loader.
        device (torch.device): The device to train on.
        num_iterations (int): The number of iterations to run.
        start_lr (float): The starting learning rate.
        end_lr (float): The ending learning rate.
        beta (float): The smoothing factor for the loss.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing the learning rates and smoothed losses.
    """
    try:
        # Store original model state
        original_model_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

        # Store original optimizer state
        original_optimizer_state = {k: v.clone().detach().cpu() for k, v in optimizer.state_dict().items()}

        # Reset the model and optimizer to their original states
        def reset_model_and_optimizer():
            model.load_state_dict(original_model_state)
            optimizer.load_state_dict(original_optimizer_state)

        reset_model_and_optimizer()

        # Learning rate schedule
        lr_schedule = torch.exp(torch.linspace(torch.log(torch.tensor(start_lr)), torch.log(torch.tensor(end_lr)), num_iterations)).to(device)

        # Initialize variables
        losses: List[float] = []
        learning_rates: List[float] = []
        avg_loss = 0.0
        best_loss = 0.0

        for i, lr in enumerate(lr_schedule):
            optimizer.param_groups[0]['lr'] = lr
            learning_rates.append(lr.item())

            # Get batch
            try:
                inputs, labels = next(iter(data_loader))
                inputs, labels = inputs.to(device), labels.to(device)
            except StopIteration:
                logging.warning("DataLoader exhausted during learning rate finder. Resetting DataLoader.")
                reset_model_and_optimizer()
                return learning_rate_finder(model, optimizer, criterion, data_loader, device, num_iterations, start_lr, end_lr, beta)

            # Train step
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_item = loss.item()

            # Smooth the loss
            avg_loss = beta * avg_loss + (1 - beta) * loss_item
            smoothed_loss = avg_loss / (1 - beta**(i+1))
            losses.append(smoothed_loss)

            # Check if loss is exploding
            if i > 0 and smoothed_loss > 4 * best_loss:
                logging.warning("Loss is exploding, stopping learning rate finder.")
                break

            # Record the best loss
            if smoothed_loss < best_loss or i == 0:
                best_loss = smoothed_loss

            # Backpropagate
            loss.backward()
            optimizer.step()

        reset_model_and_optimizer()
        return learning_rates, losses

    except Exception as e:
        logging.error(f"Error during learning rate finder: {e}")
        raise


# Example Usage:
if __name__ == '__main__':
    try:
        # Dummy model, optimizer, and data loader
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()
        data = torch.randn(64, 10)
        labels = torch.randint(0, 2, (64,))
        dataset = [(data[i], labels[i]) for i in range(64)]
        data_loader = DataLoader(dataset, batch_size=32)
        device = torch.device('cpu')

        # Run learning rate finder
        learning_rates, losses = learning_rate_finder(model, optimizer, criterion, data_loader, device)

        # Print results
        print(f"Learning rates: {learning_rates}")
        print(f"Losses: {losses}")

        # Plot the results (requires matplotlib)
        import matplotlib.pyplot as plt
        plt.plot(learning_rates, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Smoothed Loss')
        plt.title('Learning Rate Finder')
        plt.show()

    except Exception as e:
        print(f"Error in example usage: {e}")