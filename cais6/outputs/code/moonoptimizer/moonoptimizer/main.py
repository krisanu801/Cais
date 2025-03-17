import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from typing import Dict, Tuple, List

# Dynamically add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from config import load_config
from moonlib.optimizer import DARMSprop
from moonlib.model import SimpleNN
from moonlib.evaluation import calculate_accuracy


def train_model(model: nn.Module, optimizer: optim.Optimizer, train_loader: torch.utils.data.DataLoader, config: Dict) -> Tuple[List[float], List[float]]:
    """Trains the given model.

    Args:
        model: The neural network model.
        optimizer: The optimizer.
        train_loader: DataLoader for the training dataset.
        config: Configuration dictionary.

    Returns:
        Tuple of lists containing training losses and accuracies.
    """
    model.train()
    losses = []
    accuracies = []

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predicted = torch.argmax(outputs, dim=1)
        accuracy = calculate_accuracy(predicted, labels)

        losses.append(loss.item())
        accuracies.append(accuracy)
    return losses, accuracies


def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
    """Evaluates the model on the test set.

    Args:
        model: The trained neural network model.
        test_loader: DataLoader for the test dataset.

    Returns:
        The accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def plot_decision_boundary(model: nn.Module, X: np.ndarray, y: np.ndarray, title: str, folder_path: str) -> None:
    """Plots the decision boundary of the trained model.

    Args:
        model: The trained neural network model.
        X: The input data.
        y: The labels.
        title: The title of the plot.
        folder_path: The path to save the generated plot to.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    X_grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid_tensor = torch.from_numpy(X_grid).float()
    Z = model(X_grid_tensor)
    Z = torch.argmax(Z, dim=1).reshape(xx.shape)
    Z = Z.detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    file_path = os.path.join(folder_path, f"{title.replace(' ', '_')}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Decision boundary plot saved to: {file_path}")


def run_experiment(config: Dict, folder_path: str) -> None:
    """Runs the experiment with the specified configuration.

    Args:
        config: The configuration dictionary.
        folder_path: The path to save results to.
    """

    # Create the make_moons dataset
    if not config or 'dataset' not in config:
      print("Config is empty or missing dataset parameters, using default dataset parameters")
      X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
      test_size = 0.2 # Default test size
    else:
      X, y = make_moons(n_samples=config['dataset']['n_samples'], noise=config['dataset']['noise'], random_state=42)
      test_size = config['dataset']['test_size']

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    # Create DataLoader
    batch_size = 32 # Default batch size
    if config and 'training' in config and 'batch_size' in config['training']:
        batch_size = config['training']['batch_size']
    else:
        batch_size = 32

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    hidden_size = 64 # Default hidden size
    if config and 'model' in config and 'hidden_size' in config['model']:
        hidden_size = config['model']['hidden_size']
    else:
        hidden_size = 64

    model = SimpleNN(input_size=2, hidden_size=hidden_size, num_classes=2)

    # Initialize optimizers
    if config and 'optimizer' in config and 'darms' in config['optimizer'] and 'learning_rate' in config['optimizer']['darms'] and 'beta1' in config['optimizer']['darms'] and 'beta2' in config['optimizer']['darms'] and 'weight_decay' in config:
      darms_lr = config['optimizer']['darms']['learning_rate']
      darms_beta1 = config['optimizer']['darms']['beta1']
      darms_beta2 = config['optimizer']['darms']['beta2']
      weight_decay = config['weight_decay']
    else:
      print("Optimizer config is missing or incomplete, using default values")
      darms_lr = 0.001
      darms_beta1 = 0.9
      darms_beta2 = 0.999
      weight_decay = 0.0001

    darms_optimizer = DARMSprop(model.parameters(), lr=darms_lr, beta_1=darms_beta1, beta_2=darms_beta2, weight_decay=weight_decay)

    if config and 'optimizer' in config and 'adamw' in config['optimizer'] and 'learning_rate' in config['optimizer']['adamw'] and 'weight_decay' in config:
      adamw_lr = config['optimizer']['adamw']['learning_rate']
      weight_decay = config['weight_decay']
    else:
      print("AdamW optimizer config is missing or incomplete, using default values")
      adamw_lr = 0.001
      weight_decay = 0.0001


    adamw_optimizer = optim.AdamW(model.parameters(), lr=adamw_lr, weight_decay=weight_decay)

    # Train with DA-RMSprop
    print("Training with DA-RMSprop...")
    darms_losses, darms_accuracies = train_model(model, darms_optimizer, train_loader, config)
    darms_test_accuracy = evaluate_model(model, test_loader)
    print(f"DA-RMSprop Test Accuracy: {darms_test_accuracy:.4f}")

    # Plot decision boundary for DA-RMSprop
    plot_decision_boundary(model, X, y, "DA-RMSprop Decision Boundary", folder_path)

    # Reset model weights before training with AdamW
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    # Train with AdamW
    print("Training with AdamW...")
    adamw_losses, adamw_accuracies = train_model(model, adamw_optimizer, train_loader, config)
    adamw_test_accuracy = evaluate_model(model, test_loader)
    print(f"AdamW Test Accuracy: {adamw_test_accuracy:.4f}")

    # Plot decision boundary for AdamW
    plot_decision_boundary(model, X, y, "AdamW Decision Boundary", folder_path)

    # Save training loss and accuracy data to files
    np.savetxt(os.path.join(folder_path, "darms_losses.txt"), np.array(darms_losses), delimiter=",")
    np.savetxt(os.path.join(folder_path, "darms_accuracies.txt"), np.array(darms_accuracies), delimiter=",")
    np.savetxt(os.path.join(folder_path, "adamw_losses.txt"), np.array(adamw_losses), delimiter=",")
    np.savetxt(os.path.join(folder_path, "adamw_accuracies.txt"), np.array(adamw_accuracies), delimiter=",")

    # Print and save numerical results
    results_str = f"DA-RMSprop Test Accuracy: {darms_test_accuracy:.4f}\nAdamW Test Accuracy: {adamw_test_accuracy:.4f}"
    print(results_str)

    with open(os.path.join(folder_path, "results.txt"), "w") as f:
        f.write(results_str)

    # Plot training loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(darms_losses, label="DA-RMSprop")
    plt.plot(adamw_losses, label="AdamW")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(darms_accuracies, label="DA-RMSprop")
    plt.plot(adamw_accuracies, label="AdamW")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "training_plots.png"))
    plt.close()
    print(f"Training plots saved to: {os.path.join(folder_path, 'training_plots.png')}")


if __name__ == "__main__":
    # Example usage
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = load_config(config_path)
    folder_path = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists
    run_experiment(config, folder_path)
