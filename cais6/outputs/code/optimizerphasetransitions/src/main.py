import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from src.data.data_loader import MakeMoonsDataset, SimplifiedMNISTDataset
from src.models.model import MLP
from src.optimizers.learning_rate_schedulers import get_scheduler
from src.analysis.metrics import calculate_accuracy, calculate_auc
from src.analysis.loss_landscape import plot_loss_landscape
from src.analysis.hessian import approximate_hessian
from src.utils.config import load_config
from src.utils.utils import set_seed, create_directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define output directory
OUTPUT_DIR = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"

def train_model(model: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: Any,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: Dict[str, Any],
                device: torch.device) -> Tuple[List[float], List[float]]:
    """
    Trains the given model.

    Args:
        model (nn.Module): The model to train.
        optimizer (optim.Optimizer): The optimizer to use.
        scheduler (Any): The learning rate scheduler.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        config (Dict[str, Any]): The configuration dictionary.
        device (torch.device): The device to train on.

    Returns:
        Tuple[List[float], List[float]]: Training and validation losses.
    """
    criterion = nn.CrossEntropyLoss()
    epochs = config['training']['epochs']
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float('inf')
    patience = config['training']['patience']
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info("Early stopping triggered.")
                break

    return train_losses, val_losses


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the given model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): The test data loader.
        device (torch.device): The device to evaluate on.

    Returns:
        Tuple[float, float]: Accuracy and AUC.
    """
    model.eval()
    all_labels: List[int] = []
    all_predictions: List[float] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(probabilities)

    accuracy = calculate_accuracy(all_labels, np.round(all_predictions))
    auc = calculate_auc(all_labels, all_predictions)

    logging.info(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")
    return accuracy, auc


def main(config: Dict[str, Any]) -> None:
    """
    Main function to run the training, evaluation, and analysis.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    try:
        # Set seed for reproducibility
        set_seed(config['seed'])

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Dataset loading
        dataset_name = config['dataset']['name']
        if dataset_name == 'MakeMoons':
            dataset = MakeMoonsDataset(n_samples=config['dataset']['n_samples'],
                                        noise=config['dataset']['noise'],
                                        random_state=config['seed'])
        elif dataset_name == 'MNIST':
            dataset = SimplifiedMNISTDataset(num_classes=config['dataset']['num_classes'],
                                             pca_dim=config['dataset']['pca_dim'])
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Data splitting
        
        train_size = int(config['training']['train_ratio'] * len(dataset))
        val_size = int(config['training']['val_ratio'] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])



        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        # Model instantiation
        input_dim = dataset.num_features if dataset_name == 'MakeMoons' else dataset.pca_dim
        model = MLP(input_dim=input_dim,
                    hidden_dims=config['model']['hidden_dims'],
                    output_dim=dataset.num_classes).to(device)
        

        # Optimizer instantiation
        optimizer_name = config['optimizer']['name']
        lr = config['optimizer']['learning_rate']
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Learning rate scheduler instantiation
        scheduler_config = config['scheduler']
        scheduler = get_scheduler(optimizer, scheduler_config)
        

        # Training
        train_losses, val_losses = train_model(model, optimizer, scheduler, train_loader, val_loader, config, device)

        # Evaluation
        accuracy, auc = evaluate_model(model, test_loader, device)

        # Phase Transition Analysis
        if config['analysis']['plot_loss_landscape']:
            try:
                # Create the output directory if it doesn't exist
                create_directory(OUTPUT_DIR)

                # Plot loss landscape
                plot_loss_landscape(model, test_loader, criterion=nn.CrossEntropyLoss(), device=device,
                                    filename=os.path.join(OUTPUT_DIR, "loss_landscape.png"))

                # Approximate Hessian
                hessian = approximate_hessian(model, test_loader, nn.CrossEntropyLoss(), device)
                eigenvalues = np.linalg.eigvals(hessian.cpu().numpy())
                plt.figure()
                plt.hist(eigenvalues, bins=50)
                plt.title("Hessian Eigenvalue Distribution")
                plt.xlabel("Eigenvalue")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(OUTPUT_DIR, "hessian_eigenvalues.png"))
                plt.close()

                logging.info("Loss landscape and Hessian analysis completed.")

            except Exception as e:
                logging.error(f"Error during phase transition analysis: {e}")

        logging.info("Training and evaluation completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example Usage:
    # python src/main.py --config configs/config.yaml
    parser = argparse.ArgumentParser(description='Train and evaluate a model.')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run main function
    main(config)