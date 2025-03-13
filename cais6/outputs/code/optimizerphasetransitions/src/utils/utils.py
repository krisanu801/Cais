import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logging.info(f"Set seed to {seed}")
    except Exception as e:
        logging.error(f"Error setting seed: {e}")


def save_model(model: nn.Module, path: str) -> None:
    """
    Saves the model to the specified path.

    Args:
        model (nn.Module): The model to save.
        path (str): The path to save the model to.
    """
    try:
        torch.save(model.state_dict(), path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


def load_model(model: nn.Module, path: str, device: torch.device) -> None:
    """
    Loads the model from the specified path.

    Args:
        model (nn.Module): The model to load the state into.
        path (str): The path to load the model from.
        device (torch.device): The device to load the model onto.
    """
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        logging.info(f"Model loaded from {path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")


def create_directory(path: str) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        path (str): The path to the directory.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
        else:
            logging.info(f"Directory already exists: {path}")
    except Exception as e:
        logging.error(f"Error creating directory: {e}")


# Example Usage:
if __name__ == '__main__':
    try:
        # Set seed
        set_seed(42)

        # Create a dummy model
        model = nn.Linear(10, 2)

        # Define paths
        model_path = "dummy_model.pth"
        directory_path = "test_directory"

        # Save the model
        save_model(model, model_path)

        # Create a directory
        create_directory(directory_path)

        # Load the model
        device = torch.device('cpu')
        loaded_model = nn.Linear(10, 2)
        load_model(loaded_model, model_path, device)

        print("Utility functions executed successfully.")

        # Clean up (remove created files/directories)
        os.remove(model_path)
        os.rmdir(directory_path)

    except Exception as e:
        print(f"Error in example usage: {e}")