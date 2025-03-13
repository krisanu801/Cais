import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import torch
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.optim import Optimizer
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> Optional[Any]:
    """
    Returns a learning rate scheduler based on the given configuration.

    Args:
        optimizer (Optimizer): The optimizer to use with the scheduler.
        config (Dict[str, Any]): The configuration dictionary for the scheduler.

    Returns:
        Optional[Any]: The learning rate scheduler, or None if no scheduler is specified.
    """
    try:
        scheduler_name = config.get('name')

        if scheduler_name is None or scheduler_name == 'None':
            return None

        if scheduler_name == 'ExponentialLR':
            gamma = config.get('gamma', 0.9)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            logging.info(f"Using ExponentialLR scheduler with gamma={gamma}")
            return scheduler

        if scheduler_name == 'PolynomialLR':
            # Custom implementation needed as PyTorch doesn't have a built-in PolynomialLR
            # This is a placeholder.  A custom implementation would go here.
            logging.warning("PolynomialLR is not implemented. Returning None.")
            return None

        if scheduler_name == 'CosineAnnealingLR':
            T_max = config.get('T_max', 10)
            eta_min = config.get('eta_min', 0)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            logging.info(f"Using CosineAnnealingLR scheduler with T_max={T_max}, eta_min={eta_min}")
            return scheduler

        if scheduler_name == 'StepLR':
            step_size = config.get('step_size', 30)
            gamma = config.get('gamma', 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            logging.info(f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}")
            return scheduler

        if scheduler_name == 'ReduceLROnPlateau':
            mode = config.get('mode', 'min')
            factor = config.get('factor', 0.1)
            patience = config.get('patience', 10)
            threshold = config.get('threshold', 1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)
            logging.info(f"Using ReduceLROnPlateau scheduler with mode={mode}, factor={factor}, patience={patience}, threshold={threshold}")
            return scheduler

        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    except Exception as e:
        logging.error(f"Error creating scheduler: {e}")
        raise


# Example Usage:
if __name__ == '__main__':
    try:
        # Dummy optimizer
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Example 1: ExponentialLR
        config_exp = {'name': 'ExponentialLR', 'gamma': 0.95}
        scheduler_exp = get_scheduler(optimizer, config_exp)
        print(f"ExponentialLR scheduler: {scheduler_exp}")

        # Example 2: CosineAnnealingLR
        config_cos = {'name': 'CosineAnnealingLR', 'T_max': 20, 'eta_min': 0.001}
        scheduler_cos = get_scheduler(optimizer, config_cos)
        print(f"CosineAnnealingLR scheduler: {scheduler_cos}")

        # Example 3: StepLR
        config_step = {'name': 'StepLR', 'step_size': 5, 'gamma': 0.5}
        scheduler_step = get_scheduler(optimizer, config_step)
        print(f"StepLR scheduler: {scheduler_step}")

        # Example 4: ReduceLROnPlateau
        config_plateau = {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 5}
        scheduler_plateau = get_scheduler(optimizer, config_plateau)
        print(f"ReduceLROnPlateau scheduler: {scheduler_plateau}")

        # Example 5: No scheduler
        config_none = {'name': 'None'}
        scheduler_none = get_scheduler(optimizer, config_none)
        print(f"No scheduler: {scheduler_none}")

    except Exception as e:
        print(f"Error in example usage: {e}")