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
import torch.optim as optim
from src.optimizers.learning_rate_schedulers import get_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestLearningRateSchedulers(unittest.TestCase):
    """
    Unit tests for the learning rate schedulers.
    """

    def setUp(self):
        """
        Set up method to create a dummy model and optimizer for testing.
        """
        self.model = nn.Linear(10, 2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def test_exponential_lr(self):
        """
        Tests the ExponentialLR scheduler.
        """
        try:
            config = {'name': 'ExponentialLR', 'gamma': 0.9}
            scheduler = get_scheduler(self.optimizer, config)
            self.assertIsNotNone(scheduler)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

            # Check learning rate after one step
            initial_lr = self.optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            self.assertAlmostEqual(new_lr, initial_lr * 0.9)

        except Exception as e:
            logging.error(f"Error in test_exponential_lr: {e}")
            self.fail(f"ExponentialLR test failed: {e}")

    def test_cosine_annealing_lr(self):
        """
        Tests the CosineAnnealingLR scheduler.
        """
        try:
            config = {'name': 'CosineAnnealingLR', 'T_max': 10, 'eta_min': 0.001}
            scheduler = get_scheduler(self.optimizer, config)
            self.assertIsNotNone(scheduler)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

        except Exception as e:
            logging.error(f"Error in test_cosine_annealing_lr: {e}")
            self.fail(f"CosineAnnealingLR test failed: {e}")

    def test_step_lr(self):
        """
        Tests the StepLR scheduler.
        """
        try:
            config = {'name': 'StepLR', 'step_size': 5, 'gamma': 0.5}
            scheduler = get_scheduler(self.optimizer, config)
            self.assertIsNotNone(scheduler)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)

            # Check learning rate after step_size steps
            initial_lr = self.optimizer.param_groups[0]['lr']
            for _ in range(5):
                scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            self.assertAlmostEqual(new_lr, initial_lr * 0.5)

        except Exception as e:
            logging.error(f"Error in test_step_lr: {e}")
            self.fail(f"StepLR test failed: {e}")

    def test_reduce_lr_on_plateau(self):
        """
        Tests the ReduceLROnPlateau scheduler.
        """
        try:
            config = {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.2, 'patience': 5}
            scheduler = get_scheduler(self.optimizer, config)
            self.assertIsNotNone(scheduler)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        except Exception as e:
            logging.error(f"Error in test_reduce_lr_on_plateau: {e}")
            self.fail(f"ReduceLROnPlateau test failed: {e}")

    def test_no_scheduler(self):
        """
        Tests the case where no scheduler is specified.
        """
        try:
            config = {'name': 'None'}
            scheduler = get_scheduler(self.optimizer, config)
            self.assertIsNone(scheduler)

        except Exception as e:
            logging.error(f"Error in test_no_scheduler: {e}")
            self.fail(f"No scheduler test failed: {e}")

    def test_unsupported_scheduler(self):
        """
        Tests the case where an unsupported scheduler is specified.
        """
        try:
            config = {'name': 'InvalidScheduler'}
            with self.assertRaises(ValueError):
                get_scheduler(self.optimizer, config)

        except Exception as e:
            logging.error(f"Error in test_unsupported_scheduler: {e}")
            self.fail(f"Unsupported scheduler test failed: {e}")


# Example Usage:
if __name__ == '__main__':
    try:
        unittest.main()
    except Exception as e:
        print(f"Error running tests: {e}")