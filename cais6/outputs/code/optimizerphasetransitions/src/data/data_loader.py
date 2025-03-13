import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MakeMoonsDataset(Dataset):
    """
    Dataset class for the MakeMoons dataset.
    """

    def __init__(self, n_samples: int = 100, noise: float = 0.1, random_state: Optional[int] = None):
        """
        Initializes the MakeMoonsDataset.

        Args:
            n_samples (int): The number of samples to generate.
            noise (float): The amount of noise to add to the data.
            random_state (Optional[int]): The random state to use for reproducibility.
        """
        try:
            self.data, self.labels = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            self.num_classes = 2
            self.num_features = 2
        except Exception as e:
            logging.error(f"Error creating MakeMoons dataset: {e}")
            raise

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and label at the given index.
        """
        return self.data[idx], self.labels[idx]


class SimplifiedMNISTDataset(Dataset):
    """
    Dataset class for a simplified MNIST dataset (only two classes).
    """

    def __init__(self, num_classes: int = 2, pca_dim: int = 32, random_state: Optional[int] = None):
        """
        Initializes the SimplifiedMNISTDataset.

        Args:
            num_classes (int): The number of classes to include in the dataset (default: 2).
            pca_dim (int): The number of PCA components to use (default: 32).
            random_state (Optional[int]): The random state to use for PCA (default: None).
        """
        try:
            self.num_classes = num_classes
            self.pca_dim = pca_dim
            self.random_state = random_state
            self.data, self.labels = self._load_and_preprocess_data()
        except Exception as e:
            logging.error(f"Error creating SimplifiedMNIST dataset: {e}")
            raise

    def _load_and_preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the MNIST dataset, filters for the specified classes, applies PCA, and converts to tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The preprocessed data and labels.
        """
        try:
            # Load MNIST dataset
            mnist_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

            # Filter data for the first 'num_classes' digits
            indices = (mnist_dataset.targets < self.num_classes).nonzero().flatten()
            data = mnist_dataset.data[indices].float() / 255.0  # Normalize to [0, 1]
            labels = mnist_dataset.targets[indices]

            # Flatten the images
            data = data.view(data.size(0), -1)

            # Apply PCA
            pca = PCA(n_components=self.pca_dim, random_state=self.random_state)
            data_pca = pca.fit_transform(data.numpy())
            data_pca = torch.tensor(data_pca, dtype=torch.float32)

            return data_pca, labels.long()
        except Exception as e:
            logging.error(f"Error loading and preprocessing MNIST data: {e}")
            raise

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and label at the given index.
        """
        return self.data[idx], self.labels[idx]


# Example Usage:
if __name__ == '__main__':
    # MakeMoons Example
    try:
        moons_dataset = MakeMoonsDataset(n_samples=200, noise=0.2, random_state=42)
        print(f"MakeMoons dataset size: {len(moons_dataset)}")
        data, label = moons_dataset[0]
        print(f"MakeMoons data shape: {data.shape}, label: {label}")
    except Exception as e:
        print(f"Error in MakeMoons example: {e}")

    # SimplifiedMNIST Example
    try:
        mnist_dataset = SimplifiedMNISTDataset(num_classes=2, pca_dim=32, random_state=42)
        print(f"SimplifiedMNIST dataset size: {len(mnist_dataset)}")
        data, label = mnist_dataset[0]
        print(f"SimplifiedMNIST data shape: {data.shape}, label: {label}")
    except Exception as e:
        print(f"Error in SimplifiedMNIST example: {e}")