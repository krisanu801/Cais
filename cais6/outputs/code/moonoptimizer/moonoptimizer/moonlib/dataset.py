from sklearn.datasets import make_moons
import numpy as np
from typing import Tuple

def load_moon_data(n_samples: int = 100, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates the make_moons dataset.

    Args:
        n_samples: The number of samples to generate.
        noise: The standard deviation of the Gaussian noise added to the data.

    Returns:
        A tuple containing the input data and the labels.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y