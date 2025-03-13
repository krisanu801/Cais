import sys
import os
import logging
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src import config
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have the project root in your PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the output folder path
OUTPUT_FOLDER = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"


def generate_localization_uncertainty(true_location: Tuple[float, float], uncertainty_level: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a probability distribution representing the localization uncertainty.

    Args:
        true_location: A tuple containing the (RA, Dec) of the true source location.
        uncertainty_level: A measure of the uncertainty in degrees.

    Returns:
        A tuple containing the RA and Dec values of the uncertainty distribution.
    """
    try:
        ra, dec = true_location
        # Generate a Gaussian distribution around the true location
        num_samples = 1000
        ra_samples = np.random.normal(ra, uncertainty_level, num_samples)
        dec_samples = np.random.normal(dec, uncertainty_level, num_samples)

        logging.info("Localization uncertainty distribution generated.")
        return ra_samples, dec_samples

    except Exception as e:
        logging.error(f"Error generating localization uncertainty distribution: {e}")
        return np.array([]), np.array([])


def visualize_localization_uncertainty(ra_samples: np.ndarray, dec_samples: np.ndarray, filename: str = "localization_uncertainty.png") -> None:
    """
    Visualizes the localization uncertainty distribution.

    Args:
        ra_samples: An array of RA values.
        dec_samples: An array of Dec values.
        filename: The name of the file to save the plot to.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(ra_samples, dec_samples, s=5, alpha=0.5)
        plt.xlabel("Right Ascension (RA)")
        plt.ylabel("Declination (Dec)")
        plt.title("Localization Uncertainty Distribution")
        plt.grid(True)

        # Save the plot to the output folder
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        plt.savefig(filepath)
        plt.close()

        logging.info(f"Localization uncertainty visualization saved to {filepath}")

    except Exception as e:
        logging.error(f"Error visualizing localization uncertainty: {e}")


def perform_localization() -> Dict:
    """
    Performs source localization, accounting for uncertainty.

    Returns:
        A dictionary containing the localization data.
    """
    try:
        # Example parameters (replace with realistic values)
        true_location = (120.0, 45.0)  # RA, Dec
        uncertainty_level = 5.0  # degrees

        # Generate localization uncertainty distribution
        ra_samples, dec_samples = generate_localization_uncertainty(true_location, uncertainty_level)

        # Visualize the uncertainty distribution
        visualize_localization_uncertainty(ra_samples, dec_samples)

        localization_data = {
            "true_location": true_location,
            "uncertainty_level": uncertainty_level,
            "ra_samples": ra_samples,
            "dec_samples": dec_samples
        }

        logging.info("Source localization performed successfully.")
        return localization_data

    except Exception as e:
        logging.error(f"Error performing source localization: {e}")
        return {}  # Return an empty dictionary in case of error


if __name__ == "__main__":
    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/localization/localization.py
    # 3.  Check the logs folder for logging information
    # 4.  Check the outputs folder for the localization uncertainty plot

    localization_data = perform_localization()
    if localization_data:
        print("Localization Data:")
        for key, value in localization_data.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: Array of shape {value.shape}")
            else:
                print(f"{key}: {value}")