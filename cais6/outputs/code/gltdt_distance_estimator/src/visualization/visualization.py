import sys
import os
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_light_curves(time1: np.ndarray, flux1: np.ndarray, time2: np.ndarray, flux2: np.ndarray,
                      output_path: str = "light_curves.png") -> None:
    """
    Plots two light curves on the same axes.

    Args:
        time1: Time values for the first light curve.
        flux1: Flux values for the first light curve.
        time2: Time values for the second light curve.
        flux2: Flux values for the second light curve.
        output_path: The path to save the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(time1, flux1, label='Light Curve 1')
        plt.plot(time2, flux2, label='Light Curve 2')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Light Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Light curves plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting light curves: {e}", exc_info=True)
        raise


def plot_lens_model(image: np.ndarray, lens_model_params: Dict[str, Any],
                    output_path: str = "lens_model.png") -> None:
    """
    Plots the image of the lensing galaxy and overlays the lens model.

    Args:
        image: The image of the lensing galaxy (numpy array).
        lens_model_params: A dictionary containing the lens model parameters.
        output_path: The path to save the plot.
    """
    try:
        plt.figure(figsize=(8, 8))
        plt.imshow(image, origin='lower', cmap='gray')  # Adjust cmap as needed
        plt.title('Lensing Galaxy with Lens Model')

        # Overlay the lens model (example: plotting the Einstein radius)
        # This is a placeholder and should be replaced with a more accurate
        # representation of the lens model.
        # Example: Plot a circle representing the Einstein radius
        # center_x = lens_model_params.get('x0', image.shape[1] // 2)
        # center_y = lens_model_params.get('y0', image.shape[0] // 2)
        # einstein_radius = lens_model_params.get('b', 10)  # Example parameter
        # circle = plt.Circle((center_x, center_y), einstein_radius, color='red', fill=False)
        # plt.gca().add_patch(circle)

        plt.savefig(output_path)
        plt.close()
        logging.info(f"Lens model plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting lens model: {e}", exc_info=True)
        raise


def plot_hubble_constant(hubble_constant: float, output_path: str = "hubble_constant.png") -> None:
    """
    Plots the calculated Hubble constant with error bars (if available).

    Args:
        hubble_constant: The calculated Hubble constant value.
        output_path: The path to save the plot.
    """
    try:
        plt.figure(figsize=(6, 4))
        plt.errorbar(0, hubble_constant, yerr=0, fmt='o', capsize=5)  # Replace yerr with actual error if available
        plt.xlim(-1, 1)
        plt.ylim(0, 100)  # Adjust y-axis limits as needed
        plt.xlabel('')
        plt.ylabel('Hubble Constant (km/s/Mpc)')
        plt.title('Hubble Constant Estimate')
        plt.xticks([])  # Remove x-axis ticks
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Hubble constant plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting Hubble constant: {e}", exc_info=True)
        raise


def plot_results(data: Dict[str, Any], lens_model: Dict[str, Any], distance: float, output_dir: str) -> None:
    """
    Orchestrates the plotting of various results.

    Args:
        data: A dictionary containing the data used for the analysis.
        lens_model: A dictionary containing the lens model parameters.
        distance: The calculated distance (Hubble constant).
        output_dir: The directory to save the plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Example: Plot light curves
        if 'time1' in data and 'flux1' in data and 'time2' in data and 'flux2' in data:
            light_curves_path = os.path.join(output_dir, "light_curves.png")
            plot_light_curves(data['time1'], data['flux1'], data['time2'], data['flux2'], light_curves_path)

        # Example: Plot lens model
        if 'lensing_galaxy_images' in data:
            lens_model_path = os.path.join(output_dir, "lens_model.png")
            plot_lens_model(data['lensing_galaxy_images'][0], lens_model, lens_model_path)

        # Example: Plot Hubble constant
        hubble_constant_path = os.path.join(output_dir, "hubble_constant.png")
        plot_hubble_constant(distance, hubble_constant_path)

        logging.info("Results plotted and saved.")

    except Exception as e:
        logging.error(f"Error plotting results: {e}", exc_info=True)
        raise


def plot_image(image: np.ndarray, title: str, output_path: str) -> None:
    """
    Plots a 2D numpy array as an image.

    Args:
        image: The 2D numpy array representing the image.
        title: The title of the plot.
        output_path: The path to save the plot.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='viridis', origin='lower')  # Adjust cmap as needed
        plt.colorbar()
        plt.title(title)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Image plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting image: {e}", exc_info=True)
        raise


def plot_data_distribution(data: pd.DataFrame, column: str, output_path: str) -> None:
    """
    Plots the distribution of a column in a pandas DataFrame using a histogram.

    Args:
        data: The pandas DataFrame.
        column: The name of the column to plot.
        output_path: The path to save the plot.
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Data distribution plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting data distribution: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Example Usage:
    # 1. Create dummy data for light curves and lens model.
    # 2.  Run this script.

    # Create dummy data
    time1 = np.linspace(0, 10, 100)
    flux1 = np.sin(time1) + np.random.normal(0, 0.1, 100)
    time2 = np.linspace(1, 11, 100)
    flux2 = np.sin(time2) + np.random.normal(0, 0.1, 100)

    lens_model_params = {'x0': 50, 'y0': 50, 'b': 20}  # Example parameters
    image = np.random.rand(100, 100)  # Example image

    data = {
        'time1': time1,
        'flux1': flux1,
        'time2': time2,
        'flux2': flux2,
        'lensing_galaxy_images': [image]
    }

    output_dir = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"
    os.makedirs(output_dir, exist_ok=True)

    try:
        plot_results(data, lens_model_params, 73.5, output_dir)  # Example Hubble constant value
        print("Visualization completed successfully.")
    except Exception as e:
        print(f"Error: {e}")