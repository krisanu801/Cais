import sys
import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fit_lens_model(data: Dict[str, Any], time_delay: float) -> Dict[str, Any]:
    """
    Fits a lens mass model to the provided data using a Bayesian framework.

    Args:
        data: A dictionary containing observational data (image positions, flux ratios,
              kinematic data, etc.). Expected keys: 'image_positions', 'flux_ratios',
              'kinematic_data'.
        time_delay: The estimated time delay between the lensed images.

    Returns:
        A dictionary containing the best-fit lens model parameters and associated
        uncertainties.

    Raises:
        ValueError: If the input data is not in the expected format.
        Exception: For any other errors during lens model fitting.
    """
    try:
        if not all(key in data for key in ['image_positions', 'flux_ratios', 'kinematic_data']):
            raise ValueError("Input data must contain 'image_positions', 'flux_ratios', and 'kinematic_data' keys.")

        image_positions = data['image_positions']
        flux_ratios = data['flux_ratios']
        kinematic_data = data['kinematic_data']

        # Define the lens model (example: Singular Isothermal Ellipsoid - SIE)
        def sie_potential(x: np.ndarray, y: np.ndarray, x0: float, y0: float, q: float, phi: float, b: float) -> np.ndarray:
            """Calculates the SIE lens potential."""
            xp = (x - x0) * np.cos(phi) + (y - y0) * np.sin(phi)
            yp = -(x - x0) * np.sin(phi) + (y - y0) * np.cos(phi)
            psi = np.sqrt(xp**2 * q + yp**2 / q)
            return b * psi

        # Define the lens model parameters
        def lens_model(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Combines the lens potential with the source position."""
            x0, y0, q, phi, b, source_x, source_y = theta
            return sie_potential(x, y, x0, y0, q, phi, b) - ((x - source_x)**2 + (y - source_y)**2) / 2  # Fermat potential

        # Define the prior distribution
        def log_prior(theta: np.ndarray) -> float:
            """Defines the prior probabilities for the lens model parameters."""
            x0, y0, q, phi, b, source_x, source_y = theta
            if not (-10 < x0 < 10 and -10 < y0 < 10 and 0 < q < 1 and 0 < phi < np.pi and 0 < b < 5 and -10 < source_x < 10 and -10 < source_y < 10):
                return -np.inf  # Return -inf if parameters are outside the prior range
            return 0.0  # Uniform prior

        # Define the likelihood function
        def log_likelihood(theta: np.ndarray, x_obs: np.ndarray, y_obs: np.ndarray, flux_obs: np.ndarray) -> float:
            """Calculates the log-likelihood of the lens model given the observed data."""
            try:
                # Calculate the predicted image positions
                predicted_potential = lens_model(theta, x_obs, y_obs)

                # Calculate the predicted flux ratios (example: using magnification)
                # This is a simplified example and should be replaced with a more accurate calculation
                magnification = np.gradient(np.gradient(predicted_potential))
                predicted_flux_ratios = magnification[0] / magnification  # Example

                # Calculate the likelihood based on the flux ratios
                log_likelihood = np.sum(norm.logpdf(flux_obs, loc=predicted_flux_ratios, scale=0.1))  # Example scale

                return log_likelihood
            except Exception as e:
                logging.error(f"Error in log_likelihood: {e}", exc_info=True)
                return -np.inf

        # Define the posterior distribution
        def log_probability(theta: np.ndarray, x_obs: np.ndarray, y_obs: np.ndarray, flux_obs: np.ndarray) -> float:
            """Calculates the log-posterior probability."""
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, x_obs, y_obs, flux_obs)

        # Prepare the data for the optimization
        x_obs = np.array([pos[0] for pos in image_positions])
        y_obs = np.array([pos[1] for pos in image_positions])
        flux_obs = np.array(flux_ratios)

        # Initial parameter values
        initial = np.array([0.1, 0.1, 0.5, 0.5, 1.0, 0.1, 0.1])

        # Run the optimization (using MCMC)
        nwalkers = 32
        ndim = len(initial)
        pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x_obs, y_obs, flux_obs))
        sampler.run_mcmc(pos, 5000, progress=True)

        # Analyze the results
        samples = sampler.get_chain(discard=1000, thin=15, flat=True)

        # Calculate the best-fit parameters and uncertainties
        best_fit_params = np.median(samples, axis=0)
        param_uncertainties = np.std(samples, axis=0)

        # Create a corner plot
        fig = corner.corner(samples, labels=["x0", "y0", "q", "phi", "b", "source_x", "source_y"])
        corner_plot_path = os.path.join("/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results", "corner_plot.png")
        plt.savefig(corner_plot_path)
        plt.close()
        logging.info(f"Corner plot saved to: {corner_plot_path}")

        # Store the results
        lens_model_results = {
            'best_fit_params': best_fit_params,
            'param_uncertainties': param_uncertainties,
            'samples': samples
        }

        logging.info("Lens model fitted successfully.")
        return lens_model_results

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during lens model fitting: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Example Usage:
    # 1. Create dummy data for image positions, flux ratios, and kinematic data.
    # 2.  Run this script.

    # Create dummy data
    image_positions = [(1.0, 1.0), (-1.0, -1.0), (0.5, -0.5), (-0.5, 0.5)]
    flux_ratios = [1.0, 0.5, 0.3, 0.2]
    kinematic_data = {'velocity_dispersion': 200, 'error': 10}  # Example

    data = {
        'image_positions': image_positions,
        'flux_ratios': flux_ratios,
        'kinematic_data': kinematic_data
    }

    try:
        time_delay = 10.0  # Example time delay
        lens_model_results = fit_lens_model(data, time_delay)
        print("Lens Model Results:")
        print(lens_model_results['best_fit_params'])
        print(lens_model_results['param_uncertainties'])
    except Exception as e:
        print(f"Error: {e}")