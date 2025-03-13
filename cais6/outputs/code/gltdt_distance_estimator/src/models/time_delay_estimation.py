import sys
import os
import logging
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.optimize import minimize
import george
from george import kernels
import pywt

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def estimate_time_delay(data: Dict[str, Any]) -> float:
    """
    Estimates the time delay between two light curves using multiple methods.

    Args:
        data: A dictionary containing the light curves and their associated times.
              Expected keys: 'time1', 'flux1', 'error1', 'time2', 'flux2', 'error2'.

    Returns:
        The estimated time delay.

    Raises:
        ValueError: If the input data is not in the expected format.
        Exception: For any other errors during time delay estimation.
    """
    try:
        if not all(key in data for key in ['time1', 'flux1', 'error1', 'time2', 'flux2', 'error2']):
            raise ValueError("Input data must contain 'time1', 'flux1', 'error1', 'time2', 'flux2', 'error2' keys.")

        time1 = data['time1']
        flux1 = data['flux1']
        error1 = data['error1']
        time2 = data['time2']
        flux2 = data['flux2']
        error2 = data['error2']

        # Estimate time delay using cross-correlation
        time_delay_cc = cross_correlation(time1, flux1, time2, flux2)
        logging.info(f"Time delay estimated using cross-correlation: {time_delay_cc}")

        # Estimate time delay using dispersion minimization
        time_delay_dm = dispersion_minimization(time1, flux1, time2, flux2)
        logging.info(f"Time delay estimated using dispersion minimization: {time_delay_dm}")

        # Estimate time delay using Gaussian Process regression
        time_delay_gp = gaussian_process_regression(time1, flux1, error1, time2, flux2, error2)
        logging.info(f"Time delay estimated using Gaussian Process regression: {time_delay_gp}")

        # Estimate time delay using Wavelet Transform
        time_delay_wt = wavelet_transform(time1, flux1, time2, flux2)
        logging.info(f"Time delay estimated using Wavelet Transform: {time_delay_wt}")

        # Combine the estimates (example: simple average)
        time_delay = np.mean([time_delay_cc, time_delay_dm, time_delay_gp, time_delay_wt])
        logging.info(f"Combined time delay estimate: {time_delay}")

        return time_delay

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during time delay estimation: {e}", exc_info=True)
        raise


def cross_correlation(time1: np.ndarray, flux1: np.ndarray, time2: np.ndarray, flux2: np.ndarray) -> float:
    """
    Estimates the time delay between two light curves using cross-correlation.

    Args:
        time1: Time values for the first light curve.
        flux1: Flux values for the first light curve.
        time2: Time values for the second light curve.
        flux2: Flux values for the second light curve.

    Returns:
        The estimated time delay.
    """
    try:
        # Interpolate the light curves onto a common time grid
        time_common = np.linspace(max(time1.min(), time2.min()), min(time1.max(), time2.max()), 1000)
        flux1_interp = np.interp(time_common, time1, flux1)
        flux2_interp = np.interp(time_common, time2, flux2)

        # Calculate the cross-correlation
        correlation = correlate(flux1_interp, flux2_interp, mode='full')
        lags = np.arange(-len(flux1_interp) + 1, len(flux1_interp))
        lag_time = lags[np.argmax(correlation)] * (time_common[1] - time_common[0])

        return lag_time
    except Exception as e:
        logging.error(f"Error in cross-correlation: {e}", exc_info=True)
        raise


def dispersion_minimization(time1: np.ndarray, flux1: np.ndarray, time2: np.ndarray, flux2: np.ndarray) -> float:
    """
    Estimates the time delay between two light curves using dispersion minimization.

    Args:
        time1: Time values for the first light curve.
        flux1: Flux values for the first light curve.
        time2: Time values for the second light curve.
        flux2: Flux values for the second light curve.

    Returns:
        The estimated time delay.
    """
    try:
        def dispersion(tau: float) -> float:
            """Calculates the dispersion for a given time delay."""
            flux2_shifted = np.interp(time1, time2 + tau, flux2, left=np.nan, right=np.nan)
            valid_indices = ~np.isnan(flux2_shifted)
            if np.sum(valid_indices) < 2:
                return np.inf  # Return a large value if not enough overlap
            return np.nanstd(flux1[valid_indices] - flux2_shifted[valid_indices])

        # Optimize the time delay
        result = minimize(dispersion, x0=0.0, method='Nelder-Mead')
        return result.x[0]
    except Exception as e:
        logging.error(f"Error in dispersion minimization: {e}", exc_info=True)
        raise


def gaussian_process_regression(time1: np.ndarray, flux1: np.ndarray, error1: np.ndarray,
                                 time2: np.ndarray, flux2: np.ndarray, error2: np.ndarray) -> float:
    """
    Estimates the time delay between two light curves using Gaussian Process regression.

    Args:
        time1: Time values for the first light curve.
        flux1: Flux values for the first light curve.
        error1: Error values for the first light curve.
        time2: Time values for the second light curve.
        flux2: Flux values for the second light curve.
        error2: Error values for the second light curve.

    Returns:
        The estimated time delay.
    """
    try:
        # Define a Gaussian Process kernel
        kernel = kernels.ExpSquaredKernel(metric=10.0)

        # Define the Gaussian Process
        gp1 = george.GP(kernel, mean=np.mean(flux1))
        gp2 = george.GP(kernel, mean=np.mean(flux2))

        # Compute the Gaussian Process
        gp1.compute(time1, error1)
        gp2.compute(time2, error2)

        # Define the objective function to minimize
        def neg_log_likelihood(p: np.ndarray) -> float:
            gp2.compute(time2 + p[0], error2)
            return -gp2.log_likelihood(flux2)

        # Optimize the time delay
        result = minimize(neg_log_likelihood, x0=[0.0], method="L-BFGS-B")
        return result.x[0]
    except Exception as e:
        logging.error(f"Error in Gaussian Process regression: {e}", exc_info=True)
        raise


def wavelet_transform(time1: np.ndarray, flux1: np.ndarray, time2: np.ndarray, flux2: np.ndarray) -> float:
    """
    Estimates the time delay between two light curves using Wavelet Transform.

    Args:
        time1: Time values for the first light curve.
        flux1: Flux values for the first light curve.
        time2: Time values for the second light curve.
        flux2: Flux values for the second light curve.

    Returns:
        The estimated time delay.
    """
    try:
        # Perform wavelet decomposition
        coeffs1 = pywt.wavedec(flux1, 'db4', level=5)
        coeffs2 = pywt.wavedec(flux2, 'db4', level=5)

        # Calculate cross-correlation of detail coefficients at each level
        correlations = []
        for i in range(1, len(coeffs1)):
            correlation = correlate(coeffs1[i], coeffs2[i], mode='full')
            correlations.append(correlation)

        # Find the lag with the maximum correlation
        lags = np.arange(-len(coeffs1[1]) + 1, len(coeffs1[1]))
        lag_indices = [np.argmax(corr) for corr in correlations]
        lag_times = [lags[i] * (time1[1] - time1[0]) for i in lag_indices]

        # Average the lag times
        time_delay = np.mean(lag_times)
        return time_delay
    except Exception as e:
        logging.error(f"Error in Wavelet Transform: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Example Usage:
    # 1. Create dummy data for two light curves.
    # 2.  Run this script.

    # Create dummy data
    time1 = np.linspace(0, 10, 100)
    flux1 = np.sin(time1) + np.random.normal(0, 0.1, 100)
    error1 = np.ones(100) * 0.1
    time2 = np.linspace(1, 11, 100)
    flux2 = np.sin(time2) + np.random.normal(0, 0.1, 100)
    error2 = np.ones(100) * 0.1

    data = {
        'time1': time1,
        'flux1': flux1,
        'error1': error1,
        'time2': time2,
        'flux2': flux2,
        'error2': error2
    }

    try:
        time_delay = estimate_time_delay(data)
        print(f"Estimated Time Delay: {time_delay}")
    except Exception as e:
        print(f"Error: {e}")