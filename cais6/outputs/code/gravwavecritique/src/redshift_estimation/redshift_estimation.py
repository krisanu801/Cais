import sys
import os
import logging
from typing import Optional, Dict
import numpy as np

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


def estimate_redshift_from_em_counterpart(em_data: dict) -> Optional[float]:
    """
    Estimates redshift from electromagnetic counterpart data.

    Args:
        em_data: A dictionary containing electromagnetic data, including redshift if available.

    Returns:
        The estimated redshift, or None if not available.
    """
    try:
        if "redshift" in em_data:
            redshift = em_data["redshift"]
            logging.info(f"Redshift from EM counterpart: {redshift}")
            return redshift
        else:
            logging.warning("Redshift not found in EM counterpart data.")
            return None
    except Exception as e:
        logging.error(f"Error estimating redshift from EM counterpart: {e}")
        return None


def estimate_redshift_from_statistical_correlation(gw_event_location: tuple) -> Optional[float]:
    """
    Estimates redshift based on statistical correlations with galaxy distributions.
    This is a simplified example and should be replaced with a more sophisticated model.

    Args:
        gw_event_location: A tuple containing the (RA, Dec) of the gravitational wave event.

    Returns:
        The estimated redshift, or None if no correlation is found.
    """
    try:
        # Placeholder: Assume a correlation between location and redshift
        ra, dec = gw_event_location
        # Simplified model: redshift increases with RA and Dec
        redshift = 0.001 * (ra + dec)
        logging.info(f"Redshift from statistical correlation: {redshift}")
        return redshift
    except Exception as e:
        logging.error(f"Error estimating redshift from statistical correlation: {e}")
        return None


def estimate_redshift_from_gw_signal(gw_signal: dict) -> Optional[float]:
    """
    Estimates redshift directly from the gravitational wave signal analysis.
    This is a simplified example and should be replaced with a more sophisticated model.

    Args:
        gw_signal: A dictionary containing the gravitational wave signal data.

    Returns:
        The estimated redshift, or None if it cannot be estimated.
    """
    try:
        # Placeholder: Assume a correlation between signal parameters and redshift
        if "redshifted_frequency" in gw_signal:
            redshifted_frequency = gw_signal["redshifted_frequency"]
            # Simplified model: redshift is inversely proportional to redshifted frequency
            redshift = 100 / redshifted_frequency
            logging.info(f"Redshift from GW signal analysis: {redshift}")
            return redshift
        else:
            logging.warning("Redshifted frequency not found in GW signal data.")
            return None
    except Exception as e:
        logging.error(f"Error estimating redshift from GW signal analysis: {e}")
        return None


def estimate_redshift(gw_signal: dict, em_data: Optional[dict] = None, gw_event_location: Optional[tuple] = None) -> Optional[float]:
    """
    Estimates redshift using a combination of methods.

    Args:
        gw_signal: A dictionary containing the gravitational wave signal data.
        em_data: An optional dictionary containing electromagnetic data.
        gw_event_location: An optional tuple containing the (RA, Dec) of the gravitational wave event.

    Returns:
        The estimated redshift, or None if it cannot be estimated.
    """
    try:
        # 1. Use EM counterpart data if available
        if em_data:
            redshift = estimate_redshift_from_em_counterpart(em_data)
            if redshift is not None:
                return redshift

        # 2. Use statistical correlations if available
        if gw_event_location:
            redshift = estimate_redshift_from_statistical_correlation(gw_event_location)
            if redshift is not None:
                return redshift

        # 3. Use GW signal analysis
        redshift = estimate_redshift_from_gw_signal(gw_signal)
        if redshift is not None:
            return redshift

        logging.warning("Redshift could not be estimated using any available method.")
        return None

    except Exception as e:
        logging.error(f"Error estimating redshift: {e}")
        return None


if __name__ == "__main__":
    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/redshift_estimation/redshift_estimation.py
    # 3.  Check the logs folder for logging information
    # 4.  The estimated redshift will be printed to the console

    # Example data
    gw_signal_data = {"redshifted_frequency": 50.0}
    em_counterpart_data = {"redshift": 0.2}
    gw_event_location = (120.0, 45.0)  # RA, Dec

    # Estimate redshift using different methods
    redshift_em = estimate_redshift(gw_signal_data, em_data=em_counterpart_data)
    redshift_statistical = estimate_redshift(gw_signal_data, gw_event_location=gw_event_location)
    redshift_gw = estimate_redshift(gw_signal_data)

    print("Estimated Redshifts:")
    print(f"From EM counterpart: {redshift_em}")
    print(f"From statistical correlation: {redshift_statistical}")
    print(f"From GW signal analysis: {redshift_gw}")