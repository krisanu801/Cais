import sys
import os
import logging
from typing import Dict, Any
import numpy as np
from astropy import constants as const
from astropy import units as u

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_distance(lens_model: Dict[str, Any], time_delay: float) -> float:
    """
    Calculates the time-delay distance and Hubble constant based on lens model
    parameters and time delays.

    Args:
        lens_model: A dictionary containing the best-fit lens model parameters.
              Expected keys: 'best_fit_params', 'redshift_lens', 'redshift_source'.
        time_delay: The estimated time delay between the lensed images (in days).

    Returns:
        The calculated Hubble constant (H0) in km/s/Mpc.

    Raises:
        ValueError: If the input data is not in the expected format.
        Exception: For any other errors during distance calculation.
    """
    try:
        if not all(key in lens_model for key in ['best_fit_params', 'redshift_lens', 'redshift_source']):
            raise ValueError("Lens model must contain 'best_fit_params', 'redshift_lens', and 'redshift_source' keys.")

        best_fit_params = lens_model['best_fit_params']
        redshift_lens = lens_model['redshift_lens']
        redshift_source = lens_model['redshift_source']

        # Extract lens model parameters (example: SIE parameters)
        # This needs to be adapted based on the specific lens model used
        x0, y0, q, phi, b, source_x, source_y = best_fit_params

        # Calculate the Fermat potential difference (example)
        # This is a simplified example and should be replaced with a more accurate calculation
        fermat_potential_scale = b**2  # Example scaling
        delta_fermat_potential = fermat_potential_scale  # Example value

        # Convert time delay to seconds
        time_delay_seconds = time_delay * u.day.to(u.s)

        # Calculate the time-delay distance (D_dt)
        # D_dt = (1 + z_l) * D_l * D_s / D_ls
        # where D_l, D_s, and D_ls are the angular diameter distances to the lens,
        # source, and between the lens and source, respectively.
        # In this simplified example, we'll assume a flat LCDM cosmology and
        # calculate the angular diameter distances using astropy.

        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # Example cosmology

        D_l = cosmo.angular_diameter_distance(redshift_lens).to(u.Mpc)
        D_s = cosmo.angular_diameter_distance(redshift_source).to(u.Mpc)

        # Redshift needs to be higher than lens redshift
        if redshift_source <= redshift_lens:
            raise ValueError("Source redshift must be greater than lens redshift.")

        D_ls = cosmo.angular_diameter_distance_z1z2(redshift_lens, redshift_source).to(u.Mpc)

        # Ensure distances are positive
        if D_l <= 0 or D_s <= 0 or D_ls <= 0:
            raise ValueError("Invalid angular diameter distances. Check redshifts and cosmology.")

        time_delay_distance = (const.c**3 / (4 * np.pi * (1 + redshift_lens) * delta_fermat_potential)) * time_delay_seconds
        time_delay_distance = time_delay_distance.to(u.Mpc)

        # Calculate the Hubble constant (H0)
        # H0 = c / D_H, where D_H is the Hubble distance
        # D_H = D_dt * (D_ls / (D_l * D_s))
        hubble_distance = time_delay_distance * (D_ls / (D_l * D_s))
        hubble_constant = (const.c / hubble_distance).to(u.km / (u.s * u.Mpc))

        logging.info(f"Calculated Hubble constant: {hubble_constant}")
        return hubble_constant.value

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during distance calculation: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Example Usage:
    # 1. Create dummy data for lens model parameters and time delay.
    # 2.  Run this script.

    # Create dummy data
    lens_model = {
        'best_fit_params': [0.1, 0.1, 0.5, 0.5, 1.0, 0.1, 0.1],  # Example SIE parameters
        'redshift_lens': 0.5,  # Example lens redshift
        'redshift_source': 1.5  # Example source redshift
    }

    time_delay = 30.0  # Example time delay (days)

    try:
        hubble_constant = calculate_distance(lens_model, time_delay)
        print(f"Calculated Hubble Constant: {hubble_constant} km/s/Mpc")
    except Exception as e:
        print(f"Error: {e}")