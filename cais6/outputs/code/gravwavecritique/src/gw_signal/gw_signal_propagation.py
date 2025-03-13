import sys
import os
import logging
from typing import Tuple, Callable
import numpy as np
from scipy.integrate import quad

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


def cosmological_redshifting(frequency: float, redshift: float) -> float:
    """
    Applies cosmological redshifting to the frequency of a gravitational wave.

    Args:
        frequency: The original frequency of the gravitational wave (Hz).
        redshift: The redshift value.

    Returns:
        The redshifted frequency (Hz).
    """
    try:
        redshifted_frequency = frequency / (1 + redshift)
        return redshifted_frequency
    except Exception as e:
        logging.error(f"Error applying cosmological redshifting: {e}")
        return frequency  # Return original frequency in case of error


def lensing_magnification(redshift: float, matter_density: float) -> float:
    """
    Calculates the lensing magnification factor based on redshift and matter density.
    This is a simplified model and should be replaced with a more sophisticated one.

    Args:
        redshift: The redshift value.
        matter_density: The matter density along the line of sight (kg/m^3).

    Returns:
        The lensing magnification factor.
    """
    try:
        # Simplified model: magnification increases with redshift and matter density
        magnification = 1 + (redshift * matter_density)
        return magnification
    except Exception as e:
        logging.error(f"Error calculating lensing magnification: {e}")
        return 1.0  # Return 1.0 (no magnification) in case of error


def damping_coefficient(frequency: float, redshift: float, dark_matter_density: float) -> float:
    """
    Calculates the damping coefficient as a function of frequency, redshift, and dark matter density.
    This is a placeholder and should be replaced with a physically motivated model.

    Args:
        frequency: The frequency of the gravitational wave (Hz).
        redshift: The redshift value.
        dark_matter_density: The dark matter density along the line of sight (kg/m^3).

    Returns:
        The damping coefficient.
    """
    try:
        # Placeholder model: damping increases with frequency, redshift, and dark matter density
        alpha = 0.001 * frequency * redshift * dark_matter_density
        return alpha
    except Exception as e:
        logging.error(f"Error calculating damping coefficient: {e}")
        return 0.0  # Return 0.0 (no damping) in case of error


def frequency_dependent_damping(frequency: float, redshift: float, dark_matter_density: float, waveform: Callable[[float], float]) -> Callable[[float], float]:
    """
    Applies frequency-dependent damping to a gravitational wave waveform.

    Args:
        frequency: The frequency of the gravitational wave (Hz).
        redshift: The redshift value.
        dark_matter_density: The dark matter density along the line of sight (kg/m^3).
        waveform: A function representing the gravitational wave waveform.  Takes time as input.

    Returns:
        A new function representing the damped waveform.
    """
    try:
        def damped_waveform(t: float) -> float:
            alpha = damping_coefficient(frequency, redshift, dark_matter_density)
            # Simple exponential damping
            return waveform(t) * np.exp(-alpha * frequency)  # Damping is frequency dependent

        return damped_waveform
    except Exception as e:
        logging.error(f"Error applying frequency-dependent damping: {e}")
        return waveform # Return original waveform in case of error


def simulate_signal() -> dict:
    """
    Simulates a gravitational wave signal, applying redshifting, lensing, and damping.

    Returns:
        A dictionary containing the simulated signal parameters.
    """
    try:
        # Example parameters (replace with realistic values)
        original_frequency = 100.0  # Hz
        redshift = 0.5
        matter_density = 1e-26  # kg/m^3
        dark_matter_density = 1e-25  # kg/m^3

        # Apply cosmological redshifting
        redshifted_frequency = cosmological_redshifting(original_frequency, redshift)

        # Calculate lensing magnification
        magnification = lensing_magnification(redshift, matter_density)

        # Define a simple waveform (replace with a more realistic model)
        def original_waveform(t: float) -> float:
            return np.sin(2 * np.pi * original_frequency * t)

        # Apply frequency-dependent damping
        damped_waveform = frequency_dependent_damping(original_frequency, redshift, dark_matter_density, original_waveform)

        # Simulate the signal (replace with actual simulation logic)
        time = np.linspace(0, 0.1, 1000)
        signal = damped_waveform(time) * magnification

        simulated_signal = {
            "time": time,
            "signal": signal,
            "redshifted_frequency": redshifted_frequency,
            "magnification": magnification,
            "redshift": redshift,
            "matter_density": matter_density,
            "dark_matter_density": dark_matter_density
        }

        logging.info("Gravitational wave signal simulated successfully.")
        return simulated_signal

    except Exception as e:
        logging.error(f"Error simulating gravitational wave signal: {e}")
        return {}  # Return an empty dictionary in case of error


if __name__ == "__main__":
    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/gw_signal/gw_signal_propagation.py
    # 3.  Check the logs folder for logging information
    # 4.  The simulated signal parameters will be printed to the console

    simulated_signal = simulate_signal()
    if simulated_signal:
        print("Simulated Signal Parameters:")
        for key, value in simulated_signal.items():
            print(f"{key}: {value}")