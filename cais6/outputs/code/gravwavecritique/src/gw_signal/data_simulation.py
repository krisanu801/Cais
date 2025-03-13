import sys
import os
import logging
from typing import Dict
import numpy as np

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src import config
    from src.gw_signal import waveform_model
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have the project root in your PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the output folder path
OUTPUT_FOLDER = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"


def add_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Adds Gaussian noise to a signal.

    Args:
        signal: The input signal.
        noise_level: The standard deviation of the Gaussian noise.

    Returns:
        The noisy signal.
    """
    try:
        noise = np.random.normal(0, noise_level, len(signal))
        noisy_signal = signal + noise
        return noisy_signal
    except Exception as e:
        logging.error(f"Error adding noise to signal: {e}")
        return signal  # Return original signal in case of error


def simulate_gw_signal(waveform_type: str, waveform_parameters: Dict, time_duration: float, sampling_rate: float, noise_level: float) -> Dict:
    """
    Simulates a gravitational wave signal.

    Args:
        waveform_type: The type of waveform to generate ("basic", "chirp", etc.).
        waveform_parameters: A dictionary containing the parameters for the waveform model.
        time_duration: The duration of the signal (seconds).
        sampling_rate: The sampling rate of the signal (Hz).
        noise_level: The standard deviation of the Gaussian noise.

    Returns:
        A dictionary containing the simulated signal data.
    """
    try:
        # Generate time vector
        time = np.linspace(0, time_duration, int(time_duration * sampling_rate), endpoint=False)

        # Create waveform model
        waveform = waveform_model.create_waveform_model(waveform_type, waveform_parameters)

        # Generate signal
        signal = waveform(time)

        # Add noise
        noisy_signal = add_noise(signal, noise_level)

        simulated_signal = {
            "time": time,
            "signal": signal,
            "noisy_signal": noisy_signal,
            "waveform_type": waveform_type,
            "waveform_parameters": waveform_parameters,
            "time_duration": time_duration,
            "sampling_rate": sampling_rate,
            "noise_level": noise_level
        }

        logging.info("Gravitational wave signal simulated successfully.")
        return simulated_signal

    except Exception as e:
        logging.error(f"Error simulating gravitational wave signal: {e}")
        return {}  # Return an empty dictionary in case of error


if __name__ == "__main__":
    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/gw_signal/data_simulation.py
    # 3.  Check the logs folder for logging information
    # 4.  The simulated signal parameters will be printed to the console

    # Example parameters
    waveform_type = "chirp"
    waveform_parameters = {
        "initial_frequency": 100.0,
        "end_frequency": 500.0,
        "duration": 1.0,
        "amplitude": 1.0
    }
    time_duration = 2.0  # seconds
    sampling_rate = 2048.0  # Hz
    noise_level = 0.1

    # Simulate GW signal
    simulated_signal = simulate_gw_signal(waveform_type, waveform_parameters, time_duration, sampling_rate, noise_level)

    if simulated_signal:
        print("Simulated Signal Parameters:")
        for key, value in simulated_signal.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: Array of shape {value.shape}")
            else:
                print(f"{key}: {value}")