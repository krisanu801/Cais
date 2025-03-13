import sys
import os
import logging
from typing import Callable, Dict
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


def generate_basic_waveform(frequency: float, amplitude: float, phase: float = 0.0) -> Callable[[float], float]:
    """
    Generates a basic sinusoidal waveform.

    Args:
        frequency: The frequency of the waveform (Hz).
        amplitude: The amplitude of the waveform.
        phase: The phase of the waveform (radians).

    Returns:
        A function representing the waveform.
    """
    try:
        def waveform(t: float) -> float:
            return amplitude * np.sin(2 * np.pi * frequency * t + phase)

        return waveform
    except Exception as e:
        logging.error(f"Error generating basic waveform: {e}")
        return lambda t: 0.0  # Return a zero function in case of error


def extend_waveform(waveform: Callable[[float], float], envelope: Callable[[float], float]) -> Callable[[float], float]:
    """
    Extends a waveform by applying an envelope function.

    Args:
        waveform: The original waveform function.
        envelope: The envelope function.

    Returns:
        A new function representing the extended waveform.
    """
    try:
        def extended_waveform(t: float) -> float:
            return waveform(t) * envelope(t)

        return extended_waveform
    except Exception as e:
        logging.error(f"Error extending waveform: {e}")
        return waveform  # Return original waveform in case of error


def generate_chirp_waveform(initial_frequency: float, end_frequency: float, duration: float, amplitude: float) -> Callable[[float], float]:
    """
    Generates a chirp waveform (frequency increases with time).

    Args:
        initial_frequency: The initial frequency of the chirp (Hz).
        end_frequency: The end frequency of the chirp (Hz).
        duration: The duration of the chirp (seconds).
        amplitude: The amplitude of the chirp.

    Returns:
        A function representing the chirp waveform.
    """
    try:
        def chirp_waveform(t: float) -> float:
            if t > duration:
                return 0.0  # Waveform is zero after the duration
            frequency = initial_frequency + (end_frequency - initial_frequency) * (t / duration)
            return amplitude * np.sin(2 * np.pi * frequency * t)

        return chirp_waveform
    except Exception as e:
        logging.error(f"Error generating chirp waveform: {e}")
        return lambda t: 0.0  # Return a zero function in case of error


def create_waveform_model(model_type: str, parameters: Dict) -> Callable[[float], float]:
    """
    Creates a waveform model based on the specified type and parameters.

    Args:
        model_type: The type of waveform model ("basic", "chirp", etc.).
        parameters: A dictionary containing the parameters for the model.

    Returns:
        A function representing the waveform model.
    """
    try:
        if model_type == "basic":
            frequency = parameters.get("frequency", 100.0)
            amplitude = parameters.get("amplitude", 1.0)
            phase = parameters.get("phase", 0.0)
            return generate_basic_waveform(frequency, amplitude, phase)
        elif model_type == "chirp":
            initial_frequency = parameters.get("initial_frequency", 50.0)
            end_frequency = parameters.get("end_frequency", 200.0)
            duration = parameters.get("duration", 1.0)
            amplitude = parameters.get("amplitude", 1.0)
            return generate_chirp_waveform(initial_frequency, end_frequency, duration, amplitude)
        else:
            logging.warning(f"Unknown waveform model type: {model_type}")
            return lambda t: 0.0  # Return a zero function for unknown types
    except Exception as e:
        logging.error(f"Error creating waveform model: {e}")
        return lambda t: 0.0  # Return a zero function in case of error


if __name__ == "__main__":
    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/gw_signal/waveform_model.py
    # 3.  Check the logs folder for logging information
    # 4.  Example waveforms will be generated and printed to the console

    # Example: Create a basic waveform
    basic_waveform = create_waveform_model("basic", {"frequency": 440.0, "amplitude": 0.5})
    time = np.linspace(0, 0.01, 100)
    basic_signal = basic_waveform(time)
    print("Basic Waveform:", basic_signal[:5])  # Print first 5 values

    # Example: Create a chirp waveform
    chirp_waveform = create_waveform_model("chirp", {"initial_frequency": 100.0, "end_frequency": 500.0, "duration": 1.0, "amplitude": 0.8})
    chirp_signal = chirp_waveform(time)
    print("Chirp Waveform:", chirp_signal[:5])  # Print first 5 values