import sys
import os
import pytest
import numpy as np

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.gw_signal import gw_signal_propagation
    from src.gw_signal import waveform_model
    from src.gw_signal import data_simulation
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have the project root in your PYTHONPATH.")
    sys.exit(1)


def test_cosmological_redshifting():
    """Test the cosmological_redshifting function."""
    frequency = 100.0
    redshift = 0.5
    redshifted_frequency = gw_signal_propagation.cosmological_redshifting(frequency, redshift)
    assert redshifted_frequency == pytest.approx(frequency / (1 + redshift))


def test_lensing_magnification():
    """Test the lensing_magnification function."""
    redshift = 0.5
    matter_density = 1e-26
    magnification = gw_signal_propagation.lensing_magnification(redshift, matter_density)
    assert magnification > 0


def test_damping_coefficient():
    """Test the damping_coefficient function."""
    frequency = 100.0
    redshift = 0.5
    dark_matter_density = 1e-25
    alpha = gw_signal_propagation.damping_coefficient(frequency, redshift, dark_matter_density)
    assert alpha >= 0


def test_frequency_dependent_damping():
    """Test the frequency_dependent_damping function."""
    frequency = 100.0
    redshift = 0.5
    dark_matter_density = 1e-25
    waveform = lambda t: np.sin(2 * np.pi * frequency * t)
    damped_waveform = gw_signal_propagation.frequency_dependent_damping(frequency, redshift, dark_matter_density, waveform)
    assert callable(damped_waveform)
    # Check that the damped waveform has a smaller amplitude at a given time
    t = 0.01
    assert abs(damped_waveform(t)) <= abs(waveform(t))


def test_simulate_signal():
    """Test the simulate_signal function."""
    simulated_signal = gw_signal_propagation.simulate_signal()
    assert isinstance(simulated_signal, dict)
    assert "time" in simulated_signal
    assert "signal" in simulated_signal


def test_generate_basic_waveform():
    """Test the generate_basic_waveform function."""
    frequency = 440.0
    amplitude = 0.5
    waveform = waveform_model.generate_basic_waveform(frequency, amplitude)
    assert callable(waveform)


def test_extend_waveform():
    """Test the extend_waveform function."""
    frequency = 440.0
    amplitude = 0.5
    waveform = waveform_model.generate_basic_waveform(frequency, amplitude)
    envelope = lambda t: np.exp(-t)
    extended_waveform = waveform_model.extend_waveform(waveform, envelope)
    assert callable(extended_waveform)


def test_generate_chirp_waveform():
    """Test the generate_chirp_waveform function."""
    initial_frequency = 100.0
    end_frequency = 500.0
    duration = 1.0
    amplitude = 0.8
    waveform = waveform_model.generate_chirp_waveform(initial_frequency, end_frequency, duration, amplitude)
    assert callable(waveform)


def test_create_waveform_model():
    """Test the create_waveform_model function."""
    model_type = "basic"
    parameters = {"frequency": 440.0, "amplitude": 0.5}
    waveform = waveform_model.create_waveform_model(model_type, parameters)
    assert callable(waveform)

    model_type = "chirp"
    parameters = {"initial_frequency": 100.0, "end_frequency": 500.0, "duration": 1.0, "amplitude": 0.8}
    waveform = waveform_model.create_waveform_model(model_type, parameters)
    assert callable(waveform)


def test_add_noise():
    """Test the add_noise function."""
    signal = np.sin(np.linspace(0, 10, 100))
    noise_level = 0.1
    noisy_signal = data_simulation.add_noise(signal, noise_level)
    assert len(noisy_signal) == len(signal)


def test_simulate_gw_signal():
    """Test the simulate_gw_signal function."""
    waveform_type = "chirp"
    waveform_parameters = {
        "initial_frequency": 100.0,
        "end_frequency": 500.0,
        "duration": 1.0,
        "amplitude": 1.0
    }
    time_duration = 2.0
    sampling_rate = 2048.0
    noise_level = 0.1
    simulated_signal = data_simulation.simulate_gw_signal(waveform_type, waveform_parameters, time_duration, sampling_rate, noise_level)
    assert isinstance(simulated_signal, dict)
    assert "time" in simulated_signal
    assert "signal" in simulated_signal
    assert "noisy_signal" in simulated_signal