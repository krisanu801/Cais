import sys
import os
import pytest

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.redshift_estimation import redshift_estimation
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have the project root in your PYTHONPATH.")
    sys.exit(1)


def test_estimate_redshift_from_em_counterpart():
    """Test the estimate_redshift_from_em_counterpart function."""
    em_data = {"redshift": 0.2}
    redshift = redshift_estimation.estimate_redshift_from_em_counterpart(em_data)
    assert redshift == 0.2

    em_data = {}
    redshift = redshift_estimation.estimate_redshift_from_em_counterpart(em_data)
    assert redshift is None


def test_estimate_redshift_from_statistical_correlation():
    """Test the estimate_redshift_from_statistical_correlation function."""
    gw_event_location = (120.0, 45.0)
    redshift = redshift_estimation.estimate_redshift_from_statistical_correlation(gw_event_location)
    assert redshift is not None


def test_estimate_redshift_from_gw_signal():
    """Test the estimate_redshift_from_gw_signal function."""
    gw_signal = {"redshifted_frequency": 50.0}
    redshift = redshift_estimation.estimate_redshift_from_gw_signal(gw_signal)
    assert redshift is not None

    gw_signal = {}
    redshift = redshift_estimation.estimate_redshift_from_gw_signal(gw_signal)
    assert redshift is None


def test_estimate_redshift():
    """Test the estimate_redshift function."""
    gw_signal = {"redshifted_frequency": 50.0}
    em_data = {"redshift": 0.2}
    gw_event_location = (120.0, 45.0)

    # Test with EM data
    redshift = redshift_estimation.estimate_redshift(gw_signal, em_data=em_data)
    assert redshift == 0.2

    # Test with statistical correlation
    redshift = redshift_estimation.estimate_redshift(gw_signal, gw_event_location=gw_event_location)
    assert redshift is not None

    # Test with GW signal
    redshift = redshift_estimation.estimate_redshift(gw_signal)
    assert redshift is not None

    # Test with no data
    gw_signal = {}
    redshift = redshift_estimation.estimate_redshift(gw_signal)
    assert redshift is None