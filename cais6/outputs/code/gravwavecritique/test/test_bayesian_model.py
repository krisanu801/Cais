import sys
import os
import pytest
import numpy as np

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.bayesian_framework import bayesian_model
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have the project root in your PYTHONPATH.")
    sys.exit(1)


def test_run_bayesian_analysis():
    """Test the run_bayesian_analysis function."""
    gw_signal = {"signal": np.random.normal(0, 1, 100)}
    redshift = 0.2
    localization_data = {"ra_samples": np.random.normal(120, 5, 100), "dec_samples": np.random.normal(45, 5, 100)}

    results = bayesian_model.run_bayesian_analysis(gw_signal, redshift, localization_data)

    assert isinstance(results, dict)
    assert "summary" in results
    assert "trace" in results