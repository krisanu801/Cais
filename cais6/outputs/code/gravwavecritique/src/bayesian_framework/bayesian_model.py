import sys
import os
import logging
from typing import Dict, Any
import numpy as np
import pymc3 as pm
import arviz as az

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


def run_bayesian_analysis(gw_signal: Dict, redshift: float, localization_data: Dict) -> Dict:
    """
    Runs a Bayesian hierarchical model to estimate parameters.

    Args:
        gw_signal: A dictionary containing the gravitational wave signal data.
        redshift: The estimated redshift.
        localization_data: A dictionary containing the localization data.

    Returns:
        A dictionary containing the results of the Bayesian analysis.
    """
    try:
        # Example parameters (replace with realistic values and priors)
        with pm.Model() as model:
            # Priors for parameters
            distance = pm.Normal("distance", mu=100.0, sigma=50.0)  # Example prior for distance
            # Incorporate redshift estimate as a prior
            redshift_prior = pm.Normal("redshift_prior", mu=redshift, sigma=0.1)

            # Incorporate localization data (example: use RA and Dec samples)
            ra_samples = localization_data.get("ra_samples", np.array([0.0]))
            dec_samples = localization_data.get("dec_samples", np.array([0.0]))

            # Likelihood function (simplified example)
            # Replace with a realistic likelihood function based on the GW signal model
            likelihood_data = gw_signal.get("signal", np.array([0.0]))
            if len(likelihood_data) > 0:
                sigma = pm.HalfNormal("sigma", sigma=1.0)
                pm.Normal("likelihood", mu=distance * (1 + redshift_prior), sigma=sigma, observed=likelihood_data[:10]) # Using only first 10 elements for demonstration

            # MCMC sampling
            try:
                trace = pm.sample(100, tune=50, cores=1, compute_convergence_checks=False, target_accept=0.9, random_seed=42, chains=1, init='adapt_diag', step=pm.Metropolis(), progressbar=False, discard_tuned_samples=True) # Reduced samples for demonstration, disable convergence checks, increase target_accept, add random_seed, single chain, init method, explicit sampler, disable progressbar, discard tuned samples
            except Exception as e:
                logging.warning(f"Error during MCMC sampling: {e}")
                return {}

            # Analyze results
            try:
                summary = az.summary(trace)
                summary_dict = summary.to_dict()
            except Exception as e:
                logging.warning(f"Error generating arviz summary: {e}")
                summary_dict = {}

            # Convert trace to dictionary
            trace_dict = {}
            for var in trace.varnames:
                trace_dict[var] = trace[var]

            results = {
                "summary": summary_dict,
                "trace": trace_dict
            }

            logging.info("Bayesian analysis completed successfully.")
            return results

    except Exception as e:
        logging.error(f"Error running Bayesian analysis: {e}")
        return {}  # Return an empty dictionary in case of error


if __name__ == "__main__":
    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/bayesian_framework/bayesian_model.py
    # 3.  Check the logs folder for logging information
    # 4.  The results of the Bayesian analysis will be printed to the console

    # Example data
    gw_signal_data = {"signal": np.random.normal(0, 1, 100)}
    redshift = 0.2
    localization_data = {"ra_samples": np.random.normal(120, 5, 100), "dec_samples": np.random.normal(45, 5, 100)}

    # Run Bayesian analysis
    bayesian_results = run_bayesian_analysis(gw_signal_data, redshift, localization_data)

    if bayesian_results:
        print("Bayesian Analysis Results:")
        print(bayesian_results["summary"])
