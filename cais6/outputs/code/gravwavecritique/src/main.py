import sys
import os
import logging
from typing import Optional

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src import config
    from src.gw_signal import gw_signal_propagation
    from src.redshift_estimation import redshift_estimation
    from src.localization import localization
    from src.bayesian_framework import bayesian_model
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have the project root in your PYTHONPATH.")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the output folder path
OUTPUT_FOLDER = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"

def ensure_output_folder_exists(folder_path: str) -> None:
    """
    Ensures that the specified output folder exists.
    
    Args:
        folder_path: The path to the output folder.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        logging.info(f"Output folder '{folder_path}' exists or was created.")
    except OSError as e:
        logging.error(f"Error creating output folder '{folder_path}': {e}")


def main() -> None:
    """
    Main application entry point. Orchestrates the different modules.
    """
    logging.info("Starting the GravWaveCritique application.")

    # Load configuration
    try:
        cfg = config.load_config()
        logging.info("Configuration loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        return
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return

    # Ensure the output folder exists
    ensure_output_folder_exists(OUTPUT_FOLDER)

    # Example usage of the modules (replace with actual data and logic)
    try:
        # Simulate a GW signal
        simulated_signal = gw_signal_propagation.simulate_signal()
        logging.info("Simulated GW signal.")

        # Estimate redshift
        redshift = redshift_estimation.estimate_redshift(simulated_signal)
        logging.info(f"Estimated redshift: {redshift}")

        # Perform localization
        localization_data = localization.perform_localization()
        logging.info("Performed localization.")

        # Run the Bayesian model
        bayesian_results = bayesian_model.run_bayesian_analysis(simulated_signal, redshift, localization_data)
        logging.info("Bayesian analysis completed.")

        # Save results to the output folder (example)
        results_file = os.path.join(OUTPUT_FOLDER, "bayesian_results.txt")
        with open(results_file, "w") as f:
            f.write(str(bayesian_results))
        logging.info(f"Bayesian results saved to {results_file}")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

    logging.info("GravWaveCritique application finished.")


if __name__ == "__main__":
    main()

    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/main.py
    # 3.  Check the logs folder for logging information
    # 4.  Check the outputs folder for results