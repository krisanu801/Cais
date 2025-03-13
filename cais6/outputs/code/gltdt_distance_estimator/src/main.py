import sys
import os
import logging
from typing import Any

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.data.data_loader import load_data  # type: ignore
    from src.models.time_delay_estimation import estimate_time_delay  # type: ignore
    from src.models.lens_model import fit_lens_model  # type: ignore
    from src.models.machine_learning_model import train_cnn  # type: ignore
    from src.models.distance_calculation import calculate_distance  # type: ignore
    from src.models.line_of_sight_correction import correct_line_of_sight  # type: ignore
    from configs.config import load_config  # type: ignore
    from src.visualization.visualization import plot_results  # type: ignore
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have configured your PYTHONPATH correctly.")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    """
    Main application entry point. Orchestrates data loading, processing,
    modeling, and validation for GLTDT distance estimation.
    """
    try:
        # Load configuration
        config = load_config()
        logging.info("Configuration loaded successfully.")

        # Define output directory
        output_dir = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        data = load_data(config['data']['data_path'])
        logging.info("Data loaded successfully.")

        # Estimate time delay
        time_delay = estimate_time_delay(data)
        logging.info(f"Time delay estimated: {time_delay}")

        # Fit lens model
        lens_model = fit_lens_model(data, time_delay)
        logging.info("Lens model fitted successfully.")

        # Train CNN (example - adjust as needed)
        cnn_model = train_cnn(data, lens_model, config['model']['cnn_params'])
        logging.info("CNN model trained successfully.")

        # Correct for line-of-sight effects
        corrected_lens_model = correct_line_of_sight(lens_model)
        logging.info("Line-of-sight effects corrected.")

        # Calculate distance
        distance = calculate_distance(corrected_lens_model, time_delay)
        logging.info(f"Distance calculated: {distance}")

        # Visualize results
        plot_results(data, lens_model, distance, output_dir)
        logging.info("Results visualized and saved.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

    # Example Usage (Illustrative)
    # 1. Ensure you have a config.yaml file in the configs/ directory.
    # 2. Ensure you have data in the path specified in config.yaml.
    # 3. Run the script: python src/main.py
    # 4. Check the logs/ directory for logging information.
    # 5. Check the outputs/results directory for saved images and texts.