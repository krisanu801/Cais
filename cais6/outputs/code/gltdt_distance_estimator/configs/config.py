import sys
import os
import logging
from typing import Dict, Any

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.utils.utils import load_config_yaml  # type: ignore
except ImportError as e:
    print(f"ImportError: {e}.  Make sure you are running this from the project root or have configured your PYTHONPATH correctly.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILE = "configs/config.yaml"  # Relative path to the config file

def load_config() -> dict[str, Any]:
    """
    Loads the configuration from the config.yaml file.

    Returns:
        A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        Exception: For any other errors during configuration loading.
    """
    try:
        config = load_config_yaml(CONFIG_FILE)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Example Usage:
    # 1. Ensure you have a config.yaml file in the configs/ directory.
    # 2.  Run this script.

    try:
        config = load_config()
        print("Loaded Configuration:")
        print(config)
        print(f"Data path: {config['data']['data_path']}")
        print(f"Number of epochs: {config['model']['cnn_params']['epochs']}")
    except Exception as e:
        print(f"Error: {e}")