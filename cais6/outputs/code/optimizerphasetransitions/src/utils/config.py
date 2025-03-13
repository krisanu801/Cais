import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import yaml
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration from the specified YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise


# Example Usage:
if __name__ == '__main__':
    try:
        # Create a dummy config.yaml file for testing
        dummy_config = {
            'seed': 42,
            'dataset': {'name': 'MakeMoons', 'n_samples': 100}
        }
        config_path = "dummy_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dummy_config, f)

        # Load the configuration
        config = load_config(config_path)
        print(f"Configuration: {config}")

        # Clean up (remove the created file)
        os.remove(config_path)

    except Exception as e:
        print(f"Error in example usage: {e}")