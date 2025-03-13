import sys
import os
import logging
from typing import Dict, Any
import yaml

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")


def load_config(config_path: str = CONFIG_FILE_PATH) -> Dict[str, Any]:
    """
    Loads the configuration from the specified YAML file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise FileNotFoundError(f"Configuration file not found: {e}")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise ValueError(f"Error parsing configuration file: {e}")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise Exception(f"Error loading configuration: {e}")


def get_config_value(config: Dict[str, Any], key_path: str) -> Any:
    """
    Retrieves a value from the configuration dictionary using a key path.

    Args:
        config: The configuration dictionary.
        key_path: A string representing the path to the value, e.g., "gw_signal.redshift".

    Returns:
        The value at the specified key path.
    """
    try:
        keys = key_path.split(".")
        value = config
        for key in keys:
            value = value[key]
        return value
    except KeyError as e:
        logging.error(f"Key not found in configuration: {e}")
        return None  # Or raise an exception, depending on your needs
    except Exception as e:
        logging.error(f"Error retrieving configuration value: {e}")
        return None


if __name__ == "__main__":
    # Example Usage:
    # 1.  Make sure you have installed all the dependencies from requirements.txt
    # 2.  Run the script from the project root: python src/config.py
    # 3.  Check the logs folder for logging information
    # 4.  The configuration values will be printed to the console

    try:
        config = load_config()
        print("Loaded Configuration:")
        print(config)

        # Example: Get the redshift value
        redshift = get_config_value(config, "gw_signal.redshift")
        print(f"Redshift: {redshift}")

        # Example: Get the uncertainty level
        uncertainty_level = get_config_value(config, "localization.uncertainty_level")
        print(f"Uncertainty Level: {uncertainty_level}")

    except Exception as e:
        print(f"Error: {e}")