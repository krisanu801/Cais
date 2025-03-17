import yaml
import os
from typing import Dict


def load_config(config_path: str) -> Dict:
    """Loads configuration from a YAML file.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}
