import sys
import os
import logging
from typing import List, Any, Union, Dict
import numpy as np
import pandas as pd
import yaml

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame containing the data.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any other errors during data loading.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from CSV: {file_path}")
        return data
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from CSV: {e}", exc_info=True)
        raise


def save_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        data: The pandas DataFrame to save.
        file_path: The path to the CSV file.
    """
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Data saved to CSV: {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}", exc_info=True)
        raise


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalizes a numpy array to the range [0, 1].

    Args:
        data: The numpy array to normalize.

    Returns:
        The normalized numpy array.
    """
    try:
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            logging.warning("Data has zero range, cannot normalize.")
            return data  # Return original data if normalization is not possible
        normalized_data = (data - min_val) / (max_val - min_val)
        logging.info("Data normalized.")
        return normalized_data
    except Exception as e:
        logging.error(f"Error normalizing data: {e}", exc_info=True)
        raise


def calculate_mean(data: List[Union[int, float]]) -> float:
    """
    Calculates the mean of a list of numbers.

    Args:
        data: A list of numbers.

    Returns:
        The mean of the numbers.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the list is empty.
    """
    try:
        if not isinstance(data, list):
            raise TypeError("Input must be a list.")
        if not data:
            raise ValueError("List is empty, cannot calculate mean.")
        mean = np.mean(data)
        logging.info("Mean calculated.")
        return float(mean)
    except TypeError as e:
        logging.error(f"TypeError: {e}")
        raise
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise
    except Exception as e:
        logging.error(f"Error calculating mean: {e}", exc_info=True)
        raise


def load_config_yaml(file_path: str) -> dict[str, Any]:
    """
    Loads configuration parameters from a YAML file.

    Args:
        file_path: The path to the YAML file.

    Returns:
        A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from YAML: {file_path}")
        return config
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"YAML error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration from YAML: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Example Usage:
    # 1. Create a dummy CSV file named 'dummy_data.csv' in the same directory
    #    with some numerical data.
    # 2.  Run this script.

    # Create a dummy CSV file for testing
    dummy_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    dummy_csv_path = 'dummy_data.csv'
    dummy_data.to_csv(dummy_csv_path, index=False)

    try:
        loaded_data = load_data_from_csv(dummy_csv_path)
        print("Loaded Data:")
        print(loaded_data)

        normalized_col1 = normalize_data(loaded_data['col1'].values)
        print("Normalized col1:")
        print(normalized_col1)

        mean_col2 = calculate_mean(loaded_data['col2'].tolist())
        print("Mean of col2:")
        print(mean_col2)

        save_data_to_csv(loaded_data, 'output_data.csv')

        # Example loading config
        # Create a dummy config.yaml file
        config_data = {'param1': 10, 'param2': 'test'}
        with open('config.yaml', 'w') as f:
            yaml.dump(config_data, f)

        config = load_config_yaml('config.yaml')
        print("Loaded config:", config)

    except Exception as e:
        print(f"Error: {e}")