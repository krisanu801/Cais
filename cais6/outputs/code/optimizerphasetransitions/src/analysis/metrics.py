import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_accuracy(true_labels: List[int], predicted_labels: List[int]) -> float:
    """
    Calculates the accuracy.

    Args:
        true_labels (List[int]): The true labels.
        predicted_labels (List[int]): The predicted labels.

    Returns:
        float: The accuracy.
    """
    try:
        accuracy = accuracy_score(true_labels, predicted_labels)
        return accuracy
    except Exception as e:
        logging.error(f"Error calculating accuracy: {e}")
        raise


def calculate_auc(true_labels: List[int], predicted_probabilities: List[float]) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (AUC).

    Args:
        true_labels (List[int]): The true labels.
        predicted_probabilities (List[float]): The predicted probabilities for the positive class.

    Returns:
        float: The AUC score.
    """
    try:
        auc = roc_auc_score(true_labels, predicted_probabilities)
        return auc
    except ValueError as e:
        logging.warning(f"ValueError during AUC calculation (likely due to single class): {e}")
        return 0.5  # Return a neutral value if AUC cannot be calculated
    except Exception as e:
        logging.error(f"Error calculating AUC: {e}")
        raise


def calculate_f1_score(true_labels: List[int], predicted_labels: List[int]) -> float:
    """
    Calculates the F1-score.

    Args:
        true_labels (List[int]): The true labels.
        predicted_labels (List[int]): The predicted labels.

    Returns:
        float: The F1-score.
    """
    try:
        f1 = f1_score(true_labels, predicted_labels)
        return f1
    except Exception as e:
        logging.error(f"Error calculating F1-score: {e}")
        raise


# Example Usage:
if __name__ == '__main__':
    try:
        # Example data
        true_labels = [0, 1, 1, 0, 1, 0]
        predicted_labels = [0, 1, 0, 0, 1, 1]
        predicted_probabilities = [0.1, 0.8, 0.3, 0.2, 0.7, 0.6]

        # Calculate metrics
        accuracy = calculate_accuracy(true_labels, predicted_labels)
        auc = calculate_auc(true_labels, predicted_probabilities)
        f1 = calculate_f1_score(true_labels, predicted_labels)

        # Print results
        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")
        print(f"F1-score: {f1}")

    except Exception as e:
        print(f"Error in example usage: {e}")