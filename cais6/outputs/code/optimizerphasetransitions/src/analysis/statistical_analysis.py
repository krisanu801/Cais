import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def perform_t_test(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """
    Performs an independent samples t-test.

    Args:
        data1 (List[float]): The first set of data.
        data2 (List[float]): The second set of data.

    Returns:
        Tuple[float, float]: The t-statistic and p-value.
    """
    try:
        t_statistic, p_value = stats.ttest_ind(data1, data2)
        return t_statistic, p_value
    except Exception as e:
        logging.error(f"Error performing t-test: {e}")
        raise


def perform_anova(data: List[List[float]]) -> Tuple[float, float]:
    """
    Performs a one-way ANOVA test.

    Args:
        data (List[List[float]]): A list of lists, where each inner list represents a group.

    Returns:
        Tuple[float, float]: The F-statistic and p-value.
    """
    try:
        f_statistic, p_value = stats.f_oneway(*data)
        return f_statistic, p_value
    except Exception as e:
        logging.error(f"Error performing ANOVA: {e}")
        raise


def perform_tukey_hsd(data: List[List[float]], group_names: List[str]) -> Optional[pairwise_tukeyhsd]:
    """
    Performs Tukey's Honest Significant Difference (HSD) post-hoc test.

    Args:
        data (List[List[float]]): A list of lists, where each inner list represents a group.
        group_names (List[str]): A list of group names corresponding to the data.

    Returns:
        Optional[pairwise_tukeyhsd]: The Tukey HSD results, or None if an error occurs.
    """
    try:
        # Flatten the data and create group labels
        values = []
        labels = []
        for i, group_data in enumerate(data):
            values.extend(group_data)
            labels.extend([group_names[i]] * len(group_data))

        # Perform Tukey HSD
        tukey_result = pairwise_tukeyhsd(values, labels, alpha=0.05)
        return tukey_result
    except Exception as e:
        logging.error(f"Error performing Tukey HSD: {e}")
        return None


def test_variance_equality(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """
    Performs a Levene test for equality of variances.

    Args:
        data1 (List[float]): The first set of data.
        data2 (List[float]): The second set of data.

    Returns:
        Tuple[float, float]: The Levene statistic and p-value.
    """
    try:
        levene_statistic, p_value = stats.levene(data1, data2)
        return levene_statistic, p_value
    except Exception as e:
        logging.error(f"Error performing Levene test: {e}")
        raise


# Example Usage:
if __name__ == '__main__':
    try:
        # Example data
        data1 = [0.8, 0.9, 0.7, 0.85, 0.95]
        data2 = [0.6, 0.7, 0.65, 0.75, 0.8]
        data3 = [0.5, 0.6, 0.55, 0.65, 0.7]

        # T-test
        t_statistic, p_value = perform_t_test(data1, data2)
        print(f"T-test: t={t_statistic:.3f}, p={p_value:.3f}")

        # ANOVA
        anova_data = [data1, data2, data3]
        f_statistic, p_value = perform_anova(anova_data)
        print(f"ANOVA: F={f_statistic:.3f}, p={p_value:.3f}")

        # Tukey HSD
        group_names = ['Group 1', 'Group 2', 'Group 3']
        tukey_result = perform_tukey_hsd(anova_data, group_names)
        if tukey_result:
            print("Tukey HSD:")
            print(tukey_result)

        # Levene test for variance equality
        levene_statistic, levene_p_value = test_variance_equality(data1, data2)
        print(f"Levene Test: statistic={levene_statistic:.3f}, p={levene_p_value:.3f}")

    except Exception as e:
        print(f"Error in example usage: {e}")