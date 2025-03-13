# Optimizer Phase Transitions

## Project Description

This project investigates the impact of various adaptive learning rate decay schedules on the performance of Adam and RMSprop optimizers, focusing on the potential for inducing phase transitions in the learning process. We hypothesize that specific decay schedules can alter the optimizer's ability to escape local minima and converge to global optima, leading to improved generalization performance. We extend this analysis to both the MakeMoons dataset and a simplified version of the MNIST dataset to assess the generalizability of the findings.

## Project Structure

```
OptimizerPhaseTransitions/
├── src/
│   ├── data/
│   │   ├── data_loader.py  # Handles loading and preprocessing of datasets
│   ├── models/
│   │   ├── model.py        # Defines the MLP model architecture
│   ├── optimizers/
│   │   ├── learning_rate_schedulers.py # Implements LR decay schedules
│   │   ├── optimizer_utils.py          # Utility functions for optimizers
│   ├── analysis/
│   │   ├── loss_landscape.py # Visualizes the loss landscape
│   │   ├── metrics.py        # Defines evaluation metrics (AUC, F1-score)
│   │   ├── statistical_analysis.py # Performs statistical analysis
│   │   ├── hessian.py        # Approximates and analyzes the Hessian
│   ├── utils/
│   │   ├── utils.py        # General utility functions
│   │   ├── config.py       # Loads configuration from YAML
│   ├── main.py             # Main application entry point
├── configs/
│   ├── config.yaml         # Configuration file
│   ├── logging.conf        # Logging configuration
├── logs/                   # Directory for log files
├── tests/
│   ├── test_data_loader.py # Unit tests for data loading
│   ├── test_model.py       # Unit tests for the MLP model
│   ├── test_learning_rate_schedulers.py # Unit tests for LR schedulers
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── setup.py                # Installation script
├── .gitignore              # Specifies intentionally untracked files
```

## Setup Instructions

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment:**

    *   On Linux/macOS:

        ```bash
        source venv/bin/activate
        ```
    *   On Windows:

        ```bash
        venv\Scripts\activate
        ```
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

To run the main application, use the following command:

```bash
python src/main.py --config configs/config.yaml
```

You can modify the `config.yaml` file to change the experiment parameters.

## Configuration

The `config.yaml` file contains the following parameters:

*   `seed`: Random seed for reproducibility.
*   `dataset`:
    *   `name`: Dataset name (MakeMoons or MNIST).
    *   `n_samples`: Number of samples for MakeMoons.
    *   `noise`: Noise level for MakeMoons.
    *   `num_classes`: Number of classes for MNIST.
    *   `pca_dim`: PCA dimension for MNIST.
*   `model`:
    *   `hidden_dims`: List of hidden layer dimensions.
*   `optimizer`:
    *   `name`: Optimizer name (Adam or RMSprop).
    *   `learning_rate`: Learning rate.
*   `scheduler`:
    *   `name`: Scheduler name (ExponentialLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau, or None).
    *   Scheduler-specific parameters (e.g., `gamma` for ExponentialLR, `T_max` for CosineAnnealingLR).
*   `training`:
    *   `batch_size`: Batch size.
    *   `epochs`: Number of epochs.
    *   `train_ratio`: Training data ratio.
    *   `val_ratio`: Validation data ratio.
    *   `patience`: Patience for early stopping.
*   `analysis`:
    *   `plot_loss_landscape`: Whether to plot the loss landscape.

## Logging

The project uses the `logging` module for logging information, warnings, and errors. The logging configuration is defined in `configs/logging.conf`.

## Testing

To run the unit tests, use the following command:

```bash
python -m unittest discover tests
```

## Output

The project may generate images or text outputs, which are saved in the `/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results` directory.  Ensure this directory exists before running the project.

## Dependencies

The project dependencies are listed in `requirements.txt`.