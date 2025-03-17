# MoonOptimizer

This project implements a novel optimizer, DA-RMSprop, designed to improve training performance on datasets with complex geometries, such as the make_moons dataset. It compares DA-RMSprop against AdamW.

## Project Structure

```
moonoptimizer/
├── configs/
│   └── config.yaml         # Configuration file for experiment parameters
├── moonlib/
│   ├── optimizer.py      # DA-RMSprop optimizer implementation
│   ├── model.py          # Simple neural network model definition
│   ├── dataset.py        # make_moons dataset loading function
│   └── evaluation.py     # Functions for model evaluation (accuracy, loss)
├── main.py               # Main application file to run experiments
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Setup Instructions

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment:**

    ```bash
    source venv/bin/activate   # Linux/macOS
    venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**

    ```bash
    pip install -r moonoptimizer/requirements.txt
    ```

## Running the Experiment

To run the experiment, execute the `main.py` script:

```bash
python moonoptimizer/main.py
```

This will:

*   Generate the make_moons dataset.
*   Train a simple neural network model using both DA-RMSprop and AdamW optimizers.
*   Evaluate the performance of both optimizers on a test set.
*   Plot the decision boundaries of the trained models.
*   Save the training loss and accuracy data.
*   Print the test accuracies to the console.
*   Save the numerical and graphical results to `/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results`.

## Configuration

The experiment parameters are defined in `moonoptimizer/configs/config.yaml`. You can modify this file to adjust the dataset size, noise level, model architecture, learning rates, and other hyperparameters.

## Results

The results of the experiment, including plots and numerical data, are saved to the `/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results` directory.

*   `darms_losses.txt`: Training losses for DA-RMSprop.
*   `darms_accuracies.txt`: Training accuracies for DA-RMSprop.
*   `adamw_losses.txt`: Training losses for AdamW.
*   `adamw_accuracies.txt`: Training accuracies for AdamW.
*   `results.txt`: Test accuracies for both optimizers.
*   `DA-RMSprop_Decision_Boundary.png`: Decision boundary plot for DA-RMSprop.
*   `AdamW_Decision_Boundary.png`: Decision boundary plot for AdamW.
*   `training_plots.png`: Plots of training loss and accuracy for both optimizers.

## DA-RMSprop Optimizer

The DA-RMSprop optimizer implementation is in `moonoptimizer/moonlib/optimizer.py`. It includes directional momentum for better adaptation to loss landscape geometry.

```python
# Example usage within main.py
from moonoptimizer.moonlib.optimizer import DARMSprop

# Initialize the DA-RMSprop optimizer
darms_optimizer = DARMSprop(model.parameters(), lr=0.001)
```