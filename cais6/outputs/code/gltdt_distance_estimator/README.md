# GLTDT Distance Estimator

## Project Description

This project aims to improve the precision of distance estimates to supermassive black holes using Gravitational Lensing Time Delay Tomography (GLTDT). It focuses on mitigating systematic uncertainties through advanced modeling techniques, including:

*   Data acquisition and preprocessing of multi-wavelength data.
*   Refinement of time delay measurements using Gaussian Process regression and wavelet transforms.
*   Optimization of lens mass models using a Bayesian framework.
*   Machine learning aided mass modeling using U-Net CNNs and GANs.
*   Accounting for line-of-sight effects using weak lensing measurements.

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd gltdt_distance_estimator
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   On Linux/macOS:

        ```bash
        source venv/bin/activate
        ```

    *   On Windows:

        ```bash
        venv\Scripts\activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure logging:**

    Modify the `configs/logging.conf` file to adjust the logging level and output.

6.  **Configure project settings:**

    Modify the `configs/config.yaml` file to set data paths, model parameters, and training hyperparameters.

7.  **Initialize DVC (Data Version Control):**

    ```bash
    dvc init
    ```

## Project Structure

```
gltdt_distance_estimator/
├── src/
│   ├── data/
│   │   ├── data_loader.py  # Handles data loading and preprocessing
│   ├── models/
│   │   ├── time_delay_estimation.py  # Implements time delay estimation techniques
│   │   ├── lens_model.py  # Defines and fits the lens mass model
│   │   ├── machine_learning_model.py  # Contains the CNN and GAN models
│   │   ├── distance_calculation.py  # Calculates the distance and Hubble constant
│   │   ├── line_of_sight_correction.py # Corrects for line-of-sight effects
│   ├── utils/
│   │   ├── utils.py  # General utility functions
│   ├── visualization/
│   │   ├── visualization.py  # Functions for creating visualizations
│   ├── main.py  # Main application entry point
├── tests/
│   ├── test_data_loader.py  # Unit tests for the data loading module
│   ├── test_lens_model.py  # Unit tests for the lens modeling module
├── configs/
│   ├── config.yaml  # Configuration file for project settings
│   ├── config.py # Python module to load config.yaml
│   ├── logging.conf  # Configuration file for the logging module
├── data/ # Directory for storing data files
├── logs/ # Directory for storing log files
├── notebooks/ # Directory for Jupyter notebooks
├── requirements.txt  # Lists all project dependencies
├── README.md  # Project documentation
├── setup.py  # Installation script for the project
├── .gitignore # Specifies intentionally untracked files that Git should ignore
├── dvc.yaml # DVC configuration file for data version control
├── params.yaml # DVC parameters file
└── LICENSE # License file
```

## Usage Examples

1.  **Run the main script:**

    ```bash
    python src/main.py
    ```

    This will load the configuration, load the data, perform time delay estimation, fit the lens model, train the CNN (if enabled), correct for line-of-sight effects, calculate the distance, and save the results.

2.  **Run unit tests:**

    ```bash
    python -m unittest discover tests
    ```

    This will run all unit tests in the `tests/` directory.

3.  **Explore the Jupyter Notebook:**

    Open `notebooks/example_notebook.ipynb` to see an example of how to use the project's modules and functions.

## Configuration

The project is configured using the `configs/config.yaml` file. This file allows you to customize various settings, including:

*   Data paths
*   Model parameters
*   Training hyperparameters
*   Logging level

## Logging

The project uses the Python `logging` module for logging. The logging configuration is defined in the `configs/logging.conf` file. Log files are saved in the `logs/` directory.

## Data Version Control

This project uses DVC (Data Version Control) to manage and track data files. To initialize DVC, run:

```bash
dvc init
```

## License

This project is licensed under the [Specify License, e.g., MIT License]. See the `LICENSE` file for details.