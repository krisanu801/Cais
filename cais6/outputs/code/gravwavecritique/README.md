# GravWaveCritique

## Project Description

This project develops a Bayesian Hierarchical Model for Estimating Black Hole Distance and Redshift Based on Gravitational Wave Signal Propagation, Incorporating Uncertainty in Source Localization, Cosmological Parameters, and Potential Dark Matter Interactions.

## Project Structure

```
GravWaveCritique/
├── src/
│   ├── main.py               # Main application entry point
│   ├── gw_signal/            # Gravitational wave signal module
│   │   ├── gw_signal_propagation.py # Signal propagation modeling
│   │   ├── waveform_model.py      # Waveform model definition
│   │   ├── data_simulation.py     # Signal simulation
│   ├── redshift_estimation/  # Redshift estimation module
│   │   ├── redshift_estimation.py # Redshift estimation logic
│   ├── localization/         # Source localization module
│   │   ├── localization.py        # Localization logic
│   ├── bayesian_framework/   # Bayesian framework module
│   │   ├── bayesian_model.py      # Bayesian model implementation
│   ├── config.py             # Configuration loading
├── configs/
│   ├── config.yaml           # Project configuration
│   ├── logger_config.yaml    # Logging configuration
├── test/
│   ├── test_gw_signal.py    # Unit tests for gw_signal module
│   ├── test_redshift_estimation.py # Unit tests for redshift_estimation module
│   ├── test_bayesian_model.py # Unit tests for bayesian_model module
├── logs/                   # Log files
├── data/                   # Data files (if any)
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── setup.py                # Installation script
├── .gitignore              # Specifies intentionally untracked files that Git should ignore
```

## Setup Instructions

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment:**

    *   Linux/macOS:

        ```bash
        source venv/bin/activate
        ```
    *   Windows:

        ```bash
        venv\Scripts\activate
        ```
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To run the main application, execute the following command from the project root:

```bash
python src/main.py
```

## Testing

To run the unit tests, execute the following command from the project root:

```bash
pytest
```

## Configuration

The project configuration is located in `configs/config.yaml`.  Modify this file to adjust project settings such as redshift, matter density, and MCMC parameters.

## Logging

Logging is configured in `configs/logger_config.yaml`. Log files are stored in the `logs/` directory.

## Dependencies

The project depends on the following Python packages:

*   numpy
*   scipy
*   matplotlib
*   PyMC3
*   arviz
*   astropy
*   PyYAML
*   pytest

These dependencies are listed in `requirements.txt`.

## Output

The project may generate images and text files as output. These files are saved in the `outputs/results` directory. The specific files generated depend on the modules that are executed.