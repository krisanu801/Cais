import sys
import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from PIL import Image

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_cnn(data: Dict[str, Any], lens_model: Dict[str, Any], cnn_params: Dict[str, Any]) -> tf.keras.Model:
    """
    Trains a U-Net CNN to predict the lens potential from observational data.

    Args:
        data: A dictionary containing observational data (e.g., images of lensing galaxies).
              Expected keys: 'lensing_galaxy_images'.
        lens_model: A dictionary containing information about the lens model.
        cnn_params: A dictionary containing hyperparameters for the CNN training.
              Expected keys: 'epochs', 'batch_size', 'learning_rate'.

    Returns:
        A trained U-Net CNN model.

    Raises:
        ValueError: If the input data is not in the expected format.
        Exception: For any other errors during CNN training.
    """
    try:
        if 'lensing_galaxy_images' not in data:
            raise ValueError("Input data must contain 'lensing_galaxy_images' key.")

        lensing_galaxy_images = data['lensing_galaxy_images']
        epochs = cnn_params.get('epochs', 10)
        batch_size = cnn_params.get('batch_size', 32)
        learning_rate = cnn_params.get('learning_rate', 0.001)

        # Prepare training data (example: using simulated data)
        # In a real application, you would load and preprocess simulated data here
        # For this example, we'll create dummy data
        num_samples = 100
        img_height = 128
        img_width = 128
        X_train = np.random.rand(num_samples, img_height, img_width, 1)  # Simulated lensing galaxy images
        Y_train = np.random.rand(num_samples, img_height, img_width, 1)  # Simulated lens potential maps

        # Define the U-Net model
        def unet_model(img_height: int, img_width: int) -> tf.keras.Model:
            """Defines the U-Net architecture."""
            inputs = keras.layers.Input((img_height, img_width, 1))

            # Encoder
            conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

            conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
            pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)

            # Bottleneck
            conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

            # Decoder
            up4 = keras.layers.UpSampling2D((2, 2))(conv3)
            merge4 = keras.layers.concatenate([conv2, up4], axis=-1)
            conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge4)

            up5 = keras.layers.UpSampling2D((2, 2))(conv4)
            merge5 = keras.layers.concatenate([conv1, up5], axis=-1)
            conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge5)

            outputs = keras.layers.Conv2D(1, (1, 1), activation='linear')(conv5)  # Linear activation for potential
            return keras.Model(inputs, outputs)

        model = unet_model(img_height, img_width)

        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')  # Mean Squared Error loss

        # Train the model
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

        logging.info("U-Net CNN model trained successfully.")
        return model

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during CNN training: {e}", exc_info=True)
        raise


def detect_substructure(image: np.ndarray, lens_model: Dict[str, Any]) -> np.ndarray:
    """
    Detects substructure in the residual image after subtracting the smooth lens model
    using a clustering algorithm (DBSCAN).

    Args:
        image: The observed image of the lensing galaxy.
        lens_model: A dictionary containing the fitted lens model parameters.

    Returns:
        A numpy array representing the detected substructure.
    """
    try:
        # Generate a smooth lens model image (example)
        smooth_lens_model_image = generate_smooth_lens_model_image(image.shape, lens_model)

        # Calculate the residual image
        residual_image = image - smooth_lens_model_image

        # Reshape the residual image for clustering
        X = residual_image.reshape(-1, 1)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.05, min_samples=5)  # Adjust parameters as needed
        clusters = dbscan.fit_predict(X)

        # Reshape the cluster labels back to the original image shape
        substructure_map = clusters.reshape(image.shape)

        logging.info("Substructure detection completed.")
        return substructure_map

    except Exception as e:
        logging.error(f"An error occurred during substructure detection: {e}", exc_info=True)
        raise


def generate_smooth_lens_model_image(shape: Tuple[int, int], lens_model: Dict[str, Any]) -> np.ndarray:
    """
    Generates a smooth lens model image based on the fitted lens model parameters.
    This is a placeholder and should be replaced with a more accurate calculation
    based on the lens model.

    Args:
        shape: The shape of the image to generate.
        lens_model: A dictionary containing the fitted lens model parameters.

    Returns:
        A numpy array representing the smooth lens model image.
    """
    try:
        # Example: Create a simple Gaussian profile
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        center_x = shape[1] // 2
        center_y = shape[0] // 2
        sigma = 20  # Adjust as needed
        smooth_lens_model_image = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

        return smooth_lens_model_image

    except Exception as e:
        logging.error(f"An error occurred during smooth lens model image generation: {e}", exc_info=True)
        raise


def generate_lensing_galaxy_image(shape: Tuple[int, int]) -> np.ndarray:
    """
    Generates a dummy lensing galaxy image. This is a placeholder and should be
    replaced with a more realistic simulation.

    Args:
        shape: The shape of the image to generate.

    Returns:
        A numpy array representing the lensing galaxy image.
    """
    try:
        # Example: Create a simple image with a bright center
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        center_x = shape[1] // 2
        center_y = shape[0] // 2
        sigma = 10  # Adjust as needed
        lensing_galaxy_image = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)) + \
                                np.random.normal(0, 0.05, shape)  # Add noise

        return lensing_galaxy_image

    except Exception as e:
        logging.error(f"An error occurred during lensing galaxy image generation: {e}", exc_info=True)
        raise


def generate_potential_map(shape: Tuple[int, int]) -> np.ndarray:
    """
    Generates a dummy potential map. This is a placeholder and should be
    replaced with a more realistic simulation.

    Args:
        shape: The shape of the image to generate.

    Returns:
        A numpy array representing the potential map.
    """
    try:
        # Example: Create a simple potential map with a saddle point
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        center_x = shape[1] // 2
        center_y = shape[0] // 2
        potential_map = (x - center_x)**2 - (y - center_y)**2 + np.random.normal(0, 0.01, shape)  # Add noise

        return potential_map

    except Exception as e:
        logging.error(f"An error occurred during potential map generation: {e}", exc_info=True)
        raise


def save_image(image: np.ndarray, filename: str, output_dir: str) -> None:
    """
    Saves a numpy array as an image file.

    Args:
        image: The numpy array representing the image.
        filename: The name of the file to save the image to.
        output_dir: The directory to save the image in.
    """
    try:
        filepath = os.path.join(output_dir, filename)
        # Normalize the image to the range 0-255 and convert to uint8
        image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        img = Image.fromarray(image_normalized)
        img.save(filepath)
        logging.info(f"Image saved to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving image: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Example Usage:
    # 1. Create dummy data for lensing galaxy images and lens model.
    # 2.  Run this script.

    # Create dummy data
    img_height = 128
    img_width = 128
    lensing_galaxy_images = np.random.rand(10, img_height, img_width, 1)  # Example images

    data = {
        'lensing_galaxy_images': lensing_galaxy_images
    }

    lens_model = {
        'param1': 1.0,
        'param2': 2.0
    }

    cnn_params = {
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    }

    try:
        # Train the CNN
        cnn_model = train_cnn(data, lens_model, cnn_params)

        # Generate a dummy lensing galaxy image
        lensing_galaxy_image = generate_lensing_galaxy_image((img_height, img_width))

        # Detect substructure
        substructure_map = detect_substructure(lensing_galaxy_image, lens_model)

        # Save the substructure map as an image
        output_dir = "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results"
        os.makedirs(output_dir, exist_ok=True)
        save_image(substructure_map, "substructure_map.png", output_dir)

        print("Machine learning model executed successfully.")

    except Exception as e:
        print(f"Error: {e}")