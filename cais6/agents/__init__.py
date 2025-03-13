import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_agents() -> None:
    """
    Initializes the agents package.

    This function can be used to perform any setup or initialization tasks
    required by the agents package when it is imported.
    """
    try:
        logger.info("Initializing agents package...")
        # Add any initialization logic here, such as:
        # - Loading common resources for agents
        # - Setting up shared configurations

        logger.info("agents package initialized successfully.")

    except Exception as e:
        logger.error(f"Error initializing agents package: {e}")
        raise


# Example usage (optional):
# You can call initialize_agents() when the package is imported,
# or you can leave it to be called explicitly by the user.
# initialize_agents()