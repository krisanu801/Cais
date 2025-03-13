import logging
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileManager:
    """
    Utility functions for file management (saving, loading, etc.).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the FileManager with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the output directories.
        """
        try:
            self.output_dir = config.get('output_dir', 'cais6/outputs')
            self.papers_dir = os.path.join(self.output_dir, 'papers')
            self.code_dir = os.path.join(self.output_dir, 'code')
            self.datasets_dir = os.path.join(self.output_dir, 'datasets')
            self.results_dir = os.path.join(self.output_dir, 'results')

            # Create directories if they don't exist
            os.makedirs(self.papers_dir, exist_ok=True)
            os.makedirs(self.code_dir, exist_ok=True)
            os.makedirs(self.datasets_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)

            logger.info("FileManager initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing FileManager: {e}")
            raise

    def save_paper(self, paper_content: str, filename: str) -> None:
        """
        Saves the given paper content to a file in the papers directory.

        Args:
            paper_content (str): The content of the paper.
            filename (str): The name of the file to save the paper to (e.g., "final_paper.tex").
        """
        try:
            filepath = os.path.join(self.papers_dir, filename)
            with open(filepath, 'w') as f:
                f.write(paper_content)
            logger.info(f"Paper saved to: {filepath}")
        except Exception as e:
            logger.exception(f"Error saving paper to file: {e}")

    def save_code(self, code_content: str, filename: str) -> None:
        """
        Saves the given code content to a file in the code directory.

        Args:
            code_content (str): The content of the code.
            filename (str): The name of the file to save the code to (e.g., "initial_code.py").
        """
        try:
            filepath = os.path.join(self.code_dir, filename)
            with open(filepath, 'w') as f:
                f.write(code_content)
            logger.info(f"Code saved to: {filepath}")
        except Exception as e:
            logger.exception(f"Error saving code to file: {e}")

    def save_dataset(self, dataset_content: str, filename: str) -> None:
        """
        Saves the given dataset content to a file in the datasets directory.

        Args:
            dataset_content (str): The content of the dataset.
            filename (str): The name of the file to save the dataset to (e.g., "data.csv").
        """
        try:
            filepath = os.path.join(self.datasets_dir, filename)
            with open(filepath, 'w') as f:
                f.write(dataset_content)
            logger.info(f"Dataset saved to: {filepath}")
        except Exception as e:
            logger.exception(f"Error saving dataset to file: {e}")

    def save_results(self, results_content: str, filename: str) -> None:
        """
        Saves the given results content to a file in the results directory.

        Args:
            results_content (str): The content of the results.
            filename (str): The name of the file to save the results to (e.g., "initial_results.txt").
        """
        try:
            filepath = filename
            with open(filepath, 'w') as f:
                f.write(results_content)
            logger.info(f"Results saved to: {filepath}")
        except Exception as e:
            logger.exception(f"Error saving results to file: {e}")

    def save_literature(self, literature_content: Dict[str, str], filename: str) -> None:
        """
        Saves the given literature content to a file in the results directory.

        Args:
            literature_content (Dict[str, str]): The content of the literature.
            filename (str): The name of the file to save the literature to (e.g., "literature.txt").
        """
        try:
            filepath = filename
            with open(filepath, 'w') as f:
                for title, summary in literature_content.items():
                    f.write(f"Title: {title}\nSummary: {summary}\n\n")
            logger.info(f"Literature saved to: {filepath}")
        except Exception as e:
            logger.exception(f"Error saving literature to file: {e}")

    def save_citations(self, citations_content: str, filename: str) -> None:
        """
        Saves the given citations content to a file in the results directory.

        Args:
            citations_content (str): The content of the citations.
            filename (str): The name of the file to save the citations to (e.g., "citations.bib").
        """
        try:
            filepath = filename
            with open(filepath, 'w') as f:
                f.write(citations_content)
            logger.info(f"Citations saved to: {filepath}")
        except Exception as e:
            logger.exception(f"Error saving citations to file: {e}")


# Example Usage (This is just a class definition, so no direct execution here)
if __name__ != "__main__":
    # Example Usage (This won't run when imported as a module)
    # To use this, you would need to instantiate the class and call the methods.
    # For example:
    #
    # import yaml
    # from utils.file_manager import FileManager
    #
    # try:
    #     with open('../../configs/config.yaml', 'r') as f:
    #         config = yaml.safe_load(f)
    # except FileNotFoundError:
    #     print("Error: config.yaml not found.  Make sure it exists and is in the correct location.")
    #     exit()
    # except yaml.YAMLError as e:
    #     print(f"Error parsing config.yaml: {e}")
    #     exit()
    #
    # file_manager = FileManager(config)
    # file_manager.save_paper("This is a sample paper.", "sample_paper.txt")
    pass