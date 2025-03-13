import logging
from typing import Dict, Any
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeminiAPI:
    """
    Utility functions for interacting with the Gemini API via GenAI.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeminiAPI with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the Gemini API key.
        """
        try:
            self.api_key = config['gemini_api_key']
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            logger.info("GeminiAPI initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing GeminiAPI: {e}")
            raise

    def generate_content(self, prompt: str) -> str:
        """
        Generates content using the Gemini API.

        Args:
            prompt (str): The prompt to send to the API.

        Returns:
            str: The generated content.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"The prompt was blocked due to: {response.prompt_feedback.block_reason}")
                return "The prompt was blocked due to safety concerns. Please refine the prompt."

            content = response.text
            logger.info(f"Generated content:\n{content}")
            return content

        except Exception as e:
            logger.exception(f"Error generating content: {e}")
            return f"Error generating content: {e}"


# Example Usage (This is just a class definition, so no direct execution here)
if __name__ != "__main__":
    # Example Usage (This won't run when imported as a module)
    # To use this, you would need to instantiate the class and call the methods.
    # For example:
    #
    # import yaml
    # from utils.gemini_api import GeminiAPI
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
    # gemini_api = GeminiAPI(config)
    # prompt = "Write a short poem about the stars."
    # poem = gemini_api.generate_content(prompt)
    # print(poem)
    pass