import logging
from typing import Dict, Any
import google.generativeai as genai
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeneratingAgent:
    """
    Agent responsible for generating novel research ideas and methodologies.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeneratingAgent with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the Gemini API key.
        """
        try:
            self.api_key = config['gemini_api_key']
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.generation_config = {
                "temperature": 0.9,
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
            logger.info("GeneratingAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing GeneratingAgent: {e}")
            raise

    def generate_research_idea(self, research_topic: str) -> str:
        """
        Generates a novel research idea based on the given research topic using the Gemini API.

        Args:
            research_topic (str): The broad research topic to generate ideas for.

        Returns:
            str: A string containing the generated research idea.
        """
        try:
            prompt = f"""
            You are an expert research scientist. Generate a novel and specific research idea 
            based on the following broad research topic: {research_topic}. 
            The idea should be feasible to investigate with computational methods and data analysis.
            Also suggest potential methodologies to explore this idea.
            Provide the output in the following format:
            
            Research Idea: [Generated Research Idea]
            Methodology: [Suggested Methodologies]
            include mathematics if needed.
            """

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"The prompt was blocked due to: {response.prompt_feedback.block_reason}")
                return "The prompt was blocked due to safety concerns. Please refine the research topic."

            research_idea = response.text
            logger.info(f"Generated research idea: {research_idea}")
            return research_idea

        except Exception as e:
            logger.exception(f"Error generating research idea: {e}")
            return f"Error generating research idea: {e}"


if __name__ == '__main__':
    # Example Usage:
    # Assuming you have a config.yaml file with 'gemini_api_key'
    import yaml

    try:
        with open('../../configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.  Make sure it exists and is in the correct location.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        exit()

    generating_agent = GeneratingAgent(config)
    research_topic = "Artificial Intelligence in Healthcare"
    idea = generating_agent.generate_research_idea(research_topic)
    print(f"Generated Research Idea:\n{idea}")