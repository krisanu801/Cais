import logging
from typing import Dict, Any
import google.generativeai as genai
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CritiquingAgent:
    """
    Agent responsible for evaluating the feasibility and flaws of research ideas.
    """

    def __init__(self,config: Dict[str, Any]):
        """
        Initializes the CritiquingAgent with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the Gemini API key.
        """
        try:
            self.api_key = config['gemini_api_key']
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
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
            logger.info("CritiquingAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing CritiquingAgent: {e}")
            raise
    def query_gemini(self, prompt: str) -> str:
        """
        Sends a prompt to Gemini and returns the response.

        Args:
            prompt (str): The prompt to send to Gemini.

        Returns:
            str: The response from Gemini.
        """
        try:
            response = self.chat.send_message(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            self.chat_history = self.chat.history
            return response.text
        except :
            logger.error(f"Error querying Gemini: {e}")
            time.sleep(100)
            self.chat = self.model.start_chat()
            self.chat.history = self.chat_history
            return self.query_gemini(prompt)

    def critique_and_refine(self, chat_history ,research_idea: str) -> str:
        """
        Critiques the given research idea and suggests refinements using the Gemini API.

        Args:
            research_idea (str): The research idea to critique.

        Returns:
            str: A string containing the refined research idea.
        """        
        self.chat = self.model.start_chat()
        if chat_history is not None:
            self.chat.history = chat_history
        self.chat_history = chat_history
        try:
            prompt = f"""
            You are an expert research scientist. Evaluate the following research idea for feasibility, 
            potential flaws, and areas for improvement. Provide a detailed critique and suggest specific 
            refinements to make the idea more robust and achievable.

            Research Idea: {research_idea}

            Provide your critique and refined idea in the following format:

            Critique: [Detailed critique of the research idea]
            Refined Idea: [Refined version of the research idea incorporating your suggestions]
            include mathematics if needed
            """

            response = self.query_gemini(prompt)
            logger.info(f"Refined research idea: {response}")
            return response , self.chat_history

        except Exception as e:
            logger.exception(f"Error critiquing and refining research idea: {e}")
            return f"Error critiquing and refining research idea: {e}"


if __name__ == '__main__':
    # Example Usage:
    # Assuming you have a config.yaml file with 'gemini_api_key'
    import yaml

    try:
        with open('/Users/krisanusarkar/Documents/ML/unt/generated/cais6/configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.  Make sure it exists and is in the correct location.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        exit()

    critiquing_agent = CritiquingAgent(config)
    research_idea = """
    Research Idea: Develop a new deep learning model for predicting stock prices based on social media sentiment analysis.
    Methodology: Collect Twitter data, train a sentiment analysis model, and use the sentiment scores as input to a deep learning model for stock price prediction.
    """
    refined_idea = critiquing_agent.critique_and_refine(None ,research_idea)
    print(f"Refined Research Idea:\n{refined_idea}")