import logging
from typing import Dict, Any
import google.generativeai as genai
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SelfReviewAgent:
    """
    Agent responsible for providing critical feedback on the drafted paper.
    """

    def __init__(self,config: Dict[str, Any]):
        """
        Initializes the SelfReviewAgent with a configuration.

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
                "max_output_tokens": 33554432,
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
            logger.info("SelfReviewAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing SelfReviewAgent: {e}")
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

    def review_and_revise(self,chat_history ,   latex_paper: str) -> str:
        """
        Reviews the given LaTeX paper and suggests revisions for clarity, coherence, and logical soundness using the Gemini API.

        Args:
            latex_paper (str): The LaTeX paper to review.

        Returns:
            str: A string containing the revised LaTeX paper.
        """
        self.chat = self.model.start_chat()
        if chat_history is not None:
            self.chat.history = chat_history
        self.chat_history = chat_history
        try:
            prompt = f"""
            You are an expert research scientist and editor. Review the following LaTeX research paper of 6 to 7 pages for clarity, coherence, logical soundness, and grammatical correctness.
            Provide specific suggestions for improvement and revise the paper accordingly.
            Incorporate the provided results and cite relevant literature using the provided BibTeX entries.
            Include mathematical equations if needed.
            Include paths of images that you have already saved. along with proper description of them in results.
            give extremely detailed output.Incorporate the provided results and cite relevant literature using the provided BibTeX entries.
            Include mathematical equations if needed.
            Include paths of images that you have already saved. along with proper description of them in results.
            give extremely detailed output.

            LaTeX Research Paper:
            ```latex
            {latex_paper}
            ```

            Revised LaTeX Research Paper:
            ```latex
            [Revised LaTeX paper incorporating your suggestions]
            ```
            """

            revised_latex_paper = self.query_gemini(prompt)
            # Extract LaTeX code from markdown format if present
            if "```latex" in revised_latex_paper:
                revised_latex_paper = revised_latex_paper.split("```latex")[1].split("```")[0].strip()

            logger.info(f"Revised LaTeX paper:\n{revised_latex_paper}")
            return revised_latex_paper , self.chat_history

        except Exception as e:
            logger.exception(f"Error reviewing and revising LaTeX paper: {e}")
            return f"% Error reviewing and revising LaTeX paper: {e}"


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

    self_review_agent = SelfReviewAgent(None , config)
    latex_paper = """
    \\documentclass{article}
    \\title{A Simple Paper}
    \\begin{document}
    \\maketitle
    This is a simple paper. It is very simple.
    \\end{document}
    """
    revised_latex_paper = self_review_agent.review_and_revise(latex_paper)

    if "% Error reviewing and revising LaTeX paper" not in revised_latex_paper:
        print("Revised LaTeX Paper:\n", revised_latex_paper)
    else:
        print("Error reviewing and revising LaTeX paper:\n", revised_latex_paper)