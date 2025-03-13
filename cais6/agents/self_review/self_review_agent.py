import logging
from typing import Dict, Any
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SelfReviewAgent:
    """
    Agent responsible for providing critical feedback on the drafted paper.
    """

    def __init__(self, config: Dict[str, Any]):
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
                "max_output_tokens": 8192*256,
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

    def review_and_revise(self, latex_paper: str) -> str:
        """
        Reviews the given LaTeX paper and suggests revisions for clarity, coherence, and logical soundness using the Gemini API.

        Args:
            latex_paper (str): The LaTeX paper to review.

        Returns:
            str: A string containing the revised LaTeX paper.
        """
        try:
            prompt = f"""
            You are an expert research scientist and editor. Review the following LaTeX research paper for clarity, coherence, logical soundness, and grammatical correctness.
            Provide specific suggestions for improvement and revise the paper accordingly.

            LaTeX Research Paper:
            ```latex
            {latex_paper}
            ```

            Revised LaTeX Research Paper:
            ```latex
            [Revised LaTeX paper incorporating your suggestions]
            ```
            """

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"The prompt was blocked due to: {response.prompt_feedback.block_reason}")
                return "% The prompt was blocked due to safety concerns. Please refine the paper content."

            revised_latex_paper = response.text
            # Extract LaTeX code from markdown format if present
            if "```latex" in revised_latex_paper:
                revised_latex_paper = revised_latex_paper.split("```latex")[1].split("```")[0].strip()

            logger.info(f"Revised LaTeX paper:\n{revised_latex_paper}")
            return revised_latex_paper

        except Exception as e:
            logger.exception(f"Error reviewing and revising LaTeX paper: {e}")
            return f"% Error reviewing and revising LaTeX paper: {e}"


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

    self_review_agent = SelfReviewAgent(config)
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