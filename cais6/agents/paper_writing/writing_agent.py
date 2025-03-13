import logging
from typing import Dict, Any
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WritingAgent:
    """
    Agent responsible for drafting the LaTeX research paper.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the WritingAgent with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the Gemini API key.
        """
        try:
            self.api_key = config['gemini_api_key']
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.generation_config = {
                "temperature": 0.8,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 8192*256,  # Increased token limit for longer papers
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
            logger.info("WritingAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing WritingAgent: {e}")
            raise

    def draft_paper(self, research_idea: str, results: str, literature: Dict[str, str], citations: str) -> str:
        """
        Drafts a LaTeX research paper based on the given research idea, results, literature, and citations using the Gemini API.

        Args:
            research_idea (str): The research idea.
            results (str): The results of the experiments.
            literature (Dict[str, str]): A dictionary of paper titles and summaries.
            citations (str): The BibTeX citations.

        Returns:
            str: A string containing the LaTeX research paper.
        """
        try:
            prompt = f"""
            You are an expert research scientist and technical writer. Draft a complete LaTeX research paper in IEEE format based on the following information.
            Include the following sections: Introduction, Methods, Results, Discussion, and Conclusion.
            Incorporate the provided results and cite relevant literature using the provided BibTeX entries.
            Include mathematical equations if needed.

            Research Idea: {research_idea}
            Results: {results}
            Literature Review:
            {literature}
            Citations (BibTeX):
            {citations}

            LaTeX Research Paper:
            ```latex
            [Complete LaTeX research paper]
            ```
            """

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"The prompt was blocked due to: {response.prompt_feedback.block_reason}")
                return "% The prompt was blocked due to safety concerns. Please refine the research idea, results, or literature."

            latex_paper = response.text
            # Extract LaTeX code from markdown format if present
            if "```latex" in latex_paper:
                latex_paper = latex_paper.split("```latex")[1].split("```")[0].strip()

            logger.info(f"Drafted LaTeX paper:\n{latex_paper}")
            return latex_paper

        except Exception as e:
            logger.exception(f"Error drafting LaTeX paper: {e}")
            return f"% Error drafting LaTeX paper: {e}"


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

    writing_agent = WritingAgent(config)
    research_idea = "Impact of Large Language Models on Scientific Writing"
    results = "Our experiments show that LLMs can generate coherent drafts."
    literature = {
        "Attention is All You Need": "Transformer models are important.",
        "BERT": "BERT is another important model."
    }
    citations = """
    @article{vaswani2017attention,
      title={Attention is all you need},
      author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
      journal={Advances in neural information processing systems},
      volume={30},
      year={2017}
    }
    """
    latex_paper = writing_agent.draft_paper(research_idea, results, literature, citations)

    if "% Error drafting LaTeX paper" not in latex_paper:
        print("Drafted LaTeX Paper:\n", latex_paper)
    else:
        print("Error drafting LaTeX paper:\n", latex_paper)