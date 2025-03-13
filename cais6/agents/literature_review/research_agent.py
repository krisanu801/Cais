import logging
from typing import Dict, Any, List
import google.generativeai as genai
import arxiv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Agent responsible for fetching and summarizing academic papers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ResearchAgent with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the Gemini API key.
        """
        try:
            self.api_key = config['gemini_api_key']
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.generation_config = {
                "temperature": 0.6,
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
            self.max_results = config.get('max_arxiv_results', 5)  # Default to 5 results
            logger.info("ResearchAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing ResearchAgent: {e}")
            raise

    def search_arxiv(self, query: str) -> List[arxiv.Result]:
        """
        Searches arXiv for relevant papers based on the given query.

        Args:
            query (str): The search query.

        Returns:
            List[arxiv.Result]: A list of arXiv results.
        """
        try:
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = list(search.results())
            logger.info(f"Found {len(results)} arXiv results for query: {query}")
            return results
        except Exception as e:
            logger.exception(f"Error searching arXiv: {e}")
            return []

    def summarize_paper(self, paper_abstract: str) -> str:
        """
        Summarizes the given paper abstract using the Gemini API.

        Args:
            paper_abstract (str): The abstract of the paper.

        Returns:
            str: A string containing the summarized abstract.
        """
        try:
            prompt = f"""
            You are an expert research scientist. Summarize the following academic paper abstract in a concise and informative manner.

            Abstract:
            {paper_abstract}

            Summary:
            [Concise Summary of the Abstract]
            """

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"The prompt was blocked due to: {response.prompt_feedback.block_reason}")
                return "The prompt was blocked due to safety concerns. Please refine the abstract."

            summary = response.text
            logger.info(f"Generated summary: {summary}")
            return summary

        except Exception as e:
            logger.exception(f"Error summarizing paper: {e}")
            return f"Error summarizing paper: {e}"

    def conduct_literature_review(self, research_idea: str) -> Dict[str, str]:
        """
        Conducts a literature review based on the given research idea.

        Args:
            research_idea (str): The research idea to conduct a literature review for.

        Returns:
            Dict[str, str]: A dictionary containing the summaries of relevant papers, keyed by paper title.
        """
        try:
            papers = self.search_arxiv(research_idea)
            summaries = {}
            for paper in papers:
                summary = self.summarize_paper(paper.summary)
                summaries[paper.title] = summary
            logger.info(f"Conducted literature review and generated summaries for {len(summaries)} papers.")
            return summaries
        except Exception as e:
            logger.exception(f"Error conducting literature review: {e}")
            return {}


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

    research_agent = ResearchAgent(config)
    research_idea = "Large Language Models in Scientific Discovery"
    literature = research_agent.conduct_literature_review(research_idea)

    if literature:
        print("Literature Review:")
        for title, summary in literature.items():
            print(f"\nTitle: {title}")
            print(f"Summary: {summary}")
    else:
        print("No literature found or an error occurred.")