import logging
from typing import Dict, Any, List
import google.generativeai as genai
import arxiv
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CitationAgent:
    """
    Agent responsible for ensuring proper LaTeX/BibTeX citations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the CitationAgent with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the Gemini API key.
        """
        try:
            self.api_key = config['gemini_api_key']
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.generation_config = {
                "temperature": 0.5,
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
            logger.info("CitationAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing CitationAgent: {e}")
            raise

    def generate_bibtex_entry(self, paper: arxiv.Result) -> str:
        """
        Generates a BibTeX entry for the given arXiv paper using the Gemini API.

        Args:
            paper (arxiv.Result): The arXiv paper to generate a BibTeX entry for.

        Returns:
            str: A string containing the BibTeX entry.
        """
        try:
            prompt = f"""
            You are an expert in generating BibTeX entries for academic papers.
            Generate a complete and accurate BibTeX entry for the following arXiv paper.

            Title: {paper.title}
            Authors: {', '.join(author.name for author in paper.authors)}
            Abstract: {paper.summary}
            Published: {paper.published}
            ArXiv URL: {paper.pdf_url}

            BibTeX Entry:
            [Complete BibTeX entry for the paper]
            """

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"The prompt was blocked due to: {response.prompt_feedback.block_reason}")
                return "% The prompt was blocked due to safety concerns. Please refine the paper details."

            bibtex_entry = response.text
            logger.info(f"Generated BibTeX entry:\n{bibtex_entry}")
            return bibtex_entry

        except Exception as e:
            logger.exception(f"Error generating BibTeX entry: {e}")
            return f"% Error generating BibTeX entry: {e}"

    def generate_citations(self, literature: Dict[str, str]) -> str:
        """
        Generates a BibTeX file containing citations for the given literature.

        Args:
            literature (Dict[str, str]): A dictionary of paper titles and summaries.

        Returns:
            str: A string containing the BibTeX file content.
        """
        try:
            bib_db = BibDatabase()
            for title in literature.keys():
                search = arxiv.Search(query=title, max_results=1)
                paper = next(search.results(), None)  # Get the first result or None
                if paper:
                    bibtex_entry_str = self.generate_bibtex_entry(paper)
                    if "% Error generating BibTeX entry" not in bibtex_entry_str:
                        parser = BibTexParser()
                        bib_db_entry = parser.parse(bibtex_entry_str).entries
                        if bib_db_entry:
                            bib_db.entries.extend(bib_db_entry)
                        else:
                            logger.warning(f"Could not parse BibTeX entry for {title}")
                    else:
                        logger.warning(f"Could not generate BibTeX entry for {title}: {bibtex_entry_str}")
                else:
                    logger.warning(f"Could not find arXiv entry for {title}")

            writer = BibTexWriter()
            bibtex_str = writer.write(bib_db)

            logger.info("Generated BibTeX file.")
            return bibtex_str

        except Exception as e:
            logger.exception(f"Error generating citations: {e}")
            return f"% Error generating citations: {e}"


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

    citation_agent = CitationAgent(config)
    literature = {
        "Attention is All You Need": "This paper introduces the Transformer model.",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": "This paper introduces BERT."
    }
    citations = citation_agent.generate_citations(literature)

    if "% Error generating citations" not in citations:
        print("Generated Citations:\n", citations)
    else:
        print("Error generating citations:\n", citations)