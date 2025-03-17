import logging
from typing import Dict, Any, List
import google.generativeai as genai
import arxiv
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

            

class CitationAgent:
    """
    Agent responsible for ensuring proper LaTeX/BibTeX citations.
    """

    def __init__(self , config: Dict[str, Any]):
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

            bibtex_entry = self.query_gemini(prompt)
            logger.info(f"Generated BibTeX entry:\n{bibtex_entry}")
            return bibtex_entry

        except Exception as e:
            logger.exception(f"Error generating BibTeX entry: {e}")
            return f"% Error generating BibTeX entry: {e}"

    def generate_citations(self, chat_history, literature: Dict[str, str]) -> str:
        """
        Generates a BibTeX file containing citations for the given literature.

        Args:
            literature (Dict[str, str]): A dictionary of paper titles and summaries.

        Returns:
            str: A string containing the BibTeX file content.
        """
        self.chat = self.model.start_chat()
        if chat_history is not None:
            self.chat.history = chat_history
        self.chat_history = chat_history
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
            return bibtex_str , self.chat_history

        except Exception as e:
            logger.exception(f"Error generating citations: {e}")
            return f"% Error generating citations: {e}"


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
    chat_history = None

    citation_agent = CitationAgent(config)
    literature = {
        "Attention is All You Need": "This paper introduces the Transformer model.",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": "This paper introduces BERT."
    }
    citations, chat_history = citation_agent.generate_citations(chat_history, literature)

    if "% Error generating citations" not in citations:
        print("Generated Citations:\n", citations)
    else:
        print("Error generating citations:\n", citations)