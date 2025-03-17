import logging
from typing import Dict, Any, List, Tuple
import google.generativeai as genai
import arxiv
import requests
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import time
import re

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
            self.chat = None
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
            if self.chat is None:
                self.chat = self.model.start_chat()
            
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error querying Gemini: {e}")
            time.sleep(10)
            return self.query_gemini(prompt)

    def fetch_bibtex_from_semantic_scholar(self, title: str) -> str:
        """
        Fetches BibTeX entry for a given paper title from Semantic Scholar API.

        Args:
            title (str): The paper title to search.

        Returns:
            str: The BibTeX entry if found, else an empty string.
        """
        try:
            # URL encode the title for proper API request
            encoded_title = requests.utils.quote(title)
            api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_title}&fields=title,paperId"
            
            response = requests.get(api_url)
            logger.info(f"Semantic Scholar API response status: {response.status_code}")

            if response.status_code == 200:
                papers = response.json().get("data", [])
                if papers:
                    paper_id = papers[0]["paperId"]
                    bibtex_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/bibtex"
                    bibtex_response = requests.get(bibtex_url)
                    
                    if bibtex_response.status_code == 200:
                        logger.info(f"Successfully retrieved BibTeX for '{title}'")
                        return bibtex_response.text
                    else:
                        logger.warning(f"Failed to get BibTeX for paper ID {paper_id}: {bibtex_response.status_code}")
                else:
                    logger.warning(f"No papers found for title: '{title}'")
            else:
                logger.warning(f"Semantic Scholar API error: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Error fetching from Semantic Scholar: {e}")
        
        return ""

    def fetch_bibtex_from_arxiv(self, title: str) -> str:
        """
        Fetches BibTeX entry from arXiv if not found in Semantic Scholar.

        Args:
            title (str): The paper title.

        Returns:
            str: The BibTeX entry if found, else an empty string.
        """
        try:
            search = arxiv.Search(query=title, max_results=1)
            paper = next(search.results(), None)
            
            if paper:
                # Generate a citation key from the first author's last name and year
                author_last_name = paper.authors[0].name.split()[-1].lower() if paper.authors else "unknown"
                citation_key = f"{author_last_name}{paper.published.year}"
                
                # Create a properly formatted BibTeX entry
                bibtex_entry = f"""@article{{{citation_key},
  title = {{{paper.title}}},
  author = {{{' and '.join(author.name for author in paper.authors)}}},
  journal = {{arXiv preprint arXiv:{paper.entry_id.split('/')[-1]}}},
  year = {{{paper.published.year}}},
  url = {{{paper.entry_id}}},
  archivePrefix = {{arXiv}},
  primaryClass = {{{paper.primary_category if hasattr(paper, 'primary_category') else 'cs.LG'}}}
}}
"""
                logger.info(f"Successfully generated arXiv BibTeX for '{title}'")
                return bibtex_entry
            else:
                logger.warning(f"No results found on arXiv for '{title}'")
        except Exception as e:
            logger.warning(f"Error fetching from arXiv: {e}")
        
        return ""

    def create_fallback_bibtex(self, title: str) -> str:
        """
        Creates a basic BibTeX entry when no results are found from online sources.
        
        Args:
            title (str): The paper title.
            
        Returns:
            str: A basic BibTeX entry.
        """
        # Generate a simple citation key from the title
        citation_key = re.sub(r'[^a-zA-Z0-9]', '', title.split()[-1].lower() if title else "unknown")
        if citation_key:
            citation_key = f"{citation_key}{time.strftime('%Y')}"
        else:
            citation_key = f"unknown{time.strftime('%Y')}"
            
        bibtex_entry = f"""@misc{{{citation_key},
  title = {{{title}}},
  author = {{Unknown}},
  note = {{Citation details not found automatically}},
  year = {{{time.strftime('%Y')}}}
}}
"""
        logger.info(f"Created fallback BibTeX entry for '{title}'")
        return bibtex_entry

    def generate_citations(self, chat_history, literature: Dict[str, str]) -> Tuple[str, Any]:
        """
        Generates a BibTeX file containing citations for the given literature.

        Args:
            chat_history: Previous chat history if available
            literature (Dict[str, str]): A dictionary of paper titles and summaries.

        Returns:
            Tuple[str, Any]: A string containing the BibTeX file content and updated chat history.
        """
        # Initialize chat if needed
        if self.chat is None:
            self.chat = self.model.start_chat()
            
        # Load chat history if provided
        if chat_history is not None:
            self.chat.history = chat_history
        
        try:
            bib_db = BibDatabase()
            bib_db.entries = []
            
            for title, summary in literature.items():
                logger.info(f"Processing citation for: '{title}'")
                
                # Try fetching from Semantic Scholar first
                bibtex_entry_str = self.fetch_bibtex_from_semantic_scholar(title)
                
                # If not found in Semantic Scholar, try arXiv
                if not bibtex_entry_str:
                    logger.info(f"Paper '{title}' not found in Semantic Scholar, searching arXiv...")
                    bibtex_entry_str = self.fetch_bibtex_from_arxiv(title)
                
                # If still not found, create a fallback entry
                if not bibtex_entry_str:
                    logger.info(f"Paper '{title}' not found in online sources, creating fallback entry...")
                    bibtex_entry_str = self.create_fallback_bibtex(title)

                # Process the BibTeX entry if we have one
                if bibtex_entry_str:
                    try:
                        parser = BibTexParser(common_strings=True)
                        parser.ignore_nonstandard_types = False
                        bib_db_entry = parser.parse(bibtex_entry_str)
                        
                        if bib_db_entry.entries:
                            bib_db.entries.extend(bib_db_entry.entries)
                            logger.info(f"Added BibTeX entry for '{title}'")
                        else:
                            logger.warning(f"Could not parse BibTeX entry for '{title}'")
                    except Exception as parse_error:
                        logger.warning(f"Error parsing BibTeX for '{title}': {parse_error}")
                        # Try to add a fallback entry if parsing fails
                        fallback_entry = self.create_fallback_bibtex(title)
                        try:
                            bib_db_entry = parser.parse(fallback_entry)
                            if bib_db_entry.entries:
                                bib_db.entries.extend(bib_db_entry.entries)
                                logger.info(f"Added fallback BibTeX entry for '{title}'")
                        except:
                            logger.error(f"Could not add fallback entry for '{title}'")
                else:
                    logger.warning(f"Could not generate any BibTeX entry for '{title}'")

            # Write the BibTeX content
            writer = BibTexWriter()
            writer.indent = '  '  # Consistent indentation
            writer.comma_first = False  # Place commas at the end of lines
            
            if bib_db.entries:
                bibtex_str = writer.write(bib_db)
                logger.info(f"Generated BibTeX file with {len(bib_db.entries)} entries.")
            else:
                bibtex_str = "% No citation entries could be generated."
                logger.warning("No citation entries could be generated.")

            return bibtex_str, self.chat.history

        except Exception as e:
            logger.exception(f"Error generating citations: {e}")
            return f"% Error generating citations: {e}", chat_history


if __name__ == '__main__':
    # Example Usage:
    import yaml

    try:
        with open('/Users/krisanusarkar/Documents/ML/unt/generated/cais6/configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Make sure it exists and is in the correct location.")
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

    if "% Error generating citations" not in citations and "% No citation entries" not in citations:
        print("Generated Citations:\n", citations)
    else:
        print("Error or no citations generated:\n", citations)
