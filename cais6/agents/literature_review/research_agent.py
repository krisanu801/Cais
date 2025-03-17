import logging
from typing import Dict, Any, List
import google.generativeai as genai
import arxiv
import os
import fitz  # pymupdf
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import time

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
            #self.chat = chat
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

    def extract_text_from_pdf(self , pdf_path , use_ocr=False):
        """
        Extract text from PDF files, supporting both digital and scanned PDFs.
    
        Args:
            pdf_path (str): Path to the PDF file
            use_ocr (bool): Force OCR even for digital PDFs
        
        Returns:
            str: Extracted text content
        """
        # Try extracting text directly first
        if not use_ocr:
            try:
                # Method 1: PyMuPDF (fitz) - faster and handles more complex layouts
                doc = fitz.open(pdf_path)
                text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text("text")
                doc.close()
                
                # If we got meaningful text, return it
                if len(text.strip()) > 100:  # Arbitrary threshold to detect successful extraction
                    return text
                
                # Backup method for digital PDFs with complex layouts
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    
                if len(text.strip()) > 100:
                    return text
        
            except Exception as e:
                print(f"Direct extraction failed: {e}")
        
        # If direct extraction failed or force OCR is enabled, use OCR
        print("Using OCR for text extraction...")
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        text = ""
        
        # Process each page with OCR
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            page_text = pytesseract.image_to_string(image)
            text += f"\n--- Page {i+1} ---\n{page_text}"
        
        return text

    def summarize_paper(self, paper: str) -> str:
        """
        Summarizes the given paper abstract using the Gemini API.

        Args:
            paper (str): The abstract of the paper.

        Returns:
            str: A string containing the summarized abstract.
        """
        try:
            prompt = f"""
            You are an expert research scientist. Summarize the following academic paper abstract in a concise and informative manner.

            Paper: {paper}

            you can include mathematics if you want.
            """

            summary = self.query_gemini(prompt)
            logger.info(f"Generated summary: {summary}")
            return summary

        except Exception as e:
            logger.exception(f"Error summarizing paper: {e}")
            return f"Error summarizing paper: {e}"

    def finding_topic(self , research_idea : str):
        try:
            prompt = f"""
            according to the researches we have done till now , i am trying to write a research paper , 
            and i want to conduct literature review ,which topic should i search on arxiv.{research_idea}

            give me only one topic . and just the topic nothing more nothing less.
            """
            topic = self.query_gemini(prompt).strip()
            logger.info(f"topic for literature review: {topic}")
            return topic
        except Exception as e:
            logger.exception(f"Error finding topic: {e}")
            return f"Error finding topic: {e}"

    def download_arxiv_papers(self, query, num_papers=3, save_dir="/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/arxiv_papers"):
        os.makedirs(save_dir, exist_ok=True)  # Ensure folder exists

        search = arxiv.Search(
            query=query,
            max_results=num_papers
        )
        save_paths = []

        for paper in search.results():
            paper_id = paper.entry_id.split("/")[-1]  # Extract arXiv ID
            pdf_url = paper.pdf_url  # Get the PDF link
            save_path = os.path.join(save_dir, f"{paper_id}.pdf")
        
            print(f"Downloading: {paper.title}")
            paper.download_pdf(filename=save_path)  # Download full paper
        
            print(f"Saved: {save_path}")
            save_paths.append(save_path)
        return save_paths

    def conduct_literature_review(self,chat_history , research_idea: str) -> Dict[str, str]:
        """
        Conducts a literature review based on the given research idea.

        Args:
            research_idea (str): The research idea to conduct a literature review for.

        Returns:
            Dict[str, str]: A dictionary containing the summaries of relevant papers, keyed by paper title.

        """
        self.chat = self.model.start_chat()
        if chat_history is not None:
            self.chat.history = chat_history
        self.chat_history = chat_history
        research_idea = self.finding_topic(research_idea)
        logger.info(f"Research idea: {research_idea}")
        try:
            save_paths = self.download_arxiv_papers(research_idea)
            summaries = {}
            for save_path in save_paths:
                try:
                    summary = self.summarize_paper(self.extract_text_from_pdf(save_path))
                    summaries[save_path] = summary
                except Exception as e:
                    logger.exception(f"Error summarizing paper: {e}")
                    summaries[save_path] = f"Error summarizing paper: {e}"
            logger.info(f"Conducted literature review and generated summaries for {len(summaries)} papers.")
            return summaries , self.chat_history
        except Exception as e:
            logger.exception(f"Error conducting literature review: {e}")
            return {} , self.chat_history


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

    research_agent = ResearchAgent(config)
    research_idea = "create a better optimizer than rmsprop for nonconvex landscape"
    literature = research_agent.conduct_literature_review(None , research_idea)

    if literature:
        print("Literature Review:")
        for title, summary in literature.items():
            print(f"\nTitle: {title}")
            print(f"Summary: {summary}")
    else:
        print("No literature found or an error occurred.")