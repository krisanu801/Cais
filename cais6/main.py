import sys
import os
import logging
import yaml
from typing import Dict, Any
import google.generativeai as genai
import shutil

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from cais6.agents.generating.generating_agent import GeneratingAgent
from cais6.agents.critiquing.critiquing_agent import CritiquingAgent
from cais6.agents.execution.execution_agent import generate
from cais6.agents.error_handling.error_handling_agent import ErrorHandlingAgent
from cais6.agents.literature_review.research_agent import ResearchAgent
from cais6.agents.citation.citation_agent import CitationAgent
from cais6.agents.paper_writing.writing_agent import WritingAgent
from cais6.agents.self_review.self_review_agent import SelfReviewAgent
from cais6.utils.file_manager import FileManager
from cais6.agents.thinking import MultiAgentStepByStepSystem
from cais6.agents.rethink import CheckingResultsAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the configuration from a YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def delete_all_files(folder_path):
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Deletes subdirectories as well
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print("All files and subfolders deleted successfully.")



def main(research_topic: str) -> None:
    """Main function to orchestrate the AI research pipeline.

    Args:
        research_topic (str): The research topic to investigate.
    """
    try:
        # Load configuration
        config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
        config = load_config(config_path)

        # Initialize agents
        generating_agent = GeneratingAgent(config)
        critiquing_agent = CritiquingAgent(config)
        #execution_agent = ExecutionAgent(config)
        error_handling_agent = ErrorHandlingAgent(config)
        research_agent = ResearchAgent(config)
        citation_agent = CitationAgent(config)
        writing_agent = WritingAgent(config)
        self_review_agent = SelfReviewAgent(config)
        thinking_agent = MultiAgentStepByStepSystem()
        checking_results_agent = CheckingResultsAgent(config)

        # Initialize file manager
        file_manager = FileManager(config)

        chat_history = None

        # Phase 1: Idea Generation and Refinement
        #research_idea , chat_history = generating_agent.generate_research_idea(chat_history , research_topic)
        #refined_idea , chat_history = critiquing_agent.critique_and_refine(chat_history , research_idea)
        while True:
            refined_idea , chat_history = thinking_agent.run(chat_history , research_topic)
            logger.info(f"Refined Research Idea: {refined_idea}")

            # Phase 2: Code Execution
            #code = execution_agent.generate_code(refined_idea)
            results , chat_history = generate(research_topic , chat_history )
            logger.info(f"Code Execution Results: {results}")

            pursue , chat_history = checking_results_agent.critique(chat_history , results) 
            
            if pursue in {1, '1'}:
                file_manager.save_results(results, "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results/initial_results.txt")
                break
            else:
                delete_all_files("/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results")
                delete_all_files("/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/code")

        # Phase 3: Literature Review
        literature , chat_history = research_agent.conduct_literature_review(chat_history , research_topic)
        citations , chat_history = citation_agent.generate_citations(chat_history , literature)

        file_manager.save_literature(literature, "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results/literature.txt")
        file_manager.save_citations(citations, "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/results/citations.bib")

        logger.info(f"Literature Review: {literature}")
        logger.info(f"Citations: {citations}")

        # Phase 4: Paper Writing
        draft_paper , chat_history = writing_agent.draft_paper(chat_history , refined_idea, results, literature, citations)
        reviewed_paper , chat_history = self_review_agent.review_and_revise(chat_history , draft_paper)

        file_manager.save_paper(reviewed_paper, "/Users/krisanusarkar/Documents/ML/unt/generated/cais6/cais6/outputs/papers/final_paper.tex")

        logger.info(f"Final Paper: {reviewed_paper}")

        logger.info("Research pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during the research pipeline: {e}")


if __name__ == "__main__":
    # Example usage:
    research_topic = input("Enter a research topic: ")
    main(research_topic)