import logging
from typing import Dict, Any, Tuple
import google.generativeai as genai
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ErrorHandlingAgent:
    """
    Agent responsible for detecting and handling errors in code execution.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ErrorHandlingAgent with a configuration.

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
            self.max_retries = config.get('max_code_execution_retries', 3)  # Default to 3 retries
            logger.info("ErrorHandlingAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing ErrorHandlingAgent: {e}")
            raise

    def analyze_error(self, code: str, error_message: str) -> str:
        """
        Iteratively analyzes and fixes errors in the code.

        Args:
            code (str): The Python code that produced the error.
            error_message (str): The error message.

        Returns:
            str: A corrected version of the Python code.
        """
        try:
            for attempt in range(self.max_retries):
                prompt = f"""
                You are an expert Python programmer and debugger. Given the following Python code and its error message, 
                analyze the issue and return a corrected version of the full code, ensuring it runs successfully.

                Code:
                ```python
                {code}
                ```

                Error Message:
                {error_message}

                Corrected Code (full script, properly formatted):
                ```python
                Break the script into three parts:\n
            Return only one part at a time.
            for installing dependecies add according to this template at start of scipt :
            for example assuming numpy is the required dependency
            import subprocess
            import sys

            try:
                import numpy
            except ImportError:
                print("NumPy not found. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
                import numpy  # Try importing again after installation
                """

                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )

                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                    return "# The prompt was blocked due to safety concerns."

                corrected_code = response.text.strip()

                # Extract actual code from markdown formatting if present
                if "```python" in corrected_code:
                    corrected_code = corrected_code.split("```python")[1].split("```")[0].strip()

                if corrected_code != code:  # If code changed, assume fix was applied
                    return corrected_code
                
                logger.warning("No change detected in the corrected code. Retrying with refined prompt...")
                time.sleep(2)  # Avoid rapid API calls

            return f"# Failed to correct the code after {self.max_retries} retries."
        
        except Exception as e:
            logger.exception(f"Error analyzing and fixing the code: {e}")
            return f"# Error analyzing and fixing the code: {e}"

    def execute_code_with_retry(self, code: str) -> str:
        """
        Executes the given code and retries if an error occurs, up to a maximum number of retries.

        Args:
            code (str): The Python code to execute.

        Returns:
            str: The standard output of the successful execution, or an error message if all retries fail.
        """
        from cais6.agents.execution.execution_agent import ExecutionAgent  # Import here to avoid circular dependency
        execution_agent = ExecutionAgent(config={"gemini_api_key": self.api_key}) # Re-initialize with just the API key

        for attempt in range(self.max_retries):
            logger.info(f"Attempting code execution (attempt {attempt + 1}/{self.max_retries}).")
            stdout, stderr = execution_agent.execute_code(code)

            if stderr:
                logger.warning(f"Code execution failed with error:\n{stderr}")
                corrected_code = self.analyze_error(code, stderr)

                if "Error analyzing error" in corrected_code:
                    return f"Code execution failed after {self.max_retries} retries. Error analyzing the error: {corrected_code}"

                code = corrected_code  # Update the code with the corrected version
                time.sleep(2)  # Wait before retrying
            else:
                logger.info("Code execution successful.")
                return stdout

        logger.error(f"Code execution failed after {self.max_retries} retries.")
        return f"Code execution failed after {self.max_retries} retries. Last error: {stderr}"


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

    error_handling_agent = ErrorHandlingAgent(config)

    # Example code with an error
    faulty_code = """
    import numpy as np
    data = [1, 2, 3, 4, 5]
    mean = np.mean(dat) # Typo: 'dat' instead of 'data'
    print(f"Mean: {mean}")
    """

    results = error_handling_agent.execute_code_with_retry(faulty_code)
    print(f"Execution Results:\n{results}")