import logging
import subprocess
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compile_latex(latex_string: str, output_path: str = "output.pdf") -> Optional[str]:
    """
    Compiles a LaTeX string into a PDF document.

    Args:
        latex_string (str): The LaTeX code to compile.
        output_path (str, optional): The path to save the compiled PDF. Defaults to "output.pdf".

    Returns:
        Optional[str]: The path to the compiled PDF if successful, None otherwise.
    """
    try:
        # Create a temporary file to store the LaTeX code
        with open("temp.tex", "w") as f:
            f.write(latex_string)

        # Compile the LaTeX code using pdflatex
        process = subprocess.run(
            ["pdflatex", "temp.tex", "-output-directory", os.path.dirname(output_path)],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(output_path) or ".",
        )

        if process.returncode != 0:
            logger.error(f"LaTeX compilation failed:\n{process.stderr}")
            return None

        # Rename the output PDF to the desired path
        os.rename(os.path.join(os.path.dirname(output_path) or ".", "temp.pdf"), output_path)

        # Clean up temporary files
        os.remove("temp.tex")
        os.remove("temp.log")
        os.remove("temp.aux")

        logger.info(f"LaTeX compiled successfully to {output_path}")
        return output_path

    except FileNotFoundError:
        logger.error("pdflatex not found. Ensure LaTeX is installed and in your PATH.")
        return None
    except Exception as e:
        logger.exception(f"Error compiling LaTeX: {e}")
        return None


def extract_text_from_latex(latex_string: str) -> str:
    """
    Extracts plain text from a LaTeX string.  This is a very basic implementation and may not handle all LaTeX constructs correctly.

    Args:
        latex_string (str): The LaTeX code to extract text from.

    Returns:
        str: The extracted plain text.
    """
    try:
        # A very basic approach: remove LaTeX commands and environments
        text = latex_string.replace("\\documentclass{article}", "")
        text = text.replace("\\begin{document}", "")
        text = text.replace("\\end{document}", "")
        text = text.replace("\\maketitle", "")
        text = text.replace("\\title{", "")
        text = text.replace("}", "")
        text = text.replace("\\section{", "")
        text = text.replace("\\subsection{", "")
        text = text.replace("\\textit{", "")
        text = text.replace("\\textbf{", "")
        text = text.replace("\\emph{", "")
        text = text.replace("\\cite{", "")
        text = text.replace("}", "")
        text = text.replace("\\label{", "")
        text = text.replace("}", "")
        text = text.replace("\\\\", "\n")  # Replace line breaks
        text = text.replace("\\%", "%") # Handle escaped percentage signs

        # Remove other common LaTeX commands (this is not exhaustive)
        text = text.replace("\\usepackage{", "")
        text = text.replace("\\newcommand{", "")
        text = text.replace("\\def{", "")

        text = text.strip()
        logger.info("Extracted text from LaTeX string.")
        return text

    except Exception as e:
        logger.exception(f"Error extracting text from LaTeX: {e}")
        return f"Error extracting text from LaTeX: {e}"


# Example Usage (This is just a class definition, so no direct execution here)
if __name__ != "__main__":
    # Example Usage (This won't run when imported as a module)
    # To use this, you would need to call the functions.
    # For example:
    #
    # from utils.latex_utils import compile_latex, extract_text_from_latex
    #
    # latex_code = """
    # \\documentclass{article}
    # \\title{My LaTeX Document}
    # \\begin{document}
    # \\maketitle
    # This is a simple LaTeX document.
    # \\end{document}
    # """
    #
    # pdf_path = compile_latex(latex_code, "my_document.pdf")
    # if pdf_path:
    #     print(f"PDF compiled successfully to: {pdf_path}")
    #
    # plain_text = extract_text_from_latex(latex_code)
    # print(f"Extracted text:\n{plain_text}")
    pass