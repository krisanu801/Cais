# cais6: Multi-Agent AI Scientist System

## Project Description

cais6 is a Python project that aims to develop a fully autonomous, multi-agent AI Scientist system. Given a research topic, the system autonomously executes the entire research pipeline using the Gemini API via GenAI as the sole LLM model. The system is structured into collaborative agents that iteratively refine each phase before progressing.

## Key Features

- **Autonomous Research Pipeline:** Handles idea refinement, methodology selection, code execution, error handling, result analysis, literature review, and manuscript preparation without human intervention.
- **Multi-Agent Collaboration:** Employs a team of specialized agents that work together to complete the research process.
- **Iterative Refinement:** Ensures that if any phase produces weak results, it loops back for refinement before proceeding.
- **Gemini API Integration:** Leverages the Gemini API via GenAI for all LLM-based operations.
- **Comprehensive Output:** Generates a submission-ready research paper, along with all code, datasets, results, and supplementary materials.

## Agents

- **Generating Agent:** Proposes novel research ideas and methodologies.
- **Critiquing Agent:** Evaluates feasibility, points out flaws, and forces revisions.
- **Execution Agent:** Writes and runs Python scripts for data processing, simulations, or model training.
- **Error Handling Agent:** Detects failures, refines the code, and reruns experiments until valid results are obtained.
- **Research Agent:** Fetches and summarizes academic papers.
- **Citation Agent:** Ensures proper LaTeX/BibTeX citations while connecting past research to the current study.
- **Writing Agent:** Drafts a structured LaTeX research paper incorporating results, citations, and figures.
- **Self-Review Agent:** Provides critical feedback, ensuring clarity, coherence, and logical soundness.

## Project Structure

```
cais6/
├── agents/
│   ├── generating/
│   │   └── generating_agent.py
│   ├── critiquing/
│   │   └── critiquing_agent.py
│   ├── execution/
│   │   └── execution_agent.py
│   ├── error_handling/
│   │   └── error_handling_agent.py
│   ├── literature_review/
│   │   └── research_agent.py
│   ├── citation/
│   │   └── citation_agent.py
│   ├── paper_writing/
│   │   └── writing_agent.py
│   └── self_review/
│       └── self_review_agent.py
├── utils/
│   ├── gemini_api.py
│   ├── latex_utils.py
│   └── file_manager.py
├── data/
├── logs/
├── outputs/
│   ├── papers/
│   ├── code/
│   ├── datasets/
│   └── results/
├── tests/
├── configs/
│   └── config.yaml
├── main.py
├── requirements.txt
├── README.md
└── setup.py
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [repository_url]
   cd cais6
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure the `configs/config.yaml` file:**
   - Add your Gemini API key.
   - Customize other settings as needed.

6. **Run the main application:**
   ```bash
   python cais6/main.py
   ```

## Configuration

The `configs/config.yaml` file contains the following configuration options:

- `gemini_api_key`: Your Gemini API key.
- `output_dir`: The directory where generated files will be saved.
- `max_code_execution_retries`: The maximum number of times to retry code execution if an error occurs.
- `max_arxiv_results`: The maximum number of arXiv results to fetch for literature review.
- Agent-specific configurations: You can customize the behavior of individual agents by modifying their respective configurations.

## Dependencies

- google-generativeai
- python-dotenv
- requests
- beautifulsoup4
- arxiv
- PyYAML
- tiktoken
- openai
- tenacity
- matplotlib
- pandas
- scipy
- latexcodec
- bibtexparser

## Sample Output 

For a simple prompt like compare adam on mnist and makemoons , it created a well draft pdf  , along with creating code scripts  , and executing them , the final result is [paper](AIgeneratedpaper1.pdf)

## Logging

The system uses the `logging` module for logging events and errors. The logging configuration can be found in the `configs/logging.conf` file.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[License](License)
