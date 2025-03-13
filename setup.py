import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cais6",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Agent AI Scientist System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cais6",  # Replace with your repository URL
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "google-generativeai",
        "python-dotenv",
        "requests",
        "beautifulsoup4",
        "arxiv",
        "PyYAML",
        "tiktoken",
        "openai",
        "tenacity",
        "matplotlib",
        "pandas",
        "scipy",
        "latexcodec",
        "bibtexparser"
    ],
    entry_points={
        'console_scripts': [
            'cais6=cais6.main:main',  # Optional: If you want to create a command-line tool
        ],
    },
)