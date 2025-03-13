from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GravWaveCritique",
    version="1.0.0",
    author="[Your Name]",
    author_email="[Your Email]",
    description="A Bayesian Hierarchical Model for Estimating Black Hole Distance and Redshift",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="[Your Repository URL]",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "PyMC3",
        "arviz",
        "astropy",
        "PyYAML",
        "pytest"
    ],
    entry_points={
        'console_scripts': [
            'gravwavecritique=main:main',  # Replace main with your actual main module name
        ],
    },
)