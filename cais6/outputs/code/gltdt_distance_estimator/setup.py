from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gltldt_distance_estimator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for estimating distances using Gravitational Lensing Time Delay Tomography.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gltldt_distance_estimator", # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "keras",
        "PyYAML",
        "emcee",
        "corner",
        "dvc",
        "george",
        "pywavelets",
        "Pillow",
        "seaborn"
    ],
    entry_points={
        'console_scripts': [
            'gltldt = src.main:main',  # Creates a command-line tool named 'gltldt'
        ],
    },
    package_data={
        'configs': ['configs/config.yaml', 'configs/logging.conf'], # Include config files
    },
    include_package_data=True,
)