from setuptools import setup, find_packages

setup(
    name='OptimizerPhaseTransitions',
    version='0.1.0',
    description='Investigating Phase Transitions in Optimization Dynamics',
    author='[Your Name]',
    author_email='[Your Email]',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'torch',
        'torchvision',
        'matplotlib',
        'pyyaml',
        'tqdm',
        'statsmodels'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)