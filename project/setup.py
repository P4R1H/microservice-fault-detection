"""
Setup script for AIOps RCA project.

Install in editable mode:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="aiops-rca",
    version="0.1.0",
    description="AIOps Multimodal Root Cause Analysis System",
    author="Bachelor Thesis Project",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric",
        "chronos-forecasting>=1.0.0",
        "tigramite>=5.1.0",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "networkx",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest", "black", "pylint", "mypy"],
    },
)
