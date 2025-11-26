"""Setup script for classification_dynamics package."""

from setuptools import setup, find_packages

setup(
    name="classification-dynamics",
    version="0.1.0",
    description="Classification dynamics for ROA estimation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "torchmetrics>=0.11.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)


