"""TriFlowQC setup script."""
from setuptools import setup, find_packages

setup(
    name="TriFlowQC",
    version="1.0.0",
    description="Qutrit quantum neural network framework with real hardware support",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="TriFlowQC Contributors",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["triflowqc*"]),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    extras_require={
        "ibm": ["qiskit>=1.0", "qiskit-ibm-runtime>=0.20", "qiskit-aer>=0.13"],
        "qutip": ["qutip>=5.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "pytest-timeout",
            "pytest-html",
            "qiskit>=1.0",
            "qiskit-ibm-runtime>=0.20",
            "qiskit-aer>=0.13",
            "qutip>=5.0",
            "scikit-learn>=1.3",
            "black",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
