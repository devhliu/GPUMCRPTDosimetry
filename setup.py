#!/usr/bin/env python3
"""
Setup script for GPUMCRPTDosimetry package.

GPU-accelerated Monte Carlo radionuclide particle transport dosimetry.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from pyproject.toml
import re
with open("pyproject.toml", "r", encoding="utf-8") as f:
    content = f.read()
    version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    version = version_match.group(1) if version_match else "0.1.0"

setup(
    name="gpumcrpt",
    version=version,
    description="GPU-accelerated Monte Carlo radionuclide particle transport dosimetry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GPUMCRPTDosimetry contributors",
    author_email="",  # Add email if available
    url="",  # Add repository URL if available
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.10",
    install_requires=[
        "nibabel>=5.2",
        "numpy>=1.26",
        "torch>=2.2",
        "triton==3.5.1",
        "h5py>=3.10",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpumcrpt-run=gpumcrpt.scripts.run_dosimetry_nifti:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gpumcrpt": [
            "decaydb/data/*.json",  # Include ICRP-107 data if present
        ],
    },
    keywords=[
        "monte-carlo", "dosimetry", "radionuclide", "gpu", "medical-physics",
        "radiation-therapy", "nuclear-medicine", "python"
    ],
    project_urls={
        "Documentation": "",  # Add documentation URL if available
        "Source": "",  # Add repository URL if available
        "Bug Reports": "",  # Add issue tracker URL if available
    },
)
