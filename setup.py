"""
ClimateVision: Open-source ML platform for deforestation detection
"""
from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Open-source ML platform for automated deforestation detection"

setup(
    name="climatevision",
    version="0.1.0",
    author="ClimateVision Contributors",
    description="Open-source ML platform for automated deforestation detection using deep learning and satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ClimateVision",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "rasterio>=1.3.0",
        "geopandas>=0.12.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "requests>=2.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn[standard]>=0.20.0",
            "pydantic>=2.0.0",
            "python-multipart>=0.0.5",
        ],
        "processing": [
            "dask[complete]>=2023.1.0",
        ],
        "satellite": [
            "sentinelsat>=1.1.0",
            "earthengine-api>=0.1.340",
        ],
        "mlops": [
            "mlflow>=2.1.0",
            "optuna>=3.1.0",
        ],
    },
)
