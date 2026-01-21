from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README if it exists
readme_path = Path("README.md")
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Read requirements
requirements = []
req_path = Path("requirements.txt")
if req_path.exists():
    with open(req_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="business-finance-scorecard",
    version="2.0.0",
    author="Savvy Loan Products Ltd",
    author_email="info@example.com",
    description="Advanced Business Finance Analysis & Risk Assessment Platform for MCA Underwriting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/massimocristi1970/MCAV2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "isort>=5.0.0",
        ],
        "prod": [
            "gunicorn>=21.0.0",
            "redis>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "business-finance-app=run_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": [
            "models/model_artifacts/*.pkl",
            "config/*.json",
        ],
    },
)
