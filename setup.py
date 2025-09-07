"""
Setup script for FinEasy AI Backend
"""
from setuptools import setup, find_packages

setup(
    name="fineasy-ai-backend",
    version="0.1.0",
    description="AI-powered business intelligence backend for FinEasy",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "supabase>=2.0.2",
        "redis>=5.0.1",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "pandas>=2.1.4",
        "numpy>=1.24.4",
        "scikit-learn>=1.3.2",
        "scipy>=1.11.4",
        "spacy>=3.7.2",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.23.0",
        "httpx>=0.25.2",
        "requests>=2.31.0",
        "python-dateutil>=2.8.2",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "ml": [
            "statsmodels>=0.14.0",
            "prophet>=1.1.5",
            "transformers>=4.36.2",
            "torch>=2.1.2",
        ],
        "monitoring": [
            "structlog>=23.2.0",
            "prometheus-client>=0.19.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
)