# setup.py

from setuptools import setup, find_packages

setup(
    name="plagiarism_bot",
    version="1.0.0",
    author="Pousali Dolai",
    author_email="your_email@example.com",
    description="A text-only plagiarism checker using TF-IDF and semantic similarity.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plagiarism_bot",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "nltk==3.8.1",
        "scikit-learn==1.3.2",
        "sentence-transformers==2.2.2",
        "torch>=2.0.1",
        "numpy>=1.24.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "plagiarism-bot=plagiarism_bot:main",  # optional if you define a main() in plagiarism_bot.py
        ],
    },
)
