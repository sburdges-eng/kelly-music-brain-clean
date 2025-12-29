from setuptools import setup, find_packages

setup(
    name="rr-cli",
    version="0.1.0",
    description="Git + OpenAI integrated CLI tool",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "openai>=1.0.0",
        "gitpython>=3.1.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "rr=rr.cli:main",
        ],
    },
    python_requires=">=3.8",
)
