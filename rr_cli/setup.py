from setuptools import setup, find_packages

setup(
    name="rr-cli",
    version="1.0.0",
    description="Refactor-Review CLI with Multi-Version Management and Dual-Repo Sync",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "anthropic>=0.7.0",
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
