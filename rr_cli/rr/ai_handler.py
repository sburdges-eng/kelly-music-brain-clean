"""OpenAI integration module for RR CLI"""

import os
from typing import Optional
from openai import OpenAI


class AIHandler:
    """Handle OpenAI interactions"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI handler"""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4"

    def generate_commit_message(self, diff: str, style: str = "conventional") -> str:
        """Generate a commit message from git diff"""
        prompt = f"""Generate a {style} commit message based on this git diff.

Diff:
{diff}

Return only the commit message, no explanation."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()

    def analyze_code(self, code: str, analysis_type: str = "general") -> str:
        """Analyze code with AI"""
        prompt = f"""Perform a {analysis_type} analysis of the following code:

{code}

Provide insights and recommendations."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def suggest_improvements(self, content: str, context: str = "code") -> str:
        """Suggest improvements for code or documentation"""
        prompt = f"""Suggest improvements for the following {context}:

{content}

Format as a numbered list."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def explain_concept(self, topic: str) -> str:
        """Explain a programming concept"""
        prompt = f"Explain the following concept in simple terms: {topic}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def ask(self, question: str) -> str:
        """Ask a general question"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
