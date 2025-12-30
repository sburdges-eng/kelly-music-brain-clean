"""Anthropic Claude API integration module for RR CLI"""

import os
from typing import Optional
from anthropic import Anthropic


class AIHandler:
    """Handle Anthropic Claude interactions"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic Claude handler"""
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"

    def generate_commit_message(self, diff: str, style: str = "conventional") -> str:
        """Generate a commit message from git diff"""
        prompt = f"""Generate a {style} commit message based on this git diff.

Diff:
{diff}

Return only the commit message, no explanation."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def analyze_code(self, code: str, analysis_type: str = "general") -> str:
        """Analyze code with AI"""
        prompt = f"""Perform a {analysis_type} analysis of the following code:

{code}

Provide insights and recommendations."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def suggest_improvements(self, code: str) -> str:
        """Suggest code improvements"""
        prompt = f"""Review the following code and suggest improvements:

{code}

Provide specific, actionable suggestions."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def explain_concept(self, topic: str) -> str:
        """Explain a programming concept"""
        prompt = f"""Explain the following programming concept clearly and concisely:

{topic}

Include practical examples and use cases."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def teach_concept(self, topic: str) -> str:
        """Provide a detailed, microscopic-level explanation of a programming concept"""
        prompt = f"""You are an expert programming educator. Provide a DETAILED, MICROSCOPIC-LEVEL explanation of:

{topic}

Structure your response with these 9 sections:

## 📚 CONCEPT OVERVIEW
A concise 1-2 line summary of the concept.

## 🔍 MICROSCOPIC BREAKDOWN
The concept decomposed into its smallest atomic components:
- Individual pieces explained separately
- How pieces interconnect
- Mathematical notation (if relevant)
- Fundamental principles

## 🧬 MECHANISM DETAILS
Deep dive into HOW it works:
- Complete step-by-step process
- Memory and computation flow
- Data transformations at each stage
- Why each step matters
- What happens if you skip steps
- Performance characteristics

## 💡 CONCRETE CODE EXAMPLE
Working Python code showing:
- Minimal but complete implementation
- Detailed comments on EVERY line
- Variable naming that reveals meaning
- Input → Process → Output flow
- Common variations

## 🏗️ ARCHITECTURE
How the concept fits into larger systems:
- Where it's used in practice
- Dependencies and dependents
- Typical design patterns
- When to use vs. alternatives
- Integration points

## ⚠️ CRITICAL DETAILS
The tricky parts explained:
- Edge cases and handling
- Common mistakes and why they happen
- Performance implications
- Numerical stability issues
- Thread safety (if applicable)
- Off-by-one errors
- Boundary conditions

## 🎯 PRACTICAL APPLICATIONS
Real-world examples:
- Industry applications
- Production system examples
- Problems it solves
- When NOT to use it
- Alternative approaches
- Why alternatives differ

## 🧠 MENTAL MODEL
Help visualize and remember:
- Physical world analogies
- ASCII diagrams
- Mnemonic devices
- Memory aids
- Relationship to other concepts
- Visualization techniques

## 📖 FURTHER LEARNING
Deeper understanding:
- Related concepts to study
- Research papers and resources
- Variations and extensions
- Advanced techniques
- Where experts disagree
- Cutting-edge developments

Provide a comprehensive, detailed response that helps someone deeply understand this concept."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def ask_question(self, question: str) -> str:
        """Answer a programming question"""
        prompt = f"""Answer the following programming question clearly and thoroughly:

{question}

Provide practical examples and relevant context."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
