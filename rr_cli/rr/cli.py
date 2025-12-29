"""Main CLI interface for RR"""

import click
import os
from .git_handler import GitHandler
from .ai_handler import AIHandler


@click.group()
@click.version_option()
def main():
    """RR - Git + OpenAI integrated CLI tool"""
    pass


@main.group()
def git():
    """Git-related commands"""
    pass


@main.group()
def ai():
    """AI-related commands"""
    pass


# Git Commands
@git.command()
@click.option("--repo", default=".", help="Repository path")
def status(repo):
    """Show git status"""
    try:
        handler = GitHandler(repo)
        status_info = handler.get_status()
        click.echo(f"Branch: {status_info['branch']}")
        click.echo(f"Dirty: {status_info['dirty']}")
        if status_info['untracked']:
            click.echo(f"Untracked: {', '.join(status_info['untracked'])}")
        if status_info['staged']:
            click.echo(f"Staged: {', '.join(status_info['staged'])}")
        if status_info['unstaged']:
            click.echo(f"Unstaged: {', '.join(status_info['unstaged'])}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@git.command()
@click.option("--repo", default=".", help="Repository path")
@click.option("--count", default=5, help="Number of commits to show")
def log(repo, count):
    """Show recent commits"""
    try:
        handler = GitHandler(repo)
        commits = handler.get_recent_commits(count)
        for commit in commits:
            click.echo(f"{commit['hash']} - {commit['author']}: {commit['message']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@git.command()
@click.option("--repo", default=".", help="Repository path")
@click.option("--staged", is_flag=True, help="Show staged changes only")
def diff(repo, staged):
    """Show git diff"""
    try:
        handler = GitHandler(repo)
        diff_output = handler.get_diff(staged)
        click.echo(diff_output if diff_output else "No changes")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


# AI Commands
@ai.command()
@click.option("--repo", default=".", help="Repository path")
@click.option("--style", default="conventional", help="Commit message style")
@click.option("--staged", is_flag=True, help="Use staged changes only")
def commit_msg(repo, style, staged):
    """Generate AI commit message from changes"""
    try:
        git_handler = GitHandler(repo)
        ai_handler = AIHandler()

        diff_output = git_handler.get_diff(staged)
        if not diff_output.strip():
            click.echo("No changes to commit", err=True)
            return

        message = ai_handler.generate_commit_message(diff_output, style)
        click.echo(f"Generated commit message:\n{message}")

        if click.confirm("Commit with this message?"):
            if git_handler.commit(message):
                click.echo("Committed successfully!")
            else:
                click.echo("Failed to commit", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@ai.command()
@click.argument("file_path")
@click.option("--analysis-type", default="general", help="Type of analysis")
def analyze(file_path, analysis_type):
    """Analyze code with AI"""
    try:
        if not os.path.exists(file_path):
            click.echo(f"File not found: {file_path}", err=True)
            return

        with open(file_path, "r") as f:
            code = f.read()

        ai_handler = AIHandler()
        analysis = ai_handler.analyze_code(code, analysis_type)
        click.echo(analysis)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@ai.command()
@click.argument("file_path")
def suggest(file_path):
    """Suggest improvements for a file"""
    try:
        if not os.path.exists(file_path):
            click.echo(f"File not found: {file_path}", err=True)
            return

        with open(file_path, "r") as f:
            content = f.read()

        ai_handler = AIHandler()
        suggestions = ai_handler.suggest_improvements(content)
        click.echo(suggestions)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@ai.command()
@click.argument("topic")
def explain(topic):
    """Explain a programming concept"""
    try:
        ai_handler = AIHandler()
        explanation = ai_handler.explain_concept(topic)
        click.echo(explanation)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@ai.command()
@click.argument("question")
def ask(question):
    """Ask a general question"""
    try:
        ai_handler = AIHandler()
        answer = ai_handler.ask(question)
        click.echo(answer)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    main()
