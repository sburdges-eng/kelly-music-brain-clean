"""Main CLI interface for RR"""

import click
import os
from .git_handler import GitHandler
from .ai_handler import AIHandler
from .version_manager import VersionManager, BuildType, VersionStatus, ColorCoder
from .dual_repo_sync import DualRepoSync


@click.group()
@click.version_option(version="1.0.0", prog_name="rr")
def main():
    """RR - Refactor-Review CLI with Multi-Version Support"""
    pass


@main.group()
def git():
    """Git-related commands"""
    pass


@main.group()
def ai():
    """AI-related commands"""
    pass


@main.group()
def version():
    """Version management commands"""
    pass


@main.group()
def sync():
    """Dual-repository synchronization commands"""
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
        answer = ai_handler.ask_question(question)
        click.echo(answer)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


# Version Management Commands
@version.command()
@click.argument("file_path")
@click.option("--build-type", required=True, type=click.Choice(['release', 'debug', 'development', 'staging']), help="Build type")
@click.option("--repo", default=".", help="Repository path")
def register(file_path, build_type, repo):
    """Register a file version"""
    try:
        manager = VersionManager(repo)
        build_enum = BuildType(build_type)
        version = manager.register_version(file_path, build_enum)

        click.echo(f"Registered version:")
        click.echo(f"  File: {file_path}")
        click.echo(f"  Build: {ColorCoder.colorize_build(build_enum)}")
        click.echo(f"  Hash: {version.content_hash}")
        click.echo(f"  Size: {version.size} bytes")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@version.command()
@click.argument("file_path")
@click.option("--build-type-a", required=True, type=click.Choice(['release', 'debug', 'development', 'staging']), help="First build type")
@click.option("--build-type-b", required=True, type=click.Choice(['release', 'debug', 'development', 'staging']), help="Second build type")
@click.option("--repo", default=".", help="Repository path")
def compare(file_path, build_type_a, build_type_b, repo):
    """Compare two file versions"""
    try:
        manager = VersionManager(repo)
        build_a = BuildType(build_type_a)
        build_b = BuildType(build_type_b)

        click.echo(f"Analyzing {file_path} for {build_type_a} vs {build_type_b}...")
        comparison = manager.compare_versions(file_path, build_a, build_b)

        click.echo("\nComparison Result:")
        click.echo(f"  Status: {ColorCoder.colorize_status(comparison.status)}")
        click.echo(f"  Recommendation: {comparison.recommendation}")
        click.echo(f"  Merge Action: {comparison.merge_action}")
        click.echo(f"\nAnalysis:\n{comparison.analysis}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@version.command()
@click.option("--repo", default=".", help="Repository path")
def report(repo):
    """Generate comparison report"""
    try:
        manager = VersionManager(repo)
        report_text = manager.get_comparison_report()
        click.echo(report_text)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@version.command()
@click.option("--repo", default=".", help="Repository path")
@click.option("--repo-targets", multiple=True, required=True, help="Target repositories to push to")
@click.option("--message", required=True, help="Commit message")
def push_multi(repo, repo_targets, message):
    """Push changes to multiple repositories"""
    try:
        manager = VersionManager(repo)

        click.echo(f"Pushing to {len(repo_targets)} repositories...")
        results = manager.push_to_repos({}, list(repo_targets), message)

        for target_repo, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            click.echo(f"  {target_repo}: {status}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@version.command()
@click.option("--repo", default=".", help="Repository path")
def analyze(repo):
    """Analyze all registered versions"""
    try:
        manager = VersionManager(repo)

        if not manager.versions:
            click.echo("No versions registered yet.")
            return

        for file_path, versions in manager.versions.items():
            click.echo(f"\n{ColorCoder.colorize(file_path, 'bold')}:")
            for v in versions:
                build_colored = ColorCoder.colorize_build(v.build_type)
                click.echo(f"  {build_colored}: {v.content_hash} ({v.size} bytes)")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


# Sync Commands
@sync.command()
@click.option("--kelly", required=True, help="Path to kelly-project repo")
@click.option("--midikompanion", required=True, help="Path to MidiKompanion repo")
def analyze_all(kelly, midikompanion):
    """Analyze all mapped files for differences"""
    try:
        syncer = DualRepoSync(kelly, midikompanion)

        click.echo("Analyzing files across repositories...")
        comparisons = syncer.analyze_all_files()

        if not comparisons:
            click.echo("No differences found or no files to compare.")
            return

        for comp in comparisons:
            click.echo(f"\n{ColorCoder.colorize(comp.file_path, 'bold')}")
            click.echo(f"  Status: {ColorCoder.colorize_status(comp.status)}")
            click.echo(f"  Recommendation: {comp.recommendation}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@sync.command()
@click.option("--kelly", required=True, help="Path to kelly-project repo")
@click.option("--midikompanion", required=True, help="Path to MidiKompanion repo")
@click.option("--use-ai", is_flag=True, help="Use AI to select best versions")
def merge_all(kelly, midikompanion, use_ai):
    """Merge files using AI selection or defaults"""
    try:
        syncer = DualRepoSync(kelly, midikompanion)

        if use_ai:
            click.echo("Using AI to select best versions...")
            selections = syncer.sync_with_ai_selection()

            for mapping, selected in selections.items():
                click.echo(f"\n{mapping.shared_identifier}:")
                click.echo(f"  Selected from: {ColorCoder.colorize(selected, 'green')}")

                # Perform merge
                result = syncer.merge_file(mapping, selected)
                status = "✓ Success" if result.merged_successfully else "✗ Failed"
                click.echo(f"  Status: {status}")
        else:
            click.echo("Merging with default selections...")
            for mapping in syncer.mappings:
                result = syncer.merge_file(mapping, 'kelly')
                status = "✓ Success" if result.merged_successfully else "✗ Failed"
                click.echo(f"{mapping.shared_identifier}: {status}")

        # Show statistics
        stats = syncer.get_sync_statistics()
        click.echo(f"\nTotal synced: {stats['successful']}/{stats['total_files']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@sync.command()
@click.option("--kelly", required=True, help="Path to kelly-project repo")
@click.option("--midikompanion", required=True, help="Path to MidiKompanion repo")
@click.option("--kelly-msg", default="Sync: Update from MidiKompanion", help="Kelly commit message")
@click.option("--midi-msg", default="Sync: Update from kelly-project", help="MidiKompanion commit message")
def commit_all(kelly, midikompanion, kelly_msg, midi_msg):
    """Commit changes to both repositories"""
    try:
        syncer = DualRepoSync(kelly, midikompanion)

        click.echo("Committing to both repositories...")
        kelly_ok, midi_ok = syncer.stage_and_commit_all(kelly_msg, midi_msg)

        kelly_status = "✓ Success" if kelly_ok else "✗ Failed"
        midi_status = "✓ Success" if midi_ok else "✗ Failed"

        click.echo(f"kelly-project: {kelly_status}")
        click.echo(f"MidiKompanion: {midi_status}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@sync.command()
@click.option("--kelly", required=True, help="Path to kelly-project repo")
@click.option("--midikompanion", required=True, help="Path to MidiKompanion repo")
@click.option("--output", default=".sync_report.txt", help="Output file")
def report(kelly, midikompanion, output):
    """Generate sync report"""
    try:
        syncer = DualRepoSync(kelly, midikompanion)

        report_text = syncer.create_sync_report()
        click.echo(report_text)

        # Also save to file
        with open(output, 'w') as f:
            f.write(report_text)

        click.echo(f"\nReport saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@sync.command()
@click.option("--kelly", required=True, help="Path to kelly-project repo")
@click.option("--midikompanion", required=True, help="Path to MidiKompanion repo")
def status(kelly, midikompanion):
    """Show sync status and statistics"""
    try:
        syncer = DualRepoSync(kelly, midikompanion)

        stats = syncer.get_sync_statistics()

        click.echo("SYNC STATUS:")
        click.echo(f"  Total files mapped: {stats['total_files']}")
        click.echo(f"  Successfully synced: {stats['successful']}")
        click.echo(f"  Failed: {stats['failed']}")
        click.echo(f"  From kelly-project: {stats['from_kelly']}")
        click.echo(f"  From MidiKompanion: {stats['from_midikompanion']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    main()
