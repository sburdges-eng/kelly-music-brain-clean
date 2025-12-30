# RR - Refactor-Review CLI Tool

A powerful command-line tool that integrates Git, Claude AI, and multi-version management for streamlined development workflow.

## Features

- **Git Integration**: Check status, view logs, and manage commits
- **AI-Powered Commits**: Automatically generate commit messages using Claude AI
- **Code Analysis**: Analyze code with AI suggestions
- **Multi-Version Management**: Register, compare, and manage multiple build versions
- **Color-Coded Identification**: Visual identification of build types and comparison status
- **Dual-Repo Synchronization**: Automatically sync versions between kelly-project and MidiKompanion
- **Learning Tool**: Ask questions and get explanations on programming concepts
- **Improvement Suggestions**: Get AI-powered suggestions to improve your code

## Installation

### Prerequisites

- Python 3.8 or higher
- Anthropic Claude API key (get one at https://console.anthropic.com/)
- Git

### Setup

1. Clone or download the RR CLI tool:
```bash
cd rr_cli
```

2. Install dependencies:
```bash
pip install -e .
```

3. Configure your Anthropic API key:
```bash
# Set the environment variable
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

### Git Commands

#### Check repository status
```bash
rr git status
```

#### View recent commits
```bash
rr git log --count 10
```

#### View changes
```bash
rr git diff
rr git diff --staged
```

### AI Commands

#### Generate commit message
```bash
rr ai commit-msg
rr ai commit-msg --style conventional
rr ai commit-msg --staged
```

#### Analyze code
```bash
rr ai analyze path/to/file.py
rr ai analyze path/to/file.py --analysis-type security
```

#### Get improvement suggestions
```bash
rr ai suggest path/to/file.py
```

#### Explain a concept
```bash
rr ai explain "what is a closure in Python"
```

#### Ask a question
```bash
rr ai ask "how do I handle exceptions in Python?"
```

## Examples

### Generate and commit with AI
```bash
# Make some changes to your files
git add .

# Generate a smart commit message
rr ai commit-msg --staged

# If you like it, confirm to commit automatically
```

### Improve your code
```bash
# Get AI suggestions for a file
rr ai suggest src/main.py

# Analyze code for issues
rr ai analyze src/main.py --analysis-type security
```

### Learn while coding
```bash
rr ai explain "what is a decorator in Python"
rr ai ask "best practices for error handling"
```

## Multi-Version Management (NEW!)

Manage multiple versions of files across different builds with AI-powered comparison.

```bash
# Register a file version
rr version register path/to/file.py --build-type release

# Compare two versions
rr version compare path/to/file.py --build-type-a release --build-type-b debug

# View all registered versions
rr version analyze

# Generate comparison report
rr version report
```

## Dual-Repository Synchronization (NEW!)

Automatically sync files between kelly-project and MidiKompanion repositories.

```bash
# Analyze differences
rr sync analyze-all --kelly /path/kelly --midikompanion /path/midi

# Merge using AI selection
rr sync merge-all --kelly /path/kelly --midikompanion /path/midi --use-ai

# Commit to both repos
rr sync commit-all --kelly /path/kelly --midikompanion /path/midi

# View sync status
rr sync status --kelly /path/kelly --midikompanion /path/midi

# Generate report
rr sync report --kelly /path/kelly --midikompanion /path/midi
```

**For detailed instructions, see:** `QUICK_START.md` and `MULTI_VERSION_GUIDE.md`

## Command Reference

### Git Group
- `status` - Show repository status
- `log` - Show recent commits
- `diff` - Show git diff

### AI Group
- `commit-msg` - Generate commit message from changes
- `analyze` - Analyze code with AI
- `suggest` - Get improvement suggestions
- `explain` - Explain a concept
- `ask` - Ask a question

### Version Group (NEW!)
- `register` - Register a file version
- `compare` - Compare two versions
- `analyze` - View registered versions
- `report` - Generate comparison report
- `push-multi` - Push to multiple repos

### Sync Group (NEW!)
- `analyze-all` - Analyze files across repos
- `merge-all` - Merge with AI selection
- `commit-all` - Commit to both repos
- `status` - Show sync status
- `report` - Generate sync report

## Options

### Global Options
- `--help` - Show help message
- `--version` - Show version

### Repository Options
- `--repo` - Specify repository path (default: current directory)

## Troubleshooting

### "ANTHROPIC_API_KEY environment variable not set"
Make sure your API key is configured. Either:
1. Set the environment variable: `export ANTHROPIC_API_KEY="your-key-here"`
2. Create a `.env` file with your API key
3. Check the `MULTI_VERSION_GUIDE.md` for detailed setup

### "Invalid git repository"
Make sure you're running the command in a git repository or specify the repository path:
```bash
rr git status --repo /path/to/repo
```

### "File not found" errors
Use absolute paths for reliability:
```bash
rr version register /absolute/path/to/file.py --build-type release
```

### API Rate Limits
If you exceed Anthropic's rate limits, wait a moment and try again.

### Sync Issues
Check `MULTI_VERSION_GUIDE.md` for comprehensive troubleshooting.

## Configuration

### Environment Variables
- `ANTHROPIC_API_KEY` - Your Anthropic Claude API key (required)

## License

MIT License - feel free to use and modify!

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
