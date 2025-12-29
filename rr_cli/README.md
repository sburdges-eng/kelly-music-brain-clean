# RR - Git + OpenAI CLI Tool

A powerful command-line tool that integrates Git and OpenAI to streamline your development workflow.

## Features

- **Git Integration**: Check status, view logs, and manage commits
- **AI-Powered Commits**: Automatically generate commit messages using AI
- **Code Analysis**: Analyze code with AI suggestions
- **Learning Tool**: Ask questions and get explanations on programming concepts
- **Improvement Suggestions**: Get AI-powered suggestions to improve your code

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)
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

3. Configure your OpenAI API key:
```bash
# Create a .env file from the example
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

Alternatively, set the environment variable:
```bash
export OPENAI_API_KEY=your_api_key_here
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

## Options

### Global Options
- `--help` - Show help message
- `--version` - Show version

### Repository Options
- `--repo` - Specify repository path (default: current directory)

## Troubleshooting

### "OPENAI_API_KEY environment variable not set"
Make sure your API key is configured. Either:
1. Create a `.env` file with your API key
2. Set the `OPENAI_API_KEY` environment variable
3. Pass it as a parameter (check `--help`)

### "Invalid git repository"
Make sure you're running the command in a git repository or specify the repository path:
```bash
rr git status --repo /path/to/repo
```

### API Rate Limits
If you exceed OpenAI's rate limits, wait a moment and try again.

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - Model to use (optional, default: gpt-4)

## License

MIT License - feel free to use and modify!

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
