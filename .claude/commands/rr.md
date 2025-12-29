# RR CLI Integration with Claude Code

Seamlessly use the RR CLI tool (Git + OpenAI) directly within Claude Code sessions.

## Setup

1. **Install RR CLI Tool**:
   ```bash
   cd /Volumes/Extreme\ SSD/kelly-project/rr_cli
   pip install -e .
   ```

2. **Configure OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=sk-...
   ```

3. **Verify Installation**:
   ```bash
   rr --help
   ```

## Quick Commands

### Git Operations
```bash
# Check git status
rr git status

# View recent commits
rr git log --count 5

# Show changes
rr git diff
rr git diff --staged
```

### AI-Powered Features
```bash
# Generate smart commit message from staged changes
rr ai commit-msg --staged

# Analyze code with AI
rr ai analyze src/file.py

# Get improvement suggestions
rr ai suggest src/file.py

# Learn concepts
rr ai explain "multi-task learning"

# Ask questions
rr ai ask "How do I implement a custom encoder?"
```

## Usage in Claude Code

### Pattern 1: Generate Commits
When you've made changes and want an AI-generated commit message:
```bash
rr ai commit-msg --staged
```

### Pattern 2: Code Analysis
To analyze specific files for improvements:
```bash
rr ai analyze multi_task_framework/base.py --analysis-type architecture
```

### Pattern 3: Learn While Coding
Ask questions about concepts as you work:
```bash
rr ai explain "What is encoder generalization"
```

### Pattern 4: Git Management
Quick git operations:
```bash
rr git status --repo /Volumes/Extreme\ SSD/kelly-project
```

## Examples

### Auto-commit workflow
```bash
# Make changes
git add .

# Generate and commit
rr ai commit-msg --staged
# Follow the prompt to confirm commit
```

### Code improvement workflow
```bash
# Get suggestions for multi-task framework
rr ai suggest multi_task_framework/heads.py

# Analyze for best practices
rr ai analyze multi_task_framework/factory.py --analysis-type security
```

### Learning workflow
```bash
# While implementing features
rr ai explain "backwards compatibility patterns"
rr ai ask "How do I design extensible APIs"
```

## Available RR Commands

### Git Group
- `rr git status` - Repository status
- `rr git log` - Recent commits
- `rr git diff` - Show changes

### AI Group
- `rr ai commit-msg` - Generate commit message
- `rr ai analyze` - Code analysis
- `rr ai suggest` - Improvement suggestions
- `rr ai explain` - Learn concepts
- `rr ai ask` - Ask questions

## Options

### Global
- `--help` - Show help
- `--version` - Show version

### Repository
- `--repo PATH` - Specify repository path (default: current)

### AI Commits
- `--styled conventional` - Commit style (default: conventional)
- `--staged` - Use staged changes only

### Code Analysis
- `--analysis-type security/performance/architecture/general`

## Tips

1. **Set Default Repo**: Set your working directory to the project root
2. **Use Staged Changes**: Stage only relevant files before generating commits
3. **Commit Style**: Use `--style conventional` for conventional commits
4. **API Key**: Keep your API key in environment, not in code
5. **Batch Analysis**: Analyze multiple files one at a time for detailed feedback

## Troubleshooting

### "Command not found: rr"
- Make sure installation completed: `pip install -e /path/to/rr_cli`
- Check PATH includes pip packages

### "OPENAI_API_KEY not set"
- Set environment variable: `export OPENAI_API_KEY=sk-...`
- Or create `.env` file in project root

### Git errors
- Ensure you're in a valid git repository
- Use `--repo` flag to specify repository path

## Next Steps

1. Install and configure RR CLI tool
2. Try your first git status command
3. Generate your first AI commit message
4. Use analysis tools on your code
5. Learn concepts with the explain command

Happy coding with AI assistance!
