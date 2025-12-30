# Multi-Version Management System

## Overview

The RR CLI now includes a comprehensive multi-version management system that:

- **Manages multiple build versions** (Release, Debug, Development, Staging)
- **Color-codes files** for quick visual identification
- **Uses AI to compare versions** and determine which is superior
- **Automatically merges** superior versions
- **Synchronizes two repositories** atomically (kelly-project ↔ MidiKompanion)

## Features

### 1. Color-Coded Identification

```
BUILD TYPES:
  🟢 RELEASE     - Production version (Green)
  🔵 DEBUG       - Debug version (Blue)
  🟡 DEVELOPMENT - Development version (Yellow)
  🟣 STAGING     - Staging version (Magenta)

COMPARISON STATUS:
  🟢 SUPERIOR    - Clearly better (Green)
  🔵 UNKNOWN     - Not determined (Blue)
  🟡 AMBIGUOUS   - Hard to compare (Yellow)
  🔴 INFERIOR    - Clearly worse (Red)
  🔵 EQUAL       - Equivalent (Cyan)
```

### 2. AI-Powered Analysis

The system uses Claude AI to:
- Analyze code quality
- Compare implementations
- Suggest improvements
- Select optimal versions
- Generate recommendations

### 3. Dual-Repository Synchronization

Automatically sync changes between:
- `kelly-project` (primary)
- `MidiKompanion` (secondary)

## Installation

```bash
# Install rr_cli in development mode
cd /Volumes/Extreme\ SSD/kelly-project/rr_cli
pip install -e .

# Verify installation
rr --version
rr --help
```

## Quick Start

### 1. Register File Versions

```bash
# Register kelly-project version
rr version register path/to/file.py \
  --build-type release \
  --repo /Volumes/Extreme\ SSD/kelly-project

# Register MidiKompanion version
rr version register path/to/file.py \
  --build-type debug \
  --repo /Volumes/Extreme\ SSD/kelly-project/miDiKompanion
```

### 2. Compare Two Versions

```bash
rr version compare path/to/file.py \
  --build-type-a release \
  --build-type-b debug \
  --repo /Volumes/Extreme\ SSD/kelly-project
```

Output:
```
Comparing file.py for release vs debug...

Comparison Result:
  Status: SUPERIOR
  Recommendation: Use release version - clearly superior
  Merge Action: prefer_a

Analysis:
[AI-generated analysis of both versions]
```

### 3. View All Registered Versions

```bash
rr version analyze --repo /Volumes/Extreme\ SSD/kelly-project
```

Output:
```
rr_cli/rr/git_handler.py:
  RELEASE: 5f3a2c1e (2.6 KB bytes)
  DEBUG: 7a9f4b2d (2.8 KB bytes)
```

### 4. Generate Comparison Report

```bash
rr version report --repo /Volumes/Extreme\ SSD/kelly-project
```

## Advanced Usage: Dual-Repository Sync

### 1. Analyze All Mapped Files

```bash
rr sync analyze-all \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion
```

### 2. Use AI to Select Best Versions

```bash
rr sync merge-all \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion \
  --use-ai
```

This will:
1. Analyze both versions
2. Use AI to determine which is superior
3. Merge superior version to both repos
4. Show merge statistics

### 3. Commit to Both Repos

```bash
rr sync commit-all \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion \
  --kelly-msg "Sync: Update git_handler from MidiKompanion" \
  --midi-msg "Sync: Update git_handler from kelly-project"
```

### 4. View Sync Status

```bash
rr sync status \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion
```

Output:
```
SYNC STATUS:
  Total files mapped: 3
  Successfully synced: 3
  Failed: 0
  From kelly-project: 2
  From MidiKompanion: 1
```

### 5. Generate Detailed Report

```bash
rr sync report \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion \
  --output sync_report.txt
```

## File Mapping Configuration

Default mappings between repos:

```python
SyncMapping(
    kelly_path="rr_cli/rr/git_handler.py",
    midikompanion_path="mcp_web_parser/git_handler.py",
    shared_identifier="GitHandler",
    priority=5
)

SyncMapping(
    kelly_path="rr_cli/rr/ai_handler.py",
    midikompanion_path="mcp_web_parser/ai_handler.py",
    shared_identifier="AIHandler",
    priority=5
)

SyncMapping(
    kelly_path="rr_cli/rr/version_manager.py",
    midikompanion_path="version_management/version_manager.py",
    shared_identifier="VersionManager",
    priority=4
)
```

To add custom mappings, modify:
- File: `/Volumes/Extreme SSD/kelly-project/rr_cli/rr/dual_repo_sync.py`
- Section: `DEFAULT_MAPPINGS` in `DualRepoSync` class

## Complete Workflow Example

### Scenario: Sync git_handler.py between repos

```bash
# Step 1: Register versions
rr version register rr_cli/rr/git_handler.py \
  --build-type release \
  --repo /Volumes/Extreme\ SSD/kelly-project

# Step 2: Compare with MidiKompanion version
rr version compare rr_cli/rr/git_handler.py \
  --build-type-a release \
  --build-type-b debug \
  --repo /Volumes/Extreme\ SSD/kelly-project

# Step 3: Use AI to select best version across both repos
rr sync merge-all \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion \
  --use-ai

# Step 4: Review changes
git -C /Volumes/Extreme\ SSD/kelly-project diff
git -C /Volumes/Extreme\ SSD/kelly-project/miDiKompanion diff

# Step 5: Commit to both repos
rr sync commit-all \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion \
  --kelly-msg "Sync: Update git_handler from AI analysis" \
  --midi-msg "Sync: Update git_handler from AI analysis"

# Step 6: Generate report
rr sync report \
  --kelly /Volumes/Extreme\ SSD/kelly-project \
  --midikompanion /Volumes/Extreme\ SSD/kelly-project/miDiKompanion
```

## Architecture

### Components

```
RR CLI
├── git_handler.py          # Git operations (stage, commit, diff)
├── ai_handler.py           # Claude API integration
├── version_manager.py       # Version registration & comparison
│   ├── FileVersion         # Individual file version metadata
│   ├── VersionComparison   # Comparison results
│   ├── ColorCoder          # ANSI color formatting
│   └── VersionStatus       # Enum: SUPERIOR/INFERIOR/EQUAL/etc
├── dual_repo_sync.py       # Multi-repo synchronization
│   ├── SyncMapping         # File path mapping between repos
│   ├── SyncResult          # Individual sync operation result
│   └── DualRepoSync        # Main sync orchestrator
└── cli.py                  # Click CLI interface
    ├── version             # Version management commands
    └── sync                # Repository sync commands
```

### Data Flow

```
User Input
    ↓
CLI Command
    ↓
VersionManager / DualRepoSync
    ↓
GitHandler (stage/commit)
AIHandler (analysis)
    ↓
Multiple Repositories
```

## AI Analysis Criteria

When comparing versions, the AI considers:

1. **Code Quality**
   - Readability and clarity
   - Variable naming
   - Comment quality

2. **Feature Completeness**
   - All required functionality present
   - Comprehensive error handling
   - Complete documentation

3. **Performance**
   - Algorithmic efficiency
   - Memory usage
   - Execution speed

4. **Maintainability**
   - Code organization
   - Modularity
   - DRY principle compliance

5. **Best Practices**
   - Framework adherence
   - Design patterns
   - Security considerations

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY environment variable not set"

Solution:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Issue: File not found errors

Solution:
```bash
# Verify file exists
ls -la /Volumes/Extreme\ SSD/kelly-project/rr_cli/rr/git_handler.py

# Use absolute paths
rr version register /Volumes/Extreme\ SSD/kelly-project/rr_cli/rr/git_handler.py \
  --build-type release \
  --repo /Volumes/Extreme\ SSD/kelly-project
```

### Issue: Git commit fails

Solution:
```bash
# Check git status
git -C /Volumes/Extreme\ SSD/kelly-project status

# Manually commit and try again
git -C /Volumes/Extreme\ SSD/kelly-project add .
git -C /Volumes/Extreme\ SSD/kelly-project commit -m "Manual commit"
```

## Best Practices

1. **Always analyze before merging**
   - Run `rr version compare` to understand differences
   - Review AI analysis carefully

2. **Use AI selection for complex decisions**
   - Use `--use-ai` flag when versions are significantly different
   - Manual review recommended for critical files

3. **Commit atomically**
   - Always commit to both repos in same operation
   - Use descriptive commit messages

4. **Generate reports**
   - Create sync reports for audit trail
   - Save reports with timestamps

5. **Verify before push**
   - Review diffs before committing
   - Check git logs after sync

## Commands Reference

### Version Management (`rr version`)

```bash
rr version register FILE --build-type TYPE [--repo PATH]
rr version compare FILE --build-type-a TYPE_A --build-type-b TYPE_B [--repo PATH]
rr version analyze [--repo PATH]
rr version report [--repo PATH]
rr version push-multi --repo-targets REPOS... --message MSG [--repo PATH]
```

### Repository Sync (`rr sync`)

```bash
rr sync analyze-all --kelly PATH --midikompanion PATH
rr sync merge-all --kelly PATH --midikompanion PATH [--use-ai]
rr sync commit-all --kelly PATH --midikompanion PATH [--kelly-msg MSG] [--midi-msg MSG]
rr sync status --kelly PATH --midikompanion PATH
rr sync report --kelly PATH --midikompanion PATH [--output FILE]
```

### Git Operations (`rr git`)

```bash
rr git status [--repo PATH]
rr git log [--count N] [--repo PATH]
rr git diff [--staged] [--repo PATH]
```

### AI Operations (`rr ai`)

```bash
rr ai commit-msg [--style STYLE] [--staged] [--repo PATH]
rr ai analyze FILE [--analysis-type TYPE]
rr ai suggest FILE
rr ai explain TOPIC
rr ai ask QUESTION
```

## Files Modified

- `/Volumes/Extreme SSD/kelly-project/rr_cli/rr/version_manager.py` (NEW)
- `/Volumes/Extreme SSD/kelly-project/rr_cli/rr/dual_repo_sync.py` (NEW)
- `/Volumes/Extreme SSD/kelly-project/rr_cli/rr/cli.py` (UPDATED)

## Version Information

- **System Version**: 1.0.0
- **Build Types Supported**: 4 (Release, Debug, Development, Staging)
- **Comparison Status**: 5 (Unknown, Inferior, Equal, Superior, Ambiguous)
- **Max File Size for Analysis**: Configurable
- **Default Sync Mappings**: 3

## Support

For issues or questions:
1. Check troubleshooting section
2. Verify all paths are absolute
3. Ensure ANTHROPIC_API_KEY is set
4. Review generated reports for details
5. Check git logs for commit history

---

**Last Updated**: 2025-12-29
**Created by**: Claude Code with RR CLI Integration
