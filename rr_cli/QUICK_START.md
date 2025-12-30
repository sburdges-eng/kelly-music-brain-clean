# RR CLI - Quick Start Guide

Multi-Version Management for kelly-project & MidiKompanion

## Install

```bash
cd rr_cli
pip install -e .
```

## Color Guide

| Color | Type | Meaning |
|-------|------|---------|
| 🟢 Green | BUILD | Release version |
| 🔵 Blue | BUILD | Debug version |
| 🟡 Yellow | BUILD | Development version |
| 🟣 Magenta | BUILD | Staging version |
| 🟢 Green | STATUS | Superior version |
| 🔵 Cyan | STATUS | Equal versions |
| 🔴 Red | STATUS | Inferior version |
| 🟡 Yellow | STATUS | Ambiguous comparison |

## One-Line Commands

### Compare two versions
```bash
rr version compare rr/git_handler.py --build-type-a release --build-type-b debug
```

### Sync with AI selection (RECOMMENDED)
```bash
rr sync merge-all --kelly . --midikompanion ../miDiKompanion --use-ai
```

### Commit to both repos
```bash
rr sync commit-all --kelly . --midikompanion ../miDiKompanion
```

### View sync status
```bash
rr sync status --kelly . --midikompanion ../miDiKompanion
```

## Full Workflow

```bash
# 1. Register versions
rr version register rr/git_handler.py --build-type release

# 2. Analyze and compare
rr version compare rr/git_handler.py \
  --build-type-a release \
  --build-type-b debug

# 3. Merge using AI
rr sync merge-all \
  --kelly /path/to/kelly-project \
  --midikompanion /path/to/miDiKompanion \
  --use-ai

# 4. Commit to both repos
rr sync commit-all \
  --kelly /path/to/kelly-project \
  --midikompanion /path/to/miDiKompanion \
  --kelly-msg "Update git_handler from MidiKompanion" \
  --midi-msg "Update git_handler from kelly-project"

# 5. Review report
rr sync report \
  --kelly /path/to/kelly-project \
  --midikompanion /path/to/miDiKompanion
```

## Tips

- Use **absolute paths** for reliability
- Use **--use-ai flag** for automatic version selection
- Always **review diffs** before committing
- **Generate reports** for audit trail
- Set **ANTHROPIC_API_KEY** environment variable

## Troubleshooting

```bash
# Show help
rr --help
rr version --help
rr sync --help

# Check version
rr --version

# Test git access
rr git status --repo .

# Test API
rr ai ask "What is Python?"
```

## File Paths (Common)

```
kelly-project/
└── rr_cli/
    └── rr/
        ├── git_handler.py      (synced)
        ├── ai_handler.py       (synced)
        └── version_manager.py  (synced)

miDiKompanion/
└── mcp_web_parser/
    ├── git_handler.py
    ├── ai_handler.py
    └── ...

midikompanion/
└── version_management/
    └── version_manager.py
```

## What Gets Synced

By default, these files are tracked:

1. **GitHandler** - `rr_cli/rr/git_handler.py` ↔ `mcp_web_parser/git_handler.py`
2. **AIHandler** - `rr_cli/rr/ai_handler.py` ↔ `mcp_web_parser/ai_handler.py`
3. **VersionManager** - `rr_cli/rr/version_manager.py` ↔ `version_management/version_manager.py`

To add more files, edit:
- `rr_cli/rr/dual_repo_sync.py` → `DEFAULT_MAPPINGS`

---

**Need more details?** See `MULTI_VERSION_GUIDE.md`
