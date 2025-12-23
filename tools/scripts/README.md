# Migration Tools

This directory contains automated tools for migrating DAiW-Music-Brain into Kelly Music Brain 2.0 monorepo.

## Tools Overview

### 1. `deduplicate.py` - Duplicate File Detection

Find and analyze duplicate files across the repository.

**Features:**
- Content-based hashing (MD5/SHA256)
- Fuzzy filename matching
- Semantic comparison for JSON/YAML files
- Identifies `*_music-brain` and `*_penta-core` variants
- Generates deletion scripts

**Usage:**
```bash
# Scan current directory
python deduplicate.py --scan .

# Generate markdown report
python deduplicate.py --scan . --output report.md --format markdown

# Generate JSON report
python deduplicate.py --scan . --output report.json --format json

# Generate CSV for spreadsheet analysis
python deduplicate.py --scan . --output report.csv --format csv
```

**Output:**
- Duplicate groups categorized by type (exact match, name pattern, semantic)
- Recommended actions (keep/delete)
- Automated deletion script

---

### 2. `create_monorepo.py` - Monorepo Scaffold Generator

Create complete directory structure with all configuration files.

**Features:**
- Generates `packages/` structure for all components
- Creates package-level config files (pyproject.toml, CMakeLists.txt, package.json)
- Generates root workspace configuration
- Creates .gitignore patterns
- Sets up pre-commit hooks

**Usage:**
```bash
# Generate in current directory
python create_monorepo.py --output .

# Dry run to preview structure
python create_monorepo.py --output /path/to/new/repo --dry-run
```

**Generated Structure:**
```
kelly-music-brain/
├── packages/
│   ├── core/{python,cpp}
│   ├── cli/
│   ├── desktop/
│   ├── plugins/{vst3,clap}
│   ├── mobile/{ios,android}
│   └── web/
├── data/
├── vault/
├── tests/
├── tools/
├── docs/
└── examples/
```

---

### 3. `migrate_modules.py` - Git History Preservation

Migrate specific modules from DAiW to Kelly with full git history.

**Features:**
- Uses `git-filter-repo` for history preservation
- Automatic conflict detection
- Import path rewriting (`music_brain` → `kelly_core`)
- Documentation link updates
- Dependency-aware migration order

**Modules Available:**
- `core_emotion` - Emotional mapping engine
- `intent_schema` - Intent processing system
- `groove` - Groove extraction/application
- `teaching` - Teaching module
- `vault` - Knowledge base
- `cli` - Command-line interface
- `data` - Data files

**Usage:**
```bash
# List available modules
python migrate_modules.py --list-modules

# Migrate a specific module
python migrate_modules.py \
  --source /path/to/daiw \
  --target /path/to/kelly \
  --module core_emotion

# Migrate all modules
python migrate_modules.py \
  --source /path/to/daiw \
  --target /path/to/kelly \
  --module all \
  --report migration-report.md

# Dry run
python migrate_modules.py \
  --source /path/to/daiw \
  --target . \
  --module all \
  --dry-run
```

**Requirements:**
- `git-filter-repo` installed (`pip install git-filter-repo`)

---

### 4. `standardize_names.py` - Name Standardization

Standardize naming conventions across the repository.

**Features:**
- Renames packages: `music_brain` → `kelly_core`, `penta_core` → `kelly_core`
- Removes file suffixes: `*_music-brain`, `*_penta-core`
- Updates all import statements
- Fixes CMake target names
- Validates Python naming conventions

**Usage:**
```bash
# Scan for naming issues
python standardize_names.py --scan .

# Fix all naming issues
python standardize_names.py --scan . --fix

# Dry run to preview changes
python standardize_names.py --scan . --fix --dry-run
```

**Changes Made:**
- Python imports updated
- File names standardized
- CMake targets renamed
- Project names unified

---

### 5. `validate_migration.py` - Comprehensive Validation

Validate the repository migration for correctness.

**Features:**
- Import validation (no broken imports)
- Circular dependency detection
- Test suite execution
- CMake build validation
- Git history verification
- Performance benchmarks

**Usage:**
```bash
# Run all validations
python validate_migration.py --repo . --test all

# Run specific validation
python validate_migration.py --repo . --test imports
python validate_migration.py --repo . --test tests
python validate_migration.py --repo . --test build

# Generate report
python validate_migration.py --repo . --test all --report validation-report.md
```

**Validation Tests:**
- ✅ All imports resolve correctly
- ✅ No circular dependencies
- ✅ All tests pass
- ✅ CMake builds successfully
- ✅ Git history intact
- ✅ No data loss

---

### 6. `execute_merger.py` - Master Orchestrator

Execute the complete merger process end-to-end.

**Features:**
- Runs all migration steps in order
- Progress tracking and reporting
- Automatic dependency resolution
- Comprehensive final report
- Error handling and rollback

**Usage:**
```bash
# Execute full merger
python execute_merger.py \
  --source /path/to/daiw \
  --target /path/to/kelly

# Dry run
python execute_merger.py \
  --source /path/to/daiw \
  --target . \
  --dry-run

# Skip validation for faster execution
python execute_merger.py \
  --source /path/to/daiw \
  --target . \
  --skip-validation
```

**Execution Steps:**
1. Deduplication Analysis
2. Create Monorepo Structure
3. Migrate Modules (with git history)
4. Standardize Names
5. Validate Migration
6. Generate Final Report

**Output:**
- Step-by-step progress
- Individual reports for each step
- Final comprehensive report in `migration_reports/`

---

## Quick Start Guide

### Full Migration Workflow

```bash
# 1. Clone both repositories
git clone https://github.com/your-org/daiw-music-brain.git
git clone https://github.com/your-org/kelly-music-brain-clean.git
cd kelly-music-brain-clean

# 2. Run deduplication analysis
python tools/scripts/deduplicate.py --scan . --output dedup-report.md

# 3. Review and clean up duplicates manually or with generated script

# 4. Execute full merger
python tools/scripts/execute_merger.py \
  --source ../daiw-music-brain \
  --target .

# 5. Review migration reports
ls migration_reports/

# 6. Run validation
python tools/scripts/validate_migration.py --repo . --test all

# 7. Commit and push
git add .
git commit -m "Complete DAiW → Kelly 2.0 migration"
git push
```

### Testing Individual Tools

```bash
# Test deduplication
python tools/scripts/deduplicate.py --scan . | head -50

# Test monorepo generation (dry run)
python tools/scripts/create_monorepo.py --output /tmp/test-monorepo --dry-run

# Test name standardization (scan only)
python tools/scripts/standardize_names.py --scan .

# Test validation
python tools/scripts/validate_migration.py --repo . --test imports
```

---

## Troubleshooting

### git-filter-repo not found
```bash
pip install git-filter-repo
```

### Import errors after migration
```bash
python tools/scripts/standardize_names.py --scan . --fix
```

### Build failures
```bash
# Check CMake configuration
python tools/scripts/validate_migration.py --repo . --test build
```

### Circular dependencies detected
```bash
python tools/scripts/validate_migration.py --repo . --test dependencies
```

---

## Configuration

### Excluding Directories

Edit `exclude_patterns` in each script:

```python
exclude_patterns = [
    '.git', 'node_modules', '__pycache__',
    'build', 'dist', 'external', 'JUCE'
]
```

### Module Definitions

Edit `MODULES` dict in `migrate_modules.py` to add/modify modules:

```python
MODULES = {
    'my_module': ModuleConfig(
        name='my_module',
        source_paths=['path/in/source'],
        target_path='path/in/target',
        dependencies=['other_module'],
        import_rewrites={'old': 'new'}
    )
}
```

---

## Best Practices

1. **Always run dry-run first** to preview changes
2. **Backup your repository** before running migration
3. **Review deduplication report** before deleting files
4. **Validate after each major step**
5. **Commit frequently** during migration
6. **Keep migration reports** for documentation

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/sburdges-eng/kelly-music-brain-clean/issues
- Documentation: https://github.com/sburdges-eng/kelly-music-brain-clean/docs

---

*Tools developed for Kelly Music Brain 2.0 repository merger.*
