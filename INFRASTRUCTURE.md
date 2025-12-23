# Kelly Music Brain 2.0 - Repository Merger Infrastructure

This directory contains the complete infrastructure for merging DAiW-Music-Brain into Kelly Music Brain 2.0 as a monorepo.

## 🎯 Quick Start

### Option 1: Use Codespaces (Recommended)

1. Open this repository in GitHub Codespaces
2. Wait for automatic setup (`.devcontainer/` configures everything)
3. All tools and dependencies are ready to use!

### Option 2: Local Development

```bash
# Clone repository
git clone https://github.com/sburdges-eng/kelly-music-brain-clean.git
cd kelly-music-brain-clean

# Install Python dependencies
pip install -e ".[dev]"

# Build C++ components (optional)
cmake -B build && cmake --build build
```

---

## 📁 Infrastructure Components

### 1. Development Container (`.devcontainer/`)

Complete Codespace/VS Code Dev Container configuration:

**Files:**
- `devcontainer.json` - Container configuration with all extensions
- `Dockerfile` - Multi-language build environment (Python 3.11, C++20, Node.js 18)
- `post-create.sh` - Automatic setup script

**Includes:**
- Python 3.11+ with Poetry
- C++20 with gcc-12/clang-15
- CMake 3.27+, Ninja
- Qt6, JUCE, audio libraries
- Node.js 18+ with pnpm
- git-filter-repo, pre-commit
- VS Code extensions (Python, C++, CMake, GitLens)

### 2. Migration Tools (`tools/scripts/`)

**Core Tools:**
1. **`deduplicate.py`** - Find duplicate files by content, name patterns, semantic similarity
2. **`create_monorepo.py`** - Generate complete monorepo scaffold with all configs
3. **`migrate_modules.py`** - Migrate modules with git history preservation
4. **`standardize_names.py`** - Fix naming conventions across codebase
5. **`validate_migration.py`** - Comprehensive validation suite
6. **`execute_merger.py`** - Master orchestrator for full migration

**Documentation:**
- `README.md` - Detailed tool usage guide

See [tools/scripts/README.md](tools/scripts/README.md) for complete documentation.

### 3. CI/CD Pipelines (`.github/workflows/`)

**Monorepo-Aware Workflows:**
- `ci-python.yml` - Python testing, linting, coverage (runs on package changes)
- `ci-cpp.yml` - C++ build/test across platforms (runs on package changes)
- `build-plugins.yml` - VST3/CLAP plugin builds with codesigning
- `release-monorepo.yml` - Automated releases to PyPI, Docker Hub, GitHub Releases

**Existing Workflows:**
- `ci.yml` - Unified CI for all components
- `test.yml` - Test suite runner
- `release.yml` - Release automation
- `build-macos-app.yml` - macOS app bundling

### 4. Templates (`tools/templates/`)

**Available Templates:**
- `DAIW_DEPRECATION.md` - Ready-to-use deprecation notice for DAiW repository

### 5. Documentation (`docs/`)

**Guides:**
- `MIGRATION.md` - Complete migration guide for users and developers
- `guides/` - Topic-specific guides (to be populated)

---

## 🚀 Merger Execution

### Full Automated Merger

```bash
# Execute complete merger (requires DAiW source repo)
python tools/scripts/execute_merger.py \
  --source /path/to/daiw-music-brain \
  --target .
```

This runs:
1. ✅ Deduplication analysis
2. ✅ Monorepo structure generation
3. ✅ Module migration with history
4. ✅ Name standardization
5. ✅ Validation suite
6. ✅ Final report generation

### Step-by-Step Merger

```bash
# 1. Analyze duplicates
python tools/scripts/deduplicate.py --scan . --output dedup-report.md

# 2. Create monorepo structure
python tools/scripts/create_monorepo.py --output .

# 3. Migrate modules
python tools/scripts/migrate_modules.py \
  --source /path/to/daiw \
  --target . \
  --module all

# 4. Standardize naming
python tools/scripts/standardize_names.py --scan . --fix

# 5. Validate migration
python tools/scripts/validate_migration.py --repo . --test all
```

---

## 📊 Current Status

### Completed ✅

- [x] Codespace development environment
- [x] All migration tools implemented
- [x] CI/CD workflows for monorepo
- [x] Comprehensive documentation
- [x] Deduplication tested (231 duplicate groups found)
- [x] Name standardization scanner working (198 files with old imports)

### Ready for Execution ⏳

- [ ] Merge DAiW repository (requires source repo access)
- [ ] Execute full migration
- [ ] Deploy monorepo structure
- [ ] Update CI/CD configurations
- [ ] Publish Kelly 2.0 to PyPI

### Analysis Results

**Deduplication Scan:**
- 1,560 files scanned
- 231 duplicate groups found
  - 205 exact matches
  - 0 name pattern matches
  - 26 semantic matches

**Naming Issues:**
- 198 files with old imports (`music_brain`, `penta_core`, `daiw`)
- 27 files with suffix patterns (`*_music-brain`, `*_penta-core`)

---

## 🛠️ Development Workflow

### Daily Development

```bash
# Pull latest changes
git pull

# Create feature branch
git checkout -b feature/my-feature

# Make changes...

# Run linters
ruff check . && black --check .

# Run tests
pytest tests/python

# Build C++ (if needed)
cmake --build build

# Commit and push
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

### Pre-Commit Hooks

Automatically run on every commit:
- ✅ ruff (linting and formatting)
- ✅ black (code formatting)
- ✅ clang-format (C++ formatting)
- ✅ Trailing whitespace removal
- ✅ YAML/JSON validation
- ✅ Large file detection

Install: `pre-commit install`

---

## 📦 Monorepo Structure

The migration creates this structure:

```
kelly-music-brain/
├── .devcontainer/          # Development container config
├── .github/
│   └── workflows/          # CI/CD pipelines
├── packages/               # Monorepo packages
│   ├── core/
│   │   ├── python/         # Python core library
│   │   └── cpp/            # C++ audio engine
│   ├── cli/                # Command-line interface
│   ├── desktop/            # Qt6 desktop app
│   ├── plugins/            # VST3/CLAP plugins
│   ├── mobile/             # iOS/Android apps
│   └── web/                # Web interface
├── data/                   # Emotion maps, scales, genres
├── vault/                  # Knowledge base (from DAiW)
├── tests/                  # Test suites
│   ├── python/
│   └── cpp/
├── tools/                  # Migration and development tools
│   ├── scripts/            # Migration automation
│   └── templates/          # File templates
├── docs/                   # Documentation
│   ├── MIGRATION.md
│   ├── architecture/
│   ├── api/
│   └── guides/
├── examples/               # Usage examples
├── CMakeLists.txt          # Root build configuration
├── pyproject.toml          # Python workspace config
├── package.json            # Node.js workspace config
└── .pre-commit-config.yaml # Pre-commit hooks
```

---

## 🔍 Validation

### Automated Checks

```bash
# Run all validations
python tools/scripts/validate_migration.py --repo . --test all

# Specific validations
python tools/scripts/validate_migration.py --repo . --test imports
python tools/scripts/validate_migration.py --repo . --test tests
python tools/scripts/validate_migration.py --repo . --test build
python tools/scripts/validate_migration.py --repo . --test history
```

### Manual Verification

```bash
# Check for broken imports
grep -r "from music_brain" packages/ || echo "✓ No old imports"

# Verify tests pass
pytest tests/python -v

# Verify build works
cmake -B build && cmake --build build

# Check git history preserved
git log --all --graph --oneline | head -20
```

---

## 📚 Documentation

### For Users
- [Migration Guide](docs/MIGRATION.md) - Complete guide for migrating from DAiW
- [Tool Documentation](tools/scripts/README.md) - How to use migration tools
- [Deprecation Notice](tools/templates/DAIW_DEPRECATION.md) - For DAiW repository

### For Developers
- [Codespace Setup](.devcontainer/README.md) - Development environment
- [CI/CD Workflows](.github/workflows/) - Automated pipelines
- [Monorepo Guide](docs/guides/monorepo.md) - Working with monorepo structure

---

## 🎯 Merger Strategy

### Phase-Based Approach

**Week 1-2: Infrastructure Setup**
- ✅ Development environment (Codespaces)
- ✅ Migration tools
- ✅ CI/CD pipelines
- ✅ Documentation

**Week 3-4: Analysis & Cleanup**
- ⏳ Run deduplication analysis
- ⏳ Clean up duplicate files
- ⏳ Standardize naming conventions
- ⏳ Verify git history

**Week 5-6: Module Migration**
- ⏳ Migrate core emotion engine
- ⏳ Migrate intent schema system
- ⏳ Migrate groove tools
- ⏳ Migrate teaching module
- ⏳ Migrate vault knowledge base

**Week 7-8: Integration & Testing**
- ⏳ Create monorepo structure
- ⏳ Integrate all modules
- ⏳ Update import paths
- ⏳ Run full test suite
- ⏳ Fix integration issues

**Week 9-10: Validation & Release**
- ⏳ Comprehensive validation
- ⏳ Performance benchmarks
- ⏳ Documentation updates
- ⏳ Deprecate DAiW repository
- ⏳ Release Kelly 2.0

---

## 🆘 Troubleshooting

### Common Issues

**Issue: Codespace setup fails**
```bash
# Check Docker build
cat .devcontainer/post-create.sh
# Manually run setup steps
```

**Issue: Migration script errors**
```bash
# Check git-filter-repo installed
pip install git-filter-repo

# Run in dry-run mode first
python tools/scripts/migrate_modules.py --dry-run --module all
```

**Issue: Import errors after migration**
```bash
# Fix automatically
python tools/scripts/standardize_names.py --scan . --fix

# Or manually
find . -name "*.py" -exec sed -i 's/music_brain/kelly_core/g' {} +
```

**Issue: Build failures**
```bash
# Check dependencies installed
cmake --version
qt-cmake --version  # Qt6

# Reconfigure
rm -rf build
cmake -B build
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Testing requirements
- Pull request process
- Release procedures

---

## 📞 Support

- **Issues:** https://github.com/sburdges-eng/kelly-music-brain-clean/issues
- **Discussions:** https://github.com/sburdges-eng/kelly-music-brain-clean/discussions
- **Migration Help:** Open issue with `[migration]` tag

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built for Kelly Music Brain 2.0** | *From "Interrogate Before Generate" to therapeutic music creation at scale*
