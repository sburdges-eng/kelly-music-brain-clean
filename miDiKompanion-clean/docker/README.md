# Docker-Based Multi-Variant Build System

Production-grade containerized build system for generating 1000+ plugin variants (UI themes, feature sets, platform targets) with automated CI/CD pipeline support.

## Overview

This system enables building multiple variants of the miDiKompanion plugin with different:

- **Platforms**: Linux, macOS, Windows
- **UI Themes**: Dark, Light, High Contrast, Custom Branded
- **Feature Sets**: Free Tier, Pro Tier, Enterprise Tier
- **ML Models**: Various emotion recognition and groove prediction models
- **Localizations**: Multiple languages

**Total Variants**: 3 platforms × 4 themes × 3 feature sets × 4 models × 5 languages = 720+ base variants

## Quick Start

### 1. Build Base Images

```bash
# Build base builder image
docker build -t midikompanion-base-builder:latest -f docker/Dockerfile.base-builder .

# Build macOS builder (if needed)
docker build -t midikompanion-macos-builder:latest -f docker/Dockerfile.macos-cross .

# Build Windows builder (if needed)
docker build -t midikompanion-windows-builder:latest -f docker/Dockerfile.windows-cross .
```

### 2. Build Single Variant (Local Testing)

```bash
# Build a specific variant
./scripts/docker_build_local.sh linux-x64_default-dark_pro-tier_emotion-recognition-v1_en-US

# Or use the orchestrator
python3 scripts/docker_build_orchestrator.py --variant-id "linux-x64_default-dark_pro-tier_emotion-recognition-v1_en-US"
```

### 3. List All Variants

```bash
python3 scripts/docker_build_orchestrator.py --list-variants
```

### 4. Build All Variants

```bash
# Build all variants (parallel, max 8 workers)
python3 scripts/docker_build_orchestrator.py --config build_matrix.yaml --workers 8

# Build with filter
python3 scripts/docker_build_orchestrator.py --variant-id "linux-x64"
```

## Architecture

### Base Docker Images

1. **base-builder**: Linux build environment with all dependencies
2. **macos-builder**: macOS cross-compilation (placeholder, typically use native)
3. **windows-builder**: Windows cross-compilation with MinGW

### Build Configuration

`build_matrix.yaml` defines all variant combinations:

- Platforms and their CMake flags
- UI themes and CSS files
- Feature sets and enabled/disabled features
- ML models and their file paths
- Localizations and string files

### Build Orchestration

`docker_build_orchestrator.py`:

- Generates all variant combinations
- Validates combinations (e.g., no ML models in free tier)
- Builds variants in parallel Docker containers
- Extracts and packages artifacts
- Generates build reports

## CI/CD Integration

GitHub Actions workflow (`.github/workflows/build_all_variants.yml`):

- Triggers on push to main/release branches
- Generates build matrix dynamically
- Builds variants in parallel (max 10 concurrent)
- Uploads artifacts
- Creates GitHub releases for release branches

## Build Scripts

### Inside Container

- `scripts/build/build_variant.sh`: Runs inside container to build a variant
- `scripts/package/package_variant.sh`: Packages artifacts for a variant

### Local Development

- `scripts/docker_build_local.sh`: Quick local build for testing
- `scripts/build_dashboard.py`: Generate HTML dashboard from build results

## Artifacts

Build artifacts are stored in:

```
build/
├── artifacts/
│   └── <variant_id>/
│       ├── vst3/
│       ├── au/
│       ├── lv2/
│       ├── clap/
│       └── metadata.json
├── packages/
│   └── <variant_id>.zip
└── build_results.json
```

## Monitoring

Generate build dashboard:

```bash
python3 scripts/build_dashboard.py build/build_results.json
```

Opens `build_dashboard.html` with statistics by:

- Platform
- Theme
- Feature Set
- ML Model
- Localization

## Requirements

- Docker and Docker Compose
- Python 3.11+
- Python packages: `docker`, `pyyaml`

Install Python dependencies:

```bash
pip install docker pyyaml
```

## Troubleshooting

### Docker Not Running

```bash
# Check Docker status
docker info

# Start Docker Desktop (macOS/Windows)
# Or: sudo systemctl start docker (Linux)
```

### Build Failures

- Check logs in container output
- Verify CMake configuration
- Ensure all dependencies are installed in base image
- Check variant combination validity in `build_matrix.yaml`

### Out of Disk Space

```bash
# Clean up Docker images
docker system prune -a

# Clean up build artifacts
rm -rf build/artifacts build/packages
```

## Customization

### Add New Theme

1. Add theme to `build_matrix.yaml` under `ui_themes`
2. Create CSS file in `themes/` directory
3. Rebuild variants

### Add New Feature Set

1. Add feature set to `build_matrix.yaml` under `feature_sets`
2. Define enabled/disabled features
3. Update CMake configuration if needed

### Add New Platform

1. Create new Dockerfile in `docker/` directory
2. Add platform to `build_matrix.yaml`
3. Define CMake flags and output formats

## Performance

- **Parallel Builds**: Configurable max workers (default: 8)
- **Caching**: Docker layer caching for faster rebuilds
- **Incremental**: Only rebuilds changed variants
- **ccache**: Compiler cache for faster compilation

## Security

- Base images use official Ubuntu images
- No secrets in Dockerfiles
- Artifacts can be signed/notarized (macOS) in CI/CD pipeline

## Support

For issues or questions:

1. Check build logs in `build/build_results.json`
2. Review dashboard for patterns in failures
3. Verify Docker and dependencies are installed
4. Check variant combination validity
