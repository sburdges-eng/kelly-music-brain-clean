#!/bin/bash
# build-all.sh - Comprehensive Docker Build and Verification Script
# Usage: ./build-all.sh [--no-cache] [--skip-tests]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
NO_CACHE=""
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-cache] [--skip-tests]"
            exit 1
            ;;
    esac
done

# Find Docker binary
if command -v docker &> /dev/null; then
    DOCKER_CMD="docker"
elif [ -f "/Applications/Docker.app/Contents/Resources/bin/docker" ]; then
    DOCKER_CMD="/Applications/Docker.app/Contents/Resources/bin/docker"
    export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
else
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Find Docker Compose
if $DOCKER_CMD compose version &> /dev/null; then
    DOCKER_COMPOSE="$DOCKER_CMD compose"
elif $DOCKER_CMD-compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="$DOCKER_CMD-compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Docker Build and Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Build data processing image (smaller, faster)
echo -e "${YELLOW}Step 1: Building data processing image...${NC}"
echo "----------------------------------------"
if $DOCKER_CMD build -f Dockerfile.data -t midikompanion-data:latest $NO_CACHE .; then
    echo -e "${GREEN}✓ Data processing image built successfully${NC}"
else
    echo -e "${RED}✗ Data processing image build failed${NC}"
    exit 1
fi
echo ""

# Step 2: Build training image (larger, takes longer)
echo -e "${YELLOW}Step 2: Building training image...${NC}"
echo "----------------------------------------"
if $DOCKER_COMPOSE build $NO_CACHE training; then
    echo -e "${GREEN}✓ Training image built successfully${NC}"
else
    echo -e "${RED}✗ Training image build failed${NC}"
    exit 1
fi
echo ""

# Step 3: Verify images exist
echo -e "${YELLOW}Step 3: Verifying images...${NC}"
echo "----------------------------------------"
if $DOCKER_CMD images | grep -q "midikompanion-data"; then
    echo -e "${GREEN}✓ midikompanion-data:latest found${NC}"
else
    echo -e "${RED}✗ midikompanion-data:latest not found${NC}"
    exit 1
fi

if $DOCKER_CMD images | grep -q "midikompanion-training"; then
    echo -e "${GREEN}✓ midikompanion-training:latest found${NC}"
else
    echo -e "${RED}✗ midikompanion-training:latest not found${NC}"
    exit 1
fi
echo ""

# Step 4: Test basic container functionality
echo -e "${YELLOW}Step 4: Testing container functionality...${NC}"
echo "----------------------------------------"

# Test training container
echo "Testing training container..."
if $DOCKER_CMD run --rm midikompanion-training:latest python -c "import torch; import numpy; import pandas; print('All imports successful')" 2>&1; then
    echo -e "${GREEN}✓ Training container imports work${NC}"
else
    echo -e "${RED}✗ Training container imports failed${NC}"
    exit 1
fi

# Test data processing container
echo "Testing data processing container..."
if $DOCKER_CMD run --rm midikompanion-data:latest python -c "import mido; import pandas; import numpy; print('All imports successful')" 2>&1; then
    echo -e "${GREEN}✓ Data processing container imports work${NC}"
else
    echo -e "${RED}✗ Data processing container imports failed${NC}"
    exit 1
fi
echo ""

# Step 5: Run tests (optional)
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${YELLOW}Step 5: Running basic tests...${NC}"
    echo "----------------------------------------"

    # Test pytest availability
    if $DOCKER_CMD run --rm midikompanion-training:latest python -c "import pytest; print(f'pytest {pytest.__version__} available')" 2>&1; then
        echo -e "${GREEN}✓ pytest is available${NC}"
    else
        echo -e "${YELLOW}⚠ pytest not available (non-critical)${NC}"
    fi
    echo ""
fi

# Step 6: Show image sizes
echo -e "${YELLOW}Step 6: Image summary...${NC}"
echo "----------------------------------------"
$DOCKER_CMD images midikompanion-* --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""

# Final summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Start training:     ./docker-run.sh start"
echo "  2. Start with monitor:  ./docker-run.sh monitor"
echo "  3. Process data:        ./docker-run.sh data"
echo "  4. Run tests:           ./docker-run.sh test"
echo ""
