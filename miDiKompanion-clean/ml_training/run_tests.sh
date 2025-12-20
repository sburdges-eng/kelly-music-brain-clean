#!/bin/bash
# Test runner script for ML training tests
# Usage: ./run_tests.sh [unit|integration|performance|all]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default to all tests
TEST_TYPE="${1:-all}"

echo "=========================================="
echo "ML Training Test Suite"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install pytest"
    exit 1
fi

# Run tests based on type
case "$TEST_TYPE" in
    unit)
        echo -e "${GREEN}Running unit tests...${NC}"
        pytest tests/unit/ -v -m "not slow"
        ;;
    integration)
        echo -e "${GREEN}Running integration tests...${NC}"
        pytest tests/integration/ -v -m "not slow"
        ;;
    performance)
        echo -e "${GREEN}Running performance tests...${NC}"
        pytest tests/performance/ -v -m performance
        ;;
    all)
        echo -e "${GREEN}Running all tests...${NC}"
        pytest tests/ -v
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Usage: $0 [unit|integration|performance|all]"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
