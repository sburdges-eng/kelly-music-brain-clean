#!/bin/bash
# Verify Docker Installation and Setup
# Usage: ./verify-docker.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Docker Installation Verification"
echo "=========================================="
echo ""

# Check if Docker Desktop app exists
if [ -d "/Applications/Docker.app" ]; then
    echo -e "${GREEN}✓ Docker Desktop app found${NC}"
else
    echo -e "${RED}✗ Docker Desktop app not found${NC}"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker CLI is available
echo "Checking Docker CLI..."
MAX_WAIT=60
WAIT_COUNT=0

while ! command -v docker &> /dev/null; do
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo -e "${RED}✗ Docker CLI not available after ${MAX_WAIT} seconds${NC}"
        echo ""
        echo "Please ensure Docker Desktop is running:"
        echo "1. Open Docker Desktop from Applications"
        echo "2. Wait for it to fully start (whale icon in menu bar)"
        echo "3. Run this script again"
        exit 1
    fi

    echo -e "${YELLOW}Waiting for Docker CLI... (${WAIT_COUNT}/${MAX_WAIT}s)${NC}"
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
done

echo -e "${GREEN}✓ Docker CLI available${NC}"

# Check Docker version
DOCKER_VERSION=$(docker --version)
echo -e "${GREEN}✓ ${DOCKER_VERSION}${NC}"

# Check Docker Compose
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    echo -e "${GREEN}✓ Docker Compose v2: ${COMPOSE_VERSION}${NC}"
    DOCKER_COMPOSE="docker compose"
elif docker-compose --version &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    echo -e "${GREEN}✓ Docker Compose v1: ${COMPOSE_VERSION}${NC}"
    DOCKER_COMPOSE="docker-compose"
else
    echo -e "${YELLOW}⚠ Docker Compose not found${NC}"
    echo "Install with: pip install docker-compose"
    DOCKER_COMPOSE=""
fi

# Check if Docker daemon is running
echo ""
echo "Checking Docker daemon..."
if docker info &> /dev/null; then
    echo -e "${GREEN}✓ Docker daemon is running${NC}"
else
    echo -e "${RED}✗ Docker daemon is not running${NC}"
    echo ""
    echo "Please start Docker Desktop:"
    echo "1. Open Docker Desktop from Applications"
    echo "2. Wait for it to fully start"
    exit 1
fi

# Check GPU support (will fail on macOS, that's expected)
echo ""
echo "Checking GPU support..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU support available${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ GPU support not available (expected on macOS)${NC}"
    echo "   Training will use CPU (slower but works)"
    GPU_AVAILABLE=false
fi

# Summary
echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "${GREEN}✓ Docker Desktop: Installed${NC}"
echo -e "${GREEN}✓ Docker CLI: Available${NC}"
if [ -n "$DOCKER_COMPOSE" ]; then
    echo -e "${GREEN}✓ Docker Compose: Available${NC}"
else
    echo -e "${YELLOW}⚠ Docker Compose: Not found${NC}"
fi
echo -e "${GREEN}✓ Docker Daemon: Running${NC}"
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${GREEN}✓ GPU Support: Available${NC}"
else
    echo -e "${YELLOW}⚠ GPU Support: Not available (CPU mode)${NC}"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Test the setup:"
echo "   cd ml_training"
echo "   ./docker-run.sh status"
echo ""
echo "2. Build the images:"
echo "   ./docker-run.sh build"
echo ""
echo "3. Start training:"
echo "   ./docker-run.sh start"
echo ""
echo "Note: On macOS, training will use CPU (no GPU support)"
echo "      For GPU training, use Google Colab or AWS"
echo ""
