#!/bin/bash
# Docker Compose Helper Script for miDiKompanion ML Training
# Usage: ./docker-run.sh [command] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find Docker binary (check PATH first, then Docker.app)
if command -v docker &> /dev/null; then
    DOCKER_CMD="docker"
elif [ -f "/Applications/Docker.app/Contents/Resources/bin/docker" ]; then
    DOCKER_CMD="/Applications/Docker.app/Contents/Resources/bin/docker"
    export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
    echo -e "${YELLOW}Note: Using Docker from Docker.app (not in PATH)${NC}"
else
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if docker compose is available (try v2 first, then v1)
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

# Function to print usage
usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start              Start training service (default)"
    echo "  stop               Stop all services"
    echo "  restart            Restart all services"
    echo "  logs               View training logs"
    echo "  shell              Open interactive shell in training container"
    echo "  build              Build Docker images (docker-compose)"
    echo "  build-all           Build and verify all images (comprehensive)"
    echo "  rebuild             Rebuild images without cache"
    echo "  prune               Clean up unused Docker resources"
    echo "  monitor             Start TensorBoard monitoring"
    echo "  data               Start data processing service"
    echo "  evaluate           Run model evaluation"
    echo "  benchmark          Run inference benchmarks"
    echo "  test               Run test suite"
    echo "  status             Show container status"
    echo "  clean              Stop and remove all containers"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start training"
    echo "  $0 start --detach           # Start in background"
    echo "  $0 monitor                  # Start TensorBoard"
    echo "  $0 shell                    # Interactive shell"
    echo "  $0 logs -f                  # Follow logs"
}

# Main command handler
case "${1:-start}" in
    start)
        echo -e "${GREEN}Starting training service...${NC}"
        $DOCKER_COMPOSE up "${@:2}"
        ;;

    stop)
        echo -e "${YELLOW}Stopping all services...${NC}"
        $DOCKER_COMPOSE down
        ;;

    restart)
        echo -e "${YELLOW}Restarting services...${NC}"
        $DOCKER_COMPOSE restart
        ;;

    logs)
        echo -e "${GREEN}Viewing logs...${NC}"
        $DOCKER_COMPOSE logs "${@:2}" training
        ;;

    shell)
        echo -e "${GREEN}Opening interactive shell...${NC}"
        $DOCKER_COMPOSE exec training /bin/bash || \
        $DOCKER_COMPOSE run --rm training /bin/bash
        ;;

    build)
        echo -e "${GREEN}Building Docker images...${NC}"
        $DOCKER_COMPOSE build "${@:2}"
        ;;

    build-all)
        echo -e "${GREEN}Building and verifying all images...${NC}"
        if [ -f "./build-all.sh" ]; then
            ./build-all.sh "${@:2}"
        else
            echo -e "${RED}Error: build-all.sh not found${NC}"
            exit 1
        fi
        ;;

    rebuild)
        echo -e "${YELLOW}Rebuilding images without cache...${NC}"
        $DOCKER_COMPOSE build --no-cache "${@:2}"
        echo -e "${GREEN}Rebuild complete${NC}"
        ;;

    prune)
        echo -e "${YELLOW}Cleaning up unused Docker resources...${NC}"
        read -p "This will remove unused images, containers, and networks. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $DOCKER_CMD system prune -a --volumes -f
            echo -e "${GREEN}Prune complete${NC}"
        else
            echo "Cancelled"
        fi
        ;;

    monitor)
        echo -e "${GREEN}Starting TensorBoard monitoring...${NC}"
        echo "TensorBoard will be available at: http://localhost:6006"
        $DOCKER_COMPOSE --profile monitoring up "${@:2}" tensorboard
        ;;

    data)
        echo -e "${GREEN}Starting data processing service...${NC}"
        $DOCKER_COMPOSE --profile data up "${@:2}" data-processing
        ;;

    evaluate)
        echo -e "${GREEN}Running model evaluation...${NC}"
        $DOCKER_COMPOSE --profile evaluation up "${@:2}" evaluation
        ;;

    benchmark)
        echo -e "${GREEN}Running inference benchmarks...${NC}"
        $DOCKER_COMPOSE --profile benchmark up "${@:2}" benchmark
        ;;

    test)
        echo -e "${GREEN}Running test suite...${NC}"
        $DOCKER_COMPOSE --profile testing up "${@:2}" testing
        ;;

    status)
        echo -e "${GREEN}Container status:${NC}"
        $DOCKER_COMPOSE ps
        ;;

    clean)
        echo -e "${YELLOW}Cleaning up containers and volumes...${NC}"
        read -p "This will remove all containers. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $DOCKER_COMPOSE down -v --remove-orphans
            echo -e "${GREEN}Cleanup complete${NC}"
        else
            echo "Cancelled"
        fi
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        echo -e "${RED}Unknown command: $1${NC}"
        usage
        exit 1
        ;;
esac
