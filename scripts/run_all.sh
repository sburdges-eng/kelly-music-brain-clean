#!/bin/bash
# Run All Components Script
# 
# This script helps you run all components of the Kelly project.
# It provides options to run components individually or together.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    if command_exists lsof; then
        lsof -Pi :"$1" -sTCP:LISTEN -t >/dev/null 2>&1
    else
        return 1
    fi
}

# Function to start Python API
start_python_api() {
    print_info "Starting Python API Server..."
    
    # Check for virtual environment
    VENV_DIR=""
    for venv_name in ".venv" "venv" "env"; do
        if [ -d "$venv_name" ]; then
            VENV_DIR="$venv_name"
            break
        fi
    done
    
    if [ -z "$VENV_DIR" ]; then
        print_error "Virtual environment not found. Creating one..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -e ".[dev]"
    else
        source "$VENV_DIR/bin/activate"
    fi
    
    # Check if API module exists
    if ! python3 -c "import music_brain.api" 2>/dev/null; then
        print_error "music_brain.api module not found. Installing..."
        pip install -e ".[dev]"
    fi
    
    # Check port
    if port_in_use 8000; then
        print_warning "Port 8000 is already in use. Trying port 8001..."
        PORT=8001 ./scripts/start_api_server.sh &
    else
        ./scripts/start_api_server.sh &
    fi
    
    print_success "Python API starting in background..."
    sleep 2
}

# Function to start Desktop App
start_desktop_app() {
    print_info "Starting Desktop App..."
    
    if [ ! -f "package.json" ]; then
        print_warning "package.json not found in project root. Skipping desktop app."
        return
    fi
    
    if ! command_exists npm; then
        print_error "npm not found. Please install Node.js."
        return
    fi
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_info "Installing npm dependencies..."
        npm install
    fi
    
    print_info "Starting Tauri dev mode..."
    npm run tauri dev &
    print_success "Desktop app starting in background..."
}

# Function to build C++ Audio Engine
build_cpp_engine() {
    print_info "Building C++ Audio Engine..."
    
    if ! command_exists cmake; then
        print_error "cmake not found. Please install CMake."
        return
    fi
    
    cd audio-engine-cpp || {
        print_error "audio-engine-cpp directory not found"
        return
    }
    
    mkdir -p build
    cd build
    
    if [ ! -f "CMakeCache.txt" ]; then
        print_info "Configuring CMake..."
        cmake ..
    fi
    
    print_info "Building..."
    make -j8
    
    print_success "C++ Audio Engine built successfully"
    cd "$PROJECT_ROOT"
}

# Function to build JUCE Plugin
build_juce_plugin() {
    print_info "Building JUCE Plugin..."
    
    if ! command_exists cmake; then
        print_error "cmake not found. Please install CMake."
        return
    fi
    
    cd plugin-juce || {
        print_error "plugin-juce directory not found"
        return
    }
    
    mkdir -p build
    cd build
    
    if [ ! -f "CMakeCache.txt" ]; then
        print_info "Configuring CMake..."
        cmake ..
    fi
    
    print_info "Building..."
    make -j8
    
    print_success "JUCE Plugin built successfully"
    cd "$PROJECT_ROOT"
}

# Function to run tests
run_tests() {
    print_info "Running tests..."
    
    # Python tests
    if [ -d "tests_music-brain" ]; then
        print_info "Running Python tests..."
        source .venv/bin/activate 2>/dev/null || true
        pytest tests_music-brain/ -v || print_warning "Some Python tests failed"
    fi
    
    # C++ tests
    if [ -d "audio-engine-cpp/build" ]; then
        print_info "Running C++ tests..."
        cd audio-engine-cpp/build
        ctest -V || print_warning "Some C++ tests failed"
        cd "$PROJECT_ROOT"
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Component Status"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Check Python API
    if port_in_use 8000; then
        print_success "Python API: Running on port 8000"
        curl -s http://127.0.0.1:8000/health >/dev/null 2>&1 && \
            print_success "  └─ API is responding" || \
            print_warning "  └─ API port open but not responding"
    else
        print_warning "Python API: Not running"
    fi
    
    # Check Desktop App
    if port_in_use 1420; then
        print_success "Desktop App: Running (Tauri dev server on port 1420)"
    else
        print_warning "Desktop App: Not running"
    fi
    
    # Check C++ build
    if [ -f "audio-engine-cpp/build/CMakeCache.txt" ]; then
        print_success "C++ Audio Engine: Built"
    else
        print_warning "C++ Audio Engine: Not built"
    fi
    
    # Check Plugin build
    if [ -f "plugin-juce/build/CMakeCache.txt" ]; then
        print_success "JUCE Plugin: Built"
    else
        print_warning "JUCE Plugin: Not built"
    fi
    
    echo ""
}

# Main menu
show_menu() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎵 Kelly Project - Component Runner"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1) Start Python API Server"
    echo "2) Start Desktop App"
    echo "3) Build C++ Audio Engine"
    echo "4) Build JUCE Plugin"
    echo "5) Run All (API + Desktop App)"
    echo "6) Build All (C++ + Plugin)"
    echo "7) Run Tests"
    echo "8) Show Status"
    echo "9) Exit"
    echo ""
    echo -n "Select an option [1-9]: "
}

# Main loop
if [ "$1" = "--api" ]; then
    start_python_api
elif [ "$1" = "--desktop" ]; then
    start_desktop_app
elif [ "$1" = "--build-cpp" ]; then
    build_cpp_engine
elif [ "$1" = "--build-plugin" ]; then
    build_juce_plugin
elif [ "$1" = "--all" ]; then
    start_python_api
    sleep 3
    start_desktop_app
    print_success "All components started!"
    echo ""
    print_info "Press Ctrl+C to stop all services"
    wait
elif [ "$1" = "--build-all" ]; then
    build_cpp_engine
    build_juce_plugin
elif [ "$1" = "--test" ]; then
    run_tests
elif [ "$1" = "--status" ]; then
    show_status
elif [ -z "$1" ]; then
    # Interactive mode
    while true; do
        show_menu
        read -r choice
        case $choice in
            1) start_python_api ;;
            2) start_desktop_app ;;
            3) build_cpp_engine ;;
            4) build_juce_plugin ;;
            5) 
                start_python_api
                sleep 3
                start_desktop_app
                print_info "Both services running. Press Ctrl+C to stop."
                wait
                ;;
            6) 
                build_cpp_engine
                build_juce_plugin
                ;;
            7) run_tests ;;
            8) show_status ;;
            9) 
                print_info "Exiting..."
                exit 0
                ;;
            *) print_error "Invalid option. Please select 1-9." ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
else
    echo "Usage: $0 [--api|--desktop|--build-cpp|--build-plugin|--all|--build-all|--test|--status]"
    echo ""
    echo "Options:"
    echo "  --api          Start Python API server"
    echo "  --desktop      Start Desktop app"
    echo "  --build-cpp    Build C++ Audio Engine"
    echo "  --build-plugin Build JUCE Plugin"
    echo "  --all          Start API + Desktop App"
    echo "  --build-all    Build C++ + Plugin"
    echo "  --test         Run all tests"
    echo "  --status       Show component status"
    echo ""
    echo "No arguments: Interactive menu"
fi

