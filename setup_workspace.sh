#!/usr/bin/env bash
set -euo pipefail

# Basic workspace bootstrap: verify toolchain and optionally create venv.

REQ_PYTHON_MINOR=9
REQ_CMAKE_MAJOR=3
REQ_CMAKE_MINOR=22

check_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

version_ge() {
  # shellcheck disable=SC2209
  [ "$(printf '%s\n%s' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

check_python() {
  check_cmd python3
  PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  PY_MINOR=${PY_VER#*.}
  if ! version_ge "$PY_MINOR" "$REQ_PYTHON_MINOR"; then
    echo "Python 3.$REQ_PYTHON_MINOR+ required, found $PY_VER" >&2
    exit 1
  fi
  echo "Python OK: $PY_VER"
}

check_cmake() {
  check_cmd cmake
  CM_VER=$(cmake --version | awk 'NR==1{print $3}')
  CM_MAJOR=${CM_VER%%.*}
  CM_MINOR=$(echo "$CM_VER" | cut -d. -f2)
  if ! version_ge "$CM_MAJOR" "$REQ_CMAKE_MAJOR" || ! version_ge "$CM_MINOR" "$REQ_CMAKE_MINOR"; then
    echo "CMake $REQ_CMAKE_MAJOR.$REQ_CMAKE_MINOR+ required, found $CM_VER" >&2
    exit 1
  fi
  echo "CMake OK: $CM_VER"
}

check_compiler() {
  if command -v clang++ >/dev/null 2>&1; then
    clang++ --version | head -n1
  elif command -v g++ >/dev/null 2>&1; then
    g++ --version | head -n1
  else
    echo "No C++ compiler found (clang++/g++)" >&2
    exit 1
  fi
}

create_venv() {
  if [ -d "venv" ]; then
    echo "venv already exists, skipping creation"
    return
  fi
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip
  echo "Venv created at ./venv"
}

main() {
  check_python
  check_cmake
  check_compiler
  echo "Toolchain verification complete."

  if [ "${1:-}" = "--venv" ]; then
    create_venv
    echo "To activate: source venv/bin/activate"
  fi
}

main "$@"
