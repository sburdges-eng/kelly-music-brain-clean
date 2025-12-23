# Kelly Music Brain 2.0 Development Container
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc-12 \
    g++-12 \
    clang-15 \
    clang++-15 \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    git \
    git-lfs \
    curl \
    wget \
    # Python development
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Audio libraries
    libasound2-dev \
    portaudio19-dev \
    libsndfile1-dev \
    libfftw3-dev \
    libjack-jackd2-dev \
    # Qt6 for desktop GUI
    qt6-base-dev \
    qt6-multimedia-dev \
    libqt6svg6-dev \
    # Additional utilities
    vim \
    nano \
    htop \
    tree \
    zip \
    unzip \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-12 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100

# Install Poetry for Python dependency management
RUN curl -sSL https://install.python-poetry.org | python3.11 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Install pnpm for Node.js workspace
RUN npm install -g pnpm@8

# Install git-filter-repo for history preservation
RUN pip3 install git-filter-repo

# Install Python development tools
RUN pip3 install --upgrade pip \
    && pip3 install \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    ipython \
    ipdb

# Install catch2 for C++ testing
RUN git clone --branch v3.5.0 --depth 1 https://github.com/catchorg/Catch2.git /tmp/catch2 \
    && cd /tmp/catch2 \
    && cmake -B build -DBUILD_TESTING=OFF \
    && cmake --build build --target install \
    && rm -rf /tmp/catch2

# Set up ccache
ENV PATH="/usr/lib/ccache:${PATH}"
ENV CCACHE_DIR="/workspace/.ccache"
ENV CCACHE_MAXSIZE="5G"

# Create workspace directory
RUN mkdir -p /workspace

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["/bin/bash"]
