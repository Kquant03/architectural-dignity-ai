#!/bin/bash
# Pre-setup script to install system dependencies

echo "🔧 Installing system dependencies for Consciousness AI..."

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Python development packages
echo "Installing Python development packages..."
sudo apt-get install -y python3-full python3-venv python3-dev python3-pip

# Install build essentials for compiling Python packages
echo "Installing build essentials..."
sudo apt-get install -y build-essential

# Install PostgreSQL and development headers (optional, for database features)
echo "Installing PostgreSQL (optional)..."
sudo apt-get install -y postgresql postgresql-contrib libpq-dev || echo "⚠️  PostgreSQL installation skipped"

# Install system libraries for Python packages
echo "Installing system libraries..."
sudo apt-get install -y \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libatlas-base-dev \
    liblapack-dev \
    gfortran

# Install Node.js if not present
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "✅ Node.js already installed: $(node --version)"
fi

# Install Git if not present
if ! command -v git &> /dev/null; then
    echo "Installing Git..."
    sudo apt-get install -y git
fi

# Check NVIDIA drivers and CUDA (for GPU support)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "✅ NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo ""
    echo "⚠️  No NVIDIA driver detected. GPU acceleration will not be available."
    echo "   To install NVIDIA drivers:"
    echo "   sudo apt-get install nvidia-driver-525"  # or appropriate version
fi

echo ""
echo "✅ System dependencies installed!"
echo ""
echo "Now run ./setup.sh to set up the Python environment and install packages."