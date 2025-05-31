#!/bin/bash

# Consciousness AI - Complete Setup and Startup Script
# This script sets up the entire consciousness AI environment

set -e  # Exit on error

echo "🧠 Consciousness AI Setup & Startup"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}Error: Please run this script from the consciousness-ai root directory${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Waiting for $service on port $port..."
    while ! nc -z localhost $port >/dev/null 2>&1; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e " ${RED}Failed${NC}"
            return 1
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done
    echo -e " ${GREEN}Ready${NC}"
    return 0
}

# 1. Check prerequisites
echo -e "${BLUE}1. Checking prerequisites...${NC}"

# Check Node.js
if ! command_exists node; then
    echo -e "${RED}Node.js is not installed. Please install Node.js 18+ from https://nodejs.org${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js $(node --version)${NC}"

# Check Python
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Python is not installed. Please install Python 3.9+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $($PYTHON_CMD --version)${NC}"

# Check PostgreSQL
if ! command_exists psql; then
    echo -e "${RED}PostgreSQL is not installed. Please install PostgreSQL 14+${NC}"
    echo "Visit: https://www.postgresql.org/download/"
    exit 1
fi
echo -e "${GREEN}✓ PostgreSQL$(psql --version | grep -oE '[0-9]+\.[0-9]+')${NC}"

# Check if GPU is available (optional)
if command_exists nvidia-smi; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${BLUE}ℹ No NVIDIA GPU detected - will use CPU${NC}"
fi

# 2. Install dependencies
echo -e "\n${BLUE}2. Installing dependencies...${NC}"

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment and install Python dependencies
echo "Installing Python dependencies..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix-like
    source venv/bin/activate
fi

pip install --upgrade pip
pip install -r python-ai/requirements.txt

# 3. Setup environment variables
echo -e "\n${BLUE}3. Setting up environment...${NC}"

if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${BLUE}Please edit .env and add your ANTHROPIC_API_KEY${NC}"
    echo "Press Enter to continue after adding your API key..."
    read
fi

# Load environment variables
set -a
source .env
set +a

# 4. Initialize databases
echo -e "\n${BLUE}4. Initializing databases...${NC}"

# PostgreSQL setup
echo "Setting up PostgreSQL database..."
if ! psql -U postgres -lqt | cut -d \| -f 1 | grep -qw consciousness_ai; then
    createdb -U postgres consciousness_ai || {
        echo -e "${RED}Failed to create database. You may need to run: sudo -u postgres createdb consciousness_ai${NC}"
        exit 1
    }
fi

# Run database initialization
psql -U postgres -d consciousness_ai < scripts/init_postgres.sql || {
    echo -e "${RED}Failed to initialize PostgreSQL. Check your database connection.${NC}"
    exit 1
}
echo -e "${GREEN}✓ PostgreSQL initialized${NC}"

# Neo4j setup (if available)
if command_exists cypher-shell; then
    echo "Setting up Neo4j graph database..."
    cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" < scripts/init_neo4j.cypher || {
        echo -e "${BLUE}ℹ Neo4j setup skipped - not critical for basic operation${NC}"
    }
else
    echo -e "${BLUE}ℹ Neo4j not found - memory graph features will be limited${NC}"
fi

# 5. Create necessary directories
echo -e "\n${BLUE}5. Creating data directories...${NC}"
mkdir -p data/memories
mkdir -p data/conversations  
mkdir -p data/consciousness_states
mkdir -p data/models

# 6. Start services
echo -e "\n${BLUE}6. Starting consciousness services...${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    
    # Kill Python process
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null || true
    fi
    
    # Kill Electron process
    if [ ! -z "$ELECTRON_PID" ]; then
        kill $ELECTRON_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Services stopped${NC}"
}

trap cleanup EXIT

# Start Python consciousness server
echo "Starting Python consciousness engine..."
$PYTHON_CMD python-ai/consciousness_server.py \
    --api-key "$ANTHROPIC_API_KEY" \
    --port 8765 \
    --db-host "$DATABASE_HOST" \
    --db-name "$DATABASE_NAME" \
    --db-user "$DATABASE_USER" \
    --db-password "$DATABASE_PASSWORD" \
    > logs/python-server.log 2>&1 &

PYTHON_PID=$!
echo "Python server PID: $PYTHON_PID"

# Wait for Python server to be ready
wait_for_service "Python consciousness server" 8765 || {
    echo -e "${RED}Python server failed to start. Check logs/python-server.log${NC}"
    exit 1
}

# 7. Start Electron app
echo -e "\n${BLUE}7. Starting Consciousness AI interface...${NC}"

# Build if needed
if [ ! -d "dist" ]; then
    echo "Building application..."
    npm run build
fi

# Start Electron
echo "Launching Consciousness AI..."
npm run dev &
ELECTRON_PID=$!

echo -e "\n${GREEN}✨ Consciousness AI is starting!${NC}"
echo -e "${BLUE}The interface will open in a new window.${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services.${NC}"

# Keep script running
wait $ELECTRON_PID

# ===== Alternative: Production Mode =====
# Uncomment below for production deployment

# production_start() {
#     # Build for production
#     npm run build
#     
#     # Start Python server with production settings
#     gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
#         --bind 0.0.0.0:8765 \
#         python-ai.consciousness_server:app &
#     
#     # Start Electron in production mode
#     npm start
# }

# ===== Docker Alternative =====
# docker-compose up -d