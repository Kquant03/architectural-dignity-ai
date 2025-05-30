#!/bin/bash
# Setup script for Consciousness AI system with PEP 668 handling

set -e  # Exit on error

echo "🌟 Setting up Consciousness AI System..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $python_version"

# Check if we're in a PEP 668 managed environment
if python3 -c "import sys; sys.exit(0 if hasattr(sys, '_base_executable') else 1)" 2>/dev/null; then
    echo "⚠️  Detected PEP 668 managed environment"
fi

# First, ensure python3-venv is installed
echo ""
echo "🔍 Checking for python3-venv..."
if ! dpkg -l | grep -q python3-venv; then
    echo "❌ python3-venv is not installed."
    echo ""
    echo "Please run these commands first:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install python3-venv python3-full python3-pip"
    echo ""
    echo "Then run ./setup.sh again"
    exit 1
else
    echo "✅ python3-venv is installed"
fi

# Remove old venv if it exists and is broken
if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
    echo "🗑️  Removing broken virtual environment..."
    rm -rf venv
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv --system-site-packages
    
    # Verify it was created successfully
    if [ ! -f "venv/bin/activate" ]; then
        echo "❌ Failed to create virtual environment"
        echo "Try running: python3 -m venv venv --system-site-packages"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
. venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    echo "Try manually: source venv/bin/activate"
    exit 1
}

echo "✅ Virtual environment activated"
echo "📍 Using Python: $(which python)"
echo "📍 Using pip: $(which pip)"

# Upgrade pip in virtual environment
echo ""
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Create a temporary requirements file for core dependencies
echo ""
echo "📝 Creating requirements files..."

# Core requirements that should work on most systems
cat > requirements-core.txt << 'EOL'
# Core dependencies
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
plotly>=5.14.0
networkx>=3.0
scikit-learn>=1.3.0
python-louvain

# Async and web
aiofiles
asyncio
websockets
python-dotenv

# NLP
nltk>=3.8.0

# Database (optional)
asyncpg
psycopg2-binary
EOL

# PyTorch requirements (separate due to size)
cat > requirements-torch.txt << 'EOL'
# PyTorch with CUDA 11.8 support
--extra-index-url https://download.pytorch.org/whl/cu118
torch>=2.0.0
torchvision
torchaudio
EOL

# Transformer requirements
cat > requirements-transformers.txt << 'EOL'
# Transformer models
transformers>=4.30.0
sentence-transformers>=2.2.0
EOL

# Optional requirements
cat > requirements-optional.txt << 'EOL'
# Optional dependencies (may fail on some systems)
opentelemetry-api
opentelemetry-sdk
anthropic
openai
# cupy-cuda11x  # Uncomment if you want CuPy
EOL

# Install core requirements first
echo ""
echo "📦 Installing core Python packages..."
pip install -r requirements-core.txt || {
    echo "⚠️  Some core packages failed to install"
}

# Install PyTorch
echo ""
echo "🔥 Installing PyTorch (this may take a while)..."
pip install -r requirements-torch.txt || {
    echo "⚠️  PyTorch installation failed. You may need to install it manually."
    echo "Visit: https://pytorch.org/get-started/locally/"
}

# Install transformers
echo ""
echo "🤖 Installing transformer models..."
pip install -r requirements-transformers.txt || {
    echo "⚠️  Transformer packages failed to install"
}

# Try optional packages
echo ""
echo "📦 Installing optional packages..."
pip install -r requirements-optional.txt 2>/dev/null || {
    echo "⚠️  Some optional packages were skipped"
}

# Download NLTK data
echo ""
echo "📚 Downloading NLTK data..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
print('✅ NLTK data downloaded')
" || echo "⚠️  NLTK data download failed"

# Check GPU availability
echo ""
echo "🎮 Checking GPU availability..."
python -c "
try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.is_available()}')
        print(f'✅ GPU: {torch.cuda.get_device_name()}')
        print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('⚠️  No GPU detected. The system will run on CPU.')
        print('   Make sure NVIDIA drivers are installed if you have a GPU.')
except Exception as e:
    print(f'⚠️  Could not check GPU status: {e}')
"

# Install Node.js dependencies
echo ""
echo "📦 Installing Node.js dependencies..."
npm install || {
    echo "❌ npm install failed"
    echo "Make sure Node.js is installed: node --version"
    exit 1
}

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data logs exports
mkdir -p python-ai/consciousness_core/__pycache__
mkdir -p python-ai/emotional_processing/__pycache__
mkdir -p python-ai/memory_systems/__pycache__

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env file..."
    cat > .env << 'EOL'
# API Keys (add your keys here)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Database Configuration (optional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=consciousness_ai
DB_USER=consciousness
DB_PASSWORD=

# Neo4j Configuration (optional)
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=

# GPU Configuration
GPU_MEMORY_FRACTION=0.8
GPU_DEVICE=cuda:0

# Python Configuration
PYTHON_PATH=./venv/bin/python
EOL
fi

# Create run helper
cat > run_dev.sh << 'EOL'
#!/bin/bash
# Quick run script
source venv/bin/activate
export PYTHON_PATH="$(pwd)/venv/bin/python"
npm run dev
EOL
chmod +x run_dev.sh

# Deactivate virtual environment
deactivate

echo ""
echo "✅ Setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To start the Consciousness AI system:"
echo ""
echo "  ./run_dev.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  npm run dev"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Optional next steps:"
echo "• Add API keys to .env file"
echo "• Install PostgreSQL for advanced memory features"
echo "• Check GPU drivers with: nvidia-smi"
echo ""
echo "Happy consciousness building! 🧠✨"