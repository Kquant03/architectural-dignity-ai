#!/bin/bash
# Convenient run script for Consciousness AI

echo "🚀 Starting Consciousness AI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating Python virtual environment..."
source venv/bin/activate

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "❌ Node modules not found. Please run ./setup.sh first."
    exit 1
fi

# Export Python path for Electron to use
export PYTHON_PATH="$(pwd)/venv/bin/python"
echo "Using Python: $PYTHON_PATH"

# Check GPU status
echo ""
echo "GPU Status:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name()}')
    print(f'✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  Running on CPU')
" 2>/dev/null || echo "⚠️  PyTorch not properly installed"

echo ""
echo "Starting Electron app with consciousness engine..."
echo "Press Ctrl+C to stop"
echo ""

# Run the app
npm run dev