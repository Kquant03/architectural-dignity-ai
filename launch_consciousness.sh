#!/bin/bash

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set CUDA environment for RTX 3090
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Launch consciousness server in background
echo "Starting consciousness server..."
python python-ai/consciousness_server.py --api-key "$ANTHROPIC_API_KEY" --port 8765 &
SERVER_PID=$!

# Give server time to start
sleep 3

# Launch Electron app
echo "Starting Electron app..."
npm start

# Cleanup on exit
trap "kill $SERVER_PID 2>/dev/null" EXIT