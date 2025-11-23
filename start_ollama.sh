#!/bin/bash
# Script to start Ollama service
# This script helps start Ollama in the background on your DGX Spark system

echo "Starting Ollama service..."
echo ""

# Try to find Ollama binary
OLLAMA_BIN=""

# Check common locations
if command -v ollama &> /dev/null; then
    OLLAMA_BIN="ollama"
    echo "✅ Found Ollama in PATH"
elif [ -f ~/.local/bin/ollama ]; then
    OLLAMA_BIN="$HOME/.local/bin/ollama"
    echo "✅ Found Ollama at $OLLAMA_BIN"
elif [ -f /usr/local/bin/ollama ]; then
    OLLAMA_BIN="/usr/local/bin/ollama"
    echo "✅ Found Ollama at $OLLAMA_BIN"
else
    echo "❌ Could not find Ollama binary"
    echo ""
    echo "Please install Ollama first:"
    echo "  Visit https://ollama.ai/download for installation instructions"
    echo "  Or run: curl https://ollama.ai/install.sh | sh"
    exit 1
fi

# Check if Ollama is already running
if pgrep -f ollama > /dev/null; then
    echo "⚠️  Ollama appears to be already running"
    echo "   Process IDs: $(pgrep -f ollama | tr '\n' ' ')"
    echo ""
    read -p "Do you want to start it anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 0
    fi
fi

# Start Ollama in the background
echo ""
echo "Starting Ollama in the background..."
echo "Logs will be written to: ollama.log"
echo ""

# Start with nohup so it continues running after terminal closes
nohup "$OLLAMA_BIN" serve > ollama.log 2>&1 &

# Get the process ID
OLLAMA_PID=$!

# Wait a moment for it to start
sleep 2

# Check if it's still running
if ps -p $OLLAMA_PID > /dev/null; then
    echo "✅ Ollama started successfully!"
    echo "   Process ID: $OLLAMA_PID"
    echo "   Log file: ollama.log"
    echo ""
    echo "To verify it's working, run:"
    echo "  curl http://localhost:11434/api/tags"
    echo ""
    echo "To stop Ollama, run:"
    echo "  kill $OLLAMA_PID"
    echo "  or"
    echo "  pkill -f ollama"
else
    echo "❌ Ollama failed to start"
    echo "Check ollama.log for error messages"
    exit 1
fi

