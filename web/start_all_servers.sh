#!/bin/bash

# Startup script for running all game servers
# This script starts both API servers and the proxy server

echo "=================================="
echo "🎮 Starting Unified Game Servers"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "❌ Virtual environment not found at ../.venv"
    echo "Please create it first with: uv venv"
    exit 1
fi

# Activate virtual environment
source ../.venv/bin/activate

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Stopping all servers..."
    kill $PID_ML3X3 $PID_5X5 $PID_10X10 $PID_PROXY 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start 3x3 ML API Server
echo "🤖 Starting 3x3 ML API Server (port 5003)..."
cd ../3x3
python api_server.py > ml3x3.log 2>&1 &
PID_ML3X3=$!
cd ../web

# Wait a moment for it to start
sleep 2

# Start 5x5 AlphaZero API Server
echo "🎯 Starting 5x5 AlphaZero API Server (port 5001)..."
cd ../5x5
python api_server.py > alphazero5x5.log 2>&1 &
PID_5X5=$!
cd ../web

# Wait a moment for it to start
sleep 2

# Start 10x10 AlphaZero API Server
echo "🧠 Starting 10x10 AlphaZero API Server (port 5002)..."
cd ../10x10
python api_server.py > alphazero10x10.log 2>&1 &
PID_10X10=$!
cd ../web

# Wait a moment for it to start
sleep 2

# Start Proxy Server
echo "🌐 Starting Proxy Server (port 8080)..."
python proxy_server.py &
PID_PROXY=$!

echo ""
echo "=================================="
echo "✅ All servers started!"
echo "=================================="
echo ""
echo "Server Status:"
echo "  - 3x3 ML API:        http://localhost:5003  (PID: $PID_ML3X3)"
echo "  - 5x5 AlphaZero API: http://localhost:5001  (PID: $PID_5X5)"
echo "  - 10x10 AlphaZero API: http://localhost:5002  (PID: $PID_10X10)"
echo "  - Proxy Server:      http://localhost:8080  (PID: $PID_PROXY)"
echo ""
echo "Access the game at: http://localhost:8080"
echo "To share remotely: ngrok http 8080"
echo ""
echo "Logs:"
echo "  - 3x3 ML:      tail -f ../3x3/ml3x3.log"
echo "  - 5x5 AlphaZero: tail -f ../5x5/alphazero5x5.log"
echo "  - 10x10 AlphaZero: tail -f ../10x10/alphazero10x10.log"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "=================================="

# Wait for proxy server (keeps script running)
wait $PID_PROXY
