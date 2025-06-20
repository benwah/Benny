#!/bin/bash

echo "🧠 Neural Network InputServer Test Suite"
echo "========================================"

# Function to cleanup background processes
cleanup() {
    echo "🧹 Cleaning up background processes..."
    jobs -p | xargs -r kill
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "📦 Building the project..."
cargo build --example neural_network_server --bin input_server

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

echo "🚀 Starting Neural Network Server..."
cargo run --example neural_network_server &
NN_SERVER_PID=$!

# Wait a moment for the server to start
sleep 2

echo "🌐 Starting InputServer..."
cargo run --bin input_server -- \
    --network-host 127.0.0.1 \
    --network-port 8001 \
    --web-host 0.0.0.0 \
    --web-port 12000 \
    --websocket-port 12001 \
    --input-size 4 &
INPUT_SERVER_PID=$!

# Wait a moment for the input server to start
sleep 3

echo ""
echo "🎉 System is running!"
echo "================================"
echo "📡 Neural Network Server: 127.0.0.1:8001"
echo "🌐 Web Interface: http://127.0.0.1:12000"
echo "🔌 WebSocket: ws://127.0.0.1:12001"
echo ""
echo "📋 Instructions:"
echo "1. Open http://127.0.0.1:12000 in your browser"
echo "2. Use the sliders to adjust input values"
echo "3. Click 'Send to Network' to transmit data"
echo "4. Watch the console for neural network activity"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
wait