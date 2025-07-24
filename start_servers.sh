#!/bin/bash

# Picarro SearchAI - Server Startup Script
echo "🚀 Starting Picarro SearchAI servers..."

# Function to check if a port is in use
check_port() {
    lsof -ti:$1 > /dev/null 2>&1
}

# Function to kill processes on a port
kill_port() {
    if check_port $1; then
        echo "⚠️  Port $1 is in use. Killing existing processes..."
        lsof -ti:$1 | xargs kill -9
        sleep 2
    fi
}

# Kill any existing processes on our ports
kill_port 8000
kill_port 3000

# Start backend server
echo "🔧 Starting backend server on port 8000..."
cd backend
python3 app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is healthy
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend server is running and healthy"
else
    echo "❌ Backend server failed to start"
    exit 1
fi

# Start frontend server
echo "🎨 Starting frontend server on port 3000..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to initialize..."
sleep 10

# Check if frontend is running
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend server is running"
else
    echo "❌ Frontend server failed to start"
    exit 1
fi

echo ""
echo "🎉 Picarro SearchAI is now running!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
wait 