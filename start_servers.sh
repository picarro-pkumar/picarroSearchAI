#!/bin/bash

# Picarro SearchAI - Server Startup Script with Single Instance Ollama Management
echo "🚀 Starting Picarro SearchAI with Single Instance Ollama Management..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

# Function to show success
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to show warning
warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to show error
error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if a port is in use
check_port() {
    lsof -ti:$1 > /dev/null 2>&1
}

# Function to kill processes on a port
kill_port() {
    if check_port $1; then
        warning "Port $1 is in use. Killing existing processes..."
        lsof -ti:$1 | xargs kill -9
        sleep 2
    fi
}

# Kill any existing processes on our ports
kill_port 8000
kill_port 3000

# Start Ollama with single instance management
log "Starting Ollama with single instance management..."
if ./start_ollama.sh; then
    success "Ollama started successfully"
else
    error "Failed to start Ollama"
    exit 1
fi

# Start backend server
log "Starting backend server on port 8000..."
cd backend
python3 app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
log "Waiting for backend to initialize..."
sleep 5

# Check if backend is healthy
if curl -s http://localhost:8000/health > /dev/null; then
    success "Backend server is running and healthy"
else
    error "Backend server failed to start"
    exit 1
fi

# Start frontend server
log "Starting frontend server on port 3000..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
log "Waiting for frontend to initialize..."
sleep 10

# Check if frontend is running
if curl -s http://localhost:3000 > /dev/null; then
    success "Frontend server is running"
else
    error "Frontend server failed to start"
    exit 1
fi

# Final health checks
log "Performing final health checks..."

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null; then
    success "Ollama is running on http://localhost:11434"
else
    error "Ollama health check failed"
fi

echo ""
success "🎉 All services started successfully!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "🤖 Ollama: http://localhost:11434"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
warning "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    log "Shutting down all services..."
    
    # Kill backend and frontend
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    
    # Stop Ollama using the manager script
    if [ -f "/tmp/ollama.pid" ]; then
        OLLAMA_PID=$(cat /tmp/ollama.pid)
        kill $OLLAMA_PID 2>/dev/null || true
        rm -f /tmp/ollama.pid
    fi
    
    success "All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Wait for user to stop
wait 