#!/bin/bash

# Picarro SearchAI Docker Services Management Script
# This script helps manage the 24/7 deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if Ollama is running
check_ollama() {
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "Ollama is running"
        return 0
    else
        print_warning "Ollama is not running. Please start it with: ./start_ollama.sh"
        return 1
    fi
}

# Function to check if required model is available
check_model() {
    local model="llama3:latest"
    if curl -s http://localhost:11434/api/tags | jq -e ".models[] | select(.name == \"$model\")" > /dev/null 2>&1; then
        print_status "Model $model is available"
        return 0
    else
        print_warning "Model $model not found. Please pull it with: ollama pull $model"
        return 1
    fi
}

# Function to start services
start_services() {
    print_header "Starting Picarro SearchAI Services"
    
    # Check Ollama
    if ! check_ollama; then
        print_error "Cannot start services without Ollama running"
        exit 1
    fi
    
    # Check model
    if ! check_model; then
        print_error "Cannot start services without required model"
        exit 1
    fi
    
    print_status "Building and starting Docker containers..."
    docker-compose up --build -d
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are healthy
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_status "✅ Backend is healthy"
    else
        print_warning "Backend health check failed"
    fi
    
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_status "✅ Frontend is accessible"
    else
        print_warning "Frontend accessibility check failed"
    fi
    
    print_status "Services started successfully!"
    print_status "Frontend: http://localhost:3000"
    print_status "Backend API: http://localhost:8000"
    print_status "Health Check: http://localhost:8000/health"
}

# Function to stop services
stop_services() {
    print_header "Stopping Picarro SearchAI Services"
    docker-compose down
    print_status "Services stopped successfully!"
}

# Function to restart services
restart_services() {
    print_header "Restarting Picarro SearchAI Services"
    stop_services
    sleep 2
    start_services
}

# Function to show status
show_status() {
    print_header "Picarro SearchAI Services Status"
    
    echo ""
    print_status "Docker Containers:"
    docker-compose ps
    
    echo ""
    print_status "Service Health Checks:"
    
    # Backend health
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        local backend_status=$(curl -s http://localhost:8000/health | jq -r .status 2>/dev/null || echo "unknown")
        echo "  Backend: $backend_status"
    else
        echo "  Backend: not accessible"
    fi
    
    # Frontend status
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "  Frontend: accessible"
    else
        echo "  Frontend: not accessible"
    fi
    
    # Ollama status
    if check_ollama; then
        local model=$(curl -s http://localhost:11434/api/tags | jq -r '.models[0].name' 2>/dev/null || echo "unknown")
        echo "  Ollama: running (model: $model)"
    else
        echo "  Ollama: not running"
    fi
    
    echo ""
    print_status "Access URLs:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  Health Check: http://localhost:8000/health"
}

# Function to show logs
show_logs() {
    print_header "Picarro SearchAI Services Logs"
    
    if [ "$1" = "backend" ]; then
        print_status "Showing backend logs (Ctrl+C to exit):"
        docker-compose logs -f backend
    elif [ "$1" = "frontend" ]; then
        print_status "Showing frontend logs (Ctrl+C to exit):"
        docker-compose logs -f frontend
    else
        print_status "Showing all logs (Ctrl+C to exit):"
        docker-compose logs -f
    fi
}

# Function to force resync
force_resync() {
    print_header "Force Resync Confluence Data"
    
    if [ ! -f "backend/force_resync.py" ]; then
        print_error "Force resync script not found"
        exit 1
    fi
    
    print_status "Running force resync..."
    cd backend && python3 force_resync.py && cd ..
    
    print_status "Restarting backend to pick up new data..."
    docker-compose restart backend
    
    print_status "Force resync completed!"
}

# Function to show help
show_help() {
    print_header "Picarro SearchAI Services Management"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services (builds if needed)"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show status of all services"
    echo "  logs        Show logs (all services)"
    echo "  logs backend    Show backend logs only"
    echo "  logs frontend   Show frontend logs only"
    echo "  resync      Force resync Confluence data"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start        # Start all services"
    echo "  $0 status       # Check service status"
    echo "  $0 logs backend # Monitor backend logs"
    echo "  $0 resync       # Force resync data"
    echo ""
}

# Main script logic
case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    resync)
        force_resync
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 