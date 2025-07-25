#!/bin/bash

# Ollama Single Instance Manager
# Ensures only one Ollama instance runs at any time

set -e

OLLAMA_PORT=11434
OLLAMA_PID_FILE="/tmp/ollama.pid"
LOG_FILE="/tmp/ollama.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
}

# Function to check if Ollama is running
check_ollama_running() {
    if pgrep -f "ollama serve" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Function to check if port is in use
check_port_in_use() {
    if lsof -Pi :$OLLAMA_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Function to kill all Ollama processes
kill_ollama_processes() {
    log "Killing all existing Ollama processes..."
    
    # Kill by process name
    pkill -f "ollama serve" 2>/dev/null || true
    pkill -f "ollama runner" 2>/dev/null || true
    
    # Kill by port
    if check_port_in_use; then
        lsof -ti:$OLLAMA_PORT | xargs kill -9 2>/dev/null || true
    fi
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Double-check and force kill if needed
    if check_ollama_running; then
        warning "Some processes still running, force killing..."
        pkill -9 -f "ollama" 2>/dev/null || true
        sleep 1
    fi
}

# Function to start Ollama
start_ollama() {
    log "Starting Ollama server..."
    
    # Start Ollama in background and capture PID
    nohup ollama serve > "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    
    # Save PID to file
    echo $OLLAMA_PID > "$OLLAMA_PID_FILE"
    
    log "Ollama started with PID: $OLLAMA_PID"
    
    # Wait for Ollama to be ready
    local attempts=0
    local max_attempts=30
    
    while [ $attempts -lt $max_attempts ]; do
        if curl -s http://localhost:$OLLAMA_PORT/api/tags > /dev/null 2>&1; then
            success "Ollama is ready and responding on port $OLLAMA_PORT"
            return 0
        fi
        
        attempts=$((attempts + 1))
        log "Waiting for Ollama to be ready... (attempt $attempts/$max_attempts)"
        sleep 2
    done
    
    error "Ollama failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Function to verify Ollama is working
verify_ollama() {
    log "Verifying Ollama functionality..."
    
    # Check if models are available
    if ollama list > /dev/null 2>&1; then
        success "Ollama models are accessible"
    else
        error "Ollama models are not accessible"
        return 1
    fi
    
    # Test API endpoint
    if curl -s http://localhost:$OLLAMA_PORT/api/tags > /dev/null 2>&1; then
        success "Ollama API is responding"
    else
        error "Ollama API is not responding"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    log "=== Ollama Single Instance Manager ==="
    
    # Check if Ollama is already running properly
    if check_ollama_running && check_port_in_use; then
        if verify_ollama; then
            success "Ollama is already running properly"
            log "PID: $(cat $OLLAMA_PID_FILE 2>/dev/null || echo 'unknown')"
            return 0
        else
            warning "Ollama is running but not responding properly"
        fi
    fi
    
    # Kill any existing processes
    if check_ollama_running || check_port_in_use; then
        kill_ollama_processes
    fi
    
    # Start fresh instance
    if start_ollama; then
        success "Ollama started successfully as single instance"
        log "Process ID: $(cat $OLLAMA_PID_FILE)"
        log "Log file: $LOG_FILE"
        log "API endpoint: http://localhost:$OLLAMA_PORT"
        return 0
    else
        error "Failed to start Ollama"
        return 1
    fi
}

# Handle script termination
cleanup() {
    log "Shutting down Ollama..."
    if [ -f "$OLLAMA_PID_FILE" ]; then
        OLLAMA_PID=$(cat "$OLLAMA_PID_FILE")
        if kill -0 "$OLLAMA_PID" 2>/dev/null; then
            kill "$OLLAMA_PID"
            log "Sent termination signal to Ollama (PID: $OLLAMA_PID)"
        fi
        rm -f "$OLLAMA_PID_FILE"
    fi
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Run main function
main "$@" 