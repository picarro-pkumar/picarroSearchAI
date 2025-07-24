#!/bin/bash

# Quick Ollama Status Checker
echo "🔍 Ollama Status Check"
echo "======================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if Ollama processes are running
echo -e "${BLUE}📊 Process Status:${NC}"
if pgrep -f "ollama serve" > /dev/null; then
    echo -e "${GREEN}✅ Ollama serve process is running${NC}"
    ps aux | grep "ollama serve" | grep -v grep
else
    echo -e "${RED}❌ No ollama serve process found${NC}"
fi

if pgrep -f "ollama runner" > /dev/null; then
    echo -e "${GREEN}✅ Ollama runner process is running${NC}"
    ps aux | grep "ollama runner" | grep -v grep
else
    echo -e "${YELLOW}⚠️  No ollama runner process found${NC}"
fi

echo ""

# Check port usage
echo -e "${BLUE}🌐 Port Status:${NC}"
if lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Port 11434 is in use${NC}"
    lsof -i :11434
else
    echo -e "${RED}❌ Port 11434 is not in use${NC}"
fi

echo ""

# Check API response
echo -e "${BLUE}🔌 API Status:${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama API is responding${NC}"
    echo "Available models:"
    curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "  - Unable to parse models"
else
    echo -e "${RED}❌ Ollama API is not responding${NC}"
fi

echo ""

# Check PID file
echo -e "${BLUE}📁 PID File Status:${NC}"
if [ -f "/tmp/ollama.pid" ]; then
    PID=$(cat /tmp/ollama.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "${GREEN}✅ PID file exists and process is running (PID: $PID)${NC}"
    else
        echo -e "${YELLOW}⚠️  PID file exists but process is not running (PID: $PID)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No PID file found${NC}"
fi

echo ""
echo "💡 To restart with single instance: ./start_ollama.sh"
echo "💡 To start all services: ./start_servers.sh" 