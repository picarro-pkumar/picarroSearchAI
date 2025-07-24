# Ollama Single Instance Management System

## 🎯 **Overview**

This system ensures that **Ollama always runs as a single instance**, preventing the issues caused by multiple processes competing for the same port (11434) and consuming excessive memory.

## 📁 **Files**

### **Core Management Scripts**

1. **`start_ollama.sh`** - Smart Ollama single instance manager
2. **`check_ollama.sh`** - Quick status checker and diagnostics
3. **`start_servers.sh`** - Updated main startup script with Ollama management

## 🚀 **Usage**

### **Quick Start (Recommended)**
```bash
# Start everything with single instance management
./start_servers.sh
```

### **Individual Commands**
```bash
# Start only Ollama with single instance management
./start_ollama.sh

# Check Ollama status and processes
./check_ollama.sh

# Start all services (backend + frontend + Ollama)
./start_servers.sh
```

## 🔧 **How It Works**

### **Single Instance Manager (`start_ollama.sh`)**

1. **Detection**: Checks if Ollama is already running properly
2. **Cleanup**: Kills any duplicate processes if found
3. **Verification**: Ensures API is responding and models are accessible
4. **PID Management**: Tracks process ID for clean shutdown
5. **Health Monitoring**: Waits for service to be ready before proceeding

### **Status Checker (`check_ollama.sh`)**

Provides comprehensive status information:
- ✅ Process status (serve + runner)
- 🌐 Port usage (11434)
- 🔌 API responsiveness
- 📁 PID file status
- 📊 Available models

## 🛡️ **Features**

### **Automatic Process Management**
- **Kills duplicates**: Removes multiple instances automatically
- **Port conflict resolution**: Handles "address already in use" errors
- **Force cleanup**: Uses `kill -9` for stubborn processes
- **PID tracking**: Maintains process ID for clean shutdown

### **Health Verification**
- **API testing**: Verifies `/api/tags` endpoint responds
- **Model accessibility**: Checks `ollama list` command
- **Port validation**: Confirms port 11434 is properly bound
- **Startup monitoring**: Waits up to 60 seconds for service readiness

### **Error Handling**
- **Graceful failures**: Proper error messages and exit codes
- **Signal handling**: Clean shutdown on Ctrl+C
- **Logging**: Detailed logs in `/tmp/ollama.log`
- **Recovery**: Automatic restart on failures

## 📊 **Memory Optimization**

### **Before (Multiple Instances)**
```
ollama serve: 80.2 MB
ollama runner: 1.15 GB (Instance 1)
ollama runner: 1.15 GB (Instance 2)
Total: ~2.3 GB
```

### **After (Single Instance)**
```
ollama serve: ~26 MB
ollama runner: 1.15 GB (Single instance)
Total: ~1.2 GB
```

**Savings: ~1.1 GB of memory**

## 🔍 **Troubleshooting**

### **Common Issues**

1. **"Address already in use"**
   ```bash
   ./start_ollama.sh  # Automatically fixes this
   ```

2. **Multiple processes detected**
   ```bash
   ./check_ollama.sh  # Shows all processes
   ./start_ollama.sh  # Cleans up and restarts
   ```

3. **API not responding**
   ```bash
   ./check_ollama.sh  # Diagnoses the issue
   ```

### **Manual Cleanup**
```bash
# Kill all Ollama processes
pkill -f "ollama"

# Kill by port
lsof -ti:11434 | xargs kill -9

# Remove PID file
rm -f /tmp/ollama.pid
```

## 🎯 **Integration with Backend**

The backend automatically detects Ollama status:
- **Connected**: Green status, search functionality enabled
- **Disconnected**: Red status, search disabled with clear error messages
- **Recovery**: Automatic reconnection when Ollama comes back online

## 📝 **Log Files**

- **Ollama logs**: `/tmp/ollama.log`
- **PID tracking**: `/tmp/ollama.pid`
- **Backend logs**: Check backend console output

## 🚀 **Best Practices**

1. **Always use the management scripts** instead of direct `ollama serve`
2. **Check status first** with `./check_ollama.sh` before troubleshooting
3. **Use `./start_servers.sh`** for complete system startup
4. **Monitor logs** in `/tmp/ollama.log` for detailed information

## 🔄 **Restart Scenarios**

### **After System Reboot**
```bash
./start_servers.sh
```

### **After Ollama Issues**
```bash
./start_ollama.sh
```

### **Complete System Reset**
```bash
pkill -f "ollama"
./start_servers.sh
```

## ✅ **Verification**

To verify everything is working:
```bash
# Check all services
./check_ollama.sh
curl http://localhost:8000/health
curl http://localhost:3000

# Test search functionality
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What are Picarro products?"}'
```

---

**🎉 Result**: Ollama now runs as a single, stable instance with automatic management and recovery! 