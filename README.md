# Picarro SearchAI

A powerful AI-powered search and chat interface for Picarro's documentation, built with React, FastAPI, and Ollama.

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** - For containerized deployment
- **Ollama** - Local LLM server (Meta-Llama-3-8B-Instruct recommended)
- **Confluence Access** - For documentation synchronization

### 1. Clone the Repository

```bash
git clone <repository-url>
cd picarro-SearchAI
```

### 2. Set Up Environment Variables

```bash
# Copy the example environment file
cp backend/env.example backend/.env

# Edit the environment file with your Confluence credentials
nano backend/.env
```

**Required Environment Variables:**
```env
CONFLUENCE_URL=https://your-company.atlassian.net
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=YOUR_SPACE_KEY
```

### 3. Install and Start Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull llama3.2:8b-instruct-q4_0

# Start Ollama server
ollama serve
```

### 4. Start the Application

```bash
# Start both backend and frontend services
docker-compose up -d

# Check service status
docker-compose ps
```

### 5. Access the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## 📚 Initial Setup

### First-Time Data Synchronization

After starting the services, you need to sync your Confluence documentation:

```bash
# Navigate to backend directory
cd backend

# Run the initial migration
python confluence_migrate.py
```

This will:
- Download all pages from your Confluence space
- Process and chunk the content
- Store embeddings in ChromaDB
- Create the knowledge base for AI responses

## 🛠️ Development Setup

### Running Locally (Without Docker)

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python app.py
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 🔧 Common Issues & Debugging

### 1. Ollama Connection Issues

**Problem:** "LLM is offline" or connection errors

**Solutions:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve

# Verify model is available
ollama list

# Pull model if missing
ollama pull llama3.2:8b-instruct-q4_0
```

### 2. Docker Services Not Starting

**Problem:** Services fail to start or show unhealthy status

**Solutions:**
```bash
# Check Docker status
docker-compose ps

# View logs for specific service
docker-compose logs backend
docker-compose logs frontend

# Restart services
docker-compose down
docker-compose up -d

# Rebuild if needed
docker-compose build --no-cache
docker-compose up -d
```

### 3. Confluence Authentication Errors

**Problem:** "Authentication failed" or "Access denied"

**Solutions:**
- Verify your Confluence API token is correct
- Check if your account has access to the specified space
- Ensure the space key is correct
- Try regenerating your API token in Atlassian

### 4. Data Synchronization Issues

**Problem:** AI returns old or incorrect information

**Solutions:**
```bash
# Force complete resync
cd backend
python force_resync.py

# Or manually clear and resync
rm -rf chroma_db
rm confluence_sync_state.json
python confluence_migrate.py
```

### 5. Frontend Not Loading

**Problem:** Frontend shows errors or doesn't load

**Solutions:**
```bash
# Check if backend is accessible
curl http://localhost:8000/health

# Verify nginx configuration
docker-compose exec frontend nginx -t

# Check frontend logs
docker-compose logs frontend
```

### 6. Memory Issues

**Problem:** System becomes slow or crashes

**Solutions:**
- Ensure you have at least 8GB RAM available
- Close other memory-intensive applications
- Consider using a smaller model: `ollama pull llama3.2:3b-instruct-q4_0`

### 7. Port Conflicts

**Problem:** "Address already in use" errors

**Solutions:**
```bash
# Check what's using the ports
lsof -i :3000
lsof -i :8000
lsof -i :11434

# Kill conflicting processes
kill -9 <PID>

# Or change ports in docker-compose.yml
```

## 📊 Monitoring & Maintenance

### Check System Health

```bash
# Service status
docker-compose ps

# Resource usage
docker stats

# Logs
docker-compose logs -f

# Database size
du -sh backend/chroma_db
```

### Regular Maintenance

```bash
# Update dependencies
docker-compose build --no-cache

# Clean up Docker
docker system prune -f

# Backup ChromaDB (optional)
cp -r backend/chroma_db backup_chroma_db_$(date +%Y%m%d)
```

## 🔒 Security Considerations

- Keep your Confluence API token secure
- Don't commit `.env` files to version control
- Regularly update dependencies
- Use HTTPS in production
- Consider network isolation for sensitive data

## 📝 API Endpoints

### Core Endpoints
- `POST /api/search` - Search and chat with AI
- `GET /api/health` - Health check
- `POST /api/force-resync` - Force data resync
- `GET /api/chat-history` - Get chat history

### Example Usage
```bash
# Test search endpoint
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I calibrate the analyzer?"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review the logs for error messages
- Ensure all prerequisites are met
- Verify your Confluence access and credentials

## 📄 License

This project is proprietary to Picarro Inc.

---

**Developed and trained by Unified Knowledge Explorers team** 