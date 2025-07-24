# Picarro SearchAI

A complete AI-powered search and document retrieval system for Picarro, featuring a FastAPI backend with semantic search capabilities and a React frontend with ChatGPT-like interface.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Node.js 16+** with npm
3. **Ollama** running locally with `llama3:latest` model

### Installation

1. **Clone the repository** (if not already done)
2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

#### Option 1: Use the startup script (Recommended)
```bash
./start_servers.sh
```

#### Option 2: Manual startup

**Terminal 1 - Backend:**
```bash
cd backend
python3 app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### Access the Application

- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## 🏗️ Architecture

### Backend (`backend/`)
- **FastAPI** web framework
- **ChromaDB** vector database for semantic search
- **Sentence Transformers** for document embedding
- **Ollama** integration for AI responses
- **Document Processor** for chunking and indexing

### Frontend (`frontend/`)
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **ChatGPT-like interface** with dark theme
- **Persistent chat history** using localStorage
- **Responsive design** for all devices

## 📁 Project Structure

```
picarro-SearchAI/
├── backend/
│   ├── app.py              # FastAPI main application
│   ├── ai_responder.py     # AI response generation
│   ├── doc_processor.py    # Document processing and search
│   ├── config.py           # Configuration settings
│   ├── requirements.txt    # Python dependencies
│   └── chroma_db/          # Vector database storage
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── types/          # TypeScript type definitions
│   │   └── App.tsx         # Main application component
│   ├── package.json        # Node.js dependencies
│   └── tailwind.config.js  # Tailwind configuration
├── data/                   # Sample documents
├── start_servers.sh        # Startup script
└── README.md              # This file
```

## 🔧 Configuration

### Backend Configuration
- **Ollama URL:** `http://localhost:11434/api/generate`
- **Model:** `llama3:latest`
- **Chunk Size:** 1000 characters
- **Similarity Threshold:** 0.7
- **Max Results:** 5

### Frontend Configuration
- **API Endpoint:** `http://localhost:8000/search`
- **Theme:** Dark mode by default
- **Chat History:** Persisted in localStorage

## 🎯 Features

### Backend Features
- ✅ Semantic document search
- ✅ AI-powered question answering
- ✅ Document ingestion and processing
- ✅ Health monitoring and statistics
- ✅ CORS support for frontend
- ✅ Comprehensive error handling
- ✅ Domain-specific responses (Picarro only)

### Frontend Features
- ✅ ChatGPT-like chat interface
- ✅ Dark theme with modern UI
- ✅ Persistent chat history (localStorage)
- ✅ Chat history management (clear all, delete individual)
- ✅ Typing animations
- ✅ Copy response functionality
- ✅ Regenerate response option
- ✅ Responsive design
- ✅ Loading states and error handling
- ✅ Save/load notifications

## 🔍 API Endpoints

- `GET /` - Root endpoint with basic info
- `GET /health` - Health check with system status
- `POST /search` - Main search endpoint
- `POST /add-document` - Add new documents
- `GET /stats` - Knowledge base statistics

## 🛠️ Development

### Adding New Documents
```bash
curl -X POST "http://localhost:8000/add-document" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document content here",
    "metadata": {"source": "manual", "category": "technical"}
  }'
```

### Testing the Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Picarro's technology?",
    "max_results": 5
  }'
```

## 🐛 Troubleshooting

### Common Issues

1. **Port 3000 already in use:**
   ```bash
   lsof -ti:3000 | xargs kill -9
   ```

2. **Port 8000 already in use:**
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

3. **Ollama not running:**
   ```bash
   ollama serve
   ollama pull llama3:latest
   ```

4. **Python dependencies missing:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

5. **Node modules missing:**
   ```bash
   cd frontend
   npm install
   ```

### Health Check
Visit http://localhost:8000/health to verify all services are running properly.

## 📝 Notes

- The system is designed to answer questions specifically about Picarro technology and documents
- Out-of-domain questions will be politely refused
- Chat history is stored locally in the browser
- The backend requires Ollama to be running with the llama3:latest model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is proprietary to Picarro and contains confidential information. 