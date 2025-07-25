from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import uvicorn
from datetime import datetime
import os

from ai_responder import AIResponder, Source
from doc_processor import DocumentProcessor
from chat_manager import chat_manager
from models.chat_models import (
    SaveChatRequest, SaveChatResponse, ChatListResponse,
    UpdateChatTitleRequest, ChatSearchResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Picarro SearchAI API",
    description="AI-powered search and document retrieval system for Picarro",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=5000)
    max_results: Optional[int] = Field(5, description="Maximum number of results to return", ge=1, le=20)
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")

class SourceResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int

class SearchResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
    query: str
    response_time: float
    confidence_score: Optional[float] = None
    model_used: str
    timestamp: datetime

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="Document content", min_length=1)
    doc_id: Optional[str] = Field(None, description="Optional custom document ID")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Optional document metadata")

class AddDocumentResponse(BaseModel):
    doc_id: str
    message: str
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    api_status: str
    ollama_status: str
    document_processor_status: str
    timestamp: datetime
    details: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime

class LlmStatusResponse(BaseModel):
    connected: bool
    model: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime

# Global instances
document_processor = None
ai_responder = None

def initialize_services():
    """Initialize DocumentProcessor and AIResponder instances."""
    global document_processor, ai_responder
    
    try:
        logger.info("Initializing DocumentProcessor...")
        document_processor = DocumentProcessor()
        
        logger.info("Initializing AIResponder...")
        ollama_url = os.getenv("OLLAMA_URL")
        ai_responder = AIResponder(document_processor, ollama_url=ollama_url)
        
        logger.info("All services initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services when the application starts."""
    logger.info("Starting Picarro SearchAI API...")
    if not initialize_services():
        logger.error("Failed to initialize services. API may not function properly.")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Basic info endpoint.
    
    Returns:
        API information and status
    """
    return {
        "name": "Picarro SearchAI API",
        "version": "1.0.0",
        "description": "AI-powered search and document retrieval system",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "search": "/search",
            "health": "/health", 
            "add_document": "/add-document",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check system status including API, Ollama, and DocumentProcessor.
    
    Returns:
        Health status of all system components
    """
    try:
        # Check API status
        api_status = "healthy"
        
        # Check DocumentProcessor status
        doc_processor_status = "healthy"
        doc_processor_details = {}
        if document_processor:
            try:
                stats = document_processor.get_collection_stats()
                doc_processor_details = stats
            except Exception as e:
                doc_processor_status = "error"
                doc_processor_details = {"error": str(e)}
        else:
            doc_processor_status = "not_initialized"
        
        # Check Ollama status
        ollama_status = "healthy"
        ollama_details = {}
        if ai_responder:
            try:
                is_connected = ai_responder._check_ollama_connection()
                ollama_status = "healthy" if is_connected else "not_connected"
                ollama_details = {
                    "connected": is_connected,
                    "model": ai_responder.model_name,
                    "url": ai_responder.ollama_url
                }
            except Exception as e:
                ollama_status = "error"
                ollama_details = {"error": str(e)}
        else:
            ollama_status = "not_initialized"
        
        # Overall status
        overall_status = "healthy"
        if doc_processor_status != "healthy" or ollama_status not in ["healthy", "not_connected"]:
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            api_status=api_status,
            ollama_status=ollama_status,
            document_processor_status=doc_processor_status,
            timestamp=datetime.now(),
            details={
                "document_processor": doc_processor_details,
                "ollama": ollama_details
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/api/llm-status", response_model=LlmStatusResponse)
async def get_llm_status():
    """Get LLM (Ollama) connection status."""
    try:
        if not ai_responder:
            return LlmStatusResponse(
                connected=False,
                error="AIResponder not initialized",
                timestamp=datetime.now()
            )
        
        # Check Ollama connection
        is_connected = ai_responder._check_ollama_connection()
        
        if is_connected:
            return LlmStatusResponse(
                connected=True,
                model=ai_responder.model_name,
                timestamp=datetime.now()
            )
        else:
            return LlmStatusResponse(
                connected=False,
                error="Ollama connection failed",
                timestamp=datetime.now()
            )
            
    except Exception as e:
        logger.error(f"LLM status check failed: {e}")
        return LlmStatusResponse(
            connected=False,
            error=str(e),
            timestamp=datetime.now()
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_query: SearchQuery):
    """
    Main search endpoint using AIResponder for RAG-based responses.
    
    Args:
        search_query: Search query with optional parameters
        
    Returns:
        AI-generated response with sources and metadata
    """
    try:
        if not ai_responder:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI Responder is not initialized"
            )
        
        logger.info(f"Processing search query: '{search_query.query}'")
        
        # Generate AI response
        response = ai_responder.respond(
            query=search_query.query,
            filter_metadata=search_query.filter_metadata
        )
        
        # Convert sources to response format
        source_responses = []
        for source in response.sources:
            source_response = SourceResponse(
                content=source.content,
                metadata=source.metadata,
                similarity_score=source.similarity_score,
                rank=source.rank
            )
            source_responses.append(source_response)
        
        # Create search response
        search_response = SearchResponse(
            answer=response.answer,
            sources=source_responses,
            query=response.query,
            response_time=response.response_time,
            confidence_score=response.confidence_score,
            model_used=response.model_used,
            timestamp=datetime.now()
        )
        
        logger.info(f"Search completed successfully in {response.response_time:.2f}s")
        return search_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    """
    Manually add documents to the knowledge base.
    
    Args:
        request: Document content and metadata
        
    Returns:
        Confirmation with document ID
    """
    try:
        if not document_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document Processor is not initialized"
            )
        
        logger.info(f"Adding document with ID: {request.doc_id or 'auto-generated'}")
        
        # Add document to knowledge base
        doc_id = document_processor.add_document(
            content=request.text,
            metadata=request.metadata,
            document_id=request.doc_id
        )
        
        response = AddDocumentResponse(
            doc_id=doc_id,
            message="Document added successfully",
            timestamp=datetime.now()
        )
        
        logger.info(f"Document added successfully with ID: {doc_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document: {str(e)}"
        )

@app.get("/stats", response_model=Dict[str, Any])
async def get_knowledge_base_stats():
    """
    Get statistics about the knowledge base.
    
    Returns:
        Knowledge base statistics
    """
    try:
        if not document_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document Processor is not initialized"
            )
        
        stats = document_processor.get_collection_stats()
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

# Chat Management Endpoints
@app.post("/api/save-chat", response_model=SaveChatResponse)
async def save_chat(request: SaveChatRequest):
    """
    Save a chat conversation to the backend.
    
    Args:
        request: Chat data to save
        
    Returns:
        SaveChatResponse with chat details
    """
    try:
        logger.info(f"Saving chat with {len(request.messages)} messages")
        response = chat_manager.save_chat(request)
        logger.info(f"Chat saved successfully with ID: {response.chat_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to save chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save chat: {str(e)}"
        )

@app.get("/api/chat-history", response_model=ChatListResponse)
async def get_chat_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """
    Get paginated chat history.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        user_id: Optional user ID filter
        
    Returns:
        ChatListResponse with paginated chat list
    """
    try:
        logger.info(f"Getting chat history - page {page}, size {page_size}")
        chats = chat_manager.get_chat_history(user_id=user_id, page=page, page_size=page_size)
        
        # Get total count for pagination
        all_chats = chat_manager.get_chat_history(user_id=user_id, page=1, page_size=1000)
        total_count = len(all_chats)
        
        has_more = (page * page_size) < total_count
        
        return ChatListResponse(
            chats=chats,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat history: {str(e)}"
        )

@app.get("/api/chat/{chat_id}")
async def get_chat(chat_id: str):
    """
    Get a specific chat by ID.
    
    Args:
        chat_id: The chat ID to retrieve
        
    Returns:
        Chat session data
    """
    try:
        logger.info(f"Getting chat: {chat_id}")
        chat = chat_manager.load_chat(chat_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat with ID {chat_id} not found"
            )
        
        return chat
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat: {str(e)}"
        )

@app.delete("/api/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Delete a chat by ID.
    
    Args:
        chat_id: The chat ID to delete
        
    Returns:
        Success message
    """
    try:
        logger.info(f"Deleting chat: {chat_id}")
        success = chat_manager.delete_chat(chat_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat with ID {chat_id} not found"
            )
        
        return {"message": f"Chat {chat_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat: {str(e)}"
        )

@app.put("/api/chat/{chat_id}/title")
async def update_chat_title(chat_id: str, request: UpdateChatTitleRequest):
    """
    Update the title of a chat.
    
    Args:
        chat_id: The chat ID to update
        request: New title data
        
    Returns:
        Success message
    """
    try:
        logger.info(f"Updating chat title: {chat_id} -> {request.title}")
        success = chat_manager.update_chat_title(chat_id, request.title)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat with ID {chat_id} not found"
            )
        
        return {"message": f"Chat title updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update chat title {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update chat title: {str(e)}"
        )

@app.get("/api/chat/search", response_model=ChatSearchResponse)
async def search_chats(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """
    Search through chat history.
    
    Args:
        q: Search query
        limit: Maximum number of results
        user_id: Optional user ID filter
        
    Returns:
        ChatSearchResponse with search results
    """
    try:
        logger.info(f"Searching chats for: '{q}' (limit: {limit})")
        results = chat_manager.search_chats(q, user_id=user_id, limit=limit)
        
        return ChatSearchResponse(
            results=results,
            total_found=len(results),
            query=q
        )
        
    except Exception as e:
        logger.error(f"Failed to search chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search chats: {str(e)}"
        )

@app.get("/api/chat/stats")
async def get_chat_stats():
    """
    Get chat storage statistics.
    
    Returns:
        Chat storage statistics
    """
    try:
        logger.info("Getting chat stats")
        stats = chat_manager.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get chat stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat stats: {str(e)}"
        )

@app.post("/api/chat/cleanup")
async def cleanup_old_chats():
    """
    Clean up old chats based on retention policy.
    
    Returns:
        Cleanup results
    """
    try:
        logger.info("Starting chat cleanup")
        deleted_count = chat_manager.cleanup_old_chats()
        
        return {
            "message": f"Cleanup completed",
            "deleted_directories": deleted_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup chats: {str(e)}"
        )

@app.post("/api/force-resync")
async def force_resync():
    """
    Force a complete resync of Confluence data.
    This will clear all existing data and perform a fresh sync.
    
    Returns:
        Resync status and results
    """
    try:
        logger.info("🔄 Starting force resync via API")
        
        # Import here to avoid circular imports
        from force_resync import main as force_resync_main
        
        # Run the force resync
        success = force_resync_main()
        
        if success:
            return {
                "success": True,
                "message": "Force resync completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Force resync failed"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Force resync failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Force resync failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured error responses."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with structured error responses."""
    logger.error(f"Unhandled exception: {exc}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

def start_server():
    """Start the FastAPI server with uvicorn."""
    logger.info("Starting Picarro SearchAI API server...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server() 