from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class Source(BaseModel):
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatMessage(BaseModel):
    id: str
    content: str
    role: MessageRole
    timestamp: datetime
    sources: Optional[List[Source]] = None
    isTyping: Optional[bool] = False

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatHistory(BaseModel):
    chats: List[ChatSession]
    total_count: int
    page: int = 1
    page_size: int = 50

class SaveChatRequest(BaseModel):
    chat_id: Optional[str] = None
    title: Optional[str] = None
    messages: List[ChatMessage]
    user_id: Optional[str] = None

class SaveChatResponse(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class ChatSearchRequest(BaseModel):
    query: str
    limit: int = 10
    user_id: Optional[str] = None

class ChatSearchResult(BaseModel):
    chat_id: str
    title: str
    matched_message: str
    similarity_score: float
    created_at: datetime
    updated_at: datetime

class ChatSearchResponse(BaseModel):
    results: List[ChatSearchResult]
    total_found: int
    query: str

class UpdateChatTitleRequest(BaseModel):
    title: str

class ChatListResponse(BaseModel):
    chats: List[ChatSession]
    total_count: int
    page: int
    page_size: int
    has_more: bool 