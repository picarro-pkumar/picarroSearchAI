import json
import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from difflib import SequenceMatcher

from models.chat_models import (
    ChatSession, ChatMessage, MessageRole, Source,
    SaveChatRequest, SaveChatResponse, ChatSearchResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self, storage_dir: str = "chat_history", retention_days: int = 90):
        self.storage_dir = Path(storage_dir)
        self.retention_days = retention_days
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"ChatManager initialized with storage dir: {self.storage_dir}")
        
    def _get_date_dir(self, date: datetime) -> Path:
        """Get the directory for a specific date"""
        date_str = date.strftime("%Y-%m-%d")
        date_dir = self.storage_dir / date_str
        date_dir.mkdir(exist_ok=True)
        return date_dir
    
    def _generate_chat_id(self) -> str:
        """Generate a unique chat ID"""
        return str(uuid.uuid4())
    
    def _get_chat_file_path(self, chat_id: str, date: datetime) -> Path:
        """Get the file path for a specific chat"""
        date_dir = self._get_date_dir(date)
        return date_dir / f"{chat_id}.json"
    
    def _save_chat_to_file(self, chat: ChatSession, file_path: Path) -> None:
        """Save a chat session to a JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chat.dict(), f, indent=2, default=str)
            logger.info(f"Chat saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving chat to {file_path}: {e}")
            raise
    
    def _load_chat_from_file(self, file_path: Path) -> Optional[ChatSession]:
        """Load a chat session from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert string timestamps back to datetime
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                for msg in data['messages']:
                    msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                return ChatSession(**data)
        except Exception as e:
            logger.error(f"Error loading chat from {file_path}: {e}")
            return None
    
    def _generate_title_from_messages(self, messages: List[ChatMessage]) -> str:
        """Generate a title from the first user message"""
        for msg in messages:
            if msg.role == MessageRole.USER:
                content = msg.content.strip()
                if len(content) > 50:
                    return content[:47] + "..."
                return content
        return "New Chat"
    
    def save_chat(self, request: SaveChatRequest) -> SaveChatResponse:
        """Save a chat session"""
        try:
            # Generate chat ID if not provided
            chat_id = request.chat_id or self._generate_chat_id()
            
            # Generate title if not provided
            title = request.title or self._generate_title_from_messages(request.messages)
            
            # Create or update chat session
            now = datetime.now()
            chat = ChatSession(
                id=chat_id,
                title=title,
                messages=request.messages,
                created_at=now,
                updated_at=now,
                user_id=request.user_id
            )
            
            # Save to file
            file_path = self._get_chat_file_path(chat_id, now)
            self._save_chat_to_file(chat, file_path)
            
            return SaveChatResponse(
                chat_id=chat_id,
                title=title,
                created_at=now,
                updated_at=now,
                message_count=len(request.messages)
            )
            
        except Exception as e:
            logger.error(f"Error saving chat: {e}")
            raise
    
    def load_chat(self, chat_id: str) -> Optional[ChatSession]:
        """Load a specific chat by ID"""
        try:
            # Search through all date directories
            for date_dir in self.storage_dir.iterdir():
                if date_dir.is_dir():
                    chat_file = date_dir / f"{chat_id}.json"
                    if chat_file.exists():
                        return self._load_chat_from_file(chat_file)
            return None
        except Exception as e:
            logger.error(f"Error loading chat {chat_id}: {e}")
            return None
    
    def get_chat_history(self, user_id: Optional[str] = None, 
                        page: int = 1, page_size: int = 50) -> List[ChatSession]:
        """Get chat history with pagination"""
        try:
            all_chats = []
            
            # Collect all chat files
            for date_dir in sorted(self.storage_dir.iterdir(), reverse=True):
                if date_dir.is_dir():
                    for chat_file in date_dir.glob("*.json"):
                        chat = self._load_chat_from_file(chat_file)
                        if chat and (user_id is None or chat.user_id == user_id):
                            all_chats.append(chat)
            
            # Sort by updated_at (most recent first)
            all_chats.sort(key=lambda x: x.updated_at, reverse=True)
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            return all_chats[start_idx:end_idx]
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat session"""
        try:
            # Find and delete the chat file
            for date_dir in self.storage_dir.iterdir():
                if date_dir.is_dir():
                    chat_file = date_dir / f"{chat_id}.json"
                    if chat_file.exists():
                        chat_file.unlink()
                        logger.info(f"Chat {chat_id} deleted")
                        return True
            return False
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
            return False
    
    def update_chat_title(self, chat_id: str, new_title: str) -> bool:
        """Update the title of a chat"""
        try:
            chat = self.load_chat(chat_id)
            if not chat:
                return False
            
            chat.title = new_title
            chat.updated_at = datetime.now()
            
            # Save updated chat
            file_path = self._get_chat_file_path(chat_id, chat.created_at)
            self._save_chat_to_file(chat, file_path)
            
            logger.info(f"Chat {chat_id} title updated to: {new_title}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating chat title {chat_id}: {e}")
            return False
    
    def search_chats(self, query: str, user_id: Optional[str] = None, 
                    limit: int = 10) -> List[ChatSearchResult]:
        """Search through chat history"""
        try:
            results = []
            query_lower = query.lower()
            
            # Search through all chats
            for date_dir in self.storage_dir.iterdir():
                if date_dir.is_dir():
                    for chat_file in date_dir.glob("*.json"):
                        chat = self._load_chat_from_file(chat_file)
                        if not chat or (user_id and chat.user_id != user_id):
                            continue
                        
                        # Search in title and messages
                        title_score = SequenceMatcher(None, query_lower, 
                                                     chat.title.lower()).ratio()
                        
                        best_message_score = 0
                        best_message = ""
                        
                        for msg in chat.messages:
                            if msg.role == MessageRole.USER:
                                content_lower = msg.content.lower()
                                score = SequenceMatcher(None, query_lower, 
                                                       content_lower).ratio()
                                if score > best_message_score:
                                    best_message_score = score
                                    best_message = msg.content
                        
                        # Use the best score
                        best_score = max(title_score, best_message_score)
                        
                        if best_score > 0.3:  # Threshold for relevance
                            results.append(ChatSearchResult(
                                chat_id=chat.id,
                                title=chat.title,
                                matched_message=best_message or chat.title,
                                similarity_score=best_score,
                                created_at=chat.created_at,
                                updated_at=chat.updated_at
                            ))
            
            # Sort by similarity score and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching chats: {e}")
            return []
    
    def cleanup_old_chats(self) -> int:
        """Clean up chats older than retention_days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            deleted_count = 0
            
            for date_dir in self.storage_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        # Remove entire directory
                        for file in date_dir.iterdir():
                            file.unlink()
                        date_dir.rmdir()
                        deleted_count += 1
                        logger.info(f"Cleaned up old chat directory: {date_dir}")
                except ValueError:
                    # Skip directories that don't match date format
                    continue
            
            logger.info(f"Cleanup completed: {deleted_count} old directories removed")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat storage statistics"""
        try:
            total_chats = 0
            total_messages = 0
            date_counts = {}
            
            for date_dir in self.storage_dir.iterdir():
                if date_dir.is_dir():
                    date_chats = 0
                    for chat_file in date_dir.glob("*.json"):
                        chat = self._load_chat_from_file(chat_file)
                        if chat:
                            total_chats += 1
                            total_messages += len(chat.messages)
                            date_chats += 1
                    
                    if date_chats > 0:
                        date_counts[date_dir.name] = date_chats
            
            return {
                "total_chats": total_chats,
                "total_messages": total_messages,
                "date_distribution": date_counts,
                "storage_directory": str(self.storage_dir),
                "retention_days": self.retention_days
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# Global chat manager instance
chat_manager = ChatManager() 