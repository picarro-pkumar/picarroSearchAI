import logging
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import os
import re

from doc_processor import DocumentProcessor

logger = logging.getLogger(__name__)

@dataclass
class Source:
    """Represents a source document used in the response."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int

@dataclass
class AIResponse:
    """Response from the AI system."""
    answer: str
    sources: List[Source]
    query: str
    model_used: str
    response_time: float
    confidence_score: Optional[float] = None

class SimpleAIResponder:
    """Simple AI Responder that actually works without strict validation."""
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        ollama_url: str = None,
        model_name: str = "llama3:latest",
        max_retrieved_docs: int = 10,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        timeout: int = 120
    ):
        self.document_processor = document_processor
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.model_name = model_name
        self.max_retrieved_docs = max_retrieved_docs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def _is_harmful_query(self, query: str) -> bool:
        """Check if query contains harmful, inappropriate, or offensive content."""
        query_lower = query.lower().strip()
        
        # Harmful/violent content
        harmful_patterns = [
            r'\b(kill|murder|suicide|die|death|harm|hurt|violence|weapon|gun|bomb|attack)\s+(me|you|us|him|her|them|someone|anybody)\b',
            r'\bcan\s+you\s+(kill|murder|hurt|harm|attack)\b',
            r'\bhow\s+to\s+(kill|murder|make\s+bomb|hurt|harm|suicide)\b',
            r'\b(i\s+want\s+to\s+die|kill\s+myself|end\s+my\s+life)\b',
            r'\b(hate|racist|nazi|terrorist|abuse|torture|rape)\b',
            r'\b(shoot|stab|poison|strangle|suffocate)\s+(me|you|us|him|her|them)\b',
            r'\bkill\s+me\b',
            r'\bhurt\s+me\b'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"🚫 Blocked harmful query: {query[:50]}...")
                return True
        
        return False
    
    def _is_irrelevant_query(self, query: str) -> bool:
        """Check if query is irrelevant to technical/API documentation."""
        query_lower = query.lower().strip()
        
        # Personal/irrelevant content
        irrelevant_patterns = [
            r'\b(what\s+is\s+your\s+name|who\s+are\s+you|tell\s+me\s+about\s+yourself)\b',
            r'\b(weather|climate|sports|entertainment|movies|music|games)\b',
            r'\b(cooking|recipe|food|restaurant|travel|vacation)\b',
            r'\b(personal\s+life|family|relationship|dating|marriage)\b',
            r'\b(joke|funny|meme|story|poem|creative\s+writing)\b',
            r'\b(hello|hi|hey|good\s+morning|good\s+afternoon)\b.*\??\s*$'  # Simple greetings without technical context
        ]
        
        # However, allow greetings if they're followed by technical questions
        if re.search(r'\b(hello|hi|hey)\b.*\b(api|tenant|site|endpoint|documentation)\b', query_lower):
            return False
        
        for pattern in irrelevant_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"ℹ️ Query appears irrelevant to technical documentation: {query[:50]}...")
                return True
        
        return False
    
    def _get_safety_response(self, query: str) -> str:
        """Generate appropriate response for harmful queries."""
        return "I'm designed to help with technical documentation and API questions. I can't assist with harmful, inappropriate, or dangerous requests. Please ask about APIs, technical documentation, or system features instead."
    
    def _get_irrelevant_response(self, query: str) -> str:
        """Generate appropriate response for irrelevant queries."""
        return "I'm a technical documentation assistant focused on API and system information. I can help you with questions about tenant management, API endpoints, site configuration, and other technical topics from the documentation. How can I assist you with technical information?"
    
    def _add_hallucination_warning(self, answer: str, sources: List[Source]) -> str:
        """Add warning if response likely contains hallucinated information."""
        # Check for signs of hallucination
        hallucination_indicators = [
            'for example',
            'might look like this',
            'sample response',
            'example response',
            'here\'s an example',
            'example:',
            '"123e4567-e89b',  # UUID examples
            '"sample program"',
            '"2022-01-01',  # Date examples
            'replace with your actual',
            'replace "https://',
            'example.com',
            'your-api-url'
        ]
        
        answer_lower = answer.lower()
        has_hallucination_signs = any(indicator in answer_lower for indicator in hallucination_indicators)
        
        if has_hallucination_signs:
            logger.warning("⚠️ Detected potential hallucination in response")
            warning = "\n\n⚠️ **Note:** The response above may contain examples or details not explicitly stated in the documentation. Please verify specific details like response formats, example data, and exact parameter values with the official API documentation."
            return answer + warning
        
        return answer
    
    def _enhance_query_with_context(self, query: str, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """Enhance query with conversation context for better understanding."""
        if not conversation_history or len(conversation_history) < 2:
            return query
        
        # Get recent conversation (last 4 messages to avoid context overload)
        recent_messages = conversation_history[-4:]
        
        # Check if current query is a follow-up (contains words like "this", "that", "it", "explain", "details")
        follow_up_indicators = ['this', 'that', 'it', 'explain', 'details', 'how', 'what about', 'tell me more']
        is_follow_up = any(indicator in query.lower() for indicator in follow_up_indicators)
        
        if not is_follow_up:
            return query
        
        # Find the most recent assistant response that mentioned specific topics
        context_info = []
        for msg in reversed(recent_messages):
            if msg.get('role') == 'assistant' and msg.get('content'):
                content = msg['content'].lower()
                # Extract API endpoints, technical terms
                if '/v1/' in content or 'api' in content or 'endpoint' in content:
                    # Find the main topic from assistant's response
                    lines = content.split('\n')
                    for line in lines:
                        if 'endpoint:' in line.lower() or '/v1/' in line:
                            context_info.append(line.strip())
                            break
                    break
        
        if context_info:
            enhanced_query = f"{query} (referring to: {context_info[0]})"
            return enhanced_query
        
        return query
    
    def _generate_alternative_query(self, query: str) -> str:
        """Generate alternative search terms for poor-quality results."""
        query_lower = query.lower()
        
        # FOV-specific alternatives
        if 'fov' in query_lower or 'field of view' in query_lower:
            return "FOVs gas dispersion wind direction visual representation"
        
        # API-specific alternatives
        if 'api' in query_lower and 'endpoint' in query_lower:
            # Extract key terms and add 'method response'
            key_terms = query_lower.replace('api', '').replace('endpoint', '').strip()
            return f"{key_terms} method response parameters"
        
        # Monitoring-specific alternatives
        if any(term in query_lower for term in ['monitoring', 'system', 'status']):
            key_terms = query_lower.replace('monitoring', '').replace('system', '').strip()
            return f"{key_terms} data status operational"
        
        # General fallback - add related technical terms
        return f"{query} functionality implementation details"
    
    def respond(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> AIResponse:
        """Generate AI response with safety and relevance filtering."""
        start_time = time.time()
        
        # Enhance query with conversation context if available
        enhanced_query = self._enhance_query_with_context(query, conversation_history)
        logger.info(f"🔍 Processing query: '{query}'")
        if enhanced_query != query:
            logger.info(f"🔗 Enhanced with context: '{enhanced_query}'")
        
        # Safety check - block harmful content
        if self._is_harmful_query(query):
            logger.warning(f"🚫 Blocked harmful query")
            response = AIResponse(
                answer=self._get_safety_response(query),
                sources=[],
                query=query,
                model_used=self.model_name,
                response_time=time.time() - start_time,
                confidence_score=0.0
            )
            return response
        
        # Relevance check - redirect irrelevant queries
        if self._is_irrelevant_query(query):
            logger.info(f"ℹ️ Redirecting irrelevant query")
            response = AIResponse(
                answer=self._get_irrelevant_response(query),
                sources=[],
                query=query,
                model_used=self.model_name,
                response_time=time.time() - start_time,
                confidence_score=0.0
            )
            return response
        
        # Check Ollama connection
        if not self._check_ollama_connection():
            logger.error("❌ Ollama connection failed")
            raise Exception("Ollama is not accessible. Please ensure Ollama is running.")
        
        # Get relevant documents using enhanced query
        search_results = self.document_processor.search_documents(
            query=enhanced_query,
            n_results=self.max_retrieved_docs,
            filter_metadata=filter_metadata
        )
        
        logger.info(f"📚 Found {len(search_results)} relevant sources")
        
        # Convert to Source objects
        sources = []
        for i, result in enumerate(search_results, 1):
            source = Source(
                content=result['content'],
                metadata=result['metadata'],
                similarity_score=result['similarity_score'],
                rank=i
            )
            sources.append(source)
        
        # Build simple prompt using original query for display, enhanced query for context
        context = self._build_context(sources)
        prompt = self._build_simple_prompt(enhanced_query, context)
        
        # Call Ollama
        logger.info("🤖 Generating AI response...")
        answer = self._call_ollama(prompt)
        
        # Post-process to detect and warn about potential hallucination
        answer = self._add_hallucination_warning(answer, sources)
        
        # Calculate confidence
        confidence_score = 0.0
        if sources:
            confidence_score = sum(s.similarity_score for s in sources) / len(sources)
        
        response = AIResponse(
            answer=answer,
            sources=sources,
            query=query,
            model_used=self.model_name,
            response_time=time.time() - start_time,
            confidence_score=confidence_score
        )
        
        logger.info(f"✅ Response generated in {response.response_time:.2f}s")
        return response
    
    def _build_context(self, sources: List[Source]) -> str:
        """Build context from sources."""
        if not sources:
            return "No relevant sources found."
        
        context_parts = []
        for source in sources:
            title = source.metadata.get('title', 'Unknown Document')
            context_parts.append(f"## Source: {title}\n{source.content}\n")
        
        return "\n".join(context_parts)
    
    def _build_simple_prompt(self, query: str, context: str) -> str:
        """Build a simple, effective prompt that prevents hallucination."""
        return f"""You are a documentation reader. Extract and present ONLY the information that exists in the provided context.

DOCUMENTATION CONTEXT:
{context}

USER QUESTION: {query}

RESPONSE TEMPLATE - Fill in ONLY from the context above:

Based on the documentation provided, here is the information about "{query}":

[Extract the exact relevant information from the context. Do NOT add examples, sample data, or explanations not in the context]

If the documentation is incomplete, state: "The documentation shows [what exists] but does not include [what is missing, such as response format, examples, etc.]"

IMPORTANT: Do not create any examples, sample JSON, or code snippets unless they appear exactly in the context above."""
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "I apologize, but I couldn't generate a response.")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I apologize, but there was an error generating the response."
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return "I apologize, but there was an error generating the response."