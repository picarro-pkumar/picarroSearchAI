import logging
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import os
import re

from doc_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detects potential hallucination patterns in AI responses.
    """
    
    def __init__(self):
        # Warning patterns that indicate potential hallucination
        self.warning_patterns = {
            # JSON examples not from sources - more specific patterns
            'json_examples': [
                r'```json\s*\n\s*\{[^{}]*"[^"]*":\s*"[^"]*"[^{}]*\}\s*\n```',
                r'"id":\s*"123"',
                r'"id":\s*"456"',
                r'"name":\s*"example"',
                r'"description":\s*"This is an example"',
            ],
            
            # REST API endpoints without source evidence
            'rest_endpoints': [
                r'GET\s+/api/v1/\w+',
                r'POST\s+/api/v1/\w+',
                r'PUT\s+/api/v1/\w+',
                r'DELETE\s+/api/v1/\w+',
                r'/\w+/\{id\}/',
                r'/\w+/\d{3,}',
            ],
            
            # Placeholder values
            'placeholder_values': [
                r'\b123\b',
                r'\b456\b',
                r'\b789\b',
                r'example\.com',
                r'api\.example\.com',
                r'your-domain\.com',
                r'your-email@company\.com',
                r'your-api-token',
                r'<your-',
                r'\[your-',
            ],
            
            # Generic API patterns
            'generic_api_patterns': [
                r'Request\s+Parameters:\s*\n',
                r'Response\s+Format:\s*\n',
                r'Example\s+Response:\s*\n',
                r'Authentication\s+Requirements:\s*\n',
                r'Headers:\s*\n',
                r'Content-Type:\s*application/json\s*\n',
                r'Authorization:\s*Bearer\s*\n',
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.warning_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def detect_hallucination(self, response_text: str, sources: List[Any]) -> Tuple[bool, List[str], Dict[str, List[str]]]:
        """
        Detect potential hallucination patterns in the response.
        """
        detected_patterns = {}
        warning_messages = []
        is_hallucination = False
        
        # Check each category of patterns
        for category, patterns in self.compiled_patterns.items():
            category_matches = []
            for pattern in patterns:
                matches = pattern.findall(response_text)
                if matches:
                    category_matches.extend(matches)
            
            if category_matches:
                detected_patterns[category] = category_matches
                logger.debug(f"Hallucination detection: Found {category} patterns: {category_matches}")
                
                # Check if these patterns are actually supported by sources
                if not self._patterns_supported_by_sources(category_matches, sources):
                    is_hallucination = True
                    warning_messages.append(f"Potential hallucination detected: {category} patterns found without source support")
                    logger.warning(f"HALLUCINATION: {category} patterns not supported by sources: {category_matches}")
        
        return is_hallucination, warning_messages, detected_patterns
    
    def _patterns_supported_by_sources(self, patterns: List[str], sources: List[Any]) -> bool:
        """Check if detected patterns are actually supported by source documents."""
        if not sources:
            return False
        
        source_content = " ".join([getattr(source, 'content', str(source)) for source in sources])
        source_content_lower = source_content.lower()
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in source_content_lower:
                return True
        
        return False


@dataclass
class Source:
    """Represents a source document used in the response."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int


@dataclass
class AIResponse:
    """Structured response from the AI system."""
    answer: str
    sources: List[Source]
    query: str
    model_used: str
    response_time: float
    confidence_score: Optional[float] = None


class AIResponder:
    """
    Demo-Safe AI Responder with anti-hallucination protection.
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        ollama_url: str = None,
        model_name: str = "llama3.2:3b",
        max_retrieved_docs: int = 10,
        max_tokens: int = 1024,  # Reduced for demo safety
        temperature: float = 0.0,  # No creativity for demos
        timeout: int = 120,
        min_sources_required: int = 1
    ):
        """Initialize with demo-safe settings."""
        self.document_processor = document_processor
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.model_name = model_name
        self.max_retrieved_docs = max_retrieved_docs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.min_sources_required = min_sources_required
        
        # Initialize hallucination detector
        self.hallucination_detector = HallucinationDetector()
        
        logger.info(f"AIResponder initialized with DEMO-SAFE settings")
        logger.info(f"Temperature: {temperature}, Max tokens: {max_tokens}")
        logger.info(f"Ollama URL: {self.ollama_url}")
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama connection successful")
                return True
            else:
                logger.error(f"Ollama connection failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            return False
    
    def _check_ethical_content(self, query: str) -> Tuple[bool, str]:
        """Check if the query contains unethical or inappropriate content."""
        query_lower = query.lower()
        
        # Harmful keywords check
        harmful_keywords = [
            'kill', 'harm', 'attack', 'hurt', 'injure', 'destroy', 'sabotage',
            'suicide', 'self-harm', 'violence', 'assault', 'bomb', 'poison',
            'explosive', 'weapon', 'murder', 'abuse'
        ]
        
        for word in harmful_keywords:
            if word in query_lower:
                logger.warning(f"Ethical content filter triggered: harmful keyword '{word}' detected.")
                return False, f"Harmful keyword detected: '{word}'"
        
        return True, "Query passed ethical content filter"
    
    def _check_domain_relevance(self, query: str, sources: List[Source]) -> bool:
        """Check if the query is relevant to the domain."""
        domain_keywords = [
            'fenceline', 'architecture', 'monitoring', 'dashboard', 'api', 'websocket',
            'system', 'platform', 'service', 'database', 'frontend', 'backend',
            'tenant', 'site', 'user', 'data', 'analytics', 'reporting', 'alerts',
            'picarro', 'telemetry', 'concentration', 'operational', 'heartbeat'
        ]
        
        query_lower = query.lower()
        has_domain_keywords = any(keyword in query_lower for keyword in domain_keywords)
        
        # Check if sources have sufficient relevance
        relevant_sources = [s for s in sources if s.similarity_score > 0.3]
        has_relevant_sources = len(relevant_sources) > 0
        
        return has_domain_keywords or has_relevant_sources

    def _deduplicate_sources_by_document_id(self, sources: List[Source]) -> List[Source]:
        """Deduplicate sources by document ID, keeping the highest scoring chunk."""
        if not sources:
            return sources
        
        doc_groups = {}
        for source in sources:
            doc_id = source.metadata.get('document_id', 'unknown')
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(source)
        
        deduplicated_sources = []
        for doc_id, doc_sources in doc_groups.items():
            if doc_sources:
                best_source = max(doc_sources, key=lambda s: s.similarity_score)
                deduplicated_sources.append(best_source)
        
        deduplicated_sources.sort(key=lambda s: s.similarity_score, reverse=True)
        
        for i, source in enumerate(deduplicated_sources):
            source.rank = i + 1
        
        logger.info(f"Deduplicated sources: {len(sources)} -> {len(deduplicated_sources)} unique documents")
        return deduplicated_sources

    def get_demo_response(self, query: str) -> Optional[str]:
        """Get pre-written safe responses for critical demo questions."""
        DEMO_RESPONSES = {
            "websocket details": """# WebSocket Configuration for Active Monitoring Dashboard

Based on the documentation:

## Connection Setup
- **API Endpoint**: `/api/v1/webSocket`
- **Method**: Request WebSocket connection details via GET

## Namespace Configuration  
- **Namespace**: `active-monitoring-namespace`
- **Protocol**: Socket.IO

## Connection Process
1. Request connection details from `/api/v1/webSocket`
2. Establish Socket.IO connection
3. Connect to `active-monitoring-namespace`
4. Send `start_active_monitoring` event with siteProgramId

## Event Types
- **`rt_processed_conc_data`**: Real-time concentration data with wind polygon
- **`ms_heartbeat_state`**: Heartbeat monitoring (every 15 seconds)
- **`ms_operational_state`**: Operational status updates
- **`rt_wind_data`**: Wind data updates (every 5 seconds)

## Required Parameters
- **siteProgramId**: UUID (required for all operations)

Source: Active Monitoring Dashboard Documentation""",
            
            "api endpoints": """# API Endpoints

Based on the documentation:

## Telemetry Endpoint
- **Path**: `/v1/telemetry/latest`
- **Method**: GET
- **Parameter**: `siteProgramId` (UUID, required)
- **Description**: Get latest telemetry info for Active Monitoring Dashboard

## WebSocket Connection
- **Path**: `/api/v1/webSocket`
- **Purpose**: Request WebSocket connection details
- **Returns**: Connection specifications for Socket.IO

## Alert Dashboard
- **Path**: `/v1/alerts/`
- **Method**: GET
- **Parameters**: category, site_program_id, ms_id, start, end, type, status

Source: API Documentation""",
            
            "monitoring dashboard": """# Active Monitoring Dashboard

Based on the documentation:

## Features
- **Real-time gas concentration monitoring** (15-minute averages)
- **Operational health monitoring** with status indicators
- **System connectivity tracking** via heartbeat signals
- **Visual indicators** with color-coded status
- **Field of Views (FOVs)** for gas dispersion visualization
- **Wind data integration** (updated every 5 seconds)

## Visual Status Indicators
- **Red**: Concentration exceedance detected
- **Orange**: 75% of exceedance threshold
- **Green**: Safe levels (below 75% of threshold)
- **Grey**: No data received

## System Monitoring
- **Heartbeat**: Every 15 seconds connectivity check
- **Operational Status**: Real-time health monitoring
- **System Down**: Triggered when heartbeat fails

## Data Updates
- **Gas Concentrations**: 15-minute averages
- **Wind Data**: Every 5 seconds
- **Heartbeat**: Every 15 seconds

Source: Active Monitoring Dashboard Specification""",
            
            "tenant management": """# Tenant Management

Based on the documentation:

## Available Information
The documentation references tenant management in the context of:
- **Site Program Management**: Using `siteProgramId` for operations
- **Site Boundary Configuration**: Configured as part of tenant and site setup
- **Alert Filtering**: By site program identifier

## API Integration
- Most endpoints require `site_program_id` parameter
- Tenant context is maintained through site program identifiers

For complete tenant management API details, please consult the full API documentation.

Source: API Documentation References"""
        }
        
        query_lower = query.lower()
        for key, response in DEMO_RESPONSES.items():
            if key in query_lower:
                return response.strip()
        return None

    def _generate_prompt(self, query: str, sources: List[Source]) -> str:
        """Generate demo-safe prompt that extracts rich information from sources."""
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"Source {i}:\n{source.content}\n")
        context = "\n".join(context_parts)
        
        prompt = f"""You are a senior technical expert. Extract and present comprehensive technical information from the sources.

RESPONSE GUIDELINES:
1. **When sources contain technical information about the query:**
   - Extract ALL relevant API endpoints, methods, and parameters
   - Include complete request/response schemas and data structures
   - Present technical specifications in organized sections
   - Use professional technical formatting with headers and bullet points
   - Include exact field names, data types, and validation rules
   - Reference specific error codes and response formats

2. **When sources don't contain information about the requested topic:**
   - Clearly state what information is not available
   - List what topics the sources DO cover
   - Suggest consulting additional documentation

3. **When sources contain partial information:**
   - Present what information IS available comprehensively
   - Clearly indicate what details are missing
   - Organize available information professionally

TECHNICAL FORMATTING:
- Use clear section headers (##, ###)
- Present API endpoints with method and path
- Include parameter tables when applicable
- Show request/response examples from sources
- Use bullet points for lists and specifications
- Present JSON schemas exactly as documented

STRICT RULES:
- ONLY use information explicitly stated in the sources
- Extract ALL available technical details comprehensively
- NEVER add information not in sources
- NEVER create fake examples or endpoints
- Present information in a professional, technical format

SOURCES:
{context}

USER QUESTION: {query}

Extract and present ALL relevant technical information from the sources in a comprehensive, professional format.

RESPONSE:"""
        
        return prompt

    def _validate_response_safety(self, response: str, sources: List[Source]) -> bool:
        """Ultra-strict validation to prevent hallucination."""
        if not sources:
            return True
        
        source_content = " ".join([s.content.lower() for s in sources])
        response_lower = response.lower()
        
        # Only check for fake endpoint patterns, not general technical terms
        fake_patterns = [
            r'/fencelines',  # Specific fake endpoints we want to catch
            r'/users/\{id\}',  # Generic REST patterns not in sources
            r'POST\s+/fake',  # Obviously fake endpoints
            r'example\.com',  # Placeholder domains
            r'"id":\s*"123"',  # Placeholder IDs
        ]
        
        for pattern in fake_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                if not re.search(pattern, source_content, re.IGNORECASE):
                    logger.warning(f"Potential fake pattern detected: {pattern}")
                    return False
        
        # Check for completely made-up endpoints (not in sources)
        response_endpoints = re.findall(r'(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s]+)', response)
        source_endpoints = re.findall(r'(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s]+)', source_content)
        
        for method, endpoint in response_endpoints:
            endpoint_in_source = any(endpoint.lower() in source_ep.lower() for _, source_ep in source_endpoints)
            if not endpoint_in_source and endpoint not in source_content:
                logger.warning(f"Endpoint not found in sources: {method} {endpoint}")
                return False
        
        return True

    def _validate_no_guessing(self, response: str) -> bool:
        """Validate that the response doesn't contain guessing phrases."""
        forbidden_phrases = [
            "educated guess", "general knowledge", "typically you would",
            "usually", "generally", "normally", "standard practice",
            "common approach", "typical implementation", "you would need to",
            "based on experience", "in most cases", "generally speaking"
        ]
        
        response_lower = response.lower()
        for phrase in forbidden_phrases:
            if phrase in response_lower:
                logger.warning(f"Response contains guessing phrase: '{phrase}'")
                return False
        return True
    
    def _call_ollama_api(self, prompt: str) -> Tuple[str, float]:
        """Call the Ollama API to generate a response."""
        start_time = time.time()
        
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
            
            logger.info(f"Sending request to Ollama with model: {self.model_name}")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                logger.info(f"Ollama response generated in {response_time:.2f}s")
                return generated_text.strip(), response_time
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to improve search matching."""
        processed_query = query.lower()
        
        # Add common variations for better matching
        variations = []
        if 'api' in processed_query:
            variations.extend(['API', 'endpoint', 'interface'])
        if 'websocket' in processed_query:
            variations.extend(['socket', 'socketio', 'connection'])
        if 'monitoring' in processed_query:
            variations.extend(['dashboard', 'system', 'telemetry'])
        
        if variations:
            processed_query = f"{processed_query} {' '.join(variations)}"
        
        logger.info(f"Query preprocessing: '{query}' -> '{processed_query}'")
        return processed_query
    
    def get_sources_with_cascade(self, search_results: List[dict], query: str) -> List[Source]:
        """Get sources using cascade threshold system."""
        thresholds = [0.4, 0.3, 0.2, 0.15, 0.1]
        for threshold in thresholds:
            sources = []
            for result in search_results:
                if result["similarity_score"] >= threshold:
                    source = Source(
                        content=result["content"],
                        metadata=result["metadata"],
                        similarity_score=result["similarity_score"],
                        rank=result["rank"]
                    )
                    sources.append(source)
            if len(sources) >= 1:
                logger.info(f"Using threshold {threshold} - found {len(sources)} sources")
                return sources
        
        # Fallback: return top 3 results
        logger.info("Using fallback: top 3 results")
        return [Source(content=r["content"], metadata=r["metadata"], 
                      similarity_score=r["similarity_score"], rank=r["rank"]) 
                for r in search_results[:3]]

    def respond(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None) -> AIResponse:
        """Generate AI response with demo-safe anti-hallucination."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Check Ollama connection
            if not self._check_ollama_connection():
                raise Exception("Ollama is not accessible. Please ensure Ollama is running.")
            
            # # DEMO SAFETY: Check for pre-written responses first
            # demo_response = self.get_demo_response(query)
            # if demo_response:
            #     logger.info("Using pre-written demo response")
            #     return AIResponse(
            #         answer=demo_response,
            #         sources=[],
            #         query=query,
            #         model_used=self.model_name,
            #         response_time=time.time() - start_time,
            #         confidence_score=1.0
            #     )
            
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Check ethical content
            is_ethical, ethical_reason = self._check_ethical_content(query)
            if not is_ethical:
                logger.warning(f"Ethical content filter rejected query: {ethical_reason}")
                return AIResponse(
                    answer="Your request cannot be processed.",
                    sources=[],
                    query=query,
                    model_used=self.model_name,
                    response_time=time.time() - start_time,
                    confidence_score=0.0
                )
            
            # Retrieve documents
            logger.info("Retrieving relevant documents...")
            search_results = self.document_processor.search_documents(
                query=processed_query,
                n_results=self.max_retrieved_docs,
                filter_metadata=filter_metadata
            )
            
            # Get sources with cascade system
            sources = self.get_sources_with_cascade(search_results, query)
            logger.info(f"Retrieved {len(sources)} relevant documents")
            
            # Check if we have enough sources
            if len(sources) < self.min_sources_required:
                return AIResponse(
                    answer="I don't have enough relevant information to provide a reliable answer. Please try rephrasing your question.",
                    sources=[],
                    query=query,
                    model_used=self.model_name,
                    response_time=time.time() - start_time,
                    confidence_score=0.0
                )
            
            # Check domain relevance
            is_relevant = self._check_domain_relevance(query, sources)
            if not is_relevant:
                return AIResponse(
                    answer="I can only answer questions related to the available documentation.",
                    sources=[],
                    query=query,
                    model_used=self.model_name,
                    response_time=time.time() - start_time,
                    confidence_score=0.0
                )
            
            # Deduplicate sources
            sources = self._deduplicate_sources_by_document_id(sources)
            
            # Generate prompt with STRICT template system
            prompt = self._generate_prompt(query, sources)
            
            # Generate AI response
            logger.info("Generating AI response with demo-safe system...")
            answer, api_response_time = self._call_ollama_api(prompt)
            
            # DEMO SAFETY: Multiple validation layers
            if not self._validate_response_safety(answer, sources):
                logger.warning("Response failed safety validation - using safe fallback")
                answer = "I can only provide information explicitly documented in the available sources. The documentation may not contain complete details for this specific question."
            
            if not self._validate_no_guessing(answer):
                logger.warning("Response contained guessing - using safe fallback")
                answer = "Based on the available documentation, I can provide limited information. For complete details, please consult additional documentation."
            
            # Calculate confidence
            confidence_score = sum(s.similarity_score for s in sources) / len(sources) if sources else 0.0
            
            # DEMO SAFETY: More permissive confidence threshold
            if confidence_score < 0.1:  # Much more permissive
                logger.warning(f"Low confidence ({confidence_score:.3f}) - using safe response")
                answer = "The available information has low relevance to your question. Please try a different question or check additional documentation."
                confidence_score = 0.0
            
            response = AIResponse(
                answer=answer,
                sources=sources,
                query=query,
                model_used=self.model_name,
                response_time=time.time() - start_time,
                confidence_score=confidence_score
            )
            
            logger.info(f"Demo-safe response generated in {response.response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # DEMO SAFETY: Safe fallback for any errors
            return AIResponse(
                answer="I encountered an issue processing your request. Please try rephrasing your question.",
                sources=[],
                query=query,
                model_used=self.model_name,
                response_time=time.time() - start_time,
                confidence_score=0.0
            )
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the knowledge base."""
        try:
            logger.info("Adding document to knowledge base...")
            doc_id = self.document_processor.add_document(content, metadata)
            logger.info(f"Document added successfully with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            return self.document_processor.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            raise


# Production-ready Demo-Safe AI Responder for Picarro SearchAI
# Enhanced with comprehensive anti-hallucination protection