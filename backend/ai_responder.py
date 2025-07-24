import logging
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from doc_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    """Structured response from the AI system."""
    answer: str
    sources: List[Source]
    query: str
    model_used: str
    response_time: float
    confidence_score: Optional[float] = None


class AIResponder:
    """
    AI Responder that implements RAG (Retrieval-Augmented Generation) pattern
    using DocumentProcessor for retrieval and Ollama for generation.
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        ollama_url: str = "http://localhost:11434/api/generate",
        model_name: str = "llama3:latest",
        max_retrieved_docs: int = 10,  # Retrieve more initially, then filter
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 30
    ):
        """
        Initialize the AI Responder.
        
        Args:
            document_processor: DocumentProcessor instance for retrieval
            ollama_url: URL of the Ollama API
            model_name: Name of the model to use for generation
            max_retrieved_docs: Maximum number of documents to retrieve
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation (0.0-1.0)
            timeout: Timeout for API requests in seconds
        """
        self.document_processor = document_processor
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.max_retrieved_docs = max_retrieved_docs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        logger.info(f"AIResponder initialized with model: {model_name}")
        logger.info(f"Ollama URL: {ollama_url}")
    
    def _check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            # Try to list models to check connection
            response = requests.get(
                f"{self.ollama_url.replace('/api/generate', '/api/tags')}",
                timeout=5
            )
            if response.status_code == 200:
                logger.info("Ollama connection successful")
                return True
            else:
                logger.error(f"Ollama connection failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            return False
    
    def _check_domain_relevance(self, query: str, sources: List[Source]) -> bool:
        """
        Check if the query is relevant to Picarro domain.
        
        Args:
            query: User query
            sources: Retrieved source documents
            
        Returns:
            True if query is relevant to Picarro domain
        """
        # Define comprehensive FenceLine domain keywords (matching current knowledge base content)
        domain_keywords = [
            # FenceLine-specific terms (what's actually in your data)
            'fenceline', 'fence line', 'architecture', 'system architecture', 'cloud solution',
            'demo', 'template', 'troubleshooting', 'how-to', 'guide', 'documentation',
            'system', 'design', 'solution', 'platform', 'service', 'api', 'database',
            'frontend', 'backend', 'deployment', 'configuration', 'setup', 'installation',
            
            # Additional FenceLine terms from your content
            'tenant', 'site', 'hierarchy', 'organization', 'structure', 'management',
            'user', 'role', 'permission', 'access', 'security', 'authentication',
            'dashboard', 'interface', 'ui', 'ux', 'workflow', 'process', 'integration',
            'data', 'analytics', 'reporting', 'monitoring', 'alerts', 'notifications',
            'cloud', 'aws', 'azure', 'gcp', 'infrastructure', 'scalability', 'performance',
            
            # General Picarro terms for compatibility
            'picarro', 'monitoring', 'analysis', 'reporting'
        ]
        
        # Check if query contains domain-related keywords
        query_lower = query.lower()
        has_domain_keywords = any(keyword in query_lower for keyword in domain_keywords)
        
        # Check if sources have sufficient relevance (similarity score > 0.5 - more flexible)
        relevant_sources = [s for s in sources if s.similarity_score > 0.5]
        has_relevant_sources = len(relevant_sources) > 0
        
        # Query is relevant if it has domain keywords OR has highly relevant sources
        return has_domain_keywords or has_relevant_sources

    def _generate_prompt(self, query: str, sources: List[Source]) -> str:
        """
        Generate a prompt for the AI model with retrieved sources.
        
        Args:
            query: User query
            sources: Retrieved source documents
            
        Returns:
            Formatted prompt string
        """
        # Create context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"Source {i} (Score: {source.similarity_score:.3f}):\n{source.content}\n")
        
        context = "\n".join(context_parts)
        
        # Create the prompt with ENHANCED specific analysis capabilities
        prompt = f"""You are an AI assistant for Picarro's FenceLine Cloud Solution and related documentation. 

IMPORTANT: You can answer questions related to:
- Picarro's products, technology, services, and environmental monitoring applications
- FenceLine architecture, system design, and cloud solutions
- Documentation, templates, troubleshooting guides, and how-to articles
- Any content present in the provided context

If the user asks about topics completely unrelated to Picarro or the provided context, politely decline and redirect them to relevant topics.

Context Information:
{context}

User Question: {query}

CRITICAL INSTRUCTIONS FOR RESPONSE QUALITY:

1. **ANALYZE SPECIFIC CONTENT**: 
   - DO NOT give generic responses. Analyze the ACTUAL content provided in the sources.
   - Extract and describe SPECIFIC details from the source content.
   - Quote or reference exact text from the sources when describing components.
   - List actual component names, labels, and technical specifications mentioned.

2. **FOR DIAGRAMS AND ARCHITECTURE**:
   - If the context contains diagrams, architecture, or visual content:
     * Describe the EXACT components, sections, and layers as they appear in the source
     * Explain the specific relationships and connections as shown in the source
     * Mention specific file names, URLs, or identifiers from the source
     * Include technical details like IP addresses, port numbers, or configurations if mentioned
     * Describe any labels, arrows, flow directions, or annotations exactly as they appear
     * If the source mentions specific diagram elements (like "Tenant - Site Hierarchy", "Platform API Architecture"), describe them in detail
     * Extract and explain any specific technical terms, component names, or architectural patterns mentioned
   - AVOID generic statements like "appears to be" or "seems to show" - be specific about what's actually in the source
   - If the source contains raw diagram data or metadata, interpret and explain the actual content rather than making assumptions

3. **RESPONSE STRUCTURE**:
   - Start with a direct answer to the user's question
   - Provide specific details from the sources
   - If describing architecture, break it down into clear sections
   - Use bullet points or numbered lists for clarity
   - Reference specific source information when possible
   - For FenceLine architecture questions, focus on the actual diagram content and technical specifications mentioned

4. **QUALITY REQUIREMENTS**:
   - Be precise and technical when the source contains technical information
   - Avoid vague or generic descriptions
   - Focus on what's actually documented in the sources
   - If the source contains specific technical details, include them in your response

5. **If the question is NOT related to Picarro, FenceLine, or the provided context**, respond with:
   "I'm sorry, but I can only answer questions related to Picarro's solutions and the documentation available. Your question appears to be outside my area of expertise. Please ask me about Picarro's products, FenceLine architecture, or related documentation."

Answer:"""
        
        return prompt
    
    def _call_ollama_api(self, prompt: str) -> Tuple[str, float]:
        """
        Call the Ollama API to generate a response.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Tuple of (response_text, response_time)
        """
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
                self.ollama_url,
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
                
        except requests.exceptions.Timeout:
            error_msg = f"Ollama API timeout after {self.timeout}s"
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API request failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Ollama response: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error calling Ollama API: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def respond(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None) -> AIResponse:
        """
        Generate an AI response using RAG pattern.
        
        Args:
            query: User query
            filter_metadata: Optional metadata filters for document retrieval
            
        Returns:
            AIResponse object with answer and sources
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Check Ollama connection
            if not self._check_ollama_connection():
                raise Exception("Ollama is not accessible. Please ensure Ollama is running.")
            
            # Retrieve relevant documents
            logger.info("Retrieving relevant documents...")
            search_results = self.document_processor.search_documents(
                query=query,
                n_results=self.max_retrieved_docs,
                filter_metadata=filter_metadata
            )
            
            # Convert search results to Source objects and filter by relevance
            sources = []
            min_similarity_threshold = 0.3  # Only include sources with decent relevance
            
            for result in search_results:
                similarity_score = result["similarity_score"]
                
                # Only include sources that meet the minimum similarity threshold
                if similarity_score >= min_similarity_threshold:
                    source = Source(
                        content=result["content"],
                        metadata=result["metadata"],
                        similarity_score=similarity_score,
                        rank=result["rank"]
                    )
                    sources.append(source)
                else:
                    logger.info(f"Filtering out low-relevance source (score: {similarity_score:.3f}): {result['metadata'].get('title', 'Unknown')}")
            
            logger.info(f"Retrieved {len(sources)} relevant documents (filtered from {len(search_results)} total)")
            
            logger.info(f"Retrieved {len(sources)} relevant documents")
            
            # Check domain relevance
            is_relevant = self._check_domain_relevance(query, sources)
            logger.info(f"Domain relevance check: {is_relevant}")
            
            if not is_relevant:
                logger.info("Query is not relevant to Picarro domain - refusing to answer")
                return AIResponse(
                    answer="I'm sorry, but I can only answer questions related to Picarro's precision gas analyzers and environmental monitoring solutions. Your question appears to be outside my area of expertise. Please ask me about Picarro's products, technology, or environmental monitoring applications.",
                    sources=[],
                    query=query,
                    model_used=self.model_name,
                    response_time=time.time() - start_time,
                    confidence_score=0.0
                )
            
            # Generate prompt with retrieved sources
            prompt = self._generate_prompt(query, sources)
            
            # Generate AI response
            logger.info("Generating AI response...")
            answer, api_response_time = self._call_ollama_api(prompt)
            
            total_response_time = time.time() - start_time
            
            # Calculate confidence score based on source similarity scores
            if sources:
                avg_similarity = sum(s.similarity_score for s in sources) / len(sources)
                confidence_score = min(avg_similarity * 1.2, 1.0)  # Boost confidence slightly
            else:
                confidence_score = 0.0
            
            response = AIResponse(
                answer=answer,
                sources=sources,
                query=query,
                model_used=self.model_name,
                response_time=total_response_time,
                confidence_score=confidence_score
            )
            
            logger.info(f"Response generated successfully in {total_response_time:.2f}s")
            logger.info(f"Confidence score: {confidence_score:.3f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        try:
            logger.info("Adding document to knowledge base...")
            doc_id = self.document_processor.add_document(content, metadata)
            logger.info(f"Document added successfully with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        try:
            return self.document_processor.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            raise


# Production-ready AI Responder for Picarro SearchAI
# Enhanced with specific analysis capabilities for better diagram and architecture descriptions 