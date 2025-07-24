import logging
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import os

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
        ollama_url: str = None,
        model_name: str = "llama3:latest",
        max_retrieved_docs: int = 20,  # Retrieve more initially, then filter
        max_tokens: int = 4096,
        temperature: float = 0.2,
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
        # Use env var or default to host.docker.internal for Docker
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.model_name = model_name
        self.max_retrieved_docs = max_retrieved_docs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        logger.info(f"AIResponder initialized with model: {model_name}")
        logger.info(f"Ollama URL: {self.ollama_url}")
    
    def _check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            # Try to list models to check connection
            response = requests.get(
                f"{self.ollama_url}/api/tags",
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
    
    def _check_ethical_content(self, query: str) -> Tuple[bool, str]:
        """
        Check if the query contains unethical or inappropriate content.
        
        Args:
            query: User query to check
            
        Returns:
            Tuple of (is_ethical, reason) where is_ethical is True if content is acceptable
        """
        query_lower = query.lower()
        
        # Define comprehensive unethical content patterns
        unethical_patterns = {
            # Harmful content
            'harmful': [
                'how to harm', 'how to hurt', 'how to kill', 'how to injure',
                'how to damage', 'how to destroy', 'how to sabotage',
                'weapon', 'explosive', 'bomb', 'poison', 'toxic',
                'self-harm', 'suicide', 'self injury', 'cutting',
                'violence', 'assault', 'attack', 'fight', 'battle'
            ],
            
            # Illegal activities
            'illegal': [
                'how to hack', 'how to crack', 'how to steal', 'how to cheat',
                'how to scam', 'how to fraud', 'how to counterfeit',
                'illegal drugs', 'drug manufacturing', 'drug dealing',
                'money laundering', 'tax evasion', 'identity theft',
                'cyber attack', 'ddos', 'malware', 'virus', 'trojan'
            ],
            
            # Privacy violations
            'privacy': [
                'how to spy', 'how to stalk', 'how to track without permission',
                'how to access private', 'how to break into', 'how to invade privacy',
                'surveillance without consent', 'wiretapping', 'eavesdropping'
            ],
            
            # Discrimination and hate
            'discrimination': [
                'hate speech', 'racist', 'sexist', 'homophobic', 'transphobic',
                'discriminatory', 'prejudiced', 'bigoted', 'slur',
                'how to discriminate', 'how to exclude based on'
            ],
            
            # Misinformation and manipulation
            'misinformation': [
                'how to lie', 'how to deceive', 'how to manipulate',
                'how to spread fake news', 'how to create propaganda',
                'how to gaslight', 'how to brainwash'
            ],
            
            # Inappropriate content
            'inappropriate': [
                'pornographic', 'explicit', 'adult content', 'nsfw',
                'inappropriate for work', 'inappropriate for children'
            ],
            
            # Corporate espionage and sabotage
            'corporate_espionage': [
                'how to steal trade secrets', 'how to sabotage competitor',
                'how to spy on company', 'how to leak confidential',
                'how to hack company', 'how to access competitor data'
            ]
        }
        
        # Check for unethical patterns
        detected_issues = []
        for category, patterns in unethical_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    detected_issues.append(f"{category}: {pattern}")
        
        # Add standalone harmful keywords for flexible matching
        harmful_keywords = [
            'kill', 'harm', 'attack', 'hurt', 'injure', 'destroy', 'sabotage',
            'suicide', 'self-harm', 'self harm', 'violence', 'assault', 'bomb', 'poison',
            'explosive', 'weapon', 'murder', 'abuse', 'rape', 'shoot', 'stab', 'abduct', 'kidnap'
        ]
        for word in harmful_keywords:
            if word in query_lower:
                logger.warning(f"Ethical content filter triggered: harmful keyword '{word}' detected.")
                return False, f"Harmful keyword detected: '{word}'"
        
        if detected_issues:
            reason = f"Query contains unethical content: {', '.join(detected_issues[:3])}"
            logger.warning(f"Ethical content filter triggered: {reason}")
            return False, reason
        
        # Check for suspicious intent patterns
        suspicious_intent = [
            'bypass', 'circumvent', 'override', 'disable', 'remove',
            'ignore', 'skip', 'avoid', 'evade', 'escape'
        ]
        
        # Only flag if combined with potentially harmful terms
        harmful_terms = ['security', 'filter', 'block', 'restriction', 'limit', 'rule']
        
        for intent in suspicious_intent:
            for term in harmful_terms:
                if intent in query_lower and term in query_lower:
                    reason = f"Query appears to attempt bypassing {term} restrictions"
                    logger.warning(f"Ethical content filter triggered: {reason}")
                    return False, reason
        
        return True, "Query passed ethical content filter"
    
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

    def _deduplicate_sources_by_document_id(self, sources: List[Source]) -> List[Source]:
        """
        Deduplicate sources by document ID, keeping the highest scoring chunk from each document.
        
        Args:
            sources: List of source documents
            
        Returns:
            Deduplicated list of sources
        """
        if not sources:
            return sources
        
        # Group sources by document ID
        doc_groups = {}
        for source in sources:
            doc_id = source.metadata.get('document_id', 'unknown')
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(source)
        
        # For each document, keep the source with the highest similarity score
        deduplicated_sources = []
        for doc_id, doc_sources in doc_groups.items():
            if doc_sources:
                # Sort by similarity score (highest first) and take the best one
                best_source = max(doc_sources, key=lambda s: s.similarity_score)
                deduplicated_sources.append(best_source)
        
        # Sort by similarity score (highest first) to maintain relevance order
        deduplicated_sources.sort(key=lambda s: s.similarity_score, reverse=True)
        
        # Update ranks to reflect new order
        for i, source in enumerate(deduplicated_sources):
            source.rank = i + 1
        
        logger.info(f"Deduplicated sources: {len(sources)} -> {len(deduplicated_sources)} unique documents")
        return deduplicated_sources

    def _is_api_query(self, query: str) -> bool:
        api_keywords = [
            'api', 'endpoint', 'method', 'get', 'post', 'put', 'patch', 'delete',
            'request', 'response', 'parameter', 'query', 'path', 'body', 'header',
            'status', 'code', 'error', 'validation', 'schema', 'json', 'xml',
            'authentication', 'authorization', 'token', 'bearer', 'oauth',
            'rate limit', 'pagination', 'filter', 'sort', 'search'
        ]
        q = query.lower()
        return any(k in q for k in api_keywords)

    def _generate_prompt(self, query: str, sources: List[Source]) -> str:
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"Source {i} (Score: {source.similarity_score:.3f}):\n{source.content}\n")
        context = "\n".join(context_parts)
        if self._is_api_query(query):
            prompt = f"""You are a technical API documentation assistant for Picarro's FenceLine Cloud Solution.\n\nCONTEXT INFORMATION:\n{context}\n\nUSER QUESTION: {query}\n\nCRITICAL INSTRUCTIONS FOR API DOCUMENTATION RESPONSES:\n\n1. **API-SPECIFIC ANALYSIS**:\n   - When asked about API endpoints, provide EXACT details from the sources\n   - Include HTTP method, endpoint path, request/response formats\n   - List all parameters, their types, and whether they're required/optional\n   - Include example request/response bodies when available\n   - Mention status codes and error responses\n\n2. **STRUCTURED RESPONSE FORMAT**:\n   - Start with a direct answer to the specific API question\n   - Use clear headings: 'Endpoint', 'Method', 'Parameters', 'Request Body', 'Response', 'Examples'\n   - Include exact field names, data types, and validation rules\n   - Quote specific text from sources when describing API behavior\n\n3. **TECHNICAL ACCURACY**:\n   - DO NOT make assumptions about API behavior not documented in sources\n   - Include exact parameter names, types, and constraints\n   - Mention specific validation rules, error codes, and response formats\n   - If source contains JSON examples, include them in your response\n\n4. **QUALITY REQUIREMENTS**:\n   - Be precise and technical - this is API documentation\n   - Include all relevant details from the sources\n   - Use proper technical terminology\n   - Structure information logically for developers\n\n5. **If the API information is not in the sources**, say:\n   'I don't have specific information about that API endpoint in the available documentation. Please check the latest API documentation or contact the development team.'\n\nAnswer:"""
        else:
            prompt = f"""You are an AI assistant for Picarro's FenceLine Cloud Solution and related documentation.\n\nIMPORTANT: You can answer questions related to:\n- Picarro's products, technology, services, and environmental monitoring applications\n- FenceLine architecture, system design, and cloud solutions\n- Documentation, templates, troubleshooting guides, and how-to articles\n- Any content present in the provided context\n\nIf the user asks about topics completely unrelated to Picarro or the provided context, politely decline and redirect them to relevant topics.\n\nContext Information:\n{context}\n\nUser Question: {query}\n\nCRITICAL INSTRUCTIONS FOR RESPONSE QUALITY:\n\n1. **ANALYZE SPECIFIC CONTENT**: \n   - DO NOT give generic responses. Analyze the ACTUAL content provided in the sources.\n   - Extract and describe SPECIFIC details from the source content.\n   - Quote or reference exact text from the sources when describing components.\n   - List actual component names, labels, and technical specifications mentioned.\n\n2. **FOR DIAGRAMS AND ARCHITECTURE**:\n   - If the context contains diagrams, architecture, or visual content:\n     * Describe the EXACT components, sections, and layers as they appear in the source\n     * Explain the specific relationships and connections as shown in the source\n     * Mention specific file names, URLs, or identifiers from the source\n     * Include technical details like IP addresses, port numbers, or configurations if mentioned\n     * Describe any labels, arrows, flow directions, or annotations exactly as they appear\n     * If the source mentions specific diagram elements (like 'Tenant - Site Hierarchy', 'Platform API Architecture'), describe them in detail\n     * Extract and explain any specific technical terms, component names, or architectural patterns mentioned\n   - AVOID generic statements like 'appears to be' or 'seems to show' - be specific about what's actually in the source\n   - If the source contains raw diagram data or metadata, interpret and explain the actual content rather than making assumptions\n\n3. **RESPONSE STRUCTURE**:\n   - Start with a direct answer to the user's question\n   - Provide specific details from the sources\n   - If describing architecture, break it down into clear sections\n   - Use bullet points or numbered lists for clarity\n   - Reference specific source information when possible\n   - For FenceLine architecture questions, focus on the actual diagram content and technical specifications mentioned\n\n4. **QUALITY REQUIREMENTS**:\n   - Be precise and technical when the source contains technical information\n   - Avoid vague or generic descriptions\n   - Focus on what's actually documented in the sources\n   - If the source contains specific technical details, include them in your response\n\n5. **If the question is NOT related to Picarro, FenceLine, or the provided context**, respond with:\n   'I'm sorry, but I can only answer questions related to Picarro's solutions and the documentation available. Your question appears to be outside my area of expertise. Please ask me about Picarro's products, FenceLine architecture, or related documentation.'\n\nAnswer:"""
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
            
            # Check ethical content first
            is_ethical, ethical_reason = self._check_ethical_content(query)
            if not is_ethical:
                logger.warning(f"Ethical content filter rejected query: {ethical_reason}")
                return AIResponse(
                    answer="Your request is unethical and cannot be processed.",
                    sources=[],
                    query=query,
                    model_used=self.model_name,
                    response_time=time.time() - start_time,
                    confidence_score=0.0
                )
            
            # Retrieve relevant documents
            logger.info("Retrieving relevant documents...")
            search_results = self.document_processor.search_documents(
                query=query,
                n_results=self.max_retrieved_docs,
                filter_metadata=filter_metadata
            )
            
            # Convert search results to Source objects and filter by relevance
            sources = []
            min_similarity_threshold = 0.4  # Higher threshold to filter out less relevant sources
            
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
            
            # Deduplicate sources by document ID
            sources = self._deduplicate_sources_by_document_id(sources)
            
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