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
        model_name: str = "llama3.2:3b",
        max_retrieved_docs: int = 10,  # Reduced from 20 for higher quality
        max_tokens: int = 8192,  # Increased for longer, more detailed responses
        temperature: float = 0.3,  # Slightly increased for better explanations
        timeout: int = 120,  # Increased timeout for longer responses
        min_sources_required: int = 1  # Keep at 1 for small datasets
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
            min_sources_required: Minimum number of sources required to answer
        """
        self.document_processor = document_processor
        # Use env var or default to host.docker.internal for Docker
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.model_name = model_name
        self.max_retrieved_docs = max_retrieved_docs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.min_sources_required = min_sources_required
        
        logger.info(f"AIResponder initialized with model: {model_name}")
        logger.info(f"Ollama URL: {self.ollama_url}")
        logger.info(f"Anti-hallucination settings: max_docs={max_retrieved_docs}, min_sources={min_sources_required}")
    
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
        
        # Anti-hallucination base instructions
        anti_hallucination_instructions = """
CRITICAL ANTI-HALLUCINATION RULES:
1. **ONLY USE PROVIDED SOURCES**: Base your answer EXCLUSIVELY on the information in the provided sources
2. **NO ASSUMPTIONS**: Do not make assumptions or inferences beyond what's explicitly stated in the sources
3. **SOURCE VERIFICATION**: If you cannot verify a fact from the provided sources, do not include it
4. **UNCERTAINTY ACKNOWLEDGMENT**: If the sources don't contain enough information, say "I don't have enough information from the available sources"
5. **QUOTE WHEN POSSIBLE**: Use direct quotes from sources when making specific claims
6. **AVOID GENERIC RESPONSES**: Do not provide generic or template responses - be specific to the source content
"""
        
        # Technical expert instructions
        technical_instructions = """
RESPONSE QUALITY REQUIREMENTS - SENIOR TECHNICAL EXPERT:

1. **DIRECT AND PRECISE ANSWERS**: 
   - Give direct, technical responses like a senior engineer
   - Focus on specific implementation details and technical specifications
   - Avoid generic explanations of basic concepts unless specifically asked
   - Be concise and practical, not educational

2. **TECHNICAL FOCUS**:
   - Reference specific code examples, API endpoints, and data structures from documents
   - Include exact field names, parameters, and configuration details
   - Use bullet points and code snippets when relevant
   - Focus on practical implementation details

3. **STRUCTURED TECHNICAL FORMAT**:
   - Use clear technical headings
   - Include bullet points for lists of features/endpoints
   - Provide code examples and configuration snippets
   - Focus on specific technical details, not general explanations

4. **PRECISE TECHNICAL LANGUAGE**:
   - Use technical terminology appropriate for experienced developers
   - Reference exact API endpoints, data structures, and configurations
   - Avoid explaining basic concepts unless specifically requested
   - Focus on "how to" rather than "what is"

5. **IMPLEMENTATION-FOCUSED**:
   - Provide specific technical details for implementation
   - Include exact parameters, data types, and validation rules
   - Reference specific error codes and their meanings
   - Focus on practical technical solutions
"""
        
        if self._is_api_query(query):
            prompt = f"""You are a senior Picarro technical expert specializing in API documentation.

{anti_hallucination_instructions}

{technical_instructions}

CONTEXT INFORMATION:
{context}

USER QUESTION: {query}

CRITICAL INSTRUCTIONS FOR API RESPONSES:

1. **DIRECT API SPECIFICATIONS**:
   - Provide exact endpoint URLs, HTTP methods, and parameters
   - Include specific field names, data types, and validation rules
   - Reference exact request/response formats from documentation
   - Focus on implementation details, not general API concepts

2. **TECHNICAL IMPLEMENTATION**:
   - Include specific code examples and configuration snippets
   - Reference exact error codes and their meanings
   - Provide specific authentication requirements and headers
   - Focus on practical integration details

3. **PRECISE TECHNICAL FORMAT**:
   - Use bullet points for endpoint lists and parameters
   - Include code snippets for request/response examples
   - Reference specific data structures and field mappings
   - Focus on technical specifications, not educational explanations

4. **IMPLEMENTATION-FOCUSED**:
   - Provide specific technical details for immediate implementation
   - Include exact parameters, headers, and response formats
   - Reference specific business logic and data flow
   - Focus on "how to implement" rather than "what is an API"

5. **If the API information is not in the sources**, say:
   'I don't have specific information about that API endpoint in the available documentation. Please check the latest API documentation or contact the development team.'

Answer:"""
        else:
            prompt = f"""You are a senior Picarro technical expert. Give direct, precise answers based on the provided documentation.

{anti_hallucination_instructions}

{technical_instructions}

IMPORTANT: You can answer questions related to:
- Picarro's products, technology, services, and environmental monitoring applications
- FenceLine architecture, system design, and cloud solutions
- Documentation, templates, troubleshooting guides, and how-to articles
- Any content present in the provided context

If the user asks about topics completely unrelated to Picarro or the provided context, politely decline and redirect them to relevant topics.

Context Information:
{context}

User Question: {query}

CRITICAL INSTRUCTIONS FOR TECHNICAL RESPONSES:

1. **DIRECT AND PRECISE ANSWERS**:
   - Give direct, technical responses like a senior engineer
   - Focus on specific implementation details and technical specifications
   - Avoid generic explanations of basic concepts unless specifically asked
   - Be concise and practical, not educational

2. **TECHNICAL FOCUS**:
   - Reference specific code examples, API endpoints, and data structures from documents
   - Include exact field names, parameters, and configuration details
   - Use bullet points and code snippets when relevant
   - Focus on practical implementation details

3. **IMPLEMENTATION-FOCUSED**:
   - Provide specific technical details for immediate implementation
   - Include exact parameters, configurations, and validation rules
   - Reference specific error codes and their meanings
   - Focus on "how to implement" rather than "what is"

4. **PRECISE TECHNICAL FORMAT**:
   - Use bullet points for lists of features/endpoints
   - Include code examples and configuration snippets
   - Reference specific data structures and field mappings
   - Focus on technical specifications, not general explanations

5. **If the question is NOT related to Picarro, FenceLine, or the provided context**, respond with:
   'I'm sorry, but I can only answer questions related to Picarro's solutions and the documentation available. Your question appears to be outside my area of expertise. Please ask me about Picarro's products, FenceLine architecture, or related documentation.'

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
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query to improve search matching.
        
        Args:
            query: Original user query
            
        Returns:
            Preprocessed query
        """
        # Common typos and variations for better matching
        typo_fixes = {
            'managment': 'management',
            'fenceline': 'fenceline',  # Keep as is
            'fenceline': 'fenceline',  # Keep as is
            'api': 'API',
            'tenant': 'tenant',
            'site': 'site',
            'user': 'user',
            'auth': 'authentication',
            'config': 'configuration',
            'setup': 'setup',
            'install': 'installation',
            'create': 'create',
            'creation': 'creation'
        }
        
        processed_query = query.lower()
        
        # Apply typo fixes
        for typo, correction in typo_fixes.items():
            processed_query = processed_query.replace(typo, correction)
        
        # Add common variations for better matching
        variations = []
        if 'management' in processed_query:
            variations.extend(['tenant management', 'user management', 'site management'])
        if 'api' in processed_query:
            variations.extend(['API', 'endpoint', 'interface', 'contract'])
        if 'tenant' in processed_query:
            variations.extend(['tenant management', 'tenant API'])
        if 'site' in processed_query:
            variations.extend(['site management', 'site API', 'site creation', 'site setup'])
        if 'create' in processed_query or 'creation' in processed_query:
            variations.extend(['create', 'creation', 'setup', 'establish'])
        if 'monitoring' in processed_query:
            variations.extend(['monitoring system', 'monitoring API'])
        
        # Combine original query with variations
        if variations:
            processed_query = f"{processed_query} {' '.join(variations)}"
        
        logger.info(f"Query preprocessing: '{query}' -> '{processed_query}'")
        return processed_query
    
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
            
            # Preprocess query for better matching
            processed_query = self._preprocess_query(query)
            
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
                query=processed_query, # Use processed query for retrieval
                n_results=self.max_retrieved_docs,
                filter_metadata=filter_metadata
            )
            
            # Convert search results to Source objects and filter by relevance
            sources = []
            
            # DYNAMIC THRESHOLD SYSTEM - Adapts based on search results
            if search_results:
                # Calculate dynamic threshold based on result distribution
                scores = [result["similarity_score"] for result in search_results]
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                
                # Use adaptive threshold: 70% of max score, but not below 0.25
                dynamic_threshold = max(0.25, max_score * 0.7)
                
                # If max score is very low (< 0.4), be more lenient
                if max_score < 0.4:
                    dynamic_threshold = max(0.2, avg_score * 0.8)
                
                logger.info(f"Dynamic threshold: {dynamic_threshold:.3f} (max: {max_score:.3f}, avg: {avg_score:.3f})")
            else:
                dynamic_threshold = 0.3  # Fallback threshold
            
            for result in search_results:
                similarity_score = result["similarity_score"]
                content = result["content"]
                metadata = result["metadata"]
                
                # Use dynamic threshold for initial filtering
                if similarity_score >= dynamic_threshold:
                    # Additional content relevance check - require key terms to be present
                    query_terms = [term.lower() for term in query.split() if len(term) > 3]
                    content_lower = content.lower()
                    
                    # Count how many query terms are present in the content
                    matching_terms = sum(1 for term in query_terms if term in content_lower)
                    
                    # Adaptive term matching: be more lenient for high-quality matches
                    if similarity_score > 0.5:
                        required_terms = 1
                    elif similarity_score > 0.35:
                        required_terms = 1  # More lenient for medium-high scores
                    else:
                        required_terms = 2
                    
                    if matching_terms >= required_terms:
                        source = Source(
                            content=content,
                            metadata=metadata,
                            similarity_score=similarity_score,
                            rank=result["rank"]
                        )
                        sources.append(source)
                        logger.info(f"Included relevant source (score: {similarity_score:.3f}, terms: {matching_terms}/{required_terms}): {metadata.get('title', 'Unknown')}")
                    else:
                        logger.info(f"Filtering out content with insufficient terms (score: {similarity_score:.3f}, terms: {matching_terms}/{required_terms}): {metadata.get('title', 'Unknown')}")
                else:
                    logger.info(f"Filtering out low-relevance source (score: {similarity_score:.3f} < {dynamic_threshold:.3f}): {metadata.get('title', 'Unknown')}")
            
            logger.info(f"Retrieved {len(sources)} relevant documents (filtered from {len(search_results)} total)")
            
            # Anti-hallucination: Check if we have enough sources
            if len(sources) < self.min_sources_required:
                logger.warning(f"Insufficient sources ({len(sources)}) for reliable answer. Minimum required: {self.min_sources_required}")
                return AIResponse(
                    answer=f"I don't have enough relevant information from the available sources to provide a reliable answer. I found {len(sources)} relevant documents, but need at least {self.min_sources_required} for confidence. Please try rephrasing your question or ask about a different aspect of Picarro's technology.",
                    sources=sources,
                    query=query,
                    model_used=self.model_name,
                    response_time=time.time() - start_time,
                    confidence_score=0.0
                )
            
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
                
                # Anti-hallucination: Log source usage for transparency
                source_titles = [s.metadata.get('title', 'Unknown') for s in sources]
                logger.info(f"Using sources: {source_titles}")
                logger.info(f"Average similarity: {avg_similarity:.3f}, Confidence: {confidence_score:.3f}")
                
                # Anti-hallucination: Reject low confidence responses
                if confidence_score < 0.3:  # Reduced from 0.7 for more flexible confidence
                    logger.warning(f"Low confidence response ({confidence_score:.3f}) - rejecting to prevent hallucination")
                    return AIResponse(
                        answer=f"I'm not confident enough in the available information to provide a reliable answer. The sources have low relevance to your question. Please try rephrasing your question or ask about a different aspect of Picarro's technology.",
                        sources=sources,
                        query=query,
                        model_used=self.model_name,
                        response_time=time.time() - start_time,
                        confidence_score=confidence_score
                    )
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