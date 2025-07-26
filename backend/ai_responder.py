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
                r'```json\s*\n\s*\{[^{}]*"[^"]*":\s*"[^"]*"[^{}]*\}\s*\n```',  # JSON code blocks with specific structure
                r'"id":\s*"123"',  # Specific placeholder IDs
                r'"id":\s*"456"',  # Specific placeholder IDs
                r'"name":\s*"example"',  # Example names
                r'"description":\s*"This is an example"',  # Example descriptions
            ],
            
            # REST API endpoints without source evidence - more specific
            'rest_endpoints': [
                r'GET\s+/api/v1/\w+',  # Versioned API endpoints
                r'POST\s+/api/v1/\w+',  # Versioned API endpoints
                r'PUT\s+/api/v1/\w+',  # Versioned API endpoints
                r'DELETE\s+/api/v1/\w+',  # Versioned API endpoints
                r'/\w+/\{id\}/',  # Path parameters with braces
                r'/\w+/\d{3,}',  # Numeric path parameters (3+ digits)
            ],
            
            # Placeholder values - more specific
            'placeholder_values': [
                r'\b123\b',  # Common placeholder number
                r'\b456\b',  # Common placeholder number
                r'\b789\b',  # Common placeholder number
                r'example\.com',  # Example domain
                r'api\.example\.com',  # Example API domain
                r'your-domain\.com',  # Generic domain placeholder
                r'your-email@company\.com',  # Email placeholder
                r'your-api-token',  # Token placeholder
                r'<your-',  # Generic placeholders
                r'\[your-',  # Bracket placeholders
            ],
            
            # Standardized error codes without source documentation - more specific
            'error_codes': [
                r'"error_code":\s*"400"',  # JSON error codes
                r'"error_code":\s*"401"',
                r'"error_code":\s*"404"',
                r'"error_code":\s*"500"',
                r'"status":\s*400',  # Status codes in JSON
                r'"status":\s*401',
                r'"status":\s*404',
                r'"status":\s*500',
            ],
            
            # Generic API patterns - more specific
            'generic_api_patterns': [
                r'Request\s+Parameters:\s*\n',  # Generic API documentation
                r'Response\s+Format:\s*\n',  # Generic response format
                r'Example\s+Response:\s*\n',  # Example responses
                r'Authentication\s+Requirements:\s*\n',  # Generic auth
                r'Headers:\s*\n',  # Generic headers
                r'Content-Type:\s*application/json\s*\n',  # Standard headers
                r'Authorization:\s*Bearer\s*\n',  # Standard auth
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.warning_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def detect_hallucination(self, response_text: str, sources: List[Any]) -> Tuple[bool, List[str], Dict[str, List[str]]]:
        """
        Detect potential hallucination patterns in the response.
        
        Args:
            response_text: The AI-generated response text
            sources: List of source documents used for generation
            
        Returns:
            Tuple of (is_hallucination, warning_messages, detected_patterns)
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
        
        # Additional checks for specific hallucination indicators
        if self._has_unsupported_technical_details(response_text, sources):
            is_hallucination = True
            warning_messages.append("Technical details detected without source documentation")
            logger.warning("HALLUCINATION: Technical details not supported by sources")
        
        # Debug logging
        if detected_patterns:
            logger.info(f"Hallucination detection results: {detected_patterns}")
            logger.info(f"Is hallucination: {is_hallucination}")
        
        return is_hallucination, warning_messages, detected_patterns
    
    def _patterns_supported_by_sources(self, patterns: List[str], sources: List[Any]) -> bool:
        """
        Check if detected patterns are actually supported by source documents.
        
        Args:
            patterns: List of detected patterns
            sources: List of source documents
            
        Returns:
            True if patterns are supported by sources, False otherwise
        """
        if not sources:
            return False
        
        # Extract all source content
        source_content = " ".join([getattr(source, 'content', str(source)) for source in sources])
        source_content_lower = source_content.lower()
        
        # Check if any of the patterns appear in source content
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in source_content_lower:
                return True
        
        return False
    
    def _has_unsupported_technical_details(self, response_text: str, sources: List[Any]) -> bool:
        """
        Check for technical details that aren't supported by sources.
        
        Args:
            response_text: The AI-generated response text
            sources: List of source documents
            
        Returns:
            True if unsupported technical details are found
        """
        # Technical keywords that should be supported by sources
        technical_keywords = [
            'api', 'endpoint', 'json', 'request', 'response', 'schema',
            'authentication', 'headers', 'parameters', 'validation',
            'error', 'status', 'code', 'method', 'url'
        ]
        
        if not sources:
            # Only flag if multiple technical keywords are present
            found_keywords = [kw for kw in technical_keywords if kw in response_text.lower()]
            return len(found_keywords) >= 3  # Require at least 3 technical keywords
        
        # Extract source content
        source_content = " ".join([getattr(source, 'content', str(source)) for source in sources])
        source_content_lower = source_content.lower()
        
        # Check if technical keywords in response are supported by sources
        response_lower = response_text.lower()
        unsupported_keywords = []
        for keyword in technical_keywords:
            if keyword in response_lower and keyword not in source_content_lower:
                unsupported_keywords.append(keyword)
        
        # Only flag if multiple unsupported technical keywords are found
        return len(unsupported_keywords) >= 2  # Require at least 2 unsupported keywords


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
        
        # Initialize hallucination detector
        self.hallucination_detector = HallucinationDetector()
        
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
            context_parts.append(f"Source {i}:\n{source.content}\n")
        context = "\n".join(context_parts)
        
        prompt = f"""You are a senior technical expert. Extract and organize ALL technical information from the sources into a comprehensive technical response.

TECHNICAL EXTRACTION GUIDELINES:
- Extract ALL API endpoints, event names, and technical specifications
- Include ALL parameter names, data types, and field definitions
- Organize information with clear technical headings and sections
- Present data structures, schemas, and technical details professionally
- Include ALL procedural steps and implementation details
- Use bullet points and structured formatting for technical specifications
- Extract exact field names, types, and descriptions from JSON schemas
- Present WebSocket events, namespaces, and connection details completely

REQUIRED SECTIONS (if information exists in sources):
## API Endpoints
## WebSocket Configuration  
## Event Specifications
## Data Structures
## Implementation Steps
## Technical Parameters

SOURCES:
{context}

USER QUESTION: {query}

Extract and present ALL technical information from the sources in a comprehensive, well-organized technical format. Do not summarize - include all available technical details.

TECHNICAL RESPONSE:"""
        
        return prompt

    def _validate_no_guessing(self, response: str) -> bool:
        """
        Validate that the response doesn't contain guessing phrases.
        
        Args:
            response: The AI response to validate
            
        Returns:
            True if response is clean, False if it contains guessing phrases
        """
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
    
    def get_adaptive_threshold(self, scores: List[float], query: str) -> float:
        """
        Calculate adaptive threshold based on similarity scores and query.
        
        Args:
            scores: List of similarity scores from search results
            query: User query
            
        Returns:
            Adaptive threshold value
        """
        if not scores:
            return 0.15
        max_score = max(scores)
        if max_score < 0.3:
            return 0.1
        else:
            return max(0.15, max_score * 0.5)
    
    def get_sources_with_cascade(self, search_results: List[dict], query: str) -> List[Source]:
        """
        Get sources using cascade threshold system for maximum inclusivity.
        
        Args:
            search_results: Raw search results from document processor
            query: User query
            
        Returns:
            List of filtered sources
        """
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
        
        # Fallback: return top 3 results if no threshold works
        logger.info("No sources found with cascade thresholds, using top 3 results")
        return [Source(content=r["content"], metadata=r["metadata"], similarity_score=r["similarity_score"], rank=r["rank"]) for r in search_results[:3]]

    def _analyze_content_type(self, sources: List[Source], query: str) -> Dict[str, Any]:
        """
        Automatically detect what type of content we have and how to handle it.
        
        Args:
            sources: List of source documents
            query: User query
            
        Returns:
            Dictionary with content analysis results
        """
        # Analyze source content characteristics
        all_content = " ".join([s.content for s in sources]).lower()
        
        analysis = {
            'has_technical_specs': False,
            'has_code_examples': False,
            'has_api_info': False,
            'has_procedures': False,
            'content_depth': 'basic',
            'technical_keywords': [],
            'main_topics': []
        }
        
        # Universal technical indicators
        technical_patterns = {
            'api_info': ['endpoint', 'api', 'http', 'get', 'post', 'json', 'request', 'response'],
            'code_examples': ['function', 'def', 'class', 'import', 'const', 'var', '```', 'code'],
            'procedures': ['step', 'procedure', 'process', 'workflow', 'install', 'setup', 'configure'],
            'specifications': ['spec', 'requirement', 'parameter', 'field', 'property', 'attribute']
        }
        
        # Detect content types
        for category, keywords in technical_patterns.items():
            matches = sum(1 for kw in keywords if kw in all_content)
            if matches >= 2:  # Need at least 2 matching keywords
                analysis[f'has_{category}'] = True
                analysis['technical_keywords'].extend([kw for kw in keywords if kw in all_content])
        
        # Determine content depth
        total_length = sum(len(s.content) for s in sources)
        if total_length > 2000:
            analysis['content_depth'] = 'detailed'
        elif total_length > 500:
            analysis['content_depth'] = 'moderate'
        else:
            analysis['content_depth'] = 'basic'
        
        # Extract main topics (most frequent meaningful words)
        words = all_content.split()
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():  # Meaningful words only
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top topics
        analysis['main_topics'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        logger.info(f"Content analysis: {analysis}")
        return analysis

    def _get_adaptive_response_strategy(self, content_analysis: Dict[str, Any], query: str) -> str:
        """
        Determine how to respond based on content analysis.
        
        Args:
            content_analysis: Results from content analysis
            query: User query
            
        Returns:
            Response strategy string
        """
        query_lower = query.lower()
        
        # Match query intent with available content
        if any(word in query_lower for word in ['how', 'what', 'explain', 'describe']):
            if content_analysis['content_depth'] == 'detailed':
                return "comprehensive_explanation"
            elif content_analysis['has_procedures']:
                return "step_by_step_guide"
            else:
                return "basic_explanation"
        
        elif any(word in query_lower for word in ['api', 'endpoint', 'integration']):
            if content_analysis['has_api_info']:
                return "technical_api_response"
            else:
                return "api_guidance_response"
        
        elif any(word in query_lower for word in ['example', 'sample', 'demo']):
            if content_analysis['has_code_examples']:
                return "example_based_response"
            else:
                return "conceptual_example_response"
        
        else:
            return "general_informative_response"

    def _generate_adaptive_prompt(self, query: str, sources: List[Source]) -> str:
        """
        Generate adaptive prompt based on content analysis and response strategy.
        
        Args:
            query: User query
            sources: List of source documents
            
        Returns:
            Generated prompt string
        """
        # Analyze what we have
        content_analysis = self._analyze_content_type(sources, query)
        response_strategy = self._get_adaptive_response_strategy(content_analysis, query)
        
        # Build context
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"Source {i}:\n{source.content}\n")
        context = "\n".join(context_parts)
        
        # Universal base instructions
        base_instructions = """You are an expert technical assistant. Analyze the provided sources and give the most helpful response possible based on what's actually available."""
        
        # Strategy-specific instructions
        strategy_instructions = {
            "comprehensive_explanation": "Provide a detailed, well-structured explanation using all available information from the sources.",
            
            "technical_api_response": "Focus on the technical specifications, endpoints, and implementation details found in the sources.",
            
            "api_guidance_response": "Explain what you can determine about the API from the sources, and provide clear guidance on getting complete technical details.",
            
            "step_by_step_guide": "Organize the information into clear, actionable steps based on the procedures described in the sources.",
            
            "example_based_response": "Use the examples and code snippets from the sources to illustrate the concepts.",
            
            "conceptual_example_response": "Explain the concepts clearly based on the sources, and suggest where to find implementation examples.",
            
            "basic_explanation": "Provide a clear, concise explanation based on the available information.",
            
            "general_informative_response": "Give a helpful, informative response that makes the best use of the available source material."
        }
        
        instruction = strategy_instructions.get(response_strategy, strategy_instructions["general_informative_response"])
        
        # Content-aware guidelines
        guidelines = []
        if content_analysis['has_technical_specs']:
            guidelines.append("Include specific technical details and specifications mentioned in the sources.")
        if content_analysis['has_procedures']:
            guidelines.append("Reference the procedures and processes described in the sources.")
        if content_analysis['content_depth'] == 'basic':
            guidelines.append("Since the available information is limited, be clear about what is and isn't covered.")
        
        guidelines_text = "\n".join([f"• {g}" for g in guidelines])
        
        prompt = f"""{base_instructions}

RESPONSE STRATEGY: {instruction}

CONTENT GUIDELINES:
{guidelines_text}

SOURCES:
{context}

USER QUESTION: {query}

Provide the most helpful response possible based on the available sources. Always be honest about the limitations of the available information while maximizing its usefulness.

RESPONSE:"""
        
        logger.info(f"Generated adaptive prompt with strategy: {response_strategy}")
        return prompt

    def _calculate_adaptive_confidence(self, sources: List[Source], query: str, content_analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence based on content quality and query match.
        
        Args:
            sources: List of source documents
            query: User query
            content_analysis: Results from content analysis
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not sources:
            return 0.0
        
        # Base similarity confidence
        base_confidence = sum(s.similarity_score for s in sources) / len(sources)
        
        # Content quality multipliers
        quality_multiplier = 1.0
        
        if content_analysis['content_depth'] == 'detailed':
            quality_multiplier += 0.2
        elif content_analysis['content_depth'] == 'moderate':
            quality_multiplier += 0.1
        
        # Query-content match bonus
        query_words = set(query.lower().split())
        content_words = set(" ".join([s.content.lower() for s in sources]).split())
        word_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
        
        match_bonus = word_overlap * 0.3
        
        # Final confidence
        final_confidence = min(1.0, (base_confidence * quality_multiplier) + match_bonus)
        
        logger.info(f"Adaptive confidence: base={base_confidence:.3f}, quality_mult={quality_multiplier:.3f}, match_bonus={match_bonus:.3f}, final={final_confidence:.3f}")
        return final_confidence

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
            
            # ADAPTIVE CASCADE SYSTEM - Use cascade thresholds for maximum inclusivity
            sources = self.get_sources_with_cascade(search_results, query)
            
            logger.info(f"Retrieved {len(sources)} relevant documents using cascade system")
            
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
            
            # UNIVERSAL ADAPTIVE SYSTEM - Use strict anti-guessing prompt
            content_analysis = self._analyze_content_type(sources, query)
            prompt = self._generate_prompt(query, sources)  # Use strict prompt, not adaptive
            
            # Generate AI response
            logger.info("Generating AI response with adaptive system...")
            answer, api_response_time = self._call_ollama_api(prompt)
            
            # VALIDATE NO GUESSING - Check for educated guesses and replace if found
            if not self._validate_no_guessing(answer):
                logger.warning("Response contained guessing - using strict fallback")
                source_content = " ".join([s.content[:200] for s in sources])
                answer = f"Based on the available documentation: {source_content} The sources don't provide complete implementation details for this specific question."
            
            total_response_time = time.time() - start_time
            
            # Calculate adaptive confidence score
            confidence_score = self._calculate_adaptive_confidence(sources, query, content_analysis)
            
            # Log source usage for transparency
            source_titles = [s.metadata.get('title', 'Unknown') for s in sources]
            logger.info(f"Using sources: {source_titles}")
            logger.info(f"Content analysis: {content_analysis}")
            logger.info(f"Adaptive confidence: {confidence_score:.3f}")
            
            # Anti-hallucination: Reject very low confidence responses
            if confidence_score < 0.05:  # Very permissive threshold for maximum inclusivity
                logger.warning(f"Low confidence response ({confidence_score:.3f}) - rejecting to prevent hallucination")
                return AIResponse(
                    answer=f"I'm not confident enough in the available information to provide a reliable answer. The sources have low relevance to your question. Please try rephrasing your question or ask about a different aspect of Picarro's technology.",
                    sources=sources,
                    query=query,
                    model_used=self.model_name,
                    response_time=time.time() - start_time,
                    confidence_score=confidence_score
                )
            
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
            
            # HALLUCINATION DETECTION - DISABLED for maximum inclusivity
            # is_hallucination, warning_messages, detected_patterns = self.hallucination_detector.detect_hallucination(answer, sources)
            
            # TEMPORARY: Bypass hallucination detection for debugging
            # bypass_hallucination_detection = os.getenv("BYPASS_HALLUCINATION_DETECTION", "false").lower() == "true"
            
            # if is_hallucination and not bypass_hallucination_detection:
            #     # Log all potential hallucinations for review
            #     logger.warning(f"HALLUCINATION DETECTED for query: '{query}'")
            #     logger.warning(f"Warning messages: {warning_messages}")
            #     logger.warning(f"Detected patterns: {detected_patterns}")
            #     logger.warning(f"Original response: {answer[:200]}...")  # Log first 200 chars
            #     
            #     # Replace response with safe message
            #     response.answer = "I don't have enough specific technical documentation to provide detailed API information. Please refer to the official API documentation or provide more specific documentation sources."
            #     response.confidence_score = 0.0
            #     
            #     logger.info("Response replaced with safe message due to hallucination detection")
            # elif is_hallucination and bypass_hallucination_detection:
            #     # Log but don't replace response for debugging
            #     logger.warning(f"HALLUCINATION DETECTED (BYPASSED) for query: '{query}'")
            #     logger.warning(f"Warning messages: {warning_messages}")
            #     logger.warning(f"Detected patterns: {detected_patterns}")
            #     logger.info("Response kept for debugging purposes")
            
            # After filtering sources and before generating the prompt:
            if not sources:
                response.answer = "I don't have documentation about specific API endpoints for this topic."
                response.confidence_score = 0.0
            # If sources exist but none contain technical details (API, JSON, etc.)
            has_technical = any(
                any(keyword in s.content.lower() for keyword in ["api", "endpoint", "json", "request", "response", "schema"])
                for s in sources
            )
            if not has_technical:
                response.answer = "The available documentation doesn't include detailed API specifications."
                response.confidence_score = 0.0
            # Before returning the final answer, enforce confidence threshold for technical info:
            if "api" in query.lower() and response.confidence_score is not None and response.confidence_score < 0.4:
                response.answer = (
                    "The available documentation does not provide high-confidence technical API details for this topic. "
                    "Please consult the official API documentation or contact the development team."
                )
            
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