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
        max_retrieved_docs: int = 5,
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
        # Define Picarro-related keywords
        picarro_keywords = [
            'picarro', 'gas analyzer', 'environmental monitoring', 'precision', 'emissions',
            'greenhouse gas', 'air quality', 'atmospheric', 'climate', 'carbon', 'methane',
            'co2', 'n2o', 'analyzer', 'spectrometer', 'cavity ring-down', 'crd', 'monitoring',
            'calibration', 'drift', 'stability', 'reliability', 'field deployment', 'remote',
            'continuous', 'real-time', 'data validation', 'quality control', 'industrial',
            'research', 'scientific', 'ecosystem', 'flux', 'sequestration', 'paleoclimatology',
            'ocean-atmosphere', 'urban air', 'agricultural', 'forest', 'atmospheric chemistry'
        ]
        
        # Check if query contains Picarro-related keywords
        query_lower = query.lower()
        has_picarro_keywords = any(keyword in query_lower for keyword in picarro_keywords)
        
        # Check if sources have sufficient relevance (similarity score > 0.7)
        relevant_sources = [s for s in sources if s.similarity_score > 0.7]
        has_relevant_sources = len(relevant_sources) > 0
        
        # Query is relevant if it has Picarro keywords OR has highly relevant sources
        return has_picarro_keywords or has_relevant_sources

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
        
        # Create the prompt with strict domain restrictions
        prompt = f"""You are an AI assistant specifically for Picarro, a company that specializes in precision gas analyzers and environmental monitoring solutions. 

IMPORTANT: You can ONLY answer questions related to Picarro's products, technology, services, and environmental monitoring applications. You MUST refuse to answer any questions that are not related to Picarro or environmental monitoring.

If the user asks about topics unrelated to Picarro (such as general knowledge, other companies, geography, history, etc.), politely decline and redirect them to Picarro-related topics.

Context Information:
{context}

User Question: {query}

Instructions:
1. If the question is NOT related to Picarro or environmental monitoring, respond with:
   "I'm sorry, but I can only answer questions related to Picarro's precision gas analyzers and environmental monitoring solutions. Your question appears to be outside my area of expertise. Please ask me about Picarro's products, technology, or environmental monitoring applications."

2. If the question IS related to Picarro, provide a comprehensive answer based on the context information.

3. Always stay focused on Picarro's technology, products, and environmental monitoring capabilities.

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
            
            # Convert search results to Source objects
            sources = []
            for result in search_results:
                source = Source(
                    content=result["content"],
                    metadata=result["metadata"],
                    similarity_score=result["similarity_score"],
                    rank=result["rank"]
                )
                sources.append(source)
            
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


def test_ai_responder():
    """
    Test function to demonstrate the AIResponder functionality.
    """
    print("=== Testing AIResponder ===")
    
    try:
        # Initialize document processor and AI responder
        doc_processor = DocumentProcessor()
        ai_responder = AIResponder(doc_processor)
        
        # Sample documents about Picarro analyzers
        sample_documents = [
            {
                "content": """
                Picarro's G2301 gas analyzer is a high-precision instrument designed for measuring carbon dioxide (CO2), methane (CH4), and water vapor (H2O) concentrations. 
                It uses cavity ring-down spectroscopy (CRDS) technology to achieve parts-per-billion (ppb) detection limits. 
                The G2301 is widely used in atmospheric research, greenhouse gas monitoring, and industrial emissions tracking.
                """,
                "metadata": {
                    "product": "G2301",
                    "category": "gas_analyzer",
                    "technology": "CRDS",
                    "applications": "atmospheric_research,greenhouse_gas_monitoring"
                }
            },
            {
                "content": """
                The Picarro G2401 analyzer measures carbon monoxide (CO), carbon dioxide (CO2), methane (CH4), and water vapor (H2O) simultaneously. 
                This analyzer is particularly useful for air quality monitoring and urban pollution studies. 
                It features real-time data logging and can operate continuously for extended periods with minimal maintenance.
                """,
                "metadata": {
                    "product": "G2401",
                    "category": "gas_analyzer",
                    "technology": "CRDS",
                    "applications": "air_quality_monitoring,urban_pollution_studies"
                }
            },
            {
                "content": """
                Picarro's isotopic analyzers, such as the G2131-i, provide high-precision measurements of stable isotopes in carbon dioxide and water vapor. 
                These instruments are essential for understanding carbon cycling, ecosystem studies, and climate research. 
                The isotopic data helps scientists trace the sources and sinks of greenhouse gases.
                """,
                "metadata": {
                    "product": "G2131-i",
                    "category": "isotopic_analyzer",
                    "technology": "CRDS",
                    "applications": "carbon_cycling,ecosystem_studies,climate_research"
                }
            },
            {
                "content": """
                Cavity Ring-Down Spectroscopy (CRDS) is the core technology behind Picarro's analyzers. 
                This technique measures the time it takes for light to decay in an optical cavity, providing extremely precise concentration measurements. 
                CRDS technology offers superior stability, accuracy, and sensitivity compared to traditional spectroscopic methods.
                """,
                "metadata": {
                    "category": "technology",
                    "technology": "CRDS",
                    "description": "core_technology"
                }
            },
            {
                "content": """
                Picarro analyzers are deployed worldwide in various applications including environmental monitoring networks, 
                industrial process control, agricultural research, and scientific laboratories. 
                The instruments are known for their reliability in harsh environmental conditions and their ability to provide 
                continuous, unattended operation for months at a time.
                """,
                "metadata": {
                    "category": "applications",
                    "description": "deployment_info"
                }
            }
        ]
        
        # Add documents to knowledge base
        print("Adding sample documents to knowledge base...")
        doc_ids = []
        for doc in sample_documents:
            doc_id = ai_responder.add_document(doc["content"], doc["metadata"])
            doc_ids.append(doc_id)
            print(f"Added document: {doc_id}")
        
        # Test queries
        test_queries = [
            "What is the G2301 analyzer used for?",
            "How does CRDS technology work?",
            "What are the applications of Picarro analyzers?",
            "Tell me about isotopic measurements",
            "What makes Picarro analyzers reliable?"
        ]
        
        print(f"\nTesting {len(test_queries)} different queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            try:
                response = ai_responder.respond(query)
                
                print(f"Answer: {response.answer}")
                print(f"Response time: {response.response_time:.2f}s")
                print(f"Confidence: {response.confidence_score:.3f}")
                print(f"Sources used: {len(response.sources)}")
                
                for j, source in enumerate(response.sources, 1):
                    print(f"  Source {j} (Score: {source.similarity_score:.3f}):")
                    print(f"    Product: {source.metadata.get('product', 'N/A')}")
                    print(f"    Category: {source.metadata.get('category', 'N/A')}")
                    print(f"    Content: {source.content[:100]}...")
                
            except Exception as e:
                print(f"Error processing query: {e}")
        
        # Get knowledge base statistics
        print(f"\n--- Knowledge Base Statistics ---")
        stats = ai_responder.get_knowledge_base_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Clean up
        print(f"\nCleaning up - removing test documents...")
        for doc_id in doc_ids:
            try:
                doc_processor.delete_document(doc_id)
                print(f"Deleted document: {doc_id}")
            except Exception as e:
                print(f"Error deleting document {doc_id}: {e}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    test_ai_responder() 