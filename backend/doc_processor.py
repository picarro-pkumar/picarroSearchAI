import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging3
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    A production-ready document processor that uses sentence-transformers for embeddings
    and ChromaDB for vector storage with semantic search capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chroma_persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the DocumentProcessor.
        
        Args:
            model_name: Name of the sentence transformer model to use
            chroma_persist_directory: Directory to persist ChromaDB data
            chunk_size: Size of text chunks for processing large documents
            chunk_overlap: Overlap between consecutive chunks
        """
        self.model_name = model_name
        self.chroma_persist_directory = chroma_persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize sentence transformer model
        try:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
        
        # Initialize ChromaDB client
        try:
            logger.info(f"Initializing ChromaDB with persist directory: {chroma_persist_directory}")
            self.client = chromadb.PersistentClient(
                path=chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the vector store with chunking for large texts.
        
        Args:
            content: Document content
            metadata: Optional metadata for the document
            document_id: Optional custom document ID
            
        Returns:
            Document ID
        """
        try:
            if not content.strip():
                raise ValueError("Document content cannot be empty")
            
            # Generate document ID if not provided
            if document_id is None:
                document_id = str(uuid.uuid4())
            
            # Set default metadata
            if metadata is None:
                metadata = {}
            
            # Add document info to metadata
            metadata.update({
                "document_id": document_id,
                "content_length": len(content),
                "chunk_count": 0
            })
            
            # Chunk the document
            chunks = self._chunk_text(content)
            logger.info(f"Document {document_id} split into {len(chunks)} chunks")
            
            if not chunks:
                raise ValueError("No valid chunks generated from document")
            
            # Generate embeddings for chunks
            embeddings = self._generate_embeddings(chunks)
            
            # Prepare data for ChromaDB
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "is_chunk": True
                })
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            # Update metadata with chunk count
            metadata["chunk_count"] = len(chunks)
            
            logger.info(f"Successfully added document {document_id} with {len(chunks)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def search_documents(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with content, metadata, and similarity scores
        """
        try:
            if not query.strip():
                raise ValueError("Search query cannot be empty")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - distance
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })
            
            logger.info(f"Search completed for query: '{query}' - found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            raise
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of document chunks with metadata
        """
        try:
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            chunks = []
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                chunks.append({
                    "content": doc,
                    "metadata": metadata,
                    "chunk_index": metadata.get("chunk_index", i)
                })
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Get all chunk IDs for the document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                logger.warning(f"No document found with ID: {document_id}")
                return False
            
            # Delete all chunks
            self.collection.delete(ids=results['ids'])
            
            logger.info(f"Successfully deleted document {document_id} with {len(results['ids'])} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample_results = self.collection.get(limit=1000)
            
            total_chars = sum(len(doc) for doc in sample_results['documents'])
            avg_chunk_size = total_chars / len(sample_results['documents']) if sample_results['documents'] else 0
            
            # Count unique documents
            unique_docs = set()
            for metadata in sample_results['metadatas']:
                if metadata and 'document_id' in metadata:
                    unique_docs.add(metadata['document_id'])
            
            stats = {
                "total_chunks": count,
                "unique_documents": len(unique_docs),
                "average_chunk_size": round(avg_chunk_size, 2),
                "total_characters": total_chars
            }
            
            logger.info(f"Collection stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from the collection.
        
        Returns:
            List of document dictionaries with content and metadata
        """
        try:
            # Get all documents from collection
            results = self.collection.get(limit=10000)  # Large limit to get all
            
            documents = []
            for i in range(len(results['documents'])):
                doc = {
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                    'id': results['ids'][i] if results['ids'] else None
                }
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents from collection")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            raise
    
    def clear_documents(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection("documents")
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("All documents cleared from collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            raise


def test_document_processor():
    """
    Test function to demonstrate the DocumentProcessor functionality.
    """
    print("=== Testing DocumentProcessor ===")
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Test document
        test_content = """
        Picarro is a leading provider of precision gas analyzers and monitoring solutions. 
        Our technology enables real-time measurement of greenhouse gases, air quality, and industrial emissions.
        
        The company was founded in 1998 and has been at the forefront of cavity ring-down spectroscopy (CRDS) technology.
        Our analyzers are used in various applications including environmental monitoring, industrial process control, and scientific research.
        
        Picarro's products are known for their high precision, reliability, and ability to operate in challenging environments.
        We serve customers worldwide across multiple industries including oil and gas, agriculture, and environmental protection.
        """
        
        # Add document
        print("Adding test document...")
        doc_id = processor.add_document(
            content=test_content,
            metadata={
                "source": "company_website",
                "category": "company_info",
                "date_added": "2024-01-01"
            }
        )
        print(f"Document added with ID: {doc_id}")
        
        # Search for documents
        print("\nSearching for 'gas analyzers'...")
        results = processor.search_documents("gas analyzers", n_results=3)
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['similarity_score']:.3f}):")
            print(f"Content: {result['content'][:100]}...")
            print(f"Metadata: {result['metadata']}")
        
        # Search for another query
        print("\nSearching for 'environmental monitoring'...")
        results = processor.search_documents("environmental monitoring", n_results=2)
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['similarity_score']:.3f}):")
            print(f"Content: {result['content'][:100]}...")
        
        # Get document chunks
        print(f"\nRetrieving chunks for document {doc_id}...")
        chunks = processor.get_document_chunks(doc_id)
        print(f"Found {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"Chunk {i+1}: {chunk['content'][:80]}...")
        
        # Get collection stats
        print("\nCollection statistics:")
        stats = processor.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Clean up
        print(f"\nCleaning up - deleting document {doc_id}...")
        processor.delete_document(doc_id)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    test_document_processor() 