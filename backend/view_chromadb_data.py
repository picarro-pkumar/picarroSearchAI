#!/usr/bin/env python3
"""
View ChromaDB Data Script
Shows all documents stored in the ChromaDB knowledge base
"""

import json
import sys
from doc_processor import DocumentProcessor

def view_chromadb_data():
    """View all data in ChromaDB"""
    print("�� ChromaDB Knowledge Base Viewer")
    print("=" * 50)
    
    try:
        # Initialize document processor
        dp = DocumentProcessor()
        
        # Get collection statistics
        print("📊 Collection Statistics:")
        stats = dp.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
        
        # Get all documents
        print("📄 All Documents in Knowledge Base:")
        print("-" * 50)
        
        all_docs = dp.get_all_documents()
        
        if not all_docs:
            print("❌ No documents found in knowledge base")
            return
        
        # Group documents by source
        documents_by_source = {}
        for doc in all_docs:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'unknown')
            title = metadata.get('title', 'Untitled')
            
            if source not in documents_by_source:
                documents_by_source[source] = []
            
            documents_by_source[source].append({
                'id': doc.get('id'),
                'title': title,
                'content': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                'metadata': metadata
            })
        
        # Display documents by source
        for source, docs in documents_by_source.items():
            print(f"\n📁 Source: {source.upper()}")
            print(f"   Documents: {len(docs)}")
            print("-" * 30)
            
            for i, doc in enumerate(docs, 1):
                print(f"\n{i}. Document ID: {doc['id']}")
                print(f"   Title: {doc['title']}")
                print(f"   Content Preview: {doc['content']}")
                
                # Show key metadata
                metadata = doc['metadata']
                if metadata:
                    print("   Metadata:")
                    for key, value in list(metadata.items())[:5]:  # Show first 5 metadata items
                        if key not in ['content', 'content_length']:
                            print(f"     {key}: {value}")
        
        # Save to JSON file
        output_file = 'chromadb_data_export.json'
        with open(output_file, 'w') as f:
            json.dump({
                'stats': stats,
                'documents': all_docs
            }, f, indent=2)
        
        print(f"\n💾 Data exported to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error viewing ChromaDB data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    view_chromadb_data() 