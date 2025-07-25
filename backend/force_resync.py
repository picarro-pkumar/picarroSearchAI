#!/usr/bin/env python3
"""
Force Resync Script
Clears sync state and forces a complete resync of all Confluence data
"""

import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Import required modules
from confluence_connector import ConfluenceConnector
from doc_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_sync_state():
    """Clear the Confluence sync state to force a full resync"""
    try:
        sync_state_file = 'confluence_sync_state.json'
        if os.path.exists(sync_state_file):
            os.remove(sync_state_file)
            logger.info("✅ Cleared Confluence sync state")
        else:
            logger.info("ℹ️  No sync state file found")
        return True
    except Exception as e:
        logger.error(f"Failed to clear sync state: {e}")
        return False

def clear_chromadb():
    """Clear all documents from ChromaDB"""
    try:
        doc_processor = DocumentProcessor()
        doc_processor.clear_documents()
        logger.info("✅ Cleared all documents from ChromaDB")
        return True
    except Exception as e:
        logger.error(f"Failed to clear ChromaDB: {e}")
        return False

def main():
    """Main function to force a complete resync"""
    logger.info("🔄 Starting Force Resync Process")
    logger.info("=" * 50)
    
    try:
        # Step 1: Clear sync state
        logger.info("Step 1: Clearing sync state...")
        if not clear_sync_state():
            logger.error("❌ Failed to clear sync state")
            return False
        
        # Step 2: Clear ChromaDB
        logger.info("Step 2: Clearing ChromaDB...")
        if not clear_chromadb():
            logger.error("❌ Failed to clear ChromaDB")
            return False
        
        # Step 3: Run full migration
        logger.info("Step 3: Running full migration...")
        from confluence_migrate import ConfluenceMigrator
        
        migrator = ConfluenceMigrator()
        results = migrator.run_migration()
        
        # Print results
        print("\n" + "="*60)
        print("FORCE RESYNC RESULTS")
        print("="*60)
        
        if results.get('error'):
            print(f"❌ Force resync failed: {results['error']}")
            return False
        
        summary = results.get('summary', {})
        if summary.get('success'):
            print("✅ Force resync completed successfully!")
            print(f"📊 Target Space: {summary.get('target_space')}")
            print(f"📊 Pages Processed: {summary.get('pages_processed')}")
            print(f"📊 Pages Successful: {summary.get('pages_successful')}")
            print(f"📊 Pages Failed: {summary.get('pages_failed')}")
            print(f"📊 Pages Updated: {summary.get('pages_updated')}")
            print(f"📊 Pages Skipped: {summary.get('pages_skipped')}")
            print(f"📊 Total Chunks: {summary.get('total_chunks')}")
            print(f"⏱️  Duration: {summary.get('duration_minutes', 0):.1f} minutes")
            
            # Save results
            with open('force_resync_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: force_resync_results.json")
            
            return True
        else:
            print("⚠️  Force resync completed with errors")
            print(f"📊 Pages Failed: {summary.get('pages_failed')}")
            return False
            
    except Exception as e:
        logger.error(f"Force resync failed: {e}")
        print(f"\n❌ Force resync failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 