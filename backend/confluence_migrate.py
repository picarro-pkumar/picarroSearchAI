#!/usr/bin/env python3
"""
Confluence Migration Script
Syncs FenceLineD space content to ChromaDB for SearchAI
"""

import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Import required modules
from confluence_connector import ConfluenceConnector, ConfluencePage
from doc_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('confluence_migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ConfluenceMigrator:
    """Migrator for Confluence data to ChromaDB"""
    
    def __init__(self):
        self.confluence_connector = ConfluenceConnector()
        self.doc_processor = DocumentProcessor()
        self.target_space = os.getenv('CONFLUENCE_SPACES', 'FenceLineD')
        
        # Migration statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'pages_processed': 0,
            'pages_successful': 0,
            'pages_failed': 0,
            'total_chunks': 0,
            'errors': []
        }
    
    def validate_environment(self) -> Dict[str, any]:
        """Validate environment configuration"""
        logger.info("Validating environment configuration...")
        
        required_vars = [
            'CONFLUENCE_URL',
            'CONFLUENCE_USERNAME', 
            'CONFLUENCE_API_TOKEN',
            'CONFLUENCE_SPACES'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            return {
                'success': False,
                'error': f"Missing environment variables: {', '.join(missing_vars)}"
            }
        
        logger.info("✅ Environment validation passed")
        return {'success': True}
    
    def test_confluence_connection(self) -> Dict[str, any]:
        """Test Confluence connection"""
        logger.info("Testing Confluence connection...")
        
        try:
            # Test basic connection by directly accessing the space
            space_data = self.confluence_connector._make_request(f'space/{self.target_space}')
            
            logger.info(f"✅ Confluence connection successful - found space: {self.target_space}")
            return {
                'success': True,
                'space_info': space_data
            }
            
        except Exception as e:
            logger.error(f"Confluence connection test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_migration_plan(self) -> Dict[str, any]:
        """Get migration plan and statistics"""
        logger.info("Creating migration plan...")
        
        try:
            # Get pages in target space
            pages = self.confluence_connector.get_pages_in_space(self.target_space)
            
            # Analyze pages
            page_types = {}
            total_content_size = 0
            
            for page in pages:
                page_type = page.get('type', 'unknown')
                page_types[page_type] = page_types.get(page_type, 0) + 1
                
                # Estimate content size
                if page.get('body', {}).get('storage', {}).get('value'):
                    total_content_size += len(page['body']['storage']['value'])
            
            plan = {
                'target_space': self.target_space,
                'total_pages': len(pages),
                'page_types': page_types,
                'estimated_content_size_mb': total_content_size / (1024 * 1024),
                'estimated_processing_time_minutes': len(pages) * 2,  # 2 seconds per page
                'pages': pages
            }
            
            logger.info(f"Migration plan created: {len(pages)} pages to process")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create migration plan: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_page(self, page_data: Dict) -> Optional[Dict[str, any]]:
        """Process a single Confluence page"""
        try:
            page_id = page_data['id']
            title = page_data['title']
            
            logger.info(f"Processing page: {title} (ID: {page_id})")
            
            # Get full page content
            full_page_data = self.confluence_connector.get_page_content(page_id)
            if not full_page_data:
                return {
                    'success': False,
                    'page_id': page_id,
                    'title': title,
                    'error': 'Failed to retrieve page content'
                }
            
            # Extract and clean content
            body_content = full_page_data.get('body', {}).get('storage', {}).get('value', '')
            clean_content = self.confluence_connector.clean_html_content(body_content)
            
            if not clean_content.strip():
                return {
                    'success': False,
                    'page_id': page_id,
                    'title': title,
                    'error': 'Page has no content after cleaning'
                }
            
            # Extract metadata
            metadata = self.confluence_connector.extract_metadata(full_page_data)
            
            # Clean metadata - ensure no None values
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is None:
                    cleaned_metadata[key] = ''
                elif isinstance(value, list):
                    cleaned_metadata[key] = ', '.join(str(v) for v in value)
                else:
                    cleaned_metadata[key] = str(value)
            
            # Add to ChromaDB
            doc_id = self.doc_processor.add_document(
                content=clean_content,
                metadata=cleaned_metadata,
                document_id=f"confluence_{page_id}"
            )
            
            logger.info(f"✅ Successfully processed page: {title}")
            
            return {
                'success': True,
                'page_id': page_id,
                'title': title,
                'doc_id': doc_id,
                'content_length': len(clean_content),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to process page {page_data.get('id', 'unknown')}: {e}")
            return {
                'success': False,
                'page_id': page_data.get('id', 'unknown'),
                'title': page_data.get('title', 'Unknown'),
                'error': str(e)
            }
    
    def migrate_space(self, space_key: str) -> Dict[str, any]:
        """Migrate all pages in a space"""
        logger.info(f"Starting migration for space: {space_key}")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # Get all pages in space
            pages = self.confluence_connector.get_pages_in_space(space_key)
            
            if not pages:
                logger.warning(f"No pages found in space: {space_key}")
                return {
                    'success': True,
                    'pages_processed': 0,
                    'pages_successful': 0,
                    'pages_failed': 0,
                    'message': 'No pages to migrate'
                }
            
            logger.info(f"Found {len(pages)} pages to migrate")
            
            # Process each page
            results = []
            for i, page in enumerate(pages, 1):
                logger.info(f"Processing page {i}/{len(pages)}: {page.get('title', 'Unknown')}")
                
                result = self.process_page(page)
                results.append(result)
                
                # Update statistics
                self.stats['pages_processed'] += 1
                if result.get('success'):
                    self.stats['pages_successful'] += 1
                else:
                    self.stats['pages_failed'] += 1
                    self.stats['errors'].append(result.get('error', 'Unknown error'))
                
                # Rate limiting
                time.sleep(float(os.getenv('CONFLUENCE_RATE_LIMIT_DELAY', '1.0')))
            
            # Get final statistics
            final_stats = self.doc_processor.get_collection_stats()
            self.stats['total_chunks'] = final_stats.get('total_chunks', 0)
            self.stats['end_time'] = datetime.now()
            
            logger.info(f"✅ Space migration completed: {self.stats['pages_successful']}/{self.stats['pages_processed']} pages successful")
            
            return {
                'success': True,
                'pages_processed': self.stats['pages_processed'],
                'pages_successful': self.stats['pages_successful'],
                'pages_failed': self.stats['pages_failed'],
                'total_chunks': self.stats['total_chunks'],
                'errors': self.stats['errors'],
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Space migration failed: {e}")
            self.stats['end_time'] = datetime.now()
            return {
                'success': False,
                'error': str(e),
                'pages_processed': self.stats['pages_processed'],
                'pages_successful': self.stats['pages_successful'],
                'pages_failed': self.stats['pages_failed']
            }
    
    def run_migration(self) -> Dict[str, any]:
        """Run complete Confluence migration"""
        logger.info("🚀 Starting Confluence migration process...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: Validate environment
            results['steps']['environment_validation'] = self.validate_environment()
            if not results['steps']['environment_validation']['success']:
                return results
            
            # Step 2: Test connection
            results['steps']['connection_test'] = self.test_confluence_connection()
            if not results['steps']['connection_test']['success']:
                return results
            
            # Step 3: Get migration plan
            results['steps']['migration_plan'] = self.get_migration_plan()
            
            # Step 4: Perform migration
            results['steps']['migration'] = self.migrate_space(self.target_space)
            
            # Step 5: Final statistics
            results['steps']['final_stats'] = self.doc_processor.get_collection_stats()
            
            # Summary
            migration_result = results['steps']['migration']
            results['summary'] = {
                'success': migration_result.get('success', False),
                'target_space': self.target_space,
                'pages_processed': migration_result.get('pages_processed', 0),
                'pages_successful': migration_result.get('pages_successful', 0),
                'pages_failed': migration_result.get('pages_failed', 0),
                'total_chunks': migration_result.get('total_chunks', 0),
                'duration_minutes': (self.stats['end_time'] - self.stats['start_time']).total_seconds() / 60 if self.stats['end_time'] else 0
            }
            
            logger.info("✅ Confluence migration completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            results['error'] = str(e)
            return results

def main():
    """Main function"""
    try:
        migrator = ConfluenceMigrator()
        results = migrator.run_migration()
        
        # Print results
        print("\n" + "="*60)
        print("CONFLUENCE MIGRATION RESULTS")
        print("="*60)
        
        if results.get('error'):
            print(f"❌ Migration failed: {results['error']}")
            return
        
        summary = results.get('summary', {})
        if summary.get('success'):
            print("✅ Migration completed successfully!")
            print(f"📊 Target Space: {summary.get('target_space')}")
            print(f"📊 Pages Processed: {summary.get('pages_processed')}")
            print(f"📊 Pages Successful: {summary.get('pages_successful')}")
            print(f"📊 Pages Failed: {summary.get('pages_failed')}")
            print(f"📊 Total Chunks: {summary.get('total_chunks')}")
            print(f"⏱️  Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        else:
            print("⚠️  Migration completed with errors")
            print(f"📊 Pages Failed: {summary.get('pages_failed')}")
        
        # Save detailed results
        with open('confluence_migration_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: confluence_migration_results.json")
        
    except Exception as e:
        logger.error(f"Confluence migration failed: {e}")
        print(f"\n❌ Confluence migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 