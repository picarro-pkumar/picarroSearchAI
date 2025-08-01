#!/usr/bin/env python3
"""
Picarro Confluence Connector
Connects to real Picarro Confluence and syncs content for SearchAI
"""

import os
import logging
import time
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import html2text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfluencePage:
    """Represents a Confluence page with metadata"""
    page_id: str
    title: str
    content: str
    space_key: str
    space_name: str
    author: str
    created_date: str
    last_modified: str
    page_url: str
    tags: List[str]
    version: int
    status: str
    parent_id: Optional[str] = None
    children: List[str] = None

class ConfluenceConnector:
    """Connector for Picarro Confluence API"""
    
    def __init__(self):
        self.base_url = os.getenv('CONFLUENCE_URL')
        self.username = os.getenv('CONFLUENCE_USERNAME')
        self.api_token = os.getenv('CONFLUENCE_API_TOKEN')
        self.spaces = os.getenv('CONFLUENCE_SPACES', 'ENG,SUPPORT,PRODUCTS,DOCS').split(',')
        self.rate_limit_delay = float(os.getenv('CONFLUENCE_RATE_LIMIT_DELAY', '1.0'))
        self.max_retries = int(os.getenv('CONFLUENCE_MAX_RETRIES', '3'))
        self.batch_size = int(os.getenv('CONFLUENCE_BATCH_SIZE', '1000'))
        
        if not all([self.base_url, self.username, self.api_token]):
            raise ValueError("Missing required Confluence environment variables")
        
        # Setup session with retry logic
        self.session = self._setup_session()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # Don't wrap text
        self.html_converter.unicode_snob = True  # Use Unicode characters
        self.html_converter.escape_snob = True  # Escape special characters
        self.html_converter.single_line_break = True  # Single line breaks
        
        # Track sync state
        self.sync_state_file = 'confluence_sync_state.json'
        self.last_sync = self._load_sync_state()
        
        logger.info(f"Confluence connector initialized for spaces: {self.spaces}")
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with authentication and retry logic"""
        session = requests.Session()
        session.auth = (self.username, self.api_token)
        session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _load_sync_state(self) -> Dict[str, Any]:
        """Load last sync state from file"""
        try:
            if os.path.exists(self.sync_state_file):
                with open(self.sync_state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load sync state: {e}")
        return {'last_sync': None, 'page_versions': {}}
    
    def _save_sync_state(self, state: Dict[str, Any]):
        """Save sync state to file"""
        try:
            with open(self.sync_state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save sync state: {e}")
    
    def _rate_limit(self):
        """Respect rate limits"""
        time.sleep(self.rate_limit_delay)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Confluence API"""
        url = f"{self.base_url}/rest/api/{endpoint}"
        try:
            self._rate_limit()
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            raise
    
    def get_spaces(self) -> List[Dict]:
        """Get all accessible spaces"""
        # Get all spaces in one call with a large limit
        params = {
            'limit': 1000,
            'type': 'global'
        }
        
        data = self._make_request('space', params)
        spaces = data['results']
        
        logger.info(f"Retrieved {len(spaces)} total spaces from API")
        
        # Filter to specified spaces
        filtered_spaces = [s for s in spaces if s['key'] in self.spaces]
        logger.info(f"Found {len(filtered_spaces)} target spaces: {[s['key'] for s in filtered_spaces]}")
        return filtered_spaces
    
    def get_pages_in_space(self, space_key: str, since_date: Optional[str] = None) -> List[Dict]:
        """Get all pages in a space, optionally filtered by date"""
        pages = []
        start = 0
        
        while True:
            params = {
                'spaceKey': space_key,
                'start': start,
                'limit': self.batch_size,
                'expand': 'version,history,metadata.labels'
            }
            
            if since_date:
                params['created'] = f">={since_date}"
            
            data = self._make_request('content', params)
            pages.extend(data['results'])
            
            if not data.get('_links', {}).get('next'):
                break
            start += self.batch_size
        
        logger.info(f"Found {len(pages)} pages in space {space_key}")
        return pages
    
    def get_page_content(self, page_id: str) -> Optional[Dict]:
        """Get full page content with body"""
        try:
            params = {
                'expand': 'body.storage,version,history,metadata.labels,space'
            }
            return self._make_request(f'content/{page_id}', params)
        except Exception as e:
            logger.error(f"Failed to get content for page {page_id}: {e}")
            return None
    
    def clean_html_content(self, html_content: str) -> str:
        """Clean and convert HTML content to plain text with enhanced Confluence support"""
        if not html_content:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Enhanced handling for Confluence-specific elements
        
        # 1. Handle Confluence macro content (like code blocks, info panels)
        for macro in soup.find_all(['ac:structured-macro', 'ac:plain-text-body']):
            content = macro.get_text().strip()
            if content:
                # Convert to markdown code block
                macro.replace_with(BeautifulSoup(f"\n```\n{content}\n```\n", 'html.parser'))
        
        # 2. Handle Confluence code blocks
        for code_block in soup.find_all(['pre', 'code', 'ac:code']):
            content = code_block.get_text().strip()
            if content:
                if code_block.name == 'pre' or 'ac:code' in str(code_block):
                    # Convert to markdown code block
                    code_block.replace_with(BeautifulSoup(f"\n```\n{content}\n```\n", 'html.parser'))
                else:
                    # Convert inline code to markdown
                    code_block.replace_with(BeautifulSoup(f"`{content}`", 'html.parser'))
        
        # 3. Handle Confluence tables with enhanced API endpoint detection
        for table in soup.find_all('table'):
            try:
                # Check if this table contains API endpoint information
                table_text = table.get_text().lower()
                if any(keyword in table_text for keyword in ['method', 'path', 'endpoint', 'post', 'get', 'patch', 'delete', 'v1/']):
                    # Convert API table to structured markdown
                    markdown_table = self._convert_api_table_to_markdown(table)
                    table.replace_with(BeautifulSoup(markdown_table, 'html.parser'))
                else:
                    # Convert regular table to markdown
                    markdown_table = self._convert_table_to_markdown(table)
                    table.replace_with(BeautifulSoup(markdown_table, 'html.parser'))
            except Exception as e:
                logger.warning(f"Failed to convert table to markdown: {e}")
        
        # 4. Handle Confluence lists and structured content
        for list_elem in soup.find_all(['ul', 'ol']):
            # Ensure proper spacing for lists
            list_elem.insert_before(BeautifulSoup('\n', 'html.parser'))
            list_elem.insert_after(BeautifulSoup('\n', 'html.parser'))
        
        # 5. Handle Confluence headings and sections
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            # Ensure proper spacing around headings
            heading.insert_before(BeautifulSoup('\n', 'html.parser'))
            heading.insert_after(BeautifulSoup('\n', 'html.parser'))
        
        # 6. Preserve structured content like API endpoints in paragraphs
        for element in soup.find_all(['p', 'div', 'span', 'ac:structured-macro']):
            text = element.get_text().strip()
            if text:
                # Look for API endpoint patterns
                if any(pattern in text.lower() for pattern in [
                    'method:', 'path:', 'endpoint:', 'post', 'get', 'patch', 'delete',
                    'v1/tenant', 'v1/site', 'v1/monitoring', 'api/v1'
                ]):
                    # Ensure API endpoint details are preserved with proper formatting
                    element.string = text
                    element.insert_before(BeautifulSoup('\n', 'html.parser'))
                    element.insert_after(BeautifulSoup('\n', 'html.parser'))
        
        # Convert to markdown
        markdown = self.html_converter.handle(str(soup))
        
        # Clean up extra whitespace but preserve structure
        markdown = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown)
        markdown = re.sub(r' +', ' ', markdown)
        
        # Ensure proper spacing around code blocks
        markdown = re.sub(r'```\s*\n', '```\n', markdown)
        markdown = re.sub(r'\n\s*```', '\n```', markdown)
        
        # Preserve API endpoint patterns
        markdown = re.sub(r'(Method:\s*)([A-Z]+)', r'\1**\2**', markdown)
        markdown = re.sub(r'(PATH:\s*)([^\n]+)', r'\1**\2**', markdown)
        markdown = re.sub(r'(Endpoint:\s*)([^\n]+)', r'\1**\2**', markdown)
        
        return markdown.strip()
    
    def _convert_table_to_markdown(self, table) -> str:
        """Convert HTML table to markdown format"""
        markdown_lines = []
        
        # Find all rows
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        for i, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            
            # Convert cells to markdown
            cell_texts = [cell.get_text().strip() for cell in cells]
            markdown_line = '| ' + ' | '.join(cell_texts) + ' |'
            markdown_lines.append(markdown_line)
            
            # Add separator line after header row
            if i == 0:
                separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                markdown_lines.append(separator)
        
        return '\n'.join(markdown_lines)
    
    def _convert_api_table_to_markdown(self, table) -> str:
        """Convert API endpoint table to structured markdown format"""
        markdown_lines = []
        
        # Find all rows
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        for i, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            
            # Convert cells to markdown with enhanced formatting for API endpoints
            cell_texts = []
            for cell in cells:
                text = cell.get_text().strip()
                # Bold HTTP methods
                if text.upper() in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
                    text = f"**{text}**"
                # Bold API paths
                elif text.startswith('v1/') or text.startswith('/api/'):
                    text = f"**{text}**"
                cell_texts.append(text)
            
            markdown_line = '| ' + ' | '.join(cell_texts) + ' |'
            markdown_lines.append(markdown_line)
            
            # Add separator line after header row
            if i == 0:
                separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                markdown_lines.append(separator)
        
        return '\n'.join(markdown_lines)
    
    def extract_metadata(self, page_data: Dict) -> Dict[str, Any]:
        """Extract metadata from page data"""
        history = page_data.get('history', {})
        version = page_data.get('version', {})
        space = page_data.get('space', {})
        labels = page_data.get('metadata', {}).get('labels', {}).get('results', [])
        
        return {
            'page_id': page_data['id'],
            'title': page_data['title'],
            'space_key': space.get('key', ''),
            'space_name': space.get('name', ''),
            'author': history.get('createdBy', {}).get('displayName', 'Unknown'),
            'created_date': history.get('createdDate', ''),
            'last_modified': version.get('when', ''),
            'page_url': f"{self.base_url}/pages/viewpage.action?pageId={page_data['id']}",
            'tags': ', '.join([label['name'] for label in labels]) if labels else '',
            'version': version.get('number', 1),
            'status': page_data.get('status', 'current'),
            'parent_id': page_data.get('ancestors', [{}])[-1].get('id') if page_data.get('ancestors') else None
        }
    
    def should_sync_page(self, page_data: Dict) -> bool:
        """Determine if page should be synced based on last sync state"""
        page_id = page_data['id']
        current_version = page_data.get('version', {}).get('number', 1)
        last_version = self.last_sync.get('page_versions', {}).get(page_id, 0)
        
        return current_version > last_version
    
    def sync_space(self, space_key: str) -> List[ConfluencePage]:
        """Sync all pages in a space"""
        logger.info(f"Starting sync for space: {space_key}")
        
        # Get pages in space
        pages_data = self.get_pages_in_space(space_key)
        synced_pages = []
        
        for i, page_data in enumerate(pages_data, 1):
            try:
                # Check if page needs syncing
                if not self.should_sync_page(page_data):
                    logger.debug(f"Skipping page {page_data['id']} - no changes")
                    continue
                
                logger.info(f"Processing page {i}/{len(pages_data)}: {page_data['title']}")
                
                # Get full page content
                full_page_data = self.get_page_content(page_data['id'])
                if not full_page_data:
                    continue
                
                # Extract content and metadata
                body_content = full_page_data.get('body', {}).get('storage', {}).get('value', '')
                clean_content = self.clean_html_content(body_content)
                
                if not clean_content.strip():
                    logger.warning(f"Page {page_data['id']} has no content")
                    continue
                
                metadata = self.extract_metadata(full_page_data)
                
                # Create ConfluencePage object
                page = ConfluencePage(
                    content=clean_content,
                    **metadata
                )
                
                synced_pages.append(page)
                
                # Update sync state
                self.last_sync['page_versions'][page_data['id']] = page_data.get('version', {}).get('number', 1)
                
                logger.info(f"Successfully processed page: {page.title}")
                
            except Exception as e:
                logger.error(f"Failed to process page {page_data.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Completed sync for space {space_key}: {len(synced_pages)} pages synced")
        return synced_pages
    
    def sync_all_spaces(self) -> List[ConfluencePage]:
        """Sync all configured spaces"""
        logger.info("Starting full Confluence sync")
        
        all_pages = []
        spaces = self.get_spaces()
        
        for space in spaces:
            try:
                space_pages = self.sync_space(space['key'])
                all_pages.extend(space_pages)
            except Exception as e:
                logger.error(f"Failed to sync space {space['key']}: {e}")
                continue
        
        # Update sync state
        self.last_sync['last_sync'] = datetime.now().isoformat()
        self._save_sync_state(self.last_sync)
        
        logger.info(f"Confluence sync completed: {len(all_pages)} total pages synced")
        return all_pages
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get sync statistics"""
        return {
            'last_sync': self.last_sync.get('last_sync'),
            'total_pages_tracked': len(self.last_sync.get('page_versions', {})),
            'spaces_configured': self.spaces,
            'rate_limit_delay': self.rate_limit_delay,
            'max_retries': self.max_retries
        }

def main():
    """Main function for testing the connector"""
    try:
        connector = ConfluenceConnector()
        
        # Test connection
        spaces = connector.get_spaces()
        print(f"Connected successfully. Found {len(spaces)} target spaces.")
        
        # Get sync stats
        stats = connector.get_sync_stats()
        print(f"Sync stats: {json.dumps(stats, indent=2)}")
        
        # Sync all spaces
        pages = connector.sync_all_spaces()
        print(f"Synced {len(pages)} pages")
        
    except Exception as e:
        logger.error(f"Confluence sync failed: {e}")
        raise

if __name__ == "__main__":
    main() 