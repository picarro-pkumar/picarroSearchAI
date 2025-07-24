#!/usr/bin/env python3
"""
Startup Script with Auto Confluence Sync
Automatically syncs Confluence data before starting the backend server
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_confluence_sync():
    """Run Confluence sync to get latest data"""
    try:
        logger.info("🔄 Starting Confluence sync...")
        
        # Check if confluence_migrate.py exists
        if not Path("confluence_migrate.py").exists():
            logger.error("❌ confluence_migrate.py not found!")
            return False
        
        # Run the Confluence migration
        result = subprocess.run(
            [sys.executable, "confluence_migrate.py"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            logger.info("✅ Confluence sync completed successfully!")
            if result.stdout:
                logger.info(f"Sync output: {result.stdout}")
            return True
        else:
            logger.error(f"❌ Confluence sync failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during Confluence sync: {e}")
        return False

def start_backend_server():
    """Start the backend server"""
    try:
        logger.info("🚀 Starting backend server...")
        
        # Start the server using uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        logger.info("🛑 Backend server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting backend server: {e}")

def main():
    """Main startup function"""
    logger.info("🎯 Picarro SearchAI Startup with Auto-Sync")
    logger.info("=" * 50)
    
    # Step 1: Run Confluence sync
    sync_success = run_confluence_sync()
    
    if not sync_success:
        logger.warning("⚠️  Confluence sync failed, but continuing with startup...")
    
    # Step 2: Start backend server
    start_backend_server()

if __name__ == "__main__":
    main() 