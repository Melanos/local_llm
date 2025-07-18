#!/usr/bin/env python3
"""
Document Training Script - Process and add documents to RAG database
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.document_trainer import DocumentTrainer
from config import ensure_directories

def main():
    """Main entry point"""
    print("📚 Local RAG System - Document Training")
    print("=" * 45)
    
    # Ensure directories exist
    ensure_directories()
    
    # Start document training
    trainer = DocumentTrainer()
    trainer.train_documents()

if __name__ == "__main__":
    main()
