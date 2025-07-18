#!/usr/bin/env python3
"""
Image Training Script - Analyze and add images to RAG database
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.image_trainer import ImageTrainer
from config import ensure_directories

def main():
    """Main entry point"""
    print("🖼️  Local RAG System - Image Training")
    print("=" * 45)
    
    # Ensure directories exist
    ensure_directories()
    
    # Start image training
    trainer = ImageTrainer()
    trainer.train_images()

if __name__ == "__main__":
    main()
