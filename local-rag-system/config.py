"""
Local RAG System - Configuration Module
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = PROJECT_ROOT / "database"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data directories
IMAGES_DIR = DATA_DIR / "images"
DOCUMENTS_DIR = DATA_DIR / "documents" 
ANALYSIS_RESULTS_DIR = DATA_DIR / "analysis_results"

# Default configuration
DEFAULT_CONFIG = {
    "models": {
        "chat_model": "llama3.2",
        "embedding_model": "nomic-embed-text",
        "vision_model": "Salesforce/instructblip-vicuna-7b",
        # Embedding model options for comparison
        "embedding_options": {
            "nomic": "nomic-embed-text",  # Via Ollama
            "jina_base": "jinaai/jina-embeddings-v2-base-en",  # Via HuggingFace
            "jina_small": "jinaai/jina-embeddings-v2-small-en"  # Via HuggingFace
        }
    },
    "ollama": {
        "url": "http://localhost:11434"
    },
    "database": {
        "path": str(DATABASE_DIR / "rag_database"),
        "collection_name": "documents"
    },
    "chunking": {
        "chunk_size": 1000,
        "overlap": 200
    },
    "relevance": {
        "search_threshold": 0.9,
        "chat_threshold": 0.9
    },
    "features": {
        "auto_extract_images": True,   # Set to True to automatically extract images from documents
        "enhanced_image_analysis": True,  # Set to True for multi-pass detailed image analysis
        "vision_analysis_passes": 3  # Number of analysis passes for enhanced mode
    }
}

# Ensure directories exist
def ensure_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR, IMAGES_DIR, DOCUMENTS_DIR, 
        ANALYSIS_RESULTS_DIR, DATABASE_DIR, DOCS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    print("✅ Project directories initialized")

if __name__ == "__main__":
    ensure_directories()
