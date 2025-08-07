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
        "embedding_model": "openai/clip-vit-base-patch32",  # CLIP - Best performance and multimodal
        "vision_model": "Salesforce/instructblip-vicuna-7b",  # InstructBLIP with Vicuna-7B backbone for image-to-text conversion
        # Embedding model options for comparison
        "embedding_options": {
            # Current models
            "jina_v4": "jinaai/jina-embeddings-v4",        # 🥇 Best quality (2048-dim, production)
            "nomic": "nomic-embed-text",                   # 🥈 Privacy/offline (768-dim, local)
            "all_minilm": "sentence-transformers/all-MiniLM-L6-v2",  # 🔧 Baseline (384-dim, fast)
            "clip": "openai/clip-vit-base-patch32",        # 🖼️ Multimodal (512-dim, text+images)
            
            # Additional competitive models for large-scale testing
            "all_mpnet": "sentence-transformers/all-mpnet-base-v2",     # 🏆 SBERT's best (768-dim)
            "bge_large": "BAAI/bge-large-en-v1.5",                     # 🚀 BAAI large (1024-dim)
            "e5_large": "intfloat/e5-large-v2",                        # 🔬 Microsoft E5 (1024-dim)
            "instructor_xl": "hkunlp/instructor-xl",                   # 📚 Instruction-tuned (768-dim)
            "gte_large": "thenlper/gte-large",                         # 🎯 GTE large (1024-dim)
            "bge_m3": "BAAI/bge-m3",                                   # 🌐 Multilingual (1024-dim)
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
