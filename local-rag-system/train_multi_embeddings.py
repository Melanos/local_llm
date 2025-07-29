#!/usr/bin/env python3
"""
Multi-Embedding Model Training Script
Train the same content with different embedding models for comparison
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.training.image_trainer import ImageTrainer
from src.core.enhanced_rag_engine import EnhancedRAGEngine

def train_with_multiple_embeddings():
    """Train the same content with different embedding models"""
    
    print("🎯 Multi-Embedding Model Training")
    print("Training network diagrams with different embedding models for comparison")
    print("=" * 70)
    
    # Check for training data
    images_dir = Path("data/images")
    if not images_dir.exists():
        print("❌ No images directory found")
        return
    
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    if not image_files:
        print("❌ No image files found in data/images/")
        return
    
    print(f"📷 Found {len(image_files)} images to process")
    
    # Models to train with
    embedding_models = [
        ("nomic", "Ollama Nomic-Embed-Text"),
        ("jina_base", "Jina Embeddings v2 Base"),
        ("jina_small", "Jina Embeddings v2 Small")
    ]
    
    # Get analysis results
    analysis_dir = Path("data/analysis_results")
    analysis_files = list(analysis_dir.glob("*.txt")) if analysis_dir.exists() else []
    
    if not analysis_files:
        print("⚠️  No analysis files found. Running image analysis first...")
        
        # Run image analysis to generate content
        trainer = ImageTrainer()
        for image_file in image_files:
            print(f"🖼️  Analyzing: {image_file.name}")
            trainer.analyze_image(str(image_file))
        
        # Refresh analysis files list
        analysis_files = list(analysis_dir.glob("*.txt")) if analysis_dir.exists() else []
    
    if not analysis_files:
        print("❌ No analysis content available for training")
        return
    
    print(f"📄 Found {len(analysis_files)} analysis files")
    
    # Read all analysis content
    all_documents = []
    all_metadatas = []
    
    for analysis_file in analysis_files:
        print(f"📖 Reading: {analysis_file.name}")
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks for better retrieval
            chunks = split_into_chunks(content)
            
            for i, chunk in enumerate(chunks):
                all_documents.append(chunk)
                all_metadatas.append({
                    "source": analysis_file.name,
                    "chunk_id": i,
                    "type": "image_analysis",
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        except Exception as e:
            print(f"❌ Error reading {analysis_file.name}: {e}")
    
    print(f"📚 Prepared {len(all_documents)} document chunks for training")
    
    # Train with each embedding model
    training_results = {}
    
    for model_key, model_name in embedding_models:
        print(f"\n🚀 Training with {model_name}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            # Initialize RAG engine with specific embedding model
            engine = EnhancedRAGEngine(embedding_model_key=model_key)
            
            # Add documents to the database
            engine.add_documents(
                documents=all_documents,
                metadatas=all_metadatas
            )
            
            training_time = time.time() - start_time
            
            # Get final collection info
            info = engine.get_collection_info()
            
            training_results[model_key] = {
                "model_name": model_name,
                "training_time": training_time,
                "document_count": info.get('document_count', 0),
                "collection_name": info.get('collection_name', 'Unknown')
            }
            
            print(f"✅ Training complete!")
            print(f"   ⏱️  Time: {training_time:.2f}s")
            print(f"   📚 Documents: {info.get('document_count', 0)}")
            print(f"   🗃️  Collection: {info.get('collection_name', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Training failed for {model_name}: {e}")
            training_results[model_key] = {
                "model_name": model_name,
                "error": str(e)
            }
    
    # Generate training summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    
    for model_key, result in training_results.items():
        if "error" not in result:
            print(f"\n✅ {result['model_name']}:")
            print(f"   Training time: {result['training_time']:.2f}s")
            print(f"   Documents stored: {result['document_count']}")
            print(f"   Collection: {result['collection_name']}")
        else:
            print(f"\n❌ {result['model_name']}: {result['error']}")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Run: python compare_embeddings.py")
    print(f"   2. Test queries with different models")
    print(f"   3. Compare retrieval quality and performance")
    
    return training_results

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n\n')
            
            if last_period > chunk_size * 0.7:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
            elif last_newline > chunk_size * 0.5:
                chunk = chunk[:last_newline + 2]
                end = start + last_newline + 2
        
        chunks.append(chunk.strip())
        start = max(start + chunk_size - overlap, end)
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

if __name__ == "__main__":
    results = train_with_multiple_embeddings()
