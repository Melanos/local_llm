#!/usr/bin/env python3
"""
Simple test to check which embedding models are available
"""

import sys
from pathlib import Path

print("🔍 Checking Available Embedding Models")
print("=" * 50)

# Test 1: Check for sentence-transformers (for Jina and MiniLM)
try:
    import sentence_transformers
    print("✅ sentence-transformers available")
    
    # Test MiniLM
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = model.encode(["test text"])
        print(f"✅ MiniLM-L6-v2 working (dim: {len(test_embedding[0])})")
    except Exception as e:
        print(f"❌ MiniLM-L6-v2 failed: {e}")
    
    # Test Jina v4
    try:
        model = SentenceTransformer('jinaai/jina-embeddings-v4', 
                                  model_kwargs={'default_task': 'retrieval'},
                                  trust_remote_code=True)
        test_embedding = model.encode(["test text"])
        print(f"✅ Jina v4 working (dim: {len(test_embedding[0])})")
    except Exception as e:
        print(f"❌ Jina v4 failed: {e}")
        
except ImportError:
    print("❌ sentence-transformers not available")

# Test 2: Check for transformers + torch (for CLIP)
try:
    import transformers
    import torch
    from transformers import CLIPProcessor, CLIPModel
    print("✅ transformers and torch available")
    
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Test text embedding
        inputs = processor(text=["test text"], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        print(f"✅ CLIP working (dim: {text_features.shape[1]})")
        print("🖼️ CLIP supports multimodal (text + images)")
        
    except Exception as e:
        print(f"❌ CLIP failed: {e}")
        
except ImportError as e:
    print(f"❌ transformers/torch not available: {e}")

# Test 3: Check for PIL (for image processing)
try:
    from PIL import Image
    print("✅ PIL available for image processing")
except ImportError:
    print("❌ PIL not available")

# Test 4: Check for ChromaDB
try:
    import chromadb
    print("✅ ChromaDB available")
except ImportError:
    print("❌ ChromaDB not available")

print("\n💡 Recommendations:")
print("=" * 50)

print("To enable all models, install in your venv:")
print("  pip install sentence-transformers")
print("  pip install transformers torch torchvision") 
print("  pip install Pillow")
print("  pip install chromadb")

print("\n🎯 Model Comparison Benefits:")
print("=" * 50)
print("• MiniLM-L6-v2: Fast baseline (384-dim)")
print("• Jina v4: High quality (2048-dim)")  
print("• CLIP: Multimodal text+images (512-dim)")
print("• Nomic: Privacy-focused via Ollama (768-dim)")
