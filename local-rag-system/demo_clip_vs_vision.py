#!/usr/bin/env python3
"""
Demonstrate CLIP's multimodal capabilities vs image-to-text generation
"""

import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.core.enhanced_rag_engine import EnhancedRAGEngine
    from config import DEFAULT_CONFIG
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def demonstrate_clip_capabilities():
    """Show what CLIP can and cannot do"""
    print("🔍 CLIP Multimodal Capabilities Demo")
    print("=" * 50)
    
    # Initialize CLIP engine
    print("🚀 Initializing CLIP engine...")
    engine = EnhancedRAGEngine(embedding_model_key="clip")
    
    if not engine.supports_images:
        print("❌ CLIP not available")
        return
    
    print("✅ CLIP engine initialized")
    print(f"🖼️  Multimodal support: {engine.supports_images}")
    
    # Look for test images
    image_dir = project_root / "data" / "images"
    if not image_dir.exists():
        print("❌ No image directory found")
        return
    
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    
    if not image_files:
        print("❌ No images found")
        return
    
    test_image = str(image_files[0])
    print(f"📷 Testing with: {os.path.basename(test_image)}")
    
    print("\n🔍 What CLIP CAN do:")
    print("-" * 30)
    
    # 1. Generate image embedding
    print("1️⃣ Generate semantic embedding from image...")
    image_embedding = engine.embedding_engine.embed_image(test_image)
    if image_embedding:
        print(f"✅ Created 512-dimensional embedding")
        print(f"📊 First 5 values: {image_embedding[:5]}")
    
    # 2. Add image to database
    print("\n2️⃣ Add image to searchable database...")
    success = engine.add_image(test_image, {"description": "Network diagram"})
    if success:
        print("✅ Image added to vector database")
    
    # 3. Search with text queries
    print("\n3️⃣ Search images using text queries...")
    text_queries = [
        "network diagram",
        "computer infrastructure", 
        "technical drawing",
        "servers and routers"
    ]
    
    for query in text_queries:
        results = engine.search_multimodal(query, n_results=3)
        if results:
            similarity = results[0].get('similarity', 0)
            result_type = results[0].get('type', 'unknown')
            print(f"   🔍 '{query}' → {result_type} (similarity: {similarity:.4f})")
        else:
            print(f"   ❌ '{query}' → No results")
    
    # 4. Cross-modal similarity
    print("\n4️⃣ Compare text and image embeddings...")
    text_embedding = engine.embedding_engine.embed_query("network topology diagram")
    if text_embedding and image_embedding:
        # Calculate cosine similarity
        import numpy as np
        text_vec = np.array(text_embedding)
        image_vec = np.array(image_embedding)
        similarity = np.dot(text_vec, image_vec) / (np.linalg.norm(text_vec) * np.linalg.norm(image_vec))
        print(f"✅ Text-image similarity: {similarity:.4f}")
    
    print("\n❌ What CLIP CANNOT do:")
    print("-" * 30)
    print("❌ Generate text descriptions from images")
    print("❌ Answer questions about image content") 
    print("❌ Create captions or detailed descriptions")
    print("❌ Identify specific objects or text in images")
    
    print("\n💡 For Image-to-Text Generation, you'd need:")
    print("-" * 45)
    print("🔧 LLaVA (recommended for local use)")
    print("🔧 InstructBLIP")
    print("🔧 BLIP-2") 
    print("🔧 GPT-4 Vision API")
    print("🔧 Vicuna with vision capabilities")

if __name__ == "__main__":
    demonstrate_clip_capabilities()
