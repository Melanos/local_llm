#!/usr/bin/env python3
"""
Test script for new embedding models: MiniLM and CLIP
"""

import sys
from pathlib import Path
import time
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.enhanced_rag_engine import EnhancedRAGEngine
from config import DEFAULT_CONFIG

def test_embedding_model(model_key: str, model_name: str):
    """Test a specific embedding model"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model_name} ({model_key})")
    print(f"{'='*60}")
    
    try:
        # Initialize engine
        start_time = time.time()
        engine = EnhancedRAGEngine(embedding_model_key=model_key)
        init_time = time.time() - start_time
        
        print(f"✅ Model initialized in {init_time:.2f}s")
        print(f"📊 Collection info: {engine.get_collection_info()}")
        print(f"🖼️  Supports images: {engine.supports_images}")
        
        # Test text embedding
        test_text = "This is a test document about machine learning and artificial intelligence."
        start_time = time.time()
        
        # Add a test document
        engine.add_documents([test_text], [{"test": True, "model": model_key}])
        add_time = time.time() - start_time
        print(f"✅ Document added in {add_time:.3f}s")
        
        # Test search
        start_time = time.time()
        results = engine.search("machine learning", n_results=3)
        search_time = time.time() - start_time
        print(f"✅ Search completed in {search_time:.3f}s")
        print(f"📝 Found {len(results)} results")
        
        # Display first result
        if results:
            print(f"🎯 Top result similarity: {results[0]['similarity']:.4f}")
            print(f"📄 Content preview: {results[0]['content'][:100]}...")
        
        # Test image capabilities for CLIP
        if engine.supports_images:
            print(f"\n🖼️  Testing multimodal capabilities...")
            
            # Look for test images
            image_dir = project_root / "data" / "images"
            if image_dir.exists():
                image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
                
                if image_files:
                    test_image = str(image_files[0])
                    print(f"📷 Testing with image: {os.path.basename(test_image)}")
                    
                    start_time = time.time()
                    success = engine.add_image(test_image)
                    if success:
                        image_time = time.time() - start_time
                        print(f"✅ Image processed in {image_time:.3f}s")
                        
                        # Test multimodal search
                        multimodal_results = engine.search_multimodal("network diagram", n_results=5)
                        print(f"🔍 Multimodal search found {len(multimodal_results)} results")
                        
                        for i, result in enumerate(multimodal_results[:3]):
                            result_type = result.get('type', 'text')
                            similarity = result.get('similarity', 0)
                            print(f"  {i+1}. [{result_type}] Similarity: {similarity:.4f}")
                    else:
                        print("❌ Failed to process image")
                else:
                    print("ℹ️  No test images found in data/images/")
            else:
                print("ℹ️  Image directory not found")
        
        print(f"✅ {model_name} test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all new embedding models"""
    print("🚀 Testing New Embedding Models")
    print("=" * 60)
    
    # Models to test
    test_models = [
        ("minilm", "all-MiniLM-L6-v2"),
        ("clip", "CLIP ViT-B/32"),
        ("jina_v4", "Jina v4"),  # Also test existing model
        ("nomic", "Nomic Embed")  # Also test existing model
    ]
    
    results = {}
    
    for model_key, model_name in test_models:
        if model_key in DEFAULT_CONFIG["models"]["embedding_options"]:
            success = test_embedding_model(model_key, model_name)
            results[model_key] = success
        else:
            print(f"⚠️  Model {model_key} not found in config, skipping...")
            results[model_key] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for model_key, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        model_name = next((name for key, name in test_models if key == model_key), model_key)
        print(f"{model_name:20} | {status}")
    
    successful_models = sum(1 for success in results.values() if success)
    total_models = len(results)
    
    print(f"\nOverall: {successful_models}/{total_models} models working")
    
    if successful_models > 0:
        print(f"\n🎉 Ready for comparative benchmarking!")
        print(f"💡 Next steps:")
        print(f"   1. Run quality comparison across all models")
        print(f"   2. Test multimodal search with CLIP")
        print(f"   3. Performance benchmarking")
    else:
        print(f"\n⚠️  No models are working. Check dependencies and configuration.")

if __name__ == "__main__":
    main()
