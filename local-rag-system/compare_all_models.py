#!/usr/bin/env python3
"""
Comprehensive Embedding Model Comparison
Compares Jina v4, Nomic, MiniLM, and CLIP across multiple dimensions
"""

import sys
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.enhanced_rag_engine import EnhancedRAGEngine
from config import DEFAULT_CONFIG

class EmbeddingModelComparator:
    """Compare embedding models across various metrics"""
    
    def __init__(self):
        self.models = {
            "jina_v4": "Jina v4 (2048-dim)",
            "nomic": "Nomic Embed (768-dim)", 
            "minilm": "MiniLM-L6-v2 (384-dim)",
            "clip": "CLIP ViT-B/32 (512-dim)"
        }
        
        self.test_documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning uses multiple layers to model and understand complex patterns in data.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Reinforcement learning trains agents to make decisions through interaction.",
            "Data science combines statistics, programming, and domain expertise.",
            "Big data refers to large, complex datasets that require special tools to process.",
            "Cloud computing provides on-demand access to computing resources over the internet.",
            "Cybersecurity protects digital systems from unauthorized access and attacks."
        ]
        
        self.test_queries = [
            "What is artificial intelligence?",
            "How do neural networks work?",
            "Explain machine learning algorithms",
            "What is computer vision?",
            "Tell me about data processing"
        ]
        
        self.results = {}
    
    def test_model_performance(self, model_key: str, model_name: str) -> Dict[str, Any]:
        """Test a single model's performance"""
        print(f"\n🧪 Testing {model_name}")
        print("-" * 50)
        
        results = {
            "model_key": model_key,
            "model_name": model_name,
            "initialization_time": 0,
            "embedding_time": 0,
            "search_time": 0,
            "supports_images": False,
            "search_quality": [],
            "errors": [],
            "status": "pending"
        }
        
        try:
            # Test 1: Initialization time
            print("1️⃣ Testing initialization...")
            start_time = time.time()
            engine = EnhancedRAGEngine(embedding_model_key=model_key)
            results["initialization_time"] = time.time() - start_time
            results["supports_images"] = engine.supports_images
            print(f"   ✅ Initialized in {results['initialization_time']:.2f}s")
            print(f"   🖼️  Image support: {results['supports_images']}")
            
            # Test 2: Document embedding time
            print("2️⃣ Testing document embedding...")
            start_time = time.time()
            engine.add_documents(
                self.test_documents,
                [{"test_batch": True, "doc_id": i} for i in range(len(self.test_documents))]
            )
            results["embedding_time"] = time.time() - start_time
            embed_rate = len(self.test_documents) / results["embedding_time"]
            print(f"   ✅ Embedded {len(self.test_documents)} docs in {results['embedding_time']:.2f}s")
            print(f"   📊 Rate: {embed_rate:.1f} docs/second")
            
            # Test 3: Search performance and quality
            print("3️⃣ Testing search performance...")
            total_search_time = 0
            search_results = []
            
            for query in self.test_queries:
                start_time = time.time()
                query_results = engine.search_documents(query, n_results=5)
                query_time = time.time() - start_time
                total_search_time += query_time
                
                # Analyze search quality
                if query_results and query_results.get("documents") and len(query_results["documents"][0]) > 0:
                    # Process ChromaDB format results
                    documents = query_results["documents"][0]
                    distances = query_results["distances"][0]
                    
                    # Convert distances to similarities (similarity = 1 - distance)
                    similarities = [1 - dist for dist in distances]
                    avg_similarity = sum(similarities) / len(similarities)
                    top_similarity = similarities[0] if similarities else 0
                    
                    search_results.append({
                        "query": query,
                        "results_count": len(documents),
                        "avg_similarity": avg_similarity,
                        "top_similarity": top_similarity,
                        "search_time": query_time
                    })
                else:
                    # No results found
                    search_results.append({
                        "query": query,
                        "results_count": 0,
                        "avg_similarity": 0,
                        "top_similarity": 0,
                        "search_time": query_time
                    })
            
            results["search_time"] = total_search_time / len(self.test_queries)
            results["search_quality"] = search_results
            
            if search_results:
                avg_similarity = sum(r['avg_similarity'] for r in search_results) / len(search_results)
                avg_top_similarity = sum(r['top_similarity'] for r in search_results) / len(search_results)
                
                print(f"   ✅ Average search time: {results['search_time']:.3f}s")
                print(f"   🎯 Average similarity: {avg_similarity:.4f}")
                print(f"   🥇 Average top similarity: {avg_top_similarity:.4f}")
            else:
                print("   ⚠️  No search results obtained")
            
            # Test 4: Image capabilities (if supported)
            if results["supports_images"]:
                print("4️⃣ Testing multimodal capabilities...")
                
                # Look for test images
                image_dir = project_root / "data" / "images"
                if image_dir.exists():
                    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
                    
                    if image_files:
                        test_image = str(image_files[0])
                        start_time = time.time()
                        success = engine.add_image(test_image)
                        image_time = time.time() - start_time
                        
                        if success:
                            print(f"   ✅ Image processed in {image_time:.3f}s")
                            
                            # Test multimodal search
                            multimodal_results = engine.search_multimodal("network diagram", n_results=5)
                            print(f"   🔍 Multimodal search: {len(multimodal_results)} results")
                            results["multimodal_test"] = {
                                "image_processing_time": image_time,
                                "multimodal_results_count": len(multimodal_results)
                            }
                        else:
                            print("   ❌ Failed to process image")
                    else:
                        print("   ℹ️  No test images found")
                else:
                    print("   ℹ️  Image directory not found")
            
            results["status"] = "success"
            print(f"✅ {model_name} test completed")
            
        except Exception as e:
            error_msg = f"Error testing {model_name}: {e}"
            results["errors"].append(error_msg)
            results["status"] = "failed"
            print(f"❌ {error_msg}")
            
        return results
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison of all models"""
        print("🚀 COMPREHENSIVE EMBEDDING MODEL COMPARISON")
        print("=" * 60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 Test documents: {len(self.test_documents)}")
        print(f"🔍 Test queries: {len(self.test_queries)}")
        
        comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "documents_count": len(self.test_documents),
                "queries_count": len(self.test_queries)
            },
            "models": {},
            "summary": {}
        }
        
        # Test each model
        for model_key, model_name in self.models.items():
            if model_key in DEFAULT_CONFIG["models"]["embedding_options"]:
                model_results = self.test_model_performance(model_key, model_name)
                comparison_results["models"][model_key] = model_results
            else:
                print(f"⚠️  Model {model_key} not found in config, skipping...")
        
        # Generate summary
        self.generate_summary(comparison_results)
        
        return comparison_results
    
    def generate_summary(self, results: Dict[str, Any]):
        """Generate comparison summary"""
        print(f"\n{'='*60}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        successful_models = []
        failed_models = []
        
        # Collect successful models
        for model_key, model_data in results["models"].items():
            if model_data["status"] == "success":
                successful_models.append(model_data)
            else:
                failed_models.append(model_data)
        
        if not successful_models:
            print("❌ No models tested successfully")
            return
        
        # Performance comparison
        print("\n🚀 PERFORMANCE METRICS")
        print("-" * 40)
        print(f"{'Model':<20} {'Init(s)':<8} {'Embed(s)':<9} {'Search(s)':<9} {'Images':<7}")
        print("-" * 40)
        
        for model in successful_models:
            name = model["model_name"].split()[0]
            init_time = model["initialization_time"]
            embed_time = model["embedding_time"]
            search_time = model["search_time"]
            images = "✅" if model["supports_images"] else "❌"
            
            print(f"{name:<20} {init_time:<8.2f} {embed_time:<9.2f} {search_time:<9.3f} {images:<7}")
        
        # Quality comparison
        print("\n🎯 SEARCH QUALITY")
        print("-" * 40)
        print(f"{'Model':<20} {'Avg Sim':<8} {'Top Sim':<8} {'Quality':<8}")
        print("-" * 40)
        
        quality_scores = []
        for model in successful_models:
            if model["search_quality"]:
                avg_similarity = sum(r['avg_similarity'] for r in model["search_quality"]) / len(model["search_quality"])
                avg_top_similarity = sum(r['top_similarity'] for r in model["search_quality"]) / len(model["search_quality"])
                
                # Quality score: weighted average of avg and top similarities
                quality_score = (avg_similarity * 0.6) + (avg_top_similarity * 0.4)
                quality_scores.append((model["model_key"], quality_score))
                
                name = model["model_name"].split()[0]
                quality_rating = "🥇" if quality_score > 0.7 else "🥈" if quality_score > 0.5 else "🥉"
                
                print(f"{name:<20} {avg_similarity:<8.4f} {avg_top_similarity:<8.4f} {quality_rating:<8}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 40)
        
        # Best overall model
        if quality_scores:
            best_model = max(quality_scores, key=lambda x: x[1])
            best_name = next(m["model_name"] for m in successful_models if m["model_key"] == best_model[0])
            print(f"🏆 Best Overall: {best_name} (Quality: {best_model[1]:.4f})")
        
        # Fastest model
        fastest_model = min(successful_models, key=lambda x: x["search_time"])
        print(f"⚡ Fastest Search: {fastest_model['model_name']} ({fastest_model['search_time']:.3f}s)")
        
        # Most efficient (speed + quality balance)
        if quality_scores:
            efficiency_scores = []
            for model in successful_models:
                quality = next(score for key, score in quality_scores if key == model["model_key"])
                speed_score = 1 / (model["search_time"] + 0.001)  # Avoid division by zero
                efficiency = quality * speed_score
                efficiency_scores.append((model["model_key"], efficiency))
            
            best_efficiency = max(efficiency_scores, key=lambda x: x[1])
            efficient_name = next(m["model_name"] for m in successful_models if m["model_key"] == best_efficiency[0])
            print(f"⚖️  Most Efficient: {efficient_name}")
        
        # Multimodal capability
        multimodal_models = [m for m in successful_models if m["supports_images"]]
        if multimodal_models:
            print(f"🖼️  Multimodal: {', '.join(m['model_name'].split()[0] for m in multimodal_models)}")
        
        # Failed models
        if failed_models:
            print(f"\n⚠️  Failed: {', '.join(m['model_name'] for m in failed_models)}")
        
        results["summary"] = {
            "successful_models": len(successful_models),
            "failed_models": len(failed_models),
            "best_quality": best_model[0] if quality_scores else None,
            "fastest": fastest_model["model_key"],
            "multimodal_available": len(multimodal_models) > 0
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save comparison results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embedding_comparison_{timestamp}.json"
        
        filepath = project_root / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run the comprehensive comparison"""
    comparator = EmbeddingModelComparator()
    results = comparator.run_comparison()
    comparator.save_results(results)
    
    print(f"\n🎉 Comparison complete!")
    print(f"📊 Check the JSON file for detailed results")

if __name__ == "__main__":
    main()
