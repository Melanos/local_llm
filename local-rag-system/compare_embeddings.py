#!/usr/bin/env python3
"""
Embedding Model Comparison Script
Compare Nomic vs Jina embedding models for network diagram analysis
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.core.enhanced_rag_engine import EnhancedRAGEngine

def test_embedding_models():
    """Compare different embedding models on network diagram analysis"""
    
    print("🔬 Embedding Model Comparison for Network Diagram Analysis")
    print("=" * 70)
    
    # Test queries relevant to network diagrams
    test_queries = [
        "What devices are shown in the network?",
        "Describe the network topology and connections",
        "What IP addresses and subnets are visible?", 
        "What security features are present?",
        "How many routers and switches are there?"
    ]
    
    # Models to test
    models_to_test = [
        ("nomic", "Ollama Nomic-Embed-Text"),
        ("jina_base", "Jina Embeddings v2 Base"),
        ("jina_small", "Jina Embeddings v2 Small")
    ]
    
    results = {}
    
    for model_key, model_name in models_to_test:
        print(f"\n🧪 Testing {model_name}")
        print("-" * 50)
        
        try:
            # Initialize RAG engine with specific embedding model
            start_time = time.time()
            engine = EnhancedRAGEngine(embedding_model_key=model_key)
            init_time = time.time() - start_time
            
            # Get collection info
            info = engine.get_collection_info()
            print(f"📊 Collection: {info.get('collection_name', 'Unknown')}")
            print(f"📚 Documents: {info.get('document_count', 0)}")
            print(f"⏱️  Init time: {init_time:.2f}s")
            
            # Test search performance
            model_results = {
                "init_time": init_time,
                "document_count": info.get('document_count', 0),
                "queries": {}
            }
            
            for query in test_queries:
                print(f"\n🔍 Query: {query[:50]}...")
                
                start_time = time.time()
                search_results = engine.search_documents(query, n_results=3)
                search_time = time.time() - start_time
                
                # Analyze results
                documents = search_results.get('documents', [[]])[0]
                distances = search_results.get('distances', [[]])[0]
                
                if documents:
                    avg_distance = sum(distances) / len(distances) if distances else 0
                    max_doc_length = max(len(doc) for doc in documents) if documents else 0
                    
                    print(f"  📄 Found: {len(documents)} documents")
                    print(f"  📏 Avg distance: {avg_distance:.4f}")
                    print(f"  ⏱️  Search time: {search_time:.3f}s")
                    
                    model_results["queries"][query] = {
                        "search_time": search_time,
                        "num_results": len(documents),
                        "avg_distance": avg_distance,
                        "max_doc_length": max_doc_length
                    }
                else:
                    print("  ❌ No results found")
                    model_results["queries"][query] = {
                        "search_time": search_time,
                        "num_results": 0,
                        "avg_distance": 0,
                        "max_doc_length": 0
                    }
            
            results[model_key] = model_results
            print(f"\n✅ {model_name} testing complete")
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            results[model_key] = {"error": str(e)}
    
    # Generate comparison report
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    # Performance summary
    print("\n🚀 Performance Summary:")
    print("-" * 30)
    for model_key, model_name in models_to_test:
        if model_key in results and "error" not in results[model_key]:
            data = results[model_key]
            avg_search_time = sum(q["search_time"] for q in data["queries"].values()) / len(data["queries"])
            total_results = sum(q["num_results"] for q in data["queries"].values())
            
            print(f"{model_name}:")
            print(f"  Init time: {data['init_time']:.2f}s")
            print(f"  Avg search: {avg_search_time:.3f}s")
            print(f"  Total results: {total_results}")
            print()
    
    # Quality comparison (by distance scores)
    print("🎯 Quality Comparison (Lower distance = better match):")
    print("-" * 50)
    for query in test_queries:
        print(f"\nQuery: {query}")
        for model_key, model_name in models_to_test:
            if (model_key in results and 
                "error" not in results[model_key] and 
                query in results[model_key]["queries"]):
                
                query_data = results[model_key]["queries"][query]
                avg_dist = query_data["avg_distance"]
                num_results = query_data["num_results"]
                
                print(f"  {model_name}: {avg_dist:.4f} ({num_results} results)")
    
    # Recommendations
    print("\n🏆 Recommendations:")
    print("-" * 20)
    
    # Find best performing models
    if results:
        valid_models = [(k, v) for k, v in results.items() if "error" not in v]
        
        if valid_models:
            # Best for speed
            fastest_init = min(valid_models, key=lambda x: x[1]["init_time"])
            print(f"⚡ Fastest initialization: {dict(models_to_test)[fastest_init[0]]}")
            
            # Best for retrieval quality (lowest average distance)
            model_avg_distances = {}
            for model_key, data in valid_models:
                if data["queries"]:
                    avg_dist = sum(q["avg_distance"] for q in data["queries"].values()) / len(data["queries"])
                    model_avg_distances[model_key] = avg_dist
            
            if model_avg_distances:
                best_quality = min(model_avg_distances.items(), key=lambda x: x[1])
                print(f"🎯 Best retrieval quality: {dict(models_to_test)[best_quality[0]]} (avg dist: {best_quality[1]:.4f})")
    
    return results

def generate_detailed_report(results):
    """Generate a detailed markdown report"""
    
    report_path = Path("embedding_comparison_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Embedding Model Comparison Report\n\n")
        f.write(f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Models Tested\n\n")
        f.write("| Model | Type | Description |\n")
        f.write("|-------|------|-------------|\n")
        f.write("| Nomic Embed Text | Ollama | Current production model |\n")
        f.write("| Jina v2 Base | HuggingFace | 768-dim, English optimized |\n")
        f.write("| Jina v2 Small | HuggingFace | 512-dim, faster variant |\n\n")
        
        f.write("## Performance Results\n\n")
        
        # Write detailed results for each model
        for model_key, data in results.items():
            if "error" not in data:
                model_names = {
                    "nomic": "Nomic Embed Text",
                    "jina_base": "Jina v2 Base", 
                    "jina_small": "Jina v2 Small"
                }
                
                f.write(f"### {model_names.get(model_key, model_key)}\n\n")
                f.write(f"- **Initialization time**: {data['init_time']:.2f}s\n")
                f.write(f"- **Document count**: {data['document_count']}\n")
                f.write(f"- **Collection name**: {data.get('collection_name', 'N/A')}\n\n")
                
                f.write("#### Query Performance\n\n")
                f.write("| Query | Search Time | Results | Avg Distance |\n")
                f.write("|-------|-------------|---------|-------------|\n")
                
                for query, query_data in data["queries"].items():
                    f.write(f"| {query[:40]}... | {query_data['search_time']:.3f}s | "
                           f"{query_data['num_results']} | {query_data['avg_distance']:.4f} |\n")
                f.write("\n")
    
    print(f"\n📄 Detailed report saved: {report_path}")

if __name__ == "__main__":
    print("Starting embedding model comparison...")
    results = test_embedding_models()
    generate_detailed_report(results)
    print("\n✅ Comparison complete!")
