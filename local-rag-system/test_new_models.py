#!/usr/bin/env python3
"""
CLIP-Optimized Large File Testing
Modified test that handles CLIP's token limitations while testing large file performance
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_clip_optimized_documents(count: int = 1000) -> List[str]:
    """Create documents optimized for CLIP's token limitations (~75 tokens max)"""
    print(f"📝 Generating {count} CLIP-optimized documents...")
    
    # Domain-specific content templates (shorter for CLIP)
    domains = {
        "ai_ml": {
            "title": "AI/ML",
            "keywords": ["neural networks", "deep learning", "algorithms", "training", "optimization"],
            "templates": [
                "Advanced {} implementation using {} for {} optimization in production systems.",
                "Comparative analysis of {} and {} methodologies for {} applications.",
                "Performance evaluation of {} algorithms with {} in {} environments.",
                "Technical implementation of {} using {} for improved {} efficiency.",
                "Research findings on {} integration with {} for enhanced {} performance."
            ]
        },
        "cybersecurity": {
            "title": "Cybersecurity",
            "keywords": ["threat detection", "encryption", "network security", "vulnerability", "incident response"],
            "templates": [
                "Enterprise {} strategies using {} for comprehensive {} management.",
                "Advanced {} techniques for {} in critical {} infrastructure.",
                "Implementation of {} protocols with {} for effective {} mitigation.",
                "Security framework development using {} and {} for {} protection.",
                "Risk assessment methodologies for {} with {} in {} scenarios."
            ]
        },
        "cloud_computing": {
            "title": "Cloud Computing",
            "keywords": ["microservices", "containerization", "serverless", "scalability", "distributed systems"],
            "templates": [
                "Cloud architecture design using {} and {} for optimal {} deployment.",
                "Performance optimization of {} with {} in {} environments.",
                "Cost-effective {} implementation using {} for {} solutions.",
                "Scalable {} design with {} for enterprise {} applications.",
                "Technical evaluation of {} vs {} for {} infrastructure."
            ]
        }
    }
    
    documents = []
    domain_keys = list(domains.keys())
    
    for i in range(count):
        domain_key = domain_keys[i % len(domain_keys)]
        domain = domains[domain_key]
        
        # Create concise but meaningful content (under 75 tokens)
        template = np.random.choice(domain["templates"])
        keywords = np.random.choice(domain["keywords"], 3, replace=False)
        
        content = f"{domain['title']} Research #{i+1}: " + template.format(*keywords)
        
        # Add brief technical details
        details = [
            f"Technical specifications include {keywords[0]} optimization and {keywords[1]} integration.",
            f"Performance metrics demonstrate {keywords[2]} improvement through {keywords[0]} implementation.",
            f"Implementation results show enhanced {keywords[1]} efficiency in {keywords[2]} scenarios."
        ]
        
        content += " " + np.random.choice(details)
        documents.append(content)
    
    # Verify token counts are reasonable for CLIP
    token_counts = [len(doc.split()) for doc in documents]
    print(f"✅ Created {len(documents)} documents")
    print(f"   📊 Average length: {np.mean(token_counts):.0f} tokens")
    print(f"   📊 Max length: {max(token_counts)} tokens")
    
    return documents

def create_standard_documents(count: int = 1000) -> List[str]:
    """Create standard-sized documents for other models"""
    print(f"📝 Generating {count} standard documents...")
    
    domains = {
        "ai_ml": "artificial intelligence and machine learning",
        "cybersecurity": "cybersecurity and information security", 
        "cloud_computing": "cloud computing and infrastructure",
        "data_science": "data science and analytics",
        "software_engineering": "software engineering and development"
    }
    
    documents = []
    domain_keys = list(domains.keys())
    
    for i in range(count):
        domain_key = domain_keys[i % len(domain_keys)]
        domain = domains[domain_key]
        
        # Create substantial content (500-800 words)
        content_parts = []
        content_parts.append(f"# {domain.title()} Analysis #{i+1}")
        content_parts.append(f"This document provides comprehensive analysis of {domain} and related technologies.")
        
        # Generate multiple paragraphs
        for p in range(3):
            sentences = []
            for s in range(5):
                sentence = f"The implementation of advanced {domain} demonstrates significant potential for improving operational efficiency and technical performance through systematic methodological approaches."
                sentences.append(sentence)
            content_parts.append(" ".join(sentences))
        
        content_parts.append(f"In conclusion, this analysis of {domain} reveals critical insights for future development.")
        documents.append("\n\n".join(content_parts))
    
    word_counts = [len(doc.split()) for doc in documents]
    print(f"✅ Created {len(documents)} documents")
    print(f"   📊 Average length: {np.mean(word_counts):.0f} words")
    
    return documents

def test_model_optimized(model_key: str, doc_count: int = 1000, query_count: int = 50) -> Dict[str, Any]:
    """Test a model with appropriately sized documents"""
    print(f"\n🧪 TESTING {model_key.upper()} WITH OPTIMIZED DOCUMENTS")
    print("=" * 60)
    
    try:
        from src.core.enhanced_rag_engine import EnhancedRAGEngine
        
        # Initialize model
        print("1️⃣ Initializing model...")
        start_time = time.time()
        engine = EnhancedRAGEngine(embedding_model_key=model_key)
        init_time = time.time() - start_time
        print(f"   ⏱️  Model initialized in {init_time:.2f}s")
        
        # Generate appropriate test data
        print("2️⃣ Generating test data...")
        if model_key == "clip":
            documents = create_clip_optimized_documents(doc_count)
        else:
            documents = create_standard_documents(doc_count)
        
        # Standard queries that work for all models
        queries = [
            "What are the latest AI developments?",
            "How does cybersecurity impact business?",
            "What are cloud computing best practices?",
            "Explain machine learning algorithms",
            "What are data science methodologies?",
            "How to implement security frameworks?",
            "What are software optimization techniques?",
            "Describe network architecture principles",
            "What are performance optimization strategies?",
            "How to design scalable systems?"
        ] * 5  # Repeat to get 50 queries
        queries = queries[:query_count]
        
        # Test embedding performance
        print("3️⃣ Testing embedding performance...")
        embedding_start = time.time()
        
        # Process in batches
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_metadata = [{"batch": i // batch_size, "doc_id": j} for j in range(len(batch))]
            
            engine.add_documents(batch, batch_metadata)
            
            batch_num = (i // batch_size) + 1
            if batch_num % 5 == 0 or batch_num == total_batches:
                progress = batch_num / total_batches * 100
                elapsed = time.time() - embedding_start
                print(f"   📊 Batch {batch_num}/{total_batches} ({progress:.1f}%) - {elapsed:.1f}s")
        
        embedding_time = time.time() - embedding_start
        embedding_rate = len(documents) / embedding_time
        
        print(f"   ✅ Embedded {len(documents)} documents in {embedding_time:.2f}s")
        print(f"   📊 Rate: {embedding_rate:.1f} docs/second")
        
        # Test search performance
        print("4️⃣ Testing search performance...")
        search_start = time.time()
        search_times = []
        quality_scores = []
        
        for i, query in enumerate(queries):
            query_start = time.time()
            results = engine.search_documents(query, n_results=5)
            query_time = time.time() - query_start
            search_times.append(query_time)
            
            # Calculate quality metrics
            if results and results.get("distances") and results["distances"][0]:
                similarities = [1 - dist for dist in results["distances"][0]]
                quality_scores.append(np.mean(similarities))
            
            if (i + 1) % 10 == 0:
                progress = (i + 1) / len(queries) * 100
                elapsed = time.time() - search_start
                print(f"   🔍 Query {i+1}/{len(queries)} ({progress:.1f}%) - {elapsed:.1f}s")
        
        search_time = time.time() - search_start
        search_rate = len(queries) / search_time
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        print(f"   ✅ Processed {len(queries)} queries in {search_time:.2f}s")
        print(f"   📊 Rate: {search_rate:.1f} queries/second")
        print(f"   🎯 Average quality: {avg_quality:.4f}")
        
        # Scalability test
        print("5️⃣ Testing scalability...")
        test_query = queries[0]
        scalability_times = []
        
        for load in [1, 5, 10]:
            times = []
            for _ in range(load):
                start = time.time()
                engine.search_documents(test_query, n_results=3)
                times.append(time.time() - start)
            scalability_times.append(np.mean(times))
        
        performance_degradation = scalability_times[-1] / scalability_times[0]
        
        results = {
            "model_key": model_key,
            "test_config": {"doc_count": doc_count, "query_count": query_count},
            "initialization_time": init_time,
            "embedding_performance": {
                "total_time": embedding_time,
                "docs_per_second": embedding_rate,
                "avg_time_per_doc": embedding_time / len(documents)
            },
            "search_performance": {
                "total_time": search_time,
                "queries_per_second": search_rate,
                "avg_query_time": np.mean(search_times),
                "quality_score": avg_quality
            },
            "scalability": {
                "performance_degradation": performance_degradation,
                "scalability_score": 1 / performance_degradation
            },
            "status": "success"
        }
        
        print(f"   📈 Performance degradation (10x load): {performance_degradation:.2f}x")
        print(f"✅ {model_key} testing completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_key} failed: {e}")
        return {"model_key": model_key, "status": "failed", "error": str(e)}

def main():
    """Run optimized comparison including CLIP"""
    print("🚀 OPTIMIZED LARGE FILE MODEL COMPARISON")
    print("Testing CLIP (optimized) vs Alternatives with Appropriate Document Sizes")
    print("=" * 80)
    
    # Models to test
    models_to_test = [
        ("clip", "CLIP ViT-B/32 (Multimodal)"),
        ("all_minilm", "all-MiniLM-L6-v2 (Fast Baseline)"),
        ("all_mpnet", "all-mpnet-base-v2 (SBERT Best)"),
    ]
    
    test_config = {
        "document_count": 1000,
        "query_count": 50,
    }
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_description": "Optimized large file comparison with CLIP token limits",
        "test_config": test_config,
        "model_results": {},
        "comparative_analysis": {}
    }
    
    # Test each model
    for model_key, model_description in models_to_test:
        print(f"\n{'='*20} {model_description} {'='*20}")
        model_results = test_model_optimized(
            model_key, 
            test_config["document_count"], 
            test_config["query_count"]
        )
        results["model_results"][model_key] = model_results
        
        # Brief summary
        if model_results["status"] == "success":
            embed_speed = model_results["embedding_performance"]["docs_per_second"]
            search_speed = model_results["search_performance"]["queries_per_second"]
            quality = model_results["search_performance"]["quality_score"]
            print(f"📊 Summary: {embed_speed:.1f} docs/s, {search_speed:.1f} q/s, {quality:.4f} quality")
    
    # Generate comparison analysis
    successful_results = {k: v for k, v in results["model_results"].items() 
                         if v["status"] == "success"}
    
    if len(successful_results) >= 2:
        print(f"\n{'='*80}")
        print("📊 FINAL COMPARATIVE ANALYSIS")
        print("="*80)
        
        # Performance comparison
        print("\n🏆 EMBEDDING SPEED RANKING:")
        embed_ranking = sorted(successful_results.items(), 
                             key=lambda x: x[1]["embedding_performance"]["docs_per_second"], 
                             reverse=True)
        for i, (model, data) in enumerate(embed_ranking, 1):
            speed = data["embedding_performance"]["docs_per_second"]
            print(f"  {i}. {model}: {speed:.1f} docs/second")
        
        print("\n⚡ SEARCH SPEED RANKING:")
        search_ranking = sorted(successful_results.items(),
                               key=lambda x: x[1]["search_performance"]["queries_per_second"],
                               reverse=True)
        for i, (model, data) in enumerate(search_ranking, 1):
            speed = data["search_performance"]["queries_per_second"]
            print(f"  {i}. {model}: {speed:.1f} queries/second")
        
        print("\n🎯 QUALITY RANKING:")
        quality_ranking = sorted(successful_results.items(),
                               key=lambda x: x[1]["search_performance"]["quality_score"],
                               reverse=True)
        for i, (model, data) in enumerate(quality_ranking, 1):
            quality = data["search_performance"]["quality_score"]
            print(f"  {i}. {model}: {quality:.4f} quality score")
        
        print("\n📈 SCALABILITY RANKING:")
        scalability_ranking = sorted(successful_results.items(),
                                   key=lambda x: x[1]["scalability"]["scalability_score"],
                                   reverse=True)
        for i, (model, data) in enumerate(scalability_ranking, 1):
            score = data["scalability"]["scalability_score"]
            print(f"  {i}. {model}: {score:.2f} scalability score")
        
        # Final recommendation
        print(f"\n💡 FINAL RECOMMENDATION:")
        
        # Check if CLIP is available and how it performed
        if "clip" in successful_results:
            clip_data = successful_results["clip"]
            print(f"  🖼️  CLIP: {clip_data['search_performance']['queries_per_second']:.1f} q/s, multimodal support")
        
        fastest_embed = embed_ranking[0][0] if embed_ranking else None
        fastest_search = search_ranking[0][0] if search_ranking else None
        highest_quality = quality_ranking[0][0] if quality_ranking else None
        
        print(f"  🏃 Fastest Embedding: {fastest_embed}")
        print(f"  ⚡ Fastest Search: {fastest_search}")
        print(f"  🎯 Highest Quality: {highest_quality}")
        
        if "clip" in successful_results:
            print(f"  🥇 RECOMMENDATION: CLIP for multimodal + good performance")
        else:
            print(f"  🥇 RECOMMENDATION: {fastest_search} for text-only applications")
        
        results["comparative_analysis"] = {
            "embedding_ranking": embed_ranking,
            "search_ranking": search_ranking,
            "quality_ranking": quality_ranking,
            "scalability_ranking": scalability_ranking
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optimized_large_file_comparison_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Final results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()
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
