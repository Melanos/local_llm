#!/usr/bin/env python3
"""
Comprehensive Embedding Model Testing and Analysis
Tests Jina v4 against all available models with detailed metrics
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.enhanced_rag_engine import EnhancedRAGEngine
from config import DEFAULT_CONFIG

class ComprehensiveModelTester:
    """Comprehensive testing suite for embedding models"""
    
    def __init__(self):
        self.models_to_test = {
            "jina_v4": {
                "name": "Jina v4",
                "dimensions": 2048,
                "description": "Latest Jina embedding model with high quality"
            },
            "nomic": {
                "name": "Nomic Embed", 
                "dimensions": 768,
                "description": "Privacy-focused model via Ollama"
            },
            "minilm": {
                "name": "MiniLM-L6-v2",
                "dimensions": 384, 
                "description": "Fast, lightweight industry baseline"
            },
            "clip": {
                "name": "CLIP ViT-B/32",
                "dimensions": 512,
                "description": "Multimodal text+image understanding"
            }
        }
        
        # Comprehensive test documents covering various domains
        self.test_documents = [
            # Technical/IT
            "Machine learning algorithms process data to identify patterns and make predictions automatically.",
            "Network infrastructure includes routers, switches, firewalls, and load balancers for connectivity.",
            "Database optimization involves indexing, query tuning, and proper schema design principles.",
            
            # Business/Finance  
            "Financial markets fluctuate based on economic indicators, investor sentiment, and global events.",
            "Supply chain management coordinates procurement, production, and distribution processes efficiently.",
            "Customer relationship management systems track interactions and improve business relationships.",
            
            # Science/Research
            "Quantum computing leverages quantum mechanical phenomena for computational advantages.",
            "Climate change research analyzes temperature data, ice core samples, and atmospheric measurements.",
            "Pharmaceutical development requires extensive clinical trials and regulatory approval processes.",
            
            # General Knowledge
            "Renewable energy sources include solar, wind, hydroelectric, and geothermal power systems.",
            "Educational psychology studies how people learn and the most effective teaching methods.",
            "Urban planning balances residential, commercial, and industrial development with infrastructure needs."
        ]
        
        # Diverse test queries
        self.test_queries = [
            # Technical queries
            "How do machine learning algorithms work?",
            "What is network security architecture?", 
            "Database performance optimization techniques",
            
            # Business queries
            "Financial market analysis methods",
            "Supply chain efficiency strategies",
            "Customer service best practices",
            
            # Science queries
            "Quantum computing applications",
            "Climate research methodologies", 
            "Drug development process",
            
            # General queries
            "Sustainable energy solutions",
            "Effective learning strategies",
            "Smart city planning approaches"
        ]
        
        self.results = {}
    
    def test_model_comprehensive(self, model_key: str, model_info: Dict) -> Dict[str, Any]:
        """Comprehensive testing of a single model"""
        print(f"\n🧪 Testing {model_info['name']} ({model_info['dimensions']}-dim)")
        print("=" * 60)
        
        results = {
            "model_key": model_key,
            "model_info": model_info,
            "initialization_time": 0,
            "embedding_performance": {},
            "search_performance": {},
            "quality_metrics": {},
            "multimodal_capabilities": {},
            "errors": [],
            "status": "pending"
        }
        
        try:
            # 1. Initialization Test
            print("1️⃣ Initialization Test")
            start_time = time.time()
            engine = EnhancedRAGEngine(embedding_model_key=model_key)
            init_time = time.time() - start_time
            results["initialization_time"] = init_time
            print(f"   ⏱️  Initialization: {init_time:.2f}s")
            print(f"   🖼️  Image support: {engine.supports_images}")
            results["multimodal_capabilities"]["supports_images"] = engine.supports_images
            
            # 2. Embedding Performance Test
            print("\n2️⃣ Embedding Performance Test")
            start_time = time.time()
            engine.add_documents(
                self.test_documents,
                [{"test_batch": True, "doc_id": i, "domain": self._get_domain(i)} 
                 for i in range(len(self.test_documents))]
            )
            embedding_time = time.time() - start_time
            
            results["embedding_performance"] = {
                "total_time": embedding_time,
                "docs_per_second": len(self.test_documents) / embedding_time,
                "avg_time_per_doc": embedding_time / len(self.test_documents)
            }
            
            print(f"   ⏱️  Total embedding time: {embedding_time:.2f}s")
            print(f"   📊 Rate: {results['embedding_performance']['docs_per_second']:.1f} docs/sec")
            
            # 3. Search Performance and Quality Test
            print("\n3️⃣ Search Performance & Quality Test")
            search_results = []
            total_search_time = 0
            
            for i, query in enumerate(self.test_queries):
                start_time = time.time()
                query_results = engine.search_documents(query, n_results=5)
                search_time = time.time() - start_time
                total_search_time += search_time
                
                # Analyze quality
                quality_metrics = self._analyze_search_quality(query, query_results, i)
                search_results.append({
                    "query": query,
                    "search_time": search_time,
                    **quality_metrics
                })
                
                print(f"   🔍 Query {i+1}: {search_time:.3f}s, Similarity: {quality_metrics['top_similarity']:.4f}")
            
            # Aggregate search metrics
            results["search_performance"] = {
                "total_time": total_search_time,
                "avg_time_per_query": total_search_time / len(self.test_queries),
                "queries_per_second": len(self.test_queries) / total_search_time
            }
            
            # Quality metrics
            avg_similarity = np.mean([r['avg_similarity'] for r in search_results])
            avg_top_similarity = np.mean([r['top_similarity'] for r in search_results])
            relevance_score = np.mean([r['relevance_score'] for r in search_results])
            
            results["quality_metrics"] = {
                "average_similarity": avg_similarity,
                "average_top_similarity": avg_top_similarity,
                "relevance_score": relevance_score,
                "detailed_results": search_results
            }
            
            print(f"   🎯 Average similarity: {avg_similarity:.4f}")
            print(f"   🥇 Average top similarity: {avg_top_similarity:.4f}")
            print(f"   📈 Relevance score: {relevance_score:.4f}")
            
            # 4. Multimodal Test (if supported)
            if engine.supports_images:
                print("\n4️⃣ Multimodal Capabilities Test")
                multimodal_results = self._test_multimodal_capabilities(engine)
                results["multimodal_capabilities"].update(multimodal_results)
            
            # 5. Stress Test
            print("\n5️⃣ Stress Test")
            stress_results = self._perform_stress_test(engine)
            results["stress_test"] = stress_results
            
            results["status"] = "success"
            print(f"✅ {model_info['name']} testing completed successfully")
            
        except Exception as e:
            error_msg = f"Error testing {model_info['name']}: {e}"
            results["errors"].append(error_msg)
            results["status"] = "failed"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def _get_domain(self, doc_index: int) -> str:
        """Get domain category for a document"""
        if doc_index < 3:
            return "technical"
        elif doc_index < 6:
            return "business"
        elif doc_index < 9:
            return "science"
        else:
            return "general"
    
    def _analyze_search_quality(self, query: str, results: Dict, query_index: int) -> Dict:
        """Analyze the quality of search results"""
        if not results or not results.get("documents") or not results["documents"][0]:
            return {
                "results_count": 0,
                "avg_similarity": 0,
                "top_similarity": 0,
                "relevance_score": 0
            }
        
        documents = results["documents"][0]
        distances = results["distances"][0]
        
        # Convert distances to similarities
        similarities = [1 - dist for dist in distances]
        
        # Calculate relevance score based on expected domain matching
        expected_domain = self._get_query_domain(query_index)
        relevance_score = self._calculate_relevance_score(results, expected_domain)
        
        return {
            "results_count": len(documents),
            "avg_similarity": np.mean(similarities),
            "top_similarity": similarities[0] if similarities else 0,
            "relevance_score": relevance_score
        }
    
    def _get_query_domain(self, query_index: int) -> str:
        """Get expected domain for a query"""
        if query_index < 3:
            return "technical"
        elif query_index < 6:
            return "business" 
        elif query_index < 9:
            return "science"
        else:
            return "general"
    
    def _calculate_relevance_score(self, results: Dict, expected_domain: str) -> float:
        """Calculate how relevant results are to expected domain"""
        if not results.get("metadatas") or not results["metadatas"][0]:
            return 0.5  # Neutral score if no metadata
        
        metadatas = results["metadatas"][0]
        domain_matches = sum(1 for meta in metadatas if meta.get("domain") == expected_domain)
        
        return domain_matches / len(metadatas) if metadatas else 0
    
    def _test_multimodal_capabilities(self, engine) -> Dict:
        """Test multimodal capabilities"""
        results = {"image_processing": False, "multimodal_search": False}
        
        try:
            # Look for test images
            image_dir = project_root / "data" / "images"
            if image_dir.exists():
                image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
                
                if image_files:
                    test_image = str(image_files[0])
                    
                    # Test image processing
                    start_time = time.time()
                    success = engine.add_image(test_image)
                    image_time = time.time() - start_time
                    
                    if success:
                        results["image_processing"] = True
                        results["image_processing_time"] = image_time
                        print(f"   📷 Image processed in {image_time:.3f}s")
                        
                        # Test multimodal search
                        multimodal_results = engine.search_multimodal("technical diagram", n_results=3)
                        if multimodal_results:
                            results["multimodal_search"] = True
                            results["multimodal_results_count"] = len(multimodal_results)
                            print(f"   🔍 Multimodal search: {len(multimodal_results)} results")
                        
        except Exception as e:
            print(f"   ❌ Multimodal test failed: {e}")
        
        return results
    
    def _perform_stress_test(self, engine) -> Dict:
        """Perform stress testing"""
        print("   🔥 Running stress test...")
        
        # Test with many concurrent queries
        stress_queries = self.test_queries * 3  # 3x the queries
        start_time = time.time()
        
        for query in stress_queries:
            engine.search_documents(query, n_results=3)
        
        stress_time = time.time() - start_time
        
        return {
            "total_queries": len(stress_queries),
            "total_time": stress_time,
            "queries_per_second": len(stress_queries) / stress_time
        }
    
    def run_comprehensive_testing(self) -> Dict[str, Any]:
        """Run comprehensive testing on all models"""
        print("🚀 COMPREHENSIVE EMBEDDING MODEL TESTING")
        print("=" * 80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 Test documents: {len(self.test_documents)}")
        print(f"🔍 Test queries: {len(self.test_queries)}")
        print(f"🧪 Models to test: {len(self.models_to_test)}")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "documents_count": len(self.test_documents),
                "queries_count": len(self.test_queries),
                "domains": ["technical", "business", "science", "general"]
            },
            "models": {},
            "comparative_analysis": {},
            "recommendations": {}
        }
        
        # Test each model
        for model_key, model_info in self.models_to_test.items():
            if model_key in DEFAULT_CONFIG["models"]["embedding_options"]:
                model_results = self.test_model_comprehensive(model_key, model_info)
                test_results["models"][model_key] = model_results
            else:
                print(f"⚠️  Model {model_key} not found in config, skipping...")
        
        # Generate comparative analysis
        test_results["comparative_analysis"] = self._generate_comparative_analysis(test_results["models"])
        
        # Generate recommendations
        test_results["recommendations"] = self._generate_recommendations(test_results["models"])
        
        return test_results
    
    def _generate_comparative_analysis(self, models: Dict) -> Dict:
        """Generate comparative analysis between models"""
        successful_models = {k: v for k, v in models.items() if v["status"] == "success"}
        
        if not successful_models:
            return {"error": "No models tested successfully"}
        
        analysis = {
            "performance_ranking": {},
            "quality_ranking": {},
            "efficiency_ranking": {},
            "capabilities_comparison": {}
        }
        
        # Performance ranking (speed)
        speed_scores = {}
        for model_key, model_data in successful_models.items():
            search_speed = model_data["search_performance"]["queries_per_second"]
            embed_speed = model_data["embedding_performance"]["docs_per_second"]
            speed_scores[model_key] = (search_speed + embed_speed) / 2
        
        analysis["performance_ranking"] = sorted(speed_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Quality ranking
        quality_scores = {}
        for model_key, model_data in successful_models.items():
            if "quality_metrics" in model_data:
                quality_score = (
                    model_data["quality_metrics"]["average_similarity"] * 0.4 +
                    model_data["quality_metrics"]["average_top_similarity"] * 0.4 +
                    model_data["quality_metrics"]["relevance_score"] * 0.2
                )
                quality_scores[model_key] = quality_score
        
        analysis["quality_ranking"] = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Efficiency ranking (quality/speed balance)
        efficiency_scores = {}
        for model_key in successful_models.keys():
            if model_key in quality_scores and model_key in speed_scores:
                efficiency = quality_scores[model_key] * speed_scores[model_key]
                efficiency_scores[model_key] = efficiency
        
        analysis["efficiency_ranking"] = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Capabilities comparison
        analysis["capabilities_comparison"] = {
            "multimodal_models": [k for k, v in successful_models.items() 
                                if v["multimodal_capabilities"]["supports_images"]],
            "text_only_models": [k for k, v in successful_models.items() 
                               if not v["multimodal_capabilities"]["supports_images"]],
            "fastest_initialization": min(successful_models.items(), 
                                        key=lambda x: x[1]["initialization_time"])[0],
            "highest_quality": max(quality_scores.items(), key=lambda x: x[1])[0] if quality_scores else None
        }
        
        return analysis
    
    def _generate_recommendations(self, models: Dict) -> Dict:
        """Generate usage recommendations"""
        successful_models = {k: v for k, v in models.items() if v["status"] == "success"}
        
        recommendations = {
            "general_purpose": None,
            "high_performance": None,
            "multimodal_tasks": None,
            "privacy_focused": None,
            "resource_constrained": None
        }
        
        if not successful_models:
            return recommendations
        
        # Find best models for different use cases
        quality_scores = {}
        speed_scores = {}
        
        for model_key, model_data in successful_models.items():
            if "quality_metrics" in model_data:
                quality_scores[model_key] = model_data["quality_metrics"]["average_top_similarity"]
            
            if "search_performance" in model_data:
                speed_scores[model_key] = model_data["search_performance"]["queries_per_second"]
        
        # General purpose: best balance of quality and speed
        if quality_scores and speed_scores:
            balance_scores = {}
            for model_key in successful_models.keys():
                if model_key in quality_scores and model_key in speed_scores:
                    # Normalize scores and balance them
                    norm_quality = quality_scores[model_key] / max(quality_scores.values())
                    norm_speed = speed_scores[model_key] / max(speed_scores.values())
                    balance_scores[model_key] = (norm_quality + norm_speed) / 2
            
            if balance_scores:
                recommendations["general_purpose"] = max(balance_scores.items(), key=lambda x: x[1])[0]
        
        # High performance: fastest
        if speed_scores:
            recommendations["high_performance"] = max(speed_scores.items(), key=lambda x: x[1])[0]
        
        # Multimodal tasks
        multimodal_models = [k for k, v in successful_models.items() 
                           if v["multimodal_capabilities"]["supports_images"]]
        if multimodal_models:
            recommendations["multimodal_tasks"] = multimodal_models[0]  # First available
        
        # Privacy focused (Ollama-based)
        ollama_models = [k for k, v in successful_models.items() 
                        if "nomic" in k.lower()]  # Assuming nomic is privacy-focused
        if ollama_models:
            recommendations["privacy_focused"] = ollama_models[0]
        
        # Resource constrained: smallest dimension model with good performance
        dimension_info = {k: v["model_info"]["dimensions"] for k, v in successful_models.items()}
        if dimension_info:
            recommendations["resource_constrained"] = min(dimension_info.items(), key=lambda x: x[1])[0]
        
        return recommendations

def main():
    """Run comprehensive testing and save results"""
    tester = ComprehensiveModelTester()
    results = tester.run_comprehensive_testing()
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_embedding_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TESTING SUMMARY")
    print("="*80)
    
    if results["comparative_analysis"]:
        analysis = results["comparative_analysis"]
        recommendations = results["recommendations"]
        
        print("\n🏆 PERFORMANCE RANKINGS:")
        for i, (model, score) in enumerate(analysis["performance_ranking"], 1):
            print(f"  {i}. {model}: {score:.2f} ops/sec")
        
        print("\n🎯 QUALITY RANKINGS:")
        for i, (model, score) in enumerate(analysis["quality_ranking"], 1):
            print(f"  {i}. {model}: {score:.4f} quality score")
        
        print("\n⚖️  EFFICIENCY RANKINGS:")
        for i, (model, score) in enumerate(analysis["efficiency_ranking"], 1):
            print(f"  {i}. {model}: {score:.2f} efficiency score")
        
        print("\n💡 RECOMMENDATIONS:")
        for use_case, model in recommendations.items():
            if model:
                print(f"  • {use_case.replace('_', ' ').title()}: {model}")
    
    return results

if __name__ == "__main__":
    main()
