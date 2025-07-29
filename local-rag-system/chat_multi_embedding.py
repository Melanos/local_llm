#!/usr/bin/env python3
"""
Multi-Embedding Chat Interface
Chat interface that allows switching between different embedding models
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.core.enhanced_rag_engine import EnhancedRAGEngine

def multi_embedding_chat():
    """Interactive chat with multiple embedding model support"""
    
    print("🤖 Multi-Embedding RAG Chat Interface")
    print("=" * 50)
    print("Compare how different embedding models retrieve context for your queries")
    print()
    
    # Available models
    models = {
        "1": ("nomic", "Ollama Nomic-Embed-Text"),
        "2": ("jina_base", "Jina Embeddings v2 Base"), 
        "3": ("jina_small", "Jina Embeddings v2 Small"),
        "4": ("all", "Compare All Models")
    }
    
    print("Available embedding models:")
    for key, (model_key, model_name) in models.items():
        print(f"  {key}. {model_name}")
    print()
    
    while True:
        # Model selection
        choice = input("Select embedding model (1-4, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        if choice not in models:
            print("❌ Invalid choice. Please try again.")
            continue
        
        model_key, model_name = models[choice]
        
        if model_key == "all":
            # Compare all models
            print(f"\n🔬 Comparison Mode - All Models")
            print("-" * 40)
            
            query = input("\n💬 Your question: ").strip()
            if not query:
                continue
            
            compare_all_models(query)
            
        else:
            # Single model chat
            print(f"\n🤖 Chat Mode - {model_name}")
            print("-" * 40)
            
            try:
                engine = EnhancedRAGEngine(embedding_model_key=model_key)
                info = engine.get_collection_info()
                print(f"📊 Collection: {info.get('document_count', 0)} documents")
                print()
                
                single_model_chat(engine, model_name)
                
            except Exception as e:
                print(f"❌ Error initializing {model_name}: {e}")
        
        print("\n" + "="*50)

def single_model_chat(engine, model_name):
    """Chat with a single embedding model"""
    
    print(f"💬 Chatting with {model_name}")
    print("Type 'back' to return to model selection, 'quit' to exit")
    print()
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() == 'quit':
            sys.exit(0)
        elif query.lower() == 'back':
            break
        elif not query:
            continue
        
        try:
            # Search for relevant context
            search_results = engine.search_documents(query, n_results=3)
            
            if search_results and search_results['documents'][0]:
                documents = search_results['documents'][0]
                distances = search_results['distances'][0]
                
                # Combine context
                context = "\n\n".join(documents[:2])  # Use top 2 results
                
                # Create prompt
                prompt = f"""Based on the network diagram analysis, answer this question:

Question: {query}

Context from network analysis:
{context[:1500]}...

Provide a helpful answer based on the network information provided."""

                print(f"\n🤖 {model_name}:")
                
                # Show retrieval info
                print(f"📊 Found {len(documents)} relevant chunks (distances: {[f'{d:.3f}' for d in distances[:3]]})")
                
                # Get LLM response
                response = engine.call_ollama(prompt)
                print(f"💡 {response}")
                
            else:
                print(f"\n🤖 {model_name}:")
                print("❌ No relevant information found in the database.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def compare_all_models(query):
    """Compare query results across all embedding models"""
    
    models_to_compare = [
        ("nomic", "Nomic"),
        ("jina_base", "Jina Base"),
        ("jina_small", "Jina Small")
    ]
    
    print(f"\n🔍 Query: {query}")
    print("=" * 60)
    
    results = {}
    
    for model_key, model_short_name in models_to_compare:
        print(f"\n📊 {model_short_name} Results:")
        print("-" * 30)
        
        try:
            engine = EnhancedRAGEngine(embedding_model_key=model_key)
            search_results = engine.search_documents(query, n_results=3)
            
            if search_results and search_results['documents'][0]:
                documents = search_results['documents'][0]
                distances = search_results['distances'][0]
                
                print(f"✅ Found {len(documents)} documents")
                print(f"📏 Distances: {[f'{d:.4f}' for d in distances[:3]]}")
                print(f"📄 Top result preview: {documents[0][:100]}...")
                
                results[model_key] = {
                    "documents": documents,
                    "distances": distances,
                    "avg_distance": sum(distances) / len(distances) if distances else 0
                }
                
            else:
                print("❌ No results found")
                results[model_key] = {"documents": [], "distances": [], "avg_distance": 0}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results[model_key] = {"error": str(e)}
    
    # Summary comparison
    print(f"\n🏆 Summary:")
    print("-" * 20)
    
    valid_results = [(k, v) for k, v in results.items() if "error" not in v and v["documents"]]
    
    if valid_results:
        # Best by distance (lower is better)
        best_quality = min(valid_results, key=lambda x: x[1]["avg_distance"])
        print(f"🎯 Best semantic match: {dict(models_to_compare)[best_quality[0]]} (avg distance: {best_quality[1]['avg_distance']:.4f})")
        
        # Most results
        most_results = max(valid_results, key=lambda x: len(x[1]["documents"]))
        print(f"📚 Most results: {dict(models_to_compare)[most_results[0]]} ({len(most_results[1]['documents'])} documents)")
    else:
        print("❌ No valid results from any model")

if __name__ == "__main__":
    try:
        multi_embedding_chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
