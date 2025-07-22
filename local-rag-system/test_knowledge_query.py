#!/usr/bin/env python3
"""
Quick test script to query the network knowledge base
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.core.rag_engine import RAGEngine

def test_queries():
    """Test various queries against the network knowledge base"""
    
    print("🧪 Testing Network Knowledge Queries")
    print("=" * 50)
    
    # Initialize RAG engine
    rag_engine = RAGEngine()
    
    # Test queries for our clean, retrained network diagrams
    test_questions = [
        "What devices are shown in the network diagrams?",
        "Describe the network topology structure",
        "What connections and cables are visible?",
        "Are there any IP addresses or network information shown?",
        "What security components can you identify?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. 🔍 Question: {question}")
        print("-" * 40)
        
        # Search for relevant context
        search_results = rag_engine.search_documents(question, n_results=3)
        
        if search_results and search_results['documents'][0]:
            documents = search_results['documents'][0]
            context = "\n".join(documents[:2])  # Use top 2 results
            
            # Create prompt for LLM
            prompt = f"""Based on the network diagram analysis, answer this question:

Question: {question}

Context:
{context[:1000]}...

Provide a concise answer."""
            
            # Get answer
            answer = rag_engine.call_ollama(prompt)
            
            # Show answer (truncated)
            summary = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"💡 Answer: {summary}")
        else:
            print("❌ No relevant information found")
    
    print(f"\n✅ Knowledge query testing complete!")

if __name__ == "__main__":
    test_queries()
