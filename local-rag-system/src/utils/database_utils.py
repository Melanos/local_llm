"""
Database Utilities - Inspect and manage the RAG database
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_engine import RAGEngine
from config import DEFAULT_CONFIG


class DatabaseUtils:
    """Utilities for database inspection and management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.rag_engine = RAGEngine(config)
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed database statistics"""
        try:
            stats = self.rag_engine.get_stats()
            
            # Get sample documents to analyze content types
            sample_results = self.rag_engine.search_documents("", n_results=10, relevance_threshold=0.0)
            
            content_types = {}
            sources = set()
            
            if sample_results['metadatas'][0]:
                for metadata in sample_results['metadatas'][0]:
                    source_type = metadata.get('source_type', 'unknown')
                    content_types[source_type] = content_types.get(source_type, 0) + 1
                    
                    source = metadata.get('source', 'unknown')
                    sources.add(source)
            
            stats.update({
                "content_types": content_types,
                "unique_sources": len(sources),
                "sample_sources": list(sources)[:5]  # First 5 sources
            })
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def search_by_keyword(self, keyword: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for documents containing specific keywords"""
        print(f"🔍 Searching for: '{keyword}'")
        
        results = self.rag_engine.search_documents(
            keyword, n_results=n_results, relevance_threshold=0.1
        )
        
        if not results['documents'][0]:
            print(f"❌ No documents found containing '{keyword}'")
            return {"found": 0, "results": []}
        
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            relevance = round((1 - distance) * 100, 1)
            metadata = results['metadatas'][0][i]
            
            formatted_results.append({
                "relevance": relevance,
                "source": metadata.get('source', 'Unknown'),
                "source_type": metadata.get('source_type', 'Unknown'),
                "preview": doc[:200] + "..." if len(doc) > 200 else doc
            })
        
        print(f"✅ Found {len(formatted_results)} relevant document(s)")
        return {"found": len(formatted_results), "results": formatted_results}
    
    def list_sources(self) -> List[str]:
        """List all unique sources in the database"""
        try:
            # Use ChromaDB's get method to retrieve all documents
            collection = self.rag_engine.collection
            results = collection.get()
            
            sources = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    if source != 'Unknown':
                        sources.add(source)
            
            return sorted(list(sources))
            
        except Exception as e:
            print(f"❌ Error listing sources: {e}")
            return []
    
    def show_sample_documents(self, n_samples: int = 3):
        """Show sample documents from the database"""
        print(f"📄 Sample Documents (showing {n_samples}):")
        print("=" * 50)
        
        results = self.rag_engine.search_documents("", n_results=n_samples, relevance_threshold=0.0)
        
        if not results['documents'][0]:
            print("❌ No documents found in database")
            return
        
        for i, doc in enumerate(results['documents'][0], 1):
            metadata = results['metadatas'][0][i-1] if results['metadatas'] else {}
            
            print(f"\n📄 Document {i}:")
            print(f"   Source: {metadata.get('source', 'Unknown')}")
            print(f"   Type: {metadata.get('source_type', 'Unknown')}")
            print(f"   Preview: {doc[:150]}...")
            print("-" * 50)


def main():
    """Main utility functions"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Utilities")
    parser.add_argument("--stats", action="store_true", help="Show detailed database statistics")
    parser.add_argument("--search", type=str, help="Search for keyword in database")
    parser.add_argument("--sources", action="store_true", help="List all sources")
    parser.add_argument("--samples", type=int, default=3, help="Show sample documents")
    
    args = parser.parse_args()
    
    utils = DatabaseUtils()
    
    if args.stats:
        print("📊 Database Statistics")
        print("=" * 30)
        stats = utils.get_detailed_stats()
        
        print(f"Total Documents: {stats.get('total_documents', 0)}")
        print(f"Unique Sources: {stats.get('unique_sources', 0)}")
        
        if 'content_types' in stats:
            print("\nContent Types:")
            for content_type, count in stats['content_types'].items():
                print(f"  {content_type}: {count}")
        
        if 'sample_sources' in stats:
            print(f"\nSample Sources: {', '.join(stats['sample_sources'])}")
    
    elif args.search:
        results = utils.search_by_keyword(args.search)
        
        for i, result in enumerate(results['results'], 1):
            print(f"\n{i}. {result['source']} ({result['source_type']}) - {result['relevance']}%")
            print(f"   {result['preview']}")
    
    elif args.sources:
        sources = utils.list_sources()
        print(f"📚 Sources in Database ({len(sources)} total):")
        for source in sources:
            print(f"  - {source}")
    
    else:
        utils.show_sample_documents(args.samples)


if __name__ == "__main__":
    main()
