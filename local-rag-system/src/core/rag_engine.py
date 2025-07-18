"""
Core RAG Engine - Database operations and search functionality
"""

import sys
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import requests
import json
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG


class RAGEngine:
    """Core RAG functionality for document storage and retrieval"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        
        self.chat_model = self.config["models"]["chat_model"]
        self.embedding_model = self.config["models"]["embedding_model"] 
        self.ollama_url = self.config["ollama"]["url"]
        self.db_path = self.config["database"]["path"]
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Create or get collection with Ollama embedding function
        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url=self.ollama_url,
            model_name=self.embedding_model,
        )
        
        # Create collection (or get existing one)
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.config["database"]["collection_name"],
                embedding_function=self.embedding_function
            )
            print(f"✅ Created new vector database at: {self.db_path}")
        except Exception:
            self.collection = self.chroma_client.get_collection(
                name=self.config["database"]["collection_name"],
                embedding_function=self.embedding_function
            )
            print(f"✅ Connected to existing vector database at: {self.db_path}")
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the vector database"""
        doc_id = str(uuid.uuid4())
        
        # Default metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id
        })
        
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return doc_id
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            return None
    
    def search_documents(self, query: str, n_results: int = 5, 
                        relevance_threshold: float = None) -> Dict[str, Any]:
        """Search for relevant documents with relevance filtering"""
        if relevance_threshold is None:
            relevance_threshold = self.config["relevance"]["search_threshold"]
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Filter by relevance threshold
            if results and results['documents'] and results['documents'][0]:
                filtered_docs = []
                filtered_metadata = []
                filtered_distances = []
                
                for i, distance in enumerate(results['distances'][0]):
                    if distance < relevance_threshold:
                        filtered_docs.append(results['documents'][0][i])
                        filtered_metadata.append(results['metadatas'][0][i])
                        filtered_distances.append(distance)
                
                return {
                    'documents': [filtered_docs] if filtered_docs else [[]],
                    'metadatas': [filtered_metadata] if filtered_metadata else [[]],
                    'distances': [filtered_distances] if filtered_distances else [[]]
                }
            
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "database_path": self.db_path,
                "collection_name": self.config["database"]["collection_name"]
            }
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {"total_documents": 0, "error": str(e)}
    
    def clear_database(self):
        """Clear all documents from the database"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name=self.config["database"]["collection_name"])
            
            # Recreate the collection
            self.collection = self.chroma_client.create_collection(
                name=self.config["database"]["collection_name"],
                embedding_function=self.embedding_function
            )
            print("🗑️ Database cleared successfully")
            
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
    
    def call_ollama(self, prompt: str, model: str = None) -> str:
        """Make a request to Ollama API"""
        if model is None:
            model = self.chat_model
            
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"❌ Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"❌ Connection error: {e}"
