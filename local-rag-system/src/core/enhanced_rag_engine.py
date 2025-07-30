"""
Enhanced RAG Engine with Multiple Embedding Model Support
"""

import sys
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import requests
import json
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  SentenceTransformers not available. Jina models will be unavailable.")

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not available. Multimodal features will be unavailable.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available. Image processing will be unavailable.")


class EmbeddingEngine:
    """Handles multiple embedding model backends including multimodal CLIP"""
    
    def __init__(self, model_name: str, model_type: str = "ollama"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.processor = None
        
        if model_type == "huggingface" and SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"🔧 Loading HuggingFace model: {model_name}")
            # Handle models that require trust_remote_code
            if "jina-embeddings-v4" in model_name:
                self.model = SentenceTransformer(
                    model_name, 
                    model_kwargs={'default_task': 'retrieval'},
                    trust_remote_code=True
                )
            else:
                self.model = SentenceTransformer(model_name)
            print(f"✅ HuggingFace model loaded successfully")
        elif model_type == "clip" and CLIP_AVAILABLE:
            print(f"🔧 Loading CLIP model: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print(f"✅ CLIP model loaded successfully")
        elif model_type == "ollama":
            print(f"🔧 Using Ollama model: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if self.model_type == "huggingface" and self.model:
            # Use Jina/HuggingFace model
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        elif self.model_type == "clip" and self.model:
            # Use CLIP model for text
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
            return text_features.cpu().numpy().tolist()
        elif self.model_type == "ollama":
            # Use Ollama model
            embeddings = []
            for text in texts:
                try:
                    response = requests.post(
                        "http://localhost:11434/api/embeddings",
                        json={"model": self.model_name, "prompt": text},
                        timeout=30
                    )
                    if response.status_code == 200:
                        embedding = response.json().get("embedding", [])
                        embeddings.append(embedding)
                    else:
                        print(f"❌ Ollama embedding failed: {response.status_code}")
                        return []
                except Exception as e:
                    print(f"❌ Ollama embedding error: {e}")
                    return []
            return embeddings
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text"""
        if self.model_type == "huggingface" and self.model:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        elif self.model_type == "clip" and self.model:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
            return text_features.cpu().numpy()[0].tolist()
        elif self.model_type == "ollama":
            try:
                response = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json().get("embedding", [])
                else:
                    print(f"❌ Ollama query embedding failed: {response.status_code}")
                    return []
            except Exception as e:
                print(f"❌ Ollama query embedding error: {e}")
                return []
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for an image (CLIP only)"""
        if self.model_type == "clip" and self.model and PIL_AVAILABLE:
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
                return image_features.cpu().numpy()[0].tolist()
            except Exception as e:
                print(f"❌ CLIP image embedding error: {e}")
                return []
        else:
            raise ValueError(f"Image embedding only supported for CLIP models")


class EnhancedRAGEngine:
    """Enhanced RAG functionality with multiple embedding model support"""
    
    def __init__(self, config: Dict[str, Any] = None, embedding_model_key: str = None):
        self.config = config or DEFAULT_CONFIG
        
        self.chat_model = self.config["models"]["chat_model"]
        self.ollama_url = self.config["ollama"]["url"]
        
        # Set up embedding model
        if embedding_model_key and embedding_model_key in self.config["models"]["embedding_options"]:
            self.embedding_model_name = self.config["models"]["embedding_options"][embedding_model_key]
            self.embedding_model_key = embedding_model_key
        else:
            self.embedding_model_name = self.config["models"]["embedding_model"]
            self.embedding_model_key = "default"
        
        # Determine model type and initialize embedding engine
        if self.embedding_model_name.startswith("jinaai/") or self.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2":
            self.embedding_engine = EmbeddingEngine(self.embedding_model_name, "huggingface")
        elif self.embedding_model_name.startswith("openai/clip-"):
            self.embedding_engine = EmbeddingEngine(self.embedding_model_name, "clip")
        else:
            self.embedding_engine = EmbeddingEngine(self.embedding_model_name, "ollama")
        
        # Multimodal capabilities for CLIP
        self.supports_images = (self.embedding_model_name.startswith("openai/clip-"))
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize ChromaDB with custom embedding function"""
        db_path = self.config["database"]["path"]
        collection_name = f"{self.config['database']['collection_name']}_{self.embedding_model_key}"
        
        print(f"🔗 Connecting to database: {db_path}")
        print(f"📊 Using embedding model: {self.embedding_model_name}")
        print(f"🗃️  Collection: {collection_name}")
        
        try:
            # Custom embedding function that uses our EmbeddingEngine
            class CustomEmbeddingFunction:
                def __init__(self, embedding_engine):
                    self.embedding_engine = embedding_engine
                
                def name(self):
                    """Required by ChromaDB"""
                    return f"custom_{self.embedding_engine.model_type}_{self.embedding_engine.model_name}"
                
                def __call__(self, input):
                    if isinstance(input, str):
                        return self.embedding_engine.embed_query(input)
                    elif isinstance(input, list):
                        return self.embedding_engine.embed_documents(input)
                    else:
                        raise ValueError(f"Unsupported input type: {type(input)}")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Create or get collection with custom embedding function
            embedding_function = CustomEmbeddingFunction(self.embedding_engine)
            
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
                print(f"✅ Connected to existing collection: {collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
                print(f"✅ Created new collection: {collection_name}")
                
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector database"""
        if not documents:
            return
            
        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        # Generate metadatas if not provided
        if not metadatas:
            metadatas = [{"timestamp": datetime.now().isoformat()} for _ in documents]
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Added {len(documents)} documents to collection")
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            raise
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "embedding_model": self.embedding_model_name,
                "embedding_model_key": self.embedding_model_key,
                "collection_name": self.collection.name,
                "document_count": count
            }
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
            return {}
    
    def call_ollama(self, prompt: str) -> str:
        """Generate response using Ollama LLM"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.chat_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def add_image(self, image_path: str, metadata: Dict = None, image_id: str = None):
        """Add an image to the vector database (CLIP models only)"""
        if not self.supports_images:
            raise ValueError("Image embedding only supported for CLIP models")
        
        if not image_id:
            image_id = str(uuid.uuid4())
        
        if not metadata:
            metadata = {
                "type": "image",
                "path": image_path,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Generate image embedding using CLIP
            image_embedding = self.embedding_engine.embed_image(image_path)
            if not image_embedding:
                return False
            
            # Store image representation in collection
            self.collection.add(
                documents=[f"Image: {os.path.basename(image_path)}"],  # Text description
                metadatas=[metadata],
                ids=[image_id]
            )
            
            print(f"✅ Added image: {os.path.basename(image_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add image {image_path}: {e}")
            return False
    
    def search_multimodal(self, query: str, include_images: bool = True, n_results: int = 5):
        """Search both text and images using CLIP embeddings"""
        if not self.supports_images:
            return self.search(query, n_results)
        
        try:
            # Generate text embedding for the query
            query_embedding = self.embedding_engine.embed_query(query)
            
            # Search in the vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for doc, metadata, distance in zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            ):
                result_type = metadata.get("type", "text")
                formatted_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "type": result_type
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Multimodal search failed: {e}")
            return []


# Backwards compatibility - keep original RAGEngine class
class RAGEngine(EnhancedRAGEngine):
    """Original RAG Engine for backwards compatibility"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, embedding_model_key=None)


if __name__ == "__main__":
    # Test both embedding models
    print("🧪 Testing Enhanced RAG Engine")
    
    # Test with default (Ollama) model
    print("\n1. Testing with Ollama (nomic-embed-text)")
    engine1 = EnhancedRAGEngine(embedding_model_key="nomic")
    print(f"Collection info: {engine1.get_collection_info()}")
    
    # Test with Jina model (if available)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n2. Testing with Jina v2 Base (jina-embeddings-v2-base-en)")
        engine2 = EnhancedRAGEngine(embedding_model_key="jina_v2_base")
        print(f"Collection info: {engine2.get_collection_info()}")
    else:
        print("\n2. Jina models not available (install sentence-transformers)")
