"""
🗄️ RAG Chatbot with Vector Database
This creates a chatbot that can store documents as embeddings and retrieve relevant information to answer questions.
Uses ChromaDB for vector storage and your local Ollama models for chat and embeddings.
"""

import requests
import json
import chromadb
from chromadb.utils import embedding_functions
import os
import glob
from pathlib import Path
import uuid
from datetime import datetime

class RAGChatbot:
    def __init__(self, 
                 chat_model="llama3.2", 
                 embedding_model="nomic-embed-text",
                 ollama_url="http://localhost:11434",
                 db_path="./rag_database"):
        
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.db_path = db_path
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection with Ollama embedding function
        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url=ollama_url,
            model_name=embedding_model,
        )
        
        # Create collection (or get existing one)
        try:
            self.collection = self.chroma_client.create_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
            print(f"✅ Created new vector database at: {db_path}")
        except Exception:
            self.collection = self.chroma_client.get_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
            print(f"✅ Connected to existing vector database at: {db_path}")
        
        self.chat_history = []
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:  # Only if boundary is reasonable
                    end = start + boundary + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if end >= len(text):
                break
                
        return chunks
    
    def add_document(self, content, filename=None, metadata=None):
        """Add a document to the vector database"""
        if filename is None:
            filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Split into chunks
        chunks = self.chunk_text(content)
        
        # Generate IDs and metadata for each chunk
        ids = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            chunk_metadata = {
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "added_date": datetime.now().isoformat()
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            ids.append(chunk_id)
            metadatas.append(chunk_metadata)
            documents.append(chunk)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added document '{filename}' ({len(chunks)} chunks) to database")
        return len(chunks)
    
    def add_text_file(self, file_path):
        """Add a text file to the database"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(file_path)
            metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path)
            }
            
            return self.add_document(content, filename, metadata)
            
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            return 0
    
    def add_folder(self, folder_path, file_extensions=['.txt', '.md', '.py', '.json']):
        """Add all text files from a folder to the database"""
        total_chunks = 0
        files_added = 0
        
        for ext in file_extensions:
            pattern = os.path.join(folder_path, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                chunks = self.add_text_file(file_path)
                if chunks > 0:
                    total_chunks += chunks
                    files_added += 1
        
        print(f"📁 Added {files_added} files ({total_chunks} total chunks) from {folder_path}")
        return files_added, total_chunks
    
    def search_documents(self, query, n_results=5, relevance_threshold=0.3):
        """Search for relevant documents with relevance filtering"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Filter results by relevance if distances are available
            if results['distances'] and results['distances'][0]:
                filtered_docs = []
                filtered_metadatas = []
                filtered_distances = []
                
                for i, distance in enumerate(results['distances'][0]):
                    # Lower distance = higher relevance (ChromaDB uses cosine distance)
                    # Only include results that are reasonably relevant
                    if distance < relevance_threshold:
                        filtered_docs.append(results['documents'][0][i])
                        filtered_metadatas.append(results['metadatas'][0][i])
                        filtered_distances.append(distance)
                
                # Return filtered results
                return {
                    'documents': [filtered_docs],
                    'metadatas': [filtered_metadatas], 
                    'distances': [filtered_distances]
                }
            
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return None
    
    def generate_response(self, query, context_docs):
        """Generate response using retrieved context"""
        # Build context from retrieved documents with relevance scores
        context_text = ""
        sources = []
        
        if context_docs:
            context_text = "CONTEXT FROM RELEVANT DOCUMENTS:\n\n"
            for i, doc_info in enumerate(context_docs, 1):
                doc = doc_info['document']
                metadata = doc_info['metadata']
                relevance = doc_info['relevance_score']
                
                filename = metadata.get('filename', f'Document {i}')
                doc_type = metadata.get('type', 'text')
                
                context_text += f"[Source {i}: {filename} - {doc_type} (relevance: {relevance}%)]\n"
                context_text += f"{doc}\n\n"
                
                sources.append({
                    'filename': filename,
                    'type': doc_type,
                    'relevance': relevance
                })
        
        # Create system prompt with context
        system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
If the answer cannot be found in the context, say so clearly.

{context_text}

Please provide a helpful and accurate response based on the context above. Always cite which source document you're using."""
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Add recent chat history (last 4 exchanges)
        if self.chat_history:
            recent_history = self.chat_history[-8:]  # Last 4 exchanges (user + assistant)
            messages.extend(recent_history)
            messages.append({"role": "user", "content": query})
        
        # Send to Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['message']['content']
                
                # Add sources information if we have context
                if sources:
                    answer += "\n\n📖 **Sources used:**\n"
                    for i, source in enumerate(sources, 1):
                        answer += f"{i}. {source['filename']} ({source['type']}) - {source['relevance']}% relevant\n"
                else:
                    answer += "\n\n⚠️ *No relevant documents found in database for this query*"
                
                return answer
            else:
                return f"❌ Error generating response: {response.status_code}"
                
        except Exception as e:
            return f"❌ Error: {e}"
    
    def chat(self, query):
        """Main chat function with RAG"""
        print("🔍 Searching for relevant information...")
        
        # Search for relevant documents with lower threshold for better recall
        search_results = self.search_documents(query, n_results=5, relevance_threshold=0.2)
        
        if not search_results or not search_results['documents'][0]:
            print("⚠️  No relevant documents found in database")
            # Still try to answer with the base model
            context_docs = []
        else:
            # Format search results with relevance scores
            context_docs = []
            for i in range(len(search_results['documents'][0])):
                distance = search_results['distances'][0][i] if search_results['distances'] else 0
                relevance_score = round((1 - distance) * 100, 1)  # Convert to percentage
                
                context_docs.append({
                    'document': search_results['documents'][0][i],
                    'metadata': search_results['metadatas'][0][i],
                    'distance': distance,
                    'relevance_score': relevance_score
                })
            
            print(f"📚 Found {len(context_docs)} relevant documents")
            
            # Show relevance scores for debugging
            for i, doc in enumerate(context_docs, 1):
                filename = doc['metadata'].get('filename', 'Unknown')
                score = doc['relevance_score']
                print(f"   {i}. {filename} (relevance: {score}%)")
        
        # Generate response
        print("🤔 Generating response...")
        response = self.generate_response(query, context_docs)
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Keep only recent history
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        return response, context_docs
    
    def show_database_stats(self):
        """Show database statistics"""
        try:
            count = self.collection.count()
            print(f"\n📊 Database Statistics:")
            print(f"   Total chunks: {count}")
            
            # Get sample of documents to show files
            if count > 0:
                sample = self.collection.get(limit=min(count, 100))
                files = set()
                for metadata in sample['metadatas']:
                    files.add(metadata['filename'])
                
                print(f"   Files in database: {len(files)}")
                if len(files) <= 10:
                    for file in sorted(files):
                        print(f"     - {file}")
                else:
                    for file in sorted(list(files)[:10]):
                        print(f"     - {file}")
                    print(f"     ... and {len(files) - 10} more files")
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def clear_database(self):
        """Clear all documents from database"""
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection("documents")
            self.collection = self.chroma_client.create_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
            print("🗑️  Database cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
    
    def run_interactive(self):
        """Run interactive chat session"""
        print("🤖 RAG Chatbot with Vector Database")
        print("=" * 50)
        print("💡 This chatbot can answer questions based on documents you've added to its database")
        print("📚 Add documents first, then ask questions about them!")
        print()
        
        self.show_database_stats()
        
        print("\n🆘 Commands:")
        print("   /add <file_path>    - Add a text file to database")
        print("   /add_folder <path>  - Add all text files from folder")
        print("   /stats              - Show database statistics")
        print("   /clear_db           - Clear entire database")
        print("   /clear_chat         - Clear chat history")
        print("   /quit               - Exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split(' ', 1)
                    command = parts[0].lower()
                    
                    if command in ['/quit', '/q', '/exit']:
                        print("👋 Goodbye!")
                        break
                    
                    elif command == '/add' and len(parts) > 1:
                        file_path = parts[1].strip()
                        if os.path.exists(file_path):
                            self.add_text_file(file_path)
                        else:
                            print(f"❌ File not found: {file_path}")
                    
                    elif command == '/add_folder' and len(parts) > 1:
                        folder_path = parts[1].strip()
                        if os.path.exists(folder_path):
                            self.add_folder(folder_path)
                        else:
                            print(f"❌ Folder not found: {folder_path}")
                    
                    elif command == '/stats':
                        self.show_database_stats()
                    
                    elif command == '/clear_db':
                        confirm = input("⚠️  Are you sure you want to clear the entire database? (y/N): ")
                        if confirm.lower() == 'y':
                            self.clear_database()
                    
                    elif command == '/clear_chat':
                        self.chat_history = []
                        print("🧹 Chat history cleared")
                    
                    else:
                        print("❌ Unknown command. Available commands:")
                        print("   /add, /add_folder, /stats, /clear_db, /clear_chat, /quit")
                    
                    continue
                
                # Regular chat
                response, sources = self.chat(user_input)
                
                print(f"\n🤖 RAG Bot: {response}")
                
                # Show sources if any
                if sources:
                    print(f"\n📖 Sources used:")
                    for i, source in enumerate(sources, 1):
                        filename = source['metadata']['filename']
                        chunk_idx = source['metadata']['chunk_index']
                        print(f"   {i}. {filename} (chunk {chunk_idx})")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Chat ended. Goodbye!")
                break

def main():
    """Main function with setup instructions"""
    print("🚀 Setting up RAG Chatbot...")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Cannot connect to Ollama!")
            print("💡 Make sure Ollama is running with: ollama serve")
            return
    except:
        print("❌ Cannot connect to Ollama!")
        print("💡 Make sure Ollama is running with: ollama serve")
        return
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Quick setup if no documents exist
    if chatbot.collection.count() == 0:
        print("\n📚 Your database is empty. Let's add some documents!")
        print("💡 Example: You can add Python files from your current directory")
        
        # Offer to add current directory Python files
        current_dir = os.getcwd()
        python_files = glob.glob("*.py")
        
        if python_files:
            print(f"\n🐍 Found Python files in current directory:")
            for file in python_files[:5]:  # Show first 5
                print(f"   - {file}")
            if len(python_files) > 5:
                print(f"   ... and {len(python_files) - 5} more")
            
            add_them = input(f"\n📥 Add these Python files to database? (y/N): ")
            if add_them.lower() == 'y':
                for file in python_files:
                    chatbot.add_text_file(file)
                print("✅ Files added! Now you can ask questions about your code.")
    
    # Start interactive session
    chatbot.run_interactive()

if __name__ == "__main__":
    main()
