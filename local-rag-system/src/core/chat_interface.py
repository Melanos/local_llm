"""
Chat Interface - Interactive chatbot with conversation history
"""

from typing import List, Dict, Any
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_engine import RAGEngine
from config import DEFAULT_CONFIG


class ChatInterface:
    """Interactive chatbot interface with conversation history"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.rag_engine = RAGEngine(config)
        self.chat_history = []
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved context"""
        # Build context from retrieved documents with relevance scores
        context_text = ""
        sources = []
        
        if context_docs:
            context_text = "CONTEXT FROM RELEVANT DOCUMENTS:\n\n"
            for i, doc_info in enumerate(context_docs, 1):
                doc = doc_info['document']
                metadata = doc_info['metadata']
                relevance_score = doc_info['relevance_score']
                
                context_text += f"Document {i} (Relevance: {relevance_score}%):\n"
                context_text += f"{doc}\n\n"
                
                # Track sources
                source = metadata.get('source', 'Unknown')
                if source not in sources:
                    sources.append(source)
        
        # Include recent chat history for context
        history_context = ""
        if len(self.chat_history) > 0:
            history_context = "\n\nRECENT CONVERSATION:\n"
            for entry in self.chat_history[-3:]:  # Last 3 exchanges
                history_context += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
        
        # Create the prompt
        prompt = f"""You are a helpful AI assistant with access to a knowledge base. Answer the user's question based on the provided context and conversation history.

{context_text}
{history_context}

User's current question: {query}

Instructions:
- Use the context documents to provide accurate, detailed answers
- Reference specific information from the documents when relevant  
- If the context doesn't contain relevant information, say so clearly
- Maintain conversation continuity with the chat history
- Be conversational and helpful

Answer:"""

        # Get response from Ollama
        response = self.rag_engine.call_ollama(prompt)
        
        # Add source attribution if we found relevant documents
        if sources and context_docs:
            response += f"\n\n📚 Sources: {', '.join(sources)}"
            response += f" (Found {len(context_docs)} relevant document(s))"
        
        return response
    
    def get_image_info(self) -> str:
        """Get information about images in the database"""
        try:
            # Search for image documents in the main collection
            collection = self.rag_engine.collection
            results = collection.get()
            
            if not results['documents'] or not results['metadatas']:
                return "No images found in the database."
            
            image_list = []
            unique_images = set()
            
            for i, metadata in enumerate(results['metadatas']):
                source_type = metadata.get('source_type', '')
                if source_type == 'image':
                    # Use 'source' field which contains the filename
                    filename = metadata.get('source', 'Unknown')
                    if filename not in unique_images:
                        unique_images.add(filename)
                        analysis_type = metadata.get('image_type', 'Unknown')
                        processed_date = metadata.get('processed_date', 'Unknown')
                        if processed_date != 'Unknown':
                            # Format the date nicely
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                                processed_date = dt.strftime('%Y-%m-%d %H:%M')
                            except:
                                pass
                        image_list.append(f"• {filename} (analyzed as '{analysis_type}' on {processed_date})")
            
            if not image_list:
                return "No images found in the database."
            
            response = f"The database contains {len(unique_images)} images:\n\n"
            response += "\n".join(image_list)
            response += f"\n\n📊 Total image documents: {len([m for m in results['metadatas'] if m.get('source_type') == 'image'])}"
            
            return response
            
        except Exception as e:
            return f"Error retrieving image information: {str(e)}"

    def process_query(self, query: str) -> str:
        """Process a user query and return response"""
        # Check if this is an image-related query
        image_keywords = ['image', 'images', 'picture', 'pictures', 'photo', 'photos']
        if any(keyword in query.lower() for keyword in image_keywords) and ('database' in query.lower() or 'stored' in query.lower() or 'available' in query.lower()):
            return self.get_image_info()
        
        # Search for relevant documents
        relevance_threshold = self.config["relevance"]["chat_threshold"]
        search_results = self.rag_engine.search_documents(
            query, n_results=5, relevance_threshold=relevance_threshold
        )
        
        if not search_results['documents'][0]:
            # No relevant documents found, try with lower threshold
            search_results = self.rag_engine.search_documents(
                query, n_results=5, relevance_threshold=0.1
            )
            
            if not search_results['documents'][0]:
                # Still no results, use base model
                context_docs = []
            else:
                # Format search results with lower threshold warning
                context_docs = []
                for i in range(len(search_results['documents'][0])):
                    distance = search_results['distances'][0][i] if search_results['distances'] else 0
                    relevance_score = round((1 - distance) * 100, 1)
                    
                    context_docs.append({
                        'document': search_results['documents'][0][i],
                        'metadata': search_results['metadatas'][0][i],
                        'distance': distance,
                        'relevance_score': relevance_score
                    })
        else:
            # Format search results with relevance scores
            context_docs = []
            for i in range(len(search_results['documents'][0])):
                distance = search_results['distances'][0][i] if search_results['distances'] else 0
                relevance_score = round((1 - distance) * 100, 1)
                
                context_docs.append({
                    'document': search_results['documents'][0][i],
                    'metadata': search_results['metadatas'][0][i],
                    'distance': distance,
                    'relevance_score': relevance_score
                })
        
        # Generate response
        response = self.generate_response(query, context_docs)
        
        # Add to chat history
        self.chat_history.append({
            'user': query,
            'assistant': response,
            'timestamp': str(len(self.chat_history) + 1)
        })
        
        return response
    
    def start_chat(self):
        """Start the interactive chat interface"""
        print("🤖 Local RAG Chatbot Started!")
        print("💡 Commands: /stats, /clear_chat, /quit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split(' ', 1)
                    command = parts[0].lower()
                    
                    if command in ['/quit', '/q', '/exit']:
                        print("👋 Goodbye!")
                        break
                    
                    elif command == '/stats':
                        stats = self.rag_engine.get_stats()
                        print(f"\n📊 Database Stats:")
                        print(f"   📄 Total documents: {stats['total_documents']}")
                        print(f"   💾 Database path: {stats['database_path']}")
                    
                    elif command == '/clear_chat':
                        self.chat_history = []
                        print("🧹 Chat history cleared")
                    
                    else:
                        print("❌ Unknown command. Available: /stats, /clear_chat, /quit")
                    
                    continue
                
                # Process the query
                print("\n🤔 Thinking...")
                response = self.process_query(user_input)
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    chat = ChatInterface()
    chat.start_chat()


if __name__ == "__main__":
    main()
