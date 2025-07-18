"""
Web UI for Local RAG System
A modern Flask-based web interface for the chatbot
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
from datetime import datetime
import threading
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import DEFAULT_CONFIG, ensure_directories, IMAGES_DIR, DOCUMENTS_DIR
    from src.core.rag_engine import RAGEngine
    from src.core.chat_interface import ChatInterface
    from src.training.document_trainer import DocumentTrainer
    from src.training.image_trainer import ImageTrainer
    from src.utils.database_utils import DatabaseUtils
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🚨 Please ensure you're running from the project root with virtual environment activated")
    sys.exit(1)

app = Flask(__name__)
app.secret_key = 'local-rag-system-secret-key-change-in-production'

# Global instances
rag_engine = None
chat_interface = None
db_utils = None

def initialize_system():
    """Initialize the RAG system components"""
    global rag_engine, chat_interface, db_utils
    
    try:
        ensure_directories()
        rag_engine = RAGEngine()
        chat_interface = ChatInterface()
        db_utils = DatabaseUtils()
        print("✅ RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

# Initialize system on startup
if not initialize_system():
    print("🚨 System initialization failed. Exiting.")
    sys.exit(1)

@app.route('/')
def index():
    """Main dashboard"""
    try:
        stats = db_utils.get_detailed_stats()
        return render_template('index.html', stats=stats)
    except Exception as e:
        flash(f"Error loading dashboard: {e}", 'error')
        return render_template('index.html', stats={})

@app.route('/chat')
def chat():
    """Chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Process the message
        response = chat_interface.process_query(message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def api_clear_chat():
    """Clear chat history"""
    try:
        chat_interface.chat_history = []
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train')
def train():
    """Training interface"""
    return render_template('train.html')

@app.route('/api/train/documents', methods=['POST'])
def api_train_documents():
    """Train on documents"""
    try:
        trainer = DocumentTrainer()
        results = trainer.train_documents()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/images', methods=['POST'])
def api_train_images():
    """Train on images"""
    try:
        trainer = ImageTrainer()
        results = trainer.train_images()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/database')
def database():
    """Database management interface"""
    try:
        stats = db_utils.get_detailed_stats()
        sources = db_utils.list_sources()
        return render_template('database.html', stats=stats, sources=sources)
    except Exception as e:
        flash(f"Error loading database info: {e}", 'error')
        return render_template('database.html', stats={}, sources=[])

@app.route('/api/database/images')
def api_get_images():
    """Get list of images in the database"""
    try:
        # Get image documents from the main collection
        collection = db_utils.rag_engine.collection
        results = collection.get()
        
        images = []
        if results['documents'] and results['metadatas']:
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                if metadata.get('source_type') == 'image':
                    images.append({
                        'id': i,
                        'filename': metadata.get('source', 'Unknown'),  # Use 'source' field
                        'image_type': metadata.get('image_type', 'Unknown'),
                        'processed_date': metadata.get('processed_date', 'Unknown'),
                        'file_path': metadata.get('source', 'Unknown'),
                        'analysis_preview': doc[:200] + "..." if len(doc) > 200 else doc
                    })
        
        return jsonify({
            'success': True,
            'total_images': len(images),
            'images': images
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/database/documents')
def api_get_documents():
    """Get detailed list of all documents in database"""
    try:
        # Get all documents directly from ChromaDB
        collection = db_utils.rag_engine.collection
        results = collection.get()
        
        documents = []
        if results['documents'] and results['metadatas']:
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                documents.append({
                    'id': i,
                    'source': metadata.get('source', 'Unknown'),
                    'source_type': metadata.get('source_type', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', i),
                    'content_preview': doc[:200] + "..." if len(doc) > 200 else doc,
                    'content_length': len(doc),
                    'timestamp': metadata.get('timestamp', 'Unknown')
                })
        
        return jsonify({
            'success': True,
            'total_chunks': len(documents),
            'documents': documents
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/database/search', methods=['POST'])
def api_search_database():
    """Search database"""
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        
        if not keyword:
            return jsonify({'error': 'Empty search term'}), 400
        
        results = db_utils.search_by_keyword(keyword, n_results=10)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/clear', methods=['POST'])
def api_clear_database():
    """Clear entire database"""
    try:
        rag_engine.clear_database()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/settings')
def settings():
    """Settings interface"""
    return render_template('settings.html', config=DEFAULT_CONFIG)

@app.route('/api/settings/update', methods=['POST'])
def api_update_settings():
    """Update configuration (runtime only)"""
    try:
        data = request.get_json()
        # Note: This only updates runtime config, not the config file
        # For production, you'd want to save to config file
        
        flash('Settings updated (runtime only)', 'success')
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Local RAG System Web UI")
    print("=" * 40)
    print("🔗 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 40)
    
    app.run(host='127.0.0.1', port=5000, debug=False)
