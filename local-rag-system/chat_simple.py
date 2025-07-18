#!/usr/bin/env python3
"""
Simple Chat Application - Working version
"""

import os
import sys
from pathlib import Path

# Set working directory to project root
project_root = Path(__file__).parent.absolute()
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Now import modules
try:
    from config import DEFAULT_CONFIG, ensure_directories
    from src.core.rag_engine import RAGEngine
    print("✅ All modules loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Trying fallback to original chat...")
    # Fallback to original implementation
    sys.path.append(str(Path("c:/Scripts for LLM")))
    from chat import RAGChatbot
    
    def main():
        chatbot = RAGChatbot()
        chatbot.start_chat()
    
    if __name__ == "__main__":
        main()
    sys.exit()

from src.core.chat_interface import ChatInterface

def main():
    """Main entry point"""
    print("🤖 Local RAG System - Chat Interface")
    print("=" * 45)
    
    # Ensure directories exist
    ensure_directories()
    
    # Start chat interface
    chat = ChatInterface()
    chat.start_chat()

if __name__ == "__main__":
    main()
