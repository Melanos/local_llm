#!/usr/bin/env python3
"""
Main Chat Application - Entry point for the RAG chatbot
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.chat_interface import ChatInterface
from config import ensure_directories

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
