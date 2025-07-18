#!/usr/bin/env python3
"""
Quick test script to verify multi-format document support
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.chat_interface import ChatInterface

def test_multi_format_support():
    """Test questions about different document types"""
    
    print("🧪 Testing Multi-Format Document Support")
    print("=" * 50)
    
    # Initialize chat interface
    chat = ChatInterface()
    
    # Test questions for different document types
    test_questions = [
        {
            "question": "What skills does John Smith have?",
            "expected_source": "sample_text.txt",
            "description": "Testing plain text file processing"
        },
        {
            "question": "What is the technical stack mentioned in the project documentation?",
            "expected_source": "project_docs.md", 
            "description": "Testing markdown file processing"
        },
        {
            "question": "What is Igor Matsenko's programming experience?",
            "expected_source": "I.M.Resume.docx",
            "description": "Testing DOCX file processing"
        }
    ]
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {test['description']}")
        print(f"❓ Question: {test['question']}")
        print(f"📄 Expected source: {test['expected_source']}")
        
        try:
            response = chat.process_query(test['question'])
            print(f"✅ Answer: {response}")
            
            if test['expected_source'] in response:
                print(f"✅ Source verification passed!")
            else:
                print(f"⚠️  Expected source '{test['expected_source']}' not found in response")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Multi-format testing complete!")

if __name__ == "__main__":
    test_multi_format_support()
