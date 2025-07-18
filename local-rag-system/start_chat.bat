@echo off
echo 🚀 Starting Local RAG System
echo =============================

echo 📁 Navigating to project...
cd /d "C:\Scripts for LLM\local-rag-system"

echo 🐍 Activating virtual environment...
call "C:\local-rag\local-rag-env\Scripts\Activate.bat"

echo 🤖 Starting chat interface...
python chat.py

pause
