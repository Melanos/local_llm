@echo off
echo 📚 Training Documents
echo ====================

echo 📁 Navigating to project...
cd /d "C:\Scripts for LLM\local-rag-system"

echo 🐍 Activating virtual environment...
call "C:\local-rag\local-rag-env\Scripts\Activate.bat"

echo 📄 Training on documents...
python train_documents.py

pause
