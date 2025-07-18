@echo off
echo 🖼️ Training Images
echo ==================

echo 📁 Navigating to project...
cd /d "C:\Scripts for LLM\local-rag-system"

echo 🐍 Activating virtual environment...
call "C:\local-rag\local-rag-env\Scripts\Activate.bat"

echo 🔍 Analyzing images...
python train_images.py

pause
