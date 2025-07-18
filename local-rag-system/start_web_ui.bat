@echo off
echo 🌐 Starting Local RAG System Web UI
echo ===================================

echo 📁 Navigating to project...
cd /d "C:\Scripts for LLM\local-rag-system"

echo 🐍 Activating virtual environment...
call "C:\local-rag\local-rag-env\Scripts\Activate.bat"

echo 🌐 Starting web server...
echo.
echo 🔗 Open your browser and go to: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the server
echo.

python web_ui\app.py

pause
