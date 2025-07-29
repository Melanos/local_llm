@echo off
echo 🚀 Setting up Jina Embedding Model Support
echo ==========================================

echo.
echo Activating virtual environment...
call C:\local-rag\local-rag-env\Scripts\activate.bat

echo.
echo Installing additional dependencies for Jina models...
pip install sentence-transformers>=2.2.0

echo.
echo Testing enhanced RAG engine...
python src/core/enhanced_rag_engine.py

echo.
echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Run: python train_multi_embeddings.py
echo 2. Run: python compare_embeddings.py  
echo 3. Run: python chat_multi_embedding.py
echo.
pause
