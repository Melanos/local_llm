# Installation & Setup Guide

## 🎯 Overview

This guide will help you set up the Local RAG System for network diagram analysis using vision-language models. The setup includes environment configuration, model installation, and system verification.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **CPU**: 4+ cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space
- **GPU**: Optional (CUDA-compatible for better performance)

#### Recommended Setup
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 16GB+ 
- **Storage**: 50GB+ SSD
- **GPU**: 8GB+ VRAM (RTX 3070/4060 or better)

### Software Dependencies
- Python 3.12+
- Git
- CUDA 12.1+ (for GPU acceleration)
- Ollama

## 🚀 Installation Steps

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd local-rag-system
```

### Step 2: Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv local-rag-env

# Activate environment
# Windows:
local-rag-env\Scripts\activate
# Linux/macOS:
source local-rag-env/bin/activate
```

### Step 3: Install Python Dependencies
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Vision and language models
pip install transformers accelerate

# Vector database and embedding
pip install chromadb sentence-transformers

# Web interface
pip install flask flask-cors

# Document processing
pip install PyPDF2 python-docx python-multipart

# Additional utilities
pip install Pillow requests numpy pandas

# Ollama client
pip install ollama
```

### Step 4: Install Ollama
```bash
# Windows (download installer from ollama.ai)
# Or use winget:
winget install Ollama.Ollama

# Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# macOS:
brew install ollama
```

### Step 5: Download Required Models
```bash
# Start Ollama service (if not auto-started)
ollama serve

# Download chat model
ollama pull llama3.2

# Download embedding model
ollama pull nomic-embed-text
```

### Step 6: Verify Installation
```bash
# Check Python environment
python --version  # Should show 3.12+

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check ChromaDB
python -c "import chromadb; print('ChromaDB: OK')"

# Check Ollama
ollama list  # Should show llama3.2 and nomic-embed-text
```

## 🔧 Configuration

### Step 1: Configure System
```bash
# Navigate to project directory
cd local-rag-system

# Initialize project directories
python config.py
```

### Step 2: Set Up Data Directories
```bash
# Create directory structure (if not exists)
mkdir -p data/images
mkdir -p data/documents
mkdir -p data/analysis_results
mkdir -p database
```

### Step 3: Configure Models (Optional)
Edit `config.py` to customize:
```python
DEFAULT_CONFIG = {
    "models": {
        "chat_model": "llama3.2",
        "embedding_model": "nomic-embed-text",
        "vision_model": "Salesforce/instructblip-vicuna-7b"  # Best performer
    },
    "features": {
        "enhanced_image_analysis": True,    # Multi-pass analysis
        "vision_analysis_passes": 3,        # Number of analysis passes
        "auto_extract_images": True         # Extract images from docs
    }
}
```

## 📊 First Use

### Step 1: Add Sample Data
```bash
# Copy your network diagrams to:
cp your_diagrams.png data/images/

# Or copy documents with embedded diagrams to:
cp your_documents.pdf data/documents/
```

### Step 2: Train the System
```bash
# For network diagrams
python train_images.py

# For documents
python train_documents.py

# Or use batch scripts (Windows)
train_imgs.bat
train_docs.bat
```

### Step 3: Test the System
```bash
# Quick test
python test_knowledge_query.py

# Interactive chat
python chat.py

# Web interface
python web_ui/app.py
# Visit: http://localhost:5000
```

## 🧪 Verification Tests

### Test 1: Model Loading
```bash
python -c "
from src.training.image_trainer import ImageTrainer
trainer = ImageTrainer()
print('✅ Vision model loaded successfully')
"
```

### Test 2: Database Connection
```bash
python -c "
from src.core.rag_engine import RAGEngine
engine = RAGEngine()
print('✅ RAG engine initialized successfully')
"
```

### Test 3: Ollama Integration
```bash
python -c "
from src.core.rag_engine import RAGEngine
engine = RAGEngine()
response = engine.call_ollama('Hello, can you respond?')
print(f'✅ Ollama response: {response[:50]}...')
"
```

## ⚠️ Troubleshooting

### Common Issues

#### CUDA Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Check available models
ollama list
```

#### Out of Memory Errors
```python
# Edit config.py for lower memory usage
"chunking": {
    "chunk_size": 500,    # Reduced from 1000
    "overlap": 100        # Reduced from 200
}

# Or force CPU mode
"device": "cpu"  # Add this to config
```

#### Model Download Issues
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Download models manually
python -c "
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
model = InstructBlipForConditionalGeneration.from_pretrained('Salesforce/instructblip-vicuna-7b')
print('Model downloaded successfully')
"
```

### Performance Optimization

#### For Limited Resources
```python
# Use faster model in config.py
"vision_model": "Salesforce/instructblip-flan-t5-xl"
"enhanced_image_analysis": False
"vision_analysis_passes": 1
```

#### For Better Quality
```python
# Use best model with full analysis
"vision_model": "Salesforce/instructblip-vicuna-7b"
"enhanced_image_analysis": True
"vision_analysis_passes": 3
```

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
# Activate environment
source local-rag-env/bin/activate  # Linux/macOS
# or
local-rag-env\Scripts\activate     # Windows

# Update packages
pip install --upgrade torch torchvision transformers chromadb

# Update Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Database Maintenance
```bash
# Clear and rebuild database
rm -rf database/*
python train_images.py
python train_documents.py
```

### Backup Important Data
```bash
# Backup your analysis results
cp -r data/analysis_results/ backup/

# Backup configuration
cp config.py backup/
```

## 🎯 Environment-Specific Notes

### Windows
- Use PowerShell for better command compatibility
- Batch scripts (.bat) are provided for convenience
- Ensure Windows Defender doesn't interfere with model downloads

### Linux
- May need to install additional CUDA libraries
- Use virtual display for headless servers: `export DISPLAY=:0`
- Consider using tmux for long-running training sessions

### macOS
- Metal Performance Shaders (MPS) support available
- May need Xcode command line tools: `xcode-select --install`
- Use Homebrew for easy dependency management

## 📞 Getting Help

### Documentation
- Main README.md: Project overview and features
- MODEL_COMPARISON.md: Detailed model analysis
- QUICK_REFERENCE.md: Common commands and tips

### Troubleshooting Steps
1. Check system requirements
2. Verify all dependencies are installed
3. Test individual components (Ollama, PyTorch, etc.)
4. Review error logs and messages
5. Check GitHub issues for similar problems

### Support Resources
- Project documentation
- Community discussions
- Issue tracker for bug reports
- Model-specific documentation from Hugging Face

---

*Complete setup should take 15-30 minutes depending on internet speed and hardware.*
