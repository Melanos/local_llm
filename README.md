# 🤖 Local AI RAG Chatbot System

A complete local AI system that combines image analysis, document processing, and conversational AI without sending any data to external services. This system uses InstructBLIP for image analysis, ChromaDB for vector storage, and Ollama for local language models.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Training (Embeddings)](#training-embeddings)
- [Scripts Overview](#scripts-overview)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)

## ✨ Features

- 🖼️ **Image Analysis**: Analyze images using InstructBLIP (local AI model)
- 📄 **Document Processing**: Extract and process Word documents (.docx)
- 🗃️ **Vector Database**: Store embeddings in local ChromaDB
- 💬 **Conversational AI**: Chat with your documents and images using local LLMs
- 🔒 **Fully Local**: No data sent to external services
- 🚀 **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- 🔍 **Relevance Filtering**: Smart search with relevance scoring

## 🛠️ Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB+ recommended for large models
- **Storage**: 10GB+ free space for models

### Required Services
- **Ollama**: Local LLM server
  ```bash
  # Install Ollama from https://ollama.ai
  # Then install required models:
  ollama pull llama3.2
  ollama pull mistral
  ollama pull nomic-embed-text
  ```

## 🚀 Installation

### 1. Create Virtual Environment
```powershell
# Create virtual environment
cd c:\local-rag
python -m venv local-rag-env

# Activate environment
.\local-rag-env\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow chromadb requests python-docx

# Additional dependencies
pip install accelerate bitsandbytes
```

### 3. Verify GPU Setup
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Directory Structure

```
c:\Scripts for LLM\
├── README.md                    # This file
├── requirements.txt             # Dependencies list
├── chat.py                      # 💬 Main chatbot interface
├── train_images.py              # 🖼️ Image analysis and training
├── train_documents.py           # 📄 Document processing and training
├── analyze_single_image.py      # 🔍 Single image analysis utility
├── import_image_analysis.py     # 📥 Import existing image analyses
├── Documents/                   # Place your .docx files here
│   └── *.docx                  # Word documents for training
├── images/                      # Place your images here
│   └── *.jpg, *.png, etc.      # Images for analysis
├── rag_database/               # Vector database (auto-created)
│   └── chroma.sqlite3          # ChromaDB storage
└── analysis_results/           # Analysis outputs (auto-created)
    └── *_analysis.txt          # Individual image analyses
```

## 🎯 Usage

### Quick Start

1. **Activate Environment**
   ```powershell
   cd c:\local-rag
   .\local-rag-env\Scripts\Activate.ps1
   cd "c:\Scripts for LLM"
   ```

2. **Start Ollama** (in separate terminal)
   ```powershell
   ollama serve
   ```

3. **Add Your Content**
   - Place images in `./images/` folder
   - Place Word documents in `./Documents/` folder

4. **Train the System** (see Training section below)

5. **Start Chatbot**
   ```powershell
   python chat.py
   ```

### Using the Chatbot

Once running, you can ask questions like:
- "What images do you have information about?"
- "Tell me about the kitten image"
- "What are my skills from the resume?"
- "Describe any network diagrams you've seen"
- "Summarize the documents you have"

## 🏋️ Training (Embeddings)

Training means analyzing your content and creating embeddings for the RAG system.

### 1. Train on Images

```powershell
# Analyze all images in ./images/ directory
python train_images.py
```

**What this does:**
- Scans `./images/` directory for image files
- Uses InstructBLIP to generate detailed descriptions
- Creates embeddings using Ollama's nomic-embed-text model
- Stores in ChromaDB vector database
- Skips already processed images

**Supported formats:** .jpg, .jpeg, .png, .bmp, .tiff, .webp

### 2. Train on Documents

```powershell
# Process all .docx files in ./Documents/ directory
python train_documents.py
```

**What this does:**
- Scans `./Documents/` directory for .docx files
- Extracts text content from Word documents
- Chunks text into manageable pieces
- Creates embeddings and stores in ChromaDB
- Skips already processed documents

### 3. Individual Image Analysis (Alternative)

```powershell
# Analyze specific images and save results
python analyze_single_image.py
```

**Then add results to RAG:**
```powershell
python import_image_analysis.py
```

### Training Tips

- **First Run**: Process everything from scratch
- **Updates**: Only new files will be processed (automatic deduplication)
- **Large Files**: Documents are automatically chunked for better retrieval
- **GPU Usage**: Image analysis uses GPU automatically if available

## 📜 Scripts Overview

### Core Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `chat.py` | 💬 Main chat interface | User questions | AI responses with sources |
| `train_images.py` | 🖼️ Image training pipeline | `./images/` folder | Embeddings in ChromaDB |
| `train_documents.py` | 📄 Document training pipeline | `./Documents/` folder | Embeddings in ChromaDB |

### Utility Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `analyze_single_image.py` | 🔍 Standalone image analysis | Test individual images |
| `import_image_analysis.py` | 📥 Import existing analyses | Process pre-analyzed images |

### Configuration Files

- **ChromaDB**: Automatically managed in `./rag_database/`
- **Models**: Downloaded to Ollama's model directory
- **Cache**: Python cache in `__pycache__/` (auto-generated)

## 🔧 Troubleshooting

### Common Issues

#### "CUDA out of memory"
```powershell
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU
# Or restart Python session
```

#### "Module not found"
```powershell
# Ensure virtual environment is activated
.\local-rag-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### "Ollama connection failed"
```powershell
# Start Ollama service
ollama serve

# Check if models are installed
ollama list
ollama pull nomic-embed-text  # If missing
```

#### "No relevant documents found"
```powershell
# Check database contents
python -c "
import chromadb
client = chromadb.PersistentClient('./rag_database')
collection = client.get_collection('documents')
print(f'Documents: {collection.count()}')
"
```

### Performance Tips

1. **GPU Memory**: Close other applications using GPU
2. **Large Images**: Resize very large images before processing
3. **Many Documents**: Process in batches if memory issues occur
4. **Search Quality**: Use specific questions for better results

## ⚙️ Configuration

### Model Settings

Edit the model configurations in each script:

```python
# In train_images.py - Change InstructBLIP model
model_name = "Salesforce/instructblip-vicuna-7b"  # Default
# model_name = "Salesforce/instructblip-vicuna-13b"  # Larger model

# In chat.py - Change chat model
chat_model = "llama3.2"  # Default
# chat_model = "mistral"  # Alternative
```

### Database Settings

```python
# Change database location
db_path = "./rag_database"  # Default
# db_path = "/path/to/custom/database"  # Custom location
```

### Relevance Threshold

```python
# In chat.py - Adjust search sensitivity
relevance_threshold = 0.6  # Default (60% relevance)
# Lower = more results, Higher = more precise
```

## 🎉 Success Indicators

### After Training
- "✅ Added X documents to database"
- "📊 Database now contains X documents"
- Image analysis files created in `./analysis_results/`

### During Chat
- "📚 Found X relevant documents"
- Relevance scores shown (e.g., "85.2% relevant")
- Sources cited at end of responses

### System Health
```powershell
# Check GPU usage
nvidia-smi

# Check Ollama models
ollama list

# Check database size
dir rag_database
```

## 📞 Support

For issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Verify all prerequisites are installed
3. Ensure Ollama is running: `ollama serve`
4. Check virtual environment is activated

---

**Happy chatting with your local AI! 🚀**
