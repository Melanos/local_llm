# Local RAG System

A comprehensive **Retrieval-Augmented Generation (RAG)** system with **multimodal capabilities** that combines document and image processing with advanced language models for intelligent question-answering.

## 🌟 Key Features

- **🤖 Multiple Embedding Models**: CLIP ViT-B/32 (recommended), Jina v4, Nomic Embed
- **🖼️ Multimodal Support**: Process both text documents and images
- **🌐 Web Interface**: Modern Flask-based UI for easy interaction
- **💬 CLI Chat**: Command-line interface for quick queries
- **📊 Advanced Analytics**: Comprehensive model comparison and performance analysis
- **🔧 Easy Setup**: Automated installation and configuration scripts

## 🏆 Performance Highlights

Based on comprehensive testing (see `docs/COMPREHENSIVE_MODEL_ANALYSIS.md`):

- **CLIP ViT-B/32**: 🥇 **Recommended** - 50x faster than alternatives, multimodal support, superior quality
- **Search Speed**: 45+ queries/second with CLIP
- **Embedding Speed**: 55+ documents/second with CLIP  
- **Quality Score**: 0.68 semantic understanding (CLIP) vs 0.35 (Jina v4)

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **Git**
- **10GB+ free disk space** (for models)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd local-rag-system

# Set up virtual environment
python -m venv local-rag-env
source local-rag-env/bin/activate  # Linux/Mac
# OR
local-rag-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Quick Setup (Windows)
Use the automated batch files:
```bash
setup_jina.bat          # Setup Jina embedding model
train_docs.bat          # Train on documents
train_imgs.bat          # Train on images  
start_web_ui.bat        # Start web interface
start_chat.bat          # Start CLI chat
```

### 4. Manual Setup

#### Configure the System
```python
# Edit config.py to select your embedding model
DEFAULT_CONFIG = {
    "models": {
        "embedding_model": "clip",  # Recommended: clip, jina_v4, or nomic
        # ... other settings
    }
}
```

#### Train on Your Data
```bash
# Add documents to data/documents/
python train_documents.py

# Add images to data/images/  
python train_images.py

# Or train on multiple embeddings
python train_multi_embeddings.py
```

#### Start the Application
```bash
# Web interface (recommended)
python web_ui/app.py
# Open http://localhost:5000

# CLI interface
python chat.py
```

## 📁 Project Structure

```
local-rag-system/
├── src/                      # Core source code
│   ├── core/                # RAG engine and chat interface
│   │   ├── enhanced_rag_engine.py  # Main RAG engine with multimodal support
│   │   ├── chat_interface.py       # Chat logic and conversation handling
│   │   └── rag_engine.py           # Base RAG functionality
│   ├── training/            # Document and image trainers
│   │   ├── document_trainer.py     # Text document processing
│   │   └── image_trainer.py        # Image processing and analysis
│   └── utils/               # Utility functions
│       ├── database_utils.py       # ChromaDB operations
│       └── document_image_extractor.py  # Content extraction
├── web_ui/                  # Flask web interface
│   ├── app.py               # Main web application
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS, assets
├── data/                    # Training data
│   ├── documents/           # Text documents (.txt, .md, .pdf)
│   └── images/              # Image files (.png, .jpg, .jpeg)
├── docs/                    # Documentation
│   ├── COMPREHENSIVE_MODEL_ANALYSIS.md  # Performance analysis
│   ├── SETUP_GUIDE.md       # Detailed setup instructions
│   └── QUICK_REFERENCE.md   # Command reference
├── config.py                # Configuration settings
├── chat.py                  # CLI chat interface
├── train_*.py               # Training scripts
└── *.bat                    # Windows automation scripts
```

## 🎯 Usage Examples

### Web Interface
1. **Start the web server**: `python web_ui/app.py`
2. **Open browser**: `http://localhost:5000`
3. **Train data**: Use the Training tab to process documents/images
4. **Chat**: Ask questions in natural language
5. **Manage**: View database contents and search history

### CLI Interface
```bash
python chat.py
> How do neural networks work?
> What's in the network diagram image?
> Explain machine learning concepts
```

### Programmatic Usage
```python
from src.core.enhanced_rag_engine import EnhancedRAGEngine

# Initialize with CLIP (recommended)
engine = EnhancedRAGEngine(embedding_model_key="clip")

# Add documents
engine.add_documents(["AI is transforming industries..."])

# Add images (multimodal)
engine.add_image("path/to/diagram.png")

# Search across all content
results = engine.search_documents("artificial intelligence")
image_results = engine.search_multimodal("technical diagram")
```

## 🔧 Configuration

### Embedding Model Selection

**CLIP ViT-B/32** (Recommended):
```python
"embedding_model": "clip"
```
- ✅ **Best Performance**: 50x faster than alternatives
- ✅ **Multimodal**: Text + Image support
- ✅ **High Quality**: Superior semantic understanding
- ✅ **Resource Efficient**: 512-dimensional embeddings

**Jina v4** (High-Dimension):
```python  
"embedding_model": "jina_v4"
```
- ✅ **High Precision**: 2048-dimensional embeddings
- ✅ **Latest Model**: Recent improvements
- ⚠️ **Slower**: 50x slower than CLIP
- ❌ **Text Only**: No image support

**Nomic Embed** (Privacy):
```python
"embedding_model": "nomic"  
```
- ✅ **Privacy**: Local Ollama execution
- ✅ **No External Calls**: Fully offline
- ❌ **Quality Issues**: Negative similarity scores
- ⚠️ **Requires Debugging**: Not production-ready

### Advanced Configuration
```python
DEFAULT_CONFIG = {
    "models": {
        "embedding_model": "clip",
        "collection_name": "documents_clip",
        "max_chunk_size": 1000,
        "chunk_overlap": 200
    },
    "database": {
        "persist_directory": "./database/rag_database"
    },
    "ui": {
        "max_chat_history": 50,
        "default_n_results": 5
    }
}
```

## 📊 Performance Analysis

### Comprehensive Benchmarking Results

| Model | Speed (queries/sec) | Quality Score | Dimensions | Multimodal |
|-------|-------------------|---------------|------------|------------|
| **CLIP ViT-B/32** | **45.2** | **0.68** | 512 | ✅ |
| Jina v4 | 1.17 | 0.35 | 2048 | ❌ |
| Nomic Embed | 0.46 | -270 | 768 | ❌ |

**CLIP is 50-100x faster** with superior quality scores across all test domains (technical, business, science, general knowledge).

### Use Case Recommendations

- **🎯 General Purpose**: CLIP ViT-B/32
- **⚡ High Performance**: CLIP ViT-B/32  
- **🖼️ Multimodal Tasks**: CLIP ViT-B/32
- **🔒 Privacy-Focused**: Nomic Embed (with quality caveats)
- **💻 Resource-Constrained**: CLIP ViT-B/32

## 🛠️ Development

### Adding New Embedding Models
1. Update `config.py` with model configuration
2. Extend `enhanced_rag_engine.py` with model loading logic
3. Add model-specific handling in training scripts
4. Test with `comprehensive_model_test.py`

### Testing and Validation
```bash
# Comprehensive model testing (archived)
python archive/comprehensive_model_test.py

# Basic functionality test
python -c "from src.core.enhanced_rag_engine import EnhancedRAGEngine; print('✅ Setup working')"
```

## 🔍 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure virtual environment is activated
source local-rag-env/bin/activate  # Linux/Mac
local-rag-env\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**Model Loading Issues**:
```bash
# For CLIP model issues
pip install --upgrade torch torchvision transformers

# For Jina model issues  
pip install --upgrade sentence-transformers

# For Ollama/Nomic issues
# Install Ollama and pull the model:
ollama pull nomic-embed-text
```

**Database Issues**:
```bash
# Clear and rebuild database
python -c "from src.core.enhanced_rag_engine import EnhancedRAGEngine; EnhancedRAGEngine().clear_database()"
python train_documents.py
```

### Performance Optimization

1. **Use CLIP** for best performance (50x speed improvement)
2. **Enable GPU** for CUDA-compatible models
3. **Batch processing** for large document sets
4. **SSD storage** for faster database access

## 📖 Documentation

- **`docs/COMPREHENSIVE_MODEL_ANALYSIS.md`**: Detailed performance analysis
- **`docs/SETUP_GUIDE.md`**: Step-by-step setup instructions  
- **`docs/QUICK_REFERENCE.md`**: Command and API reference
- **`CLEANUP_SUMMARY.md`**: Repository cleanup and organization notes

## 🔄 Recent Updates

- **✅ Comprehensive Model Analysis**: Benchmarked all embedding models
- **✅ Repository Cleanup**: Removed test files, organized structure
- **✅ CLIP Integration**: Added multimodal support and performance optimization
- **✅ Enhanced Documentation**: Updated guides and performance metrics
- **✅ Production Ready**: Optimized for deployment with clear recommendations

## 🤝 Contributing

1. Follow the existing code structure in `src/`
2. Add tests for new functionality  
3. Update documentation for new features
4. Run performance benchmarks for model changes
5. Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for vector database functionality
- **OpenAI CLIP** for multimodal embeddings
- **Jina AI** for advanced embedding models
- **Hugging Face** for transformer models and tools
- **Flask** for the web interface framework

---

**⭐ Star this repository if you find it useful!**

For questions, issues, or contributions, please open an issue on GitHub.
