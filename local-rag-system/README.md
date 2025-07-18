# Local RAG System

A professional local Retrieval-Augmented Generation (RAG) system that processes documents and images for AI-powered question answering.

## 🚀 Features

- **📄 Document Processing**: Extract and index content from Word documents (.docx)
- **🖼️ Image Analysis**: Analyze images using InstructBLIP vision model
- **🤖 AI Chat**: Interactive chatbot with conversation history
- **🌐 Modern Web UI**: Beautiful web interface with real-time chat
- **🔍 Semantic Search**: Find relevant content using vector embeddings
- **💾 Local Storage**: All data stays on your machine
- **🎯 Smart Relevance**: Configurable relevance thresholds for better results

## 📁 Project Structure

```
local-rag-system/
├── config.py              # Project configuration
├── start_web_ui.bat        # Start web interface (NEW!)
├── start_chat.bat          # Start terminal chat
├── train_documents.py      # Document training script
├── train_images.py         # Image training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── web_ui/                # Modern web interface
│   ├── app.py             # Flask web application
│   ├── templates/         # HTML templates
│   └── static/            # CSS, JS, assets
├── data/                  # Data directories
│   ├── core/              # Core functionality
│   │   ├── rag_engine.py  # RAG database operations
│   │   └── chat_interface.py # Chat interface
│   ├── training/          # Training modules
│   │   ├── document_trainer.py # Document processing
│   │   └── image_trainer.py    # Image analysis
│   └── utils/             # Utility functions
│       └── database_utils.py   # Database utilities
├── data/                  # Data directories
│   ├── documents/         # Place .docx files here
│   ├── images/            # Place image files here
│   └── analysis_results/  # Generated analysis files
├── database/              # Vector database storage
└── docs/                  # Documentation
```

## � Quick Start (Windows)

### Easy Way - Double-click these files:
- **`start_chat.bat`** - Start the chatbot  
- **`train_docs.bat`** - Train on documents
- **`train_imgs.bat`** - Train on images

### Manual Way:
1. **Activate environment**: 
   ```cmd
   C:\local-rag\local-rag-env\Scripts\Activate.bat
   cd "C:\Scripts for LLM\local-rag-system"
   ```

2. **Train your data**:
   ```cmd
   python train_documents.py
   python train_images.py
   ```

3. **Start chatting**:
   ```cmd
   python chat.py
   ```

## 📚 Usage

### 🌐 **Option 1: Modern Web UI (Recommended)**

1. **Double-click `start_web_ui.bat`**
2. **Open browser to: http://localhost:5000**
3. **Use the beautiful web interface to:**
   - 💬 Chat with AI in real-time
   - 📚 Train documents and images
   - 🔍 Search and manage database
   - ⚙️ Adjust settings

### 💻 **Option 2: Command Line**

### 1. Add Your Data

**Documents**: Place `.docx` files in `data/documents/`
```bash
# Train on documents
python train_documents.py
```

**Images**: Place image files in `data/images/`
```bash
# Train on images  
python train_images.py
```

### 2. Start Chatting

```bash
# Start the chatbot
python chat.py
```

### 3. Chat Commands

- `/stats` - Show database statistics
- `/clear_chat` - Clear conversation history  
- `/quit` - Exit the chatbot

## 🎯 Configuration

Edit `config.py` to customize:

- **Models**: Change AI models used
- **Paths**: Modify data directories
- **Chunking**: Adjust text chunking parameters
- **Relevance**: Fine-tune search thresholds

```python
DEFAULT_CONFIG = {
    "models": {
        "chat_model": "llama3.2",
        "embedding_model": "nomic-embed-text", 
        "vision_model": "Salesforce/instructblip-vicuna-7b"
    },
    "relevance": {
        "search_threshold": 0.3,  # Lower = broader search
        "chat_threshold": 0.2     # Lower = more results
    }
}
```

## 🔧 Advanced Usage

### Database Utilities

```bash
# Show database stats
python -m src.utils.database_utils --stats

# Search for specific content
python -m src.utils.database_utils --search "technical skills"

# List all sources
python -m src.utils.database_utils --sources
```

### Custom Training

```python
from src.training.document_trainer import DocumentTrainer
from src.training.image_trainer import ImageTrainer

# Train on specific directory
doc_trainer = DocumentTrainer()
doc_trainer.train_documents(Path("path/to/docs"))

# Analyze specific image
img_trainer = ImageTrainer()
img_trainer.process_image(Path("path/to/image.jpg"))
```

## 🎨 Supported Formats

**Documents**:
- `.docx` (Microsoft Word)

**Images**:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

## 🚨 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`

2. **CUDA Memory Issues**
   - Reduce batch size in image processing
   - Use CPU instead: edit config to use `"cpu"`

3. **Poor Search Results**
   - Lower relevance thresholds in `config.py`
   - Add more specific keywords to queries
   - Check database content with `--stats`

4. **Import Errors**
   - Ensure virtual environment is activated
   - Install missing packages: `pip install -r requirements.txt`

## 📈 Performance Tips

- **Better Relevance**: Use specific, detailed questions
- **Faster Processing**: Use GPU for image analysis
- **Memory Management**: Process large datasets in batches
- **Query Optimization**: Start broad, then ask specific follow-ups

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** - Local LLM hosting
- **ChromaDB** - Vector database
- **InstructBLIP** - Image analysis model
- **Transformers** - AI model library
