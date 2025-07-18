# 🚀 Local RAG System - Quick Reference

## Professional Project Structure

### New Project Location
```
📁 c:\Scripts for LLM\local-rag-system\
```

### Essential Commands

#### 1. Setup (One-time)
```powershell
# Navigate to project
cd "c:\Scripts for LLM\local-rag-system"

# Activate environment
cd c:\local-rag
.\local-rag-env\Scripts\Activate.ps1
cd "c:\Scripts for LLM\local-rag-system"

# Install dependencies
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama serve
```

#### 2. Training Your Data
```powershell
# Train on documents (place in ./data/documents/)
python train_documents.py

# Train on images (place in ./data/images/)
python train_images.py

# Extract images from documents (NEW!)
python extract_document_images.py
```

#### 3. Chat with Your Data
```powershell
# Start the AI chatbot (console)
python chat.py

# OR: Start the Web UI (browser interface)
start_web_ui.bat
# Then open: http://localhost:5000
```

#### 4. Database Utilities
```powershell
# Check database stats
python -m src.utils.database_utils --stats

# Search for content
python -m src.utils.database_utils --search "technical skills"

# List all sources
python -m src.utils.database_utils --sources
```

## 📁 Professional Structure

```
local-rag-system/
├── 📄 config.py              # Project configuration
├── 🤖 chat.py                # Main chat application  
├── 📚 train_documents.py     # Document training
├── 🖼️ train_images.py        # Image training
├── 📷 extract_document_images.py # Extract images from documents
├── 📋 requirements.txt       # Dependencies
├── 📖 README.md             # Full documentation
├── 🔧 src/                  # Source code
│   ├── core/                # Core functionality
│   ├── training/            # Training modules
│   └── utils/               # Utilities
├── 📂 data/                 # Your data
│   ├── documents/           # Documents (.docx, .pdf, .txt, .md, etc.)
│   ├── images/              # Standalone image files (.jpg, .png, etc.)
│   └── analysis_results/    # Generated image analyses
├── 💾 database/             # Vector database
└── 📚 docs/                 # Documentation
```

## 🎯 Key Improvements

- **✅ Modular Code**: Separated into logical modules
- **✅ Professional Structure**: Standard Python project layout
- **✅ Better Configuration**: Centralized config management
- **✅ Database Utilities**: Advanced database inspection tools
- **✅ Comprehensive Docs**: Full README with examples
- **✅ Privacy Protection**: .gitignore for sensitive data

## 💡 Quick Tips

- **Configuration**: Edit `config.py` to adjust models and thresholds
- **Better Questions**: "What technical skills does Igor have with Cisco equipment?"
- **Follow-ups**: Ask specific follow-up questions in the same conversation
- **Database Inspection**: Use `--stats` to check your database content
- **Relevance Tuning**: Lower thresholds in config for broader search
- **Images in Documents**: Use `python extract_document_images.py` to extract embedded images from DOCX, PDF, and PowerPoint files
- **Auto-Extract**: Set `"auto_extract_images": true` in config.py to automatically extract images during document training
- **Supported Formats**: Documents (.docx, .pdf, .txt, .md, .py, .js, .html, .css, .json, .xml, .csv, .pptx, .xlsx) | Images (.jpg, .jpeg, .png, .bmp, .tiff, .webp)

## 🔄 Migration Complete!

Your existing data has been moved to:
- **Documents**: `./data/documents/`
- **Images**: `./data/images/`
- **Database**: `./database/rag_database/`
- **Analysis**: `./data/analysis_results/`

## 📖 Full Documentation

See `README.md` in the project folder for complete documentation, troubleshooting, and advanced usage examples.
