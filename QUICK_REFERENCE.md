# 🚀 Quick Reference Card

## Essential Commands

### 1. Setup (One-time)
```powershell
# Activate environment
cd c:\local-rag
.\local-rag-env\Scripts\Activate.ps1
cd "c:\Scripts for LLM"

# Install dependencies
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama serve
```

### 2. Training Your Data
```powershell
# Train on images (place in ./images/)
python train_images.py

# Train on documents (place in ./Documents/)
python train_documents.py
```

### 3. Chat with Your Data
```powershell
# Start the AI chatbot
python chat.py
```

### 4. Utilities
```powershell
# Analyze a single image
python analyze_single_image.py

# Import existing analysis files
python import_image_analysis.py
```

## What Each Script Does

- **`chat.py`** 💬 - Talk to your AI about images and documents
- **`train_images.py`** 🖼️ - Analyze images and add to database  
- **`train_documents.py`** 📄 - Process Word docs and add to database
- **`analyze_single_image.py`** 🔍 - Test image analysis on one file
- **`import_image_analysis.py`** 📥 - Import pre-existing analysis files

## File Locations

- **Your images**: `./images/` folder
- **Your documents**: `./Documents/` folder  
- **Database**: `./rag_database/` (auto-created)
- **Analysis results**: `./analysis_results/` (auto-created)

---
💡 **Tips for Better Results:**
- Always activate the virtual environment first!
- Ask specific questions: "What Cisco equipment does Igor have experience with?"
- Try follow-up questions: "What are his technical skills?" after asking about Igor
- Use different phrasings: "technologies", "equipment", "experience", "skills"

🎯 **For Better Relevance:**
- Start broad, then get specific: "Tell me about Igor" → "What are his skills?"
- Use technical terms: "programming languages", "networking equipment", "certifications"
- Try multiple angles: "What technologies...", "What experience with...", "List Igor's..."
- If results are poor, try /stats to check your database content
