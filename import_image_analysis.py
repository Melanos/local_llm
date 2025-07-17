"""
🖼️→📚 Image to RAG Database
This script processes images with InstructBLIP and saves the results to your RAG database
so you can later ask questions about your images!
"""

import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import os
import glob
import requests
import json
from datetime import datetime

# Import RAG components
import chromadb
from chromadb.utils import embedding_functions

def setup_instructblip():
    """Initialize InstructBLIP model and processor"""
    print("Loading InstructBLIP model...")
    
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("Using CPU (slower)")
        device = "cpu"
    
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    
    if device == "cuda":
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    
    print(f"Model loaded successfully on {device.upper()}!")
    return processor, model

def setup_rag_database(db_path="./rag_database"):
    """Initialize ChromaDB for storing image analysis results"""
    print("Setting up RAG database...")
    
    client = chromadb.PersistentClient(path=db_path)
    
    embedding_function = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="nomic-embed-text",
    )
    
    try:
        collection = client.create_collection(
            name="image_analyses",
            embedding_function=embedding_function
        )
        print(f"✅ Created new image analysis database at: {db_path}")
    except Exception:
        collection = client.get_collection(
            name="image_analyses",
            embedding_function=embedding_function
        )
        print(f"✅ Connected to existing image analysis database")
    
    return collection

def analyze_image(image_path, processor, model):
    """Analyze image with multiple prompts"""
    prompts = {
        "general_description": "Describe this image in detail, including all visible elements, colors, and activities.",
        "technical_analysis": "If this is a diagram, chart, or technical image, explain what it represents, its key components, and relationships.",
        "text_content": "Extract and transcribe any text visible in this image.",
        "objects_and_locations": "List all objects in this image and their approximate locations.",
        "context_and_purpose": "What is the context or purpose of this image? What information does it convey?"
    }
    
    try:
        image = Image.open(image_path).convert('RGB')
        results = {}
        
        for prompt_name, prompt_text in prompts.items():
            print(f"  - {prompt_name}...")
            
            inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                repetition_penalty=1.5,
                length_penalty=1.0,
            )
            
            result = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            results[prompt_name] = result
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None

def store_image_analysis_in_rag(image_path, analysis_results, collection):
    """Store image analysis results in RAG database"""
    if not analysis_results:
        return
    
    image_name = os.path.basename(image_path)
    timestamp = datetime.now().isoformat()
    
    # Create comprehensive document text
    document_parts = [
        f"IMAGE ANALYSIS: {image_name}",
        f"File: {image_path}",
        f"Analyzed: {timestamp}",
        "",
    ]
    
    for analysis_type, result in analysis_results.items():
        document_parts.append(f"{analysis_type.upper().replace('_', ' ')}:")
        document_parts.append(result)
        document_parts.append("")
    
    document_text = "\n".join(document_parts)
    
    # Create unique ID
    doc_id = f"image_{image_name}_{timestamp.replace(':', '-').replace('.', '-')}"
    
    # Metadata
    metadata = {
        "type": "image_analysis",
        "filename": image_name,
        "filepath": image_path,
        "analyzed_date": timestamp,
        "analysis_types": list(analysis_results.keys())
    }
    
    # Store in database
    collection.add(
        documents=[document_text],
        metadatas=[metadata],
        ids=[doc_id]
    )
    
    print(f"✅ Stored analysis for {image_name} in RAG database")

def process_images_to_rag(images_folder="images"):
    """Process all images and store results in RAG database"""
    print("🚀 Processing Images to RAG Database")
    print("=" * 50)
    
    # Setup models
    processor, model = setup_instructblip()
    collection = setup_rag_database()
    
    # Find images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(images_folder, f"*{ext.upper()}")))
    
    image_files = sorted(list(set(image_files)))  # Remove duplicates
    
    if not image_files:
        print(f"❌ No images found in {images_folder}")
        return
    
    print(f"📸 Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\n[{i}/{len(image_files)}] Processing: {image_name}")
        
        # Analyze image
        results = analyze_image(image_path, processor, model)
        
        # Store in RAG database
        store_image_analysis_in_rag(image_path, results, collection)
    
    print(f"\n🎉 Complete! Processed {len(image_files)} images")
    print("💬 You can now ask questions about your images using the RAG chatbot!")
    
    # Show database stats
    count = collection.count()
    print(f"📊 Database now contains {count} documents")

if __name__ == "__main__":
    process_images_to_rag()
