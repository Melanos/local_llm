"""
📚 Add Images to RAG Database
This script analyzes images using InstructBLIP and adds the results to the RAG database
so you can ask questions about your images.
"""

import os
import requests
import json
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

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
            name="documents", 
            embedding_function=embedding_function
        )
        print(f"✅ Created new RAG database")
    except Exception:
        collection = client.get_collection(
            name="documents",
            embedding_function=embedding_function
        )
        print(f"✅ Connected to existing RAG database")
    
    return collection

def setup_instructblip():
    """Initialize InstructBLIP model for image analysis"""
    print("🔧 Loading InstructBLIP model...")
    
    model_name = "Salesforce/instructblip-vicuna-7b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    print(f"✅ Model loaded on {device}")
    return processor, model, device

def detect_image_type(image_path):
    """Detect image type to choose appropriate analysis prompt"""
    filename = os.path.basename(image_path).lower()
    
    # Technical diagrams/documents
    if any(keyword in filename for keyword in ['diagram', 'chart', 'network', 'topology', 'schema', 'flow', 'technical']):
        return 'technical'
    
    # Screenshots or documents
    if any(keyword in filename for keyword in ['screenshot', 'screen', 'doc', 'document', 'pdf', 'scan']):
        return 'document'
    
    # Photos of people/animals/scenes
    if any(keyword in filename for keyword in ['photo', 'img', 'picture', 'cat', 'dog', 'person', 'face']):
        return 'general'
    
    # Default to comprehensive
    return 'comprehensive'

def get_analysis_prompt(image_type):
    """Get appropriate prompt based on image type"""
    prompts = {
        'technical': """Analyze this technical image focusing on:
1. Identify any diagrams, charts, schematics, or technical drawings
2. Describe all visible text, labels, annotations, and numbers
3. Explain technical concepts, connections, and data flows shown
4. Note symbols, components, and their relationships
5. Describe the overall technical purpose and structure
Be precise about technical details.""",
        
        'document': """Analyze this document/screenshot carefully:
1. Transcribe all visible text accurately, including titles and headings
2. Describe the document layout, structure, and formatting
3. Identify the type of document (form, webpage, report, etc.)
4. Note any tables, lists, or structured data
5. Capture any important information, numbers, or key details
Focus on text accuracy and document structure.""",
        
        'general': """Describe this image comprehensively:
1. Main subjects, objects, people, or animals present
2. Setting, environment, and background details
3. Colors, lighting, composition, and visual style
4. Activities, actions, or emotions depicted
5. Any text, signs, or written content visible
Provide rich descriptive details.""",
        
        'comprehensive': """Analyze this image in detail covering all aspects:
1. What you see (objects, people, scenes, activities)
2. Colors, composition, lighting, and visual elements
3. Any text, diagrams, technical content, or data
4. The context, purpose, and type of image
5. Important details, symbols, or notable features
Be thorough and capture both visual and informational content."""
    }
    
    return prompts.get(image_type, prompts['comprehensive'])

def analyze_image(image_path, processor, model, device):
    """Analyze a single image using InstructBLIP with dynamic prompts"""
    try:
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Detect image type and get appropriate prompt
        image_type = detect_image_type(image_path)
        prompt = get_analysis_prompt(image_type)
        
        print(f"  📝 Using {image_type} analysis prompt")
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        # Move to device
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate description with better error handling
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=3,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
                # Clear GPU cache after generation
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  GPU memory error, retrying with simpler settings...")
                torch.cuda.empty_cache()
                
                # Retry with simpler settings
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Simpler generation
                    num_beams=1,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
        
        # Decode output
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response (remove the prompt)
        if prompt in generated_text:
            description = generated_text.replace(prompt, "").strip()
        else:
            description = generated_text.strip()
        
        # Validate output quality
        if len(description.strip()) < 10:
            print(f"  ⚠️  Short description generated, may indicate processing issue")
        
        return description
        
    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg:
            print(f"  ❌ GPU Error analyzing {os.path.basename(image_path)}: {e}")
            print("  💡 Try reducing image size or using CPU mode")
        else:
            print(f"  ❌ Error analyzing {os.path.basename(image_path)}: {e}")
        return None

def process_images_from_directory(images_dir="./images"):
    """Process all images in the images directory"""
    collection = setup_rag_database()
    processor, model, device = setup_instructblip()
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        print("💡 Please create the './images' directory and add your images there")
        return
    
    # Find all image files
    image_files = []
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(os.path.join(images_dir, file))
    
    if not image_files:
        print(f"❌ No image files found in {images_dir}")
        print(f"💡 Supported formats: {', '.join(supported_formats)}")
        return
    
    print(f"🖼️  Found {len(image_files)} images to analyze")
    
    # Check for existing analysis to avoid duplicates
    existing_docs = collection.get()
    existing_images = set()
    if existing_docs['metadatas']:
        for metadata in existing_docs['metadatas']:
            if metadata.get('type') == 'image_analysis':
                existing_images.add(metadata.get('image_filename', ''))
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        
        print(f"🔍 [{i}/{len(image_files)}] Analyzing: {filename}")
        
        # Skip if already processed
        if filename in existing_images:
            print(f"  ⏭️  Already processed")
            skipped_count += 1
            continue
        
        # Analyze image
        description = analyze_image(image_path, processor, model, device)
        
        if description:
            # Get image info for metadata
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_format = img.format
            except:
                width = height = 0
                image_format = "unknown"
            
            # Create enhanced document for RAG
            doc_content = f"""Image Analysis: {filename}

File: {filename}
Dimensions: {width}x{height}
Format: {image_format}
Analysis Type: {detect_image_type(image_path)}

Detailed Description:
{description}"""
            
            # Create enhanced metadata
            metadata = {
                "type": "image_analysis",
                "filename": f"{filename}_analysis",
                "image_filename": filename,
                "image_path": image_path,
                "image_width": width,
                "image_height": height,
                "image_format": image_format,
                "analysis_type": detect_image_type(image_path),
                "added_date": datetime.now().isoformat(),
                "source": "instructblip_analysis",
                "description_length": len(description)
            }
            
            # Add to database
            doc_id = f"image_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            collection.add(
                documents=[doc_content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"  ✅ Added to RAG database ({len(description)} chars)")
            processed_count += 1
            
            # Also save to individual file for reference
            analysis_dir = "./analysis_results"
            os.makedirs(analysis_dir, exist_ok=True)
            
            analysis_file = os.path.join(analysis_dir, f"{os.path.splitext(filename)[0]}_analysis.txt")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(doc_content)
        else:
            print(f"  ❌ Failed to analyze")
            error_count += 1
    
    # Final cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Show comprehensive final stats
    total_docs = collection.count()
    print(f"\n📊 Processing Complete!")
    print(f"   Total images found: {len(image_files)}")
    print(f"   ✅ Successfully processed: {processed_count}")
    print(f"   ⏭️  Already processed: {skipped_count}")
    print(f"   ❌ Failed to process: {error_count}")
    print(f"   📁 Total documents in database: {total_docs}")
    
    if processed_count > 0:
        print(f"\n🎉 Added {processed_count} new image analyses to your RAG database!")
        print("💬 You can now ask questions about your images using the RAG chatbot!")
    elif skipped_count > 0:
        print("\n💡 All images were already processed. Add new images to analyze more content!")
    else:
        print("\n⚠️  No images were successfully processed. Check for errors above.")
def test_rag_search(query="What images do you have?"):
    """Test RAG search functionality with better output"""
    print(f"\n🔍 Testing RAG search with query: '{query}'")
    
    try:
        collection = setup_rag_database()
        
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if results['documents'][0]:
            print("✅ Search working! Found relevant documents:")
            
            for i, doc in enumerate(results['documents'][0], 1):
                metadata = results['metadatas'][0][i-1]
                distance = results['distances'][0][i-1] if results['distances'] else 0
                relevance = round((1 - distance) * 100, 1)
                
                filename = metadata.get('image_filename', 'Unknown')
                analysis_type = metadata.get('analysis_type', 'unknown')
                
                print(f"\n{i}. {filename} ({analysis_type} analysis, {relevance}% relevant)")
                print(f"   Preview: {doc[:150]}...")
        else:
            print("⚠️  No documents found")
            
    except Exception as e:
        print(f"❌ Search error: {e}")

if __name__ == "__main__":
    print("�️  Image Analysis and RAG Training")
    print("=" * 50)
    
    process_images_from_directory("./images")
    test_rag_search("What images have been analyzed?")
    
    print("\n🎉 Setup complete!")
    print("💡 Next steps:")
    print("   1. Run: python rag_chatbot.py")
    print("   2. Ask questions like:")
    print("      - 'What images do you have information about?'")
    print("      - 'Tell me about the kitten image'")
    print("      - 'What diagrams have been analyzed?'")
    print("      - 'Describe any network topology diagrams')")
