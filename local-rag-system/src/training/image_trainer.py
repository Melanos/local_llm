"""
Image Training Module - Process and analyze images for RAG database
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import uuid
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_engine import RAGEngine
from config import DEFAULT_CONFIG, IMAGES_DIR, ANALYSIS_RESULTS_DIR


class ImageTrainer:
    """Handle image analysis and training"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.rag_engine = RAGEngine(config)
        self.processor = None
        self.model = None
        self.device = None
        
    def setup_vision_model(self) -> bool:
        """Initialize InstructBLIP model for image analysis"""
        if self.processor is not None:  # Already initialized
            return True
            
        print("🔧 Loading InstructBLIP model...")
        
        try:
            model_name = self.config["models"]["vision_model"]
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.processor = InstructBlipProcessor.from_pretrained(model_name)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            print(f"✅ Vision model loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading vision model: {e}")
            return False
    
    def detect_image_type(self, image_path: Path) -> str:
        """Detect image type to choose appropriate analysis prompt"""
        filename = image_path.name.lower()
        
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
    
    def get_analysis_prompt(self, image_type: str) -> str:
        """Get appropriate prompt based on image type"""
        prompts = {
            'technical': "Analyze this technical diagram or document. Describe all technical details, components, connections, labels, and any text visible. Include technical specifications, measurements, or data shown.",
            
            'document': "Analyze this document or screenshot. Extract and describe all text content, formatting, structure, and visual elements. Include any data, tables, charts, or technical information visible.",
            
            'general': "Describe this image in detail. Include what you see, the setting, people/animals/objects present, colors, lighting, composition, and any text or signs visible.",
            
            'comprehensive': "Provide a comprehensive analysis of this image. Describe everything visible including objects, people, text, technical details, setting, colors, and context. Be thorough and detailed."
        }
        
        return prompts.get(image_type, prompts['comprehensive'])
    
    def analyze_image(self, image_path: Path) -> Tuple[str, str]:
        """Analyze a single image and return analysis text and image type"""
        if not self.setup_vision_model():
            return None, None
            
        try:
            # Load and analyze image
            image = Image.open(image_path).convert('RGB')
            image_type = self.detect_image_type(image_path)
            prompt = self.get_analysis_prompt(image_type)
            
            print(f"🔍 Analyzing as '{image_type}' image...")
            
            # Process with InstructBLIP
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate analysis
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    do_sample=True,
                    temperature=0.7
                )
            
            analysis = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return analysis, image_type
            
        except Exception as e:
            print(f"❌ Error analyzing {image_path.name}: {e}")
            return None, None
    
    def save_analysis(self, image_path: Path, analysis: str, image_type: str) -> Path:
        """Save analysis to file"""
        # Ensure analysis results directory exists
        ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create analysis filename
        analysis_filename = f"{image_path.stem}_analysis.txt"
        analysis_path = ANALYSIS_RESULTS_DIR / analysis_filename
        
        # Save analysis
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"Image Analysis Report\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Source Image: {image_path.name}\n")
            f.write(f"Image Type: {image_type}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(analysis)
        
        return analysis_path
    
    def process_image(self, image_path: Path) -> bool:
        """Process a single image and add to database"""
        print(f"🖼️  Processing: {image_path.name}")
        
        if not image_path.exists():
            print(f"❌ File not found: {image_path}")
            return False
        
        # Check for existing analysis to avoid duplicates
        existing_docs = self.rag_engine.search_documents(
            f"source:{image_path.name}", n_results=1
        )
        
        if existing_docs['documents'][0]:
            print(f"⚠️  Image {image_path.name} already analyzed")
            return False
        
        # Analyze the image
        analysis, image_type = self.analyze_image(image_path)
        
        if not analysis:
            print(f"❌ Failed to analyze {image_path.name}")
            return False
        
        # Save analysis to file
        analysis_path = self.save_analysis(image_path, analysis, image_type)
        print(f"💾 Analysis saved to: {analysis_path.name}")
        
        # Add to RAG database
        metadata = {
            'source': image_path.name,
            'source_type': 'image',
            'image_type': image_type,
            'file_size': image_path.stat().st_size,
            'analysis_file': analysis_path.name,
            'processed_date': datetime.now().isoformat()
        }
        
        doc_id = self.rag_engine.add_document(analysis, metadata)
        
        if doc_id:
            print(f"✅ Added image analysis to database")
            return True
        else:
            print(f"❌ Failed to add to database")
            return False
    
    def train_images(self, directory: Path = None) -> Dict[str, Any]:
        """Train on all images in the specified directory"""
        if directory is None:
            directory = IMAGES_DIR
        
        directory = Path(directory)
        
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return {"processed": 0, "errors": 0}
        
        print(f"🔍 Looking for images in: {directory}")
        
        # Find all supported image files
        image_files = []
        for pattern in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
            image_files.extend(directory.glob(pattern))
            # Also check uppercase extensions
            image_files.extend(directory.glob(pattern.upper()))
        
        if not image_files:
            print(f"📁 No supported image files found in {directory}")
            print("🖼️  Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
            return {"processed": 0, "errors": 0}
        
        print(f"🖼️  Found {len(image_files)} image(s) to process")
        
        # Process each image
        processed = 0
        errors = 0
        
        for image_file in image_files:
            try:
                if self.process_image(image_file):
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
                errors += 1
        
        # Show results
        print(f"\n✅ Image training complete!")
        print(f"📊 Processed: {processed} images")
        print(f"❌ Errors: {errors} images")
        
        # Show database stats
        stats = self.rag_engine.get_stats()
        print(f"💾 Total documents in database: {stats['total_documents']}")
        
        return {
            "processed": processed,
            "errors": errors,
            "total_in_db": stats['total_documents']
        }


def main():
    """Main entry point for image training"""
    print("🖼️  Image Training Module")
    print("=" * 40)
    
    trainer = ImageTrainer()
    
    # Check if images directory exists and has files
    if not IMAGES_DIR.exists():
        print(f"📁 Creating images directory: {IMAGES_DIR}")
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"🖼️  Please add image files to: {IMAGES_DIR}")
        return
    
    results = trainer.train_images()
    
    if results["processed"] > 0:
        print(f"\n🎉 Successfully analyzed {results['processed']} images!")
        print("💬 You can now chat about your images using: python chat.py")
    else:
        print(f"\n🖼️  No images processed. Add image files to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
