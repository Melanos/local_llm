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
        self.enhanced_analysis = self.config.get("features", {}).get("enhanced_image_analysis", False)
        
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
        
        # Network diagrams - specific detection
        if any(keyword in filename for keyword in ['network', 'topology', 'router', 'switch', 'lan', 'wan']):
            return 'network_diagram'
        
        # Flowcharts - specific detection  
        if any(keyword in filename for keyword in ['flow', 'process', 'workflow', 'chart', 'diagram']):
            return 'flowchart'
        
        # Technical diagrams/documents
        if any(keyword in filename for keyword in ['technical', 'schema', 'blueprint', 'circuit', 'engineering']):
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
            'technical': """Analyze this technical diagram or document step by step:
1. Identify the type of diagram (network, flowchart, schematic, etc.)
2. List all components, devices, or elements visible
3. Describe connections, relationships, and data flow
4. Extract all text, labels, and technical specifications
5. Note any measurements, values, or configuration details
6. Explain the overall purpose and functionality
Provide a comprehensive technical analysis.""",
            
            'document': """Examine this document image thoroughly:
1. Identify the document type (report, form, letter, etc.)
2. Extract ALL text content, including headers and fine print
3. Describe layout, formatting, and structure
4. Note any tables, charts, or embedded graphics
5. Identify key information like dates, names, numbers
6. Summarize the document's purpose and main points
Provide complete document analysis.""",
            
            'general': """Analyze this image in detail:
1. Describe the main subject and setting
2. List all objects, people, or animals present
3. Note colors, lighting, and composition
4. Read any visible text or signs
5. Describe activities or actions taking place
6. Provide context about the scene or situation
Give a comprehensive visual description.""",
            
            'comprehensive': """Perform a thorough analysis of this image:
1. Overall description and context
2. Detailed inventory of all visible elements
3. Text extraction and reading
4. Spatial relationships and layout
5. Technical details if applicable
6. Colors, materials, and visual properties
7. Any symbolic or contextual meaning
Provide an exhaustive analysis covering all aspects.""",
            
            'network_diagram': """Analyze this network diagram with technical precision:
1. Identify network topology type (star, mesh, ring, etc.)
2. List all network devices (routers, switches, computers, servers)
3. Describe connection types and protocols
4. Extract IP addresses, subnets, and network ranges
5. Note any security features or firewalls
6. Identify redundancy or failover mechanisms
7. Describe data flow patterns and network segments
Provide detailed network architecture analysis.""",
            
            'flowchart': """Examine this flowchart systematically:
1. Identify the process or workflow depicted
2. List all process steps, decision points, and connectors
3. Describe the logical flow and branching paths
4. Extract all text from shapes and decision nodes
5. Note start/end points and loop structures
6. Identify any parallel processes or sub-workflows
7. Explain the overall business process or algorithm
Provide complete flowchart analysis."""
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
            print(f"❌ Error analyzing image {image_path}: {e}")
            return None, None
    
    def enhanced_multi_pass_analysis(self, image_path: Path) -> Tuple[str, str]:
        """Perform enhanced multi-pass analysis for comprehensive understanding"""
        if not self.setup_vision_model():
            return None, None
            
        try:
            image = Image.open(image_path).convert('RGB')
            image_type = self.detect_image_type(image_path)
            
            analyses = []
            
            # Pass 1: Primary analysis with specialized prompt
            primary_prompt = self.get_analysis_prompt(image_type)
            inputs = self.processor(images=image, text=primary_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=512, num_beams=5, temperature=0.7)
            primary_analysis = self.processor.decode(outputs[0], skip_special_tokens=True)
            analyses.append(f"## Primary Analysis ({image_type.title()})\n{primary_analysis}")
            
            # Pass 2: Text extraction focus
            text_prompt = "Extract and transcribe ALL text visible in this image, including labels, captions, numbers, and any written content. List each text element separately."
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=256, num_beams=3, temperature=0.5)
            text_analysis = self.processor.decode(outputs[0], skip_special_tokens=True)
            analyses.append(f"## Text Content\n{text_analysis}")
            
            # Pass 3: Spatial relationships and layout
            spatial_prompt = "Describe the spatial layout and positioning of elements in this image. How are components arranged and connected?"
            inputs = self.processor(images=image, text=spatial_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=256, num_beams=3, temperature=0.6)
            spatial_analysis = self.processor.decode(outputs[0], skip_special_tokens=True)
            analyses.append(f"## Spatial Layout\n{spatial_analysis}")
            
            # Combine all analyses
            combined_analysis = "\n\n".join(analyses)
            
            return combined_analysis, f"{image_type}_enhanced"
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis for {image_path}: {e}")
            return None, None
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
        try:
            collection_name = self.rag_engine.config["database"]["collection_name"]
            collection = self.rag_engine.chroma_client.get_collection(collection_name)
            existing = collection.get(
                where={"$and": [
                    {"source_type": "image"},
                    {"source": image_path.name}
                ]}
            )
            
            if existing['ids']:
                print(f"⚠️  Image {image_path.name} already analyzed")
                return False
        except Exception as e:
            print(f"⚠️  Error checking for existing analysis: {e}")
            # Continue with analysis if check fails
        
        # Analyze the image with appropriate method
        if self.enhanced_analysis:
            print(f"🔍 Performing enhanced multi-pass analysis...")
            analysis, image_type = self.enhanced_multi_pass_analysis(image_path)
        else:
            print(f"🔍 Analyzing as '{self.detect_image_type(image_path)}' image...")
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
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file in image_files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)
        image_files = unique_files
        
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
