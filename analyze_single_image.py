"""
InstructBLIP Image to Text Converter
This script demonstrates how to use InstructBLIP to convert images to detaile    # F    # Find all image files in the folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))ll image files in the folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))t
and understand diagrams, charts, and other visual content.
"""

import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import os
import glob
from pathlib import Path

def setup_instructblip():
    """Initialize InstructBLIP model and processor"""
    print("Loading InstructBLIP model...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = "cuda"
    else:
        print("CUDA not available. Possible reasons:")
        print("1. Python 3.13 - PyTorch CUDA support not yet available for Python 3.13")
        print("2. Missing CUDA runtime libraries")
        print("Using CPU instead (will be slower)")
        device = "cpu"
    
    # Load the processor and model
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    
    if device == "cuda":
        # Optimized GPU settings for RTX 3080
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=torch.float16,  # Use half precision to save VRAM
            device_map="auto",          # Automatically distribute model across available GPUs
            low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
        )
    else:
        # CPU settings - use smaller model or optimizations
        print("Loading model for CPU processing...")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=torch.float32,  # Full precision for CPU
            device_map="cpu",
            low_cpu_mem_usage=True,     # Reduce RAM usage
        )
    
    print(f"Model loaded successfully on {device.upper()}!")
    
    return processor, model

def process_image_from_url(image_url, prompt, processor, model):
    """Process an image from URL with InstructBLIP"""
    try:
        # Download and open the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Process the image and prompt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        
        # Decode the response
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_local_image(image_path, prompt, processor, model):
    """Process a local image file with InstructBLIP"""
    try:
        # Open and convert image
        image = Image.open(image_path).convert('RGB')
        
        # Process the image and prompt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        
        # Decode the response
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_folder_images(folder_path, prompt, processor, model, output_file=None):
    """Process all images in a folder with InstructBLIP"""
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    
    # Find all image files in the folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return {}
    
    print(f"Found {len(image_files)} image(s) to process...")
    
    results = {}
    
    # Prepare output file if specified
    output_content = []
    if output_file:
        output_content.append(f"InstructBLIP Analysis Results\n")
        output_content.append(f"Folder: {folder_path}\n")
        output_content.append(f"Prompt: {prompt}\n")
        output_content.append("=" * 50 + "\n\n")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\nProcessing ({i}/{len(image_files)}): {os.path.basename(image_path)}")
        
        try:
            result = process_local_image(image_path, prompt, processor, model)
            results[image_path] = result
            
            # Show full result in console (not truncated)
            print(f"Result: {result}")
            
            # Add to output file content if specified
            if output_file:
                output_content.append(f"File: {os.path.basename(image_path)}\n")
                output_content.append(f"Path: {image_path}\n")
                output_content.append(f"Result: {result}\n")
                output_content.append("-" * 30 + "\n\n")
                
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            results[image_path] = error_msg
            print(error_msg)
            
            if output_file:
                output_content.append(f"File: {os.path.basename(image_path)}\n")
                output_content.append(f"Error: {error_msg}\n")
                output_content.append("-" * 30 + "\n\n")
    
    # Save results to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(output_content)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {str(e)}")
    
    return results

def process_images_with_multiple_prompts(folder_path, prompts_dict, processor, model, output_folder=None):
    """Process all images in a folder with multiple different prompts"""
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    
    # Find all image files in the folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return {}
    
    print(f"Found {len(image_files)} image(s) to process with {len(prompts_dict)} prompts each...")
    
    all_results = {}
    
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\nProcessing image ({i}/{len(image_files)}): {image_name}")
        
        image_results = {}
        output_content = [f"Analysis Results for: {image_name}\n"]
        output_content.append(f"Path: {image_path}\n")
        output_content.append("=" * 50 + "\n\n")
        
        for prompt_name, prompt_text in prompts_dict.items():
            print(f"  - Running: {prompt_name}")
            
            try:
                result = process_local_image(image_path, prompt_text, processor, model)
                image_results[prompt_name] = result
                
                output_content.append(f"Prompt: {prompt_name}\n")
                output_content.append(f"Query: {prompt_text}\n")
                output_content.append(f"Result: {result}\n")
                output_content.append("-" * 30 + "\n\n")
                
            except Exception as e:
                error_msg = f"Error with prompt '{prompt_name}': {str(e)}"
                image_results[prompt_name] = error_msg
                print(f"    Error: {error_msg}")
                
                output_content.append(f"Prompt: {prompt_name}\n")
                output_content.append(f"Error: {error_msg}\n")
                output_content.append("-" * 30 + "\n\n")
        
        all_results[image_path] = image_results
        
        # Save individual results file if output folder specified
        if output_folder:
            safe_name = "".join(c for c in os.path.splitext(image_name)[0] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_file = os.path.join(output_folder, f"{safe_name}_analysis.txt")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.writelines(output_content)
            except Exception as e:
                print(f"Error saving results for {image_name}: {str(e)}")
    
    return all_results

def main():
    """Main function to process LOCAL images only"""
    
    # Initialize the model
    processor, model = setup_instructblip()
    
    # Process LOCAL images only - no internet access needed
    print("\n=== Processing LOCAL Images Only ===")
    images_folder = "images"  # Your local images folder
    
    if os.path.exists(images_folder):
        print(f"Processing all images in '{images_folder}' folder...")
        folder_prompt = "Describe this image in detail, including all visible elements, colors, and activities."
        
        folder_results = process_folder_images(
            images_folder, 
            folder_prompt, 
            processor, 
            model, 
            output_file="image_analysis_results.txt"
        )
        
        print(f"Processed {len(folder_results)} images from folder")
        
        # Also run multiple prompts analysis on your local images
        print("\n=== Multiple Prompts Analysis on Local Images ===")
        prompts = example_prompts()
        # Use key prompts for comprehensive analysis
        selected_prompts = {
            "detailed_description": prompts["detailed_description"],
            "object_detection": prompts["object_detection"],
            "text_extraction": prompts["text_extraction"],
            "diagram_analysis": prompts["diagram_analysis"]
        }
        
        print(f"Processing images with {len(selected_prompts)} different prompts...")
        multi_results = process_images_with_multiple_prompts(
            images_folder,
            selected_prompts,
            processor,
            model,
            output_folder="analysis_results"
        )
        
        print(f"Multi-prompt analysis complete for {len(multi_results)} images")
        
    else:
        print(f"ERROR: Folder '{images_folder}' not found!")
        print("Please make sure you have an 'images' folder with your local images.")
        print("Current working directory:", os.getcwd())
        print("Looking for folder at:", os.path.abspath(images_folder))
    
    print("\n=== LOCAL Processing Complete ===")
    print("All processing done using LOCAL images only - no internet connection used!")

# Example usage for different types of prompts
def example_prompts():
    """Examples of different types of prompts for various use cases"""
    
    prompts = {
        "detailed_description": "Provide a comprehensive description of this image, including all visible elements, their positions, colors, and any activities taking place.",
        
        "diagram_analysis": "Analyze this diagram or chart. What type is it? What information does it convey? Describe its structure and key components.",
        
        "technical_diagram": "This appears to be a technical diagram. Explain the system or process it represents, identify key components, and describe how they relate to each other.",
        
        "flowchart_analysis": "If this is a flowchart, describe the process it represents. Identify decision points, start/end points, and the flow of operations.",
        
        "data_visualization": "Analyze this data visualization. What trends, patterns, or insights can be derived from the data presented?",
        
        "text_extraction": "Extract and transcribe any text visible in this image, including labels, captions, titles, or other written content.",
        
        "object_detection": "Identify and list all objects visible in this image, including their approximate locations and relationships to each other.",
        
        "scene_understanding": "Describe the scene, setting, or environment shown in this image. What is the context or situation being depicted?",
        
        "educational_content": "If this image contains educational content (like a lesson, tutorial, or instructional material), explain what is being taught and the key learning points.",
        
        "medical_diagram": "If this is a medical or anatomical diagram, identify the body parts or medical concepts being illustrated and explain their functions."
    }
    
    return prompts

if __name__ == "__main__":
    print("InstructBLIP Image Analysis Tool")
    print("This tool can convert images to detailed text descriptions and analyze diagrams.")
    print("Make sure you have sufficient GPU memory or the model will run on CPU (slower).")
    print("\nStarting analysis...")
    
    main()