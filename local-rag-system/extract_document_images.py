#!/usr/bin/env python3
"""
Extract Images from Documents - Standalone script
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.document_image_extractor import DocumentImageExtractor


def main():
    """Extract images from all documents"""
    print("🖼️ Document Image Extractor")
    print("=" * 50)
    print("📁 Extracting images from documents in data/documents/")
    print("💾 Saving extracted images to data/images/extracted/")
    print()
    
    extractor = DocumentImageExtractor()
    results = extractor.extract_all_document_images()
    
    print(f"\n🎯 Extraction Summary:")
    print(f"   📚 Documents processed: {results['processed']}")
    print(f"   🖼️ Images extracted: {results['extracted']}")
    print(f"   ❌ Errors: {results['errors']}")
    
    if results["extracted"] > 0:
        print(f"\n📋 Next Steps:")
        print(f"   1. Review extracted images: data/images/extracted/")
        print(f"   2. Move relevant images to: data/images/")
        print(f"   3. Train on images: python train_images.py")
        print(f"   4. Query about document images in chat!")


if __name__ == "__main__":
    main()
