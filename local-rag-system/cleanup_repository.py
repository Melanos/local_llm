#!/usr/bin/env python3
"""
Repository Cleanup Script
Removes test files, temporary files, and reorganizes the project structure
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class RepositoryCleanup:
    """Clean up repository by removing test files and organizing structure"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.backup_dir = self.project_root / "backup_removed_files"
        self.cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "removed_files": [],
            "moved_files": [],
            "kept_files": [],
            "errors": []
        }
    
    def create_backup_directory(self):
        """Create backup directory for removed files"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            print(f"📁 Created backup directory: {self.backup_dir}")
    
    def identify_files_to_remove(self):
        """Identify files that should be removed or moved"""
        files_to_remove = [
            # Test files (standalone)
            "test_embeddings_comparison.py",
            "test_knowledge_query.py", 
            "test_new_models.py",
            "test_quality_comparison.py",
            "tests/test_final_verification.py",
            
            # Comparison/analysis files (standalone testing)
            "compare_embeddings.py",
            "compare_all_models.py",
            "check_models.py",
            "demo_clip_vs_vision.py",
            
            # Temporary/experimental files
            "chat_multi_embedding.py",  # Superseded by enhanced_rag_engine
            "custom_st.py",  # Streamlit-specific, not used
            "analyze_network_diagrams.py",  # Specific use case
            "build_network_knowledge.py",  # Specific use case
            
            # Old reports and data files
            "EMBEDDING_ANALYSIS_REPORT.md",  # Superseded by new analysis
            "PROJECT_COMPLETION_SUMMARY.md",  # Old summary
            "embedding_comparison_20250730_101238.json",
            "embedding_comparison_20250730_102053.json",
            
            # Cache and temp files
            "__pycache__/",
            "src/__pycache__/",
            "src/core/__pycache__/",
        ]
        
        files_to_archive = [
            # Move to archive folder instead of delete
            "comprehensive_embedding_analysis_20250730_103213.json",
            "comprehensive_model_test.py",
        ]
        
        files_to_keep = [
            # Core application files
            "config.py",
            "chat.py",
            "train_documents.py", 
            "train_images.py",
            "train_multi_embeddings.py",
            "web_ui/app.py",
            "src/",
            "data/",
            "database/",
            "docs/",
            "README.md",
            "requirements.txt",
            
            # Batch files for easy startup
            "start_chat.bat",
            "start_web_ui.bat", 
            "train_docs.bat",
            "train_imgs.bat",
            "setup_jina.bat",
        ]
        
        return files_to_remove, files_to_archive, files_to_keep
    
    def backup_file(self, file_path: Path):
        """Backup a file before removal"""
        try:
            relative_path = file_path.relative_to(self.project_root)
            backup_path = self.backup_dir / relative_path
            
            # Create backup directories
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.is_file():
                shutil.copy2(file_path, backup_path)
                print(f"📄 Backed up: {relative_path}")
            elif file_path.is_dir():
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.copytree(file_path, backup_path)
                print(f"📁 Backed up directory: {relative_path}")
                
        except Exception as e:
            error_msg = f"Failed to backup {file_path}: {e}"
            self.cleanup_report["errors"].append(error_msg)
            print(f"❌ {error_msg}")
    
    def remove_file_or_directory(self, file_path: Path):
        """Remove a file or directory"""
        try:
            relative_path = file_path.relative_to(self.project_root)
            
            if file_path.is_file():
                file_path.unlink()
                print(f"🗑️  Removed file: {relative_path}")
                self.cleanup_report["removed_files"].append(str(relative_path))
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                print(f"🗑️  Removed directory: {relative_path}")
                self.cleanup_report["removed_files"].append(str(relative_path))
            else:
                print(f"⚠️  File not found: {relative_path}")
                
        except Exception as e:
            error_msg = f"Failed to remove {file_path}: {e}"
            self.cleanup_report["errors"].append(error_msg)
            print(f"❌ {error_msg}")
    
    def create_archive_directory(self):
        """Create archive directory for files to keep but move"""
        archive_dir = self.project_root / "archive"
        if not archive_dir.exists():
            archive_dir.mkdir(parents=True)
            print(f"📁 Created archive directory: {archive_dir}")
        return archive_dir
    
    def move_to_archive(self, file_path: Path, archive_dir: Path):
        """Move file to archive directory"""
        try:
            relative_path = file_path.relative_to(self.project_root)
            archive_path = archive_dir / relative_path.name
            
            if file_path.is_file():
                shutil.move(str(file_path), str(archive_path))
                print(f"📦 Archived: {relative_path} → archive/{relative_path.name}")
                self.cleanup_report["moved_files"].append({
                    "from": str(relative_path),
                    "to": f"archive/{relative_path.name}"
                })
            
        except Exception as e:
            error_msg = f"Failed to archive {file_path}: {e}"
            self.cleanup_report["errors"].append(error_msg)
            print(f"❌ {error_msg}")
    
    def cleanup_empty_directories(self):
        """Remove empty directories"""
        empty_dirs = []
        
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if dir_path.is_dir() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        relative_path = dir_path.relative_to(self.project_root)
                        empty_dirs.append(str(relative_path))
                        print(f"🗑️  Removed empty directory: {relative_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove directory {dir_path}: {e}")
        
        if empty_dirs:
            self.cleanup_report["removed_files"].extend(empty_dirs)
    
    def update_gitignore(self):
        """Update .gitignore with proper exclusions"""
        gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
local-rag-env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project Specific
database/
*.json
*.log
backup_removed_files/
archive/

# Temporary files
temp/
tmp/
*.tmp
*.temp

# Model files (too large for git)
models/
*.bin
*.safetensors
'''
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("📝 Updated .gitignore")
    
    def create_cleanup_summary(self):
        """Create a summary of the cleanup operation"""
        summary_path = self.project_root / "CLEANUP_SUMMARY.md"
        
        summary_content = f"""# Repository Cleanup Summary

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Operation:** Automated repository cleanup and organization

## 🗑️ Files Removed ({len(self.cleanup_report['removed_files'])})

"""
        
        for file_path in self.cleanup_report["removed_files"]:
            summary_content += f"- `{file_path}`\n"
        
        if self.cleanup_report["moved_files"]:
            summary_content += f"\n## 📦 Files Archived ({len(self.cleanup_report['moved_files'])})\n\n"
            for move_info in self.cleanup_report["moved_files"]:
                summary_content += f"- `{move_info['from']}` → `{move_info['to']}`\n"
        
        if self.cleanup_report["errors"]:
            summary_content += f"\n## ❌ Errors ({len(self.cleanup_report['errors'])})\n\n"
            for error in self.cleanup_report["errors"]:
                summary_content += f"- {error}\n"
        
        summary_content += f"""
## 📁 Repository Structure After Cleanup

```
local-rag-system/
├── src/                    # Core source code
│   ├── core/              # RAG engine and chat interface
│   ├── training/          # Document and image trainers
│   └── utils/             # Utility functions
├── web_ui/                # Flask web interface
├── data/                  # Training data
│   ├── documents/         # Text documents
│   └── images/            # Image files
├── docs/                  # Documentation
├── config.py              # Configuration
├── chat.py                # CLI chat interface
├── train_*.py             # Training scripts
├── *.bat                  # Windows batch files
└── README.md              # Main documentation
```

## 🎯 Next Steps

1. **Test the cleaned repository** to ensure functionality
2. **Update documentation** to reflect new structure  
3. **Consider the comprehensive model analysis** for production deployment
4. **Archive contains** test results and analysis data for reference

## 🔧 Recommended Configuration

Based on the comprehensive analysis:
- **Use CLIP ViT-B/32** as the primary embedding model
- **Remove Nomic Embed** from production (quality issues)
- **Keep Jina v4** for specialized high-dimension use cases

---

*This cleanup was performed automatically. Backup files are available in `backup_removed_files/` directory.*
"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📄 Created cleanup summary: {summary_path}")
    
    def run_cleanup(self):
        """Execute the complete cleanup process"""
        print("🧹 REPOSITORY CLEANUP STARTING")
        print("=" * 50)
        
        # Create backup directory
        self.create_backup_directory()
        
        # Create archive directory
        archive_dir = self.create_archive_directory()
        
        # Get file lists
        files_to_remove, files_to_archive, files_to_keep = self.identify_files_to_remove()
        
        # Backup and remove files
        print(f"\n🗑️  REMOVING {len(files_to_remove)} FILES/DIRECTORIES")
        print("-" * 30)
        
        for file_name in files_to_remove:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.backup_file(file_path)
                self.remove_file_or_directory(file_path)
        
        # Archive files
        if files_to_archive:
            print(f"\n📦 ARCHIVING {len(files_to_archive)} FILES")
            print("-" * 30)
            
            for file_name in files_to_archive:
                file_path = self.project_root / file_name
                if file_path.exists():
                    self.move_to_archive(file_path, archive_dir)
        
        # Clean empty directories
        print(f"\n🗑️  CLEANING EMPTY DIRECTORIES")
        print("-" * 30)
        self.cleanup_empty_directories()
        
        # Update .gitignore
        print(f"\n📝 UPDATING PROJECT FILES")
        print("-" * 30)
        self.update_gitignore()
        
        # Create summary
        self.create_cleanup_summary()
        
        # Save cleanup report
        report_path = self.project_root / "cleanup_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        print(f"\n✅ CLEANUP COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"📊 Removed: {len(self.cleanup_report['removed_files'])} items")
        print(f"📦 Archived: {len(self.cleanup_report['moved_files'])} items")
        print(f"❌ Errors: {len(self.cleanup_report['errors'])} items")
        print(f"📄 Summary: CLEANUP_SUMMARY.md")
        print(f"💾 Report: cleanup_report.json")
        print(f"🔙 Backup: backup_removed_files/")

def main():
    """Run the repository cleanup"""
    cleaner = RepositoryCleanup()
    cleaner.run_cleanup()

if __name__ == "__main__":
    main()
