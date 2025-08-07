#!/usr/bin/env python3
"""
Comprehensive Large File Performance Analysis for RAG System
Tests embedding models with 10-50MB files to analyze enterprise-scale performance
"""
import os
import sys
import json
import time
import tempfile
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import required libraries
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    import psutil
    import gc
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install sentence-transformers torch scikit-learn psutil")
    sys.exit(1)

class LargeFileEmbeddingAnalyzer:
    def __init__(self):
        """Initialize the large file analyzer"""
        self.results = {}
        self.test_files = {}
        self.models_config = {
            # TEXT EMBEDDING MODELS
            'all-MiniLM-L6-v2': {
                'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
                'type': 'Text Embedding',
                'description': 'Fast general-purpose text embeddings',
                'max_tokens': 256,
                'dimensions': 384
            },
            'all-mpnet-base-v2': {
                'model_id': 'sentence-transformers/all-mpnet-base-v2',
                'type': 'Text Embedding', 
                'description': 'High-quality general text embeddings',
                'max_tokens': 384,
                'dimensions': 768
            },
            'BGE-large-en': {
                'model_id': 'BAAI/bge-large-en',
                'type': 'Text Embedding',
                'description': 'State-of-the-art English text embeddings',
                'max_tokens': 512,
                'dimensions': 1024
            },
            'E5-large': {
                'model_id': 'intfloat/e5-large',
                'type': 'Text Embedding',
                'description': 'Microsoft E5 high-quality embeddings',
                'max_tokens': 512,
                'dimensions': 1024
            },
            'CLIP-ViT-B32': {
                'model_id': 'clip-ViT-B-32',
                'type': 'Multimodal Embedding',
                'description': 'Text + Image embeddings (OpenAI CLIP)',
                'max_tokens': 77,
                'dimensions': 512
            },
            'Jina-v4': {
                'model_id': 'jinaai/jina-embeddings-v4',
                'type': 'Text Embedding',
                'description': 'Jina AI high-performance embeddings',
                'max_tokens': 8192,
                'dimensions': 1024
            },
            'Nomic-Embed': {
                'model_id': 'nomic-ai/nomic-embed-text-v1',
                'type': 'Text Embedding',
                'description': 'Privacy-focused embeddings',
                'max_tokens': 2048,
                'dimensions': 768
            }
        }
        
    def create_large_test_files(self):
        """Create test files of various sizes (10-50MB)"""
        print("📄 Creating large test files...")
        
        # Base content blocks for different domains
        base_contents = {
            'business': '''
            Executive Summary: Our quarterly business analysis reveals significant market opportunities in the technology sector.
            The comprehensive financial review indicates strong performance across all business units with revenue growth exceeding projections.
            Strategic initiatives implemented this quarter have resulted in improved operational efficiency and customer satisfaction metrics.
            Market analysis shows competitive advantages in key demographic segments with potential for expansion into emerging markets.
            Financial performance indicators demonstrate sustainable growth patterns with healthy profit margins and cash flow optimization.
            Our business development team has identified strategic partnerships that align with long-term corporate objectives and growth targets.
            Customer acquisition costs have decreased while retention rates have improved, indicating effective marketing strategy implementation.
            Supply chain optimization efforts have resulted in reduced operational costs and improved delivery performance metrics.
            Technology infrastructure investments continue to provide competitive advantages in operational efficiency and data analytics capabilities.
            Human resources initiatives have improved employee satisfaction scores and reduced turnover rates across all departments.
            ''',
            'technology': '''
            Advanced machine learning algorithms and artificial intelligence systems are revolutionizing data processing capabilities.
            Cloud computing infrastructure provides scalable solutions for enterprise-level applications with improved performance metrics.
            Cybersecurity frameworks implement multi-layered protection strategies to safeguard sensitive information and system integrity.
            Software development methodologies incorporate agile practices and continuous integration for rapid deployment cycles.
            Database optimization techniques improve query performance and data retrieval efficiency for large-scale applications.
            Network architecture designs ensure high availability and fault tolerance for mission-critical systems and applications.
            API development standards facilitate seamless integration between disparate systems and third-party services.
            Microservices architecture enables modular application development with improved scalability and maintenance capabilities.
            DevOps practices streamline development workflows and automated testing procedures for quality assurance.
            Mobile application development frameworks support cross-platform compatibility and responsive user interface design.
            ''',
            'research': '''
            Scientific research methodologies employ rigorous experimental design and statistical analysis for evidence-based conclusions.
            Peer review processes ensure research quality and validity through expert evaluation and methodological scrutiny.
            Literature reviews provide comprehensive analysis of existing research findings and identify knowledge gaps for investigation.
            Data collection procedures follow established protocols to maintain research integrity and minimize bias factors.
            Statistical analysis techniques analyze complex datasets to identify patterns and relationships within research variables.
            Hypothesis testing frameworks guide research design and experimental procedures for scientific investigation.
            Research ethics guidelines ensure participant safety and data privacy protection throughout the research process.
            Publication standards require transparent reporting of methodology, results, and limitations for scientific transparency.
            Interdisciplinary collaboration enhances research quality through diverse perspectives and specialized expertise.
            Grant funding opportunities support innovative research projects and facilitate scientific advancement in various fields.
            '''
        }
        
        file_sizes = [
            ('10MB', 10 * 1024 * 1024),
            ('25MB', 25 * 1024 * 1024),  
            ('50MB', 50 * 1024 * 1024)
        ]
        
        for domain, base_content in base_contents.items():
            for size_name, target_size in file_sizes:
                filename = f"test_large_{size_name}_{domain}.txt"
                filepath = os.path.join('data', 'test_docs', filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Generate content to reach target size
                content = ""
                while len(content.encode('utf-8')) < target_size:
                    content += base_content + "\n\n"
                
                # Trim to exact size
                content_bytes = content.encode('utf-8')[:target_size]
                final_content = content_bytes.decode('utf-8', errors='ignore')
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                
                actual_size = os.path.getsize(filepath)
                self.test_files[filename] = {
                    'path': filepath,
                    'size_mb': actual_size / (1024 * 1024),
                    'domain': domain,
                    'target_size': size_name
                }
                
                print(f"✅ Created {filename}: {actual_size / (1024 * 1024):.1f}MB")

    def chunk_text(self, text, max_tokens=500, overlap=50):
        """Chunk large text into smaller pieces"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_tokens - overlap):
            chunk = ' '.join(words[i:i + max_tokens])
            chunks.append(chunk)
            
        return chunks

    def measure_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def test_model_performance(self, model_name, model_config):
        """Test a model's performance on large files"""
        print(f"\n🧪 Testing {model_name}...")
        
        try:
            # Load model
            start_mem = self.measure_memory_usage()
            load_start = time.time()
            
            model = SentenceTransformer(model_config['model_id'])
            
            load_time = time.time() - load_start
            load_mem = self.measure_memory_usage() - start_mem
            
            model_results = {
                'model_info': model_config,
                'loading': {
                    'time_seconds': load_time,
                    'memory_mb': load_mem,
                    'status': 'success'
                },
                'file_results': {}
            }
            
            # Test on each large file
            for filename, file_info in self.test_files.items():
                print(f"  📄 Processing {filename} ({file_info['size_mb']:.1f}MB)...")
                
                try:
                    # Read file
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Chunk the content
                    max_tokens = model_config.get('max_tokens', 500)
                    chunks = self.chunk_text(content, max_tokens=max_tokens)
                    
                    # Measure embedding performance
                    embed_start = time.time()
                    start_mem = self.measure_memory_usage()
                    
                    embeddings = []
                    chunk_times = []
                    
                    for i, chunk in enumerate(chunks[:100]):  # Limit to first 100 chunks for testing
                        chunk_start = time.time()
                        embedding = model.encode([chunk])
                        chunk_time = time.time() - chunk_start
                        chunk_times.append(chunk_time)
                        embeddings.append(embedding[0])
                        
                        if i % 20 == 0:  # Progress indicator
                            print(f"    Processed {i+1}/{min(100, len(chunks))} chunks...")
                    
                    embed_time = time.time() - embed_start
                    embed_mem = self.measure_memory_usage() - start_mem
                    
                    # Calculate metrics
                    chunks_processed = len(embeddings)
                    avg_chunk_time = np.mean(chunk_times)
                    chunks_per_second = chunks_processed / embed_time if embed_time > 0 else 0
                    
                    # Quality test - similarity between chunks
                    if len(embeddings) > 1:
                        similarities = []
                        for i in range(min(10, len(embeddings)-1)):
                            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                            similarities.append(sim)
                        avg_similarity = np.mean(similarities)
                    else:
                        avg_similarity = 0.0
                    
                    file_result = {
                        'file_size_mb': file_info['size_mb'],
                        'total_chunks': len(chunks),
                        'processed_chunks': chunks_processed,
                        'processing_time_seconds': embed_time,
                        'memory_usage_mb': embed_mem,
                        'chunks_per_second': chunks_per_second,
                        'avg_chunk_time_seconds': avg_chunk_time,
                        'avg_similarity_score': avg_similarity,
                        'status': 'success'
                    }
                    
                    model_results['file_results'][filename] = file_result
                    
                    print(f"    ✅ {chunks_per_second:.1f} chunks/s, {avg_similarity:.3f} similarity")
                    
                except Exception as e:
                    print(f"    ❌ Error processing {filename}: {e}")
                    model_results['file_results'][filename] = {
                        'status': 'error',
                        'error': str(e)
                    }
                
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return model_results
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return {
                'model_info': model_config,
                'loading': {
                    'status': 'error',
                    'error': str(e)
                },
                'file_results': {}
            }

    def run_comprehensive_analysis(self):
        """Run the complete large file analysis"""
        print("🚀 Starting Comprehensive Large File Analysis")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Create test files
        self.create_large_test_files()
        
        # Test each model
        print(f"\n🧪 Testing {len(self.models_config)} embedding models...")
        
        for model_name, model_config in self.models_config.items():
            try:
                result = self.test_model_performance(model_name, model_config)
                self.results[model_name] = result
            except Exception as e:
                print(f"❌ Critical error testing {model_name}: {e}")
                self.results[model_name] = {
                    'status': 'critical_error',
                    'error': str(e)
                }
        
        # Generate report
        self.generate_comprehensive_report()
        
        print("\n🎉 Large file analysis complete!")
        print("📄 Check LARGE_FILE_COMPREHENSIVE_REPORT.md for detailed results")

    def generate_comprehensive_report(self):
        """Generate detailed markdown report"""
        report_content = f"""# 📊 LARGE FILE PERFORMANCE ANALYSIS

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Test Files**: {len(self.test_files)} files (10MB, 25MB, 50MB)  
**Models Tested**: {len(self.models_config)} embedding models  
**Analysis Type**: Enterprise-scale performance testing

## 🔍 Model Categories & Specifications

### Text Embedding Models
"""
        
        # Categorize models
        text_models = {}
        multimodal_models = {}
        
        for name, config in self.models_config.items():
            if config['type'] == 'Text Embedding':
                text_models[name] = config
            else:
                multimodal_models[name] = config
        
        # Text models table
        report_content += """
| Model | Max Tokens | Dimensions | Description | Status |
|-------|------------|------------|-------------|---------|
"""
        
        for name, config in text_models.items():
            status = "✅ Working" if name in self.results and self.results[name].get('loading', {}).get('status') == 'success' else "❌ Failed"
            report_content += f"| **{name}** | {config['max_tokens']} | {config['dimensions']} | {config['description']} | {status} |\n"
        
        if multimodal_models:
            report_content += "\n### Multimodal Embedding Models\n"
            report_content += """
| Model | Max Tokens | Dimensions | Description | Status |
|-------|------------|------------|-------------|---------|
"""
            for name, config in multimodal_models.items():
                status = "✅ Working" if name in self.results and self.results[name].get('loading', {}).get('status') == 'success' else "❌ Failed"
                report_content += f"| **{name}** | {config['max_tokens']} | {config['dimensions']} | {config['description']} | {status} |\n"
        
        # Performance summary
        report_content += "\n## 🏆 Large File Performance Results\n\n"
        
        # Create performance table
        working_models = {name: data for name, data in self.results.items() 
                         if data.get('loading', {}).get('status') == 'success'}
        
        if working_models:
            report_content += """
| Model | Avg Chunks/Sec | Memory Usage (MB) | Quality Score | Best File Size | Status |
|-------|----------------|-------------------|---------------|----------------|---------|
"""
            
            for model_name, model_data in working_models.items():
                file_results = model_data.get('file_results', {})
                if file_results:
                    # Calculate averages
                    chunks_per_sec = []
                    memory_usage = []
                    quality_scores = []
                    
                    for file_result in file_results.values():
                        if file_result.get('status') == 'success':
                            chunks_per_sec.append(file_result.get('chunks_per_second', 0))
                            memory_usage.append(file_result.get('memory_usage_mb', 0))
                            quality_scores.append(file_result.get('avg_similarity_score', 0))
                    
                    if chunks_per_sec:
                        avg_chunks = np.mean(chunks_per_sec)
                        avg_memory = np.mean(memory_usage)
                        avg_quality = np.mean(quality_scores)
                        
                        # Find best performing file size
                        best_file = max(file_results.items(), 
                                      key=lambda x: x[1].get('chunks_per_second', 0) if x[1].get('status') == 'success' else 0)
                        best_size = self.test_files.get(best_file[0], {}).get('target_size', 'Unknown')
                        
                        report_content += f"| **{model_name}** | {avg_chunks:.1f} | {avg_memory:.1f} | {avg_quality:.3f} | {best_size} | ✅ Working |\n"
                    else:
                        report_content += f"| **{model_name}** | - | - | - | - | ❌ No results |\n"
                else:
                    report_content += f"| **{model_name}** | - | - | - | - | ❌ No data |\n"
        
        # Quality score explanation
        report_content += """
## 📊 Quality Score Explanation

**Quality Score Range**: 0.0 to 1.0  
**Measurement**: Cosine similarity between consecutive document chunks  
**Interpretation**:
- **0.9-1.0**: Excellent semantic consistency (90-100% accuracy)
- **0.7-0.9**: Good semantic consistency (70-90% accuracy)  
- **0.5-0.7**: Moderate semantic consistency (50-70% accuracy)
- **0.3-0.5**: Low semantic consistency (30-50% accuracy)
- **0.0-0.3**: Poor semantic consistency (0-30% accuracy)

Higher quality scores indicate better preservation of semantic meaning across document chunks, 
which translates to more accurate retrieval and better user experience in RAG applications.

## 💡 Enterprise Recommendations

### For 10-50MB Files:
"""
        
        # Find best performers
        if working_models:
            best_speed = max(working_models.items(), 
                           key=lambda x: max([fr.get('chunks_per_second', 0) 
                                            for fr in x[1].get('file_results', {}).values() 
                                            if fr.get('status') == 'success'], default=0))
            
            best_quality = max(working_models.items(),
                             key=lambda x: max([fr.get('avg_similarity_score', 0)
                                              for fr in x[1].get('file_results', {}).values()
                                              if fr.get('status') == 'success'], default=0))
            
            report_content += f"""
1. **Speed Champion**: {best_speed[0]} - Fastest processing for large documents
2. **Quality Leader**: {best_quality[0]} - Best semantic consistency
3. **Memory Efficiency**: Analyze memory usage patterns for production deployment

### Production Deployment Strategy:
- **Small files (<10MB)**: Use quality-optimized models
- **Medium files (10-25MB)**: Balance speed and quality  
- **Large files (25-50MB)**: Prioritize speed and memory efficiency
- **Enterprise scale**: Consider chunking strategies and parallel processing
"""
        
        # Detailed results
        report_content += "\n## 📋 Detailed Test Results\n\n"
        
        for model_name, model_data in self.results.items():
            report_content += f"### {model_name}\n\n"
            
            if model_data.get('loading', {}).get('status') == 'success':
                loading = model_data['loading']
                report_content += f"**Loading**: {loading['time_seconds']:.2f}s, {loading['memory_mb']:.1f}MB\n\n"
                
                file_results = model_data.get('file_results', {})
                if file_results:
                    report_content += "| File | Size | Chunks/Sec | Memory (MB) | Quality | Status |\n"
                    report_content += "|------|------|------------|-------------|---------|--------|\n"
                    
                    for filename, result in file_results.items():
                        if result.get('status') == 'success':
                            size_mb = result['file_size_mb']
                            chunks_sec = result['chunks_per_second']
                            memory = result['memory_usage_mb']
                            quality = result['avg_similarity_score']
                            status = "✅"
                        else:
                            size_mb = self.test_files.get(filename, {}).get('size_mb', 0)
                            chunks_sec = memory = quality = 0
                            status = "❌"
                        
                        report_content += f"| {filename} | {size_mb:.1f}MB | {chunks_sec:.1f} | {memory:.1f} | {quality:.3f} | {status} |\n"
                else:
                    report_content += "*No file results available*\n"
            else:
                error = model_data.get('loading', {}).get('error', 'Unknown error')
                report_content += f"**Status**: ❌ Failed to load  \n**Error**: {error}\n"
            
            report_content += "\n"
        
        # Save report
        report_file = os.path.join('docs', 'LARGE_FILE_COMPREHENSIVE_REPORT.md')
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save JSON data
        json_file = os.path.join('docs', 'large_file_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_files': self.test_files,
                'models_config': self.models_config,
                'results': self.results
            }, f, indent=2)
        
        print(f"📄 Report saved: {report_file}")
        print(f"📊 Data saved: {json_file}")

if __name__ == "__main__":
    analyzer = LargeFileEmbeddingAnalyzer()
    analyzer.run_comprehensive_analysis()
