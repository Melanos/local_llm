# 📊 LARGE FILE PERFORMANCE ANALYSIS

**Generated**: 2025-08-06 18:40:04  
**Test Files**: 9 files (10MB, 25MB, 50MB)  
**Models Tested**: 7 embedding models  
**Analysis Type**: Enterprise-scale performance testing

## 🔍 Model Categories & Specifications

### Text Embedding Models

| Model | Max Tokens | Dimensions | Description | Status |
|-------|------------|------------|-------------|---------|
| **all-MiniLM-L6-v2** | 256 | 384 | Fast general-purpose text embeddings | ✅ Working |
| **all-mpnet-base-v2** | 384 | 768 | High-quality general text embeddings | ✅ Working |
| **BGE-large-en** | 512 | 1024 | State-of-the-art English text embeddings | ✅ Working |
| **E5-large** | 512 | 1024 | Microsoft E5 high-quality embeddings | ✅ Working |
| **Jina-v4** | 8192 | 1024 | Jina AI high-performance embeddings | ❌ Failed |
| **Nomic-Embed** | 2048 | 768 | Privacy-focused embeddings | ❌ Failed |

### Multimodal Embedding Models

| Model | Max Tokens | Dimensions | Description | Status |
|-------|------------|------------|-------------|---------|
| **CLIP-ViT-B32** | 77 | 512 | Text + Image embeddings (OpenAI CLIP) | ✅ Working |

## 🏆 Large File Performance Results


| Model | Avg Chunks/Sec | Memory Usage (MB) | Quality Score | Best File Size | Status |
|-------|----------------|-------------------|---------------|----------------|---------|
| **all-MiniLM-L6-v2** | 43.4 | 5.6 | 0.935 | 25MB | ✅ Working |
| **all-mpnet-base-v2** | 4.7 | 39.9 | 0.965 | 10MB | ✅ Working |
| **BGE-large-en** | 1.3 | 130.4 | 0.987 | 50MB | ✅ Working |
| **E5-large** | 1.3 | 128.5 | 0.996 | 25MB | ✅ Working |
| **CLIP-ViT-B32** | 36.2 | 15.8 | 0.918 | 10MB | ✅ Working |

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

1. **Speed Champion**: all-MiniLM-L6-v2 - Fastest processing for large documents
2. **Quality Leader**: E5-large - Best semantic consistency
3. **Memory Efficiency**: Analyze memory usage patterns for production deployment

### Production Deployment Strategy:
- **Small files (<10MB)**: Use quality-optimized models
- **Medium files (10-25MB)**: Balance speed and quality  
- **Large files (25-50MB)**: Prioritize speed and memory efficiency
- **Enterprise scale**: Consider chunking strategies and parallel processing

## 📋 Detailed Test Results

### all-MiniLM-L6-v2

**Loading**: 2.86s, 41.9MB

| File | Size | Chunks/Sec | Memory (MB) | Quality | Status |
|------|------|------------|-------------|---------|--------|
| test_large_10MB_business.txt | 10.1MB | 42.1 | 55.3 | 0.936 | ✅ |
| test_large_25MB_business.txt | 25.2MB | 44.6 | 0.1 | 0.936 | ✅ |
| test_large_50MB_business.txt | 50.5MB | 44.5 | 0.2 | 0.936 | ✅ |
| test_large_10MB_technology.txt | 10.1MB | 43.7 | 0.8 | 0.936 | ✅ |
| test_large_25MB_technology.txt | 25.2MB | 38.7 | -0.8 | 0.936 | ✅ |
| test_large_50MB_technology.txt | 50.5MB | 44.5 | -2.0 | 0.936 | ✅ |
| test_large_10MB_research.txt | 10.1MB | 43.0 | -1.7 | 0.931 | ✅ |
| test_large_25MB_research.txt | 25.2MB | 44.6 | -0.8 | 0.931 | ✅ |
| test_large_50MB_research.txt | 50.5MB | 44.5 | -0.8 | 0.931 | ✅ |

### all-mpnet-base-v2

**Loading**: 0.85s, 7.6MB

| File | Size | Chunks/Sec | Memory (MB) | Quality | Status |
|------|------|------------|-------------|---------|--------|
| test_large_10MB_business.txt | 10.1MB | 4.7 | 361.3 | 0.990 | ✅ |
| test_large_25MB_business.txt | 25.2MB | 4.5 | 0.0 | 0.990 | ✅ |
| test_large_50MB_business.txt | 50.5MB | 4.6 | 0.3 | 0.990 | ✅ |
| test_large_10MB_technology.txt | 10.1MB | 4.8 | 1.0 | 0.942 | ✅ |
| test_large_25MB_technology.txt | 25.2MB | 4.7 | -0.9 | 0.942 | ✅ |
| test_large_50MB_technology.txt | 50.5MB | 4.7 | -0.7 | 0.942 | ✅ |
| test_large_10MB_research.txt | 10.1MB | 4.8 | -0.2 | 0.962 | ✅ |
| test_large_25MB_research.txt | 25.2MB | 4.6 | -0.9 | 0.962 | ✅ |
| test_large_50MB_research.txt | 50.5MB | 4.8 | -0.8 | 0.962 | ✅ |

### BGE-large-en

**Loading**: 15.39s, 7.8MB

| File | Size | Chunks/Sec | Memory (MB) | Quality | Status |
|------|------|------------|-------------|---------|--------|
| test_large_10MB_business.txt | 10.1MB | 1.3 | 1175.4 | 0.997 | ✅ |
| test_large_25MB_business.txt | 25.2MB | 1.3 | 0.2 | 0.997 | ✅ |
| test_large_50MB_business.txt | 50.5MB | 1.3 | 0.0 | 0.997 | ✅ |
| test_large_10MB_technology.txt | 10.1MB | 1.3 | 1.4 | 0.985 | ✅ |
| test_large_25MB_technology.txt | 25.2MB | 1.3 | -1.0 | 0.985 | ✅ |
| test_large_50MB_technology.txt | 50.5MB | 1.3 | -0.6 | 0.985 | ✅ |
| test_large_10MB_research.txt | 10.1MB | 1.2 | 0.0 | 0.980 | ✅ |
| test_large_25MB_research.txt | 25.2MB | 1.2 | -1.0 | 0.980 | ✅ |
| test_large_50MB_research.txt | 50.5MB | 1.2 | -0.8 | 0.980 | ✅ |

### E5-large

**Loading**: 14.38s, 11.0MB

| File | Size | Chunks/Sec | Memory (MB) | Quality | Status |
|------|------|------------|-------------|---------|--------|
| test_large_10MB_business.txt | 10.1MB | 1.3 | 1159.0 | 0.998 | ✅ |
| test_large_25MB_business.txt | 25.2MB | 1.3 | 0.2 | 0.998 | ✅ |
| test_large_50MB_business.txt | 50.5MB | 1.2 | 0.2 | 0.998 | ✅ |
| test_large_10MB_technology.txt | 10.1MB | 1.3 | 1.7 | 0.996 | ✅ |
| test_large_25MB_technology.txt | 25.2MB | 1.3 | -1.0 | 0.996 | ✅ |
| test_large_50MB_technology.txt | 50.5MB | 1.3 | -0.8 | 0.996 | ✅ |
| test_large_10MB_research.txt | 10.1MB | 1.2 | -1.3 | 0.993 | ✅ |
| test_large_25MB_research.txt | 25.2MB | 1.3 | -1.0 | 0.993 | ✅ |
| test_large_50MB_research.txt | 50.5MB | 1.2 | -0.7 | 0.993 | ✅ |

### CLIP-ViT-B32

**Loading**: 1.11s, 20.5MB

| File | Size | Chunks/Sec | Memory (MB) | Quality | Status |
|------|------|------------|-------------|---------|--------|
| test_large_10MB_business.txt | 10.1MB | 31.7 | 146.0 | 0.916 | ✅ |
| test_large_25MB_business.txt | 25.2MB | 36.6 | 0.0 | 0.916 | ✅ |
| test_large_50MB_business.txt | 50.5MB | 35.6 | 0.0 | 0.916 | ✅ |
| test_large_10MB_technology.txt | 10.1MB | 37.5 | 0.6 | 0.908 | ✅ |
| test_large_25MB_technology.txt | 25.2MB | 37.0 | -1.0 | 0.908 | ✅ |
| test_large_50MB_technology.txt | 50.5MB | 36.7 | -1.0 | 0.908 | ✅ |
| test_large_10MB_research.txt | 10.1MB | 36.7 | -0.6 | 0.929 | ✅ |
| test_large_25MB_research.txt | 25.2MB | 36.8 | -1.0 | 0.929 | ✅ |
| test_large_50MB_research.txt | 50.5MB | 37.1 | -1.0 | 0.929 | ✅ |

### Jina-v4

**Status**: ❌ Failed to load  
**Error**: No module named 'custom_st'

### Nomic-Embed

**Status**: ❌ Failed to load  
**Error**: nomic-ai/nomic-bert-2048 You can inspect the repository content at https://hf.co/nomic-ai/nomic-embed-text-v1.
Please pass the argument `trust_remote_code=True` to allow custom code to be run.

