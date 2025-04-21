"""
Create a FAISS index from sample documents for the TeacherForge system.
With support for smart document chunking.
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Import document processor
from retrieval.document_processor import DocumentProcessor, DocumentLoader, process_and_save_documents

# Configuration
DOCUMENTS_PATH = 'sample_documents.json'
OUTPUT_INDEX_PATH = 'faiss.index'
OUTPUT_DIR = Path('data')
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load documents
print(f"Loading documents from {DOCUMENTS_PATH}...")
with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} documents")

# Process documents with chunking (if requested)
use_chunking = os.environ.get('USE_CHUNKING', 'false').lower() == 'true'
if use_chunking:
    print("Processing documents with smart chunking...")
    processor = DocumentProcessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_metadata_to_text=True
    )
    documents = processor.process_documents(documents)
    print(f"Created {len(documents)} document chunks")

# Initialize embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Create embeddings
print("Creating document embeddings...")
texts = [doc['text'] for doc in documents]
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

# Build FAISS index
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized vectors)
index.add(np.array(embeddings).astype('float32'))

# Save index and documents
index_path = OUTPUT_DIR / OUTPUT_INDEX_PATH
faiss.write_index(index, str(index_path))
print(f"Saved FAISS index to {index_path}")

# Save documents to data directory
docs_path = OUTPUT_DIR / DOCUMENTS_PATH
with open(docs_path, 'w', encoding='utf-8') as f:
    json.dump(documents, f, indent=2)
print(f"Saved documents to {docs_path}")

# Update .env file
if os.path.exists('.env'):
    print("Updating .env file with FAISS paths...")
    with open('.env', 'r', encoding='utf-8') as f:
        env_content = f.read()
    
    # Update FAISS paths
    env_content = env_content.replace('FAISS_INDEX_PATH=path/to/your/faiss.index', f'FAISS_INDEX_PATH={str(index_path)}')
    env_content = env_content.replace('FAISS_DOCUMENTS_PATH=path/to/your/documents.json', f'FAISS_DOCUMENTS_PATH={str(docs_path)}')
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    print("Updated .env file with correct FAISS paths")

print("FAISS index creation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FAISS index from documents")
    parser.add_argument("--documents", type=str, default=DOCUMENTS_PATH, help="Path to documents JSON file")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL, help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Document chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Document chunk overlap")
    parser.add_argument("--use-chunking", action="store_true", help="Use document chunking")
    
    args = parser.parse_args()
    
    # Set environment variables for script
    os.environ['USE_CHUNKING'] = 'true' if args.use_chunking else 'false'
    
    # Override globals with command line arguments
    DOCUMENTS_PATH = args.documents
    OUTPUT_DIR = Path(args.output_dir)
    EMBEDDING_MODEL = args.model
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
