"""
Document Ingestion Tool
A command-line tool for ingesting various document formats into TeacherForge.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Import document processing modules
from retrieval.document_loaders import load_document, load_documents_from_directory
from retrieval.document_processor import DocumentProcessor, process_and_save_documents
import config

def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into TeacherForge",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input file or directory to process")
    parser.add_argument("--output", type=str, default="data/processed_documents.json",
                        help="Output file for processed documents")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively process directories")
    parser.add_argument("--create-index", action="store_true",
                        help="Create FAISS index from processed documents")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Document chunk size (in characters)")
    parser.add_argument("--chunk-overlap", type=int, default=100,
                        help="Overlap between chunks (in characters)")
    parser.add_argument("--extensions", type=str, default=".txt,.pdf,.docx,.html,.csv",
                        help="Comma-separated list of file extensions to process")
    
    args = parser.parse_args()
    
    # Process input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Set up extensions list
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    
    # Load documents
    if input_path.is_file():
        logger.info(f"Loading single file: {input_path}")
        document = load_document(str(input_path))
        if document:
            documents = [document]
        else:
            logger.error(f"Failed to load document: {input_path}")
            return 1
    else:
        logger.info(f"Loading documents from directory: {input_path}")
        documents = load_documents_from_directory(
            str(input_path),
            recursive=args.recursive,
            extensions=extensions
        )
    
    if not documents:
        logger.error("No documents were loaded")
        return 1
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Process documents with chunking
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_metadata_to_text=True
    )
    
    processed_docs = processor.process_documents(documents)
    logger.info(f"Created {len(processed_docs)} document chunks")
    
    # Save processed documents
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, indent=2)
    
    logger.info(f"Saved processed documents to {output_path}")
    
    # Create FAISS index if requested
    if args.create_index:
        try:
            logger.info("Creating FAISS index...")
            # Set environment variable for create_faiss_index.py
            os.environ['USE_CHUNKING'] = 'false'  # We've already chunked the documents
            
            # Update path in environment
            os.environ['DOCUMENTS_PATH'] = str(output_path)
            
            # Import and run index creation
            from create_faiss_index import DOCUMENTS_PATH, OUTPUT_INDEX_PATH, OUTPUT_DIR, EMBEDDING_MODEL
            
            # Override with our documents path
            DOCUMENTS_PATH = str(output_path)
            
            # Execute the module code
            import create_faiss_index
            
            logger.info("FAISS index creation complete")
            
            # Update .env file
            env_path = Path('.env')
            if env_path.exists():
                logger.info("Updating .env file with FAISS paths...")
                with open(env_path, 'r', encoding='utf-8') as f:
                    env_content = f.read()
                
                # Update document paths
                index_path = OUTPUT_DIR / OUTPUT_INDEX_PATH
                env_content = env_content.replace('FAISS_INDEX_PATH=', f'FAISS_INDEX_PATH={str(index_path)}')
                env_content = env_content.replace('FAISS_DOCUMENTS_PATH=', f'FAISS_DOCUMENTS_PATH={str(output_path)}')
                
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write(env_content)
                
                logger.info("Updated .env file with correct FAISS paths")
        
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return 1
    
    logger.info("Document ingestion completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
