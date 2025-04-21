"""
Document Processor Module
Handles document preprocessing, chunking, and preparation for embedding.
"""
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents for RAG, including chunking and metadata extraction."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        add_metadata_to_text: bool = True
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size of chunks in tokens/characters
            chunk_overlap: Overlap between chunks in tokens/characters
            add_metadata_to_text: Whether to add metadata to text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_metadata_to_text = add_metadata_to_text
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process a list of documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            processed_docs.extend(chunks)
            
        logger.info(f"Processed {len(documents)} documents into {len(processed_docs)} chunks")
        return processed_docs
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Split a document into chunks.
        
        Args:
            document: Document dictionary
            
        Returns:
            List of document chunks
        """
        # Extract document fields
        doc_id = document.get("id", "")
        title = document.get("title", "")
        text = document.get("text", "")
        source = document.get("source", "")
        
        if not text:
            logger.warning(f"Empty text for document {doc_id or title}")
            return []
        
        # Split text into chunks
        chunks = self._split_text(text)
        
        # Create document chunks
        doc_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            # Create formatted text with metadata if requested
            if self.add_metadata_to_text:
                formatted_text = f"Title: {title}\nSource: {source}\nChunk: {i+1}/{len(chunks)}\n\n{chunk_text}"
            else:
                formatted_text = chunk_text
            
            # Create chunk document
            chunk_doc = {
                "id": f"{doc_id}_{i}" if doc_id else f"chunk_{i}",
                "title": title,
                "text": formatted_text,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "original_doc_id": doc_id
            }
            
            doc_chunks.append(chunk_doc)
        
        logger.info(f"Split document '{title}' into {len(doc_chunks)} chunks")
        return doc_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text content
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # If text is shorter than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split text by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split further
            if len(paragraph) > self.chunk_size:
                # If there's content in current chunk, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                
                # Build new chunks from sentences
                sentence_chunk = ""
                for sentence in sentences:
                    if len(sentence_chunk) + len(sentence) <= self.chunk_size:
                        sentence_chunk += " " + sentence if sentence_chunk else sentence
                    else:
                        chunks.append(sentence_chunk.strip())
                        # Include some overlap for context
                        overlap_words = self._get_overlap_content(sentence_chunk)
                        sentence_chunk = overlap_words + " " + sentence
                
                if sentence_chunk:
                    current_chunk = sentence_chunk
                
            # If adding paragraph exceeds chunk size, start new chunk
            elif len(current_chunk) + len(paragraph) > self.chunk_size:
                chunks.append(current_chunk.strip())
                # Include some overlap for context
                overlap_words = self._get_overlap_content(current_chunk)
                current_chunk = overlap_words + "\n\n" + paragraph
            
            # Otherwise add paragraph to current chunk
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_content(self, text: str) -> str:
        """
        Get content for chunk overlap.
        
        Args:
            text: Text to extract overlap from
            
        Returns:
            Overlap content
        """
        # Simple approach: get last N characters for overlap
        if len(text) <= self.chunk_overlap:
            return text
            
        # Try to get complete sentences for overlap
        overlap_text = text[-self.chunk_overlap:]
        
        # Find first sentence boundary
        match = re.search(r'(?<=[.!?])\s+', overlap_text)
        if match:
            # Start from the first sentence boundary
            start_pos = match.end()
            return overlap_text[start_pos:]
        
        # If no sentence boundary, find word boundary
        match = re.search(r'\s+', overlap_text)
        if match:
            # Start from the first word boundary
            start_pos = match.end()
            return overlap_text[start_pos:]
            
        return overlap_text


class DocumentLoader:
    """Loads documents from various sources."""
    
    @staticmethod
    def load_from_jsonl(file_path: str) -> List[Dict]:
        """
        Load documents from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {file_path}: {e}")
            return []
    
    @staticmethod
    def load_from_json(file_path: str) -> List[Dict]:
        """
        Load documents from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of document dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If data is a list, assume it's a list of documents
            if isinstance(data, list):
                documents = data
            # If data is a dict, look for a documents field
            elif isinstance(data, dict) and "documents" in data:
                documents = data["documents"]
            else:
                logger.warning(f"Unexpected JSON format in {file_path}")
                documents = []
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {file_path}: {e}")
            return []
    
    @staticmethod
    def load_from_text_directory(
        directory_path: str,
        recursive: bool = True,
        extensions: List[str] = [".txt"]
    ) -> List[Dict]:
        """
        Load documents from text files in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search recursively
            extensions: List of file extensions to include
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        try:
            # Get all matching files
            path = Path(directory_path)
            if recursive:
                files = [f for ext in extensions for f in path.glob(f"**/*{ext}")]
            else:
                files = [f for ext in extensions for f in path.glob(f"*{ext}")]
            
            # Load each file
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Create document
                    doc = {
                        "id": str(file_path.stem),
                        "title": file_path.name,
                        "text": text,
                        "source": str(file_path.relative_to(path))
                    }
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
            
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from directory {directory_path}: {e}")
            return []


def process_and_save_documents(
    input_path: str,
    output_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    add_metadata: bool = True
) -> bool:
    """
    Process documents and save them to a file.
    
    Args:
        input_path: Path to input file or directory
        output_path: Path to output file
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        add_metadata: Whether to add metadata to text
        
    Returns:
        Whether processing was successful
    """
    # Determine input type and load documents
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Load from file based on extension
        if input_path.suffix == ".jsonl":
            documents = DocumentLoader.load_from_jsonl(str(input_path))
        elif input_path.suffix == ".json":
            documents = DocumentLoader.load_from_json(str(input_path))
        elif input_path.suffix == ".txt":
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents = [{
                "id": input_path.stem,
                "title": input_path.name,
                "text": text,
                "source": str(input_path)
            }]
        else:
            logger.error(f"Unsupported file type: {input_path.suffix}")
            return False
    elif input_path.is_dir():
        # Load from directory
        documents = DocumentLoader.load_from_text_directory(str(input_path))
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return False
    
    if not documents:
        logger.error(f"No documents loaded from {input_path}")
        return False
    
    # Process documents
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_metadata_to_text=add_metadata
    )
    
    processed_docs = processor.process_documents(documents)
    
    if not processed_docs:
        logger.error(f"No processed documents generated")
        return False
    
    # Save processed documents
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, indent=2)
    
    logger.info(f"Saved {len(processed_docs)} processed documents to {output_path}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents for RAG")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap")
    parser.add_argument("--no-metadata", action="store_true", help="Don't add metadata to text")
    
    args = parser.parse_args()
    
    success = process_and_save_documents(
        args.input,
        args.output,
        args.chunk_size,
        args.chunk_overlap,
        not args.no_metadata
    )
    
    if success:
        print(f"Successfully processed documents from {args.input} to {args.output}")
    else:
        print(f"Failed to process documents")
        sys.exit(1)
