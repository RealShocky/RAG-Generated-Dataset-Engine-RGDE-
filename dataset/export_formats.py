"""
Dataset Export Formats Module
Provides utilities to convert datasets to various export formats.
"""
import csv
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

def convert_to_jsonl(
    dataset: List[Dict], 
    output_path: str,
    format_type: str = "default"
) -> Tuple[bool, str]:
    """
    Export dataset to JSONL format.
    
    Args:
        dataset: List of dataset examples
        output_path: Path to save the exported dataset
        format_type: Format type (default, openai, chat, instruction)
        
    Returns:
        Tuple of (success status, message)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                if format_type == "openai":
                    # OpenAI fine-tuning format
                    converted_item = _convert_to_openai_format(item)
                elif format_type == "chat":
                    # Chat format (messages array)
                    converted_item = _convert_to_chat_format(item)
                elif format_type == "instruction":
                    # Instruction format (instruction, input, output)
                    converted_item = _convert_to_instruction_format(item)
                else:
                    # Default format - use as is
                    converted_item = item
                
                f.write(json.dumps(converted_item) + '\n')
        
        logger.info(f"Exported {len(dataset)} examples to {output_path} in {format_type} format")
        return True, f"Successfully exported {len(dataset)} examples to {output_path}"
    
    except Exception as e:
        logger.error(f"Error exporting dataset to JSONL: {e}")
        return False, f"Error exporting dataset: {str(e)}"

def convert_to_csv(
    dataset: List[Dict], 
    output_path: str
) -> Tuple[bool, str]:
    """
    Export dataset to CSV format.
    
    Args:
        dataset: List of dataset examples
        output_path: Path to save the exported dataset
        
    Returns:
        Tuple of (success status, message)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to simple key-value pairs for CSV
        flattened_data = []
        for item in dataset:
            flattened_item = {
                "question": item.get("question", ""),
                "response": item.get("generated_response", item.get("response", "")),
                "score": _extract_score(item),
                "source": _extract_source(item)
            }
            flattened_data.append(flattened_item)
        
        # Write to CSV
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            if not flattened_data:
                return False, "No data to export"
            
            writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
            writer.writeheader()
            writer.writerows(flattened_data)
        
        logger.info(f"Exported {len(dataset)} examples to {output_path} in CSV format")
        return True, f"Successfully exported {len(dataset)} examples to {output_path}"
    
    except Exception as e:
        logger.error(f"Error exporting dataset to CSV: {e}")
        return False, f"Error exporting dataset: {str(e)}"

def convert_to_parquet(
    dataset: List[Dict], 
    output_path: str
) -> Tuple[bool, str]:
    """
    Export dataset to Parquet format.
    
    Args:
        dataset: List of dataset examples
        output_path: Path to save the exported dataset
        
    Returns:
        Tuple of (success status, message)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Simplify dataset for storage efficiency
        if 'retrieved_documents' in df.columns:
            # Store only essential document info
            df['retrieved_documents'] = df['retrieved_documents'].apply(
                lambda docs: [{"title": d.get("title", ""), "id": d.get("id", "")} for d in (docs or [])]
            )
        
        # Write to Parquet
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Exported {len(dataset)} examples to {output_path} in Parquet format")
        return True, f"Successfully exported {len(dataset)} examples to {output_path}"
    
    except Exception as e:
        logger.error(f"Error exporting dataset to Parquet: {e}")
        return False, f"Error exporting dataset: {str(e)}"

def convert_to_hf_dataset(
    dataset: List[Dict], 
    output_path: str,
    create_splits: bool = True,
    train_ratio: float = 0.8
) -> Tuple[bool, str]:
    """
    Export dataset to HuggingFace dataset format (dataset saved as directory).
    
    Args:
        dataset: List of dataset examples
        output_path: Path to save the exported dataset
        create_splits: Whether to create train/test splits
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (success status, message)
    """
    try:
        # Import datasets library here to avoid requiring it for the entire module
        from datasets import Dataset, DatasetDict
        
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        if create_splits:
            # Create train/test split
            import numpy as np
            np.random.seed(42)
            
            indices = np.random.permutation(len(dataset))
            train_size = int(len(dataset) * train_ratio)
            
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            train_data = [dataset[i] for i in train_indices]
            test_data = [dataset[i] for i in test_indices]
            
            # Create HF datasets
            train_dataset = Dataset.from_list(train_data)
            test_dataset = Dataset.from_list(test_data)
            
            # Combine into DatasetDict
            dataset_dict = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
            
            # Save dataset
            dataset_dict.save_to_disk(output_path)
            
            logger.info(f"Exported dataset with {len(train_data)} train and {len(test_data)} test examples to {output_path}")
            return True, f"Successfully exported dataset with {len(train_data)} train and {len(test_data)} test examples"
        
        else:
            # Create single dataset
            hf_dataset = Dataset.from_list(dataset)
            hf_dataset.save_to_disk(output_path)
            
            logger.info(f"Exported {len(dataset)} examples to {output_path} as HuggingFace dataset")
            return True, f"Successfully exported {len(dataset)} examples to HuggingFace dataset"
    
    except Exception as e:
        logger.error(f"Error exporting dataset to HuggingFace format: {e}")
        return False, f"Error exporting dataset: {str(e)}"

# Helper functions for format conversion

def _convert_to_openai_format(item: Dict) -> Dict:
    """Convert to OpenAI fine-tuning format."""
    question = item.get("question", "")
    response = item.get("generated_response", item.get("response", ""))
    
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }

def _convert_to_chat_format(item: Dict) -> Dict:
    """Convert to chat format with messages array."""
    question = item.get("question", "")
    response = item.get("generated_response", item.get("response", ""))
    
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ],
        "metadata": item.get("metadata", {}),
        "traceability": item.get("traceability", {})
    }

def _convert_to_instruction_format(item: Dict) -> Dict:
    """Convert to instruction-input-output format."""
    question = item.get("question", "")
    response = item.get("generated_response", item.get("response", ""))
    
    return {
        "instruction": question,
        "input": "",
        "output": response,
        "metadata": item.get("metadata", {}),
        "traceability": item.get("traceability", {})
    }

def _extract_score(item: Dict) -> float:
    """Extract confidence score from item if available."""
    try:
        if "traceability" in item and "validation_metadata" in item["traceability"]:
            return item["traceability"]["validation_metadata"].get("confidence_score", 0.0)
        return 0.0
    except:
        return 0.0

def _extract_source(item: Dict) -> str:
    """Extract source information from retrieved documents if available."""
    try:
        if "retrieved_documents" in item and item["retrieved_documents"]:
            sources = []
            for doc in item["retrieved_documents"][:3]:  # Take top 3 sources
                src = doc.get("source", "")
                if src and src not in sources:
                    sources.append(src)
            return "; ".join(sources)
        return ""
    except:
        return ""
