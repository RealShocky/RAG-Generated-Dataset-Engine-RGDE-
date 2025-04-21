"""
Dataset Builder Module
Compiles final instruction datasets from processed responses.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config

import tqdm
from datasets import Dataset

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds instruction datasets from processed responses."""
    
    def __init__(self, output_dir: str = str(config.OUTPUTS_DIR)):
        """
        Initialize the dataset builder.
        
        Args:
            output_dir: Directory to save datasets
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        logger.info(f"Initialized DatasetBuilder with output directory: {output_dir}")
    
    def load_responses(self, input_file: str) -> List[Dict]:
        """
        Load processed responses from a JSONL file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of response objects
        """
        responses = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    response = json.loads(line.strip())
                    responses.append(response)
            
            logger.info(f"Loaded {len(responses)} responses from {input_file}")
            return responses
        
        except Exception as e:
            logger.error(f"Error loading responses: {e}")
            return []
    
    def format_as_instruction(self, response: Dict, format_type: str = "chat") -> Dict:
        """
        Format a response as an instruction example.
        
        Args:
            response: Response object
            format_type: Format type ('chat' or 'instruction')
            
        Returns:
            Formatted instruction example
        """
        question = response.get("question", "")
        generated_response = response.get("generated_response", "")
        
        if format_type == "chat":
            # Chat format for models like ChatGPT, Claude, etc.
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": generated_response}
                ]
            }
        elif format_type == "instruction":
            # Instruction format for models like Alpaca, Vicuna, etc.
            return {
                "instruction": question,
                "input": "",
                "output": generated_response
            }
        elif format_type == "completion":
            # Completion format (e.g., for text-davinci-003)
            return {
                "prompt": f"Question: {question}\n\nAnswer:",
                "completion": generated_response
            }
        else:
            logger.warning(f"Unknown format type: {format_type}, using chat")
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": generated_response}
                ]
            }
    
    def add_traceability(self, example: Dict, response: Dict) -> Dict:
        """
        Add traceability information to dataset example.
        
        Args:
            example: Formatted example
            response: Original response with metadata
            
        Returns:
            Example with traceability information
        """
        # Create traceability object
        traceability = {
            "retrieved_documents": [],
            "generation_metadata": response.get("generation_metadata", {}),
            "validation_metadata": response.get("validation_metadata", {}),
            "enhancement_metadata": response.get("enhancement_metadata", {})
        }
        
        # Extract minimal information from retrieved documents
        for doc in response.get("retrieved_documents", []):
            traceability["retrieved_documents"].append({
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "retrieval_score": doc.get("retrieval_score", 0.0),
                "id": doc.get("id", "")
            })
        
        # Add traceability to example
        example["traceability"] = traceability
        
        return example
    
    def build_dataset(
        self,
        responses: List[Dict],
        format_type: str = "chat",
        include_traceability: bool = True,
        include_metadata: bool = True,
        output_file: Optional[str] = None
    ) -> Dataset:
        """
        Build a dataset from responses.
        
        Args:
            responses: List of response objects
            format_type: Format type ('chat', 'instruction', or 'completion')
            include_traceability: Whether to include traceability information
            include_metadata: Whether to include metadata
            output_file: Path to output file
            
        Returns:
            Hugging Face Dataset object
        """
        examples = []
        
        logger.info(f"Building dataset with {len(responses)} examples in {format_type} format")
        
        for response in tqdm.tqdm(responses, desc="Formatting examples"):
            # Format as instruction
            example = self.format_as_instruction(response, format_type)
            
            # Add traceability information if requested
            if include_traceability:
                example = self.add_traceability(example, response)
            
            # Add metadata if requested
            if include_metadata:
                example["metadata"] = {
                    "domain": response.get("enhancement_metadata", {}).get("domain", ""),
                    "complexity": response.get("enhancement_metadata", {}).get("complexity", ""),
                    "question_type": response.get("enhancement_metadata", {}).get("question_type", ""),
                    "confidence_score": response.get("validation_metadata", {}).get("confidence_score", 0.0),
                    "is_valid": response.get("validation_metadata", {}).get("is_valid", True)
                }
            
            examples.append(example)
        
        # Create Dataset object
        dataset = Dataset.from_list(examples)
        
        # Save dataset to file if specified
        if output_file:
            self.save_dataset(dataset, output_file)
        
        return dataset
    
    def save_dataset(self, dataset: Dataset, output_file: str, format: str = "jsonl") -> None:
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset to save
            output_file: Path to output file
            format: Output format ('jsonl', 'parquet', or 'arrow')
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            if format == "jsonl":
                dataset.to_json(output_file, orient="records", lines=True)
            elif format == "parquet":
                dataset.to_parquet(output_file)
            elif format == "arrow":
                dataset.save_to_disk(output_file)
            else:
                logger.warning(f"Unknown format: {format}, saving as JSONL")
                dataset.to_json(output_file, orient="records", lines=True)
            
            logger.info(f"Saved dataset with {len(dataset)} examples to {output_file}")
        
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.9, seed: int = 42) -> Dict[str, Dataset]:
        """
        Split dataset into train and validation sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio of training examples
            seed: Random seed
            
        Returns:
            Dictionary with train and validation splits
        """
        splits = dataset.train_test_split(train_size=train_ratio, seed=seed)
        train_dataset = splits["train"]
        val_dataset = splits["test"]
        
        logger.info(f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation examples")
        
        return {
            "train": train_dataset,
            "validation": val_dataset
        }


def build_from_responses(
    input_file: str,
    output_file: Optional[str] = None,
    format_type: str = "chat",
    include_traceability: bool = True,
    include_metadata: bool = True,
    create_train_val_split: bool = True,
    train_ratio: float = 0.9
) -> Union[Dataset, Dict[str, Dataset], None]:
    """
    Build a dataset from a file of processed responses.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        format_type: Format type
        include_traceability: Whether to include traceability
        include_metadata: Whether to include metadata
        create_train_val_split: Whether to create train/val split
        train_ratio: Ratio of training examples
        
    Returns:
        Dataset or dictionary with splits, or None if no valid responses
    """
    # Initialize builder
    builder = DatasetBuilder()
    
    # Load responses
    responses = builder.load_responses(input_file)
    
    if not responses:
        logger.error(f"No responses loaded from {input_file}")
        return None
    
    # Build dataset
    dataset = builder.build_dataset(
        responses,
        format_type,
        include_traceability,
        include_metadata,
        output_file if not create_train_val_split else None
    )
    
    # Split if requested
    if create_train_val_split:
        # Check if dataset has enough examples to split
        if len(dataset) <= 1:
            logger.warning(f"Dataset has only {len(dataset)} examples, not enough to split. Returning full dataset.")
            if output_file:
                builder.save_dataset(dataset, output_file)
            return {"train": dataset, "validation": dataset}  # Return same dataset for both splits
            
        splits = builder.split_dataset(dataset, train_ratio)
        
        # Save splits if output file specified
        if output_file:
            output_base = output_file.rsplit(".", 1)[0]
            builder.save_dataset(splits["train"], f"{output_base}_train.jsonl")
            builder.save_dataset(splits["validation"], f"{output_base}_val.jsonl")
        
        return splits
    else:
        return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build instruction dataset from processed responses")
    parser.add_argument("--input", type=str, required=True, help="Input file with processed responses")
    parser.add_argument("--output", type=str, default=str(config.OUTPUTS_DIR / "dataset.jsonl"),
                        help="Output file for the dataset")
    parser.add_argument("--format", type=str, choices=["chat", "instruction", "completion"], default="chat",
                        help="Format type for the dataset")
    parser.add_argument("--no-traceability", action="store_true", help="Exclude traceability information")
    parser.add_argument("--no-metadata", action="store_true", help="Exclude metadata")
    parser.add_argument("--no-split", action="store_true", help="Do not create train/val split")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Ratio of training examples")
    
    args = parser.parse_args()
    
    # Build dataset
    result = build_from_responses(
        args.input,
        args.output,
        args.format,
        not args.no_traceability,
        not args.no_metadata,
        not args.no_split,
        args.train_ratio
    )
    
    if result:
        if isinstance(result, dict):
            print(f"Created dataset with {len(result['train'])} training and {len(result['validation'])} validation examples")
        else:
            print(f"Created dataset with {len(result)} examples")
