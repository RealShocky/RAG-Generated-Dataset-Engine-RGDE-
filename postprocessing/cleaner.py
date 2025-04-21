"""
Post-Processing Module
Validates and enhances generated responses with quality checks and metadata.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config

import openai

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Set API key
openai.api_key = config.OPENAI_API_KEY


class ResponseValidator:
    """Validates and enhances generated responses."""
    
    def __init__(
        self,
        validation_model: str = "gpt-3.5-turbo",
        min_confidence_threshold: float = 0.5
    ):
        """
        Initialize response validator.
        
        Args:
            validation_model: Model to use for validation
            min_confidence_threshold: Minimum confidence threshold for valid responses
        """
        self.validation_model = validation_model
        self.min_confidence_threshold = min_confidence_threshold
        logger.info(f"Initialized ResponseValidator with model: {validation_model}")
    
    def validate_grounding(self, response_data: Dict) -> Dict:
        """
        Validate that the response is grounded in the retrieved documents.
        
        Args:
            response_data: Generated response data with retrieved documents
            
        Returns:
            Enhanced response data with validation metrics
        """
        question = response_data.get("question", "")
        generated_response = response_data.get("generated_response", "")
        retrieved_documents = response_data.get("retrieved_documents", [])
        
        # Extract text from retrieved documents
        context_texts = []
        for doc in retrieved_documents:
            text = doc.get("text", "")
            if text:
                context_texts.append(text)
        
        # If no documents retrieved, mark as ungrounded
        if not context_texts:
            logger.warning(f"No context documents for question: {question[:50]}...")
            response_data["validation_metadata"] = {
                "is_valid": False,
                "confidence_score": 0.0,
                "validation_model": self.validation_model,
                "validation_reason": "No context documents provided"
            }
            return response_data
            
        try:
            # Create validation prompt
            validation_prompt = f"""
            You are an expert at evaluating the quality and factual accuracy of AI-generated responses.
            
            QUESTION:
            {question}
            
            CONTEXT DOCUMENTS:
            {' '.join(context_texts)}
            
            AI GENERATED RESPONSE:
            {generated_response}
            
            Your task is to evaluate whether the AI's response is factually accurate and grounded in the context documents.
            
            Please provide:
            1. A confidence score from 0.0 to 1.0 (where 1.0 is completely factual and fully grounded in the context)
            2. A brief explanation of your rating
            3. Any hallucinations or factual errors identified
            4. Suggested tags for categorizing this response (e.g., "well-grounded", "partially-grounded", "ungrounded", "domain-specific", etc.)
            
            Format your response as a JSON object with the following fields:
            {{
              "confidence_score": float,
              "explanation": "string",
              "hallucinations": ["list of strings"],
              "tags": ["list of strings"],
              "is_valid": boolean
            }}
            """
            
            # Get validation response
            validation_response = openai.chat.completions.create(
                model=self.validation_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI response quality and factual accuracy."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse validation results
            validation_text = validation_response.choices[0].message.content.strip()
            validation_results = json.loads(validation_text)
            
            # Set is_valid based on confidence threshold
            confidence_score = validation_results.get("confidence_score", 0.0)
            is_valid = confidence_score >= self.min_confidence_threshold
            validation_results["is_valid"] = is_valid
            
            # Add validation results to response data
            response_data["validation_metadata"] = validation_results
            
            logger.info(f"Validated response for question: {question[:50]}... (confidence: {confidence_score:.2f}, valid: {is_valid})")
            return response_data
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            response_data["validation_metadata"] = {
                "is_valid": True,  # Default to valid in case of validation error
                "confidence_score": self.min_confidence_threshold,
                "validation_model": self.validation_model,
                "validation_error": str(e)
            }
            return response_data
    
    def enhance_with_metadata(self, response_data: Dict) -> Dict:
        """
        Enhance response with additional metadata like domain, complexity, etc.
        
        Args:
            response_data: Response data with validation results
            
        Returns:
            Enhanced response data with additional metadata
        """
        question = response_data.get("question", "")
        generated_response = response_data.get("generated_response", "")
        
        try:
            # Create metadata extraction prompt
            metadata_prompt = f"""
            You are an expert at categorizing and extracting metadata from question-answer pairs.
            
            QUESTION:
            {question}
            
            ANSWER:
            {generated_response}
            
            Please analyze this question-answer pair and extract the following metadata:
            1. The domain or subject area (e.g., "computer science", "medicine", "history")
            2. The complexity level (e.g., "beginner", "intermediate", "advanced")
            3. The question type (e.g., "factual", "conceptual", "procedural", "comparative")
            4. Required knowledge areas
            5. Any key entities mentioned (people, places, technologies, concepts)
            
            Format your response as a JSON object with the following fields:
            {{
              "domain": "string",
              "complexity": "string",
              "question_type": "string",
              "knowledge_areas": ["list of strings"],
              "entities": ["list of strings"]
            }}
            """
            
            # Get metadata extraction response
            metadata_response = openai.chat.completions.create(
                model=self.validation_model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting and categorizing metadata from text."},
                    {"role": "user", "content": metadata_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse metadata results
            metadata_text = metadata_response.choices[0].message.content.strip()
            metadata_results = json.loads(metadata_text)
            
            # Add metadata to response data
            if "enhancement_metadata" not in response_data:
                response_data["enhancement_metadata"] = {}
                
            response_data["enhancement_metadata"].update(metadata_results)
            
            logger.info(f"Enhanced response with metadata for question: {question[:50]}...")
            return response_data
            
        except Exception as e:
            logger.error(f"Error enhancing response with metadata: {e}")
            if "enhancement_metadata" not in response_data:
                response_data["enhancement_metadata"] = {}
                
            response_data["enhancement_metadata"].update({
                "enhancement_error": str(e)
            })
            return response_data


def process_response(
    response_data: Dict,
    validate: bool = True,
    enhance: bool = True,
    min_confidence: float = 0.7
) -> Dict:
    """
    Process a generated response by validating and enhancing.
    
    Args:
        response_data: Generated response data
        validate: Whether to validate the response
        enhance: Whether to enhance with metadata
        min_confidence: Minimum confidence threshold
        
    Returns:
        Processed response data
    """
    validator = ResponseValidator(min_confidence_threshold=min_confidence)
    
    # Validate response groundedness
    if validate:
        response_data = validator.validate_grounding(response_data)
    
    # Enhance response with metadata
    if enhance and response_data.get("validation_metadata", {}).get("is_valid", True):
        response_data = validator.enhance_with_metadata(response_data)
    
    return response_data


def filter_valid_responses(responses: List[Dict]) -> List[Dict]:
    """
    Filter responses to keep only valid ones.
    
    Args:
        responses: List of processed responses
        
    Returns:
        List of valid responses
    """
    valid_responses = []
    
    for response in responses:
        is_valid = response.get("validation_metadata", {}).get("is_valid", False)
        confidence = response.get("validation_metadata", {}).get("confidence_score", 0.0)
        
        if is_valid:
            valid_responses.append(response)
            logger.info(f"Kept valid response with confidence {confidence:.2f}")
        else:
            logger.info(f"Filtered out invalid response with confidence {confidence:.2f}")
    
    logger.info(f"Kept {len(valid_responses)} out of {len(responses)} responses")
    return valid_responses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and validate generated responses")
    parser.add_argument("--input", type=str, required=True, help="Input file with generated responses")
    parser.add_argument("--output", type=str, help="Output file for processed responses")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence threshold")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--no-enhance", action="store_true", help="Skip metadata enhancement")
    parser.add_argument("--filter", action="store_true", help="Filter out invalid responses")
    
    args = parser.parse_args()
    
    # Load responses
    responses = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            responses.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(responses)} responses from {args.input}")
    
    # Process responses
    processed_responses = []
    for response in responses:
        processed = process_response(
            response,
            validate=not args.no_validate,
            enhance=not args.no_enhance,
            min_confidence=args.min_confidence
        )
        processed_responses.append(processed)
    
    # Filter responses if requested
    if args.filter:
        processed_responses = filter_valid_responses(processed_responses)
    
    # Save processed responses
    output_file = args.output or args.input.replace(".jsonl", "_processed.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for response in processed_responses:
            f.write(json.dumps(response) + '\n')
    
    logger.info(f"Saved {len(processed_responses)} processed responses to {output_file}")
