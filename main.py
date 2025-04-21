"""
TeacherForge - RAG-Generated Dataset Engine (RGDE)
Main orchestration script for the full pipeline.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import config
from prompts.generate_prompts import generate_questions, load_questions_from_jsonl
from generation.rag_generator import generate_rag_response
from postprocessing.cleaner import process_response, filter_valid_responses
from dataset.build_dataset import build_from_responses

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def run_pipeline(
    questions_file: str,
    output_dir: str = str(config.OUTPUTS_DIR),
    num_questions: int = 0,
    domain: str = "general knowledge",
    top_k: int = config.RAG_TOP_K,
    min_confidence: float = 0.7,
    dataset_format: str = "chat",
    include_traceability: bool = True,
    filter_responses: bool = True,
    create_splits: bool = True,
    provider: str = config.RAG_MODEL_PROVIDER,
    model_name: str = config.RAG_MODEL_NAME,
    temperature: float = config.RAG_TEMPERATURE,
    model_path: str = config.LOCAL_MODEL_PATH,
    backend: str = config.LOCAL_MODEL_BACKEND
) -> Dict:
    """
    Run the full TeacherForge pipeline.
    
    Args:
        questions_file: Path to questions file (or will be created if it doesn't exist)
        output_dir: Directory for outputs
        num_questions: Number of questions to generate (if file doesn't exist)
        domain: Domain for question generation
        top_k: Number of documents to retrieve for each question
        min_confidence: Minimum confidence threshold for valid responses
        dataset_format: Format for final dataset
        include_traceability: Whether to include traceability information
        filter_responses: Whether to filter invalid responses
        create_splits: Whether to create train/val splits
        
    Returns:
        Dictionary with pipeline results and statistics
    """
    start_time = time.time()
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    results = {
        "pipeline_start_time": start_time,
        "questions_file": questions_file,
        "output_dir": output_dir,
        "domain": domain,
        "statistics": {}
    }
    
    # Step 1: Load or generate questions
    logger.info("Step 1: Loading or generating questions")
    questions_path = Path(questions_file)
    
    if questions_path.exists():
        questions = load_questions_from_jsonl(str(questions_path))
        logger.info(f"Loaded {len(questions)} questions from {questions_file}")
    else:
        if num_questions <= 0:
            num_questions = 10
            logger.warning(f"No number of questions specified, defaulting to {num_questions}")
        
        logger.info(f"Generating {num_questions} new questions for domain: {domain}")
        questions = generate_questions(num_questions, domain, str(questions_path))
    
    results["statistics"]["num_questions"] = len(questions)
    
    # Step 2: Generate RAG responses
    logger.info("Step 2: Generating RAG responses")
    responses_file = Path(output_dir) / "responses.jsonl"
    
    with open(responses_file, 'w', encoding='utf-8') as f:
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            # Generate RAG response with the specified provider
            provider_kwargs = {
                "model_name": model_name,
                "temperature": temperature
            }
            
            # Add provider-specific kwargs
            if provider == "local" and model_path:
                provider_kwargs["model_path"] = model_path
                provider_kwargs["backend"] = backend
            
            response = generate_rag_response(
                question=question,
                top_k=top_k,
                temperature=temperature,
                provider=provider,
                **provider_kwargs
            )
            
            # Save raw response
            f.write(json.dumps(response) + '\n')
            f.flush()
    
    logger.info(f"Generated and saved {len(questions)} RAG responses to {responses_file}")
    
    # Step 3: Post-process responses
    logger.info("Step 3: Post-processing responses")
    processed_file = Path(output_dir) / "responses_processed.jsonl"
    
    processed_responses = []
    with open(responses_file, 'r', encoding='utf-8') as f_in, open(processed_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            response = json.loads(line.strip())
            
            # Process response
            processed = process_response(
                response,
                validate=True,
                enhance=True,
                min_confidence=min_confidence
            )
            
            processed_responses.append(processed)
            f_out.write(json.dumps(processed) + '\n')
    
    logger.info(f"Processed {len(processed_responses)} responses and saved to {processed_file}")
    results["statistics"]["num_responses_processed"] = len(processed_responses)
    
    # Step 4: Filter responses if requested
    if filter_responses:
        logger.info("Step 4: Filtering responses")
        filtered_file = Path(output_dir) / "responses_filtered.jsonl"
        
        valid_responses = filter_valid_responses(processed_responses)
        
        with open(filtered_file, 'w', encoding='utf-8') as f:
            for response in valid_responses:
                f.write(json.dumps(response) + '\n')
        
        logger.info(f"Filtered to {len(valid_responses)} valid responses and saved to {filtered_file}")
        results["statistics"]["num_responses_valid"] = len(valid_responses)
        dataset_input_file = str(filtered_file)
    else:
        logger.info("Step 4: Skipping response filtering")
        results["statistics"]["num_responses_valid"] = len(processed_responses)
        dataset_input_file = str(processed_file)
    
    # Step 5: Build dataset
    logger.info("Step 5: Building dataset")
    dataset_file = Path(output_dir) / "dataset.jsonl"
    
    dataset_result = build_from_responses(
        dataset_input_file,
        str(dataset_file),
        dataset_format,
        include_traceability,
        True,  # always include metadata
        create_splits
    )
    
    if dataset_result is None:
        logger.warning("No dataset was created, possibly due to lack of valid responses")
        results["statistics"]["dataset_created"] = False
        results["statistics"]["num_dataset_examples"] = 0
        logger.info("No dataset created due to lack of valid responses")
    elif create_splits:
        results["statistics"]["dataset_created"] = True
        results["statistics"]["num_train_examples"] = len(dataset_result["train"])
        results["statistics"]["num_val_examples"] = len(dataset_result["validation"])
        logger.info(f"Created dataset with {len(dataset_result['train'])} training and "
                   f"{len(dataset_result['validation'])} validation examples")
    else:
        results["statistics"]["dataset_created"] = True
        results["statistics"]["num_dataset_examples"] = len(dataset_result)
        logger.info(f"Created dataset with {len(dataset_result)} examples")
    
    # Pipeline complete
    end_time = time.time()
    duration = end_time - start_time
    results["pipeline_end_time"] = end_time
    results["pipeline_duration_seconds"] = duration
    
    logger.info(f"Pipeline completed in {duration:.2f} seconds")
    
    # Save pipeline results
    results_file = Path(output_dir) / "pipeline_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TeacherForge pipeline")
    parser.add_argument("--questions", type=str, default=str(config.PROMPTS_DIR / "prompts.jsonl"),
                       help="Path to questions file (will be created if it doesn't exist)")
    parser.add_argument("--output", type=str, default=str(config.OUTPUTS_DIR),
                       help="Output directory")
    parser.add_argument("--num", type=int, default=0,
                       help="Number of questions to generate if file doesn't exist")
    parser.add_argument("--domain", type=str, default="general knowledge",
                       help="Domain for question generation")
    parser.add_argument("--top-k", type=int, default=config.RAG_TOP_K,
                       help="Number of documents to retrieve for each question")
    parser.add_argument("--provider", type=str, default=config.RAG_MODEL_PROVIDER,
                       help="LLM provider to use (openai, anthropic, huggingface, local)")
    parser.add_argument("--model", type=str, default=config.RAG_MODEL_NAME,
                       help="Model name to use with the provider")
    parser.add_argument("--temperature", type=float, default=config.RAG_TEMPERATURE,
                       help="Temperature for text generation")
    parser.add_argument("--model-path", type=str, default=config.LOCAL_MODEL_PATH,
                       help="Path to local model (only used with local provider)")
    parser.add_argument("--backend", type=str, default=config.LOCAL_MODEL_BACKEND,
                       help="Backend for local models (llamacpp, vllm)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum confidence threshold for valid responses")
    parser.add_argument("--format", type=str, choices=["chat", "instruction", "completion"], default="chat",
                       help="Format for final dataset")
    parser.add_argument("--no-traceability", action="store_true",
                       help="Exclude traceability information")
    parser.add_argument("--no-filter", action="store_true",
                       help="Do not filter invalid responses")
    parser.add_argument("--no-splits", action="store_true",
                       help="Do not create train/val splits")
    
    args = parser.parse_args()
    
    # Run the pipeline
    try:
        results = run_pipeline(
            questions_file=args.questions,
            output_dir=args.output,
            num_questions=args.num,
            domain=args.domain,
            top_k=args.top_k,
            min_confidence=args.min_confidence,
            dataset_format=args.format,
            include_traceability=not args.no_traceability,
            filter_responses=not args.no_filter,
            create_splits=not args.no_splits
        )
        
        print("\n--- Pipeline Results ---")
        print(f"Questions processed: {results['statistics']['num_questions']}")
        print(f"Valid responses: {results['statistics']['num_responses_valid']}")
        
        # Check if dataset was created successfully
        if results['statistics'].get('dataset_created', False):
            if "num_train_examples" in results["statistics"]:
                print(f"Training examples: {results['statistics']['num_train_examples']}")
                print(f"Validation examples: {results['statistics']['num_val_examples']}")
            else:
                print(f"Dataset examples: {results['statistics']['num_dataset_examples']}")
        else:
            print("No dataset created due to lack of valid responses")
            
        print(f"Pipeline duration: {results['pipeline_duration_seconds']:.2f} seconds")
        print(f"Results saved to: {os.path.join(args.output, 'pipeline_results.json')}")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        print(f"Error: {str(e)}")
        sys.exit(1)
