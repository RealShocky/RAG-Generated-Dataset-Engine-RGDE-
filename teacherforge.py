#!/usr/bin/env python
"""
TeacherForge CLI
Command-line interface for interacting with the TeacherForge system.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import config
from prompts.generate_prompts import generate_questions, load_questions_from_jsonl
from generation.rag_generator import generate_rag_response
from postprocessing.cleaner import process_response
from dataset.build_dataset import build_from_responses
from main import run_pipeline

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="TeacherForge - RAG-Generated Dataset Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")
    
    # Generate questions command
    gen_questions_parser = subparsers.add_parser("generate-questions", help="Generate questions")
    gen_questions_parser.add_argument("--num", type=int, default=10, help="Number of questions to generate")
    gen_questions_parser.add_argument("--domain", type=str, default="general knowledge", help="Domain for questions")
    gen_questions_parser.add_argument("--output", type=str, default=str(config.PROMPTS_DIR / "prompts.jsonl"),
                                    help="Output file path")
    
    # Generate RAG response command
    rag_parser = subparsers.add_parser("generate-response", help="Generate a RAG response for a question")
    rag_parser.add_argument("--question", type=str, required=True, help="Input question")
    rag_parser.add_argument("--output", type=str, help="Output file for the response")
    rag_parser.add_argument("--top-k", type=int, default=config.RAG_TOP_K, help="Number of documents to retrieve")
    rag_parser.add_argument("--provider", type=str, default=config.RAG_MODEL_PROVIDER, 
                          help="LLM provider to use (openai, anthropic, huggingface, local)")
    rag_parser.add_argument("--model", type=str, default=config.RAG_MODEL_NAME, 
                          help="Model name to use with the provider")
    rag_parser.add_argument("--temperature", type=float, default=config.RAG_TEMPERATURE, 
                          help="Temperature for text generation")
    rag_parser.add_argument("--model-path", type=str, default=config.LOCAL_MODEL_PATH, 
                          help="Path to local model (only used with local provider)")
    rag_parser.add_argument("--backend", type=str, default=config.LOCAL_MODEL_BACKEND, 
                          help="Backend for local models (llamacpp, vllm)")
    
    # Process response command
    process_parser = subparsers.add_parser("process-response", help="Process and validate a response")
    process_parser.add_argument("--input", type=str, required=True, help="Input file with response")
    process_parser.add_argument("--output", type=str, help="Output file for processed response")
    process_parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence threshold")
    
    # Build dataset command
    dataset_parser = subparsers.add_parser("build-dataset", help="Build a dataset from responses")
    dataset_parser.add_argument("--input", type=str, required=True, help="Input file with processed responses")
    dataset_parser.add_argument("--output", type=str, default=str(config.OUTPUTS_DIR / "dataset.jsonl"),
                              help="Output file for the dataset")
    dataset_parser.add_argument("--format", type=str, choices=["chat", "instruction", "completion"], default="chat",
                              help="Format type for the dataset")
    dataset_parser.add_argument("--no-traceability", action="store_true", help="Exclude traceability information")
    dataset_parser.add_argument("--no-split", action="store_true", help="Do not create train/val split")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run the full TeacherForge pipeline")
    pipeline_parser.add_argument("--questions", type=str, default=str(config.PROMPTS_DIR / "prompts.jsonl"),
                               help="Path to questions file (will be created if it doesn't exist)")
    pipeline_parser.add_argument("--output", type=str, default=str(config.OUTPUTS_DIR),
                               help="Output directory")
    pipeline_parser.add_argument("--num", type=int, default=0,
                               help="Number of questions to generate if file doesn't exist")
    pipeline_parser.add_argument("--domain", type=str, default="general knowledge",
                               help="Domain for question generation")
    pipeline_parser.add_argument("--top-k", type=int, default=config.RAG_TOP_K,
                               help="Number of documents to retrieve for each question")
    pipeline_parser.add_argument("--min-confidence", type=float, default=0.7,
                               help="Minimum confidence threshold for valid responses")
    pipeline_parser.add_argument("--format", type=str, choices=["chat", "instruction", "completion"], default="chat",
                               help="Format for final dataset")
    pipeline_parser.add_argument("--no-traceability", action="store_true",
                               help="Exclude traceability information")
    pipeline_parser.add_argument("--no-filter", action="store_true",
                               help="Do not filter invalid responses")
    pipeline_parser.add_argument("--no-splits", action="store_true",
                               help="Do not create train/val splits")
    pipeline_parser.add_argument("--provider", type=str, default=config.RAG_MODEL_PROVIDER, 
                               help="LLM provider to use (openai, anthropic, huggingface, local)")
    pipeline_parser.add_argument("--model", type=str, default=config.RAG_MODEL_NAME, 
                               help="Model name to use with the provider")
    pipeline_parser.add_argument("--temperature", type=float, default=config.RAG_TEMPERATURE, 
                               help="Temperature for text generation")
    pipeline_parser.add_argument("--model-path", type=str, default=config.LOCAL_MODEL_PATH, 
                               help="Path to local model (only used with local provider)")
    pipeline_parser.add_argument("--backend", type=str, default=config.LOCAL_MODEL_BACKEND, 
                               help="Backend for local models (llamacpp, vllm)")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == "generate-questions":
            questions = generate_questions(args.num, args.domain, args.output)
            print(f"Generated {len(questions)} questions and saved to {args.output}")
        
        elif args.command == "generate-response":
            # Get additional provider arguments
            provider_kwargs = {
                "model_name": args.model,
                "temperature": args.temperature
            }
            
            # Add provider-specific arguments
            if args.provider == "local" and args.model_path:
                provider_kwargs["model_path"] = args.model_path
                provider_kwargs["backend"] = args.backend
            
            response = generate_rag_response(
                question=args.question, 
                top_k=args.top_k,
                temperature=args.temperature,
                provider=args.provider,
                **provider_kwargs
            )
            
            if args.output:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(response, f, indent=2)
                print(f"Response saved to {args.output}")
            else:
                print("\n--- Question ---")
                print(response["question"])
                
                print("\n--- Retrieved Documents ---")
                for i, doc in enumerate(response["retrieved_documents"]):
                    print(f"\n[{i+1}] {doc.get('title', f'Document {i+1}')}")
                    print(f"Score: {doc.get('retrieval_score', 'N/A')}")
                    print(f"Text: {doc.get('text', 'N/A')[:200]}...")
                    
                print("\n--- Generated Response ---")
                print(response["generated_response"])
        
        elif args.command == "process-response":
            import json
            with open(args.input, 'r', encoding='utf-8') as f:
                response = json.load(f)
            
            processed = process_response(
                response,
                validate=True,
                enhance=True,
                min_confidence=args.min_confidence
            )
            
            output_file = args.output or args.input.replace(".json", "_processed.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed, f, indent=2)
            
            print(f"Processed response saved to {output_file}")
            print(f"Confidence score: {processed.get('validation_metadata', {}).get('confidence_score', 'N/A')}")
            print(f"Valid: {processed.get('validation_metadata', {}).get('is_valid', 'N/A')}")
        
        elif args.command == "build-dataset":
            result = build_from_responses(
                args.input,
                args.output,
                args.format,
                not args.no_traceability,
                True,  # always include metadata
                not args.no_split
            )
            
            if isinstance(result, dict):
                print(f"Created dataset with {len(result['train'])} training and {len(result['validation'])} validation examples")
                print(f"Saved to {args.output}_train.jsonl and {args.output}_val.jsonl")
            else:
                print(f"Created dataset with {len(result)} examples")
                print(f"Saved to {args.output}")
        
        elif args.command == "run-pipeline":
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
            
            if "num_train_examples" in results["statistics"]:
                print(f"Training examples: {results['statistics']['num_train_examples']}")
                print(f"Validation examples: {results['statistics']['num_val_examples']}")
            else:
                print(f"Dataset examples: {results['statistics']['num_dataset_examples']}")
                
            print(f"Pipeline duration: {results['pipeline_duration_seconds']:.2f} seconds")
            print(f"Results saved to: {os.path.join(args.output, 'pipeline_results.json')}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
