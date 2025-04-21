#!/usr/bin/env python
"""
TeacherForge Feedback CLI
Command-line interface for the TeacherForge feedback loop system
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import config
from feedback.feedback_loop import run_feedback_pipeline
from feedback.evaluator import TeacherStudentEvaluator
from feedback.refinement import ResponseRefiner

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TeacherForge Feedback System - Self-improving RAG dataset generation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Feedback loop command
    feedback_parser = subparsers.add_parser(
        "run-feedback", help="Run the complete feedback loop on a dataset"
    )
    feedback_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset file (JSONL)"
    )
    feedback_parser.add_argument(
        "--teacher-provider", type=str, default=config.RAG_MODEL_PROVIDER,
        help=f"Teacher LLM provider (default: {config.RAG_MODEL_PROVIDER})"
    )
    feedback_parser.add_argument(
        "--teacher-model", type=str, default=config.RAG_MODEL_NAME,
        help=f"Teacher model name (default: {config.RAG_MODEL_NAME})"
    )
    feedback_parser.add_argument(
        "--student-model", type=str, default=None,
        help="Path to student model or HuggingFace model ID (optional)"
    )
    feedback_parser.add_argument(
        "--iterations", type=int, default=1,
        help="Number of feedback iterations to run (default: 1)"
    )
    feedback_parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size for processing (default: 10)"
    )
    feedback_parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of samples to evaluate (default: all)"
    )
    feedback_parser.add_argument(
        "--output-dir", type=str, default="outputs/feedback",
        help="Directory to save results (default: outputs/feedback)"
    )
    feedback_parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    
    # Evaluate-only command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate student model against teacher model"
    )
    eval_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset file (JSONL)"
    )
    eval_parser.add_argument(
        "--teacher-provider", type=str, default=config.RAG_MODEL_PROVIDER,
        help=f"Teacher LLM provider (default: {config.RAG_MODEL_PROVIDER})"
    )
    eval_parser.add_argument(
        "--teacher-model", type=str, default=config.RAG_MODEL_NAME,
        help=f"Teacher model name (default: {config.RAG_MODEL_NAME})"
    )
    eval_parser.add_argument(
        "--student-model", type=str, required=True,
        help="Path to student model or HuggingFace model ID"
    )
    eval_parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of samples to evaluate (default: all)"
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size for processing (default: 10)"
    )
    eval_parser.add_argument(
        "--output-file", type=str, default="outputs/evaluation_results.json",
        help="File to save evaluation results (default: outputs/evaluation_results.json)"
    )
    
    # Refine-only command
    refine_parser = subparsers.add_parser(
        "refine", help="Refine responses based on evaluation results"
    )
    refine_parser.add_argument(
        "--evaluation-file", type=str, required=True,
        help="Path to evaluation results file (JSON)"
    )
    refine_parser.add_argument(
        "--refiner-provider", type=str, default=config.RAG_MODEL_PROVIDER,
        help=f"Refiner LLM provider (default: {config.RAG_MODEL_PROVIDER})"
    )
    refine_parser.add_argument(
        "--refiner-model", type=str, default=config.RAG_MODEL_NAME,
        help=f"Refiner model name (default: {config.RAG_MODEL_NAME})"
    )
    refine_parser.add_argument(
        "--threshold", type=float, default=7.0,
        help="Score threshold for refinement (default: 7.0)"
    )
    refine_parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size for processing (default: 10)"
    )
    refine_parser.add_argument(
        "--output-file", type=str, default="outputs/refinement_results.json",
        help="File to save refinement results (default: outputs/refinement_results.json)"
    )
    
    return parser.parse_args()

def run_feedback_command(args):
    """Run the complete feedback loop."""
    logger.info(f"Running feedback loop on dataset: {args.dataset}")
    logger.info(f"Teacher model: {args.teacher_provider}/{args.teacher_model}")
    
    if args.student_model:
        logger.info(f"Student model: {args.student_model}")
    else:
        logger.info("No student model provided. Will use teacher for both roles.")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the feedback pipeline
    results = run_feedback_pipeline(
        dataset_path=args.dataset,
        teacher_provider=args.teacher_provider,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        output_dir=args.output_dir,
        iterations=args.iterations,
        batch_size=args.batch_size,
        samples=args.samples,
        temperature=args.temperature
    )
    
    # Print summary
    summary = results.get("summary", {})
    logger.info("-" * 50)
    logger.info("Feedback Loop Summary:")
    logger.info(f"Initial dataset: {summary.get('initial_dataset')}")
    logger.info(f"Final dataset: {summary.get('final_dataset')}")
    logger.info(f"Iterations: {summary.get('iterations')}")
    
    improvement = summary.get("improvement", {})
    if improvement:
        logger.info(f"Refined {improvement.get('refined_count', 0)} out of {improvement.get('total_examples', 0)} examples")
        logger.info(f"Average improvement: {improvement.get('average_improvement', 0):.2f} points")
    
    logger.info(f"Full results saved to: {os.path.join(args.output_dir, 'feedback_summary.json')}")
    logger.info("-" * 50)

def run_evaluate_command(args):
    """Run evaluation only."""
    logger.info(f"Evaluating student model against teacher model")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Teacher model: {args.teacher_provider}/{args.teacher_model}")
    logger.info(f"Student model: {args.student_model}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load dataset
    with open(args.dataset, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Subsample if needed
    if args.samples and args.samples < len(dataset):
        import random
        dataset = random.sample(dataset, args.samples)
        logger.info(f"Sampled {args.samples} examples from dataset")
    
    # Extract questions and contexts
    questions = []
    contexts = []
    for item in dataset:
        questions.append(item.get("question", ""))
        contexts.append(item.get("context", []))
    
    # Determine student provider based on model
    student_provider = "local"
    if args.student_model and '/' in args.student_model and not os.path.exists(args.student_model):
        student_provider = "huggingface"
    
    # Initialize evaluator
    evaluator = TeacherStudentEvaluator(
        teacher_provider=args.teacher_provider,
        teacher_model=args.teacher_model,
        student_provider=student_provider,
        student_model=args.student_model
    )
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_responses(
        questions=questions,
        contexts=contexts,
        batch_size=args.batch_size,
        output_file=args.output_file
    )
    
    # Identify improvement areas
    improvement_areas = evaluator.identify_improvement_areas(evaluation_results)
    
    improvement_file = args.output_file.replace(".json", "_improvements.json")
    with open(improvement_file, 'w', encoding='utf-8') as f:
        json.dump(improvement_areas, f, indent=2, ensure_ascii=False)
    
    # Print summary
    summary = evaluation_results.get("summary", {})
    logger.info("-" * 50)
    logger.info("Evaluation Summary:")
    
    overall = summary.get("overall_scores", {})
    if overall:
        logger.info(f"Overall Score: {overall.get('mean', 0):.2f}/10")
        logger.info(f"Total examples: {overall.get('total_examples', 0)}")
    
    logger.info(f"Full results saved to: {args.output_file}")
    logger.info(f"Improvement analysis saved to: {improvement_file}")
    logger.info("-" * 50)
    
    # Return worst performing metric for user info
    worst_metric = None
    worst_score = 10.0
    for metric, data in summary.get("per_metric", {}).items():
        mean_score = data.get("mean", 10.0)
        if mean_score < worst_score:
            worst_score = mean_score
            worst_metric = metric
    
    if worst_metric:
        logger.info(f"Lowest performing metric: {worst_metric} with score {worst_score:.2f}/10")
        logger.info(f"Consider focusing improvements on this area.")

def run_refine_command(args):
    """Run refinement only based on evaluation results."""
    logger.info(f"Refining responses based on evaluation results")
    logger.info(f"Evaluation file: {args.evaluation_file}")
    logger.info(f"Refiner model: {args.refiner_provider}/{args.refiner_model}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load evaluation results
    with open(args.evaluation_file, 'r', encoding='utf-8') as f:
        evaluation_results = json.load(f)
    
    # Initialize refiner
    refiner = ResponseRefiner(
        refiner_provider=args.refiner_provider,
        refiner_model=args.refiner_model,
        refinement_threshold=args.threshold
    )
    
    # Run refinement
    refinement_results = refiner.refine_responses(
        evaluation_results=evaluation_results,
        batch_size=args.batch_size,
        output_file=args.output_file
    )
    
    # Print summary
    summary = refinement_results.get("summary", {})
    logger.info("-" * 50)
    logger.info("Refinement Summary:")
    logger.info(f"Total examples: {summary.get('total_examples', 0)}")
    logger.info(f"Refined responses: {summary.get('refined_count', 0)}")
    logger.info(f"Average improvement: {summary.get('average_improvement', 0):.2f} points")
    logger.info(f"Full results saved to: {args.output_file}")
    logger.info("-" * 50)

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "run-feedback":
        run_feedback_command(args)
    elif args.command == "evaluate":
        run_evaluate_command(args)
    elif args.command == "refine":
        run_refine_command(args)
    else:
        logger.error("No command specified. Use --help for options.")
        sys.exit(1)

if __name__ == "__main__":
    main()
