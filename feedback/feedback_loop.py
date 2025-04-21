"""
Feedback Loop Module
Implements a complete feedback loop for self-improving RAG dataset generation
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from tqdm import tqdm

# Add project root to path to import config
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config
from generation.rag_generator import get_generator, ProviderGenerator
from feedback.evaluator import TeacherStudentEvaluator
from feedback.refinement import ResponseRefiner

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class FeedbackLoop:
    """
    Implements a complete feedback loop for self-improving RAG-generated datasets.
    
    The feedback loop process:
    1. Generate responses with both teacher and student models
    2. Evaluate and compare the responses
    3. Refine problematic responses
    4. Update the dataset with refined responses
    5. (Optional) Retrain the student model with the improved dataset
    """
    
    def __init__(
        self,
        teacher_provider: str = config.RAG_MODEL_PROVIDER,
        teacher_model: str = config.RAG_MODEL_NAME,
        student_provider: str = "local",
        student_model: str = None,
        evaluator_provider: str = None,
        evaluator_model: str = None,
        refiner_provider: str = None,
        refiner_model: str = None,
        refinement_threshold: float = 7.0,
        metrics: List[str] = None,
        output_dir: str = "outputs/feedback",
        **kwargs
    ):
        """
        Initialize the feedback loop.
        
        Args:
            teacher_provider: Provider for teacher model
            teacher_model: Teacher model name
            student_provider: Provider for student model
            student_model: Student model name or path
            evaluator_provider: Provider for evaluation model
            evaluator_model: Evaluation model name
            refiner_provider: Provider for refinement model
            refiner_model: Refinement model name
            refinement_threshold: Score threshold for refinement
            metrics: Evaluation metrics
            output_dir: Directory to save results
            **kwargs: Additional provider-specific parameters
        """
        self.teacher_provider = teacher_provider
        self.teacher_model = teacher_model
        self.student_provider = student_provider
        self.student_model = student_model
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize teacher generator
        self.teacher_generator = get_generator(
            model_provider=teacher_provider,
            model_name=teacher_model,
            **kwargs
        )
        logger.info(f"Initialized teacher model: {teacher_model} with provider {teacher_provider}")
        
        # Initialize student generator if model path is provided
        if student_model:
            student_kwargs = kwargs.copy()
            if student_provider == "local":
                student_kwargs["model_path"] = student_model
                
            self.student_generator = get_generator(
                model_provider=student_provider,
                model_name=student_model,
                **student_kwargs
            )
            logger.info(f"Initialized student model: {student_model} with provider {student_provider}")
        else:
            self.student_generator = None
            logger.warning("No student model provided. Student evaluation will be skipped.")
        
        # Initialize evaluator
        evaluator_kwargs = kwargs.copy()
        self.evaluator = TeacherStudentEvaluator(
            teacher_provider=teacher_provider,
            teacher_model=teacher_model,
            student_provider=student_provider,
            student_model=student_model,
            evaluator_provider=evaluator_provider or teacher_provider,
            evaluator_model=evaluator_model or teacher_model,
            metrics=metrics,
            **evaluator_kwargs
        )
        
        # Initialize refiner
        refiner_kwargs = kwargs.copy()
        self.refiner = ResponseRefiner(
            refiner_provider=refiner_provider or teacher_provider,
            refiner_model=refiner_model or teacher_model,
            refinement_threshold=refinement_threshold,
            **refiner_kwargs
        )
        
    def run_feedback_loop(
        self,
        dataset_path: str,
        iterations: int = 1,
        batch_size: int = 10,
        evaluation_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete feedback loop on a dataset.
        
        Args:
            dataset_path: Path to the dataset file (JSONL)
            iterations: Number of feedback iterations to run
            batch_size: Batch size for processing
            evaluation_samples: Number of samples to evaluate (None = all)
            
        Returns:
            Dictionary with feedback loop results
        """
        results = {
            "iterations": [],
            "summary": {
                "initial_dataset": dataset_path,
                "iterations": iterations,
                "final_dataset": "",
                "improvement": {}
            }
        }
        
        current_dataset = dataset_path
        
        # Load initial dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            initial_data = [json.loads(line) for line in f]
        
        # Subsample for evaluation if specified
        if evaluation_samples and evaluation_samples < len(initial_data):
            import random
            evaluation_data = random.sample(initial_data, evaluation_samples)
            logger.info(f"Subsampled {evaluation_samples} examples from dataset of size {len(initial_data)}")
        else:
            evaluation_data = initial_data
        
        # Extract questions and contexts from the dataset
        questions = []
        contexts = []
        for item in evaluation_data:
            questions.append(item.get("question", ""))
            contexts.append(item.get("context", []))
        
        # Run iterations
        for iteration in range(iterations):
            logger.info(f"Starting feedback loop iteration {iteration+1}/{iterations}")
            
            # Create iteration output directory
            iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration+1}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            iteration_results = {
                "iteration": iteration + 1,
                "dataset": current_dataset,
                "evaluation_results": "",
                "refinement_results": "",
                "improved_dataset": ""
            }
            
            # Step 1: Evaluate teacher and student responses
            logger.info("Step 1: Evaluating responses...")
            evaluation_file = os.path.join(iteration_dir, "evaluation_results.json")
            
            evaluation_results = self.evaluator.evaluate_responses(
                questions=questions,
                contexts=contexts,
                batch_size=batch_size,
                output_file=evaluation_file
            )
            
            iteration_results["evaluation_results"] = evaluation_file
            
            # Step 2: Identify improvement areas
            logger.info("Step 2: Identifying improvement areas...")
            improvement_areas = self.evaluator.identify_improvement_areas(evaluation_results)
            
            improvement_file = os.path.join(iteration_dir, "improvement_areas.json")
            with open(improvement_file, 'w', encoding='utf-8') as f:
                json.dump(improvement_areas, f, indent=2, ensure_ascii=False)
                
            iteration_results["improvement_areas"] = improvement_file
            
            # Step 3: Refine responses
            logger.info("Step 3: Refining responses...")
            refinement_file = os.path.join(iteration_dir, "refinement_results.json")
            
            refinement_results = self.refiner.refine_responses(
                evaluation_results=evaluation_results,
                batch_size=batch_size,
                output_file=refinement_file
            )
            
            iteration_results["refinement_results"] = refinement_file
            
            # Step 4: Update dataset with refined responses
            logger.info("Step 4: Updating dataset with refined responses...")
            improved_dataset = os.path.join(iteration_dir, "improved_dataset.jsonl")
            
            self._update_dataset(
                dataset_path=current_dataset,
                refinement_results=refinement_results,
                output_path=improved_dataset
            )
            
            iteration_results["improved_dataset"] = improved_dataset
            
            # Update current dataset for next iteration
            current_dataset = improved_dataset
            
            # Add iteration results
            results["iterations"].append(iteration_results)
            
            logger.info(f"Completed feedback loop iteration {iteration+1}")
            logger.info(f"Refined {refinement_results['summary']['refined_count']} responses")
            logger.info(f"Average improvement: {refinement_results['summary']['average_improvement']:.2f} points")
        
        # Set final dataset in summary
        results["summary"]["final_dataset"] = current_dataset
        
        # Compare initial and final evaluation (if we did multiple iterations)
        if iterations > 1:
            init_eval = results["iterations"][0]["evaluation_results"]
            final_eval = results["iterations"][-1]["evaluation_results"]
            
            # In a real implementation, you would load and compare these evaluations
            # For simplicity, we'll just estimate improvement from refinement results
            final_refinement = refinement_results["summary"]
            results["summary"]["improvement"] = {
                "refined_count": final_refinement["refined_count"],
                "average_improvement": final_refinement["average_improvement"],
                "total_examples": final_refinement["total_examples"]
            }
        
        # Save overall results
        summary_file = os.path.join(self.output_dir, "feedback_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Feedback loop completed. Summary saved to {summary_file}")
        
        return results
    
    def _update_dataset(
        self,
        dataset_path: str,
        refinement_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Update the dataset with refined responses.
        
        Args:
            dataset_path: Path to the original dataset
            refinement_results: Results from the refinement process
            output_path: Path to save the updated dataset
        """
        # Load the original dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        
        # Create a mapping from question_idx to refined response
        refined_map = {}
        for item in refinement_results.get("refined_responses", []):
            idx = item.get("question_idx")
            refined_response = item.get("refined_response")
            if idx is not None and refined_response:
                refined_map[idx] = refined_response
        
        # Update the dataset with refined responses
        updated = 0
        for idx, item in enumerate(dataset):
            if idx in refined_map:
                # Update the response in the dataset
                # This assumes the dataset has a "generated_response" field - adjust as needed
                if "generated_response" in item:
                    item["generated_response"] = refined_map[idx]
                    item["is_refined"] = True
                    updated += 1
                # For datasets using a messages format (e.g., for chat models)
                elif "messages" in item:
                    # Find and update the assistant message
                    for msg in item["messages"]:
                        if msg.get("role") == "assistant":
                            msg["content"] = refined_map[idx]
                            item["is_refined"] = True
                            updated += 1
                            break
        
        logger.info(f"Updated {updated} items in the dataset with refined responses")
        
        # Write the updated dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved updated dataset to {output_path}")


def run_feedback_pipeline(
    dataset_path: str,
    teacher_provider: str = config.RAG_MODEL_PROVIDER,
    teacher_model: str = config.RAG_MODEL_NAME,
    student_model: str = None,
    output_dir: str = "outputs/feedback",
    iterations: int = 1,
    batch_size: int = 10,
    samples: int = None,
    **kwargs
):
    """
    Convenience function to run the feedback pipeline with minimal setup.
    
    Args:
        dataset_path: Path to the dataset file (JSONL)
        teacher_provider: Provider for teacher model
        teacher_model: Teacher model name
        student_model: Path to student model (if available)
        output_dir: Directory to save results
        iterations: Number of feedback iterations
        batch_size: Batch size for processing
        samples: Number of samples to evaluate (None = all)
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Dictionary with feedback loop results
    """
    # Determine student provider based on student model
    student_provider = "local"
    if student_model:
        # Check if it's a HuggingFace model ID
        if '/' in student_model and not os.path.exists(student_model):
            student_provider = "huggingface"
    
    # Initialize feedback loop
    feedback_loop = FeedbackLoop(
        teacher_provider=teacher_provider,
        teacher_model=teacher_model,
        student_provider=student_provider,
        student_model=student_model,
        output_dir=output_dir,
        **kwargs
    )
    
    # Run feedback loop
    results = feedback_loop.run_feedback_loop(
        dataset_path=dataset_path,
        iterations=iterations,
        batch_size=batch_size,
        evaluation_samples=samples
    )
    
    return results
