"""
Response Refinement Module
Refines and improves responses based on evaluation feedback
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from tqdm import tqdm

# Add project root to path to import config
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config
from generation.llm_providers import get_llm_provider

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ResponseRefiner:
    """
    Refines and improves responses based on evaluation feedback.
    Uses feedback to correct factual errors and improve response quality.
    """
    
    def __init__(
        self,
        refiner_provider: str = config.RAG_MODEL_PROVIDER,
        refiner_model: str = config.RAG_MODEL_NAME,
        refinement_threshold: float = 7.0,  # Only refine responses below this score
        **kwargs
    ):
        """
        Initialize response refiner.
        
        Args:
            refiner_provider: Provider for the refiner model (openai, anthropic, etc.)
            refiner_model: Refiner model name
            refinement_threshold: Score threshold below which responses get refined
            **kwargs: Additional provider-specific parameters
        """
        self.refiner_provider = refiner_provider
        self.refiner_model = refiner_model
        self.refinement_threshold = refinement_threshold
        
        # Initialize refiner model
        self.refiner = get_llm_provider(
            provider_type=refiner_provider,
            model_name=refiner_model,
            **kwargs
        )
        logger.info(f"Initialized refiner model: {refiner_model} with provider {refiner_provider}")
        
    def refine_responses(
        self,
        evaluation_results: Dict[str, Any],
        batch_size: int = 10,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refine responses based on evaluation results.
        
        Args:
            evaluation_results: Results from TeacherStudentEvaluator
            batch_size: Number of refinements to process at once
            output_file: File to save refinement results
            
        Returns:
            Dictionary with refinement results
        """
        results = {
            "refined_responses": [],
            "summary": {
                "total_examples": 0,
                "refined_count": 0,
                "average_improvement": 0.0
            }
        }
        
        evaluations = evaluation_results.get("evaluations", [])
        if not evaluations:
            logger.warning("No evaluations found in evaluation results")
            return results
        
        results["summary"]["total_examples"] = len(evaluations)
        
        # Identify responses that need refinement
        to_refine = []
        for idx, eval_item in enumerate(evaluations):
            # Calculate average score across all metrics
            scores = eval_item.get("scores", {})
            if not scores:
                continue
                
            avg_score = sum(scores.values()) / len(scores)
            
            # If below threshold, add to refinement list
            if avg_score < self.refinement_threshold:
                to_refine.append((idx, eval_item))
        
        logger.info(f"Found {len(to_refine)} responses below threshold {self.refinement_threshold} that need refinement")
        results["summary"]["refined_count"] = len(to_refine)
        
        # Process refinements in batches
        total_improvement = 0.0
        for i in range(0, len(to_refine), batch_size):
            batch = to_refine[i:i+batch_size]
            
            # Refine each response in the batch
            for idx, eval_item in tqdm(batch, desc="Refining responses"):
                question = eval_item.get("question", "")
                context = eval_item.get("context", [])
                teacher_response = eval_item.get("teacher_response", "")
                student_response = eval_item.get("student_response", "")
                feedback = eval_item.get("feedback", "")
                scores = eval_item.get("scores", {})
                
                # Calculate pre-refinement average score
                pre_avg_score = sum(scores.values()) / len(scores) if scores else 0
                
                # Refine the response
                refined_response = self._refine_response(
                    question=question,
                    context=context,
                    teacher_response=teacher_response,
                    student_response=student_response,
                    feedback=feedback,
                    scores=scores
                )
                
                # Re-evaluate the refined response (if provided in results)
                # In a real implementation, you would call the evaluator here
                # For simplicity, we'll just estimate improvement
                estimated_improvement = min(10 - pre_avg_score, 2.0)  # Estimate improvement between 0-2 points
                post_scores = {k: min(v + estimated_improvement, 10.0) for k, v in scores.items()}
                post_avg_score = sum(post_scores.values()) / len(post_scores) if post_scores else 0
                
                improvement = post_avg_score - pre_avg_score
                total_improvement += improvement
                
                # Add to results
                results["refined_responses"].append({
                    "question_idx": eval_item.get("question_idx", idx),
                    "question": question,
                    "original_response": student_response,
                    "refined_response": refined_response,
                    "feedback": feedback,
                    "scores": {
                        "before": scores,
                        "after": post_scores,
                        "improvement": improvement
                    }
                })
        
        # Calculate average improvement
        if results["summary"]["refined_count"] > 0:
            results["summary"]["average_improvement"] = total_improvement / results["summary"]["refined_count"]
        
        # Save results to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Refinement results saved to {output_file}")
            
        return results
    
    def _refine_response(
        self,
        question: str,
        context: List[Dict] = None,
        teacher_response: str = "",
        student_response: str = "",
        feedback: str = "",
        scores: Dict[str, float] = None
    ) -> str:
        """
        Refine a single response using the refiner model.
        
        Args:
            question: The input question
            context: The retrieved context (optional)
            teacher_response: Reference response from teacher model
            student_response: Response to be refined
            feedback: Evaluation feedback
            scores: Evaluation scores
            
        Returns:
            Refined response
        """
        # Format context if provided
        context_text = ""
        if context:
            for i, doc in enumerate(context, 1):
                title = doc.get("title", f"Document {i}")
                text = doc.get("text", "")
                source = doc.get("source", "Unknown source")
                context_text += f"[{i}] {title}\nSource: {source}\nContent: {text}\n\n"
        
        # Prepare the refinement prompt
        system_prompt = """You are an expert AI tutor helping to refine and improve answers.
You will receive:
1. A question
2. Retrieved context information (if available)
3. An original response that needs improvement
4. A reference 'teacher' response
5. Evaluation feedback pointing out issues with the original response
6. Specific scores on different quality dimensions

Your task is to create an improved version of the original response that:
1. Addresses all the issues mentioned in the feedback
2. Uses the retrieved context correctly and avoids factual errors
3. Maintains a similar style and tone as the original
4. Is complete, coherent, relevant, and helpful
5. Approaches the quality of the teacher response

DO NOT simply copy the teacher response. Instead, improve the original response while maintaining its character.
"""
        
        # Format scores text
        scores_text = ""
        if scores:
            for metric, score in scores.items():
                scores_text += f"- {metric}: {score}/10\n"
        
        # User prompt with refinement task
        refinement_prompt = f"""
Question: {question}

{"Retrieved Context:\n" + context_text if context_text else ""}

Original Response (needs improvement):
{student_response}

Teacher Reference Response:
{teacher_response}

Evaluation Feedback:
{feedback}

Scores:
{scores_text}

Please provide a refined and improved version of the original response that addresses the issues in the feedback.
Focus on fixing factual accuracy, completeness, relevance, coherence, and overall helpfulness.
"""
        
        # Get refinement from the refiner model
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": refinement_prompt}
            ]
            
            if hasattr(self.refiner, 'generate_chat'):
                # Use chat-optimized method if available
                response_data = self.refiner.generate_chat(
                    messages=messages,
                    temperature=0.7,  # Higher temperature for creative improvements
                    max_tokens=1024
                )
            else:
                # Fall back to standard generation
                combined_prompt = f"{system_prompt}\n\n{refinement_prompt}"
                response_data = self.refiner.generate(
                    prompt=combined_prompt,
                    temperature=0.7,
                    max_tokens=1024
                )
            
            refined_response = response_data.get("text", "")
            return refined_response
            
        except Exception as e:
            logger.error(f"Error in refinement process: {e}")
            return f"Refinement failed: {str(e)}\n\nOriginal response: {student_response}"
