"""
Teacher-Student Evaluator Module
Evaluates and compares responses from teacher and student models
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path to import config
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config
from generation.rag_generator import get_generator
from generation.llm_providers import get_llm_provider

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class TeacherStudentEvaluator:
    """
    Evaluates responses from teacher and student models using various metrics.
    Provides comparison and scoring to identify areas for improvement.
    """
    
    def __init__(
        self,
        teacher_provider: str = config.RAG_MODEL_PROVIDER,
        teacher_model: str = config.RAG_MODEL_NAME,
        student_provider: str = "local",
        student_model: str = None,
        evaluator_provider: str = None,
        evaluator_model: str = None,
        metrics: List[str] = None,
        **kwargs
    ):
        """
        Initialize teacher-student evaluator.
        
        Args:
            teacher_provider: Provider for teacher model (openai, anthropic, etc.)
            teacher_model: Teacher model name
            student_provider: Provider for student model (likely 'local' if fine-tuned)
            student_model: Path to student model or model name
            evaluator_provider: Provider for evaluation model (if None, uses teacher)
            evaluator_model: Model to use for evaluation (if None, uses teacher)
            metrics: List of metrics to use for evaluation
            **kwargs: Additional provider-specific parameters
        """
        self.teacher_provider = teacher_provider
        self.teacher_model = teacher_model
        self.student_provider = student_provider
        self.student_model = student_model
        
        # Set default metrics if none provided
        self.metrics = metrics or [
            "factual_accuracy", 
            "relevance",
            "completeness",
            "coherence",
            "helpfulness"
        ]
        
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
        
        # Initialize evaluator model (use teacher by default)
        self.evaluator_provider = evaluator_provider or teacher_provider
        self.evaluator_model = evaluator_model or teacher_model
        
        evaluator_kwargs = kwargs.copy()
        self.evaluator = get_llm_provider(
            provider_type=self.evaluator_provider,
            model_name=self.evaluator_model,
            **evaluator_kwargs
        )
        logger.info(f"Initialized evaluator model: {self.evaluator_model} with provider {self.evaluator_provider}")
        
    def evaluate_responses(
        self,
        questions: List[str],
        contexts: List[List[Dict]],
        teacher_responses: Optional[List[str]] = None,
        student_responses: Optional[List[str]] = None,
        batch_size: int = 10,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate teacher and student responses for a set of questions.
        
        Args:
            questions: List of input questions
            contexts: List of retrieved contexts for each question
            teacher_responses: Pre-generated teacher responses (optional)
            student_responses: Pre-generated student responses (optional)
            batch_size: Number of evaluations to process at once
            output_file: File to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            "metrics": self.metrics,
            "evaluations": [],
            "summary": {}
        }
        
        # Generate teacher responses if not provided
        if teacher_responses is None and self.teacher_generator:
            logger.info("Generating teacher responses...")
            teacher_responses = []
            for i, (question, context) in enumerate(tqdm(zip(questions, contexts), total=len(questions))):
                response = self.teacher_generator.generate(question, context)
                teacher_responses.append(response.get("generated_response", ""))
        
        # Generate student responses if not provided
        if student_responses is None and self.student_generator:
            logger.info("Generating student responses...")
            student_responses = []
            for i, (question, context) in enumerate(tqdm(zip(questions, contexts), total=len(questions))):
                response = self.student_generator.generate(question, context)
                student_responses.append(response.get("generated_response", ""))
        
        # If we have both teacher and student responses, evaluate them
        if teacher_responses and student_responses:
            logger.info(f"Evaluating {len(questions)} response pairs...")
            
            # Process in batches
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i+batch_size]
                batch_contexts = contexts[i:i+batch_size]
                batch_teacher = teacher_responses[i:i+batch_size]
                batch_student = student_responses[i:i+batch_size]
                
                # Evaluate each pair
                for j, (question, context, t_resp, s_resp) in enumerate(zip(
                    batch_questions, batch_contexts, batch_teacher, batch_student
                )):
                    idx = i + j
                    logger.debug(f"Evaluating response pair {idx+1}/{len(questions)}")
                    
                    eval_result = self._evaluate_response_pair(
                        question=question,
                        context=context,
                        teacher_response=t_resp,
                        student_response=s_resp
                    )
                    
                    results["evaluations"].append({
                        "question_idx": idx,
                        "question": question,
                        "teacher_response": t_resp,
                        "student_response": s_resp,
                        "scores": eval_result["scores"],
                        "feedback": eval_result["feedback"]
                    })
        
        # Calculate summary statistics
        if results["evaluations"]:
            summary = {"overall_scores": {}, "per_metric": {}}
            
            # Extract scores for each metric
            for metric in self.metrics:
                scores = [e["scores"].get(metric, 0) for e in results["evaluations"]]
                summary["per_metric"][metric] = {
                    "mean": float(np.mean(scores)),
                    "median": float(np.median(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                }
                
            # Calculate overall scores (average across all metrics)
            overall_scores = []
            for eval_result in results["evaluations"]:
                metric_scores = [eval_result["scores"].get(m, 0) for m in self.metrics]
                overall_scores.append(np.mean(metric_scores))
                
            summary["overall_scores"] = {
                "mean": float(np.mean(overall_scores)),
                "median": float(np.median(overall_scores)),
                "std": float(np.std(overall_scores)),
                "min": float(np.min(overall_scores)),
                "max": float(np.max(overall_scores)),
                "total_examples": len(results["evaluations"])
            }
            
            results["summary"] = summary
        
        # Save results to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to {output_file}")
            
        return results
    
    def _evaluate_response_pair(
        self,
        question: str,
        context: List[Dict],
        teacher_response: str,
        student_response: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single teacher-student response pair using the evaluator model.
        
        Args:
            question: The input question
            context: The retrieved context used for generation
            teacher_response: Response from teacher model
            student_response: Response from student model
            
        Returns:
            Dictionary with scores and feedback
        """
        # Format context
        context_text = ""
        for i, doc in enumerate(context, 1):
            title = doc.get("title", f"Document {i}")
            text = doc.get("text", "")
            source = doc.get("source", "Unknown source")
            context_text += f"[{i}] {title}\nSource: {source}\nContent: {text}\n\n"
        
        # Prepare the evaluation prompt
        system_prompt = """You are an expert evaluator assessing the quality of AI responses to questions based on provided context.
You will be given: 
1. A question
2. Retrieved context that should be used to answer the question
3. A teacher model's response (considered the reference)
4. A student model's response (being evaluated)

Evaluate the student's response compared to the teacher's response on the following metrics:
- factual_accuracy (1-10): How factually accurate is the student response based on the retrieved context?
- relevance (1-10): How relevant is the student response to the question asked?
- completeness (1-10): How complete is the student response compared to the teacher's?
- coherence (1-10): How well-structured and coherent is the student response?
- helpfulness (1-10): How helpful would the student response be to a human asking this question?

Your evaluation must be fair and objective. Provide a JSON output with scores and feedback.
"""
        
        # User prompt with the evaluation task
        evaluation_prompt = f"""
Question: {question}

Retrieved Context:
{context_text}

Teacher Response:
{teacher_response}

Student Response:
{student_response}

Please provide a detailed evaluation comparing the student's response to the teacher's response.
Score each metric on a scale of 1-10 and provide feedback explaining the scores.
Return your evaluation as a JSON object with the structure:
{{
  "scores": {{
    "factual_accuracy": <score>,
    "relevance": <score>,
    "completeness": <score>,
    "coherence": <score>,
    "helpfulness": <score>
  }},
  "feedback": "<detailed feedback explaining scores and suggestions for improvement>"
}}
"""
        
        # Get evaluation from the evaluator model
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            if hasattr(self.evaluator, 'generate_chat'):
                # Use chat-optimized method if available
                response_data = self.evaluator.generate_chat(
                    messages=messages,
                    temperature=0.3,  # Low temperature for consistent evaluations
                    max_tokens=1024
                )
            else:
                # Fall back to standard generation
                combined_prompt = f"{system_prompt}\n\n{evaluation_prompt}"
                response_data = self.evaluator.generate(
                    prompt=combined_prompt,
                    temperature=0.3,
                    max_tokens=1024
                )
            
            response_text = response_data.get("text", "")
            
            # Extract JSON from response
            try:
                # Find JSON block in the response
                json_str = response_text.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                # Parse the JSON
                evaluation = json.loads(json_str)
                
                # Ensure all metrics are present
                for metric in self.metrics:
                    if metric not in evaluation.get("scores", {}):
                        evaluation.setdefault("scores", {})[metric] = 5  # Default mid-range score
                
                # Ensure feedback is present
                if "feedback" not in evaluation:
                    evaluation["feedback"] = "No specific feedback provided by evaluator."
                    
                return evaluation
                
            except (json.JSONDecodeError, IndexError, ValueError) as e:
                logger.warning(f"Failed to parse evaluator response as JSON: {e}")
                logger.debug(f"Raw response: {response_text}")
                
                # Fallback to default evaluation
                return {
                    "scores": {metric: 5 for metric in self.metrics},
                    "feedback": "Failed to parse evaluator response. Raw response: " + response_text[:200] + "..."
                }
                
        except Exception as e:
            logger.error(f"Error in evaluation process: {e}")
            return {
                "scores": {metric: 0 for metric in self.metrics},
                "feedback": f"Evaluation error: {str(e)}"
            }

    def identify_improvement_areas(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze evaluation results to identify areas for improvement.
        
        Args:
            evaluation_results: Results from evaluate_responses
            
        Returns:
            Dictionary with identified improvement areas
        """
        if not evaluation_results.get("evaluations"):
            return {"improvement_areas": [], "message": "No evaluations found"}
        
        # Identify weakest metrics
        per_metric = evaluation_results.get("summary", {}).get("per_metric", {})
        metric_means = {metric: stats.get("mean", 0) for metric, stats in per_metric.items()}
        sorted_metrics = sorted(metric_means.items(), key=lambda x: x[1])
        
        # Find worst-performing examples
        evaluations = evaluation_results.get("evaluations", [])
        worst_examples = []
        
        # For each metric, find the worst examples
        improvement_areas = {}
        for metric, score in sorted_metrics:
            # Find examples with lowest scores for this metric
            metric_examples = sorted(
                [(i, e) for i, e in enumerate(evaluations) if metric in e.get("scores", {})],
                key=lambda x: x[1]["scores"].get(metric, 10)
            )[:3]  # Get 3 worst examples
            
            worst_for_metric = []
            for idx, example in metric_examples:
                worst_for_metric.append({
                    "question_idx": example.get("question_idx", idx),
                    "question": example.get("question", ""),
                    "score": example["scores"].get(metric, 0),
                    "feedback": example.get("feedback", "")
                })
            
            improvement_areas[metric] = {
                "mean_score": metric_means[metric],
                "worst_examples": worst_for_metric,
                "suggestions": self._generate_improvement_suggestions(metric, worst_for_metric)
            }
        
        # Overall summary
        return {
            "improvement_areas": sorted_metrics,
            "details": improvement_areas,
            "overall_score": evaluation_results.get("summary", {}).get("overall_scores", {}).get("mean", 0)
        }
    
    def _generate_improvement_suggestions(self, metric: str, examples: List[Dict]) -> str:
        """
        Generate suggestions for improving a specific metric.
        
        Args:
            metric: The metric to generate suggestions for
            examples: List of worst examples for this metric
            
        Returns:
            Suggestions for improvement
        """
        # Prepare prompt for the evaluator model
        system_prompt = """You are an expert AI tutor helping to improve an AI student model.
Based on evaluation results, provide specific suggestions for improvement in a particular metric."""
        
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Question: {example.get('question', '')}\n"
            examples_text += f"Score: {example.get('score', 0)}/10\n"
            examples_text += f"Feedback: {example.get('feedback', '')}\n\n"
        
        user_prompt = f"""
The AI student model is performing poorly on the metric: {metric}.

Here are some examples of poor performance:

{examples_text}

Based on these examples, please provide 3-5 specific suggestions for how the student model could improve on the {metric} metric.
Focus on concrete, actionable tips that would help improve performance.
"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if hasattr(self.evaluator, 'generate_chat'):
                response_data = self.evaluator.generate_chat(
                    messages=messages,
                    temperature=0.5,
                    max_tokens=512
                )
            else:
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                response_data = self.evaluator.generate(
                    prompt=combined_prompt,
                    temperature=0.5,
                    max_tokens=512
                )
            
            suggestions = response_data.get("text", "")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return f"Failed to generate improvement suggestions: {str(e)}"
