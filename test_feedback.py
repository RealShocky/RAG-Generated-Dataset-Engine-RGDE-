#!/usr/bin/env python
"""
Test script for the TeacherForge feedback system
This demonstrates the functionality of the feedback loop system with a sample dataset
"""

import argparse
import json
import logging
import os
from pathlib import Path

import config
from feedback.feedback_loop import FeedbackLoop
from feedback.evaluator import TeacherStudentEvaluator
from feedback.refinement import ResponseRefiner

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def create_sample_dataset(output_path, num_samples=5):
    """Create a sample dataset for testing."""
    
    # Sample questions about machine learning
    questions = [
        "What is the difference between supervised and unsupervised learning?",
        "Explain how gradient descent works in neural networks.",
        "What is overfitting and how can it be prevented?",
        "Describe the concept of regularization in machine learning.",
        "What are the advantages of using ensemble methods?",
        "How does a convolutional neural network work?",
        "What is the purpose of cross-validation?",
        "Explain the difference between precision and recall.",
        "What is the curse of dimensionality?",
        "How does reinforcement learning differ from supervised learning?"
    ]
    
    # Sample context documents (simplified for testing)
    sample_contexts = [
        [
            {
                "title": "Machine Learning Basics",
                "text": "Supervised learning is a type of machine learning where the model is trained on labeled data, meaning that each training example comes with an associated target or label. The model learns to predict the label from the input features. In contrast, unsupervised learning involves training on data without labels, and the model must find patterns or structure in the data on its own.",
                "source": "ML Textbook"
            }
        ],
        [
            {
                "title": "Neural Network Optimization",
                "text": "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In neural networks, we use gradient descent to adjust the weights in order to minimize the loss function. The process involves calculating the gradient of the loss function with respect to each weight, and then updating the weights in the opposite direction of the gradient.",
                "source": "Deep Learning Guide"
            }
        ],
        [
            {
                "title": "Model Evaluation",
                "text": "Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor performance on new, unseen data. It happens when a model is too complex relative to the amount and noisiness of the training data. To prevent overfitting, techniques such as cross-validation, regularization, early stopping, and using simpler models can be employed.",
                "source": "Data Science Handbook"
            }
        ],
        [
            {
                "title": "Regularization Techniques",
                "text": "Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. This penalty discourages complex models, effectively reducing variance at the cost of a slight increase in bias. Common regularization methods include L1 regularization (Lasso), which can lead to sparse models by forcing some weights to zero, and L2 regularization (Ridge), which penalizes large weight values without necessarily eliminating them.",
                "source": "Statistical Learning Methods"
            }
        ],
        [
            {
                "title": "Ensemble Learning",
                "text": "Ensemble methods combine the predictions of multiple machine learning models to produce a final prediction that is often more accurate than any individual model's prediction. The advantages of ensemble methods include improved accuracy, reduced variance, increased robustness, and better handling of complex datasets. Common ensemble techniques include bagging (e.g., Random Forests), boosting (e.g., AdaBoost, Gradient Boosting), and stacking.",
                "source": "Advanced Machine Learning"
            }
        ]
    ]
    
    # Create minimal dataset for testing
    n = min(num_samples, len(questions))
    dataset = []
    for i in range(n):
        item = {
            "question": questions[i],
            "context": sample_contexts[i % len(sample_contexts)],
            "generated_response": "",  # To be filled by the test
            "metadata": {
                "domain": "machine_learning",
                "difficulty": "intermediate",
                "source": "test_dataset"
            }
        }
        dataset.append(item)
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Created sample dataset with {n} questions at {output_path}")
    return output_path

def test_teacher_generation(dataset_path, output_path, model_provider="openai", model_name="gpt-3.5-turbo"):
    """Generate teacher responses for the sample dataset."""
    from generation.rag_generator import get_generator
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Initialize generator
    generator = get_generator(
        model_provider=model_provider,
        model_name=model_name,
        temperature=0.7
    )
    
    # Generate responses
    logger.info(f"Generating teacher responses using {model_provider}/{model_name}...")
    for item in dataset:
        question = item["question"]
        context = item["context"]
        
        try:
            response = generator.generate(question, context)
            item["generated_response"] = response.get("generated_response", "")
            logger.info(f"Generated response for: {question[:30]}...")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            item["generated_response"] = f"Error: {str(e)}"
    
    # Save updated dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved dataset with teacher responses to {output_path}")
    return output_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the TeacherForge feedback system with a sample dataset"
    )
    
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API key (if not set in .env)"
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo",
        help="Model name (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Number of samples to use (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/feedback_test",
        help="Output directory for test results (default: outputs/feedback_test)"
    )
    parser.add_argument(
        "--run-full", action="store_true",
        help="Run the full feedback loop (requires API credits)"
    )
    
    return parser.parse_args()

def main():
    """Main test function."""
    args = parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create sample dataset
    logger.info("Step 1: Creating sample dataset")
    sample_dataset = os.path.join(args.output_dir, "sample_dataset.jsonl")
    create_sample_dataset(sample_dataset, args.samples)
    
    if args.run_full:
        # Step 2: Generate teacher responses
        logger.info("Step 2: Generating teacher responses")
        teacher_dataset = os.path.join(args.output_dir, "teacher_dataset.jsonl")
        test_teacher_generation(
            sample_dataset, 
            teacher_dataset,
            model_provider=args.provider,
            model_name=args.model
        )
        
        # Step 3: Run feedback loop
        logger.info("Step 3: Running feedback loop")
        feedback_loop = FeedbackLoop(
            teacher_provider=args.provider,
            teacher_model=args.model,
            output_dir=os.path.join(args.output_dir, "feedback_results")
        )
        
        results = feedback_loop.run_feedback_loop(
            dataset_path=teacher_dataset,
            iterations=1,
            batch_size=args.samples,
            evaluation_samples=args.samples
        )
        
        # Print summary
        summary = results.get("summary", {})
        logger.info("-" * 50)
        logger.info("Feedback Loop Test Summary:")
        logger.info(f"Initial dataset: {summary.get('initial_dataset')}")
        logger.info(f"Final dataset: {summary.get('final_dataset')}")
        logger.info("-" * 50)
    else:
        logger.info("Skipping full feedback loop test (use --run-full to enable)")
        logger.info("Sample dataset created successfully at:")
        logger.info(sample_dataset)
    
    logger.info("Test completed!")

if __name__ == "__main__":
    main()
