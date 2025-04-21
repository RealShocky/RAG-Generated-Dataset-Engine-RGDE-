# TeacherForge Feedback System

This document describes the feedback loop system in TeacherForge that enables self-improving RAG-generated datasets.

## Overview

The feedback system implements a complete lifecycle for evaluating, refining, and improving RAG-generated datasets:

1. **Evaluation**: Compare teacher and student model outputs, identifying areas for improvement
2. **Refinement**: Automatically improve low-quality responses based on evaluation feedback
3. **Feedback Loop**: Orchestrate the evaluation-refinement cycle for continuous improvement

![Feedback Loop Diagram](https://raw.githubusercontent.com/RealShocky/RAG-Generated-Dataset-Engine-RGDE-/master/docs/images/feedback_loop.png)

## Components

### TeacherStudentEvaluator

The evaluator component compares teacher and student model responses, providing detailed metrics and feedback on quality.

**Key Features:**
- Multi-dimensional evaluation (factual accuracy, relevance, coherence, etc.)
- Improvement area identification
- Detailed performance metrics and statistics

**Usage:**
```python
from feedback.evaluator import TeacherStudentEvaluator

evaluator = TeacherStudentEvaluator(
    teacher_provider="openai",
    teacher_model="gpt-4o",
    student_provider="local",
    student_model="path/to/student_model"
)

evaluation_results = evaluator.evaluate_responses(
    questions=questions,
    contexts=contexts,
    batch_size=10,
    output_file="evaluation_results.json"
)

# Identify improvement areas
improvement_areas = evaluator.identify_improvement_areas(evaluation_results)
```

### ResponseRefiner

The refiner component automatically improves low-quality responses based on evaluation feedback.

**Key Features:**
- Targeted refinement based on evaluation scores
- Preservation of original response style and character
- Score-based thresholding to focus on problematic responses

**Usage:**
```python
from feedback.refinement import ResponseRefiner

refiner = ResponseRefiner(
    refiner_provider="openai",
    refiner_model="gpt-4o",
    refinement_threshold=7.0  # Only refine responses below this score
)

refinement_results = refiner.refine_responses(
    evaluation_results=evaluation_results,
    batch_size=10,
    output_file="refinement_results.json"
)
```

### FeedbackLoop

The feedback loop orchestrates the evaluation and refinement process across multiple iterations.

**Key Features:**
- Multi-iteration improvement cycle
- Comprehensive metrics tracking
- Dataset versioning and evolution

**Usage:**
```python
from feedback.feedback_loop import FeedbackLoop

feedback_loop = FeedbackLoop(
    teacher_provider="openai",
    teacher_model="gpt-4o",
    student_provider="local",
    student_model="path/to/student_model",
    output_dir="outputs/feedback"
)

results = feedback_loop.run_feedback_loop(
    dataset_path="path/to/dataset.jsonl",
    iterations=3,
    batch_size=10,
    evaluation_samples=100
)
```

## Command-Line Interface

TeacherForge provides a comprehensive command-line interface for the feedback system in `feedback_cli.py`.

### Run Complete Feedback Loop

```bash
python feedback_cli.py run-feedback \
    --dataset outputs/your_dataset/dataset.jsonl \
    --teacher-model "gpt-4o" \
    --student-model "outputs/your_dataset/student_model" \
    --iterations 2 \
    --batch-size 10 \
    --samples 50 \
    --output-dir "outputs/feedback_results"
```

### Evaluate Only

```bash
python feedback_cli.py evaluate \
    --dataset outputs/your_dataset/dataset.jsonl \
    --teacher-model "gpt-4o" \
    --student-model "outputs/your_dataset/student_model" \
    --samples 50 \
    --output-file "outputs/evaluation_results.json"
```

### Refine Only

```bash
python feedback_cli.py refine \
    --evaluation-file "outputs/evaluation_results.json" \
    --refiner-model "gpt-4o" \
    --threshold 7.0 \
    --output-file "outputs/refinement_results.json"
```

## Configuration

The feedback system uses the same configuration system as the rest of TeacherForge. Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_MODEL_PROVIDER` | Provider for teacher/evaluator models | `openai` |
| `RAG_MODEL_NAME` | Model name for teacher/evaluator | `gpt-3.5-turbo` |
| `OPENAI_API_KEY` | OpenAI API key for OpenAI providers | - |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude providers | - |
| `HF_API_TOKEN` | HuggingFace API token for HF providers | - |

## Workflow Integration

The feedback system is designed to integrate seamlessly with the existing TeacherForge workflow:

1. Generate initial dataset with `teacherforge.py run-pipeline`
2. Train a student model with `training.train_student`
3. Run the feedback loop with `feedback_cli.py run-feedback`
4. Retrain the student model with the improved dataset

This creates a virtuous cycle of dataset improvement and model training.

## Advanced Features

### Custom Evaluation Metrics

You can customize the evaluation metrics by providing a list of metrics to the evaluator:

```python
evaluator = TeacherStudentEvaluator(
    metrics=["factual_accuracy", "relevance", "completeness", "coherence", "helpfulness"]
)
```

### Progressive Refinement Threshold

Lower the refinement threshold in successive iterations to focus on increasingly subtle issues:

```python
for iteration in range(3):
    threshold = 7.0 - (iteration * 0.5)  # 7.0, 6.5, 6.0
    refiner = ResponseRefiner(refinement_threshold=threshold)
```

## Limitations and Future Work

- Currently, the evaluator uses the teacher model for evaluation, which may introduce bias
- The system does not yet support multi-modal content (images, audio, etc.)
- Future versions will implement RLHF-style reward models for more objective evaluation
