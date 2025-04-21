"""
Prompt Generator Module
Generates diverse, domain-specific questions for dataset creation.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import config
from config import PROMPT_TEMPLATES

import openai

# Set up logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Set API key
openai.api_key = config.OPENAI_API_KEY


def generate_questions(
    num_prompts: int = 10,
    domain: str = "general knowledge",
    output_file: Optional[str] = None
) -> List[str]:
    """
    Generate domain-specific questions using an LLM.
    
    Args:
        num_prompts: Number of questions to generate
        domain: Domain for the questions
        output_file: Path to save the generated questions
        
    Returns:
        List of generated questions
    """
    logger.info(f"Generating {num_prompts} questions for domain: {domain}")
    
    try:
        # Generate questions using OpenAI
        prompt = PROMPT_TEMPLATES["prompt_generator"].format(
            num_prompts=num_prompts,
            domain_specific=f"focused on {domain}" if domain != "general knowledge" else "general knowledge"
        )
        
        response = openai.chat.completions.create(
            model=config.RAG_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse, interesting questions for dataset creation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        # Extract questions from response
        question_text = response.choices[0].message.content.strip()
        questions = [q.strip() for q in question_text.split('\n') if q.strip()]
        
        logger.info(f"Successfully generated {len(questions)} questions")
        
        # Save to output file if specified
        if output_file:
            save_questions_to_jsonl(questions, output_file)
            
        return questions
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return []


def save_questions_to_jsonl(questions: List[str], output_file: str) -> None:
    """
    Save generated questions to a JSONL file.
    
    Args:
        questions: List of question strings
        output_file: Path to output file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps({"question": question}) + '\n')
    
    logger.info(f"Saved {len(questions)} questions to {output_file}")


def load_questions_from_jsonl(input_file: str) -> List[str]:
    """
    Load questions from a JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        
    Returns:
        List of question strings
    """
    questions = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if "question" in data:
                questions.append(data["question"])
    
    logger.info(f"Loaded {len(questions)} questions from {input_file}")
    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate domain-specific questions")
    parser.add_argument("--num", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--domain", type=str, default="general knowledge", help="Domain for questions")
    parser.add_argument("--output", type=str, default=str(config.PROMPTS_DIR / "prompts.jsonl"),
                        help="Output file path for generated questions")
    parser.add_argument("--input", type=str, help="Input file with existing questions (optional)")
    
    args = parser.parse_args()
    
    if args.input:
        # Load existing questions
        questions = load_questions_from_jsonl(args.input)
        print(f"Loaded {len(questions)} existing questions")
    else:
        # Generate new questions
        questions = generate_questions(args.num, args.domain, args.output)
        print(f"Generated {len(questions)} questions")
