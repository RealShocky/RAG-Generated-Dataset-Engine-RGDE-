# TeacherForge: RAG-Generated Dataset Engine (RGDE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, automated pipeline that uses RAG (Retrieval-Augmented Generation) to generate high-quality, domain-grounded training datasets for fine-tuning smaller, cost-effective student models.

**Status: Implemented and Tested ✅**

![TeacherForge Pipeline](https://user-images.githubusercontent.com/YOUR_USERNAME/RAG-Generated-Dataset-Engine-RGDE-/raw/main/docs/images/pipeline_diagram.png)

> **Note**: Add the pipeline_diagram.png to your repository after pushing to GitHub

## Overview

TeacherForge enables you to:
- Generate synthetic instruction datasets using RAG models as teachers
- Maintain traceability between questions, retrieved documents, and generated answers
- Create datasets formatted for supervised fine-tuning and LoRA/QLoRA training
- Build toward self-improving AI systems

## Project Structure

```
/teacherforge/
├── prompts/              # Question generation and storage
│   └── prompts.jsonl     # Example/generated questions
├── retrieval/            # Document retrieval components
│   └── retriever.py      # Vector DB and document retriever
├── generation/           # RAG answer generation
│   └── rag_generator.py  # LLM generation with retrieved context
├── postprocessing/       # Validation and filtering
│   └── cleaner.py        # Answer quality checks and metadata tagging
├── dataset/              # Dataset compilation 
│   └── build_dataset.py  # Format final instruction datasets
├── training/             # Student model training
│   └── train_student.py  # LoRA/QLoRA fine-tuning
├── outputs/              # Generated artifacts
│   ├── dataset.jsonl     # Output datasets
│   ├── logs/             # Process logs
│   └── checkpoints/      # Model checkpoints
├── config.py             # Configuration settings
├── main.py               # Main orchestration
└── requirements.txt      # Dependencies
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/teacherforge.git
cd teacherforge

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys for the desired LLM providers
# - OpenAI API key for GPT models
# - Anthropic API key for Claude models
# - HuggingFace API token for their models
# - Path to model file for local models

# Create FAISS index with sample documents
python create_faiss_index.py

# Process and ingest documents
python ingest_documents.py --source your_documents_folder --format pdf

# Generate questions
python teacherforge.py generate-questions --num 10 --domain "your domain"

# Run the full pipeline
python teacherforge.py run-pipeline --questions prompts/prompts.jsonl --output outputs/your_dataset

# Launch the web interface (optional)
python web_interface.py
```

## Configuration

Create a `.env` file in the project root with your API keys (a template is provided):

```
# API Keys for various LLM providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HF_API_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key

# LLM Provider settings
RAG_MODEL_PROVIDER=openai  # Options: openai, anthropic, huggingface, local
RAG_MODEL_NAME=gpt-4  # Model name for the selected provider
RAG_TEMPERATURE=0.1  # Generation temperature

# Local model settings (for local provider)
LOCAL_MODEL_PATH=path/to/your/model.gguf  # Path to local model file
LOCAL_MODEL_BACKEND=llamacpp  # Options: llamacpp, vllm

# Vector DB settings
VECTOR_DB_TYPE=faiss  # Options: faiss, qdrant, weaviate, chroma
VECTOR_DB_URL=  # Qdrant URL (if using Qdrant)
VECTOR_DB_COLLECTION=teacherforge  # Collection/index name

# FAISS settings (for FAISS)
FAISS_INDEX_PATH=data/faiss.index
FAISS_DOCUMENTS_PATH=data/sample_documents.json

# Weaviate settings (for Weaviate)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=

# Chroma settings (for Chroma)
CHROMA_PERSIST_DIR=./chroma_db
```

## Usage

### 1. Prepare your document corpus

Index your documents into a vector database (e.g., FAISS, Qdrant).

### 2. Generate questions or provide your own

```bash
python -m prompts.generate_prompts --num 100 --domain "machine learning"
```

### 3. Run the full pipeline

```bash
# Using OpenAI
python teacherforge.py run-pipeline --questions prompts/prompts.jsonl --output outputs/your_dataset --provider openai

# Using Anthropic Claude
python teacherforge.py run-pipeline --questions prompts/prompts.jsonl --output outputs/your_dataset --provider anthropic --model claude-3-opus-20240229

# Using HuggingFace
python teacherforge.py run-pipeline --questions prompts/prompts.jsonl --output outputs/your_dataset --provider huggingface --model mistralai/Mistral-7B-Instruct-v0.2

# Using a local model
python teacherforge.py run-pipeline --questions prompts/prompts.jsonl --output outputs/your_dataset --provider local --model-path path/to/your/model.gguf --backend llamacpp
```

### 4. Train a student model

```bash
python -m training.train_student --dataset outputs/dataset.jsonl --model "mistralai/Mistral-7B-v0.1"
```

## Components

- **Prompt Generator**: Creates diverse, domain-specific questions
- **Document Processor**: Smart document chunking with overlap control for different document formats
- **Retriever**: Queries vector databases (FAISS, Qdrant, Weaviate, Chroma) and returns relevant documents
- **Generator**: Uses a large language model with RAG to generate answers, with support for multiple LLM providers:
  - **OpenAI**: GPT-3.5 and GPT-4 models
  - **Anthropic**: Claude models (opus, sonnet, haiku)
  - **HuggingFace**: Access to all HuggingFace Inference API models
  - **Local Models**: Support for running local models via llama.cpp or vLLM
- **Post-processor**: Validates and enhances the generated dataset
- **Dataset Builder**: Formats data for instruction fine-tuning
- **Training Pipeline**: Trains smaller models via LoRA/QLoRA
- **Web Interface**: Streamlit dashboard for dataset visualization, document exploration, pipeline configuration, and dataset export capabilities with dark mode styling for improved readability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* OpenAI and other LLM providers for their incredible models
* The open-source community for their invaluable tools and libraries
* Everyone who contributes to making AI more accessible and useful
