# TeacherForge Documentation

## Overview

TeacherForge is a modular pipeline for creating high-quality instruction datasets using RAG (Retrieval-Augmented Generation). 
It uses a large language model as a "teacher" to generate grounded answers for fine-tuning smaller "student" models.

**Implementation Status**: Fully implemented and tested with sample data. The system successfully generates questions, retrieves relevant documents, produces grounded responses, validates them, and creates instruction datasets.

## System Architecture

TeacherForge consists of the following core components:

1. **Prompt Generator**: Creates diverse, domain-specific questions
2. **Retriever**: Queries a vector database to find relevant documents
3. **Generator**: Uses an LLM with RAG to produce grounded answers
4. **Post-Processor**: Validates and enhances the generated content
5. **Dataset Builder**: Formats data for instruction fine-tuning
6. **Training Pipeline**: Trains student models via LoRA/QLoRA

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- OpenAI API key for RAG generation
- (Optional) Hugging Face API token for model access
- (Optional) Weights & Biases API key for experiment tracking

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/teacherforge.git
   cd teacherforge
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   - Copy `.env.template` to `.env`
   - Add your API keys and configuration

## Configuration

TeacherForge is configured through environment variables and the `config.py` file:

- **API Keys**: Set your OpenAI, Hugging Face, and W&B keys
- **Vector DB**: Configure the vector database type and connection
- **RAG Settings**: Set model, temperature, and retrieval parameters
- **Training Settings**: Configure LoRA parameters and batch sizes

Example `.env` configuration:

```
OPENAI_API_KEY=your_openai_api_key
HF_API_TOKEN=your_huggingface_token
VECTOR_DB_TYPE=faiss
FAISS_INDEX_PATH=path/to/your/faiss.index
FAISS_DOCUMENTS_PATH=path/to/your/documents.json
```

## Vector Database Setup

TeacherForge requires a vector database with your document corpus. The system supports:

### FAISS (Local)

1. Create your document collection as a JSON file:
   ```json
   [
     {
       "id": "doc1",
       "title": "Document Title",
       "text": "Document content...",
       "source": "Document source"
     },
     ...
   ]
   ```

2. Build FAISS index:
   ```python
   from sentence_transformers import SentenceTransformer
   import faiss
   import json
   import numpy as np
   
   # Load documents
   with open('documents.json', 'r') as f:
       documents = json.load(f)
   
   # Initialize embedding model
   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
   
   # Create embeddings
   texts = [doc['text'] for doc in documents]
   embeddings = model.encode(texts, normalize_embeddings=True)
   
   # Build FAISS index
   dimension = embeddings.shape[1]
   index = faiss.IndexFlatIP(dimension)
   index.add(np.array(embeddings).astype('float32'))
   
   # Save index
   faiss.write_index(index, 'faiss.index')
   ```

3. Configure TeacherForge to use your index:
   ```
   VECTOR_DB_TYPE=faiss
   FAISS_INDEX_PATH=path/to/your/faiss.index
   FAISS_DOCUMENTS_PATH=path/to/your/documents.json
   ```

### Qdrant (Hosted or Local)

1. Install Qdrant client:
   ```bash
   pip install qdrant-client
   ```

2. Create a collection and upload documents:
   ```python
   from qdrant_client import QdrantClient
   from qdrant_client.models import VectorParams, Distance
   from sentence_transformers import SentenceTransformer
   import json
   
   # Load documents
   with open('documents.json', 'r') as f:
       documents = json.load(f)
   
   # Initialize Qdrant client
   client = QdrantClient(host="localhost", port=6333)
   
   # Initialize embedding model
   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
   
   # Create collection
   client.create_collection(
       collection_name="teacherforge",
       vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
   )
   
   # Upload documents
   for i, doc in enumerate(documents):
       embedding = model.encode(doc['text'], normalize_embeddings=True)
       client.upsert(
           collection_name="teacherforge",
           points=[
               {
                   "id": i,
                   "vector": embedding.tolist(),
                   "payload": doc
               }
           ]
       )
   ```

3. Configure TeacherForge to use Qdrant:
   ```
   VECTOR_DB_TYPE=qdrant
   VECTOR_DB_URL=http://localhost:6333
   VECTOR_DB_COLLECTION=teacherforge
   ```

## Usage

### Command Line Interface

TeacherForge provides a CLI for accessing all functionality:

```bash
python teacherforge.py [command] [options]
```

Available commands:

- `generate-questions`: Create domain-specific questions
- `generate-response`: Generate a RAG response for a single question
- `process-response`: Validate and enhance a generated response
- `build-dataset`: Create a dataset from processed responses
- `run-pipeline`: Execute the full pipeline

### Generate Questions

```bash
python teacherforge.py generate-questions --num 20 --domain "machine learning" --output prompts/ml_questions.jsonl
```

### Run Full Pipeline

```bash
python teacherforge.py run-pipeline --questions prompts/ml_questions.jsonl --output outputs/ml_dataset --num 20 --domain "machine learning"
```

### Train Student Model

```bash
python -m training.train_student --dataset outputs/ml_dataset/dataset_train.jsonl --val-dataset outputs/ml_dataset/dataset_val.jsonl --model "mistralai/Mistral-7B-v0.1" --epochs 3 --batch-size 4 --use-4bit
```

## Pipeline Components

### Prompt Generator

The prompt generator creates diverse questions for your domain of interest. In our testing, we successfully generated AI-related questions:

```python
from prompts.generate_prompts import generate_questions

questions = generate_questions(
    num_prompts=10,
    domain="machine learning",
    output_file="prompts/ml_questions.jsonl"
)
```

Example command-line usage:
```bash
python teacherforge.py generate-questions --num 5 --domain "AI and language models" --output prompts/ai_questions.jsonl
```

### RAG Generator

The RAG generator combines retrieval and generation:

```python
from generation.rag_generator import generate_rag_response

response = generate_rag_response(
    question="What is transfer learning?",
    top_k=4,
    temperature=0.1
)
```

### Post-Processing

Validate and enhance generated responses:

```python
from postprocessing.cleaner import process_response

processed = process_response(
    response_data=response,
    validate=True,
    enhance=True,
    min_confidence=0.7
)
```

### Dataset Builder

Build instruction-tuning datasets:

```python
from dataset.build_dataset import build_from_responses

dataset = build_from_responses(
    input_file="outputs/responses_processed.jsonl",
    output_file="outputs/dataset.jsonl",
    format_type="chat",
    include_traceability=True,
    create_train_val_split=True
)
```

## Advanced Customization

### Custom Retrievers

To implement a custom retriever, extend the `BaseRetriever` class:

```python
from retrieval.retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def __init__(self, custom_param, embedding_model=config.EMBEDDING_MODEL):
        super().__init__(embedding_model)
        self.custom_param = custom_param
        
    def retrieve(self, query, top_k=config.RAG_TOP_K):
        # Implement custom retrieval logic
        # ...
        return retrieved_docs
```

### Custom Generators

To implement a custom generator, extend the `BaseGenerator` class:

```python
from generation.rag_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    def __init__(self, model_name=config.RAG_MODEL_NAME):
        super().__init__(model_name)
        
    def generate(self, question, documents, temperature=config.RAG_TEMPERATURE, max_tokens=config.RAG_MAX_TOKENS):
        # Implement custom generation logic
        # ...
        return result
```

## Troubleshooting

### API Key Issues

If you experience authentication errors, check that:
- Your API keys are correctly set in the `.env` file
- The API keys have sufficient permissions and credits

### Response Validation Issues

If too many responses are being filtered out during validation:
- Lower the `min_confidence` parameter when running the pipeline (e.g., `--min-confidence 0.4`)
- Check if your document corpus contains relevant information for the questions being asked
- Ensure your questions are aligned with the domain of your document corpus

### Empty Dataset Issues

If you encounter errors related to empty datasets:
- The system has been updated to handle cases where all responses are filtered out
- Check the validation logs to see why responses are being marked as invalid

### Vector Database Connection

If retrieval fails:
- Ensure your vector database server is running
- Check connection URLs and authentication
- Verify the collection/index exists and contains documents

### Memory Issues

For large datasets or models:
- Use quantization (4-bit or 8-bit) for training
- Reduce batch size and increase gradient accumulation steps
- Use a model with fewer parameters

## Contact and Support

For questions and support, please open an issue on the repository or contact the maintainers at:
- Email: support@teacherforge.ai
- Discord: [TeacherForge Community](https://discord.gg/teacherforge)
