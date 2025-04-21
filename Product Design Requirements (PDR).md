# 🧾 Product Design Requirements (PDR)

## 📛 Project Name:
**RAG-Generated Dataset Engine (RGDE)**  
*Internal codename: “TeacherForge”*

---

## 🧩 Objective

To build a modular, automated pipeline that uses a RAG (Retrieval-Augmented Generation) system as a **teacher** to generate high-quality, domain-grounded training datasets. These datasets will be used to fine-tune or LoRA-train smaller, cost-effective **student** models.

The system will process prompts, retrieve relevant documents, synthesize grounded answers via the RAG model, and curate instruction-style Q&A datasets suitable for supervised fine-tuning.

---

## 🎯 Goals

- Automate synthetic dataset creation using RAG-based retrieval and generation.
- Maintain traceability between input question, retrieved documents, and generated output.
- Enable metadata tagging and dataset filtering for better slicing.
- Support downstream training (e.g., LoRA/QLoRA) using Hugging Face and PEFT tooling.
- Build with future extensibility toward a self-improving RL-style loop.

---

## 📐 System Components

### 1. **Prompt Generator Module (optional)**
- Generates a list of diverse, domain-specific questions.
- Can be manual input, extracted from logs, or LLM-generated.
- Output: `questions.jsonl` or DB collection.

### 2. **Retriever (RAG Input Stage)**
- Accepts a question.
- Queries vector database (e.g., FAISS, Qdrant, Weaviate).
- Retrieves top-K relevant passages.
- Output: `retrieved_documents`, `retrieval_score`, etc.

### 3. **Generator (RAG Output Stage)**
- Uses a large language model (e.g., GPT-4, Claude, LLaMA with RAG) to generate an answer.
- Inputs: `question + retrieved context`
- Outputs: Structured object including:
  - `question`
  - `retrieved_documents`
  - `generated_response`
  - `generation_metadata` (e.g., model version, timestamp)

### 4. **Post-Processing & Validation Module**
- Removes hallucinated answers or low-quality generations.
- Optional: Grade response quality using another LLM.
- Supports manual review or active learning feedback.
- Adds metadata such as:
  - `confidence_score`
  - `is_valid` flag
  - `tags` (domain, difficulty, etc.)

### 5. **Dataset Builder**
- Compiles final `{prompt, response}` entries.
- Formats in instruction-tuning friendly structure:
```json
{
  "messages": [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation..."}
  ]
}
```
- Output: JSONL or HF `datasets.Dataset` object.

### 6. **Training Pipeline (Student Model)**
- Uses the final dataset to train a smaller student model via:
  - Full fine-tuning or PEFT (LoRA/QLoRA).
- Framework: HuggingFace Transformers, PEFT, Datasets, TRL/Axolotl.
- Includes loss tracking, logging (e.g., Weights & Biases), and checkpointing.

---

## 🔁 Optional Feedback Loop (Future Scope)
- Reinsert student model into RAG pipeline for comparative evaluation.
- Self-improvement loop: student challenges teacher’s answers, proposes refinements.
- Reward models or RLHF-compatible reward signal loop.

---

## 🧱 Stack & Tooling

| Component | Suggested Tools |
|----------|-----------------|
| RAG Model | OpenAI GPT-4, Claude, LLaMA + custom retriever |
| Vector Store | FAISS, Weaviate, Qdrant |
| Dataset | Hugging Face Datasets (`datasets`), JSONL |
| Training | Axolotl, TRL, PEFT, Transformers |
| Logging & Tracking | Weights & Biases (W&B), MLflow |
| API Serving (optional) | FastAPI, Streamlit, Gradio |
| Orchestration | Airflow, LangChain, Prefect (optional) |

---

## 📦 Deliverables

| Phase | Output |
|-------|--------|
| Phase 1 | Prompt generator + retrieval + RAG response logging |
| Phase 2 | Post-processing + dataset formatter |
| Phase 3 | Training pipeline for LoRA student |
| Phase 4 | Optional: Streamlit interface for review and export |

---

## 📁 Directory Structure (Proposed)

```
/teacherforge/
├── prompts/
│   └── prompts.jsonl
├── retrieval/
│   └── retriever.py
├── generation/
│   └── rag_generator.py
├── postprocessing/
│   └── cleaner.py
├── dataset/
│   └── build_dataset.py
├── training/
│   └── train_student.py
├── outputs/
│   └── dataset.jsonl
│   └── logs/
│   └── checkpoints/
```

---

## 🔐 Security & Privacy Considerations
- Ensure sensitive content in retrieved docs is handled appropriately.
- Option to mask/redact sensitive entities in final datasets.
- Version all generations for auditability and rollback.

---

## ✅ Success Criteria

- 1,000–10,000 high-quality question/response pairs generated, with metadata.
- Dataset can train a student model to match 80–90% of the teacher's performance on held-out prompts.
- System is modular and can be expanded into a feedback loop later.

