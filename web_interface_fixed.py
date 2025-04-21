"""
TeacherForge Web Interface
A Streamlit interface for dataset visualization and management.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset

# Add local modules to path
sys.path.append(str(Path(__file__).parent.absolute()))
import config
from retrieval.document_loaders import load_document, load_documents_from_directory
from retrieval.document_processor import DocumentProcessor
from prompts.generate_prompts import load_questions_from_jsonl

# Page configuration
st.set_page_config(
    page_title="TeacherForge Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #718096;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #ffff99;
        padding: 2px;
        border-radius: 3px;
    }
    .document-card {
        background-color: #f1f5f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        color: #333333;
    }
    /* Fix for text visibility in light-mode */
    .stTextArea textarea, .stTextInput input {
        color: #333333 !important;
    }
    /* Add contrast to dataset viewer cards */
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    /* Make sure text is clearly visible */
    p, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
        color: #f0f2f6;
    }
    /* Higher contrast for document text */
    .document-card p, .document-card h1, .document-card h2, .document-card h3 {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                items.append(json.loads(line.strip()))
        return items
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

def get_projects() -> List[str]:
    """Get list of output directories as projects."""
    try:
        output_dir = Path(config.OUTPUTS_DIR)
        return [d.name for d in output_dir.iterdir() if d.is_dir()]
    except Exception as e:
        st.error(f"Error reading output directory: {e}")
        return []

def format_question_answer(item: Dict) -> Dict:
    """Extract question and answer from dataset item in various formats."""
    question = ""
    answer = ""
    
    if "messages" in item:
        # Chat format
        for message in item["messages"]:
            if message["role"] == "user":
                question = message["content"]
            elif message["role"] == "assistant":
                answer = message["content"]
    elif "instruction" in item:
        # Instruction format
        question = item["instruction"]
        if "input" in item and item["input"]:
            question += f"\n{item['input']}"
        answer = item.get("output", "")
    else:
        # Raw format - look for common keys
        question = item.get("question", "")
        answer = item.get("generated_response", item.get("response", ""))
    
    return {"question": question, "answer": answer, "metadata": item.get("metadata", {}),
            "traceability": item.get("traceability", {})}

# Main App Function
def main():
    # Sidebar
    st.sidebar.markdown("<div class='main-header'>TeacherForge</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sub-header'>Dataset Management</div>", unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Dashboard", "Document Explorer", "Dataset Viewer", "Pipeline Configuration"])
    
    # Dashboard Page
    if page == "Dashboard":
        display_dashboard()
    
    # Document Explorer Page
    elif page == "Document Explorer":
        display_document_explorer()
    
    # Dataset Viewer Page
    elif page == "Dataset Viewer":
        display_dataset_viewer()
    
    # Pipeline Configuration Page
    elif page == "Pipeline Configuration":
        display_pipeline_config()

# Page Functions
def display_dashboard():
    st.markdown("<div class='main-header'>TeacherForge Dashboard</div>", unsafe_allow_html=True)
    st.markdown("### Overview of your RAG-based dataset generation system")
    
    # System Status
    st.subheader("System Status")
    
    col1, col2, col3 = st.columns(3)
    
    # Count number of projects
    projects = get_projects()
    num_projects = len(projects)
    
    # Count documents
    try:
        docs_path = Path(os.getenv("FAISS_DOCUMENTS_PATH", config.FAISS_DOCUMENTS_PATH))
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                num_docs = len(docs)
        else:
            num_docs = 0
    except:
        num_docs = 0
    
    # Count questions
    num_questions = 0
    try:
        prompts_dir = Path(config.PROMPTS_DIR)
        if prompts_dir.exists():
            question_files = list(prompts_dir.glob("*.jsonl"))
            for qf in question_files:
                questions = load_questions_from_jsonl(str(qf))
                num_questions += len(questions)
    except:
        pass
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Documents", num_docs)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Questions", num_questions)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Projects", num_projects)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent Projects
    st.subheader("Recent Projects")
    
    if projects:
        recent_projects_df = pd.DataFrame({
            "Project": projects,
            "Status": ["Complete" for _ in projects]
        })
        st.dataframe(recent_projects_df, use_container_width=True)
    else:
        st.info("No projects found. Run the pipeline to create your first project.")
    
    # Quick Actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Generate Questions", use_container_width=True):
            st.info("Redirecting to Pipeline Configuration...")
            # This is just a placeholder since we can't actually change pages in this demo
    
    with col2:
        if st.button("Process Documents", use_container_width=True):
            st.info("Redirecting to Document Explorer...")
    
    with col3:
        if st.button("Run Pipeline", use_container_width=True):
            st.info("Redirecting to Pipeline Configuration...")

def display_document_explorer():
    st.markdown("<div class='main-header'>Document Explorer</div>", unsafe_allow_html=True)
    st.markdown("### Explore and process your document corpus")
    
    # Document Source
    st.subheader("Document Source")
    doc_source = st.radio("Select Document Source", ["Current Corpus", "Upload New Documents"])
    
    if doc_source == "Current Corpus":
        # Show current documents
        try:
            docs_path = Path(os.getenv("FAISS_DOCUMENTS_PATH", config.FAISS_DOCUMENTS_PATH))
            if docs_path.exists():
                with open(docs_path, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
                
                st.write(f"Found {len(docs)} documents in corpus")
                
                # Show document list
                doc_titles = [doc.get("title", f"Document {i}") for i, doc in enumerate(docs)]
                selected_doc = st.selectbox("Select a document to view", doc_titles)
                
                # Display selected document
                if selected_doc:
                    doc_index = doc_titles.index(selected_doc)
                    doc = docs[doc_index]
                    
                    st.markdown(f"### {doc.get('title', 'Untitled Document')}")
                    st.markdown(f"**Source:** {doc.get('source', 'Unknown')}")
                    
                    # Display metadata if present
                    if "metadata" in doc:
                        st.markdown("**Metadata:**")
                        st.json(doc["metadata"])
                    
                    # Display content
                    st.markdown("**Content:**")
                    st.markdown(f"<div class='document-card'>{doc.get('text', 'No content')}</div>", unsafe_allow_html=True)
            else:
                st.warning(f"Document file not found: {docs_path}")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
    
    else:  # Upload New Documents
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "html", "docx", "csv"])
        
        if uploaded_file:
            # Save file temporarily
            file_path = Path("uploaded_file")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process document
            st.write("Document uploaded. Processing...")
            
            # Chunking options
            st.subheader("Document Processing Options")
            use_chunking = st.checkbox("Enable Smart Chunking", value=True)
            
            if use_chunking:
                col1, col2 = st.columns(2)
                with col1:
                    chunk_size = st.slider("Chunk Size (characters)", 100, 1000, 500)
                with col2:
                    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 250, 100)
                
                # Display stats 
                st.write(f"Document will be processed with chunk size {chunk_size} and overlap {chunk_overlap}")
                
                if st.button("Process Document"):
                    st.info("Document processing initiated (simulated in this demo)")
                    st.success("Document processed and added to corpus")
            
            # Clean up
            file_path.unlink(missing_ok=True)

def display_dataset_viewer():
    st.markdown("<div class='main-header'>Dataset Viewer</div>", unsafe_allow_html=True)
    st.markdown("### Review and analyze generated datasets")
    
    # Dataset Selection
    projects = get_projects()
    
    if not projects:
        st.warning("No projects found. Run the pipeline to create a dataset first.")
        return
    
    selected_project = st.selectbox("Select Project", projects)
    
    if not selected_project:
        return
    
    # Find dataset files in the project
    project_dir = Path(config.OUTPUTS_DIR) / selected_project
    dataset_files = list(project_dir.glob("*.jsonl"))
    
    if not dataset_files:
        st.warning(f"No dataset files found in project {selected_project}")
        return
    
    # Dataset file selection
    dataset_file_options = [f.name for f in dataset_files]
    selected_file = st.selectbox("Select Dataset File", dataset_file_options)
    
    if not selected_file:
        return
    
    # Load dataset
    dataset_path = project_dir / selected_file
    items = load_jsonl(str(dataset_path))
    
    if not items:
        st.warning(f"No items found in dataset {selected_file}")
        return
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Examples", len(items))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Try to extract confidence scores if available
        confidence_scores = []
        for item in items:
            if "traceability" in item and "validation_metadata" in item["traceability"]:
                score = item["traceability"]["validation_metadata"].get("confidence_score")
                if score is not None:
                    confidence_scores.append(float(score))
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Average Confidence", f"{avg_confidence:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Dataset Export
    st.subheader("Export Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_format = st.selectbox(
            "Export Format", 
            ["JSONL (Default)", "JSONL (OpenAI)", "JSONL (Chat)", "JSONL (Instruction)", "CSV", "Parquet", "HuggingFace"]
        )
    
    with col2:
        export_filename = st.text_input(
            "Export Filename", 
            value=f"{selected_project}_{selected_file.split('.')[0]}_export"
        )
    
    with col3:
        create_splits = st.checkbox("Create Train/Test Splits", value=True)
        if create_splits:
            train_ratio = st.slider("Train Split Ratio", 0.5, 0.9, 0.8, 0.05)
    
    if st.button("Export Dataset", use_container_width=True):
        with st.spinner("Exporting dataset..."):
            try:
                # Import the export formats module
                from dataset.export_formats import (
                    convert_to_jsonl, convert_to_csv, 
                    convert_to_parquet, convert_to_hf_dataset
                )
                
                # Determine file extension and export function
                if export_format.startswith("JSONL"):
                    extension = ".jsonl"
                    format_type = "default"
                    
                    if "OpenAI" in export_format:
                        format_type = "openai"
                    elif "Chat" in export_format:
                        format_type = "chat"
                    elif "Instruction" in export_format:
                        format_type = "instruction"
                    
                    export_path = f"exports/{export_filename}{extension}"
                    success, message = convert_to_jsonl(items, export_path, format_type)
                
                elif export_format == "CSV":
                    extension = ".csv"
                    export_path = f"exports/{export_filename}{extension}"
                    success, message = convert_to_csv(items, export_path)
                
                elif export_format == "Parquet":
                    extension = ".parquet"
                    export_path = f"exports/{export_filename}{extension}"
                    success, message = convert_to_parquet(items, export_path)
                
                elif export_format == "HuggingFace":
                    export_path = f"exports/{export_filename}"
                    success, message = convert_to_hf_dataset(
                        items, export_path, 
                        create_splits=create_splits, 
                        train_ratio=train_ratio if create_splits else 0.8
                    )
                
                # Show success or error message
                if success:
                    st.success(message)
                    st.info(f"Exported to: {os.path.abspath(export_path)}")
                else:
                    st.error(message)
            
            except Exception as e:
                st.error(f"Error exporting dataset: {e}")
    
    # Dataset examples
    st.subheader("Dataset Examples")
    
    # Process items
    processed_items = []
    for i, item in enumerate(items):
        processed = format_question_answer(item)
        processed["id"] = i
        processed_items.append(processed)
    
    # Create example selector
    example_ids = [item["id"] for item in processed_items]
    selected_example = st.selectbox("Select Example", example_ids, format_func=lambda x: f"Example {x+1}")
    
    if selected_example is not None:
        example = processed_items[selected_example]
        
        # Display question and answer with improved styling for better contrast
        st.markdown("#### Question")
        st.markdown(f"<div class='document-card' style='background-color: #2c3e50; color: #ffffff; border: 1px solid #4a6282;'>{example['question']}</div>", unsafe_allow_html=True)
        
        st.markdown("#### Answer")
        st.markdown(f"<div class='document-card' style='background-color: #1e3a8a; color: #ffffff; border: 1px solid #3b5cb8;'>{example['answer']}</div>", unsafe_allow_html=True)
        
        # Display traceability information if available
        if example["traceability"]:
            with st.expander("Traceability Information"):
                # Retrieved documents
                if "retrieved_documents" in example["traceability"]:
                    st.markdown("#### Retrieved Documents")
                    docs = example["traceability"]["retrieved_documents"]
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Document {i+1}:** {doc.get('title', 'Untitled')}")
                        st.markdown(f"**Score:** {doc.get('retrieval_score', 'N/A')}")
                        st.markdown(f"**Source:** {doc.get('source', 'Unknown')}")
                        st.markdown("---")
                
                # Validation information
                if "validation_metadata" in example["traceability"]:
                    validation = example["traceability"]["validation_metadata"]
                    st.markdown("#### Validation Metadata")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence Score", validation.get("confidence_score", "N/A"))
                    with col2:
                        st.metric("Valid", str(validation.get("is_valid", "N/A")))
                    
                    if "explanation" in validation:
                        st.markdown(f"**Explanation:** {validation['explanation']}")
                    
                    if "tags" in validation:
                        st.markdown(f"**Tags:** {', '.join(validation['tags'])}")

def display_pipeline_config():
    st.markdown("<div class='main-header'>Pipeline Configuration</div>", unsafe_allow_html=True)
    st.markdown("### Configure and run the TeacherForge pipeline")
    
    # Configuration form
    with st.form("pipeline_config_form"):
        st.subheader("Question Generation")
        
        col1, col2 = st.columns(2)
        with col1:
            use_existing = st.radio("Questions Source", ["Generate New Questions", "Use Existing Questions"])
        
        with col2:
            questions_file = None
            num_questions = 10
            domain = "general knowledge"
            
            if use_existing == "Use Existing Questions":
                # Find existing question files
                prompts_dir = Path(config.PROMPTS_DIR)
                try:
                    question_files = list(prompts_dir.glob("*.jsonl"))
                    question_file_options = [f.name for f in question_files]
                    
                    if question_file_options:
                        questions_file = st.selectbox("Select Questions File", question_file_options)
                    else:
                        st.warning("No question files found")
                except Exception as e:
                    st.error(f"Error loading question files: {e}")
            else:
                num_questions = st.number_input("Number of Questions", min_value=1, max_value=100, value=10)
                domain = st.text_input("Domain", value="general knowledge")
        
        st.subheader("RAG Configuration")
        
        # Add LLM Provider selection
        llm_provider_options = ["openai", "anthropic", "huggingface", "local"]
        llm_provider = st.selectbox("LLM Provider", llm_provider_options, index=llm_provider_options.index(config.RAG_MODEL_PROVIDER) if config.RAG_MODEL_PROVIDER in llm_provider_options else 0)
        
        # Model selection based on provider
        if llm_provider == "openai":
            model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            model_name = st.selectbox("OpenAI Model", model_options, index=0)
        elif llm_provider == "anthropic":
            model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"]
            model_name = st.selectbox("Anthropic Model", model_options, index=0)
        elif llm_provider == "huggingface":
            model_options = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf", "google/gemma-7b-it"]
            model_name = st.selectbox("HuggingFace Model", model_options, index=0)
            st.warning("Using HuggingFace requires a valid HF_API_TOKEN in your .env file")
        elif llm_provider == "local":
            model_path = st.text_input("Local Model Path", value=config.LOCAL_MODEL_PATH or "")
            backend_options = ["llamacpp", "vllm"]
            backend = st.selectbox("Local Model Backend", backend_options, index=backend_options.index(config.LOCAL_MODEL_BACKEND) if config.LOCAL_MODEL_BACKEND in backend_options else 0)
            st.warning("Local models require appropriate libraries installed (llama-cpp-python or vLLM)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Number of Documents to Retrieve", min_value=1, max_value=20, value=4)
        
        with col2:
            min_confidence = st.slider("Minimum Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        with col3:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(config.RAG_TEMPERATURE), step=0.05)
        
        st.subheader("Dataset Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            dataset_format = st.selectbox("Dataset Format", ["chat", "instruction", "completion"])
        
        with col2:
            project_name = st.text_input("Project Name", value=f"project_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
        
        include_traceability = st.checkbox("Include Traceability Information", value=True)
        filter_responses = st.checkbox("Filter Invalid Responses", value=True)
        create_splits = st.checkbox("Create Train/Val Splits", value=True)
        
        # Submit button
        submit_button = st.form_submit_button("Run Pipeline")
    
    # Handle form submission
    if submit_button:
        # Construct command
        if use_existing == "Use Existing Questions":
            if questions_file:
                try:
                    questions_path = str(Path(config.PROMPTS_DIR) / questions_file)
                    st.write(f"Using existing questions from {questions_path}")
                except Exception as e:
                    st.error(f"Error with questions file: {e}")
                    return
            else:
                st.error("No question file selected")
                return
        else:
            # Generate new questions
            try:
                questions_path = str(Path(config.PROMPTS_DIR) / f"gen_questions_{project_name}.jsonl")
                st.write(f"Will generate {num_questions} questions about {domain}")
            except Exception as e:
                st.error(f"Error preparing questions: {e}")
                return
        
        # Output directory
        try:
            output_dir = str(Path(config.OUTPUTS_DIR) / project_name)
        except Exception as e:
            st.error(f"Error with output directory: {e}")
            return
        
        # Display command
        st.subheader("Pipeline Command")
        
        try:
            if use_existing == "Use Existing Questions":
                command = f"python teacherforge.py run-pipeline --questions {questions_path} --output {output_dir} --top-k {top_k} --min-confidence {min_confidence} --format {dataset_format} --provider {llm_provider} --temperature {temperature}"
            else:
                command = f"python teacherforge.py run-pipeline --questions {questions_path} --output {output_dir} --num {num_questions} --domain \"{domain}\" --top-k {top_k} --min-confidence {min_confidence} --format {dataset_format} --provider {llm_provider} --temperature {temperature}"
            
            if not include_traceability:
                command += " --no-traceability"
            
            if not filter_responses:
                command += " --no-filter"
            
            if not create_splits:
                command += " --no-splits"
            
            st.code(command)
        except Exception as e:
            st.error(f"Error building command: {e}")
            return
        
        # Run Pipeline - would be an actual execution in a real deployment
        st.info("Pipeline execution initiated (simulated in this demo)")
        
        # Show progress bar
        progress_bar = st.progress(0)
        for i in range(101):
            # Update progress bar
            progress_bar.progress(i)
            import time
            time.sleep(0.01)
        
        st.success(f"Pipeline completed successfully! Results saved to {output_dir}")

# Run the app
if __name__ == "__main__":
    main()
