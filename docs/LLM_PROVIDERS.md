# TeacherForge LLM Providers

This document details the LLM (Large Language Model) providers available in TeacherForge and how to configure and use them.

## Available Providers

TeacherForge supports the following LLM providers:

1. **OpenAI** (Default)
   - Models: GPT-4, GPT-4-turbo, GPT-3.5-turbo, etc.
   - Best for: High-quality responses with good grounding in retrieved contexts
   - Requires: OpenAI API key

2. **Anthropic**
   - Models: Claude-3-opus, Claude-3-sonnet, Claude-3-haiku, Claude-2.1
   - Best for: Nuanced, safety-aligned responses with good context understanding
   - Requires: Anthropic API key

3. **HuggingFace**
   - Models: Any model available on HuggingFace Inference API
   - Best for: Access to a wide variety of open models
   - Requires: HuggingFace API token

4. **Local Models**
   - Models: Any GGUF format model (for llama.cpp) or supported model (for vLLM)
   - Best for: Privacy, offline usage, or custom fine-tuned models
   - Requires: Model file on local disk and appropriate inference libraries

## Configuration

### Environment Variables

Configure your preferred providers in the `.env` file:

```
# OpenAI (default)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# HuggingFace
HF_API_TOKEN=your_huggingface_token_here

# Global LLM Settings
RAG_MODEL_PROVIDER=openai  # Options: openai, anthropic, huggingface, local
RAG_MODEL_NAME=gpt-4      # Default model for the selected provider
RAG_TEMPERATURE=0.1       # Response temperature (0-1)
RAG_MAX_TOKENS=1000       # Maximum output tokens

# Local Model settings
LOCAL_MODEL_PATH=path/to/your/model.gguf  # Path to local model file
LOCAL_MODEL_BACKEND=llamacpp              # Options: llamacpp, vllm
LOCAL_MODEL_N_GPU_LAYERS=-1               # -1 means use all GPU layers
LOCAL_MODEL_CONTEXT_LENGTH=4096           # Context window size
```

### Command Line

Specify providers when running the pipeline:

```bash
# Using OpenAI
python teacherforge.py run-pipeline --provider openai --model gpt-4

# Using Anthropic
python teacherforge.py run-pipeline --provider anthropic --model claude-3-opus-20240229

# Using HuggingFace
python teacherforge.py run-pipeline --provider huggingface --model mistralai/Mistral-7B-Instruct-v0.2

# Using a local model
python teacherforge.py run-pipeline --provider local --model-path path/to/model.gguf --backend llamacpp
```

### Web Interface

The web interface (launched via `python web_interface_fixed.py`) provides a dropdown menu to select the LLM provider and corresponding model options.

## Provider-Specific Notes

### OpenAI

- Default and most widely tested provider
- Requires OpenAI API key with appropriate usage limits
- Models support function calling for enhanced output parsing (not yet implemented)

### Anthropic

- Claude models excel at following complex instructions
- Requires Anthropic API key
- More focused on safety and alignment

### HuggingFace

- Provides access to open-source models via the Inference API
- Requires a HuggingFace API token with appropriate permissions
- Performance and capabilities vary widely by model

### Local Models

- Requires installing additional dependencies:
  - For llama.cpp: `pip install llama-cpp-python`
  - For vLLM: `pip install vllm`
- CPU inference is possible but slow; GPU recommended for decent performance
- Memory requirements vary based on model size and quantization level

## Adding a New Provider

To add a new LLM provider:

1. Create a new provider class in `generation/llm_providers/` that inherits from `BaseLLMProvider`
2. Implement the required abstract methods
3. Add the provider to the factory function in `provider_factory.py`
4. Update configuration and command-line options as needed

## Testing Providers

You can run the tests to verify provider functionality:

```bash
python test_providers.py
```

This will test all configured providers and report any issues.
