# Advanced Chunking

The techniques described here aim to enhance retrieval-augmented generation (RAG) systems through intelligent document decomposition and content generation.

## Overview

Advanced Chunking provides tools for breaking down documents into meaningful chunks and generating various types of content from those chunks, including:

- **Abstractive Summarization**: Generate concise summaries from document chunks
- **Question-Answer Generation**: Create natural Q&A pairs for RAG systems
- **Factoid Extraction**: Decompose text into atomic, context-independent propositions

The library leverages modern LLM models and the Docling framework for intelligent document processing and chunking strategies.

## Features

- **Hybrid Chunking**: Uses Docling's HybridChunker for optimal text segmentation
- **LLM Integration**: Seamless integration with OpenAI models via Instructor
- **Structured Output**: Pydantic models ensure consistent, validated responses
- **Configurable**: YAML-based configuration for easy customization
- **Token Management**: Built-in token counting and chunk size optimization

## Architecture

The project is organized into three main modules:

```
src/advanced_chunking/
├── abstractive_summarization/
│   └── abstractive_summarizer.py
├── q_and_a_generation/
│   └── qa_generator.py
├── factoid_generation/
│   └── factoid_generator.py
└── __init__.py
```

### Core Components

1. **Abstractive Summarizer**: Generates concise summaries from document chunks
2. **QA Generator**: Creates question-answer pairs for RAG system indexing
3. **Factoid Generator**: Extracts atomic propositions using COSTAR methodology

## Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key
- UV package manager (recommended)

### Setup

1. **Download the repository**:
   ```bash
   cd advanced_chunking
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Add your OpenAI API key to .env
   OPENAI_API_KEY=your_api_key_here
   ```

## Configuration

The library uses a YAML configuration file (`config.yaml`) to manage settings:

```yaml
abstractive_summarization:
  model: gpt-4o-mini
  chunk_size: 20000
  summary_size: 2000

factoid_generation:
  model: gpt-4o-mini
  chunk_size: 1000

qa_generation:
  model: gpt-4o-mini
  chunk_size: 1000
```

### Configuration Options

- **model**: OpenAI model to use for generation
- **chunk_size**: Maximum tokens per chunk
- **summary_size**: Maximum tokens for generated summaries

## Usage

### Abstractive Summarization

```python
from advanced_chunking.abstractive_summarization.abstractive_summarizer import (
    generate_summary_from_files,
    generate_summary_from_text
)

# Summarize from text
summary = generate_summary_from_text("Your long text here...")

# Summarize from files
summary = generate_summary_from_files(["file1.txt", "file2.txt"])
```

### Question-Answer Generation

```python
from advanced_chunking.q_and_a_generation.qa_generator import (
    generate_qa_pairs_from_files,
    generate_qa_pairs_from_text
)

# Generate QA pairs from text
qa_pairs = generate_qa_pairs_from_text("Your text content...")

# Generate QA pairs from files
qa_pairs = generate_qa_pairs_from_files(["document1.txt", "document2.txt"])
```

### Factoid Generation

```python
from advanced_chunking.factoid_generation.factoid_generator import (
    generate_factoids_from_files,
    generate_factoids_from_text
)

# Extract factoids from text
factoids = generate_factoids_from_text("Your text content...")

# Extract factoids from files
factoids = generate_factoids_from_files(["document1.txt", "document2.txt"])
```

## Prompts

The library uses carefully crafted prompts for optimal content generation:

### Question-Answer Generator Prompt

Located in `prompts/question_answer_generator.md`, this prompt guides the LLM to:
- Generate diverse, accurate QA pairs
- Cover different levels of granularity
- Ensure answers are standalone and interpretable
- Produce natural, information-seeking questions

### Propositioner Prompt

Located in `prompts/propositioner.md`, this COSTAR-based prompt:
- Extracts atomic, context-independent factoids
- Maintains original phrasing where possible
- Ensures each factoid is interpretable out of context
- Uses neutral, factual language

## Dependencies

- **svlearn-bootcamp**: Core framework and utilities
- **tiktoken**: Token counting and management
- **docling**: Document processing and chunking
- **instructor**: Structured LLM outputs
- **openai**: OpenAI API integration
- **pydantic**: Data validation and serialization

## Development

### Project Structure

```
advanced_chunking/
├── src/advanced_chunking/          # Source code
├── prompts/                        # LLM prompts
├── tests/                          # Test suite
├── docs/                           # Documentation
├── config.yaml                     # Configuration
├── pyproject.toml                  # Project metadata
└── README.md                       # This file
```

### Building Documentation

```bash
./build_docs.sh
./serve_docs.sh
```

## API Reference

### Abstractive Summarizer

- `generate_summary_from_text(text: str) -> str`: Generate summary from text
- `generate_summary_from_files(file_paths: List[str]) -> str`: Generate summary from files
- `summarize_chunk(chunk: str, model: str) -> Summary`: Summarize individual chunk

### QA Generator

- `generate_qa_pairs_from_text(text: str) -> List[QAPair]`: Generate QA pairs from text
- `generate_qa_pairs_from_files(file_paths: List[str]) -> List[QAPair]`: Generate QA pairs from files
- `extract_qa_pairs_from_chunk(chunk: str) -> QAPairs`: Extract QA pairs from chunk

### Factoid Generator

- `generate_factoids_from_text(text: str) -> List[str]`: Extract factoids from text
- `generate_factoids_from_files(file_paths: List[str]) -> List[str]`: Extract factoids from files
- `extract_factoids_from_chunk(chunk: str) -> List[str]`: Extract factoids from chunk

## Examples

### Basic Text Processing

```python
from advanced_chunking import config
from advanced_chunking.abstractive_summarization.abstractive_summarizer import generate_summary_from_text

# Generate a summary
text = "Your long document text here..."
summary = generate_summary_from_text(text)
print(f"Summary: {summary}")
```

### Batch File Processing

```python
from advanced_chunking.factoid_generation.factoid_generator import generate_factoids_from_files

# Process multiple files
files = ["document1.txt", "document2.txt", "document3.txt"]
factoids = generate_factoids_from_files(files)

for i, factoid in enumerate(factoids):
    print(f"Factoid {i+1}: {factoid}")
```

