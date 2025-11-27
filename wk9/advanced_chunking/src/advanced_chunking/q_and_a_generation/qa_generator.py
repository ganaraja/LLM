#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
"""
This module contains the question-answer pair generator.
"""
import os
import json
import time
import tiktoken
import tempfile
from typing import List
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import instructor
import openai
from pydantic import BaseModel, Field
from advanced_chunking import config
from instructor.mode import Mode
model = config["qa_generation"]["model"] 
chunk_size = config["qa_generation"]["chunk_size"]

# Create a Docling document
converter = DocumentConverter()
converter.convert

# Chunk the text using Docling
chunker = HybridChunker(max_tokens=chunk_size)

client = instructor.patch(openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1"),
                         mode=Mode.JSON)

class QAPair(BaseModel):
    question: str = Field(description="A natural question that a user might ask when searching for information")
    answer: str = Field(description="A concise and direct answer grounded in the text")

class QAPairs(BaseModel):
    qa_pairs: List[QAPair] = Field(description="A list of question-answer pairs")

# Read the QA generation prompt
with open("prompts/question_answer_generator.md", "r") as f:
    qa_prompt = f.read()

def split_into_docling_chunks(text: str) -> List[str]:
    """
    Split text into chunks of specified token size.
    Args:
        text: The text to split into chunks
    Returns:
        A list of chunks
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_file:
        temp_file.write(text)
        temp_file_path = temp_file.name  # Store the file path
    doc = converter.convert(source=temp_file_path).document
    chunks = list(chunker.chunk(dl_doc=doc))
    os.remove(temp_file_path)
    return [chunk.text for chunk in chunks]  

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text using tiktoken.
    Args:
        text: The text to count the tokens of
    Returns:
        The number of tokens in the text
    """
    try:
        # Use cl100k_base encoding which is commonly used for modern models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate token count (roughly 4 characters per token)
        return len(text) // 4

def extract_qa_pairs_from_chunk(chunk: str) -> QAPairs:
    """
    Extract question-answer pairs from a chunk of text using the QA generation prompt.
    Args:
        chunk: The chunk of text to extract QA pairs from
    Returns:
        A QAPairs object containing the generated QA pairs
    """

    # Add the chunk of text to the QA generation prompt
    user_prompt = qa_prompt + f"\n\nContent:\n{chunk}"
    # Call the LLM
    return client.chat.completions.create(
        model=model,
        response_model=QAPairs,
        messages=[{"role": "user", "content": user_prompt}],
        max_completion_tokens=chunk_size * 5,
        temperature=0.5
    )


def generate_qa_pairs_from_files(
    input_folder: str = "data/input", 
    output_folder: str = "data/qa_pairs") -> None:
    """
    For all files in the input folder, perform the following steps:
    1. Read the file content
    2. Chunk the file into chunks of size roughly {chunk_size} tokens. Use Docling for this.
    3. Extract question-answer pairs from each chunk using the QA generation prompt.
    4. Save all QA pairs to a subfolder named after the input file
    
    Args:
        input_folder: The folder containing the files to process
        output_folder: The base folder to write the QA pairs to
    Returns:
        None
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            
            # Create subfolder for this file's QA pairs
            file_base_name = os.path.splitext(filename)[0]
            file_output_folder = os.path.join(output_folder, file_base_name)
            os.makedirs(file_output_folder, exist_ok=True)
            
            print(f"Processing file: {filename}")
            
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    text = f.read()
                
                print(f"File {filename} has {count_tokens(text)} tokens")
                
                # Split the text into chunks
                chunks = split_into_docling_chunks(text)
                print(f"Split into {len(chunks)} chunks")
                
                # Write chunks file first
                chunks_filename = f"{file_base_name}_chunks.jsonl"
                chunks_filepath = os.path.join(file_output_folder, chunks_filename)
                
                with open(chunks_filepath, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(chunks):
                        chunk_line = {"chunk_id": i+1, "chunk_text": chunk}
                        f.write(json.dumps(chunk_line, ensure_ascii=False) + '\n')
                
                all_qa_pairs = []  # Collect all QA pairs from all chunks
                
                # Process each chunk to extract QA pairs
                for i, chunk in enumerate(chunks):
                    try:
                        print(f"Processing chunk {i+1}/{len(chunks)} ({count_tokens(chunk)} tokens)")
                        
                        # Extract QA pairs from the chunk using the QA generation prompt
                        chunk_qa_pairs = extract_qa_pairs_from_chunk(chunk)
                        
                        # Save QA pairs for this chunk as JSONL file
                        qa_pairs_filename = f"chunk_{i+1:03d}_qa_pairs.jsonl"
                        qa_pairs_filepath = os.path.join(file_output_folder, qa_pairs_filename)
                        
                        with open(qa_pairs_filepath, 'w', encoding='utf-8') as f:
                            for qa_pair in chunk_qa_pairs.qa_pairs:
                                json_line = {"chunk_id": i+1, "question": qa_pair.question, "answer": qa_pair.answer}
                                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
                        
                        # Collect data for combined files
                        for qa_pair in chunk_qa_pairs.qa_pairs:
                            all_qa_pairs.append({"chunk_id": i+1, "question": qa_pair.question, "answer": qa_pair.answer})
                        
                        print(f"Extracted {len(chunk_qa_pairs.qa_pairs)} QA pairs from chunk {i+1}")
                        
                        # Rate limiting to avoid API limits
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error processing chunk {i+1}: {e}")
                        continue

                # Save combined QA pairs file
                combined_qa_pairs_filename = f"{file_base_name}_all_qa_pairs.jsonl"
                combined_qa_pairs_filepath = os.path.join(file_output_folder, combined_qa_pairs_filename)
                
                with open(combined_qa_pairs_filepath, 'w', encoding='utf-8') as f:
                    for qa_pair_data in all_qa_pairs:
                        f.write(json.dumps(qa_pair_data, ensure_ascii=False) + '\n')
                
                print(f"Completed processing {filename}. Total QA pairs: {len(all_qa_pairs)}")
                print(f"QA pairs and chunks saved to: {file_output_folder}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

if __name__ == "__main__":
    generate_qa_pairs_from_files()
