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
This module contains the factoid generator.
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
model=config["factoid_generation"]["model"] 
chunk_size=config["factoid_generation"]["chunk_size"]

# Create a Docling document
converter = DocumentConverter()
converter.convert

# Chunk the text using Docling
chunker = HybridChunker(max_tokens=chunk_size)

client = instructor.patch(openai.OpenAI(api_key="ollama",base_url="http://localhost:11434/v1"),
                         mode=Mode.JSON)

class Factoids(BaseModel):
    factoids: List[str] = Field(description= '''
    A list of factoids, each of which is an atomic proposition.
    ''')

# Read the COSTAR prompt
with open("prompts/propositioner.md", "r") as f:
    costar_prompt = f.read()

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

def extract_factoids_from_chunk(chunk: str) -> List[str]:
    """
    Extract factoids from a chunk of text using the COSTAR prompt.
    Args:
        chunk: The chunk of text to extract factoids from
    Returns:
        A list of factoids
    """

    # Add the chunk of text to the COSTAR prompt
    user_prompt = costar_prompt + f"\n\nContent:\n{chunk}"
    # Call the LLM
    return client.chat.completions.create(
        model=model,
        response_model=Factoids,
        messages=[{"role": "user", "content": user_prompt}],
        max_completion_tokens=chunk_size * 5,
        temperature=0.5
    )


def generate_factoids_from_files(
    input_folder: str = "data/input", 
    output_folder: str = "data/factoids") -> None:
    """
    For all files in the input folder, perform the following steps:
    1. Read the file content
    2. Chunk the file into chunks of size roughly {chunk_size} tokens.  Use Docling for this.
    3. Extract factoids from each chunk using the COSTAR prompt.
    4. Save all factoids to a subfolder named after the input file
    
    Args:
        input_folder: The folder containing the files to process
        output_folder: The base folder to write the factoids to
    Returns:
        None
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            
            # Create subfolder for this file's factoids
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
                
                all_factoids = []  # Collect all factoids from all chunks
                
                # Process each chunk to extract factoids
                for i, chunk in enumerate(chunks):
                    try:
                        print(f"Processing chunk {i+1}/{len(chunks)} ({count_tokens(chunk)} tokens)")
                        
                        # Extract factoids from the chunk using the COSTAR prompt
                        chunk_factoids = extract_factoids_from_chunk(chunk)
                        
                        # Save factoids for this chunk as JSONL file
                        factoids_filename = f"chunk_{i+1:03d}_factoids.jsonl"
                        factoids_filepath = os.path.join(file_output_folder, factoids_filename)
                        
                        with open(factoids_filepath, 'w', encoding='utf-8') as f:
                            for factoid in chunk_factoids.factoids:
                                json_line = {"chunk_id": i+1, "factoid": factoid}
                                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
                        
                        # Collect data for combined files
                        for factoid in chunk_factoids.factoids:
                            all_factoids.append({"chunk_id": i+1, "factoid": factoid})
                        
                        print(f"Extracted {len(chunk_factoids.factoids)} factoids from chunk {i+1}")
                        
                        # Rate limiting to avoid API limits
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error processing chunk {i+1}: {e}")
                        continue

                # Save combined factoids file
                combined_factoids_filename = f"{file_base_name}_all_factoids.jsonl"
                combined_factoids_filepath = os.path.join(file_output_folder, combined_factoids_filename)
                
                with open(combined_factoids_filepath, 'w', encoding='utf-8') as f:
                    for factoid_data in all_factoids:
                        f.write(json.dumps(factoid_data, ensure_ascii=False) + '\n')
                
                print(f"Completed processing {filename}. Total factoids: {len(all_factoids)}")
                print(f"Factoids and chunks saved to: {file_output_folder}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

if __name__ == "__main__":
    generate_factoids_from_files()
