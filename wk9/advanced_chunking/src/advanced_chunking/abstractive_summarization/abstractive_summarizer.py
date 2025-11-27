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
This module contains the abstractive summarizer.
"""
import openai
import instructor
from pydantic import BaseModel, Field
from typing import List
import os
import tiktoken
import time
import tempfile
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from advanced_chunking import config

model=config["abstractive_summarization"]["model"] 
chunk_size=config["abstractive_summarization"]["chunk_size"]
summary_size=config["abstractive_summarization"]["summary_size"]

# Read the COSTAR summarizer prompt
with open("prompts/summarizer.md", "r") as f:
    summarizer_prompt_template = f.read()

class Summary(BaseModel):
    summary: str = Field(description= f'''
    A concise summary of the text. 
    The summary should be between {summary_size - 100} and {summary_size + 100} tokens.
    ''')

# Modify below to not use base_url if you are using OpenAI GPT models.
#client = instructor.patch(openai.OpenAI(base_url="http://localhost:11434/v1"), mode=Mode.JSON)
client = instructor.patch(openai.OpenAI())

# Create a Docling document
converter = DocumentConverter()
converter.convert

# Chunk the text using Docling
chunker = HybridChunker(max_tokens=chunk_size)

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

def combine_summaries(summaries: List[str]) -> str:
    """
    Combine the summaries into a final summary.
    Args:
        summaries: The summaries to combine
    Returns:
        The combined summary
    """
    return "\n".join(summaries)

def summarize_chunk(chunk: str, model: str) -> Summary:
    """
    Summarize a chunk of text using the LLM model specified above.
    Args:
        chunk: The chunk of text to summarize
        model: The LLM model to use
    Returns:
        The summary of the chunk
    """
    # Format the COSTAR prompt with token range values
    min_tokens = summary_size - 100
    max_tokens = summary_size + 100
    system_prompt = summarizer_prompt_template.format(
        min_tokens=min_tokens,
        max_tokens=max_tokens
    )
    result = client.chat.completions.create(
        model=model,
        response_model = Summary,
        messages=[
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": chunk}
        ],
        #max_completion_tokens=max_tokens + 200,  # Use max_completion_tokens for better compatibility
    )
    
    # Validate token count and retry if too short
    summary_text = result.summary
    actual_tokens = count_tokens(summary_text)
    if actual_tokens < min_tokens:
        print(f"Warning: Summary only has {actual_tokens} tokens (target: {min_tokens}-{max_tokens}). The model may not be following token requirements.")
    
    return result

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

def split_into_token_chunks(text: str, chunk_size: int = chunk_size) -> List[str]:
    """
    Split text into chunks of specified token size.
    Args:
        text: The text to split into chunks
        chunk_size: The size of the chunks
    Returns:
        A list of chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def summarize_files(
    input_folder: str = "data/input", 
    output_folder: str = "data/summaries", 
    model: str = model) -> None:
    """
    For all the files in the folder, read the file and perform the following steps:
    1. Semantically chunk the file into chunks of size roughly {chunk_size} tokens.  Use Docling HybridChunker to do this.
    2. Summarize each chunk using the LLM model specified above.
    3. Combine the summaries into a final summary.
    4. If the final summary is bigger than {summary_size * 2} tokens, then recursively repeat the process.
    5. Write one summary for each file in the secified output folder
    Args:
        input_folder: The folder containing the files to summarize
        output_folder: The folder to write the summaries to
        model: The LLM model to use
    Returns:
        None
    """

    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        with open(file_path, "r") as f:
            text = f.read()
        final_summary = text
        print(f"Processing file {file} with {count_tokens(text)} tokens")
        summary_level = 1
        while count_tokens(text) > summary_size * 2: 
            print(f"Summarizing text with {count_tokens(text)} tokens")
            # Split the text into chunks of {chunk_size} tokens each
            #chunks = split_into_token_chunks(text, chunk_size)
            if count_tokens(text) < chunk_size * 2:
                chunks = [text]
                print(f"Text is less than {chunk_size * 2} tokens, so no splitting needed")
            else:
                chunks = split_into_docling_chunks(text)
                print(f"Split into {len(chunks)} chunks")
            summaries = []
            chunking_happened = False
            for chunk in chunks:
                try:
                    if count_tokens(chunk) < summary_size:
                        print(f"Chunk is less than {summary_size} tokens, so no summarizing needed")
                        summaries.append(chunk)
                    else:
                        print(f"Summarizing chunk with {count_tokens(chunk)} tokens")
                        summary = summarize_chunk(chunk, model).summary
                        summaries.append(summary)
                        print(f"Summarized chunk with {count_tokens(summary)} tokens")
                        chunking_happened = True
                    time.sleep(1)
                except Exception as e:
                    print(f"Error summarizing chunk: {e}")
                    continue
                if len(summaries) % 10 == 0:
                    print(f"Summarized {len(summaries)} chunks")

            if not chunking_happened:
                print("No chunking happened, so summarizing the text and breaking the loop")
                final_summary = summarize_chunk(text, model).summary
                print(f"Summarized full text with {count_tokens(final_summary)} tokens")
                break

            print(f"Summarized {len(summaries)} chunks")
            final_summary = combine_summaries(summaries)
            print(f"Final summary has {count_tokens(final_summary)} tokens")
            if len(summaries) <=1:
                print("No further summarization needed, breaking the loop")
                break
            text = final_summary
            
            # Adding intermediate summary files at each summary level also to the output folder
            summary_file_name = file.replace(".txt", f"_summary_level_{summary_level}.txt")
            with open(os.path.join(output_folder, summary_file_name), "w") as f:
                f.write(final_summary)
            summary_level += 1

        # Write the final summary to the output folder
        with open(os.path.join(output_folder, file), "w") as f:
            f.write(final_summary)

if __name__ == "__main__":
    summarize_files()