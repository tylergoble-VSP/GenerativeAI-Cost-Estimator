"""
Pipeline module for high-level orchestration of the LLM cost estimation workflow.

This module provides functions to:
- Load documents (text and PDF files)
- Chunk text using different strategies
- Embed chunks with full token and time tracking
- Generate questions from chunks
- Save results to disk
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .ollama_client import generate, get_embedding
from .timing_metrics import MetricsStore, TimingContext
from .token_accounting import count_tokens, estimate_embedding_tokens


def load_document(filepath: Path) -> str:
    """
    Load a document from a file (supports .txt and .pdf).
    
    This function automatically detects the file type based on the extension
    and uses the appropriate method to extract text.
    
    Args:
        filepath: Path to the document file (.txt or .pdf)
    
    Returns:
        String containing the full text of the document
    
    Raises:
        ValueError: If file extension is not .txt or .pdf
        FileNotFoundError: If the file doesn't exist
    """
    # Convert to Path object if it's a string
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Get file extension (lowercase for case-insensitive comparison)
    extension = filepath.suffix.lower()
    
    # Handle text files
    if extension == ".txt":
        # Open file in read mode with UTF-8 encoding
        # UTF-8 handles most text characters including special characters
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read entire file content
            return f.read()
    
    # Handle PDF files
    elif extension == ".pdf":
        try:
            # Try using pypdf first (simpler, more common)
            import pypdf
            
            # Open PDF file in read-binary mode ('rb')
            with open(filepath, 'rb') as f:
                # Create PDF reader object
                pdf_reader = pypdf.PdfReader(f)
                
                # Extract text from each page and join with newlines
                text_parts = []
                for page in pdf_reader.pages:
                    # Extract text from this page
                    text_parts.append(page.extract_text())
                
                # Join all pages with newlines between them
                return "\n".join(text_parts)
        
        except ImportError:
            # If pypdf not available, try pdfplumber
            try:
                import pdfplumber
                
                # Open PDF file
                with pdfplumber.open(filepath) as pdf:
                    # Extract text from each page
                    text_parts = []
                    for page in pdf.pages:
                        text_parts.append(page.extract_text())
                    
                    # Join all pages
                    return "\n".join(text_parts)
            
            except ImportError:
                # Neither library available
                raise ImportError(
                    "PDF support requires either 'pypdf' or 'pdfplumber'. "
                    "Install one with: pip install pypdf"
                )
    
    else:
        # Unsupported file type
        raise ValueError(f"Unsupported file type: {extension}. Only .txt and .pdf are supported.")


def chunk_text_fixed_window(text: str, chunk_size_tokens: int, 
                           overlap_tokens: int, 
                           tokenizer_model: str = "google/gemma-2-2b") -> List[Dict[str, any]]:
    """
    Chunk text using a fixed token window with overlap.
    
    This strategy splits text into chunks of approximately the same size
    (in tokens), with some overlap between chunks to maintain context.
    
    Args:
        text: The full text to chunk
        chunk_size_tokens: Target size of each chunk in tokens
        overlap_tokens: Number of tokens to overlap between chunks
        tokenizer_model: Model name for tokenization
    
    Returns:
        List of dictionaries, each containing:
            - 'text': The chunk text
            - 'chunk_id': Unique identifier (e.g., 'chunk_0')
            - 'start_char': Character position where chunk starts
            - 'end_char': Character position where chunk ends
            - 'token_count': Number of tokens in this chunk
    """
    chunks = []
    chunk_id = 0
    
    # Split text into sentences for better chunking
    # This regex splits on sentence endings (. ! ?) followed by whitespace
    sentences = re.split(r'([.!?]\s+)', text)
    
    # Rejoin sentences with their punctuation (split keeps the delimiters)
    # We pair each sentence with its following punctuation
    sentence_parts = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_parts.append(sentences[i] + sentences[i + 1])
        else:
            sentence_parts.append(sentences[i])
    if len(sentences) % 2 == 1:
        sentence_parts.append(sentences[-1])
    
    current_chunk_text = ""
    current_chunk_start = 0
    char_position = 0
    
    for sentence in sentence_parts:
        # Try adding this sentence to current chunk
        test_chunk = current_chunk_text + sentence
        test_tokens = count_tokens(test_chunk, tokenizer_model)
        
        # If adding this sentence would exceed chunk size, save current chunk
        if test_tokens > chunk_size_tokens and current_chunk_text:
            # Save the current chunk
            chunk_tokens = count_tokens(current_chunk_text, tokenizer_model)
            chunks.append({
                "text": current_chunk_text.strip(),
                "chunk_id": f"chunk_{chunk_id}",
                "start_char": current_chunk_start,
                "end_char": char_position,
                "token_count": chunk_tokens
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            # Find where to start based on overlap_tokens
            overlap_text = ""
            overlap_chars = 0
            for prev_sentence in reversed(sentence_parts[:sentence_parts.index(sentence)]):
                test_overlap = prev_sentence + overlap_text
                if count_tokens(test_overlap, tokenizer_model) >= overlap_tokens:
                    overlap_text = test_overlap
                    break
                overlap_text = prev_sentence + overlap_text
                overlap_chars += len(prev_sentence)
            
            current_chunk_text = overlap_text + sentence
            current_chunk_start = char_position - overlap_chars
        else:
            # Add sentence to current chunk
            current_chunk_text += sentence
        
        char_position += len(sentence)
    
    # Don't forget the last chunk
    if current_chunk_text.strip():
        chunk_tokens = count_tokens(current_chunk_text, tokenizer_model)
        chunks.append({
            "text": current_chunk_text.strip(),
            "chunk_id": f"chunk_{chunk_id}",
            "start_char": current_chunk_start,
            "end_char": char_position,
            "token_count": chunk_tokens
        })
    
    return chunks


def chunk_text_paragraph_based(text: str) -> List[Dict[str, any]]:
    """
    Chunk text by paragraphs.
    
    This strategy splits text at paragraph boundaries (double newlines).
    Each paragraph becomes a chunk. This preserves semantic units better
    than fixed windows, but chunks may vary significantly in size.
    
    Args:
        text: The full text to chunk
    
    Returns:
        List of dictionaries, each containing:
            - 'text': The chunk text (paragraph)
            - 'chunk_id': Unique identifier (e.g., 'chunk_0')
            - 'start_char': Character position where chunk starts
            - 'end_char': Character position where chunk ends
            - 'token_count': Number of tokens in this chunk
    """
    chunks = []
    
    # Split text by double newlines (paragraph breaks)
    # Strip whitespace from each paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    char_position = 0
    
    for idx, paragraph in enumerate(paragraphs):
        # Count tokens in this paragraph
        token_count = count_tokens(paragraph)
        
        # Create chunk dictionary
        start_char = char_position
        end_char = char_position + len(paragraph)
        
        chunks.append({
            "text": paragraph,
            "chunk_id": f"chunk_{idx}",
            "start_char": start_char,
            "end_char": end_char,
            "token_count": token_count
        })
        
        # Update character position (add paragraph length plus 2 for \n\n)
        char_position = end_char + 2
    
    return chunks


def chunk_text(text: str, config: Config) -> List[Dict[str, any]]:
    """
    Chunk text using the strategy specified in config.
    
    This is the main chunking function that delegates to the appropriate
    strategy based on the configuration.
    
    Args:
        text: The full text to chunk
        config: Configuration object with chunking strategy and parameters
    
    Returns:
        List of chunk dictionaries (see chunk_text_fixed_window or chunk_text_paragraph_based)
    """
    strategy = config.chunking_strategy
    
    if strategy == "fixed_token_window":
        # Use fixed token window strategy
        return chunk_text_fixed_window(
            text,
            config.chunk_size_tokens,
            config.chunk_overlap_tokens
        )
    
    elif strategy == "paragraph_based":
        # Use paragraph-based strategy
        return chunk_text_paragraph_based(text)
    
    elif strategy == "both":
        # Use both strategies and return combined results
        # Mark each chunk with its strategy
        fixed_chunks = chunk_text_fixed_window(
            text,
            config.chunk_size_tokens,
            config.chunk_overlap_tokens
        )
        para_chunks = chunk_text_paragraph_based(text)
        
        # Add strategy marker to each chunk
        for chunk in fixed_chunks:
            chunk["strategy"] = "fixed_token_window"
        for chunk in para_chunks:
            chunk["strategy"] = "paragraph_based"
        
        return fixed_chunks + para_chunks
    
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def embed_chunks(chunks: List[Dict[str, any]], config: Config,
                metrics_store: MetricsStore) -> List[Dict[str, any]]:
    """
    Embed all chunks using Ollama, with full token and time tracking.
    
    This function:
    1. Calls Ollama's embedding API for each chunk
    2. Tracks how long each call takes
    3. Counts tokens for each chunk
    4. Stores all metrics in the metrics_store
    
    Args:
        chunks: List of chunk dictionaries (from chunk_text)
        config: Configuration object with model names and endpoint
        metrics_store: MetricsStore to record timing and token metrics
    
    Returns:
        List of chunk dictionaries with added 'embedding' field containing the vector
    """
    embedded_chunks = []
    
    # Process each chunk
    for chunk in chunks:
        chunk_text = chunk["text"]
        chunk_id = chunk["chunk_id"]
        
        # Count input tokens before making the API call
        input_tokens = estimate_embedding_tokens(chunk_text)
        
        # Time the embedding call
        with TimingContext() as timer:
            # Call Ollama API to get embedding
            embedding, metadata = get_embedding(
                chunk_text,
                config.embedding_model,
                config.ollama_endpoint
            )
        
        # Get the duration from the timer
        duration = timer.duration
        
        # Get embedding size (length of the vector)
        embedding_size = len(embedding) if embedding else None
        
        # Add embedding to chunk dictionary
        chunk_with_embedding = chunk.copy()
        chunk_with_embedding["embedding"] = embedding
        chunk_with_embedding["embedding_size"] = embedding_size
        
        # Record metrics
        metrics_store.add_embedding_metric(
            duration=duration,
            input_tokens=input_tokens,
            chunk_id=chunk_id,
            embedding_size=embedding_size
        )
        
        embedded_chunks.append(chunk_with_embedding)
    
    return embedded_chunks


def generate_questions(chunk: Dict[str, any], num_questions: int, config: Config,
                      metrics_store: MetricsStore,
                      question_prefix: str = "q") -> List[Dict[str, any]]:
    """
    Generate questions from a chunk using the generation model.
    
    This function asks the model to generate questions about the chunk content.
    Useful for creating a question-answer dataset or testing the system.
    
    Args:
        chunk: Chunk dictionary with 'text' field
        num_questions: How many questions to generate
        config: Configuration object with model names and endpoint
        metrics_store: MetricsStore to record timing and token metrics
        question_prefix: Prefix for question IDs (default: 'q')
    
    Returns:
        List of question dictionaries, each containing:
            - 'question_id': Unique identifier
            - 'question_text': The generated question
            - 'chunk_id': Which chunk this question is about
            - 'response_text': Full response from model
    """
    # Build the prompt to ask the model to generate questions
    # Use a more explicit format to get clean questions
    prompt = f"""Generate exactly {num_questions} questions about the following text. Output ONLY the questions, one per line, with no explanations, numbering, or prefixes.

Text:
{chunk['text']}

Questions:"""

    # Count prompt tokens
    prompt_tokens = count_tokens(prompt)
    
    # Time the generation call
    with TimingContext() as timer:
        # Call Ollama API to generate response
        response_text, metadata = generate(
            prompt,
            config.generation_model,
            config.ollama_endpoint
        )
    
    # Get the duration
    duration = timer.duration
    
    # Count response tokens
    response_tokens = count_tokens(response_text)
    
    # Parse questions from response with better filtering
    # Split by newlines and process each line
    all_lines = response_text.split('\n')
    question_lines = []
    
    # Common prefixes/patterns to remove
    prefixes_to_remove = [
        r'^question\s*\d*[\.\):]\s*',  # "Question 1:", "Question:", etc.
        r'^q\s*\d*[\.\):]\s*',          # "Q1:", "Q:", etc.
        r'^\d+[\.\)]\s*',                # "1.", "2)", etc.
        r'^-\s*',                        # "- Question"
        r'^\*\s*',                       # "* Question"
    ]
    
    # Words/phrases that indicate non-question lines
    skip_patterns = [
        r'^here are',
        r'^based on',
        r'^the following',
        r'^questions?:',
        r'^text:',
        r'^answer:',
        r'^note:',
        r'^please',
        r'^output',
        r'^generate',
    ]
    
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip lines that are clearly not questions (instructions, explanations)
        line_lower = line.lower()
        if any(re.match(pattern, line_lower) for pattern in skip_patterns):
            continue
        
        # Remove common prefixes
        cleaned_line = line
        for pattern in prefixes_to_remove:
            cleaned_line = re.sub(pattern, '', cleaned_line, flags=re.IGNORECASE)
        
        cleaned_line = cleaned_line.strip()
        
        # Only keep lines that look like questions (end with ? or are substantial)
        if cleaned_line and (cleaned_line.endswith('?') or len(cleaned_line) > 10):
            question_lines.append(cleaned_line)
    
    # If we didn't get enough questions, try a simpler fallback approach
    # Look for any lines ending with question marks that we might have missed
    if len(question_lines) < num_questions:
        # Get all lines ending with '?' that aren't already in question_lines
        fallback_questions = []
        for line in all_lines:
            line = line.strip()
            if not line or not line.endswith('?'):
                continue
            
            # Clean up prefixes
            cleaned = line
            for pattern in prefixes_to_remove:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
            
            # Skip if it's an instruction or explanation
            cleaned_lower = cleaned.lower()
            if any(re.match(pattern, cleaned_lower) for pattern in skip_patterns):
                continue
            
            # Only add if it's substantial and not already in our list
            if cleaned and len(cleaned) > 10 and cleaned not in question_lines:
                fallback_questions.append(cleaned)
        
        # Add fallback questions to our list
        question_lines.extend(fallback_questions[:num_questions - len(question_lines)])
    
    # Filter out any remaining invalid questions
    question_lines = [q for q in question_lines if q and len(q) > 10 and not any(re.match(pattern, q.lower()) for pattern in skip_patterns)]
    
    # Create question dictionaries
    questions = []
    num_questions_found = min(len(question_lines), num_questions)
    
    # If no questions were found, log a warning but don't crash
    if num_questions_found == 0:
        print(f"Warning: No valid questions extracted from response for chunk {chunk['chunk_id']}")
        print(f"Response was: {response_text[:200]}...")
        return questions
    
    for idx, question_text in enumerate(question_lines[:num_questions]):
        question_id = f"{question_prefix}_{chunk['chunk_id']}_{idx}"
        
        question_dict = {
            "question_id": question_id,
            "question_text": question_text,
            "chunk_id": chunk["chunk_id"],
            "response_text": response_text
        }
        
        # Record metrics for this question
        # Divide duration and tokens by number of questions found
        metrics_store.add_inference_metric(
            duration=duration / num_questions_found,  # Average duration per question
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens // num_questions_found if num_questions_found > 0 else response_tokens,  # Average tokens per question
            question_id=question_id,
            response_text=response_text
        )
        
        questions.append(question_dict)
    
    return questions


def save_chunks(chunks: List[Dict[str, any]], filepath: Path):
    """
    Save chunks to a JSON file.
    
    Note: Embeddings are large, so we might want to save them separately
    or use a more efficient format. For now, we save everything.
    
    This function automatically adds a timestamp to the saved data
    so you know when the chunks were exported.
    
    Args:
        chunks: List of chunk dictionaries
        filepath: Path where to save the chunks
    """
    import json
    from datetime import datetime
    
    # Convert Path to string if needed
    if isinstance(filepath, Path):
        filepath = str(filepath)
    
    # For large embeddings, we might want to save them separately
    # For now, save everything (this could be large!)
    chunks_to_save = []
    for chunk in chunks:
        chunk_copy = chunk.copy()
        # Optionally, we could save embeddings to a separate file
        # For now, include them
        chunks_to_save.append(chunk_copy)
    
    # Create a wrapper dictionary that includes metadata
    # This includes a timestamp so we know when the data was exported
    export_data = {
        "export_timestamp": datetime.now().isoformat(),  # ISO format: "2024-01-15T14:30:45.123456"
        "export_timestamp_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Human-readable: "2024-01-15 14:30:45"
        "num_chunks": len(chunks_to_save),  # Number of chunks in this export
        "chunks": chunks_to_save  # The actual chunk data
    }
    
    # Write to JSON file
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)


def save_metrics(metrics_store: MetricsStore, filepath: Path):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics_store: MetricsStore containing all collected metrics
        filepath: Path where to save the metrics
    """
    metrics_store.export_to_json(filepath)

