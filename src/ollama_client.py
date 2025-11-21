"""
Ollama client module for interacting with Ollama API.

This module provides functions to call Ollama's REST API for:
- Getting embeddings from text
- Generating text responses (chat/completion)

All functions include error handling and return both the result
and any metadata provided by Ollama.
"""

import json
from typing import Any, Dict, List, Tuple

import requests


def get_embedding(text: str, model: str, base_url: str = "http://localhost:11434") -> Tuple[List[float], Dict[str, Any]]:
    """
    Get an embedding vector for the given text using Ollama's embedding API.
    
    An embedding is a numerical representation of text that captures its meaning.
    Similar texts have similar embeddings, which is useful for search and comparison.
    
    Args:
        text: The text string to embed (convert to numbers)
        model: The name of the embedding model to use (e.g., 'embeddinggemma')
        base_url: The base URL where Ollama is running (default: localhost:11434)
    
    Returns:
        A tuple containing:
            - List of floats: The embedding vector (list of numbers representing the text)
            - Dictionary: Metadata from Ollama (may include usage info, model info, etc.)
    
    Raises:
        requests.RequestException: If the API call fails (network error, server error, etc.)
    """
    # Construct the full URL for the embeddings endpoint
    # Ollama's embeddings API is at /api/embeddings
    url = f"{base_url}/api/embeddings"
    
    # Prepare the request payload (data to send to the API)
    # Ollama expects 'model' and 'prompt' fields
    payload = {
        "model": model,  # Which model to use
        "prompt": text   # The text to embed
    }
    
    # Send POST request to Ollama API
    # POST is used because we're sending data (the text to embed)
    response = requests.post(url, json=payload)
    
    # Check if the request was successful
    # HTTP status code 200 means success
    response.raise_for_status()  # Raises an exception if status code indicates error
    
    # Parse the JSON response from Ollama
    # JSON is a text format for structured data
    result = response.json()
    
    # Extract the embedding vector from the response
    # Ollama returns the embedding in the 'embedding' field
    embedding = result.get("embedding", [])
    
    # Create metadata dictionary with any additional info from the response
    # This might include model info, usage stats, etc.
    metadata = {
        "model": result.get("model"),
        "usage": result.get("usage", {}),  # Token usage info if available
        "raw_response": result  # Keep full response for debugging
    }
    
    return embedding, metadata


def generate(prompt: str, model: str, base_url: str = "http://localhost:11434", 
             system: str = None, stream: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a text response using Ollama's chat API.
    
    This function sends a prompt (question or instruction) to the model
    and gets back a generated text response.
    
    Args:
        prompt: The user's question or instruction (what you want the model to respond to)
        model: The name of the generation model to use (e.g., 'gemma3:1b')
        base_url: The base URL where Ollama is running (default: localhost:11434)
        system: Optional system message that sets the model's behavior/role
        stream: Whether to stream the response (get tokens as they're generated)
               For now, we set this to False to get the complete response at once
    
    Returns:
        A tuple containing:
            - String: The generated text response from the model
            - Dictionary: Metadata from Ollama (usage info, model info, etc.)
    
    Raises:
        requests.RequestException: If the API call fails
    """
    # Construct the full URL for the chat endpoint
    # Ollama's chat API is at /api/chat
    url = f"{base_url}/api/chat"
    
    # Build the messages list for the chat API
    # Ollama's chat API expects a list of message objects
    messages = []
    
    # Add system message if provided
    # System messages set the model's behavior (e.g., "You are a helpful assistant")
    if system:
        messages.append({
            "role": "system",  # System role sets the model's behavior
            "content": system   # The system instruction
        })
    
    # Add the user's prompt as a user message
    messages.append({
        "role": "user",    # User role indicates this is from the user
        "content": prompt  # The actual question or instruction
    })
    
    # Prepare the request payload
    payload = {
        "model": model,      # Which model to use
        "messages": messages, # The conversation messages
        "stream": stream     # Whether to stream (False = get full response)
    }
    
    # Send POST request to Ollama API
    response = requests.post(url, json=payload)
    
    # Check if request was successful
    response.raise_for_status()
    
    # Parse the JSON response
    result = response.json()
    
    # Extract the generated text from the response
    # Ollama returns the message in result['message']['content']
    generated_text = result.get("message", {}).get("content", "")
    
    # Create metadata dictionary with additional info
    metadata = {
        "model": result.get("model"),
        "usage": result.get("usage", {}),  # Token usage (prompt tokens, completion tokens, etc.)
        "done": result.get("done", True),  # Whether generation is complete
        "raw_response": result  # Keep full response for debugging
    }
    
    return generated_text, metadata

