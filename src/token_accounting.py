"""
Token accounting module for accurate token counting.

This module provides token counting functionality for use with Ollama.
It prioritizes actual token counts from Ollama's usage field, with fallback
to tiktoken or simple estimation methods.

Token counting is important for:
- Estimating costs (many LLM APIs charge per token)
- Ensuring prompts fit within model limits
- Tracking usage accurately

Priority order for token counting:
1. Ollama actual tokens (from usage field in API responses) - most accurate
2. tiktoken estimation - good approximation
3. Simple word-based estimation - fallback when others unavailable
"""

import re
from typing import Any, Dict, Optional, Tuple


def extract_tokens_from_usage(usage: Dict[str, Any], call_type: str) -> Optional[Dict[str, int]]:
    """
    Extract token counts from Ollama's usage field in API responses.
    
    Ollama's API responses include a 'usage' field with actual token counts
    from the model. This function extracts those counts in a standardized format.
    
    For embedding calls, the usage field typically contains:
    - prompt_eval_count: Number of tokens in the input text
    
    For generation/chat calls, the usage field typically contains:
    - prompt_eval_count: Number of tokens in the prompt (input)
    - eval_count: Number of tokens in the generated response (output)
    
    Args:
        usage: Dictionary from Ollama's API response containing usage information
        call_type: Type of call - 'embedding' or 'inference'/'generation'
    
    Returns:
        Dictionary with token counts, or None if usage is missing/malformed:
        - For embeddings: {'input_tokens': int}
        - For inference: {'prompt_tokens': int, 'response_tokens': int, 'total_tokens': int}
        Returns None if usage field is missing or doesn't contain expected fields
    """
    # Check if usage is valid (not None, not empty, and is a dictionary)
    if not usage or not isinstance(usage, dict):
        return None
    
    # Handle embedding calls
    if call_type == "embedding":
        # For embeddings, we typically get prompt_eval_count as input tokens
        prompt_eval_count = usage.get("prompt_eval_count")
        if prompt_eval_count is not None:
            try:
                return {
                    "input_tokens": int(prompt_eval_count)
                }
            except (ValueError, TypeError):
                # If conversion fails, return None
                return None
        return None
    
    # Handle inference/generation calls
    elif call_type in ["inference", "generation"]:
        # For generation, we get both prompt and response token counts
        prompt_eval_count = usage.get("prompt_eval_count")
        eval_count = usage.get("eval_count")
        
        # Both fields should be present for a valid inference call
        if prompt_eval_count is not None and eval_count is not None:
            try:
                prompt_tokens = int(prompt_eval_count)
                response_tokens = int(eval_count)
                return {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens
                }
            except (ValueError, TypeError):
                # If conversion fails, return None
                return None
        return None
    
    # Unknown call type
    return None


def _estimate_tokens_simple(text: str) -> int:
    """
    Estimate tokens using a simple word-based method.
    
    This is a fallback method that doesn't require any external dependencies
    or authentication. It approximates tokens by counting words and punctuation.
    
    The estimation: ~1 token per word + ~0.5 tokens per punctuation mark
    This is a rough approximation - actual token counts may vary.
    
    Args:
        text: The text to estimate tokens for
    
    Returns:
        Integer: Estimated number of tokens
    """
    # Count words (split on whitespace)
    words = len(text.split())
    
    # Count punctuation marks (common punctuation)
    punctuation = len(re.findall(r'[.,!?;:()\[\]{}\-"\']', text))
    
    # Rough estimation: 1 token per word + 0.5 tokens per punctuation
    # Round up to be conservative
    estimated_tokens = words + int(punctuation * 0.5)
    
    return estimated_tokens


def _count_tokens_with_tiktoken(text: str) -> Optional[int]:
    """
    Try to count tokens using tiktoken (if available).
    
    tiktoken is a fast tokenizer that doesn't require authentication.
    We use the cl100k_base encoding which is similar to GPT models.
    While not exactly Gemma's tokenization, it's a good approximation.
    
    Args:
        text: The text to count tokens for
    
    Returns:
        Integer token count if tiktoken is available, None otherwise
    """
    try:
        import tiktoken
        # Use cl100k_base encoding (used by GPT-4, similar tokenization style)
        # This is a reasonable approximation for Gemma
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # tiktoken not installed, return None to use fallback
        return None


def count_tokens(text: str, model_name: Optional[str] = None, token: Optional[str] = None,
                 usage: Optional[Dict[str, Any]] = None, call_type: str = "inference") -> Tuple[int, str]:
    """
    Count the number of tokens in a text string using priority-based fallback chain.
    
    Tokens are the basic units that language models work with.
    A token can be a word, part of a word, or punctuation.
    For example, "Hello world!" might be 3 tokens: ["Hello", " world", "!"]
    
    This function uses a priority-based approach:
    1. Ollama actual tokens (from usage field) - most accurate
    2. tiktoken estimation - good approximation
    3. Simple word-based estimation - fallback
    
    Args:
        text: The text string to count tokens in
        model_name: Not used (kept for compatibility)
        token: Not used (kept for compatibility)
        usage: Optional usage dictionary from Ollama API response
        call_type: Type of call - 'embedding' or 'inference' (default: 'inference')
    
    Returns:
        Tuple of (token_count, method_used) where:
        - token_count: Integer number of tokens
        - method_used: String indicating method - "ollama_actual", "tiktoken", or "simple_estimation"
    
    Example:
        >>> count_tokens("Hello, world!")
        (3, "tiktoken")  # (example - actual count depends on method used)
    """
    # Priority 1: Try to extract from Ollama usage field if provided
    if usage is not None:
        usage_tokens = extract_tokens_from_usage(usage, call_type)
        if usage_tokens is not None:
            # For embeddings, return input_tokens
            if call_type == "embedding" and "input_tokens" in usage_tokens:
                return (usage_tokens["input_tokens"], "ollama_actual")
            # For inference, we need the text to determine which count to use
            # Since we only have text (not prompt/response separately), use total if available
            elif "total_tokens" in usage_tokens:
                return (usage_tokens["total_tokens"], "ollama_actual")
            elif "prompt_tokens" in usage_tokens:
                # If only prompt_tokens available, use that (assumes text is the prompt)
                return (usage_tokens["prompt_tokens"], "ollama_actual")
    
    # Priority 2: Try tiktoken (fast and accurate, no auth needed)
    tiktoken_count = _count_tokens_with_tiktoken(text)
    if tiktoken_count is not None:
        return (tiktoken_count, "tiktoken")
    
    # Priority 3: Fallback to simple estimation
    simple_count = _estimate_tokens_simple(text)
    return (simple_count, "simple_estimation")


def count_prompt_and_response(prompt: str, response: str, 
                              model_name: Optional[str] = None,
                              token: Optional[str] = None,
                              usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Count tokens in both prompt and response separately using priority-based fallback.
    
    This is useful for tracking:
    - How many tokens you sent to the model (prompt tokens)
    - How many tokens the model generated (response tokens)
    - Total tokens used in the interaction
    
    Prioritizes actual tokens from Ollama usage field when available.
    
    Args:
        prompt: The input text sent to the model (user's question/instruction)
        response: The text generated by the model (model's answer)
        model_name: Not used (kept for compatibility)
        token: Not used (kept for compatibility)
        usage: Optional usage dictionary from Ollama API response
    
    Returns:
        Dictionary with keys:
            - 'prompt_tokens': Number of tokens in the prompt
            - 'response_tokens': Number of tokens in the response
            - 'total_tokens': Sum of prompt and response tokens
            - 'prompt_method': Method used for prompt counting
            - 'response_method': Method used for response counting
    
    Example:
        >>> count_prompt_and_response("What is AI?", "AI is artificial intelligence.")
        {'prompt_tokens': 4, 'response_tokens': 6, 'total_tokens': 10, 
         'prompt_method': 'tiktoken', 'response_method': 'tiktoken'}
    """
    # Priority 1: Try to extract from Ollama usage field if provided
    if usage is not None:
        usage_tokens = extract_tokens_from_usage(usage, "inference")
        if usage_tokens is not None and "prompt_tokens" in usage_tokens and "response_tokens" in usage_tokens:
            return {
                "prompt_tokens": usage_tokens["prompt_tokens"],
                "response_tokens": usage_tokens["response_tokens"],
                "total_tokens": usage_tokens["total_tokens"],
                "prompt_method": "ollama_actual",
                "response_method": "ollama_actual"
            }
    
    # Priority 2: Fall back to estimation methods
    # Count tokens in the prompt (input)
    prompt_tokens, prompt_method = count_tokens(prompt, model_name, token=token, call_type="inference")
    
    # Count tokens in the response (output)
    response_tokens, response_method = count_tokens(response, model_name, token=token, call_type="inference")
    
    # Calculate total tokens
    total_tokens = prompt_tokens + response_tokens
    
    # Return as a dictionary for easy access
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "prompt_method": prompt_method,
        "response_method": response_method
    }


def estimate_embedding_tokens(text: str, model_name: Optional[str] = None,
                              token: Optional[str] = None,
                              usage: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
    """
    Count tokens for embedding calls using priority-based fallback.
    
    For embedding models, we typically only count input tokens
    (the text being embedded), not output tokens (the embedding vector).
    
    This function prioritizes actual tokens from Ollama usage field when available.
    
    Args:
        text: The text to embed
        model_name: Not used (kept for compatibility)
        token: Not used (kept for compatibility)
        usage: Optional usage dictionary from Ollama API response
    
    Returns:
        Tuple of (token_count, method_used) where:
        - token_count: Integer number of tokens in the input text
        - method_used: String indicating method - "ollama_actual", "tiktoken", or "simple_estimation"
    """
    return count_tokens(text, model_name, token=token, usage=usage, call_type="embedding")

