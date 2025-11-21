"""
Configuration module for LLM Cost Estimator.

This module defines the Config class that holds all configuration settings
for the project, including model names, Ollama endpoint, chunking strategies,
and file paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """
    Configuration class that holds all settings for the LLM Cost Estimator.
    
    This class uses dataclasses to automatically generate initialization
    and other methods. It stores:
    - Model names for embeddings and generation
    - Ollama API endpoint
    - Chunking configuration
    - File paths for data and results
    """
    
    # Ollama API endpoint - this is where Ollama is running locally
    # Default is the standard Ollama port
    ollama_endpoint: str = "http://localhost:11434"
    
    # Embedding model name - this is the Gemma model used for creating embeddings
    # Default is 'embeddinggemma' as specified
    embedding_model: str = "embeddinggemma"
    
    # Generation model name - this is the Gemma model used for text generation
    # Using gemma3:1b as a reasonable default for local inference
    generation_model: str = "gemma3:1b"
    
    # Chunk size in tokens - how many tokens each chunk should contain
    # This is used when chunking strategy is 'fixed_token_window'
    chunk_size_tokens: int = 512
    
    # Chunk overlap in tokens - how many tokens should overlap between chunks
    # Overlap helps maintain context across chunk boundaries
    chunk_overlap_tokens: int = 50
    
    # Chunking strategy - determines how we split the document
    # 'fixed_token_window' = split by fixed number of tokens
    # 'paragraph_based' = split by paragraphs
    # 'both' = use both strategies (for comparison)
    chunking_strategy: Literal["fixed_token_window", "paragraph_based", "both"] = "fixed_token_window"
    
    # Base directory for the project - defaults to current directory
    base_dir: Path = Path(".")
    
    # Directory where input documents are stored
    data_dir: Path = Path("data")
    
    # Directory where results (embeddings, metrics) are saved
    results_dir: Path = Path("results")
    
    def __post_init__(self):
        """
        Post-initialization method that runs after the dataclass is created.
        
        This converts string paths to Path objects and ensures directories exist.
        Path objects are easier to work with for file operations.
        """
        # Convert base_dir to Path object if it's a string
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        
        # Convert data_dir to Path object if it's a string
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        # If data_dir is relative, make it relative to base_dir
        elif not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir
        
        # Convert results_dir to Path object if it's a string
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        # If results_dir is relative, make it relative to base_dir
        elif not self.results_dir.is_absolute():
            self.results_dir = self.base_dir / self.results_dir
        
        # Create directories if they don't exist
        # exist_ok=True means don't raise error if directory already exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_embeddings_path(self) -> Path:
        """
        Returns the path where embeddings should be saved.
        
        Returns:
            Path object pointing to embeddings file in results directory
        """
        return self.results_dir / "embeddings.json"
    
    def get_metrics_path(self) -> Path:
        """
        Returns the path where metrics should be saved.
        
        Returns:
            Path object pointing to metrics file in results directory
        """
        return self.results_dir / "metrics.json"
    
    def get_chunks_path(self) -> Path:
        """
        Returns the path where chunk metadata should be saved.
        
        Returns:
            Path object pointing to chunks file in results directory
        """
        return self.results_dir / "chunks.json"

