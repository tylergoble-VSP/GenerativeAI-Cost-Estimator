"""
Timing metrics module for tracking call durations and performance.

This module provides:
- TimingContext: A context manager for measuring how long operations take
- MetricsStore: A class to collect and store all timing and token metrics

Uses time.perf_counter() for high-resolution timing measurements.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


class TimingContext:
    """
    Context manager for measuring the duration of code blocks.
    
    A context manager is a Python feature that allows you to use
    the 'with' statement. When you enter the 'with' block, __enter__ runs.
    When you exit (even if there's an error), __exit__ runs.
    
    This makes it easy to time code:
        with TimingContext() as timer:
            # do some work
        duration = timer.duration  # Get how long it took
    
    Attributes:
        start_time: When the timing started (high-resolution timestamp)
        end_time: When the timing ended (None if not finished yet)
        duration: How long it took in seconds (None if not finished)
    """
    
    def __init__(self):
        """
        Initialize the timing context.
        
        Sets start_time to None initially. It will be set when we enter
        the 'with' block.
        """
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        """
        Called when entering the 'with' block.
        
        Records the current time using perf_counter(), which gives
        high-resolution timing (more accurate than time.time()).
        
        Returns:
            self: So we can access the timer object in the 'with' statement
        """
        # Record the start time
        # perf_counter() gives the best available timer resolution
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting the 'with' block (even if there's an error).
        
        Records the end time so we can calculate duration.
        
        Args:
            exc_type: Exception type if an exception occurred (None otherwise)
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        # Record the end time
        self.end_time = time.perf_counter()
    
    @property
    def duration(self) -> Optional[float]:
        """
        Calculate the duration in seconds.
        
        This is a property, which means you can access it like an attribute
        (timer.duration) but it's actually computed by this method.
        
        Returns:
            Duration in seconds, or None if timing hasn't finished
        """
        if self.start_time is None or self.end_time is None:
            return None
        
        # Calculate duration: end time minus start time
        return self.end_time - self.start_time


class MetricsStore:
    """
    Store for collecting and managing all metrics (timing, tokens, etc.).
    
    This class acts like a database for metrics. You can add metrics
    for different operations (embeddings, inference) and then export
    them to files for analysis.
    
    Attributes:
        metrics: List of all metric records collected so far
    """
    
    def __init__(self):
        """
        Initialize an empty metrics store.
        
        Creates an empty list to hold metric records.
        Each record is a dictionary with information about one operation.
        """
        self.metrics: List[Dict[str, Any]] = []
    
    def add_metric(self, call_type: str, duration: float, 
                   token_counts: Dict[str, int],
                   chunk_id: Optional[str] = None,
                   question_id: Optional[str] = None,
                   additional_info: Optional[Dict[str, Any]] = None):
        """
        Add a metric record to the store.
        
        Each metric record represents one operation (embedding call or
        inference call) with its timing and token information.
        
        Args:
            call_type: Type of call - 'embedding' or 'inference'
            duration: How long the call took in seconds
            token_counts: Dictionary with token counts (e.g., {'input_tokens': 100})
            chunk_id: Optional identifier for which chunk this was (for embeddings)
            question_id: Optional identifier for which question this was (for inference)
            additional_info: Optional dictionary with any other relevant information
        """
        # Create a metric record dictionary
        metric = {
            "call_type": call_type,        # What kind of operation
            "duration_seconds": duration,  # How long it took
            "token_counts": token_counts,  # Token usage info
            "timestamp": time.time(),      # When this happened (Unix timestamp)
        }
        
        # Add optional identifiers if provided
        if chunk_id is not None:
            metric["chunk_id"] = chunk_id
        
        if question_id is not None:
            metric["question_id"] = question_id
        
        # Add any additional information if provided
        if additional_info:
            metric.update(additional_info)
        
        # Add this metric to our list
        self.metrics.append(metric)
    
    def add_embedding_metric(self, duration: float, input_tokens: int,
                             chunk_id: str, embedding_size: Optional[int] = None):
        """
        Convenience method to add an embedding metric.
        
        This makes it easier to add embedding metrics without having to
        construct the token_counts dictionary manually.
        
        Args:
            duration: How long the embedding call took in seconds
            input_tokens: Number of tokens in the input text
            chunk_id: Identifier for which chunk was embedded
            embedding_size: Optional size of the embedding vector (number of dimensions)
        """
        # Build token counts dictionary
        token_counts = {
            "input_tokens": input_tokens
        }
        
        # Build additional info dictionary
        additional_info = {}
        if embedding_size is not None:
            additional_info["embedding_size"] = embedding_size
        
        # Use the general add_metric method
        self.add_metric(
            call_type="embedding",
            duration=duration,
            token_counts=token_counts,
            chunk_id=chunk_id,
            additional_info=additional_info if additional_info else None
        )
    
    def add_inference_metric(self, duration: float, prompt_tokens: int,
                            response_tokens: int, question_id: str,
                            response_text: Optional[str] = None):
        """
        Convenience method to add an inference metric.
        
        This makes it easier to add inference metrics without having to
        construct the token_counts dictionary manually.
        
        Args:
            duration: How long the inference call took in seconds
            prompt_tokens: Number of tokens in the prompt (input)
            response_tokens: Number of tokens in the response (output)
            question_id: Identifier for which question this was
            response_text: Optional text of the response (for debugging/analysis)
        """
        # Build token counts dictionary
        token_counts = {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens
        }
        
        # Build additional info dictionary
        additional_info = {}
        if response_text is not None:
            additional_info["response_text"] = response_text
        
        # Use the general add_metric method
        self.add_metric(
            call_type="inference",
            duration=duration,
            token_counts=token_counts,
            question_id=question_id,
            additional_info=additional_info if additional_info else None
        )
    
    def export_to_json(self, filepath: Path) -> None:
        """
        Export all metrics to a JSON file.
        
        JSON (JavaScript Object Notation) is a text format for storing
        structured data. It's human-readable and can be easily loaded
        by other programs.
        
        This function automatically adds a timestamp to the exported data
        so you know when the metrics were exported.
        
        Args:
            filepath: Path where to save the JSON file
        """
        from datetime import datetime
        
        # Convert Path object to string if needed
        if isinstance(filepath, Path):
            filepath = str(filepath)
        
        # Create a wrapper dictionary that includes metadata
        # This includes a timestamp so we know when the data was exported
        export_data = {
            "export_timestamp": datetime.now().isoformat(),  # ISO format: "2024-01-15T14:30:45.123456"
            "export_timestamp_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Human-readable: "2024-01-15 14:30:45"
            "num_metrics": len(self.metrics),  # Number of metric records in this export
            "metrics": self.metrics  # The actual metric data
        }
        
        # Open file for writing ('w' = write mode)
        with open(filepath, 'w') as f:
            # json.dump writes the Python data structure to the file as JSON
            # indent=2 makes it pretty-printed (easier to read)
            json.dump(export_data, f, indent=2)
    
    def export_to_csv(self, filepath: Path) -> None:
        """
        Export all metrics to a CSV file.
        
        CSV (Comma-Separated Values) is a simple text format where each
        line is a record and fields are separated by commas. It's easy
        to open in Excel or pandas.
        
        Note: This is a simplified CSV export. For complex nested data,
        you might want to use pandas DataFrame.to_csv() instead.
        
        This function automatically adds a timestamp column to the CSV
        so you know when each export was created.
        
        Args:
            filepath: Path where to save the CSV file
        """
        from datetime import datetime
        
        if not self.metrics:
            # If no metrics, create an empty file with just timestamp header
            with open(filepath, 'w') as f:
                export_timestamp = datetime.now().isoformat()
                f.write(f"# Export timestamp: {export_timestamp}\n")
                f.write("# No metrics to export\n")
            return
        
        # Convert Path object to string if needed
        if isinstance(filepath, Path):
            filepath = str(filepath)
        
        # Get all unique keys from all metrics (column names)
        # This handles cases where different metrics have different fields
        all_keys = set()
        for metric in self.metrics:
            all_keys.update(metric.keys())
        
        # Add timestamp column to the set of columns
        # This will be the first column in the CSV
        all_keys.add("export_timestamp")
        
        # Sort keys for consistent column order, but put timestamp first
        columns = sorted([k for k in all_keys if k != "export_timestamp"])
        columns = ["export_timestamp"] + columns
        
        # Get the export timestamp (same for all rows in this export)
        export_timestamp = datetime.now().isoformat()
        
        # Open file for writing
        with open(filepath, 'w') as f:
            # Write header row (column names)
            f.write(",".join(columns) + "\n")
            
            # Write each metric as a row
            for metric in self.metrics:
                # Get value for each column, convert to string
                # Use empty string if key doesn't exist in this metric
                # For export_timestamp, use the current timestamp
                values = []
                for col in columns:
                    if col == "export_timestamp":
                        values.append(export_timestamp)
                    else:
                        values.append(str(metric.get(col, "")))
                # Join values with commas and write to file
                f.write(",".join(values) + "\n")
    
    def get_metrics_by_type(self, call_type: str) -> List[Dict[str, Any]]:
        """
        Get all metrics of a specific type.
        
        Useful for filtering metrics (e.g., get all embedding metrics
        separately from inference metrics).
        
        Args:
            call_type: The type to filter by ('embedding' or 'inference')
        
        Returns:
            List of metric dictionaries matching the call type
        """
        # List comprehension: create a list of metrics where call_type matches
        return [m for m in self.metrics if m.get("call_type") == call_type]
    
    def clear(self):
        """
        Clear all metrics from the store.
        
        Useful for starting fresh or resetting between runs.
        """
        self.metrics = []

