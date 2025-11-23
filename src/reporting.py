"""
Reporting module for aggregating metrics and creating visualizations.

This module provides functions to:
- Load metrics from saved files
- Aggregate metrics (sums, averages, percentiles)
- Create summary tables
- Generate visualizations (plots, charts)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_actual_tokens(token_counts: Dict[str, any], call_type: str) -> Dict[str, int]:
    """
    Extract actual tokens from token_counts dictionary with fallback to estimated.
    
    This helper function prioritizes actual token counts from Ollama's usage field
    when available, falling back to estimated tokens if actual are not present.
    
    For embedding calls, returns:
        - input_tokens: Actual or estimated input tokens
    
    For inference calls, returns:
        - prompt_tokens: Actual or estimated prompt tokens
        - response_tokens: Actual or estimated response tokens
        - total_tokens: Sum of prompt and response tokens
    
    Args:
        token_counts: Dictionary containing token count information
        call_type: Type of call - 'embedding' or 'inference'
    
    Returns:
        Dictionary with token counts (actual preferred, estimated as fallback)
    """
    if not isinstance(token_counts, dict):
        # If token_counts is not a dict, return zeros
        if call_type == "embedding":
            return {"input_tokens": 0}
        else:
            return {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
    
    if call_type == "embedding":
        # For embeddings, prioritize actual_input_tokens, fallback to input_tokens
        input_tokens = token_counts.get("actual_input_tokens")
        if input_tokens is None:
            input_tokens = token_counts.get("input_tokens", 0)
        return {"input_tokens": input_tokens}
    
    elif call_type == "inference":
        # For inference, prioritize actual tokens, fallback to estimated
        prompt_tokens = token_counts.get("actual_prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = token_counts.get("prompt_tokens", 0)
        
        response_tokens = token_counts.get("actual_response_tokens")
        if response_tokens is None:
            response_tokens = token_counts.get("response_tokens", 0)
        
        # Calculate total (use actual if available, otherwise use provided total or sum)
        total_tokens = token_counts.get("total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + response_tokens
        
        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens
        }
    
    # Unknown call type, return empty dict
    return {}


def load_metrics(filepath: Path) -> pd.DataFrame:
    """
    Load metrics from a JSON file into a pandas DataFrame.
    
    A DataFrame is like a spreadsheet - rows are records, columns are fields.
    This makes it easy to analyze and visualize the data.
    
    This function handles both old format (direct list of metrics) and
    new format (wrapper with timestamp metadata). It's backward compatible.
    
    Args:
        filepath: Path to the JSON file containing metrics
    
    Returns:
        pandas DataFrame with all metrics
    """
    # Convert Path to string if needed
    if isinstance(filepath, Path):
        filepath = str(filepath)
    
    # Read JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if this is the new format (has export_timestamp key)
    # New format: {"export_timestamp": "...", "metrics": [...]}
    # Old format: [{"metric1": ...}, {"metric2": ...}]
    if isinstance(data, dict) and "metrics" in data:
        # New format with timestamp metadata
        metrics = data["metrics"]
        # Note: We could add the export_timestamp to each row if needed
        # For now, we just extract the metrics list
    elif isinstance(data, list):
        # Old format (direct list)
        metrics = data
    else:
        # Unexpected format, try to use as-is
        metrics = data
    
    # Convert list of dictionaries to DataFrame
    # Each dictionary becomes a row, keys become columns
    df = pd.DataFrame(metrics)
    
    return df


def aggregate_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Aggregate metrics to compute summary statistics.
    
    This function calculates:
    - Total tokens by call type (embedding vs inference)
    - Total time by call type
    - Average latencies
    - Percentiles (p50, p95, etc.)
    - Throughput (tokens per second)
    
    Args:
        df: DataFrame containing metrics
    
    Returns:
        Dictionary with aggregated statistics
    """
    results = {}
    
    # Separate metrics by call type
    embedding_df = df[df['call_type'] == 'embedding'] if 'call_type' in df.columns else pd.DataFrame()
    inference_df = df[df['call_type'] == 'inference'] if 'call_type' in df.columns else pd.DataFrame()
    
    # Aggregate embedding metrics
    if not embedding_df.empty:
        # Extract token counts using helper function (prioritizes actual tokens)
        embedding_input_tokens = []
        for idx, row in embedding_df.iterrows():
            token_counts = row.get('token_counts', {})
            tokens = get_actual_tokens(token_counts, 'embedding')
            embedding_input_tokens.append(tokens.get('input_tokens', 0))
        
        embedding_durations = embedding_df['duration_seconds'].tolist() if 'duration_seconds' in embedding_df.columns else []
        
        results['embedding'] = {
            'total_calls': len(embedding_df),
            'total_tokens': sum(embedding_input_tokens),
            'total_time_seconds': sum(embedding_durations),
            'avg_tokens_per_call': np.mean(embedding_input_tokens) if embedding_input_tokens else 0,
            'avg_latency_seconds': np.mean(embedding_durations) if embedding_durations else 0,
            'p50_latency_seconds': np.percentile(embedding_durations, 50) if embedding_durations else 0,
            'p95_latency_seconds': np.percentile(embedding_durations, 95) if embedding_durations else 0,
            'tokens_per_second': sum(embedding_input_tokens) / sum(embedding_durations) if sum(embedding_durations) > 0 else 0
        }
    
    # Aggregate inference metrics
    if not inference_df.empty:
        # Extract token counts using helper function (prioritizes actual tokens)
        inference_prompt_tokens = []
        inference_response_tokens = []
        inference_total_tokens = []
        
        for idx, row in inference_df.iterrows():
            token_counts = row.get('token_counts', {})
            tokens = get_actual_tokens(token_counts, 'inference')
            inference_prompt_tokens.append(tokens.get('prompt_tokens', 0))
            inference_response_tokens.append(tokens.get('response_tokens', 0))
            inference_total_tokens.append(tokens.get('total_tokens', 0))
        
        inference_durations = inference_df['duration_seconds'].tolist() if 'duration_seconds' in inference_df.columns else []
        
        results['inference'] = {
            'total_calls': len(inference_df),
            'total_prompt_tokens': sum(inference_prompt_tokens),
            'total_response_tokens': sum(inference_response_tokens),
            'total_tokens': sum(inference_total_tokens),
            'total_time_seconds': sum(inference_durations),
            'avg_prompt_tokens': np.mean(inference_prompt_tokens) if inference_prompt_tokens else 0,
            'avg_response_tokens': np.mean(inference_response_tokens) if inference_response_tokens else 0,
            'avg_latency_seconds': np.mean(inference_durations) if inference_durations else 0,
            'p50_latency_seconds': np.percentile(inference_durations, 50) if inference_durations else 0,
            'p95_latency_seconds': np.percentile(inference_durations, 95) if inference_durations else 0,
            'tokens_per_second': sum(inference_total_tokens) / sum(inference_durations) if sum(inference_durations) > 0 else 0
        }
    
    # Overall totals
    all_tokens = []
    all_durations = []
    
    for idx, row in df.iterrows():
        call_type = row.get('call_type', 'unknown')
        token_counts = row.get('token_counts', {})
        
        # Use helper function to get actual tokens (prioritizes actual over estimated)
        tokens = get_actual_tokens(token_counts, call_type)
        if call_type == 'embedding':
            all_tokens.append(tokens.get('input_tokens', 0))
        else:
            all_tokens.append(tokens.get('total_tokens', 0))
        
        if 'duration_seconds' in row:
            all_durations.append(row['duration_seconds'])
        else:
            all_durations.append(0)
    
    results['overall'] = {
        'total_calls': len(df),
        'total_tokens': sum(all_tokens),
        'total_time_seconds': sum(all_durations),
        'tokens_per_second': sum(all_tokens) / sum(all_durations) if sum(all_durations) > 0 else 0
    }
    
    return results


def create_summary_tables(metrics: Dict[str, any]) -> pd.DataFrame:
    """
    Create a summary table from aggregated metrics.
    
    This formats the metrics into a nice table that's easy to read
    in a Jupyter notebook.
    
    Args:
        metrics: Dictionary from aggregate_metrics()
    
    Returns:
        pandas DataFrame formatted as a summary table
    """
    rows = []
    
    # Add embedding row
    if 'embedding' in metrics:
        emb = metrics['embedding']
        rows.append({
            'Call Type': 'Embedding',
            'Total Calls': emb['total_calls'],
            'Total Tokens': emb['total_tokens'],
            'Total Time (s)': f"{emb['total_time_seconds']:.2f}",
            'Avg Latency (s)': f"{emb['avg_latency_seconds']:.2f}",
            'P95 Latency (s)': f"{emb['p95_latency_seconds']:.2f}",
            'Tokens/sec': f"{emb['tokens_per_second']:.2f}"
        })
    
    # Add inference row
    if 'inference' in metrics:
        inf = metrics['inference']
        rows.append({
            'Call Type': 'Inference',
            'Total Calls': inf['total_calls'],
            'Total Tokens': inf['total_tokens'],
            'Total Time (s)': f"{inf['total_time_seconds']:.2f}",
            'Avg Latency (s)': f"{inf['avg_latency_seconds']:.2f}",
            'P95 Latency (s)': f"{inf['p95_latency_seconds']:.2f}",
            'Tokens/sec': f"{inf['tokens_per_second']:.2f}"
        })
    
    # Add overall row
    if 'overall' in metrics:
        ovr = metrics['overall']
        rows.append({
            'Call Type': 'Overall',
            'Total Calls': ovr['total_calls'],
            'Total Tokens': ovr['total_tokens'],
            'Total Time (s)': f"{ovr['total_time_seconds']:.2f}",
            'Avg Latency (s)': '-',
            'P95 Latency (s)': '-',
            'Tokens/sec': f"{ovr['tokens_per_second']:.2f}"
        })
    
    # Create DataFrame from rows
    return pd.DataFrame(rows)


def plot_token_distribution(chunks_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a histogram showing the distribution of tokens per chunk.
    
    A histogram shows how many chunks fall into each token count range.
    This helps you understand if your chunking is consistent or varies a lot.
    
    Args:
        chunks_df: DataFrame with chunk data (must have 'token_count' column)
        ax: Optional matplotlib axes to plot on (creates new figure if None)
    
    Returns:
        matplotlib Axes object with the plot
    """
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract token counts
    token_counts = chunks_df['token_count'].tolist() if 'token_count' in chunks_df.columns else []
    
    # Create histogram
    # bins=30 means divide the range into 30 bars
    ax.hist(token_counts, bins=30, edgecolor='black', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Tokens per Chunk', fontsize=12)
    ax.set_ylabel('Number of Chunks', fontsize=12)
    ax.set_title('Distribution of Tokens per Chunk', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_latency_vs_tokens(calls_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a scatter plot showing latency vs token count for each call.
    
    This helps you see if there's a relationship between token count and latency.
    Generally, more tokens = longer latency, but the relationship might not be linear.
    
    Args:
        calls_df: DataFrame with call metrics (must have token_counts and duration_seconds)
        ax: Optional matplotlib axes to plot on
    
    Returns:
        matplotlib Axes object with the plot
    """
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    tokens = []
    latencies = []
    call_types = []
    
    for idx, row in calls_df.iterrows():
        call_type = row.get('call_type', 'unknown')
        token_counts = row.get('token_counts', {})
        
        # Use helper function to get actual tokens (prioritizes actual over estimated)
        tokens_dict = get_actual_tokens(token_counts, call_type)
        if call_type == 'embedding':
            total_tokens = tokens_dict.get('input_tokens', 0)
        else:
            total_tokens = tokens_dict.get('total_tokens', 0)
        tokens.append(total_tokens)
        
        latencies.append(row.get('duration_seconds', 0))
        call_types.append(call_type)
    
    # Create scatter plot with different colors for different call types
    for call_type in set(call_types):
        # Filter data for this call type
        type_tokens = [t for t, ct in zip(tokens, call_types) if ct == call_type]
        type_latencies = [l for l, ct in zip(latencies, call_types) if ct == call_type]
        
        # Plot with label
        ax.scatter(type_tokens, type_latencies, label=call_type, alpha=0.6, s=50)
    
    # Add labels and title
    ax.set_xlabel('Tokens per Call', fontsize=12)
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Latency vs Token Count', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_throughput_comparison(metrics: Dict[str, any], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a bar chart comparing throughput (tokens/second) between call types.
    
    Throughput shows how fast the system processes tokens.
    Higher is better - it means the system is faster.
    
    Args:
        metrics: Dictionary from aggregate_metrics()
        ax: Optional matplotlib axes to plot on
    
    Returns:
        matplotlib Axes object with the plot
    """
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract throughput data
    call_types = []
    throughputs = []
    
    if 'embedding' in metrics:
        call_types.append('Embedding')
        throughputs.append(metrics['embedding']['tokens_per_second'])
    
    if 'inference' in metrics:
        call_types.append('Inference')
        throughputs.append(metrics['inference']['tokens_per_second'])
    
    if 'overall' in metrics:
        call_types.append('Overall')
        throughputs.append(metrics['overall']['tokens_per_second'])
    
    # Create bar chart
    bars = ax.bar(call_types, throughputs, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Add labels and title
    ax.set_ylabel('Tokens per Second', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_latency_distribution(calls_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a histogram showing the distribution of latencies.
    
    This helps you understand the variability in call times.
    A narrow distribution means consistent performance.
    A wide distribution means some calls are much slower than others.
    
    Args:
        calls_df: DataFrame with call metrics (must have duration_seconds and call_type)
        ax: Optional matplotlib axes to plot on
    
    Returns:
        matplotlib Axes object with the plot
    """
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by call type
    for call_type in calls_df['call_type'].unique() if 'call_type' in calls_df.columns else ['unknown']:
        type_df = calls_df[calls_df['call_type'] == call_type]
        latencies = type_df['duration_seconds'].tolist() if 'duration_seconds' in type_df.columns else []
        
        # Plot histogram
        ax.hist(latencies, bins=30, label=call_type, alpha=0.6, edgecolor='black')
    
    # Add labels and title
    ax.set_xlabel('Latency (seconds)', fontsize=12)
    ax.set_ylabel('Number of Calls', fontsize=12)
    ax.set_title('Distribution of Call Latencies', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

