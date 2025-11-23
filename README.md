# LLM Cost Estimator

A comprehensive Python tool for estimating and tracking tokens and timing metrics when using Ollama with Gemma models for embeddings and inference on text documents.

## Overview

This project demonstrates how to:
- **Estimate and track tokens** for LLM embeddings and inference
- **Measure time** taken for embedding + inference operations
- **Generate questions** from text-heavy documents using local LLMs via Ollama
- **Produce clear, beautiful reports** summarizing all metrics
- **Perform cost sensitivity analysis** by running multiple experiments with varying parameters

## Features

- ✅ Support for both `.txt` and `.pdf` files
- ✅ Configurable chunking strategies (fixed token window, paragraph-based, or both)
- ✅ Rigorous token accounting using tiktoken (no authentication required)
- ✅ High-resolution timing measurements using `time.perf_counter()`
- ✅ Comprehensive metrics tracking and storage
- ✅ Beautiful Jupyter notebook reports with visualizations
- ✅ Cost sensitivity analysis with automated experiment execution
- ✅ Support for both API pricing (per-token) and local GPU pricing (per-hour)
- ✅ Modular, well-documented codebase

## Tech Stack

- **Language**: Python 3.8+
- **LLM Host**: Ollama (local instance)
- **Models**: 
  - Embedding: `embeddinggemma` (configurable)
  - Generation: `gemma3:1b` or `gemma3:4b` (configurable)
- **Libraries**: tiktoken, pandas, matplotlib, jupyter

## Requirements

- Python 3.8 or higher
- Ollama installed and running locally
- Gemma models pulled in Ollama (see Setup section)

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Run Ollama

1. Download and install Ollama from [https://ollama.ai](https://ollama.ai)
2. Start Ollama (it runs as a service)
3. Verify it's running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### 3. Pull Gemma Models

Pull the required Gemma models in Ollama:

```bash
# Pull the embedding model
ollama pull embeddinggemma

# Pull the generation models (Gemma3 1B and/or 4B)
ollama pull gemma3:1b
ollama pull gemma3:4b  # Optional, for larger model experiments
```

### 4. Prepare Your Document

Place your text document (`.txt` or `.pdf`) in the project root directory, or in the `data/` directory. The example uses `MobyDick.txt` which is already in the project root.

## Usage

### Running the Notebooks

The project consists of seven Jupyter notebooks organized into different categories:

**Setup and Diagnostics (run first if needed):**

0. **`notebooks/00_test_ollama_connection.ipynb`**
   - Tests connection to Ollama API
   - Lists available models in your Ollama installation
   - Verifies Ollama is running and accessible
   - Helps troubleshoot connection issues before running experiments

6. **`notebooks/06_diagnose_ollama_404.ipynb`**
   - Diagnostic tool for troubleshooting Ollama API 404 errors
   - Tests Ollama server connectivity
   - Verifies API endpoints (`/api/chat`, `/api/embeddings`)
   - Checks if specific models exist
   - Provides detailed error diagnostics and troubleshooting steps
   - Use this if you encounter 404 errors when running experiments

**Core Workflow (run in order):**

1. **`notebooks/01_ingest_and_embed.ipynb`**
   - Loads and chunks the document
   - Embeds all chunks using Ollama
   - Tracks tokens and timing for embedding operations
   - Saves results to `results/` directory
   - All exported data includes timestamps for traceability

2. **`notebooks/02_inference_and_question_generation.ipynb`**
   - Loads embedded chunks from notebook 01
   - Demonstrates user-defined questions
   - Auto-generates questions from chunks
   - Tracks tokens and timing for inference operations
   - Saves inference metrics with timestamps

3. **`notebooks/03_reporting_and_visualization.ipynb`**
   - Loads all metrics from previous notebooks
   - Creates comprehensive summary tables
   - Generates visualizations (histograms, scatter plots, bar charts)
   - Provides interpretation and analysis
   - Handles both old and new timestamp formats (backward compatible)

**Cost Sensitivity Analysis (optional, can run independently):**

4. **`notebooks/04_cost_sensitivity_analysis.ipynb`**
   - Runs multiple experiments with varying parameters automatically
   - Supports both API pricing (per-token) and local GPU pricing (per-hour)
   - Calculates costs for each experiment configuration
   - Performs sensitivity analysis on key parameters (chunk size, model size, document count, etc.)
   - Generates comprehensive visualizations comparing experiments
   - Produces tables and plots showing cost relationships
   - Can reuse existing chunks from notebook 01 for faster execution
   - All experiment results include timestamps

**Advanced Experimental Design (optional, for large-scale analysis):**

5. **`notebooks/05_nolh_experimental_design.ipynb`**
   - Uses Nearly Orthogonal Latin Hypercube (NOLH) sampling for efficient parameter space exploration
   - Generates well-distributed sample points across multiple dimensions
   - Explores parameter combinations with fewer experiments than full factorial design
   - Supports multiple generation and embedding models
   - Automatically runs experiments and collects results
   - Includes comprehensive CSV export functionality:
     * Full experiment summary
     * NOLH coverage metrics
     * Model performance comparisons
     * Cost analysis summaries
     * Parameter sensitivity analysis
   - All exports include timestamps and are saved to `results/experiments/nolh/csv_exports/`
   - Validates design quality with coverage metrics
   - Produces extensive visualizations and analysis

### Starting Jupyter

```bash
jupyter notebook
```

Then navigate to the `notebooks/` directory and run the notebooks in order.

## Project Structure

```
GenerativeAI-Cost-Estimator/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
├── .gitignore              # Git ignore rules
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── ollama_client.py    # Ollama API wrapper
│   ├── token_accounting.py # Token counting with Gemma tokenizer
│   ├── timing_metrics.py  # Timing and metrics collection
│   ├── pipeline.py         # High-level workflow orchestration
│   └── reporting.py        # Metrics aggregation and visualization
├── notebooks/              # Jupyter notebooks
│   ├── 00_test_ollama_connection.ipynb      # Test Ollama setup
│   ├── 01_ingest_and_embed.ipynb            # Core: Document ingestion and embedding
│   ├── 02_inference_and_question_generation.ipynb  # Core: Question generation
│   ├── 03_reporting_and_visualization.ipynb  # Core: Reporting and analysis
│   ├── 04_cost_sensitivity_analysis.ipynb    # Cost sensitivity experiments
│   ├── 05_nolh_experimental_design.ipynb     # Advanced: NOLH experimental design
│   └── 06_diagnose_ollama_404.ipynb          # Diagnostic: Troubleshoot Ollama errors
├── data/                   # Input documents (gitignored)
│   └── .gitignore
├── results/                # Output files (gitignored)
│   ├── .gitignore
│   ├── metrics.json        # Combined metrics from notebooks 01-02 (with timestamps)
│   ├── chunks.json         # Chunk metadata and embeddings (with timestamps)
│   └── experiments/        # Experiment results
│       ├── {experiment_name}_metrics.json  # Per-experiment metrics (with timestamps)
│       ├── summary.csv     # Aggregated experiment summary
│       └── nolh/           # NOLH experimental design results (from notebook 05)
│           ├── csv_exports/  # Comprehensive CSV exports with all analysis views
│           └── {experiment_name}_metrics.json  # NOLH experiment metrics
└── MobyDick.txt           # Example document
```

## Configuration

Configuration is managed through the `Config` class in `src/config.py`. Key settings:

- **`embedding_model`**: Gemma embedding model name (default: `embeddinggemma`)
- **`generation_model`**: Gemma generation model name (default: `gemma3:1b`)
- **`ollama_endpoint`**: Ollama API endpoint (default: `http://localhost:11434`)
- **`chunk_size_tokens`**: Target chunk size in tokens (default: 512)
- **`chunk_overlap_tokens`**: Overlap between chunks (default: 50)
- **`chunking_strategy`**: Chunking method - `fixed_token_window`, `paragraph_based`, or `both`

You can modify these in the notebooks or create a custom config object.

## Token and Time Accounting

### Token Accounting

The project uses `tiktoken` (a fast, lightweight tokenizer) to count tokens. This approach:
- Requires no authentication (unlike Hugging Face tokenizers)
- Provides accurate token counts for cost estimation
- Uses the `cl100k_base` encoding (similar to GPT models) as a reasonable approximation
- Falls back to simple word-based estimation if tiktoken is not available

For the most accurate token counts, you can also check the `usage` field in Ollama's API responses, which provides actual token counts from the model.

Token counts are tracked for:
- Document text (total tokens)
- Each chunk (input tokens for embeddings)
- Each prompt and response (for inference)

### Timing

All operations are timed using `time.perf_counter()` for high-resolution measurements. The system tracks:

- Duration per call (embedding or inference)
- Total time per operation type
- Average and percentile latencies (p50, p95)
- Throughput (tokens per second)

### Metrics Storage

All metrics are saved to JSON files in the `results/` directory with automatic timestamp tracking:
- `results/metrics.json`: All timing and token metrics from notebooks 01-02 (includes export timestamp)
- `results/chunks.json`: Chunk metadata and embeddings (includes export timestamp)
- `results/experiments/{experiment_name}_metrics.json`: Individual experiment metrics (includes export timestamp)
- `results/experiments/summary.csv`: Aggregated summary of all experiments
- `results/experiments/nolh/csv_exports/`: Comprehensive CSV exports from NOLH experimental design:
  - `01_full_experiment_summary.csv`: Complete results from all experiments
  - `02_nolh_coverage_metrics.csv`: NOLH design quality metrics
  - `03_model_performance_by_*.csv`: Model performance comparisons
  - `04_cost_analysis_*.csv`: Cost breakdowns and comparisons
  - `05_parameter_*.csv`: Parameter sensitivity analysis

All exported data includes both ISO format timestamps (`export_timestamp`) and human-readable timestamps (`export_timestamp_readable`) for traceability.

## Reporting

### Standard Reporting (Notebook 03)

The reporting notebook (`03_reporting_and_visualization.ipynb`) produces a comprehensive report including:

1. **Executive Summary**: High-level overview and objectives
2. **Metrics Dashboard**: Summary tables with:
   - Total tokens (embedding vs inference)
   - Total time (embedding vs inference)
   - Average and P95 latencies
   - Throughput metrics
3. **Visualizations**:
   - Token distribution histogram
   - Latency vs token count scatter plot
   - Throughput comparison bar chart
   - Latency distribution histogram
4. **Interpretation**: Analysis of what the metrics mean and practical implications

### Cost Sensitivity Analysis (Notebook 04)

The cost sensitivity analysis notebook (`04_cost_sensitivity_analysis.ipynb`) allows you to:

1. **Define Multiple Experiments**: Configure experiments with varying parameters:
   - Model configurations (different Gemma variants)
   - Workload characteristics (number of docs, chunk sizes, questions per doc)
   - Pricing modes (API per-token or local per-hour)

2. **Run Experiments Automatically**: Execute all experiments in sequence with error handling

3. **Calculate Costs**: 
   - **API Mode**: Cost = (input_tokens/1000 × price_per_1k_input) + (output_tokens/1000 × price_per_1k_output)
   - **Local Mode**: Cost = (runtime_hours × dollars_per_gpu_hour)

4. **Sensitivity Analysis**: 
   - Compare how different parameters affect cost, tokens, and runtime
   - Calculate percentage changes when varying single parameters
   - Analyze scaling efficiency

5. **Comprehensive Visualizations**:
   - Total cost vs number of documents
   - Cost per 1K tokens vs chunk size/model
   - Runtime vs tokens
   - Cost per document vs concurrency
   - Multi-panel sensitivity analysis plots

**Example Experiment Configuration:**
```python
experiments = [
    {
        "name": "baseline_gemma3_1b",
        "gen_model": "gemma3:1b",
        "embed_model": "embeddinggemma",
        "num_docs": 10,
        "chunk_size": 512,
        "num_questions_per_doc": 2,
        "pricing": {
            "mode": "api",
            "price_per_1k_tokens_input": 0.00250,
            "price_per_1k_tokens_output": 0.01000
        }
    },
    # ... more experiments
]
```

## How It Works

### Token Accounting Implementation

The `token_accounting.py` module:
- Uses `tiktoken` for fast token counting (no authentication required)
- Falls back to simple word-based estimation if tiktoken is unavailable
- Provides functions to count tokens in text strings
- Separately counts prompt and response tokens for inference

### Timing Implementation

The `timing_metrics.py` module:
- Provides a `TimingContext` context manager for easy timing
- Implements a `MetricsStore` class to collect all metrics
- Stores metrics with call type, duration, token counts, and identifiers
- Exports metrics to JSON and CSV formats

### Pipeline Flow

1. **Document Loading**: Supports both `.txt` and `.pdf` files
2. **Chunking**: Splits document using configured strategy
3. **Embedding**: Calls Ollama API for each chunk, tracks metrics
4. **Inference**: Generates questions or answers, tracks metrics
5. **Reporting**: Aggregates and visualizes all collected metrics

## Results

Results are stored in the `results/` directory:
- Metrics are saved as JSON for easy loading and analysis
- Chunks and embeddings are saved for reuse
- All files are gitignored by default (add to `.gitignore` if needed)

## Troubleshooting

### Ollama Connection Issues

If you get connection errors:
1. Run `notebooks/00_test_ollama_connection.ipynb` to verify Ollama setup
2. Verify Ollama is running: `curl http://localhost:11434/api/tags`
3. Check the endpoint in your config matches your Ollama setup
4. Ensure models are pulled: `ollama list`
5. Use `start_ollama.sh` script to start Ollama if needed

### Ollama 404 Errors

If you encounter 404 errors when calling the Ollama API:
1. Run `notebooks/06_diagnose_ollama_404.ipynb` for detailed diagnostics
2. Check if the model exists: `ollama list`
3. Verify the model name format (e.g., `gemma3:1b` not `gengemma3_1b`)
4. Ensure Ollama version supports `/api/chat` endpoint (upgrade if needed)
5. Check improved error messages in `src/ollama_client.py` for specific guidance

### Model Not Found

If you get model not found errors:
1. Pull the required models: 
   ```bash
   ollama pull embeddinggemma
   ollama pull gemma3:1b
   ollama pull gemma3:4b  # If using 4B model in experiments
   ```
2. Verify model names match your config
3. Check available models: `ollama list`

### Tokenizer Issues

If tokenization fails:
1. Ensure `tiktoken` library is installed: `pip install tiktoken`
2. The system will fall back to simple word-based estimation if tiktoken is unavailable
3. For best accuracy, ensure tiktoken is installed: `pip install -r requirements.txt`

## Cost Sensitivity Analysis

The cost sensitivity analysis notebook (`04_cost_sensitivity_analysis.ipynb`) is a powerful tool for understanding how different parameters affect your LLM costs. 

### Key Features

- **Automated Experiment Execution**: Define multiple experiments and run them automatically
- **Flexible Pricing Models**: Support for both API pricing (per-token) and local GPU pricing (per-hour)
- **Parameter Variation**: Easily test different:
  - Model sizes (gemma3:1b vs gemma3:4b)
  - Chunk sizes (256, 512, 1024 tokens)
  - Document counts (10, 20, 30, 60+)
  - Questions per document (2, 3, 5+)
  - Concurrency levels

### Usage

1. **Edit Experiment Configurations**: Modify the `experiments` list in Section 2 of the notebook
2. **Run Experiments**: Execute Section 4 to run all experiments (can skip already-run experiments)
3. **Analyze Results**: Review tables and plots in Sections 5-8
4. **Interpret Sensitivity**: Use Section 8 to understand parameter impacts

### Example Insights

The sensitivity analysis helps answer questions like:
- How does model size (1B vs 4B) affect cost and latency?
- What's the optimal chunk size for cost efficiency?
- How does cost scale with document count?
- Is API pricing or local GPU pricing more cost-effective for my workload?

## Contributing

This is a demonstration project. Feel free to:
- Experiment with different models and configurations
- Add new chunking strategies
- Extend the reporting capabilities
- Add more experiment parameters to the sensitivity analysis
- Improve error handling and robustness

## License

This project is provided as-is for educational and demonstration purposes.

## Acknowledgments

- Ollama for providing local LLM infrastructure
- Google for the Gemma model series
- The open-source Python ecosystem (transformers, pandas, matplotlib, etc.)

