# LLM Cost Estimator

A comprehensive Python tool for estimating and tracking tokens and timing metrics when using Ollama with Gemma models for embeddings and inference on text documents.

## Overview

This project demonstrates how to:
- **Estimate and track tokens** for LLM embeddings and inference
- **Measure time** taken for embedding + inference operations
- **Generate questions** from text-heavy documents using local LLMs via Ollama
- **Produce clear, beautiful reports** summarizing all metrics

## Features

- ✅ Support for both `.txt` and `.pdf` files
- ✅ Configurable chunking strategies (fixed token window, paragraph-based, or both)
- ✅ Rigorous token accounting using Gemma tokenizer
- ✅ High-resolution timing measurements using `time.perf_counter()`
- ✅ Comprehensive metrics tracking and storage
- ✅ Beautiful Jupyter notebook reports with visualizations
- ✅ Modular, well-documented codebase

## Tech Stack

- **Language**: Python 3.8+
- **LLM Host**: Ollama (local instance)
- **Models**: 
  - Embedding: `embeddinggemma` (configurable)
  - Generation: `gemma3:1b` (configurable)
- **Libraries**: transformers, pandas, matplotlib, jupyter

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

# Pull the generation model (Gemma3 1B)
ollama pull gemma3:1b
```

### 4. Prepare Your Document

Place your text document (`.txt` or `.pdf`) in the project root directory, or in the `data/` directory. The example uses `MobyDick.txt` which is already in the project root.

## Usage

### Running the Notebooks

The project consists of three Jupyter notebooks that should be run in order:

1. **`notebooks/01_ingest_and_embed.ipynb`**
   - Loads and chunks the document
   - Embeds all chunks using Ollama
   - Tracks tokens and timing for embedding operations
   - Saves results to `results/` directory

2. **`notebooks/02_inference_and_question_generation.ipynb`**
   - Loads embedded chunks from notebook 01
   - Demonstrates user-defined questions
   - Auto-generates questions from chunks
   - Tracks tokens and timing for inference operations
   - Saves inference metrics

3. **`notebooks/03_reporting_and_visualization.ipynb`**
   - Loads all metrics from previous notebooks
   - Creates comprehensive summary tables
   - Generates visualizations (histograms, scatter plots, bar charts)
   - Provides interpretation and analysis

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
│   ├── 01_ingest_and_embed.ipynb
│   ├── 02_inference_and_question_generation.ipynb
│   └── 03_reporting_and_visualization.ipynb
├── data/                   # Input documents (gitignored)
│   └── .gitignore
├── results/                # Output files (gitignored)
│   └── .gitignore
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

The project uses the Gemma tokenizer from the `transformers` library to accurately count tokens the same way the Gemma models see them. This ensures:

- Accurate token counts for cost estimation
- Proper chunk sizing based on token limits
- Consistent tokenization across operations

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

All metrics are saved to JSON files in the `results/` directory:
- `results/metrics.json`: All timing and token metrics
- `results/chunks.json`: Chunk metadata and embeddings
- `results/embeddings.json`: Embedding vectors (if saved separately)

## Reporting

The final notebook (`03_reporting_and_visualization.ipynb`) produces a comprehensive report including:

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

## How It Works

### Token Accounting Implementation

The `token_accounting.py` module:
- Loads the Gemma tokenizer once and caches it
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
1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check the endpoint in your config matches your Ollama setup
3. Ensure models are pulled: `ollama list`

### Model Not Found

If you get model not found errors:
1. Pull the required models: `ollama pull embeddinggemma` and `ollama pull gemma3:1b`
2. Verify model names match your config
3. Check available models: `ollama list`

### Tokenizer Issues

If tokenization fails:
1. Ensure `transformers` library is installed: `pip install transformers`
2. First run will download tokenizer files (requires internet)
3. Check you have sufficient disk space for model files

## Contributing

This is a demonstration project. Feel free to:
- Experiment with different models and configurations
- Add new chunking strategies
- Extend the reporting capabilities
- Improve error handling and robustness

## License

This project is provided as-is for educational and demonstration purposes.

## Acknowledgments

- Ollama for providing local LLM infrastructure
- Google for the Gemma model series
- The open-source Python ecosystem (transformers, pandas, matplotlib, etc.)

