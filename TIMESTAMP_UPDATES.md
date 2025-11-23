# Timestamp Updates Summary

All exported data now includes timestamps. Here's what has been updated:

## ✅ Completed Updates

### 1. Core Functions (Updated)
- **`src/pipeline.py::save_chunks()`** - Now saves chunks with timestamp metadata
- **`src/timing_metrics.py::export_to_json()`** - Now exports metrics with timestamp metadata  
- **`src/timing_metrics.py::export_to_csv()`** - Now includes timestamp column in CSV exports
- **`src/reporting.py::load_metrics()`** - Updated to handle both old and new formats (backward compatible)

### 2. Export Format Changes

#### Chunks Export Format
**Old format:**
```json
[
  {"chunk_id": "chunk_0", "text": "...", ...},
  {"chunk_id": "chunk_1", "text": "...", ...}
]
```

**New format:**
```json
{
  "export_timestamp": "2024-01-15T14:30:45.123456",
  "export_timestamp_readable": "2024-01-15 14:30:45",
  "num_chunks": 772,
  "chunks": [
    {"chunk_id": "chunk_0", "text": "...", ...},
    {"chunk_id": "chunk_1", "text": "...", ...}
  ]
}
```

#### Metrics Export Format
**Old format:**
```json
[
  {"call_type": "embedding", "duration_seconds": 0.5, ...},
  {"call_type": "inference", "duration_seconds": 1.2, ...}
]
```

**New format:**
```json
{
  "export_timestamp": "2024-01-15T14:30:45.123456",
  "export_timestamp_readable": "2024-01-15 14:30:45",
  "num_metrics": 100,
  "metrics": [
    {"call_type": "embedding", "duration_seconds": 0.5, ...},
    {"call_type": "inference", "duration_seconds": 1.2, ...}
  ]
}
```

#### CSV Export Format
CSV exports now include an `export_timestamp` column as the first column, with the same timestamp for all rows in that export.

## 📝 Manual Updates Needed in Notebooks

### Notebook: `05_nolh_experimental_design.ipynb`

#### 1. Update chunk loading code (around line 848)
**Current code:**
```python
with open(chunks_path, 'r') as f:
    all_chunks = json.load(f)
print(f"Loaded {len(all_chunks)} existing chunks")
```

**Should be:**
```python
with open(chunks_path, 'r') as f:
    data = json.load(f)

# Handle both old format (direct list) and new format (with timestamp wrapper)
if isinstance(data, dict) and "chunks" in data:
    # New format with timestamp metadata
    all_chunks = data["chunks"]
    export_time = data.get("export_timestamp_readable", data.get("export_timestamp", "unknown"))
    print(f"Loaded {len(all_chunks)} existing chunks (exported: {export_time})")
elif isinstance(data, list):
    # Old format (direct list) - backward compatibility
    all_chunks = data
    print(f"Loaded {len(all_chunks)} existing chunks")
else:
    # Unexpected format, try to use as-is
    all_chunks = data if isinstance(data, list) else []
    print(f"Loaded {len(all_chunks)} existing chunks (unexpected format)")
```

#### 2. Add timestamps to experiment results (around line 920)
**Add these lines before the `results = {` dictionary:**
```python
# Get current timestamp for this experiment
from datetime import datetime
experiment_timestamp = datetime.now().isoformat()
experiment_timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

**Then add these fields at the beginning of the `results` dictionary:**
```python
results = {
    # Timestamp fields - record when this experiment was completed
    'experiment_timestamp': experiment_timestamp,
    'experiment_timestamp_readable': experiment_timestamp_readable,
    # ... rest of the fields
}
```

### Notebook: `04_cost_sensitivity_analysis.ipynb`

Apply the same updates as above for chunk loading and experiment results.

### Notebook: `02_inference_and_question_generation.ipynb`

Update chunk loading code (around line 86) to handle the new format, similar to the update above.

## 🔄 Backward Compatibility

All loading functions are backward compatible:
- They check if data is in old format (direct list) or new format (wrapper with timestamp)
- Old files will continue to work without modification
- New exports will include timestamps automatically

## 📊 Benefits

1. **Traceability**: Know exactly when data was exported
2. **Version Control**: Track when experiments were run
3. **Debugging**: Identify which export corresponds to which run
4. **Analysis**: Filter and group results by timestamp

