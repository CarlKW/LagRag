# Pipeline Testing Guide

This directory contains test scripts for the RAG pipeline.

## Test Scripts

### 1. `test_pipeline.py` - Full Pipeline Test

Runs the complete pipeline and shows detailed information about chunks and query results.

**Usage:**
```bash
python src/indexing/test_pipeline.py
```

**What it does:**
1. Loads documents from the filtered JSONL file
2. Chunks documents into paragraph-level chunks
3. Creates embeddings using TTC-L2V-supervised-2
4. Stores chunks in ChromaDB
5. Tests several queries and shows results

**Features:**
- Shows chunk summary (total chunks, chunks per paragraph)
- Displays sample chunks with metadata
- Tests multiple queries and shows top results with scores
- Saves vector store to `./chroma_db_test`

**Customization:**
Edit the script to change:
- `num_docs`: Number of documents to process (default: 5 for faster testing)
- `test_queries`: List of queries to test
- `persist_directory`: Where to save the vector store

### 2. `quick_test.py` - Interactive Query Testing

Interactive script for testing queries against an existing vector store.

**Usage:**
```bash
python src/indexing/quick_test.py
```

**What it does:**
- Loads an existing ChromaDB vector store
- Allows you to enter queries interactively
- Shows top results with similarity scores

**Requirements:**
- Must run `test_pipeline.py` first to create the vector store

## Example Workflow

1. **Run the full pipeline test:**
   ```bash
   python src/indexing/test_pipeline.py
   ```
   
   This will:
   - Process documents and create chunks
   - Show you what chunks look like
   - Test some default queries
   - Save the vector store

2. **Test custom queries interactively:**
   ```bash
   python src/indexing/quick_test.py
   ```
   
   Then enter queries like:
   - "skatt"
   - "bostadslån"
   - "arbetsmiljö"
   - Type 'quit' to exit

## Understanding the Output

### Chunk Information
- **SFS nr**: The law/regulation number (e.g., "1991:1867")
- **Type**: Either "lag" or "förordning"
- **Paragraph**: The paragraph identifier (e.g., "1 §", "31 a §")
- **Subchunk index**: If a paragraph was split, shows which part (0, 1, 2...)
- **Merged paragraphs**: If paragraphs were merged, shows which ones

### Query Results
- **Score**: Similarity score (lower is more similar for some models, higher for others)
- **Top results**: Most relevant chunks for your query
- Results include full chunk content and metadata

## Troubleshooting

### "No module named 'sentence_transformers'"
Install the required package:
```bash
pip install sentence-transformers
```

### "Model TTC-L2V-supervised-2 not found"
The model will be downloaded automatically on first use. Make sure you have internet connection and sufficient disk space.

### "Vector store not found"
Make sure you've run `test_pipeline.py` first to create the vector store.

## Next Steps

After testing, you can:
1. Process all documents (set `num_docs=None` in `test_pipeline.py`)
2. Integrate the pipeline into your main application
3. Use the vector store for RAG queries in your application

