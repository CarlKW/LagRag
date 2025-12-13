# LagRag Development Summary

## Overview

LagRag is a Retrieval-Augmented Generation (RAG) system for Swedish laws and regulations (SFS - Svensk författningssamling). The system processes legal documents from 1990 onwards, creating a searchable vector database for legal question-answering.

**Current Dataset:** ~5,000 filtered laws and regulations (1990+)
**Processing Time:** ~50 minutes for full dataset (100 docs/min)

---

## Models and Technologies

### Embedding Model
- **Base Model:** `jealk/llm2vec-scandi-mntp-v2`
- **Supervised Adapter:** `jealk/TTC-L2V-supervised-2`
- **Framework:** LLM2Vec with PEFT (Parameter-Efficient Fine-Tuning)
- **Pooling:** Mean pooling
- **Device:** CUDA (bfloat16) or CPU (float32)
- **Batch Size:** 32 documents per batch
- **Dependencies:** `llm2vec`, `peft`, `transformers`, `torch`, `accelerate`

### Vector Database
- **Database:** ChromaDB
- **Storage:** Persistent on disk (default: `./chroma_db_test`)
- **Integration:** LangChain Chroma wrapper

### Libraries
- **LangChain:** Document handling and vector store integration
- **Transformers:** Model loading and inference
- **PEFT:** Adapter loading and merging
- **LLM2Vec:** Model wrapper for embedding generation

---

## File Structure and Functions

### `/src/ingestion/` - Data Ingestion

#### `download_sfs_laws.py`
Downloads SFS documents from Riksdagen's API.

**Key Functions:**
- `normalize_url(url: str) -> str`: Normalizes URLs to HTTPS format
- `get_text_url(doc_meta: dict) -> str | None`: Extracts text URL from document metadata
- `fetch_fulltext(url: str) -> str`: Downloads full text content
- `fetch_doclist_page(url: str, label: str, max_retries: int = 3) -> dict`: Fetches paginated document lists with retry logic
- `main()`: Main download loop (1990 to current year)

**Output:** `sfs_lagboken_1990plus.jsonl` (raw JSONL with all SFS documents)

**API Endpoint:** `https://data.riksdagen.se/dokumentlista/`

#### `filter_laws.py`
Filters and cleans the downloaded JSONL file.

**Key Functions:**
- `clean_text(text: str) -> str`: Normalizes whitespace, removes excessive formatting
- `filter_and_clean_jsonl(input_file: str, output_file: str = None) -> None`: Filters documents containing "Lag" or "Förordning" in title, cleans text

**Filters:**
- Keeps only documents with "Lag" or "Förordning" in title
- Removes `/r1/` markers from titles
- Cleans text: normalizes line breaks, collapses whitespace, removes excessive dashes/underscores

**Output:** `sfs_lagboken_1990plus_filtered.jsonl` (~5,000 documents)

#### `loader.py`
Converts filtered JSONL to LangChain Documents.

**Key Functions:**
- `determine_type(titel: str) -> str`: Classifies as "lag" or "förordning"
- `load_sfs_documents(jsonl_path: str) -> List[Document]`: Loads all documents into memory
- `load_sfs_documents_lazy(jsonl_path: str) -> Iterator[Document]`: Lazy loader (generator) for memory efficiency

**Document Metadata:**
- `sfs_nr`: SFS number (beteckning)
- `titel`: Document title
- `typ`: "lag" or "förordning"
- `dok_id`: Document ID
- `dokument_url_html`: HTML URL
- `organ`: Issuing organization

**Returns:** LangChain `Document` objects with `page_content` (fulltext) and `metadata`

---

### `/src/indexing/` - Indexing Pipeline

#### `chunker.py`
Splits documents into paragraph-level chunks.

**Key Functions:**
- `find_paragraph_starts(text: str) -> List[tuple[int, str]]`: Finds paragraph markers (e.g., "1 §", "31 a §")
- `split_into_sentences(text: str) -> List[str]`: Splits text into sentences
- `count_words(text: str) -> int`: Word count utility
- `chunk_documents(docs: List[Document], min_words: int = 50, max_words: int = 400, overlap_sentences: int = 2) -> List[Document]`: Main chunking function

**Chunking Strategy:**
- **Paragraph Detection:** Regex pattern `(?:^|\n\n)(\d+(?:\s+[a-z])?)\s+§(?!§)` finds paragraph markers
- **Long Paragraphs (>400 words):** Split by sentences with 2-sentence overlap
- **Short Paragraphs (<50 words):** Merged with next paragraph
- **Normal Paragraphs:** Kept as single chunk

**Chunk Metadata:**
- Inherits all document metadata
- `paragraf`: Paragraph identifier (e.g., "1 §", "31 a §")
- `subchunk_index`: Index if paragraph was split (0, 1, 2...)
- `merged_paragraphs`: Comma-separated list if paragraphs were merged

**Default Parameters:**
- `min_words`: 50
- `max_words`: 400
- `overlap_sentences`: 2

#### `embedder.py`
Creates embeddings using TTC-L2V-supervised-2 model.

**Key Classes:**
- `TTCEmbeddings(Embeddings)`: LangChain-compatible embedding class

**Key Functions:**
- `TTCEmbeddings.__init__(base_model_name, adapter_name)`: Loads base model, applies MNTP adapter, then supervised adapter, merges all
- `TTCEmbeddings.embed_documents(texts: List[str]) -> List[List[float]]`: Embeds multiple documents (batch size 32)
- `TTCEmbeddings.embed_query(text: str) -> List[float]`: Embeds single query
- `get_embedding_function(base_model_name, adapter_name) -> Embeddings`: Factory function returning embedding instance

**Model Loading Process:**
1. Load base model `jealk/llm2vec-scandi-mntp-v2`
2. Try to load MNTP adapter from base model repo (if separate)
3. Load supervised adapter `jealk/TTC-L2V-supervised-2`
4. Merge all adapters into base model
5. Wrap with LLM2Vec (mean pooling)
6. Set to eval mode

**Device/Dtype:**
- CUDA: `torch.bfloat16`
- CPU: `torch.float32`

#### `test_pipeline.py`
End-to-end pipeline test script.

**Key Functions:**
- `print_chunk_info(chunks: List[Document], max_chunks: int = 10)`: Prints chunk statistics
- `test_query(vectorstore: Chroma, query: str, k: int = 5, show_scores: bool = True)`: Tests a query and displays results
- `run_pipeline_test(jsonl_path: str, persist_directory: str = "./chroma_db_test", num_docs: int = 10, test_queries: List[str] = None) -> Chroma`: Complete pipeline execution

**Pipeline Steps:**
1. Load documents from JSONL
2. Chunk documents
3. Create embeddings and vector store
4. Test queries

**Default Test Queries:**
- "skatt"
- "bostad"
- "arbetsrätt"
- "miljöskydd"
- "förvaltning"

#### `quick_test.py`
Interactive query testing against existing vector store.

**Key Functions:**
- `interactive_query_test(persist_directory: str = "./chroma_db_test")`: Interactive CLI for testing queries

**Usage:** Run after `test_pipeline.py` to test queries interactively

---

### `/src/inspection/` - Document Inspection

#### `inspect_documents.py`
Document analysis and inspection tool.

**Key Functions:**
- `estimate_tokens(text: str, model: str = "gpt-4") -> Optional[int]`: Token estimation (requires tiktoken)
- `detect_sections(text: str) -> List[Dict[str, str]]`: Detects Swedish legal sections (kapitel, paragraf, artikel, stycke)
- `analyze_text_structure(text: str) -> Dict[str, any]`: Comprehensive text statistics
- `format_metadata(metadata: Dict[str, any]) -> str`: Formats metadata for display
- `inspect_single_document(doc, index: Optional[int] = None) -> None`: Full document inspection report
- `inspect_batch(documents: List, limit: Optional[int] = None) -> None`: Batch statistics analysis
- `find_document_by_sfs(documents: List, sfs_number: str) -> Optional[Tuple[int, any]]`: Find document by SFS number
- `main()`: CLI entry point

**CLI Arguments:**
- `--data-file`: Path to JSONL file (optional)
- `--index N`: Inspect document at index N
- `--sfs "1991:1867"`: Inspect document by SFS number
- `--batch`: Batch analysis of all documents
- `--limit N`: Limit batch analysis to N documents

**Analysis Output:**
- Metadata structure
- Content statistics (chars, words, sentences, lines, paragraphs)
- Token estimates
- Structural patterns (sections detected)
- Full document content

---

## API Reference for Development

### Document Loading

```python
from src.ingestion.loader import load_sfs_documents, load_sfs_documents_lazy

# Load all documents
documents = load_sfs_documents("data/sfs_lagboken_1990plus_filtered.jsonl")

# Lazy loading (memory efficient)
for doc in load_sfs_documents_lazy("data/sfs_lagboken_1990plus_filtered.jsonl"):
    process(doc)
```

### Chunking

```python
from src.indexing.chunker import chunk_documents

chunks = chunk_documents(
    documents,
    min_words=50,      # Minimum chunk size
    max_words=400,     # Maximum chunk size
    overlap_sentences=2  # Overlap when splitting
)
```

### Embedding

```python
from src.indexing.embedder import get_embedding_function

# Get embedding function
embeddings = get_embedding_function()

# Embed documents
vectors = embeddings.embed_documents(["text1", "text2"])

# Embed query
query_vector = embeddings.embed_query("search query")
```

### Vector Store

```python
from langchain_community.vectorstores import Chroma
from src.indexing.embedder import get_embedding_function

# Create vector store
embeddings = get_embedding_function()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Load existing vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Search
results = vectorstore.similarity_search_with_score("query", k=5)
```

### Document Inspection

```python
from src.inspection.inspect_documents import (
    inspect_single_document,
    inspect_batch,
    find_document_by_sfs
)

# Inspect single document
inspect_single_document(document, index=0)

# Batch analysis
inspect_batch(documents, limit=100)

# Find by SFS number
result = find_document_by_sfs(documents, "1991:1867")
if result:
    idx, doc = result
    inspect_single_document(doc, index=idx)
```

---

## Pipeline Roadmap

### Phase 1: Data Ingestion ✅
1. **Download** (`download_sfs_laws.py`)
   - Fetch documents from Riksdagen API (1990-current year)
   - Handle pagination and retries
   - Download fulltext for each document
   - Output: Raw JSONL file

2. **Filter** (`filter_laws.py`)
   - Filter for "Lag" and "Förordning" only
   - Clean text formatting
   - Remove metadata artifacts
   - Output: Filtered JSONL file (~5,000 documents)

3. **Load** (`loader.py`)
   - Convert JSONL to LangChain Documents
   - Extract and structure metadata
   - Classify document types
   - Output: List of Document objects

### Phase 2: Indexing ✅
1. **Chunk** (`chunker.py`)
   - Detect paragraph boundaries (§ markers)
   - Split long paragraphs (>400 words)
   - Merge short paragraphs (<50 words)
   - Preserve metadata and paragraph identifiers
   - Output: List of chunk Documents

2. **Embed** (`embedder.py`)
   - Load TTC-L2V-supervised-2 model
   - Generate embeddings for all chunks
   - Batch processing (32 chunks/batch)
   - Output: Embedding vectors

3. **Store** (`test_pipeline.py`)
   - Create ChromaDB vector store
   - Store chunks with embeddings
   - Persist to disk
   - Output: Persistent vector database

### Phase 3: Query & Retrieval ✅
1. **Query Interface** (`quick_test.py`)
   - Load existing vector store
   - Embed query
   - Similarity search
   - Return top-k results with scores

2. **Testing** (`test_pipeline.py`)
   - End-to-end pipeline validation
   - Query testing with sample queries
   - Result inspection

### Phase 4: Inspection & Analysis ✅
1. **Document Inspection** (`inspect_documents.py`)
   - Single document analysis
   - Batch statistics
   - Structural pattern detection
   - Token estimation

### Phase 5: Future Development (Not Yet Implemented)
1. **RAG Application**
   - Query interface (web/CLI/API)
   - Context retrieval
   - LLM integration for answer generation
   - Citation and source tracking

2. **Optimization**
   - Incremental updates (new documents)
   - Chunk size optimization
   - Embedding model fine-tuning
   - Query performance tuning

3. **Features**
   - Multi-query expansion
   - Re-ranking
   - Temporal filtering (by year)
   - Document type filtering (lag vs förordning)
   - Cross-referencing between laws

---

## Data Flow

```
1. Riksdagen API
   ↓
2. download_sfs_laws.py
   → sfs_lagboken_1990plus.jsonl (raw)
   ↓
3. filter_laws.py
   → sfs_lagboken_1990plus_filtered.jsonl (~5,000 docs)
   ↓
4. loader.py
   → List[Document] (LangChain format)
   ↓
5. chunker.py
   → List[Document] (chunked, ~10,000-50,000+ chunks)
   ↓
6. embedder.py
   → List[List[float]] (embedding vectors)
   ↓
7. ChromaDB
   → Persistent vector store (./chroma_db_test)
   ↓
8. Query Interface
   → Similarity search results
```

---

## Key Metadata Fields

### Document Metadata
- `sfs_nr`: SFS number (e.g., "1991:1867")
- `titel`: Full document title
- `typ`: "lag" or "förordning"
- `dok_id`: Document ID from Riksdagen
- `dokument_url_html`: Source URL
- `organ`: Issuing organization

### Chunk Metadata (inherits document + adds)
- `paragraf`: Paragraph identifier (e.g., "1 §", "31 a §")
- `subchunk_index`: Index if paragraph was split (0, 1, 2...)
- `merged_paragraphs`: Comma-separated list if paragraphs merged

---

## Performance Notes

- **Processing Speed:** ~100 documents/minute
- **Full Dataset:** ~5,000 documents = ~50 minutes
- **Chunking:** Creates 2-10x chunks per document (varies by document length)
- **Embedding:** Batch size 32, GPU recommended for speed
- **Storage:** ChromaDB persists to disk, can be reloaded

---

## Dependencies

### Core
- `langchain` / `langchain-community` / `langchain-core`
- `chromadb`
- `transformers`
- `torch`
- `peft`
- `llm2vec`
- `accelerate`

### Optional
- `tiktoken` (for token estimation in inspection)

---

## Common Development Tasks

### Adding New Documents
1. Run `download_sfs_laws.py` (updates START_YEAR or adds new year range)
2. Run `filter_laws.py` to filter new documents
3. Re-run indexing pipeline or implement incremental updates

### Changing Chunking Strategy
Modify `chunker.py`:
- Adjust `min_words`, `max_words`, `overlap_sentences`
- Modify paragraph detection regex
- Change merge/split logic

### Changing Embedding Model
Modify `embedder.py`:
- Update `base_model_name` and `adapter_name` in `get_embedding_function()`
- Ensure model is compatible with LLM2Vec wrapper

### Query Optimization
- Adjust `k` parameter (number of results)
- Implement re-ranking
- Add metadata filtering (by year, type, etc.)
- Experiment with query expansion

---

## Notes for AI/LLM Developers

1. **Model Architecture:** The embedding model uses a base transformer with two PEFT adapters (MNTP + supervised) that are merged before use. This is a specialized Swedish legal text embedding model.

2. **Chunking Philosophy:** Paragraph-based chunking preserves legal document structure. Paragraphs are the natural semantic units in Swedish law.

3. **Metadata Preservation:** All original document metadata is preserved in chunks, enabling filtering and citation tracking.

4. **Swedish Language:** The system is designed for Swedish legal text. Paragraph markers follow Swedish conventions (§ symbols).

5. **Memory Considerations:** Use `load_sfs_documents_lazy()` for large datasets to avoid loading all documents into memory at once.

6. **Vector Store Persistence:** ChromaDB persists automatically. Reload with same `persist_directory` and `embedding_function` to reuse.

7. **Batch Processing:** Embedding uses batch size 32. Adjust in `TTCEmbeddings.embed_documents()` if needed.

8. **Error Handling:** All modules include error handling for malformed JSON, missing fields, and API failures.

---

## File Locations

- **Data:** `data/sfs_lagboken_1990plus_filtered.jsonl`
- **Vector Store:** `./chroma_db_test/` (default)
- **Source Code:** `src/ingestion/`, `src/indexing/`, `src/inspection/`
- **Test Scripts:** `src/indexing/test_pipeline.py`, `src/indexing/quick_test.py`

