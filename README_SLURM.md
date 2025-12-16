# SLURM Batch Scripts for LagRag

This directory contains SLURM batch scripts for running the LagRag pipeline on the Minerva cluster.

## Available Scripts

### 1. `run_pipeline.slurm`
**Purpose**: Run the full pipeline test with default settings (limited documents for testing)

**Resources**:
- GPU: 1 GPU
- Memory: 32GB
- CPUs: 8
- Time: 4 hours
- Partition: gpu

**Usage**:
```bash
sbatch run_pipeline.slurm
```

**What it does**:
- Loads documents from `data/sfs_lagboken_1990plus_filtered.jsonl`
- Chunks documents
- Creates embeddings (requires GPU)
- Builds vector store
- Tests queries

### 2. `run_pipeline_full.slurm`
**Purpose**: Run the full pipeline on ALL documents (production run)

**Resources**:
- GPU: 1 GPU
- Memory: 64GB
- CPUs: 16
- Time: 12 hours
- Partition: gpu

**Usage**:
```bash
sbatch run_pipeline_full.slurm
```

**What it does**:
- Same as `run_pipeline.slurm` but processes ALL documents
- Saves to `./chroma_db_full` instead of `./chroma_db_test`

### 3. `run_chunker_test.slurm`
**Purpose**: Test chunking only (no GPU needed)

**Resources**:
- Memory: 16GB
- CPUs: 8
- Time: 2 hours
- Partition: cpu (no GPU)

**Usage**:
```bash
sbatch run_chunker_test.slurm
```

**What it does**:
- Tests the chunker on sample documents
- Writes output to `data/chunks_output.txt`
- Useful for debugging chunking issues

## Setup Instructions

### 1. Create logs directory
```bash
mkdir -p logs
```

### 2. Configure environment
Edit the SLURM scripts and uncomment/modify:
- Module loading commands (if Minerva uses modules)
- Virtual environment activation (conda or venv)
- PyTorch installation (if needed)

### 3. Install dependencies
On a GPU node or login node:
```bash
# Install CPU-only dependencies
pip install -r requirements-minerva.txt

# On a GPU node, install PyTorch with CUDA
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Set HuggingFace cache (optional but recommended)
The scripts set:
```bash
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface
```

Make sure `/scratch/$USER` exists and has sufficient space.

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View output
```bash
# View latest output
tail -f logs/pipeline_*.out

# View latest errors
tail -f logs/pipeline_*.err
```

### Cancel a job
```bash
scancel <job_id>
```

## Customization

### Modify resources
Edit the `#SBATCH` directives at the top of each script:
- `--time`: Maximum runtime
- `--mem`: Memory requirement
- `--cpus-per-task`: Number of CPUs
- `--gres=gpu:1`: Number of GPUs

### Modify script parameters
Edit the Python command in the script to:
- Change number of documents: `num_docs=100`
- Change output directory: `persist_directory='./chroma_db_custom'`
- Add custom queries: Modify the `test_queries` list

## Troubleshooting

### Out of memory
- Increase `--mem` in the SLURM script
- Process fewer documents at once

### GPU not found
- Check that `--gres=gpu:1` is set
- Verify you're using the `gpu` partition
- Check GPU availability: `sinfo -p gpu`

### Module errors
- Uncomment and configure module loading commands
- Or use conda/venv instead

### Import errors
- Ensure `PYTHONPATH` is set correctly
- Activate your virtual environment
- Install all dependencies from `requirements-minerva.txt`

## Example: Running with custom parameters

Create a custom script or modify the Python command:

```bash
python -u -c "
from src.indexing.test_pipeline import run_pipeline_test
from pathlib import Path

vectorstore = run_pipeline_test(
    jsonl_path='data/sfs_lagboken_1990plus_filtered.jsonl',
    persist_directory='./chroma_db_custom',
    num_docs=500,  # Process 500 documents
    test_queries=['your', 'custom', 'queries']
)
"
```

