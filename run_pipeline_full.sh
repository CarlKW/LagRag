#!/bin/bash
#SBATCH --job-name=lagrag_full
#SBATCH --output=logs/pipeline_full_%j.out
#SBATCH --error=logs/pipeline_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on Minerva's module system)
# Uncomment and modify as needed for your Minerva setup
# module load Python/3.11.0-GCCcore-12.3.0
# module load CUDA/12.1.0

# Activate virtual environment if you have one
# If using conda:
# source activate lagrag_env
# If using venv:
# source venv/bin/activate

# Set project root
PROJECT_ROOT="/data/users/spreitz/LagRag"
cd $PROJECT_ROOT

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Run the pipeline with all documents (modify test_pipeline.py to accept num_docs=None)
echo "Starting full pipeline (all documents)..."
python -u -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from src.indexing.test_pipeline import run_pipeline_test

project_root = Path('.')
jsonl_file = project_root / 'data' / 'sfs_lagboken_1990plus_filtered.jsonl'

custom_queries = [
    'När får bidraget betalas ut för investeringsbidrag?',
    'skatt för singapor',
]

vectorstore = run_pipeline_test(
    jsonl_path=str(jsonl_file),
    persist_directory='./chroma_db_full',
    num_docs=None,  # Process all documents
    test_queries=custom_queries
)

if vectorstore:
    print('\n' + '='*80)
    print('Pipeline test completed successfully!')
    print('Vector store saved to: ./chroma_db_full')
    print('='*80)
"

echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="

