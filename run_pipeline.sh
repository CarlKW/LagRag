#!/bin/bash
#SBATCH --job-name=lagrag_pipeline
#SBATCH --output=logs/pipeline%j.out
#SBATCH --error=logs/pipeline%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1


# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Resource Information
echo "=========================================="
echo "Resource Information"
echo "=========================================="
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Allocated GPUs: $SLURM_GPUS"
echo ""
echo "CPU Count: $(nproc)"
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "  nvidia-smi not available"
echo "=========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0

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

# Run the RAG pipeline with queries
echo "=========================================="
echo "Starting RAG Pipeline"
echo "=========================================="

# Set ChromaDB path (use existing database from indexing)
CHROMA_DB="./chroma_db_pipeline"
# Alternative: use test database if pipeline database doesn't exist
# CHROMA_DB="./chroma_db_test"

# Output file for results/metrics
RESULTS_FILE="logs/pipeline_results_${SLURM_JOB_ID}.json"
METRICS_FILE="logs/pipeline_metrics_${SLURM_JOB_ID}.txt"
QUERIES_FILE="${PROJECT_ROOT}/src/generation/queries.txt"

echo "ChromaDB path: $CHROMA_DB"
echo "Queries file: $QUERIES_FILE"
echo "Results will be saved to: $RESULTS_FILE"
echo "Metrics will be saved to: $METRICS_FILE"
echo ""

# Run the pipeline
python -u src/pipeline.py \
    --chroma-db "$CHROMA_DB" \
    --collection-name "sfs_paragraphs" \
    --retriever-type "reranking" \
    --k-initial 50 \
    --k-final 1 \
    --reranker-model "jinaai/jina-reranker-v2-base-multilingual" \
    --lm-model "gpt2" \
    --max-retrieval-rounds 2 \
    --high-threshold 0.75 \
    --low-threshold 0.40 \
    --query-file "$QUERIES_FILE" \
    --output "$RESULTS_FILE" \
    2>&1 | tee "$METRICS_FILE"

# Extract and display summary metrics
echo ""
echo "=========================================="
echo "Pipeline Metrics Summary"
echo "=========================================="

if [ -f "$RESULTS_FILE" ]; then
    python -u -c "
import json
import sys
from pathlib import Path

try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    
    if not results:
        print('No results found.')
        sys.exit(0)
    
    total_queries = len(results)
    answered = sum(1 for r in results if r.get('status') == 'answered')
    need_context = sum(1 for r in results if r.get('status') == 'need_more_context')
    cannot_answer = sum(1 for r in results if r.get('status') == 'cannot_answer')
    
    avg_score = sum(r.get('score', 0) for r in results) / total_queries if total_queries > 0 else 0
    avg_retrieval_rounds = sum(r.get('num_retrieval_rounds', 0) for r in results) / total_queries if total_queries > 0 else 0
    avg_chunks_used = sum(r.get('num_chunks_used', 0) for r in results) / total_queries if total_queries > 0 else 0
    
    print(f'Total Queries: {total_queries}')
    print(f'Answered (high confidence): {answered} ({answered/total_queries*100:.1f}%)')
    print(f'Need More Context: {need_context} ({need_context/total_queries*100:.1f}%)')
    print(f'Cannot Answer: {cannot_answer} ({cannot_answer/total_queries*100:.1f}%)')
    print(f'')
    print(f'Average Score: {avg_score:.4f}')
    print(f'Average Retrieval Rounds: {avg_retrieval_rounds:.2f}')
    print(f'Average Chunks Used: {avg_chunks_used:.2f}')
    print(f'')
    print('Per-query results:')
    print('-' * 80)
    for i, r in enumerate(results, 1):
        print(f'{i}. Query: {r.get(\"query\", \"N/A\")[:60]}...')
        print(f'   Status: {r.get(\"status\", \"N/A\")} | Score: {r.get(\"score\", 0):.4f} | Rounds: {r.get(\"num_retrieval_rounds\", 0)} | Chunks: {r.get(\"num_chunks_used\", 0)}')
        print(f'   Answer: {r.get(\"answer\", \"N/A\")[:100]}...')
        print('')
except Exception as e:
    print(f'Error reading results: {e}')
    sys.exit(1)
"
else
    echo "Results file not found: $RESULTS_FILE"
fi


echo "=========================================="
echo "Full results saved to: $RESULTS_FILE"
echo "Metrics saved to: $METRICS_FILE"
echo "=========================================="

echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="

