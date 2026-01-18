#!/bin/bash
#SBATCH --job-name=retriever_eval
#SBATCH --output=logs/eval_retriever%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:L4

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
# After line 25, add:
echo "GPU Allocation Check:"
echo "  SLURM_GPUS: '${SLURM_GPUS:-NOT SET}'"
echo "  CUDA_VISIBLE_DEVICES: '${CUDA_VISIBLE_DEVICES:-NOT SET}'"

echo ""
echo "CPU Count: $(nproc)"
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null || echo "  nvidia-smi not available"
echo "=========================================="

# Set up environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

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

# Run the retriever evaluation
echo "=========================================="
echo "Starting Retriever Evaluation"
echo "=========================================="
echo ""

# Run the evaluation script
PYTHONUNBUFFERED=1 python src/retrieval/retriever_eval.py \
    2>&1 | tee "$REPORT_FILE"

echo "=========================================="
echo "Evaluation completed"
echo "Report saved to: $REPORT_FILE"
echo "=========================================="

echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="