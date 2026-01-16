#!/bin/bash
#SBATCH --job-name=lagrag_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=01:00:00

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Auto-detect available GPU with most free memory
echo "Detecting available GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$GPU_INFO" ]; then
    echo "Warning: nvidia-smi not available, defaulting to GPU 0"
    SELECTED_GPU=0
else
    # Find GPU with most free memory (in MB)
    SELECTED_GPU=$(echo "$GPU_INFO" | awk -F', ' '{print $1, $2}' | sort -k2 -rn | head -1 | awk '{print $1}')
    FREE_MEM=$(echo "$GPU_INFO" | awk -F', ' -v gpu=$SELECTED_GPU '$1==gpu {print $2}')
    echo "Selected GPU $SELECTED_GPU with $FREE_MEM MB free memory"
fi

# Set up environment to use selected GPU
export CUDA_VISIBLE_DEVICES=$SELECTED_GPU

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null || echo "  nvidia-smi not available"

echo "GPU Processes Before Job Start:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "  No compute processes found"

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

# Install PyTorch with CUDA if not already installed
# This should be done once, but included here for completeness
# pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Set project root
PROJECT_ROOT="/data/users/spreitz/LagRag"
cd $PROJECT_ROOT

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

#REMOVE OLD CHROMADB
rm -rf ./chroma_db_pipeline

# Clear cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Run the pipeline
echo "Starting pipeline..."
python -u src/indexing/test_pipeline.py

echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="
