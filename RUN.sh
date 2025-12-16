#!/bin/bash
#SBATCH --job-name=lagrag_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# #SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


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
PROJECT_ROOT="/data/users/kyleback/LagRag"
cd $PROJECT_ROOT

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Run the pipeline
echo "Starting pipeline..."
# python -u src/indexing/test_pipeline.py
python -u src/generator/retriever.py

echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="