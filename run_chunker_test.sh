#!/bin/bash
#SBATCH --job-name=lagrag_chunker
#SBATCH --output=logs/chunker_%j.out
#SBATCH --error=logs/chunker_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on Minerva's module system)
# Uncomment and modify as needed for your Minerva setup
# module load Python/3.11.0-GCCcore-12.3.0

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

# Run the chunker test (no GPU needed)
echo "Starting chunker test..."
python -u src/indexing/chunker.py

echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="

