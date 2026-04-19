#!/bin/bash
#SBATCH --partition=A100devel
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --nodelist=node-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=test_env
#SBATCH --output=log/test_env_%j.out
#SBATCH --error=log/test_env_%j.err

# Clear any inherited modules and set the correct MODULEPATH for AMD
module purge
module use /software/easybuild-AMD_A100/modules/all

# Load the exact Python and CUDA modules (AMD versions)
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Create log folder
mkdir -p log

# Activate virtual environment
source ~/Disentangling-Memory-and-Reasoning/venv/bin/activate

# Run the test
echo "=== Starting test at $(date) ==="
python test_env.py
echo "=== Test finished at $(date) ==="