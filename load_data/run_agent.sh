#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=qwen_data_gen
#SBATCH --output=log/data_gen_%j.out
#SBATCH --error=log/data_gen_%j.err

# Set model directory
export HF_HOME=~/models/
mkdir -p $HF_HOME

# Module loading
module purge
module use /software/easybuild-AMD_A100/modules/all
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Activate virtual environment
source ~/Disentangling-Memory-and-Reasoning/venv/bin/activate


# Ensure necessary libraries for local HuggingFace inference are present
# pip install -q --upgrade transformers accelerate bitsandbytes sentencepiece tiktoken

cd ~/Disentangling-Memory-and-Reasoning/load_data/

echo "=== Starting Data Generation at $(date) ==="
# Example: Run StrategyQA on train split
python data_agent-qwen.py --dataset StrategyQA --mode train
# python data_agent.py --dataset StrategyQA --mode train
echo "=== Process finished at $(date) ==="