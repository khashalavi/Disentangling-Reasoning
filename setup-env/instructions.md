best Option is to make it in an interative session

 	# 1. Get an interactive session on an A100 node
  	srun --partition=A100devel --time=00:30:00 --nodes=1 --nodelist=node-06 --pty bash
	srun --partition=A100devel --time=01:00:00 --nodes=1 --nodelist=node-06 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash <-- es hat damit funktioniert 

  	# 2. Now check the available Python modules (on the AMD node)
	module avail python


check functionality
	module load Python/3.10.4-GCCcore-11.3.0   # or whatever name is correct
	python --version
	python -c "print('Works on AMD!')"


then laod the model and create virtual ennviromennt and Import the requirement txt file 
	cd ~/Disentangling-Memory-and-Reasoning
	python -m venv venv
	source venv/bin/activate
	pip install --upgrade pip
	sed -i 's/absl-py==2.1.0/absl-py==1.4.0/g' requirements.txt
	sed -i '/torchaudio==/d' requirements.txt
	pip install -r requirements.txt


load CUDA and test GPU access

	module avail cuda          # see available versions
	module load CUDA/12.1      # or CUDA/12.1.1, etc. (use the version that matches torch's CUDA)




SLURM
============================================================================================================

writing scriptgsg requires me to run it, by loading the AMD stack 


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