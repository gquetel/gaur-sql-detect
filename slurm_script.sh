#!/bin/bash
#SBATCH --job-name=training-gpu       # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=cpu              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:0                  # Request 1 GPU
#SBATCH --cpus-per-task=16            # Request 8 CPU cores
#SBATCH --mem=32G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/xp-gaur/
source venv-3.12.3/bin/activate
cd scripts

# Execute the Python script with specific arguments
srun python ./run_eval.py --use-datadir

# Print job completion time
echo "Job finished at: $(date)"
