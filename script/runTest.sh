#!/bin/bash
#SBATCH --job-name=runTest
#SBATCH --partition=general
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=name@example.com

####################################################################
# Load the required modules.
####################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

####################################################################
# Run the test.
####################################################################

python3 -m pytest -v source
