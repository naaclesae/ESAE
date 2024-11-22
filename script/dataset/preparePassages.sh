#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=general
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=name@example.com

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
# Prepare the MS MARCO dataset.
##############################################################################

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--numShards 4"
python3 -m $ENTRYPOINT preparePassages $SHAREDCMDS
