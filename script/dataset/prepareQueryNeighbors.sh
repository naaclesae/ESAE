#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=256G
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
SHAREDCMDS="--embedding miniCPM --gpuDevice 0 1 --batchSize 2048 --topK 128"
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition train
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition dev
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition eval

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--embedding bgeBase --gpuDevice 0 1 --batchSize 2048 --topK 128"
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition train
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition dev
python3 -m $ENTRYPOINT prepareQueryNeighbors $SHAREDCMDS --partition eval
