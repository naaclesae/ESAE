#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=name@example.com
#SBATCH --array=0-7

####################################################################
# Load the required modules.
####################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

####################################################################
# Prepare the MS MARCO dataset.
####################################################################

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--embedding miniCPM --gpuDevice 0 --batchSize 128 --numShards 152 --workerCnt $SLURM_ARRAY_TASK_COUNT"
python3 -m $ENTRYPOINT preparePassageEmbeddings $SHAREDCMDS --workerIdx $SLURM_ARRAY_TASK_ID

ENTRYPOINT="source.dataset.textRetrieval.msMarco"
SHAREDCMDS="--embedding bgeBase --gpuDevice 0 --batchSize 512 --numShards 51 --workerCnt $SLURM_ARRAY_TASK_COUNT"
python3 -m $ENTRYPOINT preparePassageEmbeddings $SHAREDCMDS --workerIdx $SLURM_ARRAY_TASK_ID
