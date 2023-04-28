#!/bin/bash

#SBATCH --job-name=fgot
#SBATCH --output=../log_slurm/fgot_classification_%A_%a.log
#SBATCH --array=0-99
#SBATCH --time=06:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ~/GOT/src

python -u graph_classification_fgot.py "$SLURM_ARRAY_TASK_ID" IPFP ENZYMES
