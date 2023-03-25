#!/bin/bash

#SBATCH --job-name=fgot_distances
#SBATCH --output=../log_slurm/fgot_classification_%A_%a.log
#SBATCH --array=0-2
#SBATCH --time=01:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=6

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ../src

python graph_classification_fgot.py "$SLURM_ARRAY_TASK_ID"
