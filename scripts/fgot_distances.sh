#!/bin/bash

#SBATCH --job-name=fgot_distances
#SBATCH --output=../log_slurm/fgot_classification_%A_%a.log
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=100

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ../src

for g1 in $(seq 0 99)
do
    for g2 in $(seq 0 99)
    do
        python graph_classification_fgot.py "$g1" "$g2" "$SLURM_ARRAY_TASK_ID" &
    done
    wait
done
wait