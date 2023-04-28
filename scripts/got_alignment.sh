#!/bin/bash

#SBATCH --job-name=got
#SBATCH --output=../log_slurm/got_alignment_%A_%a.log
#SBATCH --array=0-24
#SBATCH --time=06:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=24

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ~/GOT/src

algorithms=(GW GOT L2 random QAP)
for algo in ${algorithms[*]}
do
  python -u graph_alignment.py "$algo" "$SLURM_ARRAY_TASK_ID" --path ../results --allow_soft_assignment &
done
