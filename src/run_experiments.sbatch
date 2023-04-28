#!/bin/bash
#
#SBATCH --job-name=got
#SBATCH --output=../log_slurm/slurm-%A_%a.log
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time 150
#SBATCH --account=thes1398

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

command=$(sed -n ${SLURM_ARRAY_TASK_ID}p < $1)
eval $command
wait
