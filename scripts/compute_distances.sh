#!/bin/bash

#SBATCH --job-name=fgot
#SBATCH --output=../log_slurm/fgot_classification_%A.log
#SBATCH --time=06:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=24

export CONDA_ROOT=$HOME/miniconda3
. "$CONDA_ROOT/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ../src || exit

for epsilon in 0.003 0.006 0.009 0.015 0.03; do
    srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u create_distance_matrix.py fGOT "$1" --filter "$2" --epsilon $epsilon &
done
wait
