#!/bin/bash

#SBATCH --job-name=fgot
#SBATCH --output=../log_slurm/fgot_distances_%A.log
#SBATCH --time=12:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=24

export CONDA_ROOT=$HOME/miniconda3
. "$CONDA_ROOT/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ../src || exit

for epsilon in 0.002 0.004 0.006 0.008 0.01 0.02 0.04 0.06 0.08 0.1; do
    srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u create_distance_matrix.py fGOT "$1" --filter "$2" --epsilon $epsilon &
done
wait
