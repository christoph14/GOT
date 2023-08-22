#!/bin/bash

#SBATCH --job-name=fgot
#SBATCH --output=../log_slurm/fgot_distances_%A_%a.log
#SBATCH --array=0-9
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

export CONDA_ROOT=$HOME/miniconda3
. "$CONDA_ROOT/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ../src || exit

epsilon_range=(0.002 0.004 0.006 0.008 0.01 0.02 0.04 0.06 0.08 0.1)

srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u create_distance_matrix.py "$1" "$2" --filter "$3" --epsilon ${epsilon_range[$SLURM_ARRAY_TASK_ID]}
