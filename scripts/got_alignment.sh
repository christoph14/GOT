#!/bin/bash

#SBATCH --job-name=got
#SBATCH --output=../log_slurm/got_alignment_%A_%a.log
#SBATCH --array=0-24
#SBATCH --time=06:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ~/GOT/src || exit

algorithms=(GOT L2 GW fGOT QAP random)
for algo in ${algorithms[*]}
do
  srun -N1 -n1 -c1 --exact python -u graph_alignment.py "$algo" "$SLURM_ARRAY_TASK_ID" --path ../results --allow_soft_assignment &
done
