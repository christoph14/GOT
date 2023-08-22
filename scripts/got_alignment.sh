#!/bin/bash

#SBATCH --job-name=got
#SBATCH --output=../log_slurm/got_alignment_%A_%a.log
#SBATCH --array=1-25
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

export CONDA_ROOT=$HOME/miniconda3
. "$CONDA_ROOT/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ~/GOT/src || exit

p_out_list=(0.05 0.1)
p_in_list=(0.7 0.8 0.9)

for p_out in ${p_out_list[*]}
do
    for p_in in ${p_in_list[*]}
    do
        python -u graph_alignment.py GOT "$SLURM_ARRAY_TASK_ID" --between_probability "$p_out" --within_probability "$p_in"
        python -u graph_alignment.py L2 "$SLURM_ARRAY_TASK_ID" --between_probability "$p_out" --within_probability "$p_in"
        python -u graph_alignment.py fGOT "$SLURM_ARRAY_TASK_ID" --filter got --epsilon 0.008 --between_probability "$p_out" --within_probability "$p_in"
        python -u graph_alignment.py random "$SLURM_ARRAY_TASK_ID" --between_probability "$p_out" --within_probability "$p_in"
        python -u graph_alignment.py GW "$SLURM_ARRAY_TASK_ID" --between_probability "$p_out" --within_probability "$p_in"
    done
done
wait