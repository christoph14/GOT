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
. "$CONDA_ROOT/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ~/GOT/src || exit

p_out_list=(0.05 0.1)
for p_out in ${p_out_list[*]}
do
  srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u graph_alignment.py GOT "$SLURM_ARRAY_TASK_ID" --path ../results_"$p_out" --between_probability "$p_out" &
  srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u graph_alignment.py L2 "$SLURM_ARRAY_TASK_ID" --path ../results_"$p_out" --between_probability "$p_out" &
  srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u graph_alignment.py fGOT "$SLURM_ARRAY_TASK_ID" --filter got --epsilon 0.008 --path ../results_"$p_out" --between_probability "$p_out" &
  srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u graph_alignment.py random "$SLURM_ARRAY_TASK_ID" --path ../results_"$p_out" --between_probability "$p_out" &
  srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u graph_alignment.py GW "$SLURM_ARRAY_TASK_ID" --path ../results_"$p_out" --between_probability "$p_out" &
done
wait