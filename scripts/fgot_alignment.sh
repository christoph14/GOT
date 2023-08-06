#!/bin/bash

#SBATCH --job-name=fgot
#SBATCH --output=../log_slurm/fgot_alignment_%A_%a.log
#SBATCH --array=1-50
#SBATCH --time=06:00:00
#SBATCH --partition=c18m
#SBATCH --account=thes1398
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

export CONDA_ROOT=$HOME/miniconda3
. "$CONDA_ROOT/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate graph

cd ~/GOT/src || exit

python -u graph_alignment_fgot.py stochastic-fGOT "$SLURM_ARRAY_TASK_ID" --path ../results --allow_soft_assignment --lr 50 --sampling_size 5 --it 10 --tau 1 --filter sq
python -u graph_alignment_fgot.py stochastic-fGOT "$SLURM_ARRAY_TASK_ID" --path ../results --allow_soft_assignment --lr 50 --sampling_size 5 --it 10 --tau 1 --filter got
python -u graph_alignment_fgot.py QAP "$SLURM_ARRAY_TASK_ID" --path ../results --filter sq
python -u graph_alignment_fgot.py QAP "$SLURM_ARRAY_TASK_ID" --path ../results --filter got
