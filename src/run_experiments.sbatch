#!/bin/bash
#
#SBATCH --job-name=got
#SBATCH --output=../log_slurm/slurm-%A_%a.log
#SBATCH --get-user-env
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time 60

module load python/3.9.6

command=$(sed -n ${SLURM_ARRAY_TASK_ID}p < $1)
eval $command
wait
