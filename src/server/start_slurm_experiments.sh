#! /bin/sh

num_lines=$(cat experiment_commands.txt | wc -l)
num_parallel=600

echo "sbatch --array=1-$num_lines%$num_parallel run_experiments.sbatch experiment_commands.txt"
sbatch --array=1-$num_lines%$num_parallel run_experiments.sbatch experiment_commands.txt
wait
