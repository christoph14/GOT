# Optimal Transport Distance for Graphs

This repository contains the code for the thesis "Optimal Transport Distance for Graphs".

## Setup

The `requirements.txt` file contains all required Python packages. To install them, run the following command:

    pip install -r requirements.txt

## Structure
The repository is structured as follows:

 - The `notebooks` folder contains Jupyter notebooks that are mainly used for plotting results.
 - The `scripts` folder contains bash scripts that are used to execute multiple experiments on a Slurm server.
 - The `src` folder contains the main code.
 - The `graphkernels-review` is a submodule that is forked from [graphkernels-review](https://github.com/BorgwardtLab/graphkernels-review). To install the requirements for this submodule, follow the instructions in the `graphkernels-review` readme. The code requires the `graphkernels` package, which is difficult to install on Windows and might cause problems. Hence, the requirements are not contained in the main repository.

## Run Experiments
Most of the experiments use the following arguments:
 - `STRATEGY` is the alignment algorithm and can be `[GOT, L2, fGOT, stochastic-fGOT, GW, RRWM, QAP, blowup-QAP, random]`.
 - `SEED` is the used random seed and can be any integer.
 - `FILTER` is the graph signal filter. This argument is only required for some strategies (e.g., fGOT and QAP). Valid filters are
   -  `got` for $g(L) = \sqrt{L^\dagger}$
   -  `sq` for $g(L) = L^2$
   -  `L` for $g(L) = L$
   -  `sqrtL` for $g(L) = \sqrt{L}$
   -  `heat` for $g(L) = \exp{-\tau L}$

### Graph Alignment Experiments (Sections 4.1 and 4.2)
To run the original GOT alignment experiments (Section 4.1), use the following command:

    python src/graph_alignment.py STRATEGY SEED --filter FILTER --path results

For further parameters, such as the parameters of the stochastic block model or the GOT hyperparameters, see the documentation of the file.

To run the QAP alignment experiments (Section 4.2), use the following command:

    python src/graph_alignment_fGOT.py STRATEGY SEED --filter FILTER --path results

To run the QAP alignment experiments with varying graph sizes (Section 4.2), use the following command:

    python src/graph_alignment_fGOT.py STRATEGY SEED --filter FILTER --path results --add_noise

For the first two alignment tasks, we use the Hungarian method to ensure that the computed alignment is a permutation matrix. This allows for a fair comparison of the strategies since different types of alignments might achieve better results. However, it changes the computed alignment (especially of the fGOT algorithm). If this is not desired, use the argument `--allow_soft_assignment`. If `--add_noise` is used, soft alignments are automatically allowed since permutation matrices are not possible in this case.
The results of the alignment tasks are stored in a database in the `results` folder.

The scripts in the `scripts` folder use the above commands and can be used to run multiple experiments in parallel.
To evaluate the results, use the notebook `visualize_results`.

### Experimental Evaluation of the fGOT Framework (Section 4.3)
The classification evaluation requires precomputed distance matrices. To compute the distance matrix for one strategy on one dataset, use the following command:

    python src/create_distance_matrix.py STRATEGY DATASET --filter FILTER --path distances --max_graphs MAX_GRAPHS

The `DATASET` is the name of one TUDataset. `MAX_GRAPHS` is the maximum number of sampled graphs. To create samples with constant graph size, use the argument `same_size`. For fGOT, the argument `--epsilon` must be specified.

To evaluate the classification performance, use the following command:

    python src/evaluate_classification.py STRATEGY DATASET --filter FILTER --path distances

The command evaluates the performance for each available epsilon value.

### Comparison with Graph Kernels (Section 4.4)
For the comparison with graph kernels, follow the instructions in the submodule `graphkernels-review`.
