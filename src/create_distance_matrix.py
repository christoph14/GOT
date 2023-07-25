import argparse
import functools
import itertools
import multiprocessing
import os
from time import time

import numpy as np

from utils.dataset import tud_to_networkx
from utils.distances import compute_distance

if __name__ == '__main__':
    t0 = time()
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('algorithm', type=str, help='the alignment algorithm')
    parser.add_argument('dataset', type=str, help='the benchmark data set')
    parser.add_argument('--seed', type=int, default=None, help='the used random seed')
    parser.add_argument('--path', type=str, default='../distances/', help='the path to store the output files')
    # fGOT arguments
    parser.add_argument('--filter', type=str, default='got')
    parser.add_argument('--epsilon', type=float, default=0.006)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    # Load graph data set
    graphs = tud_to_networkx(args.dataset)
    X = np.empty(len(graphs), dtype=object)
    X[:] = graphs
    y = np.array([G.graph['classes'] for G in X])
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.algorithm}")
    print(f'Compute distance matrix for {len(graphs)} graphs')

    # Determine number of cores
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()
    print(f'Use {number_of_cores} cores')

    # Compute and save distances
    strategy_args = {
        'filter_name': args.filter,
        'epsilon': args.epsilon,
        'epochs': args.epochs,
        'scale': True,
        'seed': args.seed,
    }
    f = functools.partial(compute_distance, strategy=args.algorithm, strategy_args=strategy_args, distance='got')
    with multiprocessing.Pool(number_of_cores) as pool:
        result = pool.starmap(f, itertools.product(X, X))
    distances = np.reshape(result, (len(X), len(X)))
    number_errors = np.count_nonzero(np.isnan(distances))
    if number_errors > 0:
        print(f'Warning: {number_errors} NaNs in distance matrix')
    np.fill_diagonal(distances, 0)

    # Save distance matrix and labels
    os.makedirs(f"{args.path}/{args.dataset}", exist_ok=True)
    np.savetxt(f'{args.path}/{args.dataset}/{args.algorithm}-{args.filter}-{args.epsilon}.csv', distances)
    np.savetxt(f'{args.path}/{args.dataset}/labels.csv', y)

    print(f'Completed task in {time() - t0:.0f}s')
