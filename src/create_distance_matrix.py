import argparse
import functools
import itertools
import json
import multiprocessing
import os
from time import time

import numpy as np

from utils.dataset import tud_to_networkx
from utils.distances import compute_distance

if __name__ == '__main__':
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
    print(f"Strategy: {args.algorithm} with filter {args.filter} and epsilon={args.epsilon}")
    print(f"Seed: {args.seed}")
    print(f'Compute distance matrix for {len(graphs)} graphs')

    # Determine number of cores
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()
    print(f'Use {number_of_cores} cores')

    distance_measures = [
        'got', 'got-label', 'approx', 'approx-label', 'fro', 'fro-label',
    ]

    # Compute and save distances
    strategy_args = {
        'filter_name': args.filter,
        'epsilon': args.epsilon,
        'scale': True,
        'seed': args.seed,
    }
    f = functools.partial(compute_distance, strategy=args.algorithm, strategy_args=strategy_args)
    t0 = time()
    with multiprocessing.Pool(number_of_cores) as pool:
        result = pool.starmap(f, itertools.product(X, X))
    computing_time = time() - t0
    all_distances = {k: np.reshape([dic[k] for dic in result], (len(X), len(X))) for k in distance_measures}

    # Save computing time
    os.makedirs(f"{args.path}/{args.dataset}", exist_ok=True)
    try:
        with open(f'{args.path}/{args.dataset}/time.json', "rt") as file:
            data = json.load(file)
    except IOError:
        data = {}

    data[f"{args.algorithm}-{args.filter}-{args.epsilon}"] = computing_time
    with open(f'{args.path}/{args.dataset}/time.json', "wt") as file:
        json.dump(data, file)

    # Save labels and distances
    np.savetxt(f'{args.path}/{args.dataset}/labels.csv', y)
    for measure in distance_measures:
        np.savetxt(
            f'{args.path}/{args.dataset}/{args.algorithm}-{args.filter}-{args.epsilon}-{measure}.csv',
            all_distances[measure]
        )
    print(f'Completed task in {computing_time:.0f}s')
