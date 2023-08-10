"""Compute the fGOT alignments and different distance measures."""

import argparse
import functools
import itertools
import json
import multiprocessing
import os
from time import time

import numpy as np
from tqdm import tqdm

from utils.dataset import tud_to_networkx
from utils.distances import compute_distance


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('algorithm', type=str, help='the alignment algorithm')
    parser.add_argument('dataset', type=str, help='the benchmark data set')
    parser.add_argument('--seed', type=int, default=None, help='the used random seed')
    parser.add_argument('--path', type=str, default='../distances/', help='the path to store the output files')
    parser.add_argument('--max_graphs', type=int, default=None, help='the maximum number of graphs.')
    parser.add_argument(
        '--same_size',
        action='store_const', const=True, default=False,
        help='allow soft assignment instead of a permutation matrix'
    )
    # fGOT arguments
    parser.add_argument('--filter', type=str, default='got')
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    # Load graph data set
    graphs = tud_to_networkx(args.dataset)
    if args.same_size:
        values, counts = np.unique([G.number_of_nodes() for G in graphs], return_counts=True)
        n_nodes = values[np.argmax(counts)]
        graphs = [G for G in graphs if G.number_of_nodes() == n_nodes]
    X = np.empty(len(graphs), dtype=object)
    X[:] = graphs
    if args.max_graphs is not None and args.max_graphs < len(graphs):
        rng = np.random.default_rng(args.seed)
        X = rng.choice(X, args.max_graphs, replace=False)
    y = np.array([G.graph['classes'] for G in X])
    print(f"Dataset: {args.dataset}")
    epsilon_text = f" and epsilon={args.epsilon}" if args.epsilon is not None else ""
    print(f"Strategy: {args.algorithm} with filter {args.filter}" + epsilon_text)
    print(f"Seed: {args.seed}")
    print(f'Compute distance matrix for {len(X)} graphs')

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
        result = list(tqdm(pool.imap(f, itertools.product(X, X)), total=len(X)*len(X)))
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
        json.dump(data, file, indent=4)

    # Save labels and distances
    np.savetxt(f'{args.path}/{args.dataset}/labels.csv', y)
    for measure in distance_measures:
        np.savetxt(
            f'{args.path}/{args.dataset}/{args.algorithm}-{args.filter}-{args.epsilon}-{measure}.csv',
            all_distances[measure]
        )
    print(f'Completed task in {computing_time:.0f}s')
