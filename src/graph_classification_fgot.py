import argparse
import functools
import itertools
import multiprocessing
import os
import sqlite3
from time import time

from sklearn.metrics import zero_one_loss
import networkx as nx
import numpy as np

from utils.dataset import tud_to_networkx
from utils.strategies import get_strategy, get_filters


def compute_distance(G1, G2, strategy, strategy_args):
    strategy = get_strategy(strategy, it=10, tau=1, n_samples=30, lr=0.2, **strategy_args)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    P = strategy(L1, L2)
    # distance = np.linalg.norm(L1 - P.T @ L2 @ P, ord='fro')
    gL1 = get_filters(L1, strategy_args['filter_name'])
    gL2 = get_filters(L2, strategy_args['filter_name'])
    distance = np.trace(gL1**2) + np.trace(gL2**2) - 2 * np.trace(gL1 @ P.T @ gL2 @ P)
    return distance

if __name__ == '__main__':
    t0 = time()
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('seed', type=int, help='the used random seed')
    parser.add_argument('algorithm', type=str, help='the alignment algorithm')
    parser.add_argument('dataset', type=str, help='the benchmark data set')
    parser.add_argument('--path', type=str, default='../results/', help='the path to store the output files')
    parser.add_argument('--n_graphs', type=int, default=100, help='the number of sampled graphs')
    # fGOT arguments
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--epsilon', type=float, default=0.006)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--scale', action='store_const', const=True, default=False, help='scale soft assignment')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load graph data set
    graphs = tud_to_networkx(args.dataset)
    X = np.empty(len(graphs), dtype=object)
    X[:] = graphs
    X = rng.choice(X, args.n_graphs)
    y = np.array([G.graph['classes'] for G in X])
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.algorithm}")
    print(f'Compute distance matrix for {args.n_graphs} graphs')

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
        'scale': args.scale,
    }
    f = functools.partial(compute_distance, strategy=args.algorithm, strategy_args=strategy_args)
    with multiprocessing.Pool(number_of_cores) as pool:
        result = pool.starmap(f, itertools.product(X, X))
    distances = np.reshape(result, (len(X), len(X)))
    number_errors = np.count_nonzero(np.isnan(distances))
    if number_errors > 0:
        print(f'Warning: {number_errors} NaNs in distance matrix')
    np.fill_diagonal(distances, np.nan)

    # Compute and save accuracy
    nearest_neighbors = np.nanargmin(distances, axis=0)
    y_pred = y[nearest_neighbors]
    accuracy = args.n_graphs - zero_one_loss(y, y_pred, normalize=False)
    print(f'Accuracy: {accuracy}/{args.n_graphs}')

    # Save results in database
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(f"{args.path}/distances/{args.dataset}", exist_ok=True)
    con = sqlite3.connect(f'{args.path}/results_fgot.db', timeout=60)
    cur = con.cursor()
    try:
        cur.execute('''CREATE TABLE classification (
                           STRATEGY TEXT NOT NULL,
                           DATA TEXT NOT NULL,
                           SEED TEXT NOT NULL,
                           FILTER TEXT,
                           ACCURACY REAL,
                           unique (STRATEGY, DATA, SEED, FILTER)
                       )''')
    except sqlite3.OperationalError:
        pass

    data = (args.algorithm, args.dataset, args.seed, args.filter, int(accuracy), time() - t0)
    cur.execute("INSERT INTO classification VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT DO UPDATE SET accuracy=excluded.accuracy, time=excluded.time;", data)
    con.commit()
    cur.close()
    con.close()

    # Save distance matrix
    np.savetxt(f'{args.path}/distances/{args.dataset}/{args.algorithm}-{args.filter}#{args.seed}.csv', distances)

    print(f'Completed task in {time() - t0:.0f}s')
