import argparse
import itertools
import multiprocessing
import os
import pickle
import sqlite3
from time import time

from sklearn.metrics import zero_one_loss
import networkx as nx
import numpy as np

from utils.dataset import tud_to_networkx
from utils.strategies import get_strategy


def compute_distance(G1, G2):
    strategy = get_strategy('fgot', it=10, tau=1, n_samples=30, epochs=1000, lr=0.2)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    P = strategy(L1, L2)
    distance = np.linalg.norm(L1 - P.T @ L2 @ P, ord='fro')
    return distance

if __name__ == '__main__':
    t0 = time()
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('seed', type=int, help='the used random seed')
    parser.add_argument('dataset', type=str, help='the benchmark data set')
    parser.add_argument('--path', type=str, default='../results/', help='the path to store the output files')
    parser.add_argument('--n_graphs', type=int, default=100, help='the number of sampled graphs')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load graph data set
    graphs = tud_to_networkx(args.dataset)
    X = rng.choice(graphs, args.n_graphs)
    y = np.array([G.graph['classes'] for G in X])
    print(f"Dataset: {args.dataset}")
    print(f'Compute distance matrix for {args.n_graphs} graphs')

    # Determine number of cores
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()
    print(f'Use {number_of_cores} cores')

    # Compute and save distances
    with multiprocessing.Pool(number_of_cores) as pool:
        result = pool.starmap(compute_distance, itertools.product(X, X))
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
    con = sqlite3.connect(f'{args.path}/results_fgot.db', timeout=60)
    cur = con.cursor()
    try:
        cur.execute('''CREATE TABLE classification (
                           STRATEGY TEXT NOT NULL,
                           DATA TEXT NOT NULL,
                           SEED TEXT NOT NULL,
                           ACCURACY REAL,
                           unique (STRATEGY, DATA, SEED)
                       )''')
    except sqlite3.OperationalError:
        pass

    data = ('Pgot', args.dataset, args.seed, int(accuracy))
    cur.execute("INSERT INTO classification VALUES (?, ?, ?, ?) "
                "ON CONFLICT DO UPDATE SET accuracy=excluded.accuracy;", data)
    con.commit()
    cur.close()
    con.close()
    print(f'Completed task in {time() - t0:.0f}s')
