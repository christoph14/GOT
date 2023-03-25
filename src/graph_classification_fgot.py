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

from utils.strategies import get_strategy


def compute_distance(G1, G2):
    strategy = get_strategy('fgot', it=10, tau=1, n_samples=30, epochs=10, lr=0.2)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    P = strategy(L1, L2)
    distance = np.linalg.norm(L1 - P.T @ L2 @ P, ord='fro')
    return distance

if __name__ == '__main__':
    # Parse arguments
    print('Start job')
    t0 = time()
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('seed', type=int, help='the used random seed')
    parser.add_argument('--path', type=str, default='../results/', help='the path to store the output files')
    parser.add_argument('--n_graphs', type=int, default=100, help='the number of sampled graphs')
    args = parser.parse_args()
    t1 = time()
    print(f'Args parsed in {t1 - t0:.3f}s')

    rng = np.random.default_rng(args.seed)

    # Load graph data set
    path = "../data/ENZYMES/enzymes.pkl"
    with open(path, 'rb') as file:
        graphs = pickle.load(file)
    X = rng.choice(graphs, args.n_graphs)
    y = np.array([G.graph['label'] for G in X])
    t2 = time()
    print(f'Loaded graphs in {t2 - t1:.3f}s')

    # Create arguments
    arguments = list(itertools.product(X, X))

    # Compute and save distances
    with multiprocessing.Pool(6) as pool:
        result = pool.starmap(compute_distance, arguments)
    distances = np.reshape(result, (len(X), len(X)))
    np.savetxt(f'../results/distances_{args.seed}.csv', distances)

    # Compute and save accuracy
    nearest_neighbors = np.argmin(distances, axis=0)
    y_pred = y[nearest_neighbors]
    accuracy = args.n_graphs - zero_one_loss(y, y_pred, normalize=False)
    print(f'Accuracy: {accuracy}')

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

    data = ('Pgot', 'ENZYMES', args.seed, int(accuracy))
    cur.execute("INSERT INTO classification VALUES (?, ?, ?, ?) "
                "ON CONFLICT DO UPDATE SET accuracy=excluded.accuracy;", data)
    con.commit()
    cur.close()
    con.close()
    print(f'Completed task in {time() - t0:.3f}s.')
