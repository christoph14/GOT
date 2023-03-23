#!/home/we661512/miniconda3/envs/graph/bin/python
#SBATCH --job-name=fgot_distances
#SBATCH --output=../log_slurm/fgot_classification_%A.log
#SBATCH --mem-per-cpu=3G
#SBATCH --time 360
#SBATCH --account=thes1398
#SBATCH --ntasks=200


import argparse
import itertools
import multiprocessing
import os
import pickle
import sqlite3
import sys

from sklearn.metrics import zero_one_loss

sys.path.append(os.getcwd())

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

from utils.strategies import get_strategy



def compute_distance(args):
    G1, G2 = args
    strategy = get_strategy('fgot', it=10, tau=1, n_samples=30, epochs=1000, lr=0.2)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    P = strategy(L1, L2)
    distance = np.linalg.norm(L1 - P.T @ L2 @ P, ord='fro')
    return distance

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('seed', type=int, help='the used random seed')
    parser.add_argument('--path', type=str, default='../results/', help='the path to store the output files')
    args = parser.parse_args()
    random_state = args.seed

    # Load graph data set
    n_graphs = 100
    path = "../data/ENZYMES/enzymes.pkl"
    with open(path, 'rb') as file:
        graphs = pickle.load(file)
    y = np.array([G.graph['label'] for G in graphs])
    X_train, X_test, y_train, y_test = train_test_split(graphs, y, test_size=n_graphs, random_state=random_state)

    # Create arguments
    arguments = list(itertools.product(X_train, X_test))

    # Compute and save distances
    with multiprocessing.Pool(200) as pool:
        result = pool.map(compute_distance, arguments)
    distances = np.reshape(result, (len(X_train), len(X_test)))
    np.savetxt(f'../results/distances_{random_state}.csv', distances)

    # Compute and save accuracy
    nearest_neighbors = np.argmin(distances, axis=0)
    y_pred = y_train[nearest_neighbors]
    accuracy = n_graphs - zero_one_loss(y_test, y_pred, normalize=False)

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
