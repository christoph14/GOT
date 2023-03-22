#!/usr/bin/python3

#SBATCH --job-name=got
#SBATCH --output=../log_slurm/fgot_classification.log
#SBATCH --get-user-env
#SBATCH --mem-per-cpu=3G
#SBATCH --time 360
#SBATCH --account=thes1398
#SBATCH --ntasks=500


import itertools
import multiprocessing
import pickle

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

from utils.strategies import get_strategy


def load_dataset(path, random_state=0, train_size=10):
    with open(path, 'rb') as file:
        graphs = pickle.load(file)
    y = [G.graph['label'] for G in graphs]
    X_train, X_test, y_train, y_test = train_test_split(graphs, y, train_size=train_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def f(args):
    G1, G2 = args
    # strategy = get_strategy('fgot', it=10, tau=1, n_samples=30, epochs=1000, lr=0.2)
    strategy = get_strategy('fgot', it=10, tau=1, n_samples=30, epochs=10, lr=0.2)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    P = strategy(L1, L2)
    distance = np.linalg.norm(L1 - P.T @ L2 @ P, ord='fro')
    print('done')
    return distance

if __name__ == '__main__':
    # Load graph data set
    n_graphs = 100
    path = "../data/ENZYMES/enzymes.pkl"
    X_train, X_test, y_train, y_test = load_dataset(path, train_size=n_graphs)

    # Create arguments
    arguments = list(itertools.product(X_train, X_test))

    # Compute and save distances
    with multiprocessing.Pool() as pool:
        distances = np.reshape(pool.map(f, arguments), (n_graphs, 10))
    np.savetxt('../results/distances.csv', distances)
