import argparse
import functools
import itertools
import multiprocessing
import os
from time import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils.dataset import tud_to_networkx
from utils.strategies import get_strategy, get_filters


def compute_distance(G1, G2, strategy, strategy_args):
    strategy = get_strategy(strategy, it=10, tau=1, n_samples=30, lr=0.2, epochs=1000, **strategy_args)
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
    parser = argparse.ArgumentParser(description='Evaluates fGOT parameters on given data set.')
    parser.add_argument('dataset', type=str, help='the benchmark data set')
    parser.add_argument('--seed', type=int, default=None, help='the used random seed')
    parser.add_argument('--path', type=str, default='../results/', help='the path to store the output files')
    parser.add_argument('--n_graphs', type=int, default=100, help='the number of sampled graphs')
    parser.add_argument('--filter', type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load graph data set
    graphs = tud_to_networkx(args.dataset)
    X = np.empty(len(graphs), dtype=object)
    X[:] = graphs
    n_graphs = min(args.n_graphs, len(X))
    X = rng.choice(X, n_graphs, replace=False)
    y = np.array([G.graph['classes'] for G in X])
    print(f"Evaluate parameters on data set {args.dataset}")
    print(f'Compute distance matrix for {n_graphs} graphs')

    # Determine number of cores
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()
    print(f'Use {number_of_cores} cores')

    # Compute and save distances
    strategy_args = {
        'filter_name': args.filter,
        'scale': True,
    }
    parameters = np.round(np.arange(0.004, 0.021, 0.001), 4)
    classifiers = {
        '1NN': KNeighborsClassifier(n_neighbors=1, metric='precomputed'),
        '5NN': KNeighborsClassifier(n_neighbors=5, metric='precomputed'),
        '10NN': KNeighborsClassifier(n_neighbors=10, metric='precomputed'),
    }
    for C in 10. ** np.arange(-3, 4):
        classifiers[f'SVM-{C}'] = SVC(kernel='precomputed', C=C, max_iter=100000)

    scores = pd.DataFrame(index=np.array(classifiers.keys()))
    for epsilon in parameters:
        strategy_args['epsilon'] = epsilon
        f = functools.partial(compute_distance, strategy='fGOT', strategy_args=strategy_args)
        with multiprocessing.Pool(number_of_cores) as pool:
            result = pool.starmap(f, itertools.product(X, X))
        distances = np.reshape(result, (len(X), len(X)))
        gamma = 0.2
        K = np.exp(-gamma * distances)
        distances -= np.min(distances)

        number_errors = np.count_nonzero(np.isnan(distances))
        number_errors += np.count_nonzero(np.isinf(distances))
        if number_errors > 0:
            print(f'Warning: {number_errors} NaNs/infs in distance matrix')
            scores[epsilon] = np.zeros(len(classifiers))
            continue

        distances -= np.min(distances)

        score = np.zeros((5, len(classifiers)))
        for i, (train, test) in enumerate(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(distances)):
            X_train = distances[train][:, train]
            X_test = distances[test][:, train]
            y_train = y[train]
            y_test = y[test]
            for j, (name, clf) in enumerate(classifiers.items()):
                if "SVM" in name:
                    X_train = K[train][:, train]
                    X_test = K[test][:, train]
                clf.fit(X_train, y_train)
                score[i, j] = clf.score(X_test, y_test)
        scores[epsilon] = np.mean(score, axis=0)
        print(f'epsilon={epsilon} done')
    print(time() - t0)
    scores.to_csv(f'../param_evaluation_{args.dataset}#{args.filter}.csv')
