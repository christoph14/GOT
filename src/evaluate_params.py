import argparse
import functools
import itertools
import json
import multiprocessing
import os
from time import time

import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.svm import SVC

from utils.dataset import tud_to_networkx
from utils.distances import compute_distance

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
    print(f"Evaluate parameters on data set {args.dataset} with filter {args.filter}.")
    print(f'Compute distance matrix for {n_graphs} graphs')

    # Determine number of cores
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()
    print(f'Use {number_of_cores} cores')

    distance_measures = ['got', 'got-approx', 'fro', 'got-label']
    strategy_args = {
        'filter_name': args.filter,
        'scale': True,
    }

    # Define possible values for fGOT regularization parameter
    epsilon_range = np.append(
        10**(-3) * np.arange(2, 12, 2),
        10**(-2) * np.arange(2, 12, 2)
    )

    scores = dict()
    for epsilon in epsilon_range:
        # Compute distance matrix
        strategy_args['epsilon'] = epsilon
        f = functools.partial(compute_distance, strategy='fGOT', strategy_args=strategy_args)
        with multiprocessing.Pool(number_of_cores) as pool:
            result = pool.starmap(f, itertools.product(X, X))
        # Convert list of dicts to dict of lists, and reshape distance matrices
        all_distances = {k: np.reshape([dic[k] for dic in result], (len(X), len(X))) for k in distance_measures}

        scores[epsilon] = {}
        for measure, distances in all_distances.items():
            # Check computed distance matrix
            n_errors = np.count_nonzero(np.isnan(distances) | np.isinf(distances))
            if n_errors > 0:
                print(f'Warning: {n_errors} NaNs/infs in distance matrix.')
                if (np.isnan(distances) | np.isinf(distances)).all():
                    scores[epsilon][measure] = 0, np.inf
                    continue
                else:
                    distances = np.nan_to_num(distances, nan=np.nanmax(distances))

            # Ensure that the distance matrix is non-negative
            if np.min(distances) < 0:
                print(f'Warning: negative values in {measure} distance matrix.')
                distances -= np.min(distances)

            C_range = np.logspace(-3, 3, 7)
            gamma_range = np.logspace(-9, 3, 13)
            gamma_range = np.concatenate((gamma_range, [0.2]))
            grid = ParameterGrid({'C': C_range, 'gamma': gamma_range})
            result = dict()
            result['param_C'] = []
            result['param_gamma'] = []
            result['mean_test_score'] = []
            for params in grid:
                C = params['C']
                gamma = params['gamma']
                result['param_C'].append(C)
                result['param_gamma'].append(gamma)
                clf = SVC(kernel='precomputed', C=C, max_iter=100000)
                K = np.exp(-gamma * distances)

                kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
                kfold_score = []
                for i, (train, test) in enumerate(kfold.split(distances)):
                    X_train = K[train][:, train]
                    X_test = K[test][:, train]
                    y_train = y[train]
                    y_test = y[test]
                    clf.fit(X_train, y_train)
                    kfold_score.append(clf.score(X_test, y_test))
                result['mean_test_score'].append(np.mean(kfold_score))
            scores[epsilon][measure] = np.round(np.max(result['mean_test_score']), 5), np.sum(distances)
        print(f'epsilon={epsilon} done')
    print(time() - t0)
    with open(f'../svm_param_evaluation_{args.dataset}-{args.filter}.json', 'w') as f:
        json.dump(scores, f, indent=4)
