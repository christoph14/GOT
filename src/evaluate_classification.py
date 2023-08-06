import argparse
from time import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from utils.validation import nested_cross_validation

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates fGOT parameters on given data set.')
    parser.add_argument('dataset', type=str, help='the benchmark data set')
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None, help='the used random seed')
    parser.add_argument('--path', type=str, default='../distances', help='the path to store the output files')
    args = parser.parse_args()

    print(f"Evaluate parameters on data set {args.dataset} with filter {args.filter}.")

    distance_measures = [
        'got',
        # 'got-label',
        'approx',
        # 'approx-label',
        'fro',
        # 'fro-label',
    ]

    y = np.loadtxt(f'{args.path}/{args.dataset}/labels.csv')
    _, labels = np.unique(y.flatten(), return_counts=True)
    print('Class occurrences:', labels / np.sum(labels) * 100)

    t0 = time()
    scores = {}
    for measure in distance_measures:
        print('#################################################')
        print(f"Comparison for distance measure {measure}.")
        # Load distance matrices
        epsilon_range = np.array([0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1])
        # epsilon_range = np.array([0.002, 0.004, 0.006, 0.008, 0.01, 0.06, 0.08, 0.1])
        # epsilon_range = np.array([0.008, 0.02])
        # epsilon_range = np.array([None])
        distances = {
            epsilon: np.loadtxt(f"{args.path}/{args.dataset}/fGOT-{args.filter}-{epsilon}-{measure}.csv")
            for epsilon in epsilon_range
        }

        svm_scores = []
        knn_scores = []
        baseline_scores = []
        for i in tqdm(range(5)):
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)

            # SVM classification
            clf = SVC(
                kernel='precomputed',
                max_iter=100000,
                class_weight='balanced',
                probability=True,
            )
            grid = ParameterGrid({
                'C': np.logspace(-3, 3, 7),
                # 'gamma': np.logspace(-10, 10, 21),
                'gamma': np.logspace(-3, 3, 7),
                'epsilon': epsilon_range,
            })
            svm_score = nested_cross_validation(distances, y, clf, grid, inner_cv, outer_cv, n_jobs=10)
            svm_scores.append(svm_score)

            # k-NN classification
            clf = KNeighborsClassifier(metric='precomputed')
            grid = ParameterGrid({
                'n_neighbors': np.arange(1, 11),
                'epsilon': epsilon_range,
            })
            knn_score = nested_cross_validation(distances, y, clf, grid, inner_cv, outer_cv, n_jobs=10)
            knn_scores.append(knn_score)

            # Baseline classifier
            clf = DummyClassifier(strategy='most_frequent')
            grid = None
            baseline_score = nested_cross_validation(distances, y, clf, grid, inner_cv, outer_cv)
            baseline_scores.append(baseline_score)

        print(f'kNN score: {np.mean(knn_scores)*100:.2f}')
        print(f'SVM score: {np.mean(svm_scores)*100:.2f}')
        print(f'BASELINE score: {np.mean(baseline_scores)*100:.2f}')
    print(f"Finished in {time() - t0:.0f}s")
