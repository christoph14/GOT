"""Create the predictions of a baseline classifier."""

import argparse

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils.dataset import tud_to_networkx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name',
        type=str,
        help='Data set name',
        required=True
    )
    args = parser.parse_args()

    graphs = tud_to_networkx(args.name)
    X = np.empty(len(graphs), dtype=object)
    X[:] = graphs
    y = np.array([G.graph['classes'] for G in X])

    clf = DummyClassifier(strategy='most_frequent')

    n_iterations = 10
    n_folds = 10

    accuracies = []
    for iteration in range(n_iterations):
        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42 + iteration
        )

        fold_accuracies = []
        for fold_index, (train_index, test_index) in enumerate(cv.split(X, y)):
            clf.fit(X[train_index], y[train_index])
            y_test = y[test_index]
            y_pred = clf.predict(X[test_index])
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
        accuracies.append(np.mean(fold_accuracies ))
    accuracies = np.array(accuracies)
    print(f'{accuracies.mean()*100:2.2f} \\pm {accuracies.std():2.2f}')
