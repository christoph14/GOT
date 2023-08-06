import functools
import multiprocessing
import warnings

import numpy as np
from sklearn import clone
from sklearn.exceptions import ConvergenceWarning

from utils import check_distance_matrix


def grid_search(distances, train_idx, y, estimator, grid, cv):
    result = {'mean_test_score': []}
    best_parameters = None
    for parameters in grid:
        # Save parameters in results dict and update estimator
        epsilon = parameters['epsilon']
        X = np.array(distances[epsilon])[train_idx][:, train_idx]
        X = check_distance_matrix(X)

        if 'gamma' in parameters.keys():
            gamma = parameters['gamma']
            X = np.exp(-gamma * X)
        clf_params = {k: v for k, v in parameters.items() if k not in ['epsilon', 'gamma']}
        clf = clone(estimator).set_params(**clf_params)
        for k, v in parameters.items():
            if f'param_{k}' not in result.keys():
                result[f'param_{k}'] = []
            result[f'param_{k}'].append(v)

        # Evaluate parameters
        kfold_score = []
        for train, test in cv.split(X, y):
            X_train = X[train][:, train]
            X_test = X[test][:, train]
            y_train = y[train]
            y_test = y[test]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                clf.fit(X_train, y_train)
            kfold_score.append(clf.score(X_test, y_test))
        mean_test_score = np.round(np.mean(kfold_score), 10)
        result['mean_test_score'].append(mean_test_score)
        if mean_test_score == np.max(result['mean_test_score']):
            best_parameters = parameters
    result['best_score'] = np.max(result['mean_test_score'])
    result['best_parameters'] = best_parameters
    best_clf_params = {k: v for k, v in best_parameters.items() if k not in ['epsilon', 'gamma']}
    result['best_clf'] = clone(estimator).set_params(**best_clf_params)
    return result


def _param_validation(split, distances, y, estimator, grid, inner_cv):
    train, test = split
    y_train = y[train]
    y_test = y[test]

    if grid is not None:
        # Find the best classifier parameters
        res = grid_search(distances, train, y_train, estimator, grid, inner_cv)
        clf = res['best_clf']
        epsilon = res['best_parameters']['epsilon']
        X = np.array(distances[epsilon])
        X = check_distance_matrix(X, log=False)
        X_train = X[train][:, train]
        X_test = X[test][:, train]
        # Use kernel matrix instead of distance matrix if estimator is SVM
        if 'gamma' in res['best_parameters'].keys():
            gamma = res['best_parameters']['gamma']
            X_train = np.exp(-gamma * X_train)
            X_test = np.exp(-gamma * X_test)
    else:
        clf = clone(estimator)
        X_train = np.ones((len(train), len(train)))
        X_test = np.ones((len(test), len(test)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def nested_cross_validation(distances, y, estimator, grid, inner_cv, outer_cv, n_jobs=None):
    y = np.array(y)

    splits = list(outer_cv.split(np.arange(len(y)), y))

    f = functools.partial(
        _param_validation, distances=distances, y=y, estimator=estimator, grid=grid, inner_cv=inner_cv
    )
    if n_jobs is None:
        kfold_score = list(map(f, splits))
    else:
        with multiprocessing.Pool(n_jobs) as pool:
            kfold_score = list(pool.imap(f, splits))
    return float(np.round(np.mean(kfold_score), 10))
