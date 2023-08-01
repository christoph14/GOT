import numpy as np
from sklearn import clone


def grid_search(X, y, estimator, grid, cv):
    result = {'mean_test_score': []}
    distances = np.array(X)
    best_parameters = None
    for parameters in grid:
        # Save parameters in results dict and update estimator
        if 'gamma' in parameters.keys():
            gamma = parameters['gamma']
            X = np.exp(-gamma * distances)
        clf = clone(estimator).set_params(**parameters)
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
            clf.fit(X_train, y_train)
            kfold_score.append(clf.score(X_test, y_test))
        mean_test_score = np.round(np.mean(kfold_score), 10)
        result['mean_test_score'].append(mean_test_score)
        if mean_test_score == np.max(result['mean_test_score']):
            best_parameters = parameters
    result['best_score'] = np.max(result['mean_test_score'])
    result['best_parameters'] = best_parameters
    result['best_clf'] = clone(estimator).set_params(**best_parameters)
    return result


def nested_cross_validation(X, y, estimator, grid, inner_cv, outer_cv):
    X = np.array(X)
    y = np.array(y)
    kfold_score = []
    for train, test in outer_cv.split(X, y):
        X_train = X[train][:, train]
        X_test = X[test][:, train]
        y_train = y[train]
        y_test = y[test]

        # Find the best classifier parameters
        if grid is not None:
            res = grid_search(X_train, y_train, estimator, grid, inner_cv)
            clf = res['best_clf']
            # Use kernel matrix instead of distance matrix if estimator is SVM
            if 'gamma' in res['best_parameters'].keys():
                gamma = res['best_parameters']['gamma']
                X_train = np.exp(-gamma * X_train)
                X_test = np.exp(-gamma * X_test)
        else:
            clf = clone(estimator)
        clf.fit(X_train, y_train)
        kfold_score.append(clf.score(X_test, y_test))
    return float(np.round(np.mean(kfold_score), 10))
