from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def grid_search_cv(net, X, y, param_grid, n_splits=3, verbose=0):
    """Selects a model on a dataset with cross validation and grid search."""
    cv = KFold(n_splits, shuffle=True)
    param_grid = ParameterGrid(param_grid)

    # Create a list with every possible training configuration
    configs = []
    for train_idx, val_idx in cv.split(X, y):
        for parameters in param_grid:
            configs.append((
                (train_idx, val_idx),
                parameters
            ))

    if verbose > 0:
        print("Training on %d folds for a total of %d models "
              "(%d folds * %d params configurations).\n"
              % (n_splits, len(configs), n_splits, len(param_grid)))

    # Training with early stopping for each configuration
    pool = ProcessPoolExecutor(len(configs))
    futures = []
    for (train_idx, val_idx), parameters in configs:
        futures.append(pool.submit(fit_and_score, clone(net), X, y,
                                   train_idx, val_idx,
                                   parameters,
                                   early_stopping=True))

    out = []
    for future in futures:
        out.append(future.result())
    pool.shutdown()

    # Format the results
    scores, _, opt_tr_accs, params = zip(*out)
    shape = (n_splits, len(param_grid))
    scores = np.array(scores).reshape(shape)
    opt_tr_accs = np.array(opt_tr_accs).reshape(shape)
    params = np.array(params).reshape(shape)

    # Calculate mean of validation scores for each parameter configuration
    # and find the best parameter index
    params_avgs = np.mean(scores, axis=0)
    best_param_index = np.argmax(params_avgs)

    # Find the avg training accuracy achieved
    # by the nets trained with the best params
    avg_tr_acc = np.mean(opt_tr_accs[:, best_param_index])
    avg_val_acc = np.mean(scores[:, best_param_index])
    best_params = params[0, best_param_index]
    if verbose > 0:
        print("\nGrid search results:")
        print("\tBest parameter configuration: %s" % best_params)
        print("\tAverage training accuracy: %.2f" % avg_tr_acc)
        print("\tAverage validation accuracy: %.2f" % (100*avg_val_acc))
        print("\n")

    if verbose > 0:
        print("Retraining on the entire dataset with the best parameters.\n")

    final_model = clone(net).set_params(**best_params)
    final_model.fit(X, y, target_tr_acc=avg_tr_acc)

    return final_model


def nested_cross_validation(net, X, y, param_grid, inner_splits=3, outer_splits=5, verbose=0):
    """Assesses a model on a dataset through nested cross validation."""
    outer_cv = KFold(outer_splits, shuffle=True)
    inner_cv = KFold(inner_splits)
    param_grid = ParameterGrid(param_grid)

    # Create a list with every possible inner training configuration
    configs = []
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        X_train_outer = X[outer_train_idx]
        y_train_outer = y[outer_train_idx]
        for inner_train_idx, val_idx in inner_cv.split(X_train_outer, y_train_outer):
            for parameters in param_grid:
                configs.append((
                    (outer_train_idx, outer_test_idx),
                    (inner_train_idx, val_idx),
                    parameters
                ))

    if verbose > 0:
        print("Fitting on inner folds for a total of %d models "
              "(%d outer folds * %d inner folds * %d params configurations).\n"
              % (len(configs), outer_splits, inner_splits, len(param_grid)))

    # Inner training with early stopping for each configuration
    pool = ProcessPoolExecutor(len(configs))
    futures = []
    for (outer_train_idx, _), (inner_train_idx, val_idx), parameters in configs:
        futures.append(pool.submit(fit_and_score, clone(net), X, y,
                                   inner_train_idx, val_idx,
                                   parameters,
                                   outer_train_idx=outer_train_idx,
                                   early_stopping=True))

    out = []
    for future in futures:
        out.append(future.result())
    pool.shutdown()

    # Format the results
    scores, n_units, opt_tr_accs, params = zip(*out)
    shape = (outer_splits, inner_splits, len(param_grid))
    scores = np.array(scores).reshape(shape)
    n_units = np.array(n_units).reshape(shape)
    opt_tr_accs = np.array(opt_tr_accs).reshape(shape)
    params = np.array(params).reshape(shape)

    # Calculate mean of validation scores for each parameter configuration
    # and find the best parameter indices for each outer split
    params_avgs = np.mean(scores, axis=1)
    best_param_indices = np.argmax(params_avgs, axis=1)

    # Find the number of avg units of the nets trained with the best params
    # and the best params for each outer split
    avg_n_units = []
    avg_tr_accs = []
    avg_val_accs = []
    best_params = []
    for i in range(outer_splits):
        avg_n_units.append(np.around(np.mean(n_units[i, :, best_param_indices[i]])))
        avg_tr_accs.append(np.mean(opt_tr_accs[i, :, best_param_indices[i]]))
        avg_val_accs.append(np.mean(scores[i, :, best_param_indices[i]]))
        best_params.append(params[i, 0, best_param_indices[i]])
        if verbose > 0:
            print("Outer fold: %d" % i)
            print("\tBest parameter configuration: %s" % best_params[i])
            print("\tAverage training accuracy: %.2f" % avg_tr_accs[i])
            print("\tAverage validation accuracy: %.2f" % (100*avg_val_accs[i]))
            print("\tAverage num. of units: %d" % avg_n_units[i])
            print("\n")

    if verbose > 0:
        print("Fitting %d models on the outer folds with the best parameters.\n"
              % outer_splits)

    # Retrain on entire outer training sets with the best params (in parallel)
    # and get scores on outer test sets.
    # No early stopping, prefixed target accuracy for each outer fold.
    pool = ProcessPoolExecutor(outer_splits)
    futures = []
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        futures.append(pool.submit(fit_and_score, clone(net), X, y,
                                   train_idx, test_idx,
                                   best_params[i],
                                   # fit_params={"n_units": avg_n_units[i]}))
                                   fit_params={"target_tr_acc": avg_tr_accs[i]}))

    out = []
    for future in futures:
        out.append(future.result())
    pool.shutdown()

    test_scores, _, _, _ = zip(*out)
    avg_test_score, std_test_score = np.mean(test_scores), np.std(scores)

    if verbose > 0:
        print("Accuracies on outer folds: ", test_scores)
        print("Average accuracy: %.2f +- %.2f" % (avg_test_score*100, std_test_score*100))
        print("\n")
    return test_scores


def fit_and_score(net, X, y, train_idx, test_idx, parameters,
                  outer_train_idx=None, fit_params=None, early_stopping=False):
    """Trains a single network on a training set."""
    if outer_train_idx is not None:
        X, y = X[outer_train_idx], y[outer_train_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    net.set_params(**parameters)
    fit_params = fit_params if fit_params is not None else {}

    # Train the net
    if early_stopping:
        net.fit(X_train, y_train, X_test, y_test, **fit_params)
    else:
        net.fit(X_train, y_train, **fit_params)

    # Calculate accuracy on test set
    test_score = net.score(X_test, y_test)

    return (test_score, len(net.hidden_units), net.opt_tr_acc, parameters)
