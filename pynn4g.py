from model import NN4G
from validation import nested_cross_validation
from validation import grid_search_cv
from data import BenchmarkGraphDataset
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import json
import sys
import os

# Set up the CLI
parser = argparse.ArgumentParser(description="Python implementation of NN4G, "
                                 "a constructive neural network for graphs")
subparsers = parser.add_subparsers(title="Commands", metavar="<command>", dest="command")

# Train command
train_descr = "Train a single network"
train_cmd = subparsers.add_parser("train", help=train_descr,
                                  description=train_descr)
train_cmd.add_argument("ds_dir", type=str,
                       help="the dataset directory (check supported dataset format)")
train_cmd.add_argument("--params", type=str,
                       help="path of the hyperparameters file (omit to use defaults)")                   
train_cmd.add_argument("--plot", type=str, default=None, const="plots", action='store', nargs="?",
                       help="enable plotting and store plots in the specified directory "
                       "(default: /plots)")
train_cmd.add_argument("--early_stopping", default=True, action='store_true',
                       help="enable early stopping")
train_cmd.add_argument("--no_early_stopping", action='store_false', dest="early_stopping",
                       help="disable early stopping and stop training "
                       "at a prefixed num. of units (see --units)")
train_cmd.add_argument("--val_size", type=float, default=0.2,
                       help="determines how much of the data set is "
                       "used as a validation set for early stopping (default: 0.2) (range: [0, 1])")
train_cmd.add_argument("--units", type=int, default=10,
                       help="stop training when a number of hidden units is reached (default: 10)")
train_cmd.add_argument("--samples", type=int,
                       help="use only the specified number of samples from the shuffled data set")
train_cmd.add_argument("--deg_encoding", default=False, action='store_true',
                       help="enable degree encoding for each node")

# Model selection command
select_descr = "Do model selection on a given dataset (grid search cross validation)"
select_cmd = subparsers.add_parser("select", help=select_descr,
                                   description=select_descr)
select_cmd.add_argument("ds_dir", type=str,
                        help="the dataset directory (check supported dataset format)")
select_cmd.add_argument("params_path", type=str,
                        help="path of the hyperparameter grid file")
select_cmd.add_argument("folds", type=int,
                        help="number of folds in the cross validation")
select_cmd.add_argument("model_path", type=str,
                        help="path where the final model will be saved")
select_cmd.add_argument("--deg_encoding", default=False, action='store_true',
                        help="enable degree encoding for each node")

# Assessment command
assess_descr = "Estimate the performance of NN4G on a given dataset (nested cross validation)"
assess_cmd = subparsers.add_parser("assess", help=assess_descr,
                                   description=assess_descr)
assess_cmd.add_argument("ds_dir", type=str,
                        help="the dataset directory (check supported dataset format)")
assess_cmd.add_argument("params_path", type=str,
                        help="path of the hyperparameter grid file")
assess_cmd.add_argument("inner_folds", type=int,
                        help="number of folds for the inner cv")
assess_cmd.add_argument("outer_folds", type=int,
                        help="number of folds for the outer cv")
assess_cmd.add_argument("--deg_encoding", default=False, action='store_true',
                        help="enable degree encoding for each node")

# Predictor command
predictor_descr = "Predict new data with a previously trained model"
predictor_cmd = subparsers.add_parser("predict", help=predictor_descr,
                                      description=predictor_descr)
predictor_cmd.add_argument("ds_dir", type=str,
                           help="directory that contains the data to be predicted "
                           "(check supported dataset format)")
predictor_cmd.add_argument("model_path", type=str,
                           help="path of a trained model")
predictor_cmd.add_argument("--deg_encoding", default=False, action='store_true',
                           help="enable degree encoding for each node")

# If no arguments, print help message and exit
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

if args.command == "train":
    # Dataset files
    ds_dir = args.ds_dir
    A = os.path.join(ds_dir, "A.txt")
    graph_indicator = os.path.join(ds_dir, "graph_indicator.txt")
    graph_labels = os.path.join(ds_dir, "graph_labels.txt")
    node_labels = os.path.join(ds_dir, "node_labels.txt")
    if not os.path.exists(node_labels):
        node_labels = None

    print("Parsing the dataset...")

    # Create dataset from files
    dataset = BenchmarkGraphDataset()
    dataset.from_files(
        A_file=A,
        graph_indicator_file=graph_indicator,
        graph_labels_file=graph_labels,
        node_labels_file=node_labels,
        deg_encoding=args.deg_encoding
    )
    dataset.shuffle()

    print("Dataset parsed.")
    print("Dataset length: %d samples" % len(dataset))
    print("Number of attributes for each node: %d" % dataset.n_attributes)
    print("Number of samples used for training: %d" % args.samples)

    if args.samples is not None:
        X = np.array(dataset.inputs[0:args.samples])
        y = np.array(dataset.targets[0:args.samples])
    else:
        X = np.array(dataset.inputs)
        y = np.array(dataset.targets)

    if args.params is not None:
        # Load hyperparameters from JSON file
        with open(args.params) as f:
            params = json.load(f)

        net = NN4G(dataset.n_attributes, dataset.max_nodes,
                   verbose=2, plot_dir=args.plot, **params)
    else:
        net = NN4G(dataset.n_attributes, dataset.max_nodes, verbose=2, plot_dir=args.plot)

    if args.early_stopping:
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=args.val_size)
        net.fit(X_train, y_train, X_val, y_val)
    else:
        net.fit(X, y, n_units=args.units)

elif args.command == "assess":
    # Dataset files
    ds_dir = args.ds_dir
    A = os.path.join(ds_dir, "A.txt")
    graph_indicator = os.path.join(ds_dir, "graph_indicator.txt")
    graph_labels = os.path.join(ds_dir, "graph_labels.txt")
    node_labels = os.path.join(ds_dir, "node_labels.txt")
    if not os.path.exists(node_labels):
        node_labels = None

    print("Parsing the dataset...")

    # Create dataset from files
    dataset = BenchmarkGraphDataset()
    dataset.from_files(
        A_file=A,
        graph_indicator_file=graph_indicator,
        graph_labels_file=graph_labels,
        node_labels_file=node_labels,
        deg_encoding=args.deg_encoding
    )
    dataset.shuffle()

    print("Dataset parsed.")
    print("Dataset length: %d samples" % len(dataset))
    print("Number of attributes for each node: %d" % dataset.n_attributes)

    X = np.array(dataset.inputs)
    y = np.array(dataset.targets)

    # Create hyperparameters grid from JSON file
    with open(args.params_path) as f:
        param_grid = json.load(f)

    net = NN4G(dataset.n_attributes, dataset.max_nodes, verbose=2)
    nested_cross_validation(net, X, y, param_grid,
                            inner_splits=args.inner_folds,
                            outer_splits=args.outer_folds,
                            verbose=1)

elif args.command == "select":
    # Dataset files
    ds_dir = args.ds_dir
    A = os.path.join(ds_dir, "A.txt")
    graph_indicator = os.path.join(ds_dir, "graph_indicator.txt")
    graph_labels = os.path.join(ds_dir, "graph_labels.txt")
    node_labels = os.path.join(ds_dir, "node_labels.txt")
    if not os.path.exists(node_labels):
        node_labels = None

    print("Parsing the dataset...")

    # Create dataset from files
    dataset = BenchmarkGraphDataset()
    dataset.from_files(
        A_file=A,
        graph_indicator_file=graph_indicator,
        graph_labels_file=graph_labels,
        node_labels_file=node_labels,
        deg_encoding=args.deg_encoding
    )
    dataset.shuffle()

    print("Dataset parsed.")
    print("Dataset length: %d samples" % len(dataset))
    print("Number of attributes for each node: %d" % dataset.n_attributes)

    X = np.array(dataset.inputs)
    y = np.array(dataset.targets)

    # Create hyperparameters grid from JSON file
    with open(args.params_path) as f:
        param_grid = json.load(f)

    # Do grid search and save the final model on disk
    net = NN4G(dataset.n_attributes, dataset.max_nodes, verbose=2)
    selected_model = grid_search_cv(net, X, y, param_grid, verbose=1)
    selected_model.save(args.model_path)

    print("Model saved in %s" % args.model_path)

elif args.command == "predict":
    # Dataset files
    ds_dir = args.ds_dir
    A = os.path.join(ds_dir, "A.txt")
    graph_indicator = os.path.join(ds_dir, "graph_indicator.txt")
    node_labels = os.path.join(ds_dir, "node_labels.txt")
    if not os.path.exists(node_labels):
        node_labels = None

    print("Parsing the dataset...")

    # Create dataset from files
    dataset = BenchmarkGraphDataset()
    dataset.from_files(
        A_file=A,
        graph_indicator_file=graph_indicator,
        node_labels_file=node_labels,
        deg_encoding=args.deg_encoding
    )

    print("Dataset parsed.")
    print("Dataset length: %d samples" % len(dataset))
    print("Number of attributes for each node: %d" % dataset.n_attributes)

    X = np.array(dataset.inputs)

    # Load a trained model from disk
    net = NN4G.load(args.model_path)

    print("Model loaded.")

    # Predict data
    print(net.predict(X))
