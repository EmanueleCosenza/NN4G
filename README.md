# NN4G (Neural Network For Graphs)
NN4G is a constructive neural network for graphs defined in [Micheli, Alessio. "Neural network for graphs: A contextual constructive approach." IEEE Transactions on Neural Networks 20.3 (2009): 498-511](https://ieeexplore.ieee.org/abstract/document/4773279).
This repository contains a Python implementation of NN4G with new architectural and training variants, a validation system for the network and `pynn4g`, a basic command line interface.
This project has been developed as part of my undergraduatethesis at the University of Pisa under the supervision of professor Alessio Micheli.

## CLI usage
`pynn4g` is a basic CLI that can be used to experiment with the model. It offers 4 main functionalities through 4 commands.

`train` trains a single network on a dataset (early stopping is optional). Learning curves can be plotted with the `--plot` option. In the following example, a network is trained on the NCI1 dataset using early stopping. `tests/NCI1.json` is a JSON file which contains the network hyperparameters.
```
python pynn4g.py train data/NCI1 --early stopping --params=tests/NCI1.json --plot
```


`assess` estimates the performance of NN4G on a dataset through nested cross validation. In the example below, NN4G is assessed on the MUTAG dataset using a 3-fold inner CV and a 5-fold outer CV. `tests/MUTAG.json` is a JSON file which contains an hyperparameter grid.
```
python pynn4g.py assess data/MUTAG tests/MUTAG.json 3 5
```

`select` selects a model on a dataset, using k-fold cross validation and grid search for hyperparameter optimization. The selected model is then trained on the entire dataset and a final network is returned. The trained model can be saved on file for future predictions. In the example, `pynn4g` selects a model on the dataset MUTAG using a 3-fold CV. The model is saved in `trained/mutag.model`. `tests/MUTAG.json` is a JSON file which contains an hyperparameter grid.
```
python pynn4g.py select data/MUTAG tests/mutag.json 3 trained/mutag.model
```

The `predict` command represents a predictor, which predicts outputs for a set of unlabeled graphs. In the example, `pynn4g` predicts  the outputs for each graph in the dataset contained in `DS_DIR`. The file `trained/mutag.model` contains a trained model (saved previously with the `select` command). The trained model is used to compute the predictions.
```
python pynn4g.py predict DS_DIR trained/mutag.model
```


## Project structure
- `model.py` contains everything related to the NN4G model.
- `data.py` contains the definition of a graph dataset.
- `validation.py` contains functions used for model selection and model assessment.
- `pynn4g.py` contains the implementation of a CLI which can be used to train a network, assess the model, select a model and predict new data with a trained network. It depends on all the previous modules.
- The `data` directory contains MUTAG, NCI1 and IMDB-M, three graph datasets used for model assessment.
- The `tests` directory contains examples of hyperparameter grids in JSON format.

## CLI implementation
`pynn4g` has been implemented using the standard Python module for argument parsing, argparse. In `pynn4g.py`, different actions are taken for each `pynn4g` command using the interfaces exposed by the other modules.

## Model implementation
The `Layer` class represents a layer of the neural network. Objects of this type are used inside the `NN4G` class to represent hidden layers and output layers. `Layer` inherits from PyTorch's `Module` and therefore implements a `forward` method, which computes the layer outputs given its inputs.\
The activation function of the units in a layer can be chosen between `PyTorch`'s `Tanh`, `Sigmoid` and `Softmax` (for the output layer).

The `NN4G` class represents a neural network. In this particular implementation, NN4G can be either a binary classifier or a multiclassifier.\
The class inherits from scikit-learn's `BaseEstimator` implementing its base methods `get_params` and `set_params`, which respectively get and set the network parameters. In order to be a scikit-learn classifier, `NN4G` implements `fit` and `predict`. The `fit` method is used to train the network on a training set, optionally using a validation set for the early stopping procedure, while the `predict` method is used to calculate predictions associated to graphs contained in a list. The class also implements `score`, which computes the network accuracy on a dataset. Each network hyperparameter can be set using `NN4G`'s constructor.\
The class interface is used by the validation module to do model selection and to assess the model on a dataset.

###### Training procedure
In each iteration of the training loop in the `fit` method:
1. A new hidden unit is trained and added to the network, maximizing the correlation between the unit output and the residual errors in the output layer for the previous iteration.\
Multiple units can be trained in parallel to find the best correlation. The training of a single hidden unit happens inside `generate_candidate_unit`. Then, in the `generate_hidden_unit`, results from multiple trainings are gathered and, finally, the best unit is added to the network.
2. The output layer is trained by minimizing a loss function (MSE or cross entropy loss). The training is implemented inside `generate_output_unit`.
3. New predictions and residual errors are computed for each training set sample.

Minimization and maximization problems are solved by means of the algorithm of gradient ascent/descent.\
In both cases, the algorithms are implemented using PyTorch's `Optimizer`s with weight decay (L2 loss). In this way, gradients are automatically computed by PyTorch's through its computational graph and weights are updated calling the `Tensor`'s `backward` method.

## Model selection and model assessment implementation
The model selection and model assessment procedures are defined inside the `validation.py` module.\
Model selection is implemented inside the `grid_search_cv` function, using k-fold cross validation and grid search for hyperparameter optimization. Model assessment is implemented inside the `nested_cross_validation` function, using a nested cross validation procedure.\
While `grid_search_cv` returns the final network trained with the best hyperparameters on the entire input dataset, `nested_cross_validation` returns an estimate of the model performance on a dataset.\
Python's `ProcessPoolExecutor` class is used in both functions to parallelize trainings on multiple CPUs. Both functions accept an hyperparameter grid as a Python dictionary.

## Dataset creation
Inside the `data.py` module, the `BenchmarkGraphDataset` class represents a graph dataset. Its `from_files` method constructs a dataset parsed from files in the format specified in http://graphkernels.cs.tu-dortmund.de. A dataset can also be created from an already existing list of graphs and targets.

The `Graph` class represents a graph in a graph dataset. It extends NetworkX's `DiGraph`, a directed graph. Undirected graphs can be created adding for each edge u->v of a graph the edge v->u.

## Datasets
The MUTAG, NCI1 and IMDB-M datasets are all taken from http://graphkernels.cs.tu-dortmund.de. MUTAG and NCI1 are both chemical datasets for binary classification, while IMDB-M is a social network dataset for multiclassification.

