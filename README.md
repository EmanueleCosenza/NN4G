# NN4G (Neural Network For Graphs)
NN4G is a constructive neural network for graphs defined in [Micheli, Alessio. "Neural network for graphs: A contextual constructive approach." IEEE Transactions on Neural Networks 20.3 (2009): 498-511](https://ieeexplore.ieee.org/abstract/document/4773279).
This repository contains a Python implementation of NN4G with new architectural and training variants, a validation system for the network and `pynn4g`, a basic command line interface.
This project has been developed as part of my undergraduate thesis at the University of Pisa under the supervision of Professor Alessio Micheli.

## Project structure
- `model.py` contains everything related to the NN4G model.
- `data.py` contains the definition of a graph dataset.
- `validation.py` contains functions used for model selection and model assessment.
- `pynn4g.py` contains the implementation of a CLI which can be used to train a network, assess the model, select a model and predict new data with a trained network.
- The `data` directory contains MUTAG, NCI1 and IMDB-M, three graph datasets used for model assessment.
- The `tests` directory contains hyperparameter grids in JSON format.

## Datasets
The datasets MUTAG, NCI1 and IMDB-M were taken from http://graphkernels.cs.tu-dortmund.de. MUTAG and NCI1 are both chemical datasets for binary classification, while IMDB-M is a social network dataset for multiclassification.

