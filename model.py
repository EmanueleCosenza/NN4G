import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt
import os


class Layer(nn.Module):
    """A layer in NN4G.

    This can represent an hidden layer as well as an ouput layer.
    The number of units in the layer is equal to the output dimension,
    `out_dim`. When creating an hidden layer, which is composed of a single unit,
    `out_dim` is set to 1. In an output layer, `out_dim` is set to
    the number of its outputs."""

    def __init__(self, in_dim, out_dim, f="sigmoid", weight_std=0.3):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        if f == "sigmoid":
            self.f = nn.Sigmoid()
        elif f == "tanh":
            self.f = nn.Tanh()
        elif f == "softmax":
            self.f = nn.Identity()
        self.init_weights(weight_std)

    def init_weights(self, weight_std):
        nn.init.normal_(self.linear.weight, std=weight_std)
        nn.init.normal_(self.linear.bias, std=weight_std)
        """ nn.init.uniform_(self.linear.weight, a=-0.7, b=0.7)
        nn.init.uniform_(self.linear.bias, a=-0.7, b=0.7) """

    def forward(self, input):
        return self.f(self.linear.forward(input))


class Candidate():
    """A trained candidate hidden unit in NN4G.

    This class contains the unit with its weights, the maximum
    correlation reached during the unit training,
    the state values x(v) calculated by the final unit for each vertex
    in the dataset and the averages X(g) for each graph in the dataset."""

    def __init__(self, unit, correlation, unit_state_values, unit_graphs_avgs):
        self.unit = unit
        self.correlation = correlation
        self.unit_state_values = unit_state_values
        self.unit_graphs_avgs = unit_graphs_avgs


class NN4G(BaseEstimator):
    """This class represents NN4G, a constructive neural network for graphs.

    Parameters:
    ----------
    n_candidates : int
        Number of candidate hidden units to be trained.
    neighbourhood : {"open", "closed"}
        Indicates whether to use a closed or open definition
        for the set of neighbours of a vertex
        (default: "open").
    parallel : boolean
        If true and `n_candidates` is greater than 1, candidate
        units are all trained in parallel. Otherwise, they
        are trained sequentially.
    hidden_lr : float
        Learning rate used in the training of each hidden unit.
    hidden_lambda : float
        Weight decay coefficient (L2 regularization) used
        in the training of each hidden unit.
    hidden_eps : float
        Threshold value for gradient ascent.
    max_hidden_epochs: int
        Number of maximum epochs for gradient ascent.
    output_lr : float
        Learning rate used in the training of the output layer.
    output_lambda : float
        Weight decay coefficient (L2 regularization) used
        in the training of the output layer.
    output_eps : float
        Threshold value for the gradient descent.
    max_output_epochs : int
        Number of maximum epochs of the gradient descent.
    patience : int
        Patience value used in the early stopping procedure.
    max_hidden_units : int
        Maximum number of hidden units allowed in NN4G.
    f_hidden : {"sigmoid", "tanh"}
        Activation function of the hidden units
        (default: "sigmoid").
    f_output : {"sigmoid", "softmax"}
        Activation function used in the output layer
        (default: "sigmoid").
    loss : {"cross_entropy", "mse"}
        Loss function used in the training of the output layer
        (default: "cross_entropy").
    verbose : int
        Verbosity of the training procedure. If set to 0,
        nothing will be printed.
    plot_dir : str
        Directory in which the training plots will be saved.

    """

    def __init__(self, n_attributes, max_nodes=None, n_candidates=1, neighbourhood="open",
                 wide=True, parallel=False, hidden_lr=0.1, hidden_lambda=0.01, hidden_eps=1e-3,
                 max_hidden_epochs=200, output_lr=0.1, output_lambda=0.01, output_eps=1e-3,
                 max_output_epochs=200, patience=6, max_hidden_units=35,
                 f_hidden="sigmoid", f_output="sigmoid", loss="cross_entropy",
                 verbose=0, plot_dir=None):
        self.hidden_units = []
        self.output_unit = None
        self.n_candidates = n_candidates
        self.neighbourhood = neighbourhood
        self.wide = wide
        self.parallel = parallel
        self.n_attributes = n_attributes
        self.max_nodes = max_nodes
        self.opt_tr_acc = None
        self.verbose = verbose
        self.plot_dir = plot_dir
        if self.plot_dir is not None:
            self.create_plot_dirs()
        self.f_hidden = f_hidden
        self.f_output = f_output
        self.loss = loss
        self.hidden_lr = hidden_lr
        self.output_lr = output_lr
        self.hidden_lambda = hidden_lambda
        self.output_lambda = output_lambda
        self.hidden_eps = hidden_eps
        self.output_eps = output_eps
        self.max_hidden_epochs = max_hidden_epochs
        self.max_output_epochs = max_output_epochs
        self.max_hidden_units = max_hidden_units
        self.patience = patience

    def create_plot_dirs(self):
        """Creates the directories in which the training plots will be saved."""
        if not os.path.exists(os.path.join(self.plot_dir, "correlation")):
            os.makedirs(os.path.join(self.plot_dir, "correlation"))
        if not os.path.exists(os.path.join(self.plot_dir, "losses")):
            os.makedirs(os.path.join(self.plot_dir, "losses"))
        if not os.path.exists(os.path.join(self.plot_dir, "accuracies")):
            os.makedirs(os.path.join(self.plot_dir, "accuracies"))

    def get_params(self, deep=True):
        """Returns the NN4G parameters in a dictionary."""
        params = {
            "n_attributes": self.n_attributes,
            "max_nodes": self.max_nodes,
            "n_candidates": self.n_candidates,
            "neighbourhood": self.neighbourhood,
            "wide": self.wide,
            "verbose": self.verbose,
            "plot_dir": self.plot_dir,
            "f_hidden": self.f_hidden,
            "f_output": self.f_output,
            "loss": self.loss,
            "hidden_lr": self.hidden_lr,
            "output_lr": self.output_lr,
            "hidden_lambda": self.hidden_lambda,
            "output_lambda": self.output_lambda,
            "hidden_eps": self.hidden_eps,
            "output_eps": self.output_eps,
            "max_hidden_epochs": self.max_hidden_epochs,
            "max_output_epochs": self.max_output_epochs,
            "max_hidden_units": self.max_hidden_units,
            "patience": self.patience
        }
        return params

    def set_params(self, **parameters):
        """Sets the NN4G parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def save(self, path):
        """Saves a trained model."""
        torch.save(self, path)

    @staticmethod
    def load(path):
        """Loads a trained model."""
        net = torch.load(path)
        return net

    def predict(self, graphs):
        """Predicts the output associated with every input graph."""
        n_hidden = len(self.hidden_units)
        if n_hidden == 0:
            return 0
        graphs_avgs = [{} for i in range(n_hidden)]
        state_values = [{} for i in range(n_hidden)]
        # Calculate X(i) for every hidden unit (gradient is not needed)
        with torch.no_grad():
            for i in range(n_hidden):
                unit = self.hidden_units[i]
                for graph in graphs:
                    graph_state_values = {}
                    for node in graph.nodes():
                        graph_state_values[node] = self.compute_state_value(node, graph, unit,
                                                                            i, state_values)
                    graphs_avgs[i][graph] = self.compute_graph_avg(graph, i,
                                                                   graph_state_values)
                    state_values[i].update(graph_state_values)
                graphs_avgs[i] = list(graphs_avgs[i].values())
            outputs = self.output_unit.forward(torch.t(torch.Tensor(graphs_avgs)))
            if self.n_outputs > 1:
                outputs = torch.nn.Softmax(dim=0)(outputs)
            predictions = self.predictions_from_outputs(outputs)
            return predictions

    def predictions_from_outputs(self, outputs):
        """Converts the outputs computed by the output layer
        to class predictions."""
        if outputs.shape[1] == 1:
            # Binary classification (threshold: 0.5)
            pred = torch.round(outputs)
        elif outputs.shape[1] > 1:
            # Multiclass
            # For each output vector, set to 1 its highest entry
            # and set the other entries to 0
            indices = torch.argmax(outputs, dim=1)
            pred = torch.zeros(outputs.shape)
            pred[torch.arange(outputs.shape[0]), indices] = 1
        return pred

    def score(self, graphs, targets):
        """Computes the accuracy of NN4G on a graph dataset."""
        predictions = self.predict(graphs)
        return accuracy_score(targets, predictions)

    def return_to_best_configuration(self, best_configuration):
        """Restores the best configuration reached in
        the training procedure."""
        n_hidden = best_configuration["n_hidden"]
        output_unit = best_configuration["output_unit"]

        self.output_unit = output_unit
        self.hidden_units = self.hidden_units[:n_hidden]

    def fit(self, X_train, y_train,
            X_val=None, y_val=None, n_units=None, target_tr_acc=None):
        """Trains NN4G on a training set."""
        self.early_stopping = (True if X_val is not None and y_val is not None
                               else False)
        self.n_outputs = len(y_train[0])
        self.n_samples = len(y_train)
        state_values = []
        graphs_avgs = []
        # Errors on initial minimal network (always 0 output):
        # errors[g] = (0 - target[g])
        errors = dict(zip(X_train, 0 - torch.FloatTensor(y_train)))
        no_improvement = 0
        best_configuration = None
        max_val_acc = 0
        tr_acc = 0
        val_acc = 0

        # Plot variables
        tr_curve_x = []
        tr_curve_y = []
        val_curve_x = []
        val_curve_y = []

        if self.verbose > 0:
            print("Started training.")

        # Main training loop
        while (len(self.hidden_units) < self.max_hidden_units):
            if self.early_stopping and no_improvement >= self.patience:
                break
            if self.verbose > 0:
                print("Num. of hidden units: %d" % (len(self.hidden_units) + 1))

            # Generate hidden and output units
            self.generate_hidden_unit(X_train, errors, state_values, graphs_avgs)
            best_outputs = self.generate_output_unit(X_train, y_train, graphs_avgs)
            errors = dict(zip(X_train, best_outputs - torch.FloatTensor(y_train)))
            predictions = self.predictions_from_outputs(best_outputs)
            tr_acc = 100 * accuracy_score(np.array(y_train), predictions)

            if self.early_stopping:
                # Check if accuracy on VL has improved
                val_acc = 100 * self.score(X_val, np.array(y_val))
                if (val_acc > max_val_acc):
                    self.opt_tr_acc = tr_acc
                    max_val_acc = val_acc
                    if no_improvement > 0:
                        no_improvement = 0
                    best_configuration = {
                        "n_hidden": len(self.hidden_units),
                        "output_unit": self.output_unit
                    }
                else:
                    no_improvement += 1

            if self.verbose > 1:
                print("Training accuracy: ", tr_acc)
                if self.early_stopping:
                    print("Validation accuracy: ", val_acc)
                    print("Best validation accuracy: ", max_val_acc)

            # Plot accuracies on training and validation set
            if self.plot_dir is not None:
                tr_curve_x.append(len(self.hidden_units))
                tr_curve_y.append(tr_acc)
                plt.plot(tr_curve_x, tr_curve_y, color="red", label="training set")
                if self.early_stopping:
                    val_curve_x.append(len(self.hidden_units))
                    val_curve_y.append(val_acc)
                    plt.plot(val_curve_x, val_curve_y,
                             color="green", label="validation set")
                plt.legend(loc="lower right")
                plt.xlabel("Unità nascoste")
                plt.ylabel("Accuracy")
                plt.ylim([0, 100])
                plt.xticks(tr_curve_x)
                plt.savefig(os.path.join(self.plot_dir, "accuracies",
                                         str(len(self.hidden_units)) + ".jpg"))
                plt.close()

            if n_units is not None:
                if len(self.hidden_units) >= n_units:
                    # Reached prefixed number of units
                    break
            if target_tr_acc is not None:
                if tr_acc >= target_tr_acc:
                    # Reached target training accuracy
                    break

        if self.early_stopping and best_configuration is not None:
            # Return to the configuration with the best validation accuracy
            self.return_to_best_configuration(best_configuration)

    def k(self):
        """Value used in the average operator"""
        return self.max_nodes

    def compute_state_value(self, node, graph, unit, layer, state_values):
        """Computes the state value x(v) associated to a vertex in a graph.
        The computation is done by the indicated hidden unit."""
        attributes = torch.Tensor(graph.node[node]["attributes"])
        prev_units_inputs = \
            self.prev_units_inputs(node, graph, layer, state_values)
        # Unit inputs: node attributes and inputs from previous units
        unit_inputs = torch.cat((attributes, prev_units_inputs))
        return unit.forward(unit_inputs)

    def prev_units_inputs(self, node, graph, layer, state_values):
        """Computes the inputs coming from the previous units
        for an hidden unit."""
        inputs = []
        if self.wide:
            # Consider all the previous units
            for i in range(layer):
                sum = 0
                for neighbour in graph[node]:
                    sum += state_values[i][neighbour]
                if self.neighbourhood == "closed":
                    sum += state_values[i][node]
                inputs.append(sum)
        elif layer > 0:
            # Consider only the previous unit
            sum = 0
            for neighbour in graph[node]:
                sum += state_values[layer-1][neighbour]
            if self.neighbourhood == "closed":
                sum += state_values[layer-1][node]
            inputs.append(sum)
        return torch.Tensor(inputs)

    def compute_graph_avg(self, graph, layer, graph_state_values):
        """Computes the average of all the state values of a graph.
        The computation is relative to an hidden layer of NN4G."""
        graph_avg = 0
        for node in graph.nodes():
            graph_avg += graph_state_values[node]
        graph_avg /= self.k()
        return graph_avg

    def compute_graph_state_values(self, graph, unit, layer, state_values):
        """Gets the state values computed by an hidden unit
        for each vertex in a graph."""
        graph_state_values = {}
        for node in graph.nodes():
            graph_state_values[node] = self.compute_state_value(node, graph, unit,
                                                                layer, state_values)
        return graph_state_values

    def correlation(self, graphs, errors, mean_errors, layer, unit_graphs_avgs):
        """Computes the correlation between the errors and the unit outputs
        over a set of graphs for an hidden layer."""
        avgs_sum = sum(unit_graphs_avgs.values())
        mean_graphs_avgs = avgs_sum / len(unit_graphs_avgs)
        correlation = 0
        for i in range(self.n_outputs):
            output_sum = 0
            for graph in graphs:
                output_sum += (errors[graph][i] - mean_errors[i]) * \
                    (unit_graphs_avgs[graph] - mean_graphs_avgs)
            correlation += torch.abs(output_sum)
        return correlation

    def generate_candidate_unit(self, graphs, errors, state_values):
        """Trains a candidate hidden unit with gradient ascent on correlation."""
        i = len(self.hidden_units)
        if self.wide:
            # Consider all the previous units
            in_dim = self.n_attributes + i
        else:
            # Consider only the previous unit
            if i > 0:
                in_dim = self.n_attributes + 1
            else:
                in_dim = self.n_attributes
        unit = Layer(in_dim, 1, self.f_hidden)
        last_best_unit = Layer(in_dim, 1, self.f_hidden)
        mean_errors = [0] * self.n_outputs
        for output in range(self.n_outputs):
            for e in errors.values():
                mean_errors[output] += e[output]
            mean_errors[output] = mean_errors[output] / len(errors)
        epochs = 0
        max_correlation = 0
        no_improvement = 0
        optimizer = torch.optim.Adam(
            params=unit.parameters(),
            lr=self.hidden_lr,
            weight_decay=self.hidden_lambda
        )

        epoch_x = []
        correlation_y = []

        while (epochs < self.max_hidden_epochs and no_improvement < 20):
            optimizer.zero_grad()
            unit_graphs_avgs = {}
            # Compute state values and averages for every graph
            for graph in graphs:
                graph_state_values = self.compute_graph_state_values(graph, unit, i,
                                                                     state_values)
                unit_graphs_avgs[graph] = self.compute_graph_avg(graph, i,
                                                                 graph_state_values)
            # Compute correlation
            neg_correlation = -self.correlation(graphs, errors, mean_errors,
                                                i, unit_graphs_avgs)
            epoch_x.append(epochs)
            correlation_y.append(-(neg_correlation.item()))
            neg_correlation.backward()
            optimizer.step()
            if -neg_correlation.item() > max_correlation + self.hidden_eps:
                if no_improvement > 0:
                    no_improvement = 0
            else:
                no_improvement += 1
            if -neg_correlation.item() > max_correlation:
                # A new maximum has been found, save the weights
                max_correlation = -neg_correlation.item()
                last_best_unit.load_state_dict(unit.state_dict())
            epochs += 1
        # Load the best weights
        if no_improvement > 0:
            unit.load_state_dict(last_best_unit.state_dict())

        unit_state_values = {}
        # Compute the state values with the best unit
        for graph in graphs:
            graph_state_values = self.compute_graph_state_values(graph, unit,
                                                                 i, state_values)
            unit_state_values.update(graph_state_values)
            unit_graphs_avgs[graph] = self.compute_graph_avg(graph, i,
                                                             graph_state_values)

        # Store floats instead of tensors to get rid of gradients
        for graph in graphs:
            for node in graph:
                unit_state_values[node] = unit_state_values[node].item()
            unit_graphs_avgs[graph] = unit_graphs_avgs[graph].item()

        if self.plot_dir is not None:
            plt.plot(epoch_x, correlation_y)
            plt.title("Unità nascoste: " + str(len(self.hidden_units)))
            plt.xlabel("Epoche")
            plt.ylabel("Correlazione")
            plt.savefig(os.path.join(self.plot_dir, "correlation",
                                     str(len(self.hidden_units)) + ".png"))
            plt.close()

        return Candidate(unit, max_correlation,
                         unit_state_values, unit_graphs_avgs)

    def generate_hidden_unit(self, graphs, errors, state_values, graphs_avgs):
        """Adds a new trained unit to NN4G."""
        winner = None
        if self.n_candidates == 1:
            # Train a single unit
            winner = self.generate_candidate_unit(graphs, errors, state_values)
        elif self.n_candidates > 1:
            # Train n_candidates units and choose the one with the best correlation
            if self.parallel:
                # A pool of units is trained sequentially
                max_correlation = 0
                for i in range(self.n_candidates):
                    candidate = self.generate_candidate_unit(graphs, errors, state_values)
                    if candidate.correlation > max_correlation:
                        max_correlation = candidate.correlation
                        winner = candidate
            else:
                # A pool of units is trained in parallel
                pool = ProcessPoolExecutor(self.n_candidates)
                futures = []
                for i in range(self.n_candidates):
                    futures.append(pool.submit(self.generate_candidate_unit, graphs, errors,
                                               state_values, graphs_avgs))
                max_correlation = 0
                for future in futures:
                    candidate = future.result()
                    if candidate.correlation > max_correlation:
                        max_correlation = candidate.correlation
                        winner = candidate

        # Caching of state values and averages
        state_values.append(winner.unit_state_values)
        graphs_avgs.append(winner.unit_graphs_avgs)

        # Add winner unit to the hidden units
        self.hidden_units.append(winner.unit)

        if self.verbose > 1:
            print("Max correlation: ", winner.correlation)

    def generate_output_unit(self, graphs, targets, graphs_avgs):
        """Trains the output layer of NN4G."""
        graphs = np.array(graphs)
        criterion = None
        if self.n_outputs == 1:
            if self.loss == "mse":
                criterion = torch.nn.MSELoss()  # TODO: Change
            elif self.loss == "cross_entropy":
                criterion = torch.nn.BCELoss()  # TODO: Change
        else:
            if self.f_output == "softmax":
                criterion = torch.nn.CrossEntropyLoss()
            elif self.f_output == "sigmoid":
                criterion = torch.nn.MSELoss()
        self.output_layer = Layer(len(self.hidden_units), self.n_outputs, self.f_output)
        optimizer = torch.optim.Adam(
            params=self.output_layer.parameters(),
            lr=self.output_lr,
            weight_decay=self.output_lambda
        )
        epoch_x = []
        ace_y = []
        epochs = 0
        no_improvement = 0
        min_loss = float("inf")
        best_outputs = []
        last_best_layer = Layer(len(self.hidden_units), self.n_outputs, self.f_output)

        while (epochs < self.max_output_epochs and no_improvement < 20):
            outputs = []
            loss = 0
            optimizer.zero_grad()
            # Do a forward pass to accumulate the gradients
            for g, graph in enumerate(graphs):
                inputs = []
                for i in range(len(self.hidden_units)):
                    inputs.append(graphs_avgs[i][graph])
                outputs.append(self.output_layer.forward(torch.Tensor(inputs)))
                target = torch.Tensor(targets[g])
                if self.n_outputs == 1:
                    loss += criterion(outputs[g], target)
                else:
                    if self.f_output == "softmax":
                        _, class_idx = target.max(dim=0)
                        loss += criterion(outputs[g].view(1, -1), class_idx.view(1))
                        outputs[g] = torch.nn.Softmax(dim=0)(outputs[g])
                    elif self.f_output == "sigmoid":
                        loss += criterion(outputs[g], target)
            loss /= len(graphs)
            # Plot
            epoch_x.append(epochs)
            ace_y.append(loss.item()/len(graphs))
            loss.backward()
            optimizer.step()
            # Stop condition
            if loss.item() < min_loss - self.output_eps:
                if no_improvement > 0:
                    no_improvement = 0
            else:
                no_improvement += 1
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_outputs = outputs
                last_best_layer.load_state_dict(self.output_layer.state_dict())
            epochs += 1

        if no_improvement > 0:
            self.output_layer.load_state_dict(last_best_layer.state_dict())

        if self.plot_dir is not None:
            plt.plot(epoch_x, ace_y, color="red")
            plt.title("Unità nascoste: " + str(len(self.hidden_units)))
            plt.xlabel("Epoche")
            plt.ylabel("Cross-entropy loss (media)")
            plt.savefig(os.path.join(self.plot_dir, "losses",
                        str(len(self.hidden_units)) + ".png"))
            plt.close()

        if self.verbose > 1:
            print("Epochs: ", epochs)
            print("Loss: ", min_loss/len(graphs))

        # Return column vector of best outputs (detached to get rid of the gradients)
        best_outputs = torch.stack(best_outputs).view(self.n_samples,
                                                      -1).detach()
        return best_outputs
