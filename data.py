from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer
import networkx
import pandas as pd
import random


class Graph(networkx.DiGraph):
    """A graph in a graph dataset."""

    def __init__(self, graph_id):
        super().__init__()
        self.graph_id = graph_id

    def __hash__(self):
        return self.graph_id

    def __eq__(self, other):
        return self.graph_id == other.graph_id


class BenchmarkGraphDataset(Dataset):
    """This class represents a graph dataset."""

    def __init__(self, n_attributes=None):
        self.inputs = []
        self.targets = []
        self.n_attributes = n_attributes
        self.max_nodes = 0

    def from_files(self, A_file, graph_indicator_file, graph_labels_file=None,
                   node_labels_file=None, neg_label=0, pos_label=1, deg_encoding=False):
        """Parses the indicated files to construct a graph dataset."""
        out_enc = LabelBinarizer()
        node_enc = LabelBinarizer(neg_label=neg_label, pos_label=pos_label)
        A = pd.read_csv(A_file, header=None).to_numpy()
        graph_indicator = pd.read_csv(graph_indicator_file, header=None).to_numpy()
        if graph_labels_file is not None:
            graph_labels = pd.read_csv(graph_labels_file, header=None).to_numpy()
            graph_labels = out_enc.fit_transform(graph_labels).tolist()
        if node_labels_file is not None:
            node_labels = pd.read_csv(node_labels_file, header=None).to_numpy()
            node_labels = node_enc.fit_transform(node_labels)
            self.n_attributes = len(node_enc.classes_)
        else:
            node_labels = [[1]] * len(graph_indicator)
            self.n_attributes = 1
        curr_graph_id = 1
        graph = Graph(curr_graph_id)
        for i, node_ids in enumerate(A):
            if (graph_indicator[node_ids[0]-1][0] != curr_graph_id):
                # Finished parsing the current graph
                n_nodes = len(graph)
                if n_nodes > self.max_nodes:
                    self.max_nodes = n_nodes
                self.inputs.append(graph)
                if graph_labels_file is not None:
                    target = graph_labels[curr_graph_id-1]
                    self.targets.append(target)
                curr_graph_id = graph_indicator[node_ids[0]-1][0].item()
                graph = Graph(curr_graph_id)
            # One-hot encoding for the attributes
            attr_0 = node_labels[node_ids[0]-1]
            attr_1 = node_labels[node_ids[1]-1]
            graph.add_node(node_ids[0], attributes=attr_0)
            graph.add_node(node_ids[1], attributes=attr_1)
            graph.add_edge(node_ids[0], node_ids[1])

        self.inputs.append(graph)
        if graph_labels_file is not None:
            target = graph_labels[curr_graph_id-1]
            self.targets.append(target)

        if deg_encoding:
            self.encode_degree()

    def encode_degree(self):
        """Uses one-hot degree encoding to set the attributes of each node
        in the dataset."""
        enc = LabelBinarizer()
        degrees = []
        for graph in self.inputs:
            for node in graph.nodes():
                degrees.append(graph.degree[node])
        node_labels = enc.fit_transform(degrees)
        self.n_attributes = len(enc.classes_)

        i = 0
        for graph in self.inputs:
            for node in graph:
                graph.node[node]["attributes"] = node_labels[i]
            i += 1

    def from_lists(self, inputs, targets, n_attributes):
        """Constructs a dataset from a list of graphs and a list of targets."""
        self.inputs = inputs
        self.targets = targets
        self.n_attributes = n_attributes
        self.max_nodes = 0
        for graph in inputs:
            if len(graph) > self.max_nodes:
                self.max_nodes = len(graph)

    def shuffle(self):
        """Randomly permutes the dataset samples."""
        patterns = list(zip(self.inputs, self.targets))
        random.shuffle(patterns)
        self.inputs, self.targets = zip(*patterns)
        self.inputs = list(self.inputs)
        self.targets = list(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
