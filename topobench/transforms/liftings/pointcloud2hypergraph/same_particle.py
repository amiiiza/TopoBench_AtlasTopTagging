"""Mixture of Gaussians and Minimum Spanning Tree (MoGMST) Lifting."""

import numpy as np
import torch
import torch_geometric

from topobench.transforms.liftings.pointcloud2hypergraph.base import (
    PointCloud2HypergraphLifting,
)


def compute_distance(p1, p2):
    """Compute Euclidean distance between two points.

    Parameters
    ----------
    p1 : np.ndarray
        The first point.
    p2 : np.ndarray
        The second point.

    Returns
    -------
    float
        The Euclidean distance between the two points.
    """
    return np.linalg.norm(p1 - p2)


class SameParticleLifting(PointCloud2HypergraphLifting):
    r"""Lift a point cloud to a hypergraph.

    We create an hyperedge for each detector layer. Then we also connect with hyperedges nodes that have close enough angles and are in neighboring layers.

    Parameters
    ----------
    threshold : float, optional
        The threshold for the distance between nodes to be connected. Default is 0.4.
    max_neighbors : int, optional
        The maximum number of neighbors to connect. If -1, connect all neighbors within the threshold. Default is -1.
    n_detector_layers : int, optional
        The number of detector layers. Default is 4.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        threshold=0.4,
        max_neighbors=-1,
        n_detector_layers=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_neighbors = max_neighbors
        self.n_detector_layers = n_detector_layers

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        """Lift the topology of a pointcloud to a hypergraph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        num_nodes = data.x.shape[0]
        nodes_layer = data.x[:, 0].long()
        nodes_coords = data.x[:, 1:3]

        incidence_same_layer = torch.zeros([num_nodes, self.n_detector_layers])
        incidence_same_layer[torch.arange(num_nodes), nodes_layer.long()] = 1

        lists = find_lists(
            nodes_layer, nodes_coords, self.threshold, self.max_neighbors
        )
        hyperedges = [torch.tensor(ll) for ll in lists if len(ll) > 1]
        incidence_neighboring_layers = torch.zeros(
            [num_nodes, len(hyperedges)]
        )
        for i, hyperedge in enumerate(hyperedges):
            incidence_neighboring_layers[hyperedge, i] = 1

        incidence = torch.cat(
            [incidence_same_layer, incidence_neighboring_layers], dim=1
        )

        incidence = incidence[
            :, incidence.sum(dim=0) > 1
        ]  # Remove hyperedges with zero or one node
        incidence = incidence.clone().detach().to_sparse_coo()
        return {
            "incidence_hyperedges": incidence,
            "num_hyperedges": incidence.shape[1],
            "x_0": data.x,
        }


def find_lists(nodes_layer, nodes_coords, threshold, max_neighbors):
    """Find all possible lists of connected nodes across layers.

    Parameters
    ----------
    nodes_layer : torch.Tensor
        The layer of each node.
    nodes_coords : torch.Tensor
        The coordinates of each node.
    threshold : float
        The threshold for the distance between nodes to be connected.
    max_neighbors : int
        The maximum number of neighbors to connect.

    Returns
    -------
    list
        The list of connected nodes.
    """
    unique_layers = torch.unique(nodes_layer).tolist()
    layer_dict = {layer: [] for layer in unique_layers}

    # Organize nodes by layer
    for i, layer in enumerate(nodes_layer):
        layer_dict[layer.item()].append(
            (i, nodes_coords[i].numpy())
        )  # Store index and coordinates

    used_nodes = set()  # Track nodes already assigned to lists
    all_lists = []  # Store all node groups

    def build_lists(start_node, start_layer):
        """Build multiple lists starting from a node, allowing branching.

        Parameters
        ----------
        start_node : tuple
            The node to start from.
        start_layer : int
            The layer to start from.

        Returns
        -------
        list
            The list of connected nodes.
        """
        if start_layer not in layer_dict:
            return []

        current_node_index = start_node[0]
        current_node_coords = start_node[1]

        # Mark node as used in any list it appears
        used_nodes.add(current_node_index)

        # If it's the last layer, return a single-element list
        next_layer = start_layer + 1
        if next_layer not in layer_dict:
            return [[current_node_index]]

        candidates = layer_dict[next_layer]

        # Compute distances for all candidates
        candidate_distances = [
            (candidate, compute_distance(current_node_coords, candidate[1]))
            for candidate in candidates
        ]

        # Filter and sort by distance
        neighbors = [
            candidate
            for candidate, dist in sorted(
                candidate_distances, key=lambda x: x[1]
            )
            if dist < threshold
        ]

        if max_neighbors != -1:
            neighbors = neighbors[:max_neighbors]

        # If no valid neighbors, return a list with just the current node
        if not neighbors:
            return [[current_node_index]]

        # Generate separate lists for each neighbor
        lists = []
        for neighbor in neighbors:
            sub_lists = build_lists(
                neighbor, next_layer
            )  # Get paths from the neighbor
            lists.extend(
                [[current_node_index, *sub_list] for sub_list in sub_lists]
            )

        return lists

    # Iterate through each layer and find lists
    for layer in unique_layers:
        for node in layer_dict[layer]:
            if node[0] not in used_nodes:
                new_lists = build_lists(node, layer)
                all_lists.extend(new_lists)  # Store all independent paths

    return all_lists
