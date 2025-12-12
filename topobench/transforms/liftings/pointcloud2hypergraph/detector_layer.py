"""Mixture of Gaussians and Minimum Spanning Tree (MoGMST) Lifting."""

import torch
import torch_geometric

from topobench.transforms.liftings.pointcloud2hypergraph.base import (
    PointCloud2HypergraphLifting,
)


class DetectorLayerLifting(PointCloud2HypergraphLifting):
    r"""Lift a point cloud to a hypergraph.

    We create an hyperedge for each detector layer. The hyperedges are disconnected.

    Parameters
    ----------
    layers : tuple, optional
        The layer to consider when creating the hyperedges.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        layers=(0, 1, 2, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = layers

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
        num_detector_layers = len(self.layers)
        incidence = torch.zeros([num_nodes, num_detector_layers])
        nodes_to_include = torch.isin(data.x[:, 0], torch.tensor(self.layers))
        idx_nodes = torch.arange(num_nodes)[nodes_to_include]
        layers = data.x[nodes_to_include, 0].long()
        incidence[idx_nodes, layers] = 1
        incidence = incidence.clone().detach().to_sparse_coo()
        return {
            "incidence_hyperedges": incidence,
            "num_hyperedges": incidence.shape[1],
            "x_0": data.x,
        }
