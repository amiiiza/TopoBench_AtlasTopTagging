"""MaxDifference feature lifting class."""

import torch
import torch_geometric


class MaxDifference(torch_geometric.transforms.BaseTransform):
    r"""Lift r-cell features to r+1-cells by projection.

    Parameters
    ----------
    keep_original : bool, optional
        Whether to keep the original features or not. (default: :obj:`True`).
    detector_idx : int, optional
        The index, in the data features, that indicates the detector. (default: :obj:`0`).
    aggr : str, optional
        The aggregation method to use. (default: :obj:`"mean"`).
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        keep_original=True,
        detector_idx=0,
        aggr="mean",
        **kwargs,
    ):
        super().__init__()
        self.keep_original = keep_original
        self.detector_idx = detector_idx
        if aggr == "mean":
            self.aggr = torch.mean
        elif aggr == "sum":
            self.aggr = torch.sum
        else:
            raise ValueError(f"Invalid aggregation method: {aggr}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keep_original={self.keep_original}, detector_idx={self.detector_idx}, aggr={self.aggr})"

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Project r-cell features of a graph to r+1-cell structures.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The data with the lifted features.
        """
        keys = sorted(
            [
                key.split("_")[1]
                for key in data
                if ("incidence" in key and "-" not in key)
            ]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                incidence = data[f"incidence_{elem}"]
                hyperedge_indices = torch.unique(incidence.indices()[1])
                all_features = []
                for h_i in hyperedge_indices:
                    nodes_indices = incidence.indices()[
                        0, torch.where(incidence.indices()[1, :] == h_i)[0]
                    ]
                    nodes_features = data[f"x_{idx_to_project}"][
                        nodes_indices, :
                    ]
                    max_diffs = (
                        torch.max(nodes_features, dim=0).values
                        - torch.min(nodes_features, dim=0).values
                    )
                    max_energy = torch.max(nodes_features[:, 3], dim=0).values
                    max_energy = max_energy.unsqueeze(0)
                    if self.keep_original:
                        aggregated_features = self.aggr(nodes_features, dim=0)
                        all_features.append(
                            torch.cat(
                                [aggregated_features, max_diffs, max_energy]
                            )
                        )
                    else:
                        all_features.append(torch.cat([max_diffs, max_energy]))
                if len(all_features) == 0:
                    data["x_" + elem] = torch.empty(
                        0,
                        data[f"x_{idx_to_project}"].shape[1]
                        * (1 + self.keep_original)
                        + 1,
                    )
                else:
                    data["x_" + elem] = torch.stack(all_features)
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Apply the lifting to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        data = self.lift_features(data)
        return data
