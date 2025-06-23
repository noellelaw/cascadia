import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# Placeholder: need to replace with full FloodFeatureGraph
class FloodFeatureGraph:
    """
    Constructs a graph from coastal configuration and elevation metadata.
    Each node represents a spatial patch or pixel, and edges encode local connectivity.

    Inputs:
        depth_map (torch.Tensor): Tensor of shape (B, 1, H, W), representing flood depth.
        elevation_map (torch.Tensor): Tensor of shape (B, 1, H, W), representing elevation.
        landcover (torch.Tensor): Optional tensor of shape (B, H, W), integer land class labels.
        slr_scalar (torch.Tensor): Optional tensor of shape (B,), representing SLR scenario.

    Output:
        List[torch_geometric.data.Data]: Batch of graphs, one per input image.
    """

    def __init__(self, radius: float = 5.0):
        self.radius = radius

    def make_graph(self, depth_map, elevation_map, landcover=None, slr_scalar=None):
        B, _, H, W = depth_map.shape
        graphs = []

        for b in range(B):
            # Node positions (grid coordinates)
            x_grid = torch.arange(W)
            y_grid = torch.arange(H)
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
            pos = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()  # (N, 2)

            # Node features: concatenate elevation, depth, landcover, slr
            elevation = elevation_map[b].squeeze().reshape(-1, 1)  # (N, 1)
            depth = depth_map[b].squeeze().reshape(-1, 1)  # (N, 1)
            features = [elevation, depth]

            if landcover is not None:
                lc = F.one_hot(landcover[b].long(), num_classes=20).float().reshape(-1, 20)
                features.append(lc)

            if slr_scalar is not None:
                slr = slr_scalar[b].expand(H * W, 1)  # Broadcast to all nodes
                features.append(slr)

            x = torch.cat(features, dim=-1)  # Final node features

            # Edge indices via radius graph
            edge_index = radius_graph(pos, r=self.radius, loop=False)

            data = Data(x=x, edge_index=edge_index, pos=pos)
            graphs.append(data)

        return graphs

class BackboneEncoderGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph):
        # Simplified placeholder: assumes node features are concatenated with edge aggregates
        node_repr = torch.cat([graph.nodes, graph.edges], dim=-1)
        return self.encoder(node_repr)


class FloodGraphDesign(nn.Module):
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=1):
        super().__init__()
        self.graph_encoder = FloodBackboneEncoderGNN(node_dim, edge_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, depth, coastal_geom, slr_meta):
        # Build graph
        graph = FloodFeatureGraph(depth, coastal_geom, slr_meta)
        graph = graph.to(depth.device)

        # Encode graph
        h = self.graph_encoder(graph)

        # Decode to flood prediction (e.g., depth, risk)
        out = self.decoder(h)
        return out
