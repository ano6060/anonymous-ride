from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def build_station_adjacency_from_node_links(node_links: pd.DataFrame) -> dict[int, set[int]]:
    """Build undirected station adjacency from silver node_links."""
    adjacency: dict[int, set[int]] = {}
    u_nodes = pd.to_numeric(node_links["u_node_id"]).astype(np.int64).to_numpy()
    v_nodes = pd.to_numeric(node_links["v_node_id"]).astype(np.int64).to_numpy()
    for u, v in zip(u_nodes, v_nodes):
        if u not in adjacency:
            adjacency[u] = set()
        if v not in adjacency:
            adjacency[v] = set()
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def compute_laplacian_embeddings(
    adjacency: dict[int, set[int]],
    embedding_dim: int,
) -> tuple[np.ndarray, list[int], nx.Graph]:
    """Compute normalized Laplacian eigenmap embeddings."""
    nodes = list(adjacency.keys())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_matrix = np.zeros((n, n), dtype=np.float64)
    for u in nodes:
        i = node_to_idx[u]
        for v in adjacency[u]:
            j = node_to_idx[v]
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0
    np.fill_diagonal(adj_matrix, 0.0)
    graph = nx.from_numpy_array(adj_matrix)
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    nonzero_mask = np.abs(eigenvalues) >= 1e-12
    nonzero_idx = np.where(nonzero_mask)[0]
    embeddings = eigenvectors[:, nonzero_idx[:embedding_dim]]
    embeddings = embeddings / np.maximum(1e-8, np.linalg.norm(embeddings, axis=1, keepdims=True))
    return embeddings, nodes, graph


def build_positions_from_op_nodes(
    op_nodes: pd.DataFrame,
    node_order: list[int],
) -> dict[int, tuple[float, float]]:
    """Return graph positions by node index as (lon, lat) for plotting."""
    op_nodes = op_nodes.set_index("op_id")
    positions: dict[int, tuple[float, float]] = {}
    for idx, op_id in enumerate(node_order):
        row = op_nodes.loc[int(op_id)]
        positions[idx] = (float(row["lon"]), float(row["lat"]))
    return positions


def create_station_embeddings_from_silver(
    *,
    node_links_path: Path,
    op_nodes_path: Path,
    embedding_dim: int,
) -> tuple[np.ndarray, list[int], nx.Graph, dict[int, tuple[float, float]], dict[int, set[int]]]:
    """Create station embeddings from silver node_links/op_nodes."""
    node_links = pd.read_parquet(node_links_path, columns=["u_node_id", "v_node_id"])
    adjacency = build_station_adjacency_from_node_links(node_links)
    embeddings, node_order, graph = compute_laplacian_embeddings(adjacency, embedding_dim)
    op_nodes = pd.read_parquet(op_nodes_path, columns=["op_id", "lat", "lon"])
    positions = build_positions_from_op_nodes(op_nodes, node_order)
    return embeddings, node_order, graph, positions, adjacency


def plot_embedding_components_on_graph(
    *,
    graph: nx.Graph,
    embeddings: np.ndarray,
    positions: dict[int, tuple[float, float]],
    nb_components: int = 16,
    figsize: int = 16,
) -> None:
    """Plot embedding values for components on the station graph."""
    ncols = 4
    nrows = int(np.ceil(nb_components / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize, figsize))
    axes = np.array(axes).reshape(-1)

    vmin = float(np.min(embeddings[:, :nb_components]))
    vmax = float(np.max(embeddings[:, :nb_components]))
    sm = ScalarMappable(cmap="plasma", norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    nodes_with_pos = list(positions.keys())
    subgraph = graph.subgraph(nodes_with_pos)
    mean_lat = float(np.mean([positions[n][1] for n in nodes_with_pos]))
    aspect = 1.0 / np.cos(np.deg2rad(mean_lat))

    for comp_idx in range(nb_components):
        ax = axes[comp_idx]
        node_colors = embeddings[nodes_with_pos, comp_idx]
        nx.draw(
            subgraph,
            pos={n: positions[n] for n in nodes_with_pos},
            ax=ax,
            with_labels=False,
            node_size=8,
            node_color=node_colors,
            edge_color="gray",
            width=0.2,
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect(aspect, adjustable="box")
        ax.set_title(f"Component {comp_idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[nb_components:]:
        ax.axis("off")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Embedding Value", rotation=270, labelpad=12)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
