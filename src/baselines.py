"""Baselines to contextualize the from-scratch KL result: networkx's own KL
implementation (correctness cross-check), a spectral (Fiedler-vector) bisection,
and unconstrained Louvain community detection."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from networkx.algorithms.community import kernighan_lin_bisection as nx_kl
from networkx.algorithms.community import louvain_communities, modularity


def _cut_size(graph: nx.Graph, group0: set) -> float:
    return float(nx.cut_size(graph, group0, weight="weight"))


def networkx_kl_baseline(graph: nx.Graph, seed: int = 0) -> dict[str, Any]:
    group0, group1 = nx_kl(graph, weight="weight", seed=seed)
    return {
        "cut_size": _cut_size(graph, group0),
        "modularity": modularity(graph, [group0, group1], weight="weight"),
        "group_sizes": [len(group0), len(group1)],
    }


def spectral_bisection_baseline(graph: nx.Graph) -> dict[str, Any]:
    """Balanced bisection by median-splitting the Fiedler (algebraic-connectivity)
    eigenvector, so it's comparable to KL under the same size-balance constraint."""
    nodes = list(graph.nodes())
    fiedler = nx.fiedler_vector(graph, weight="weight")
    order = np.argsort(fiedler)
    half = len(nodes) // 2
    group0 = {nodes[i] for i in order[:half]}
    group1 = {nodes[i] for i in order[half:]}
    return {
        "cut_size": _cut_size(graph, group0),
        "modularity": modularity(graph, [group0, group1], weight="weight"),
        "group_sizes": [len(group0), len(group1)],
    }


def louvain_baseline(graph: nx.Graph, seed: int = 0) -> dict[str, Any]:
    """Unconstrained community detection (no k=2 constraint) for context."""
    communities = louvain_communities(graph, weight="weight", seed=seed)
    return {
        "n_communities": len(communities),
        "community_sizes": sorted((len(c) for c in communities), reverse=True),
        "modularity": modularity(graph, communities, weight="weight"),
    }
