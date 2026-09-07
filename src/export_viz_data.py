"""Serialize the graph, KL result, and evaluation metrics into one JSON file for
the interactive D3 frontend (no matplotlib in this pipeline)."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx


def build_viz_payload(
    graph: nx.Graph,
    partition_of: dict,
    user_labels: dict[str, str],
    kl_trace: dict[str, Any],
    metrics: dict[str, Any],
    baseline_results: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    nodes = []
    for node in graph.nodes():
        nodes.append(
            {
                "id": node,
                "group": partition_of[node],
                "flair_label": user_labels.get(node, "unknown"),
                "n_comments": graph.nodes[node].get("n_comments", 0),
                "degree": graph.degree(node, weight="weight"),
            }
        )

    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                "weight": data.get("weight", 1),
                "cross": partition_of[u] != partition_of[v],
            }
        )

    # Trim the pass-level detail for the frontend payload (kept in full in kl_log.json).
    passes_summary = [
        {
            "pass": p["pass"],
            "k_star": p["k_star"],
            "gain_applied": p["gain_applied"],
            "cut_after_pass": p["cut_after_pass"],
        }
        for p in kl_trace["passes"]
    ]

    return {
        "meta": meta,
        "nodes": nodes,
        "edges": edges,
        "kl_trace": {
            "initial_cut": kl_trace["initial_cut"],
            "final_cut": kl_trace["final_cut"],
            "cut_history": kl_trace["cut_history"],
            "passes": passes_summary,
            "n_restarts": kl_trace.get("n_restarts"),
            "restarts": kl_trace.get("restarts"),
        },
        "metrics": metrics,
        "baselines": baseline_results,
    }


def write_viz_payload(payload: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
