"""Score a KL bisection against ground-truth flair labels."""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.community import modularity
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score

_LABELS = ("supporter", "nonsupporter")


def evaluate_partition(
    graph: nx.Graph,
    group0: set,
    group1: set,
    partition_of: dict,
    user_labels: dict[str, str],
) -> dict[str, Any]:
    cut = float(nx.cut_size(graph, group0, weight="weight"))
    mod = modularity(graph, [group0, group1], weight="weight")

    known_nodes = [n for n in graph.nodes() if user_labels.get(n) in _LABELS]
    coverage = len(known_nodes) / graph.number_of_nodes() if graph.number_of_nodes() else 0.0

    y_true = [user_labels[n] for n in known_nodes]
    y_pred_raw = [partition_of[n] for n in known_nodes]

    def accuracy_for(mapping: dict[int, str]) -> tuple[float, list[str]]:
        y_pred = [mapping[p] for p in y_pred_raw]
        if not y_true:
            return 0.0, y_pred
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true), y_pred

    mapping_a = {0: "supporter", 1: "nonsupporter"}
    mapping_b = {0: "nonsupporter", 1: "supporter"}
    acc_a, pred_a = accuracy_for(mapping_a)
    acc_b, pred_b = accuracy_for(mapping_b)
    if acc_a >= acc_b:
        accuracy, y_pred, mapping = acc_a, pred_a, mapping_a
    else:
        accuracy, y_pred, mapping = acc_b, pred_b, mapping_b

    nmi = normalized_mutual_info_score(y_true, y_pred_raw) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(_LABELS)) if y_true else [[0, 0], [0, 0]]

    return {
        "cut_size": cut,
        "modularity": mod,
        "group_sizes": [len(group0), len(group1)],
        "ground_truth_coverage": coverage,
        "n_labeled_nodes": len(known_nodes),
        "accuracy": accuracy,
        "nmi": float(nmi),
        "label_mapping": mapping,
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        "confusion_matrix_labels": list(_LABELS),
    }
