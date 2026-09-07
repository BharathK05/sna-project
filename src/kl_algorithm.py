"""Kernighan-Lin graph bisection, implemented from scratch (no library call).

Standard KL mechanics, vectorized with numpy for speed:
  - D_v = external cost - internal cost of node v w.r.t. its current side.
  - gain(a, b) = D_a + D_b - 2*w(a, b) for a candidate cross-side swap pair.
  - Each pass: repeatedly lock in the max-gain unlocked pair, updating D-values
    of the remaining unlocked nodes in O(n) per swap; then roll back to the
    prefix of swaps that maximizes cumulative gain (which may be < all of them).
  - Repeat passes until a pass finds no positive-gain prefix (convergence).
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np
from tqdm import tqdm


def _dense_weight_matrix(graph: nx.Graph, nodes: list) -> tuple[np.ndarray, dict]:
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    weights = np.zeros((n, n))
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1)
        i, j = index[u], index[v]
        weights[i, j] = w
        weights[j, i] = w
    return weights, index


def _cut_from_mask(weights: np.ndarray, mask: np.ndarray) -> float:
    a = np.where(mask)[0]
    b = np.where(~mask)[0]
    return float(weights[np.ix_(a, b)].sum())


def kernighan_lin_bisection(
    graph: nx.Graph, seed: int | None = None, max_passes: int = 100, verbose: bool = False
) -> tuple[set, set, dict, dict[str, Any]]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    n = len(nodes)
    weights, _ = _dense_weight_matrix(graph, nodes)

    order = list(range(n))
    rng.shuffle(order)
    half = n // 2
    mask = np.zeros(n, dtype=bool)
    mask[order[:half]] = True  # True = group A

    initial_cut = _cut_from_mask(weights, mask)
    cut_history = [initial_cut]
    pass_records = []

    for pass_idx in range(max_passes):
        row_sum_a = weights @ mask.astype(float)
        row_sum_b = weights @ (~mask).astype(float)
        d_values = np.where(mask, row_sum_b - row_sum_a, row_sum_a - row_sum_b)

        locked = np.zeros(n, dtype=bool)
        n_pairs = min(int(mask.sum()), int((~mask).sum()))
        gain_seq: list[float] = []
        swap_seq: list[tuple[int, int]] = []
        d_cur = d_values.copy()

        for _ in range(n_pairs):
            avail_a = np.where(mask & ~locked)[0]
            avail_b = np.where(~mask & ~locked)[0]
            if len(avail_a) == 0 or len(avail_b) == 0:
                break
            da = d_cur[avail_a]
            db = d_cur[avail_b]
            w_ab = weights[np.ix_(avail_a, avail_b)]
            gain_matrix = da[:, None] + db[None, :] - 2 * w_ab
            flat_idx = int(np.argmax(gain_matrix))
            ai, bi = np.unravel_index(flat_idx, gain_matrix.shape)
            a, b = int(avail_a[ai]), int(avail_b[bi])
            gain_seq.append(float(gain_matrix[ai, bi]))
            swap_seq.append((a, b))
            locked[a] = True
            locked[b] = True

            still_a = np.where(mask & ~locked)[0]
            still_b = np.where(~mask & ~locked)[0]
            d_cur[still_a] += 2 * weights[still_a, a] - 2 * weights[still_a, b]
            d_cur[still_b] += 2 * weights[still_b, b] - 2 * weights[still_b, a]

        cum = np.cumsum(gain_seq) if gain_seq else np.array([])
        if len(cum) > 0:
            k_star = int(np.argmax(cum))
            best_cum = float(cum[k_star])
        else:
            k_star = -1
            best_cum = 0.0

        improved = best_cum > 1e-9
        if improved:
            for a, b in swap_seq[: k_star + 1]:
                mask[a] = False
                mask[b] = True

        new_cut = _cut_from_mask(weights, mask)
        pass_records.append(
            {
                "pass": pass_idx,
                "n_swaps_considered": len(gain_seq),
                "k_star": k_star,
                "cumulative_gain_sequence": [float(x) for x in cum],
                "gain_applied": best_cum,
                "cut_after_pass": new_cut,
            }
        )
        cut_history.append(new_cut)
        if verbose:
            print(f"    pass {pass_idx}: swaps_applied={k_star + 1 if improved else 0}, "
                  f"gain={best_cum:.1f}, cut={new_cut:.1f}", flush=True)
        if not improved:
            break

    partition_of = {nodes[i]: (0 if mask[i] else 1) for i in range(n)}
    group0 = {nodes[i] for i in range(n) if mask[i]}
    group1 = {nodes[i] for i in range(n) if not mask[i]}
    trace: dict[str, Any] = {
        "n_nodes": n,
        "n_edges": int(graph.number_of_edges()),
        "initial_cut": initial_cut,
        "final_cut": cut_history[-1],
        "cut_history": cut_history,
        "passes": pass_records,
        "seed": seed,
    }
    return group0, group1, partition_of, trace


def kernighan_lin_multi_restart(
    graph: nx.Graph,
    n_restarts: int = 10,
    seed: int = 0,
    max_passes: int = 100,
    verbose_passes: bool = False,
) -> tuple[set, set, dict, dict[str, Any]]:
    """KL is a local-search heuristic sensitive to its random initial partition;
    run several restarts and keep the lowest-cut result."""
    best: tuple[set, set, dict, dict[str, Any]] | None = None
    restart_summaries = []
    progress = tqdm(range(n_restarts), desc="KL restarts", unit="restart")
    for i in progress:
        if verbose_passes:
            print(f"  restart {i} (seed={seed + i}):", flush=True)
        result = kernighan_lin_bisection(
            graph, seed=seed + i, max_passes=max_passes, verbose=verbose_passes
        )
        trace = result[3]
        restart_summaries.append(
            {
                "restart": i,
                "seed": seed + i,
                "initial_cut": trace["initial_cut"],
                "final_cut": trace["final_cut"],
                "n_passes": len(trace["passes"]),
            }
        )
        if best is None or trace["final_cut"] < best[3]["final_cut"]:
            best = result
        progress.set_postfix(best_cut=best[3]["final_cut"], this_cut=trace["final_cut"])

    assert best is not None
    best[3]["restarts"] = restart_summaries
    best[3]["n_restarts"] = n_restarts
    return best
