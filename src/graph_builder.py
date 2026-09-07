"""Build a weighted user-user reply-interaction graph from a flattened utterance table."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import pandas as pd
from tqdm import tqdm


def build_reply_graph(df: pd.DataFrame, min_comments: int = 15, max_nodes: int = 800) -> nx.Graph:
    """Nodes = the `max_nodes` most active users with >= min_comments comments
    (dense O(n^2)-per-swap KL needs n kept in the low thousands at most to stay
    fast). Edge (u, v) weight = number of direct-reply exchanges between u and v
    (self-replies excluded). Returns only the largest connected component, since
    bisection assumes a connected graph."""
    activity = df["author"].value_counts()
    active_authors = set(activity[activity >= min_comments].index)
    if len(active_authors) > max_nodes:
        active_authors = set(activity[activity >= min_comments].nlargest(max_nodes).index)

    id_to_author = df.set_index("utt_id")["author"].to_dict()

    edge_counter: Counter[tuple[str, str]] = Counter()
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Building reply graph"):
        if row.author not in active_authors:
            continue
        parent_id = row.reply_to
        if parent_id is None:
            continue
        parent_author = id_to_author.get(parent_id)
        if parent_author is None or parent_author not in active_authors:
            continue
        if parent_author == row.author:
            continue
        edge = tuple(sorted((row.author, parent_author)))
        edge_counter[edge] += 1

    graph = nx.Graph()
    for author in active_authors:
        graph.add_node(author, n_comments=int(activity[author]))
    for (u, v), weight in edge_counter.items():
        graph.add_edge(u, v, weight=weight)

    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        return graph

    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc).copy()
