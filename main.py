import argparse
import json
import os
import time

from src.baselines import louvain_baseline, networkx_kl_baseline, spectral_bisection_baseline
from src.data_loader import build_user_labels, extract_utterances_df, load_corpus
from src.evaluate import evaluate_partition
from src.export_viz_data import build_viz_payload, write_viz_payload
from src.graph_builder import build_reply_graph
from src.kl_algorithm import kernighan_lin_multi_restart

CACHE_DIR = "cache"
OUTPUT_DIR = "output"


def get_utterances_df(subreddit: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{subreddit}_utterances.parquet")
    if os.path.exists(cache_path):
        import pandas as pd

        print(f"Loading cached utterance table from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Downloading/loading ConvoKit corpus subreddit-{subreddit} ...")
    corpus = load_corpus(subreddit)
    print(f"Corpus loaded: {len(corpus.get_utterance_ids())} utterances, "
          f"{len(corpus.get_speaker_ids())} speakers")

    df = extract_utterances_df(corpus)
    df.to_parquet(cache_path)
    print(f"Cached utterance table to {cache_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddit", default="AskTrumpSupporters")
    parser.add_argument("--min-comments", type=int, default=300)
    parser.add_argument("--max-nodes", type=int, default=800)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = get_utterances_df(args.subreddit)

    print(f"\nBuilding reply graph (min_comments={args.min_comments}, max_nodes={args.max_nodes}) ...", flush=True)
    t0 = time.time()
    graph = build_reply_graph(df, min_comments=args.min_comments, max_nodes=args.max_nodes)
    print(f"Graph built in {time.time() - t0:.1f}s: "
          f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
          f"(largest connected component)", flush=True)

    user_labels = build_user_labels(df)
    label_counts = {}
    for node in graph.nodes():
        lbl = user_labels.get(node, "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"Ground-truth label distribution on graph nodes: {label_counts}")

    print(f"\nRunning from-scratch Kernighan-Lin ({args.n_restarts} random restarts) ...")
    t0 = time.time()
    group0, group1, partition_of, kl_trace = kernighan_lin_multi_restart(
        graph, n_restarts=args.n_restarts, seed=args.seed, verbose_passes=True
    )
    print(f"KL finished in {time.time() - t0:.1f}s")
    print(f"Cut size trajectory across passes: {kl_trace['cut_history']}")
    print(f"Initial cut: {kl_trace['initial_cut']:.1f} -> Final cut: {kl_trace['final_cut']:.1f} "
          f"(restart summaries in kl_log.json)")

    metrics = evaluate_partition(graph, group0, group1, partition_of, user_labels)
    print("\n=== Evaluation vs ground-truth flair ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== Baselines ===")
    baseline_results = {
        "networkx_kl": networkx_kl_baseline(graph, seed=args.seed),
        "spectral_bisection": spectral_bisection_baseline(graph),
        "louvain_unconstrained": louvain_baseline(graph, seed=args.seed),
    }
    print(json.dumps(baseline_results, indent=2))

    meta = {
        "subreddit": args.subreddit,
        "min_comments": args.min_comments,
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
    }

    kl_log_path = os.path.join(OUTPUT_DIR, "kl_log.json")
    with open(kl_log_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "trace": kl_trace}, f, indent=2)
    print(f"\nFull KL trace written to {kl_log_path}")

    payload = build_viz_payload(
        graph, partition_of, user_labels, kl_trace, metrics, baseline_results, meta
    )
    viz_path = os.path.join(OUTPUT_DIR, "graph_data.json")
    write_viz_payload(payload, viz_path)
    print(f"Visualization data written to {viz_path}")


if __name__ == "__main__":
    main()
