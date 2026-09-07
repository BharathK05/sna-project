# Bipartisan Polarized Debate Partitioning — Project Summary

Course case study in community detection: apply **Kernighan–Lin (KL) graph
bisection** to a real online-forum dataset to test whether minimizing
cross-group interaction recovers the two political camps in a debate, the
way it's assumed to for echo chambers on platforms like Twitter.

## Problem statement

Given the user–user reply-interaction network of a political discussion
forum, can a balanced two-way graph partition that minimizes cross-group
edge weight (Kernighan–Lin bisection) recover the forum's actual political
camps? The hypothesis, mirroring the "detect echo chambers by partitioning a
user network to minimize cross-connections" framing, is that structurally
separated communities *are* the ideological camps. This project tests that
hypothesis on data, rather than assuming it.

## Dataset

**r/AskTrumpSupporters**, pulled via [ConvoKit](https://convokit.cornell.edu/)
(`download("subreddit-AskTrumpSupporters")` — no Reddit API, no scraping, no
rate limits; ~282MB corpus, 1.37M usable utterances after bot/deleted
filtering, 39K speakers). Chosen over the sample "merge two partisan
subreddits" approach because it gives **real, user-supplied ground truth**:
the subreddit's own flair convention has users self-identify as
`Trump Supporter` (many joke variants: "Nimble Navigator", "CENTIPEDE",
"MAGA", "Build The Wall", ...) or `Non-Trump Supporter`/`Nonsupporter`. A
substring classifier (`src/data_loader.py::classify_flair`) buckets every
utterance's flair into `supporter` / `nonsupporter` / `undecided` / `unknown`,
then each user is majority-labeled from their own history
(`build_user_labels`). This label is used **only for evaluation**, never fed
to the algorithm.

## What was built

| File | Role |
|---|---|
| `src/data_loader.py` | ConvoKit download/load, utterance flattening, flair→camp labeling |
| `src/graph_builder.py` | Builds the weighted user–user reply graph; caps to the `max_nodes` most active users (dense O(n³)-ish KL needs n in the low thousands) and reduces to the largest connected component |
| `src/kl_algorithm.py` | **Kernighan–Lin bisection implemented from scratch** (numpy-vectorized D-values/gains, full per-pass trace logging) + multi-restart wrapper |
| `src/baselines.py` | networkx's own KL (correctness cross-check), spectral (Fiedler-vector) bisection, unconstrained Louvain |
| `src/evaluate.py` | Cut size, modularity, best-alignment accuracy, NMI, confusion matrix vs. flair ground truth |
| `src/export_viz_data.py` | Serializes graph + KL trace + metrics to `output/graph_data.json` |
| `main.py` | CLI orchestrating the full pipeline, with `tqdm` progress bars throughout |
| `viz/template.html` + `build_viz.py` | Interactive D3/canvas force-graph frontend; `build_viz.py` embeds `graph_data.json` into the template to produce the published artifact |

No `matplotlib` — the deliverable visualization is the interactive artifact,
not a static plot.

## How to run it

```
uv sync
uv run python main.py --subreddit AskTrumpSupporters --min-comments 300 --max-nodes 800 --n-restarts 10
uv run python build_viz.py
```

First run downloads and caches the corpus (`cache/*.parquet`); subsequent
runs reuse the cache. `output/kl_log.json` holds the full per-pass D-value/gain
trace (evidence for the math-proof section of the written report);
`output/graph_data.json` and `output/viz.html` feed the frontend.

## Algorithm

Classic Kernighan–Lin, implemented directly (not just called from a library):
random balanced initial partition → per pass, repeatedly lock in the
max-gain cross-side pair (`gain(a,b) = D_a + D_b − 2·w(a,b)`), updating
D-values in O(n) per swap → roll back to the prefix of swaps maximizing
cumulative gain → repeat passes until no improving pass is found. Ten random
restarts guard against a bad initial partition, since KL is a local-search
heuristic. **Sanity-checked** on a synthetic two-clique-plus-bridge graph
(recovers the exact bisection, cut=1) and **cross-validated** against
`networkx.algorithms.community.kernighan_lin_bisection` on the real graph —
both land on the identical cut size (67,770), confirming the from-scratch
implementation is correct.

## Results (n=800 users, min 300 comments each, 53,186 weighted reply-edges)

| | |
|---|---|
| Initial cut (random split) | 174,082 |
| Final cut (best of 10 restarts) | 67,770 (−61%) |
| Modularity | 0.281 |
| Accuracy vs. true flair | **57.9%** |
| NMI vs. true flair | **0.018** |

networkx's KL matched exactly (67,770); spectral bisection was worse
(105,645); unconstrained Louvain found 4 communities at slightly higher
modularity (0.310), suggesting the natural community structure isn't a clean
2-way split either.

### The finding worth stating plainly

**KL successfully minimizes the cut (−61%, exactly matching the reference
implementation) but the resulting bisection is only marginally better than
chance at recovering political camps** (57.9% accuracy, NMI≈0.018 — near
statistical independence). This is not a bug: r/AskTrumpSupporters is
*structurally* a cross-camp Q&A forum — nonsupporters ask questions,
supporters answer them — so the interaction graph's densest ties are
*between* camps, not within them. Minimizing cross-group edges optimizes
against the very thing the forum is built to produce. This is a real,
useful negative result for the case study's conclusion: the
"detect echo chambers via min-cut" intuition, valid on platforms built for
homophilous interaction (retweet networks, follow graphs), does not transfer
to a forum explicitly designed for cross-ideological engagement. A next
step worth naming (out of scope here) would be a *signed* graph formulation
— agreement/disagreement-weighted edges rather than raw reply counts — where
minimizing cross-camp *agreement* might behave differently than minimizing
cross-camp *contact*.

## Interactive visualization

Published as a Claude Artifact ("Debate Cut") — a canvas force-directed
graph of all 800 users, with a toggle between coloring by KL-assigned group
and coloring by true flair (the mismatch between the two colorings *is* the
57.9%-accuracy finding, made visible), an edge-strength filter, a cut-edge
highlight, per-pass convergence and restart-consistency charts, a confusion
matrix, and the baseline comparison table. Link: shared separately in the
conversation.
