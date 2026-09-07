# Bipartisan Polarized Debate Partitioning

A community-detection case study: does **Kernighan–Lin graph bisection**
recover real political camps in an online debate forum, the way it's
assumed to for echo chambers on platforms like Twitter?

Full write-up (problem, method, results, interpretation) is in
[`plan.md`](plan.md). This README is just setup/run instructions.

Dataset (r/AskTrumpSupporters, via ConvoKit) and pipeline outputs are
committed in `cache/` and `output/`, so everything below works immediately
after cloning — no re-download needed unless you want to re-run the
pipeline from scratch.

## 1. Install `uv`

**Windows** (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac** (Terminal):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
(or `brew install uv`)

Restart your terminal afterwards so `uv` is on your `PATH`.

## 2. Clone and install dependencies

```bash
git clone https://github.com/BharathK05/sna-project.git
cd sna-project
uv sync
```

`uv sync` creates `.venv` and installs everything from `uv.lock` — same
versions on every machine.

## 3. View the visualization (no pipeline run needed)

The built visualization is already at `output/viz.html`. Serve it locally:

**Windows** (PowerShell):
```powershell
.\serve_viz.ps1
```

**Mac / Linux** (Terminal):
```bash
chmod +x serve_viz.sh   # first time only
./serve_viz.sh
```

Both do the same thing: `uv run python -m http.server 8000 --directory output`.
Open **http://localhost:8000/viz.html** in your browser.

To stop the server: press **Ctrl+C** in that terminal. If you ever lose
track of it:
```powershell
# Windows
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```
```bash
# Mac / Linux
lsof -ti:8000 | xargs kill -9
```

## 4. (Optional) Re-run the full pipeline

Only needed if you want to regenerate results (e.g. with different
parameters). Uses the cached corpus in `cache/`, so it's fast and doesn't
need internet:

```bash
uv run python main.py --subreddit AskTrumpSupporters --min-comments 300 --max-nodes 800 --n-restarts 10
uv run python build_viz.py
```

This overwrites `output/kl_log.json`, `output/graph_data.json`, and
`output/viz.html`. Re-run step 3's server (or just refresh the browser tab
if it's already running) to see the new result.

If `cache/AskTrumpSupporters_utterances.parquet` is ever deleted,
`main.py` will re-download the corpus from ConvoKit automatically
(~280MB, one-time).

## Project layout

```
main.py              # pipeline entrypoint (CLI)
build_viz.py          # embeds output/graph_data.json into viz/template.html -> output/viz.html
src/
  data_loader.py       # ConvoKit download/load, flair -> political-camp labeling
  graph_builder.py      # builds the weighted user-reply graph
  kl_algorithm.py        # Kernighan-Lin bisection, implemented from scratch
  baselines.py            # networkx KL / spectral / Louvain, for comparison
  evaluate.py               # accuracy, NMI, confusion matrix vs. ground truth
  export_viz_data.py         # serializes results to output/graph_data.json
viz/template.html      # interactive D3/canvas frontend (source)
cache/                  # cached ConvoKit utterance table (committed)
output/                  # pipeline results + built visualization (committed)
plan.md                   # full project write-up
```
