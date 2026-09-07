#!/usr/bin/env bash
# Serves the interactive visualization at http://localhost:8000/viz.html
# Run it, open the link, press Ctrl+C in this terminal when you're done.

cd "$(dirname "$0")"
echo "Serving output/viz.html at http://localhost:8000/viz.html"
echo "Press Ctrl+C to stop."
uv run python -m http.server 8000 --directory output
