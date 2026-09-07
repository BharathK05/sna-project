"""Embed output/graph_data.json into viz/template.html to produce a
self-contained artifact HTML file at output/viz.html."""

import json

with open("output/graph_data.json", "r", encoding="utf-8") as f:
    payload = json.load(f)

compact_json = json.dumps(payload, separators=(",", ":"))

with open("viz/template.html", "r", encoding="utf-8") as f:
    template = f.read()

if "__GRAPH_DATA_JSON__" not in template:
    raise SystemExit("template.html is missing the __GRAPH_DATA_JSON__ placeholder")

html = template.replace("__GRAPH_DATA_JSON__", compact_json)

with open("output/viz.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote output/viz.html ({len(html) / 1e6:.2f} MB)")
