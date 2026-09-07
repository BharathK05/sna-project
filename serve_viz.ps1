# Serves the interactive visualization at http://localhost:8000/viz.html
# Run it, open the link, press Ctrl+C in this window when you're done.

Set-Location $PSScriptRoot
Write-Host "Serving output/viz.html at http://localhost:8000/viz.html"
Write-Host "Press Ctrl+C to stop."
uv run python -m http.server 8000 --directory output
