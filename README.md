---
title: Attribution Gallery
sdk: static
---

# Attribution Gallery

This repository contains a static gallery for comparing precomputed text-attribution visualizations. Each visualization is loaded in an iframe from the `attributions/` folder and displayed side-by-side for comparison. There is no server or runtime Python required at deployment time.

## Project layout

- `index.html`, `styles.css`, `app.js`: Static UI (vanilla HTML/CSS/JS)
- `manifest.json`: Generated index of available models, samples, and methods
- `tools/build_manifest.py`: Manifest builder script (run locally)
- `assets/<model>/<sample>/<method_and_params>.html`: Precomputed explanation files

## Add attribution files

Place your explanation HTML files under:

`attributions/<model>/<sample>/<method_and_params>.html`

Example:

`attributions/llama2/sample-001/integrated_gradients_steps50.html`

The UI uses the filename as the method identifier and displays a prettified version in the gallery.

## Build the manifest

Run the manifest builder any time you add or remove attribution files:

```bash
python tools/build_manifest.py
```

This writes `manifest.json` at the repository root. Commit the updated manifest before uploading to Hugging Face Spaces.

## Run locally

Use a simple static file server (recommended) so `manifest.json` can be fetched:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Then visit `http://localhost:8000/` in your browser.

## Deploy to Hugging Face Spaces (Static)

1. Ensure `manifest.json` is up to date.
2. Upload the repository to a new Space with SDK set to `Static`.
3. The app will load `manifest.json` and render the gallery.
