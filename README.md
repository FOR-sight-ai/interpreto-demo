---
title: Interpreto Explanation Gallery
sdk: static
---

# Interpreto Explanation Gallery

This repository contains a static gallery for comparing precomputed explanation visualizations (attribution + concept-based). Each visualization is loaded in an iframe from the `explanations/` folder and displayed side-by-side for comparison. There is no server or runtime Python required at deployment time.

## Project layout

- `index.html`, `styles.css`, `app.js`: Static UI (vanilla HTML/CSS/JS)
- `manifest.json`: Generated index of available models and explanations
- `scripts/build_manifest.py`: Manifest builder script (run locally)
- `scripts/_common.py`: Shared helpers used by the explanation-generation scripts (activation caching, concept model save/load, snippet formatters)
- `explanations/`: Precomputed explanation HTML files
- `data/`: Cached activations and fitted concept explainers (git-ignored). Populated by the explanation-generation scripts to avoid recomputing on every run.
- `assets/`: Logos and static images

## Model identifiers

Use a task prefix so the UI can group models by task:

- Classification: `clf:<dataset>:<model>` (example: `clf:imdb:bert-base`)
- Generation: `gen:<model>` (example: `gen:gpt2`)

Including the dataset in the classification model id is recommended.

## Add explanation files

Place your explanation HTML files under `explanations/` using the hierarchy below.

### File hierarchy

```
explanations/
  <model_id>/
    attribution/
      single-class/
        <sample_id>/
          <method>.html
      all-classes/
        <sample_id>/
          <method>.html
      general/
        <sample_id>/
          <method>.html
    concept/
      general/                       # classification: model-level
        <method>.html
      local/                         # generation: sample-linked
        <sample_id>/
          <method>.html
```

Notes:

- Classification concepts are model-level: place methods directly under `concept/general/`, no `<sample_id>` folder.
- Generation concepts are sample-linked: use `concept/local/<sample_id>/<method>.html`.
- Attribution explanations are always sample-linked.
- Methods are derived from the HTML filenames, so use consistent names (e.g. `lime.html`, `kernel_shap.html`).

### Examples

- Classification, attribution, all classes:
  `explanations/clf:emotion:bert-base/attribution/all-classes/sample-001/lime.html`
- Classification, attribution, single class:
  `explanations/clf:emotion:bert-base/attribution/single-class/sample-001/lime.html`
- Classification, concept (model-level):
  `explanations/clf:emotion:bert-base/concept/general/semi_nmf.html`
- Generation, attribution, general:
  `explanations/gen:gpt2/attribution/general/sample-001/integrated_gradients.html`
- Generation, concept, local:
  `explanations/gen:gpt2/concept/local/sample-001/vanilla_sae.html`

## Select explanations in the UI

Use the filters to lock everything except the method:

1. Task
2. Model
3. Explanation type (attribution vs concept)
4. Scope (single-class vs all-classes for attribution; general or local for concept)
5. Sample (when applicable)
6. Methods (multi-select for comparison)

## Build the manifest

Run the manifest builder any time you add or remove explanation files:

```bash
python scripts/build_manifest.py
```

This writes `manifest.json` at the repository root. The manifest also
carries the installed `interpreto` version (rendered as a small pill
next to the title in the UI). Commit the updated manifest before
uploading to Hugging Face Spaces.

## Regenerate explanations

The `scripts/` folder contains one script per explanation family. Each
script writes both the visualization HTML and a matching minimal `.py`
snippet under `explanations/<model_id>/...`.

```bash
# 10 samples, all 10 attribution methods, both single-class and all-classes
python scripts/classification_attributions.py

# 10 attribution methods on 3 fixed generation samples
python scripts/generation_attributions.py

# Concept methods on the full classification train sets
python scripts/classification_concepts.py

# Concept methods on Wikipedia (per-model sample counts in MODEL_CONFIGS)
python scripts/generation_concepts.py
```

Model selection is done by editing the `model_id` variable at the top
of each script. Activations are cached under `data/<model_id>/activations.pt`
(float32 for classification, float16 for generation to shrink disk usage)
and fitted concept explainers under `data/<model_id>/explainers/<method>.pt`,
so re-running a script only trains what is missing. `NUM_SAMPLES` can be
overridden per run via the `DEBUG_SAMPLES` environment variable.

## Shrink explanation HTML files

Interpreto exports embed CSS and JS in every HTML file. To deduplicate and load those assets once, run:

```bash
python scripts/externalize_explanations.py
```

This rewrites the files under `explanations/` to reference `assets/css/visualization.css` and the scripts under `assets/js/`. Re-run the script after regenerating explanations. To extract a single bundle from inline assets, run:

```bash
python scripts/externalize_explanations.py --mode extract --css-path assets/css/visualization.css --js-path assets/js/visualization_bundle.js
```

## Run locally

Use a simple static file server (recommended) so `manifest.json` can be fetched:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Then visit `http://localhost:8000/` in your browser.

## Single line local deployment

```bash
python scripts/build_manifest.py; python scripts/externalize_explanations.py; python -m http.server 8000 --bind 127.0.0.1
```

## Deploy to Hugging Face Spaces (Static)

1. Ensure `manifest.json` is up to date.
2. Upload the repository to a new Space with SDK set to `Static`.
3. The app will load `manifest.json` and render the gallery.
