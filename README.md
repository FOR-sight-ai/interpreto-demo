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
- `explanations/`: Precomputed explanation HTML files
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
      class-wise/
        <sample_id>/
          <method>.html
      general/
        <sample_id>/
          <method>.html
```

Notes:

- All explanations are linked to a sample except **classification + concept**, which is model-level.
- For classification concepts, omit the `<sample_id>` folder and place methods directly under the scope.
- For generation models, concept explanations are still sample-linked:
  `explanations/gen:gpt2/concept/general/<sample_id>/<method>.html`
- Methods are derived from the HTML filenames, so use consistent names (e.g. `lime.html`, `kernel_shap.html`).

### Examples

- Classification, attribution, all classes:
  `explanations/clf:emotion:bert-base/attribution/all-classes/sample-001/lime.html`
- Classification, attribution, single class:
  `explanations/clf:emotion:bert-base/attribution/single-class/sample-001/lime.html`
- Classification, concept, class-wise:
  `explanations/clf:emotion:bert-base/concept/class-wise/tcav.html`
- Generation, attribution, general:
  `explanations/gen:gpt2/attribution/general/sample-001/integrated_gradients.html`
- Generation, concept, general:
  `explanations/gen:gpt2/concept/general/sample-001/concept_excitation.html`

## Select explanations in the UI

Use the filters to lock everything except the method:

1. Task
2. Model
3. Explanation type (attribution vs concept)
4. Scope (single-class vs all-classes, class-wise vs general)
5. Sample (when applicable)
6. Methods (multi-select for comparison)

## Build the manifest

Run the manifest builder any time you add or remove explanation files:

```bash
python scripts/build_manifest.py
```

This writes `manifest.json` at the repository root. Commit the updated manifest before uploading to Hugging Face Spaces.

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
